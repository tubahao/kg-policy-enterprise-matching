#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比学习训练GAT模型

目标：学习政策-企业的相关性表示，而非简单的链接预测
- 正样本：有supports/targetsIndustry关系的(policy, enterprise)对
- 负样本：随机采样的无关(policy, enterprise)对

训练目标：InfoNCE对比损失
- 让相关的policy-enterprise嵌入更近
- 让不相关的policy-enterprise嵌入更远
"""

import json
import sys
import random
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, GATConv
from torch.utils.data import DataLoader, Dataset

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class HeteroGATContrastive(nn.Module):
    """用于对比学习的异构图GAT模型（含Jumping Knowledge残差连接）"""
    
    def __init__(self, in_dims: Dict[str, int], hidden_dim: int = 128, 
                 out_dim: int = 64, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1,
                 edge_types: Optional[List[str]] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.input_projs = nn.ModuleDict()
        for ntype, in_dim in in_dims.items():
            self.input_projs[ntype] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        self.layers = nn.ModuleList()
        if not edge_types:
            edge_types = ['transmitsTo', 'transmitsFrom', 'belongsTo', 'includesCompany', 'targetsIndustry', 'targetedByPolicy', 'supports', 'supportedByPolicy']
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            conv_dict = {}
            for etype in edge_types:
                conv_dict[etype] = GATConv(
                    in_dim, hidden_dim, num_heads,
                    feat_drop=dropout, attn_drop=dropout,
                    activation=F.elu if i < num_layers - 1 else None,
                    allow_zero_in_degree=True
                )
            self.layers.append(HeteroGraphConv(conv_dict, aggregate='mean'))
        
        # Jumping Knowledge: 拼接底层投影特征与最终拓扑特征后降维
        jk_in_dim = hidden_dim + hidden_dim * num_heads
        self.output_projs = nn.ModuleDict()
        for ntype in in_dims.keys():
            self.output_projs[ntype] = nn.Sequential(
                nn.Linear(jk_in_dim, out_dim),
                nn.LayerNorm(out_dim)
            )
        
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, g: dgl.DGLHeteroGraph, 
                node_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        # 输入投影
        h = {}
        for ntype in g.ntypes:
            if ntype in node_features and ntype in self.input_projs:
                h[ntype] = self.input_projs[ntype](node_features[ntype])
            else:
                num_nodes = g.number_of_nodes(ntype)
                h[ntype] = torch.zeros(num_nodes, self.hidden_dim, device=device)
        
        # 保存底层投影特征用于残差连接
        h_initial = {k: v.clone() for k, v in h.items()}
        
        # GAT消息传递
        for layer in self.layers:
            h_new = layer(g, h)
            h = {k: v.flatten(1) if len(v.shape) > 2 else v for k, v in h_new.items()}
        
        # Jumping Knowledge: 拼接底层特征与拓扑聚合特征
        out = {}
        for ntype in g.ntypes:
            if ntype in h and ntype in self.output_projs:
                h_init = h_initial.get(ntype)
                h_gat = h[ntype]
                if h_init is not None and h_init.shape[0] == h_gat.shape[0]:
                    combined = torch.cat([h_init, h_gat], dim=1)
                else:
                    pad = torch.zeros(h_gat.shape[0], self.hidden_dim, device=device)
                    combined = torch.cat([pad, h_gat], dim=1)
                out[ntype] = self.output_projs[ntype](combined)
            else:
                num_nodes = g.number_of_nodes(ntype)
                out[ntype] = torch.zeros(num_nodes, self.out_dim, device=device)
        
        out = {k: F.normalize(v, p=2, dim=1) for k, v in out.items()}
        return out
    
    def contrastive_loss(self, policy_emb: torch.Tensor, company_emb: torch.Tensor,
                         positive_pairs: torch.Tensor, num_negatives: int = 10,
                         company_to_industry: Dict = None,
                         industry_to_companies: Dict = None,
                         policy_to_positive_companies: Dict[int, Set[int]] = None) -> torch.Tensor:
        """
        InfoNCE对比损失，支持困难负样本挖掘。

        当提供行业映射时，优先从同一行业大类中采样困难负样本。
        """
        batch_size = positive_pairs.shape[0]
        device = policy_emb.device
        
        policy_pos = policy_emb[positive_pairs[:, 0]]
        company_pos = company_emb[positive_pairs[:, 1]]
        
        pos_sim = (policy_pos * company_pos).sum(dim=1) / self.temperature
        
        num_companies = company_emb.shape[0]

        all_company_ids = list(range(num_companies))

        if company_to_industry is not None and industry_to_companies is not None:
            neg_indices = []
            for i in range(batch_size):
                pid = positive_pairs[i, 0].item()
                cid = positive_pairs[i, 1].item()
                known_positive = set()
                if policy_to_positive_companies is not None:
                    known_positive.update(policy_to_positive_companies.get(pid, set()))
                known_positive.add(cid)

                industries = company_to_industry.get(cid, [])
                candidates = set()
                for ind in industries:
                    candidates.update(industry_to_companies.get(ind, []))
                candidates = [c for c in candidates if c < num_companies and c not in known_positive]
                
                if len(candidates) >= num_negatives:
                    sampled = random.sample(candidates, num_negatives)
                else:
                    sampled = list(candidates)
                    remaining = num_negatives - len(sampled)
                    global_candidates = [c for c in all_company_ids if c not in known_positive]
                    if len(global_candidates) >= remaining:
                        sampled += random.sample(global_candidates, remaining)
                    elif len(global_candidates) > 0:
                        sampled += random.choices(global_candidates, k=remaining)
                    else:
                        sampled += random.choices(all_company_ids, k=remaining)
                neg_indices.append(sampled)
            neg_indices = torch.tensor(neg_indices, dtype=torch.long, device=device)
        else:
            neg_indices = torch.randint(0, num_companies, (batch_size, num_negatives), device=device)
        
        company_neg = company_emb[neg_indices]
        neg_sim = torch.bmm(company_neg, policy_pos.unsqueeze(2)).squeeze(2) / self.temperature
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


class PolicyEnterpriseDataset(Dataset):
    """政策-企业正样本对数据集"""
    
    def __init__(self, positive_pairs: List[Tuple[int, int]]):
        self.pairs = positive_pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.pairs[idx], dtype=torch.long)


def load_fused_features(
    project_root: Path,
    device: torch.device,
    override_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, torch.Tensor]:
    """加载融合特征（优先使用512D对齐特征）。override_paths 可指定 policy/company/industry 的 .npy 绝对路径。"""
    features: Dict[str, torch.Tensor] = {}
    override_paths = override_paths or {}

    if "policy" in override_paths:
        p = Path(override_paths["policy"])
        feat = np.load(p)
        features["policy"] = torch.tensor(feat, dtype=torch.float32).to(device)
        print(f"加载政策特征(override): {p} {feat.shape}", flush=True)
    else:
        for path_name, label in [
            ("features/policy_feature_aligned.npy", "政策对齐特征"),
            ("features/policy_feature_fused.npy", "政策融合特征"),
            ("embeddings/policy_text_concat_emb.npy", "政策文本嵌入"),
        ]:
            p = project_root / path_name
            if p.exists():
                feat = np.load(p)
                features["policy"] = torch.tensor(feat, dtype=torch.float32).to(device)
                print(f"加载{label}: {feat.shape}", flush=True)
                break

    if "company" in override_paths:
        p = Path(override_paths["company"])
        feat = np.load(p)
        features["company"] = torch.tensor(feat, dtype=torch.float32).to(device)
        print(f"加载企业特征(override): {p} {feat.shape}", flush=True)
    else:
        for path_name, label in [
            ("features/enterprise_feature_aligned.npy", "企业对齐特征"),
            ("features/enterprise_feature_fused.npy", "企业融合特征"),
            ("embeddings/enterprise_text_emb.npy", "企业文本嵌入"),
        ]:
            p = project_root / path_name
            if p.exists():
                feat = np.load(p)
                features["company"] = torch.tensor(feat, dtype=torch.float32).to(device)
                print(f"加载{label}: {feat.shape}", flush=True)
                break

    if "industry" in override_paths:
        p = Path(override_paths["industry"])
        industry_feat = np.load(p)
        features["industry"] = torch.tensor(industry_feat, dtype=torch.float32).to(device)
        print(f"加载行业嵌入(override): {p} {industry_feat.shape}", flush=True)
    else:
        industry_path = project_root / "embeddings/industry_text_emb.npy"
        if industry_path.exists():
            industry_feat = np.load(industry_path)
            features["industry"] = torch.tensor(industry_feat, dtype=torch.float32).to(device)
            print(f"加载行业嵌入: {industry_feat.shape}", flush=True)

    return features


def build_positive_pairs(
    project_root: Path,
    node_maps: Dict,
    use_supports_only: bool = False,
    max_industry_pairs_per_policy: int = 30,
    p2e_parquet: Optional[Path] = None,
) -> Tuple[
    List[Tuple[int, int]], Dict[int, List[int]], Dict[int, List[int]], Dict[int, Set[int]]
]:
    """
    构建正样本对（policy_node_id, company_node_id），
    同时构建企业→行业、行业→企业映射（用于困难负样本挖掘）。

    Returns:
        (positive_pairs, company_to_industry, industry_to_companies_dict, policy_to_positive_companies)
    """
    positive_pairs: Set[Tuple[int, int]] = set()
    
    p2e_path = p2e_parquet if p2e_parquet is not None else project_root / "data_intermediate/triples_policy_entity.parquet"
    if not p2e_path.exists():
        print(f"警告: {p2e_path} 不存在")
        return [], {}, {}, {}
    
    p2e_df = pd.read_parquet(p2e_path)
    
    policy_map = node_maps.get("policy", {})
    company_map = node_maps.get("company", {})
    industry_map = node_maps.get("industry", {})
    policy_to_positive_companies: Dict[int, Set[int]] = {}
    
    supports_count = 0
    for _, row in p2e_df.iterrows():
        pred = str(row["predicate"]).lower()
        if pred == "supports":
            policy_title = str(row["subject"])
            company_name = str(row["object"])
            policy_node = policy_map.get(policy_title)
            company_node = company_map.get(company_name)
            if policy_node is not None and company_node is not None:
                pid = int(policy_node)
                cid = int(company_node)
                positive_pairs.add((pid, cid))
                policy_to_positive_companies.setdefault(pid, set()).add(cid)
                supports_count += 1
    print(f"从supports边构建正样本: {supports_count} 对")
    
    policy_to_industries: Dict[int, Set[int]] = {}
    for _, row in p2e_df.iterrows():
        pred = str(row["predicate"]).lower()
        if pred == "targetsindustry":
            policy_title = str(row["subject"])
            industry_name = str(row["object"])
            policy_node = policy_map.get(policy_title)
            industry_node = industry_map.get(industry_name)
            if policy_node is not None and industry_node is not None:
                policy_to_industries.setdefault(int(policy_node), set()).add(int(industry_node))
    
    industry_to_companies: Dict[int, Set[int]] = {}
    company_to_industry: Dict[int, List[int]] = {}
    for _, row in p2e_df.iterrows():
        pred = str(row["predicate"]).lower()
        if pred == "belongsto":
            company_name = str(row["subject"])
            industry_name = str(row["object"])
            company_node = company_map.get(company_name)
            industry_node = industry_map.get(industry_name)
            if company_node is not None and industry_node is not None:
                cid = int(company_node)
                iid = int(industry_node)
                industry_to_companies.setdefault(iid, set()).add(cid)
                company_to_industry.setdefault(cid, []).append(iid)
    
    industry_pairs_count = 0
    sampled_industry_pairs_count = 0
    if not use_supports_only:
        for policy_node, industries in policy_to_industries.items():
            candidates = set()
            for industry_node in industries:
                candidates.update(industry_to_companies.get(industry_node, set()))
            industry_pairs_count += len(candidates)
            # 避免将已经存在的 supports 正样本重复加入
            candidates = candidates - policy_to_positive_companies.get(policy_node, set())
            candidates = [c for c in candidates if c in company_map.values()]
            if max_industry_pairs_per_policy > 0 and len(candidates) > max_industry_pairs_per_policy:
                candidates = random.sample(candidates, max_industry_pairs_per_policy)
            sampled_industry_pairs_count += len(candidates)
            for company_node in candidates:
                positive_pairs.add((policy_node, int(company_node)))
                policy_to_positive_companies.setdefault(policy_node, set()).add(int(company_node))

    print(f"从行业关系可构建候选正样本: {industry_pairs_count} 对")
    if use_supports_only:
        print("行业扩展正样本: 已关闭（仅使用supports）")
    else:
        print(
            f"行业扩展正样本: 已采样 {sampled_industry_pairs_count} 对 "
            f"(每个policy最多 {max_industry_pairs_per_policy})"
        )
    print(f"总正样本对数: {len(positive_pairs)}")

    ind_to_comp_list = {k: list(v) for k, v in industry_to_companies.items()}
    
    return list(positive_pairs), company_to_industry, ind_to_comp_list, policy_to_positive_companies


def train_gat_contrastive(
    project_root: Path,
    hidden_dim: int = 128,
    out_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    num_epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    num_negatives: int = 20,
    use_supports_only: bool = False,
    max_industry_pairs_per_policy: int = 30,
    max_pairs_per_epoch: int = 200000,
    transmit_drop_rate: float = 0.4,
    reverse_neighbor_cap: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    graph_bin: Optional[str] = None,
    meta_json: Optional[str] = None,
    policies_clean_path: Optional[str] = None,
    feature_override: Optional[Dict[str, str]] = None,
    checkpoint_best: Optional[str] = None,
    checkpoint_final: Optional[str] = None,
    policy_emb_out: Optional[str] = None,
    company_emb_out: Optional[str] = None,
    industry_emb_out: Optional[str] = None,
):
    """训练对比学习GAT。

    数据规模实验可传入 graph_bin / meta_json / policies_clean_path / feature_override 等，
    使图与特征来自子目录而无需覆盖默认 graph/ 与 data_intermediate/。
    """
    
    print("=" * 60, flush=True)
    print("对比学习GAT训练", flush=True)
    print("=" * 60, flush=True)
    
    device = torch.device(device)
    
    # 1. 加载图
    print("\n1. 加载图数据...", flush=True)
    g_bin = project_root / (graph_bin or "graph/graph_data.bin")
    graphs, _ = dgl.load_graphs(str(g_bin))
    g = graphs[0].to(device)
    print(f"  节点类型: {g.ntypes}", flush=True)
    print(f"  边类型: {g.canonical_etypes}", flush=True)
    for ntype in g.ntypes:
        print(f"  {ntype}: {g.number_of_nodes(ntype)} 节点", flush=True)
    
    # 2. 加载节点映射
    meta_path = project_root / (meta_json or "graph/meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        node_maps = meta.get("node_maps", {})

    # 仅对主评估集政策计算损失（避免辅助政策节点扰动训练目标）
    allowed_policy_ids: Set[int] = set()
    pol_path = Path(policies_clean_path) if policies_clean_path else project_root / "data_intermediate/policies_clean.parquet"
    pol_path = pol_path if pol_path.is_absolute() else project_root / pol_path
    if pol_path.exists():
        pol_df = pd.read_parquet(pol_path)
        allowed_policy_ids = set(pol_df["policy_id"].astype(int).tolist())
        print(f"  目标政策节点掩码: {len(allowed_policy_ids)} 个政策用于loss ({pol_path})", flush=True)

    # 3. 加载特征
    print("\n2. 加载节点特征...", flush=True)
    fo: Optional[Dict[str, Path]] = None
    if feature_override:
        fo = {k: Path(v) if Path(v).is_absolute() else project_root / v for k, v in feature_override.items()}
    features = load_fused_features(project_root, device, override_paths=fo)
    
    # 检查特征维度并对齐节点数量
    in_dims = {}
    aligned_features = {}
    
    for ntype in g.ntypes:
        num_nodes = g.number_of_nodes(ntype)
        if ntype in features:
            feat = features[ntype]
            if feat.shape[0] >= num_nodes:
                aligned_features[ntype] = feat[:num_nodes]
            else:
                # 填充零向量
                pad = torch.zeros(num_nodes - feat.shape[0], feat.shape[1], device=device)
                aligned_features[ntype] = torch.cat([feat, pad], dim=0)
            in_dims[ntype] = aligned_features[ntype].shape[1]
            print(f"  {ntype}: {aligned_features[ntype].shape}", flush=True)
        else:
            # 没有特征的节点用默认维度
            in_dims[ntype] = 768
            aligned_features[ntype] = torch.randn(num_nodes, 768, device=device) * 0.01
            print(f"  {ntype}: {num_nodes} 节点 (随机初始化)", flush=True)
    
    # 4. 构建正样本对与行业映射
    print("\n3. 构建正样本对...", flush=True)
    p2e_for_pairs = None
    if policies_clean_path:
        # 子图数据与三元组同目录
        pc = Path(policies_clean_path)
        if not pc.is_absolute():
            pc = project_root / pc
        cand = pc.parent / "triples_policy_entity.parquet"
        if cand.is_file():
            p2e_for_pairs = cand

    positive_pairs, company_to_industry, industry_to_companies_map, policy_to_positive_companies = build_positive_pairs(
        project_root,
        node_maps,
        use_supports_only=use_supports_only,
        max_industry_pairs_per_policy=max_industry_pairs_per_policy,
        p2e_parquet=p2e_for_pairs,
    )
    if not positive_pairs:
        print("错误: 无法构建正样本对", flush=True)
        return
    # 训练掩码：仅保留allowed_policy_ids中的政策节点参与InfoNCE
    if allowed_policy_ids:
        positive_pairs = [(pid, cid) for (pid, cid) in positive_pairs if pid in allowed_policy_ids]
        policy_to_positive_companies = {
            pid: comps for pid, comps in policy_to_positive_companies.items() if pid in allowed_policy_ids
        }
        print(f"  掩码后正样本对数: {len(positive_pairs)}", flush=True)
        if not positive_pairs:
            print("错误: 掩码后无正样本对", flush=True)
            return
    print(f"  行业映射: {len(company_to_industry)} 个企业, {len(industry_to_companies_map)} 个行业", flush=True)
    print(
        f"  策略: use_supports_only={use_supports_only}, "
        f"max_industry_pairs_per_policy={max_industry_pairs_per_policy}, "
        f"max_pairs_per_epoch={max_pairs_per_epoch}",
        flush=True,
    )
    
    # 5. 创建模型
    print("\n4. 创建模型...", flush=True)
    model = HeteroGATContrastive(
        in_dims=in_dims,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        edge_types=list(g.etypes),
    ).to(device)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 6. 训练
    print("\n5. 开始训练...", flush=True)
    best_loss = float('inf')
    train_start_time = time.time()
    random_baseline = math.log(num_negatives + 1)
    print(f"  理论随机基线(InfoNCE): log(1+{num_negatives}) = {random_baseline:.4f}", flush=True)
    print(f"  transmits 边dropout率: {transmit_drop_rate}", flush=True)
    print(f"  反向边邻居上限(cap): {reverse_neighbor_cap if reverse_neighbor_cap > 0 else '关闭'}", flush=True)

    def _build_epoch_graph(base_g: dgl.DGLHeteroGraph, drop_rate: float, reverse_cap: int) -> dgl.DGLHeteroGraph:
        if drop_rate <= 0 and reverse_cap <= 0:
            return base_g
        keep_edges = {}
        reverse_cap_etypes = {
            ("company", "supportedByPolicy", "policy"),
            ("industry", "targetedByPolicy", "policy"),
        }
        for c_etype in base_g.canonical_etypes:
            etype_name = c_etype[1]
            ecount = base_g.number_of_edges(c_etype)
            eids = torch.arange(ecount, device=base_g.device)
            if etype_name in ("transmitsTo", "transmitsFrom") and ecount > 0:
                keep_num = max(1, int(ecount * (1.0 - drop_rate)))
                perm = torch.randperm(ecount, device=base_g.device)[:keep_num]
                eids = eids[perm]
            if reverse_cap > 0 and c_etype in reverse_cap_etypes and eids.numel() > reverse_cap:
                _, dst_all = base_g.edges(etype=c_etype)
                # 仅对保留边集合做按目标policy节点分组采样
                eids_cpu = eids.detach().cpu().numpy().tolist()
                dst_cpu = dst_all.detach().cpu().numpy().tolist()
                dst_to_eids: Dict[int, List[int]] = {}
                for eid in eids_cpu:
                    d = int(dst_cpu[eid])
                    dst_to_eids.setdefault(d, []).append(eid)
                sampled: List[int] = []
                for _, eid_list in dst_to_eids.items():
                    if len(eid_list) > reverse_cap:
                        sampled.extend(random.sample(eid_list, reverse_cap))
                    else:
                        sampled.extend(eid_list)
                if sampled:
                    eids = torch.tensor(sampled, dtype=torch.int64, device=base_g.device)
            keep_edges[c_etype] = eids
        return dgl.edge_subgraph(base_g, keep_edges, relabel_nodes=False)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()
        if max_pairs_per_epoch > 0 and len(positive_pairs) > max_pairs_per_epoch:
            epoch_pairs = random.sample(positive_pairs, max_pairs_per_epoch)
        else:
            epoch_pairs = positive_pairs
        dataset = PolicyEnterpriseDataset(epoch_pairs)
        drop_last = len(epoch_pairs) > batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
        total_batches = len(dataloader)
        progress_interval = max(1, total_batches // 10)
        g_train = _build_epoch_graph(g, transmit_drop_rate, reverse_neighbor_cap)
        print(
            f"  [Epoch {epoch+1}/{num_epochs}] 开始训练，共 {total_batches} 个 batch "
            f"(样本数={len(epoch_pairs)})",
            flush=True,
        )
        
        for batch_idx, batch_pairs in enumerate(dataloader, start=1):
            optimizer.zero_grad()
            
            embeddings = model(g_train, aligned_features)
            policy_emb = embeddings["policy"]
            company_emb = embeddings["company"]
            
            batch_pairs = batch_pairs.to(device)
            loss = model.contrastive_loss(
                policy_emb, company_emb, batch_pairs, num_negatives,
                company_to_industry=company_to_industry,
                industry_to_companies=industry_to_companies_map,
                policy_to_positive_companies=policy_to_positive_companies,
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

            if batch_idx % progress_interval == 0 or batch_idx == total_batches:
                elapsed = time.time() - epoch_start_time
                avg_loss_so_far = total_loss / max(num_batches, 1)
                pct = 100.0 * batch_idx / max(total_batches, 1)
                eta = (elapsed / batch_idx) * (total_batches - batch_idx) if batch_idx > 0 else 0.0
                print(
                    f"    Batch {batch_idx}/{total_batches} ({pct:.1f}%) "
                    f"avg_loss={avg_loss_so_far:.4f} elapsed={_format_seconds(elapsed)} "
                    f"eta={_format_seconds(eta)}",
                    flush=True,
                )
        
        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        epoch_elapsed = time.time() - epoch_start_time
        total_elapsed = time.time() - train_start_time
        remaining_epochs = num_epochs - (epoch + 1)
        eta_all = (total_elapsed / (epoch + 1)) * remaining_epochs if epoch >= 0 else 0.0
        print(
            f"  [Epoch {epoch+1}/{num_epochs}] 完成: loss={avg_loss:.4f}, "
            f"lr={scheduler.get_last_lr()[0]:.6f}, epoch_time={_format_seconds(epoch_elapsed)}, "
            f"total_elapsed={_format_seconds(total_elapsed)}, eta_all={_format_seconds(eta_all)}",
            flush=True,
        )
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_best_p = Path(checkpoint_best) if checkpoint_best else project_root / "graph/checkpoints/gat_contrastive_best.pt"
            if not ckpt_best_p.is_absolute():
                ckpt_best_p = project_root / ckpt_best_p
            ckpt_best_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_best_p)
            print(f"    [Checkpoint] best loss 更新为 {best_loss:.4f} -> {ckpt_best_p}", flush=True)
    
    # 7. 保存最终嵌入
    print("\n6. 保存节点嵌入...", flush=True)
    model.eval()
    with torch.no_grad():
        final_emb = model(g, aligned_features)
        
        # 保存
        pol_out = Path(policy_emb_out) if policy_emb_out else project_root / "graph/gat_policy_emb_contrastive.npy"
        com_out = Path(company_emb_out) if company_emb_out else project_root / "graph/gat_company_emb_contrastive.npy"
        if not pol_out.is_absolute():
            pol_out = project_root / pol_out
        if not com_out.is_absolute():
            com_out = project_root / com_out
        pol_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(pol_out, final_emb["policy"].cpu().numpy())
        np.save(com_out, final_emb["company"].cpu().numpy())
        if "industry" in final_emb:
            ind_out = Path(industry_emb_out) if industry_emb_out else project_root / "graph/gat_industry_emb_contrastive.npy"
            if not ind_out.is_absolute():
                ind_out = project_root / ind_out
            ind_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(ind_out, final_emb["industry"].cpu().numpy())
    
    print(f"  政策嵌入: {final_emb['policy'].shape}", flush=True)
    print(f"  企业嵌入: {final_emb['company'].shape}", flush=True)
    
    ckpt_final_p = Path(checkpoint_final) if checkpoint_final else project_root / "graph/checkpoints/gat_contrastive_final.pt"
    if not ckpt_final_p.is_absolute():
        ckpt_final_p = project_root / ckpt_final_p
    ckpt_final_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_final_p)
    
    print("\n训练完成！", flush=True)
    print(f"  最佳损失: {best_loss:.4f}", flush=True)
    _pol = Path(policy_emb_out) if policy_emb_out else project_root / "graph/gat_policy_emb_contrastive.npy"
    _com = Path(company_emb_out) if company_emb_out else project_root / "graph/gat_company_emb_contrastive.npy"
    if not _pol.is_absolute():
        _pol = project_root / _pol
    if not _com.is_absolute():
        _com = project_root / _com
    print(f"  政策嵌入: {_pol}", flush=True)
    print(f"  企业嵌入: {_com}", flush=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="对比学习训练GAT")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_negatives", type=int, default=20)
    parser.add_argument("--use_supports_only", action="store_true", help="仅使用supports关系构建正样本")
    parser.add_argument("--max_industry_pairs_per_policy", type=int, default=30, help="每个policy最多采样的行业扩展正样本数")
    parser.add_argument("--max_pairs_per_epoch", type=int, default=200000, help="每轮训练最多使用的正样本数")
    parser.add_argument("--transmit_drop_rate", type=float, default=0.4, help="transmitsTo/transmitsFrom边dropout率")
    parser.add_argument("--reverse_neighbor_cap", type=int, default=0, help="反向边每个policy节点最大邻居数(<=0关闭)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--feature_policy_aligned",
        type=str,
        default=None,
        help="覆盖政策节点输入特征（512D 对齐 .npy），相对项目根；用于 A2 等独立流水线而不覆盖 policy_feature_aligned.npy",
    )
    parser.add_argument("--policy_emb_out", type=str, default=None, help="政策 GAT 输出 .npy 路径（相对项目根）")
    parser.add_argument("--company_emb_out", type=str, default=None, help="企业 GAT 输出 .npy 路径（相对项目根）")
    parser.add_argument("--industry_emb_out", type=str, default=None, help="行业 GAT 输出 .npy 路径（相对项目根）")
    parser.add_argument("--checkpoint_best", type=str, default=None, help="最佳 checkpoint .pt（相对项目根）")
    parser.add_argument("--checkpoint_final", type=str, default=None, help="最终 checkpoint .pt（相对项目根）")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    fo = None
    if args.feature_policy_aligned:
        fo = {"policy": args.feature_policy_aligned}
    
    train_gat_contrastive(
        project_root,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_negatives=args.num_negatives,
        use_supports_only=args.use_supports_only,
        max_industry_pairs_per_policy=args.max_industry_pairs_per_policy,
        max_pairs_per_epoch=args.max_pairs_per_epoch,
        transmit_drop_rate=args.transmit_drop_rate,
        reverse_neighbor_cap=args.reverse_neighbor_cap,
        device=args.device,
        feature_override=fo,
        policy_emb_out=args.policy_emb_out,
        company_emb_out=args.company_emb_out,
        industry_emb_out=args.industry_emb_out,
        checkpoint_best=args.checkpoint_best,
        checkpoint_final=args.checkpoint_final,
    )

