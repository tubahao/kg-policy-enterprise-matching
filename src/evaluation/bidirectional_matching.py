#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双向匹配模块：
1. 企业/行业→政策查询：使用Attention-BLSTM计算查询相似度
2. 政策→企业检索：使用GraphRAG + GNN编码子图结构
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import os
# 设置Hugging Face镜像源（国内加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from .torch_metadata_fix import apply_torch_metadata_fix

apply_torch_metadata_fix()

# transformers 仅在 PolicyQueryMatcher 初始化时加载，便于仅依赖 PolicyToEnterpriseRetriever 的脚本运行

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def _build_policy_structure_norm_by_pid(
    gat_policy_emb: np.ndarray,
    policies_df: pd.DataFrame,
    node_maps: Dict,
) -> Dict[int, float]:
    """按 policy_id 从图节点取 GAT 行，用 L2 范数经 min-max 得到 E→P 结构分（norm 模式）。"""
    policy_map = node_maps.get("policy", {}) if node_maps else {}
    title_to_pid: Dict[str, int] = {
        str(r["title"]): int(r["policy_id"]) for _, r in policies_df.iterrows()
    }
    raw: Dict[int, float] = {}
    for title, nid in policy_map.items():
        pid = title_to_pid.get(str(title))
        if pid is None:
            continue
        ni = int(nid)
        if 0 <= ni < len(gat_policy_emb):
            raw[int(pid)] = float(np.linalg.norm(gat_policy_emb[ni]))
    if not raw:
        return {}
    vals = np.array(list(raw.values()), dtype=np.float64)
    mn, mx = float(vals.min()), float(vals.max())
    if mx - mn < 1e-12:
        return {k: 0.0 for k in raw}
    return {k: (raw[k] - mn) / (mx - mn) for k in raw}


def _build_policy_gat_alignment(
    gat_policy_emb: np.ndarray,
    policy_index: Dict,
    policies_df: pd.DataFrame,
    node_maps: Dict,
    n_emb: int,
    device: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """将图中政策节点的 GAT 向量对齐到 BERT 政策嵌入行（按 policy_id）。"""
    policy_map = node_maps.get("policy", {}) if node_maps else {}
    title_to_pid: Dict[str, int] = {
        str(r["title"]): int(r["policy_id"]) for _, r in policies_df.iterrows()
    }
    pid_to_nid: Dict[int, int] = {}
    for title, nid in policy_map.items():
        pid = title_to_pid.get(str(title))
        if pid is not None:
            pid_to_nid[int(pid)] = int(nid)
    gat_d = int(gat_policy_emb.shape[1])
    t = torch.zeros(n_emb, gat_d, dtype=torch.float32, device=device)
    valid = torch.zeros(n_emb, dtype=torch.bool, device=device)
    for pid_str, emb_idx_s in policy_index.items():
        try:
            pid = int(pid_str)
        except (ValueError, TypeError):
            continue
        emb_idx = int(emb_idx_s)
        if emb_idx < 0 or emb_idx >= n_emb:
            continue
        nid = pid_to_nid.get(pid)
        if nid is None or nid < 0 or nid >= len(gat_policy_emb):
            continue
        t[emb_idx] = torch.from_numpy(gat_policy_emb[nid].astype(np.float32)).to(device)
        valid[emb_idx] = True
    return t, valid


def _apply_rank_cutoff(
    ranked_pairs: List[Tuple[int, float]],
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    adaptive_quantile: Optional[float] = None,
    relative_drop_threshold: Optional[float] = None,
    max_output_cap: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """统一执行分数阈值、断崖截断与输出数量上限。"""
    if not ranked_pairs:
        return []

    pairs = sorted(ranked_pairs, key=lambda x: x[1], reverse=True)
    eff_threshold = score_threshold

    if (
        eff_threshold is None
        and adaptive_quantile is not None
        and 0.0 < adaptive_quantile < 1.0
        and pairs
    ):
        score_arr = np.array([s for _, s in pairs], dtype=float)
        eff_threshold = float(np.quantile(score_arr, adaptive_quantile))

    if eff_threshold is not None:
        pairs = [(nid, s) for nid, s in pairs if s >= eff_threshold]

    if relative_drop_threshold is not None and 0.0 < relative_drop_threshold < 1.0 and len(pairs) > 1:
        kept = [pairs[0]]
        for nid, s in pairs[1:]:
            prev_s = kept[-1][1]
            if prev_s > 0:
                drop_ratio = (prev_s - s) / max(prev_s, 1e-12)
                if drop_ratio > relative_drop_threshold:
                    break
            kept.append((nid, s))
        pairs = kept

    if top_k is not None and top_k > 0:
        pairs = pairs[:top_k]

    if max_output_cap is not None and max_output_cap > 0:
        pairs = pairs[:max_output_cap]

    return pairs


class AttentionBLSTM(nn.Module):
    """Attention-BLSTM模型，用于编码查询文本"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.blstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)  # 双向LSTM输出维度是hidden_dim * 2
        self.hidden_dim = hidden_dim
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) token ids
            mask: (batch_size, seq_len) attention mask
        Returns:
            query_embedding: (batch_size, hidden_dim * 2)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # BLSTM
        lstm_out, _ = self.blstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Attention机制
        attention_scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # 加权求和
        query_embedding = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_dim * 2)
        
        return query_embedding


class PolicyQueryMatcher:
    """企业/行业→政策查询匹配器（含行业增强重排序）"""
    
    def __init__(
        self,
        policy_embeddings: np.ndarray,
        policy_index: Dict[int, int],
        model_name: str = "bert-base-chinese",
        industry_mapping: Optional[Dict] = None,
        policy_industry_map: Optional[Dict[int, List[str]]] = None,
        policy_structure_scores: Optional[Dict[int, float]] = None,
        policy_importance_scores: Optional[Dict[int, float]] = None,
        allowed_policy_ids: Optional[set] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gat_emb_by_idx: Optional[torch.Tensor] = None,
        gat_emb_valid: Optional[torch.Tensor] = None,
        structure_score_mode: Optional[str] = None,
        structure_context_top_m: Optional[int] = None,
    ):
        self.policy_embeddings = torch.tensor(policy_embeddings, dtype=torch.float32).to(device)
        self.policy_index = policy_index
        self.device = device
        self.industry_mapping = industry_mapping or {}
        self.policy_industry_map = policy_industry_map or {}
        self.policy_structure_scores = policy_structure_scores or {}
        self.policy_importance_scores = policy_importance_scores or {}
        self.allowed_policy_ids = set(allowed_policy_ids) if allowed_policy_ids else None
        self.gat_emb_by_idx = gat_emb_by_idx
        self.gat_emb_valid = gat_emb_valid
        # 默认用 policy_structure_scores（GAT 行 L2 范数 min-max），与主实验冻结 JSON 口径一致。
        # gat_query_context：用当前查询的语义 Top-M 候选的 GAT 向量作上下文，再与候选余弦；
        # 在 0.45 结构权重下易与语义/重要性冲突，导致 E→P 排序与 GT 几乎无交集，仅建议显式开启评测。
        self.structure_score_mode = (
            (structure_score_mode or os.environ.get("KGE_E2P_STRUCTURE_MODE", "structure_norm")).strip().lower()
        )
        if structure_context_top_m is not None:
            self.structure_context_top_m = max(1, int(structure_context_top_m))
        else:
            try:
                self.structure_context_top_m = max(1, int(os.environ.get("KGE_E2P_STRUCTURE_TOP_M", "5")))
            except ValueError:
                self.structure_context_top_m = 5

        from transformers import AutoModel, AutoTokenizer

        # 使用BERT作为查询编码器（替代Att-BLSTM，因为BERT已经包含attention机制）
        # 使用本地缓存优先，如果不存在则从镜像源下载
        print(f"加载BERT模型: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
            self.query_encoder = AutoModel.from_pretrained(model_name, local_files_only=False).to(device)
        except Exception as e:
            print(f"警告: 加载BERT模型失败: {e}")
            print("尝试使用本地缓存...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.query_encoder = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
        self.query_encoder.eval()
        
        # 查询和政策的投影矩阵（用于注意力计算）
        self.d_k = policy_embeddings.shape[1]  # 向量维度
        
        # 获取BERT的hidden_dim
        bert_hidden_dim = self.query_encoder.config.hidden_size
        
        # 如果BERT维度与政策嵌入维度不同，需要投影层
        if bert_hidden_dim != self.d_k:
            self.query_proj = nn.Linear(bert_hidden_dim, self.d_k).to(device)
        else:
            self.query_proj = None
        
        # 评估阶段不训练投影层，使用恒等映射避免随机初始化带来的评估抖动
        self.Wq = nn.Identity().to(device)
        self.Wk = nn.Identity().to(device)

        # 预构建 embedding_idx -> policy_id 映射，并按允许集合过滤
        self.index_to_policy_id: Dict[int, int] = {}
        for pid_str, emb_idx in self.policy_index.items():
            try:
                pid = int(pid_str)
            except (ValueError, TypeError):
                continue
            if self.allowed_policy_ids is not None and pid not in self.allowed_policy_ids:
                continue
            self.index_to_policy_id[int(emb_idx)] = pid
        
    def encode_query(self, query_text: str) -> torch.Tensor:
        """使用BERT编码查询文本"""
        encoded = self.tokenizer(
            query_text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            output = self.query_encoder(**encoded)
            # 使用[CLS] token的表示作为查询向量
            query_emb = output.last_hidden_state[:, 0, :]  # (1, hidden_dim)
            
            # 如果BERT的hidden_dim与policy_embeddings的维度不同，需要投影
            if self.query_proj is not None:
                query_emb = self.query_proj(query_emb)
        
        return query_emb  # (1, d_k)
    
    def _resolve_query_industries(self, query_text: str) -> List[str]:
        """根据查询文本尝试识别其行业大类。"""
        if not self.industry_mapping:
            return []
        majors = self.industry_mapping.get(query_text)
        if majors:
            return majors
        matched = []
        for key, vals in self.industry_mapping.items():
            if key in query_text or query_text in key:
                matched.extend(vals)
        return list(set(matched))

    def compute_similarity_scores(
        self, 
        query_text: str, 
        top_k: int = 10,
        candidate_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        adaptive_quantile: Optional[float] = 0.75,
        relative_drop_threshold: Optional[float] = 0.15,
        max_output_cap: Optional[int] = 100,
        semantic_weight: float = 0.7,
        structure_weight: float = 0.2,
        importance_weight: float = 0.1,
        industry_boost: float = 0.05,
    ) -> List[Tuple[int, float]]:
        """
        计算查询与政策的相似度分数（含行业增强重排序）。
        """
        Q = self.encode_query(query_text)

        Q_proj = self.Wq(Q)
        V_policy = self.policy_embeddings
        V_policy_proj = self.Wk(V_policy)

        Q_norm = F.normalize(Q_proj, p=2, dim=1)
        V_norm = F.normalize(V_policy_proj, p=2, dim=1)

        scores = torch.matmul(Q_norm, V_norm.t())
        scores = scores.squeeze(0).detach().cpu().numpy()
        scores = (scores + 1) / 2

        # 初始候选集：可配置，默认取全部避免Top-K截断导致召回上限受限
        if candidate_k is None or candidate_k <= 0:
            candidate_k = len(scores)
        candidate_k = min(candidate_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:candidate_k]

        query_industries = set(self._resolve_query_industries(query_text))

        use_ctx = (
            self.structure_score_mode == "gat_query_context"
            and self.gat_emb_by_idx is not None
            and self.gat_emb_valid is not None
        )
        ranked_cand = sorted(
            [(int(idx), float(scores[idx])) for idx in top_indices if int(idx) in self.index_to_policy_id],
            key=lambda x: -x[1],
        )
        m_ctx = int(self.structure_context_top_m)

        results = []
        for idx in top_indices:
            if idx not in self.index_to_policy_id:
                continue
            policy_id = self.index_to_policy_id[idx]
            semantic_sim = float(scores[idx])
            idx_i = int(idx)
            if use_ctx:
                others_idx = [i for i, _ in ranked_cand if i != idx_i][:m_ctx]
                ctx_rows = [
                    self.gat_emb_by_idx[oi]
                    for oi in others_idx
                    if bool(self.gat_emb_valid[oi].item())
                ]
                if (
                    ctx_rows
                    and idx_i < self.gat_emb_by_idx.shape[0]
                    and bool(self.gat_emb_valid[idx_i].item())
                ):
                    ctx = torch.stack(ctx_rows, dim=0).mean(dim=0)
                    ctx_n = F.normalize(ctx.unsqueeze(0), p=2, dim=1)
                    cand_n = F.normalize(self.gat_emb_by_idx[idx_i].unsqueeze(0), p=2, dim=1)
                    cos = float((ctx_n * cand_n).sum().item())
                    structure_score = (cos + 1.0) / 2.0
                else:
                    structure_score = float(self.policy_structure_scores.get(policy_id, 0.0))
            else:
                structure_score = float(self.policy_structure_scores.get(policy_id, 0.0))
            importance_score = float(self.policy_importance_scores.get(policy_id, 0.0))

            s = (
                semantic_weight * semantic_sim
                + structure_weight * structure_score
                + importance_weight * importance_score
            )

            if query_industries and self.policy_industry_map:
                policy_industries = set(self.policy_industry_map.get(policy_id, []))
                if query_industries & policy_industries:
                    s += industry_boost

            results.append((policy_id, s))

        return _apply_rank_cutoff(
            results,
            top_k=top_k,
            score_threshold=score_threshold,
            adaptive_quantile=adaptive_quantile,
            relative_drop_threshold=relative_drop_threshold,
            max_output_cap=max_output_cap,
        )


class PolicyToEnterpriseRetriever:
    """政策→企业检索器（GraphRAG + GAT对比学习嵌入）"""
    
    def __init__(
        self,
        graph: dgl.DGLHeteroGraph,
        company_embeddings: np.ndarray,
        company_index: Dict[str, int],
        node_maps: Optional[Dict] = None,
        policies_df: Optional[pd.DataFrame] = None,
        gat_policy_emb: Optional[np.ndarray] = None,
        gat_company_emb: Optional[np.ndarray] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.graph = graph
        self.company_embeddings = torch.tensor(company_embeddings, dtype=torch.float32).to(device)
        self.company_index = company_index
        self.node_maps = node_maps or {}
        self.policies_df = policies_df
        self.device = device
        
        self.policy_id_to_title = {}
        if policies_df is not None:
            for _, row in policies_df.iterrows():
                policy_id = int(row["policy_id"])
                title = str(row["title"])
                self.policy_id_to_title[policy_id] = title
        
        self.gat_policy_emb = None
        self.gat_company_emb = None
        if gat_policy_emb is not None:
            self.gat_policy_emb = torch.tensor(gat_policy_emb, dtype=torch.float32).to(device)
            self.gat_policy_emb = F.normalize(self.gat_policy_emb, p=2, dim=1)
        if gat_company_emb is not None:
            self.gat_company_emb = torch.tensor(gat_company_emb, dtype=torch.float32).to(device)
            self.gat_company_emb = F.normalize(self.gat_company_emb, p=2, dim=1)

        # 预构建 policy -> 直接supports企业集合，用于先验重排
        self.policy_direct_support_companies: Dict[int, set] = {}
        try:
            if ("policy", "supports", "company") in self.graph.canonical_etypes:
                src, dst = self.graph.edges(etype=("policy", "supports", "company"))
                src_np = src.numpy()
                dst_np = dst.numpy()
                for p, c in zip(src_np.tolist(), dst_np.tolist()):
                    self.policy_direct_support_companies.setdefault(int(p), set()).add(int(c))
        except Exception:
            # 失败时忽略，不影响主流程
            self.policy_direct_support_companies = {}

    def _resolve_policy_node_id(self, policy_id: int) -> Optional[int]:
        """将 policies_clean 的 policy_id 映射到图中的 policy 节点ID。"""
        policy_node_id = None
        if "policy" in self.node_maps:
            if policy_id in self.policy_id_to_title:
                policy_title = self.policy_id_to_title[policy_id]
                policy_node_id = self.node_maps["policy"].get(policy_title)
            if policy_node_id is None:
                for _, node_id in self.node_maps["policy"].items():
                    if node_id == policy_id:
                        policy_node_id = node_id
                        break
        return policy_node_id
    
    def sample_enterprise_subgraphs(
        self, 
        policy_id: int, 
        top_k: int = 50,
        k_hop: int = 2
    ) -> List[int]:
        """
        Step1: 从企业图谱采样子图（PageRank选Top-k相关企业）
        
        Args:
            policy_id: 政策ID（policies_clean.parquet中的policy_id）
            top_k: 选择前k个相关企业
            k_hop: k-hop子图采样
            
        Returns:
            企业节点ID列表
        """
        # 获取政策节点在图中的ID
        policy_node_id = self._resolve_policy_node_id(policy_id)
        
        if policy_node_id is None:
            print(f"警告: 无法找到政策ID {policy_id} 对应的图节点")
            print(f"  提示: node_maps中有 {len(self.node_maps.get('policy', {}))} 个政策节点")
            if self.policy_id_to_title:
                print(f"  提示: policy_id_to_title中有 {len(self.policy_id_to_title)} 个政策")
            return []
        
        # 采样k-hop子图
        try:
            # 检查政策节点是否存在
            if policy_node_id >= self.graph.number_of_nodes("policy"):
                print(f"警告: 政策节点ID {policy_node_id} 超出范围（图中只有 {self.graph.number_of_nodes('policy')} 个政策节点）")
                return []
            
            subgraph, _ = dgl.khop_out_subgraph(
                self.graph, 
                {"policy": [policy_node_id]}, 
                k=k_hop
            )
            
            # 提取企业节点
            if "company" in subgraph.ntypes:
                company_nodes = subgraph.nodes("company").numpy().tolist()
                
                if len(company_nodes) == 0:
                    print(f"提示: 政策节点 {policy_node_id} 的 {k_hop}-hop子图中没有企业节点")
                    # 尝试检查是否有直接连接的政策-企业边
                    if ("policy", "supports", "company") in self.graph.canonical_etypes:
                        src, dst = self.graph.edges(etype=("policy", "supports", "company"))
                        src_np = src.numpy()
                        dst_np = dst.numpy()
                        mask = src_np == policy_node_id
                        if mask.any():
                            direct_companies = dst_np[mask].tolist()
                            if direct_companies:
                                print(f"  找到 {len(direct_companies)} 个直接连接的企业")
                                return direct_companies[:top_k]
                    return []
                
                # 如果企业节点太多，使用PageRank选择top-k
                if len(company_nodes) > top_k:
                    # 尝试构建企业子图（如果有company-company边）
                    try:
                        # 检查是否有company-company边
                        company_etypes = [etype for etype in subgraph.canonical_etypes if etype[0] == "company" and etype[2] == "company"]
                        if company_etypes:
                            company_subgraph = subgraph[company_etypes[0]]
                            if company_subgraph.num_nodes() > 0:
                                import networkx as nx
                                nx_g = company_subgraph.to_networkx().to_undirected()
                                pr = nx.pagerank(nx_g)
                                sorted_companies = sorted(pr.items(), key=lambda x: x[1], reverse=True)
                                company_nodes = [cid for cid, _ in sorted_companies[:top_k]]
                        else:
                            # 如果没有company-company边，直接取前top_k个
                            company_nodes = company_nodes[:top_k]
                    except Exception as e:
                        # 如果出错，直接取前top_k个
                        company_nodes = company_nodes[:top_k]
                
                return company_nodes
            else:
                print(f"提示: 子图中没有company节点类型")
                return []
        except Exception as e:
            print(f"采样子图时出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def encode_subgraph_with_gnn(
        self, 
        policy_node_id: int,
        company_nodes: List[int],
        direct_support_boost: float = 0.2,
    ) -> List[Tuple[int, float]]:
        """使用GAT对比学习嵌入的余弦相似度对企业排序。"""
        priority_scores = []

        policy_emb = None
        if self.gat_policy_emb is not None and policy_node_id < self.gat_policy_emb.shape[0]:
            policy_emb = self.gat_policy_emb[policy_node_id]
        direct_support_set = self.policy_direct_support_companies.get(int(policy_node_id), set())

        for cid in company_nodes:
            score = 0.0

            if policy_emb is not None and self.gat_company_emb is not None and cid < self.gat_company_emb.shape[0]:
                company_emb = self.gat_company_emb[cid]
                score = (policy_emb * company_emb).sum().item()
                score = (score + 1.0) / 2.0
            else:
                cid_key = f"enterprise_{cid}"
                cid_str = str(cid)
                emb_idx = (self.company_index.get(cid_key) or
                           self.company_index.get(cid_str) or
                           self.company_index.get(cid))
                if emb_idx is not None and emb_idx < len(self.company_embeddings):
                    company_emb = self.company_embeddings[emb_idx]
                    score = torch.norm(company_emb).item()

            # 先验重排：若图中存在直接 supports 边，给予固定加分
            if cid in direct_support_set:
                score += direct_support_boost

            priority_scores.append((cid, score))
        
        priority_scores.sort(key=lambda x: x[1], reverse=True)
        return priority_scores
    
    def retrieve_enterprises(
        self, 
        policy_id: int, 
        top_k: int = 50,
        k_hop: int = 2,
        score_threshold: Optional[float] = None,
        candidate_k: Optional[int] = None,
        adaptive_quantile: Optional[float] = None,
        relative_drop_threshold: Optional[float] = 0.15,
        max_output_cap: Optional[int] = 100,
        direct_support_boost: float = 0.2,
    ) -> List[Tuple[int, float]]:
        """
        检索与政策相关的企业（按优先级排序）
        
        Args:
            policy_id: 政策ID（policies_clean.parquet中的policy_id）
            top_k: 返回前k个企业
            k_hop: k-hop子图采样
            
        Returns:
            List of (company_id, priority_score) tuples
        """
        # Step1: 采样子图
        sample_k = candidate_k if candidate_k is not None and candidate_k > 0 else top_k
        company_nodes = self.sample_enterprise_subgraphs(policy_id, sample_k, k_hop)
        
        if not company_nodes:
            return []
        
        # Step2: 用GAT嵌入的余弦相似度排序（必须使用图节点ID）
        policy_node_id = self._resolve_policy_node_id(policy_id)
        if policy_node_id is None:
            return []
        try:
            priority_scores = self.encode_subgraph_with_gnn(
                policy_node_id,
                company_nodes,
                direct_support_boost=direct_support_boost,
            )
        except Exception as e:
            print(f"编码子图时出错: {e}")
            priority_scores = [(cid, 1.0) for cid in company_nodes]

        return _apply_rank_cutoff(
            priority_scores,
            top_k=top_k,
            score_threshold=score_threshold,
            adaptive_quantile=adaptive_quantile,
            relative_drop_threshold=relative_drop_threshold,
            max_output_cap=max_output_cap,
        )


def load_policy_to_enterprise_retriever(
    project_root: Path,
    company_emb_path: str = "embeddings/enterprise_text_emb.npy",
    company_index_path: str = "embeddings/enterprise_index.json",
    graph_path: str = "graph/graph_data.bin",
    graph_meta_path: str = "graph/meta.json",
    device: Optional[str] = None,
) -> PolicyToEnterpriseRetriever:
    """
    仅加载政策→企业检索所需资源，不加载 BERT / PolicyQueryMatcher。
    供传导效能、行业覆盖率等脚本复用 retrieve_enterprises 逻辑。
    """
    root = Path(project_root)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("加载企业嵌入向量…")
    company_emb = np.load(root / company_emb_path)
    with open(root / company_index_path, "r", encoding="utf-8") as f:
        company_index = json.load(f)
    print("加载图数据…")
    graphs, _ = dgl.load_graphs(str(root / graph_path))
    graph = graphs[0]
    node_maps: Dict = {}
    if (root / graph_meta_path).exists():
        with open(root / graph_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            node_maps = meta.get("node_maps", {})
    policies_df = None
    try:
        policies_df = pd.read_parquet(root / "data_intermediate/policies_clean.parquet")
        print(f"加载政策主表: {len(policies_df)} 条")
    except Exception as e:
        print(f"警告: 无法加载 policies_clean: {e}")
    gat_policy_emb = None
    gat_company_emb = None
    gat_policy_path = root / "graph/gat_policy_emb_contrastive.npy"
    gat_company_path = root / "graph/gat_company_emb_contrastive.npy"
    if gat_policy_path.exists() and gat_company_path.exists():
        gat_policy_emb = np.load(gat_policy_path)
        gat_company_emb = np.load(gat_company_path)
        print(f"加载 GAT 嵌入: policy {gat_policy_emb.shape}, company {gat_company_emb.shape}")
    return PolicyToEnterpriseRetriever(
        graph,
        company_emb,
        company_index,
        node_maps=node_maps,
        policies_df=policies_df,
        gat_policy_emb=gat_policy_emb,
        gat_company_emb=gat_company_emb,
        device=device,
    )


class BidirectionalMatcher:
    """双向匹配主类"""
    
    def __init__(
        self,
        project_root: Path,
        policy_emb_path: Optional[str] = None,
        policy_index_path: Optional[str] = None,
        company_emb_path: str = "embeddings/enterprise_text_emb.npy",
        company_index_path: str = "embeddings/enterprise_index.json",
        graph_path: str = "graph/graph_data.bin",
        graph_meta_path: str = "graph/meta.json",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gat_artifact_tag: Optional[str] = None,
        policy_importance_parquet: Optional[str] = None,
        ignore_env_importance_override: bool = False,
    ):
        """
        初始化双向匹配器
        
        Args:
            project_root: 项目根目录
            policy_emb_path: 政策嵌入向量路径（None 时由环境变量 KGE_POLICY_TEXT_MODE 或默认 concat 决定）
            policy_index_path: 政策索引映射路径（须与 policy_emb_path 成对指定或同时为 None）
            company_emb_path: 企业嵌入向量路径
            company_index_path: 企业索引映射路径
            graph_path: 图数据路径
            graph_meta_path: 图元信息路径（包含节点映射）
            device: 计算设备
            gat_artifact_tag: 非空时加载带后缀的 GAT 嵌入与重要性 Parquet（如 a2_joint）；默认读环境变量 KGE_GAT_ARTIFACT_TAG
            policy_importance_parquet: 相对项目根的重要性 parquet；非空时优先于环境变量（见 gat_importance_defaults）
            ignore_env_importance_override: True 时在有 tag 且未显式传 parquet 路径下忽略 KGE_POLICY_IMPORTANCE_PARQUET
        """
        from .policy_embedding_defaults import resolve_policy_embedding_paths
        from .gat_importance_defaults import resolve_gat_importance_paths

        policy_emb_path, policy_index_path = resolve_policy_embedding_paths(
            policy_emb_path, policy_index_path
        )
        gat_policy_rel, gat_company_rel, importance_rel = resolve_gat_importance_paths(
            gat_artifact_tag,
            importance_parquet=policy_importance_parquet,
            ignore_env_importance_override=ignore_env_importance_override,
        )
        self.project_root = Path(project_root)
        self.device = device
        
        # 加载行业映射
        industry_mapping = {}
        policy_industry_map: Dict[int, List[str]] = {}
        ind_map_path = self.project_root / "industry_mapping_complete.json"
        if ind_map_path.exists():
            with open(ind_map_path, 'r', encoding='utf-8') as f:
                ind_data = json.load(f)
            for fine, info in ind_data.get("fine_industry_mapping", {}).items():
                industry_mapping[fine] = info.get("majors", [])
            for major in ind_data.get("major_industries", []):
                industry_mapping[major] = [major]
            print(f"加载行业映射: {len(industry_mapping)} 条")

        # 构建 policy_id -> 行业大类列表 映射
        p2e_path = self.project_root / "data_intermediate/triples_policy_entity.parquet"
        if p2e_path.exists():
            df_p2e = pd.read_parquet(p2e_path)
            policies_clean_path = self.project_root / "data_intermediate/policies_clean.parquet"
            title_to_pid: Dict[str, int] = {}
            if policies_clean_path.exists():
                _df_pol = pd.read_parquet(policies_clean_path)
                title_to_pid = {str(r["title"]): int(r["policy_id"]) for _, r in _df_pol.iterrows()}
            for _, row in df_p2e.iterrows():
                if str(row.get("predicate", "")).lower() == "targetsindustry":
                    pid = title_to_pid.get(str(row["subject"]))
                    if pid is not None:
                        policy_industry_map.setdefault(pid, []).append(str(row["object"]))

        # 加载政策数据
        print("加载政策嵌入向量...")
        policy_emb = np.load(self.project_root / policy_emb_path)
        with open(self.project_root / policy_index_path, 'r', encoding='utf-8') as f:
            policy_index = json.load(f)

        # 提前加载政策主表，用于 policy_id 合法集合过滤
        policies_df = None
        valid_policy_ids = None
        try:
            policies_df = pd.read_parquet(self.project_root / "data_intermediate/policies_clean.parquet")
            valid_policy_ids = set(policies_df["policy_id"].astype(int).tolist())
            print(f"加载政策数据: {len(policies_df)} 个政策")
        except Exception as e:
            print(f"警告: 无法加载政策数据: {e}")

        node_maps: Dict = {}
        _meta_path = self.project_root / graph_meta_path
        if _meta_path.is_file():
            with open(_meta_path, "r", encoding="utf-8") as f:
                node_maps = json.load(f).get("node_maps", {})

        # 加载GAT对比学习嵌入（如果存在）并构建结构分数
        gat_policy_emb = None
        gat_company_emb = None
        policy_structure_scores: Dict[int, float] = {}
        gat_emb_by_idx: Optional[torch.Tensor] = None
        gat_emb_valid: Optional[torch.Tensor] = None
        gat_policy_path = self.project_root / gat_policy_rel
        gat_company_path = self.project_root / gat_company_rel
        if gat_policy_path.exists() and gat_company_path.exists():
            gat_policy_emb = np.load(gat_policy_path)
            gat_company_emb = np.load(gat_company_path)
            if policies_df is not None and len(policies_df) > 0:
                policy_structure_scores = _build_policy_structure_norm_by_pid(
                    gat_policy_emb, policies_df, node_maps
                )
                gat_emb_by_idx, gat_emb_valid = _build_policy_gat_alignment(
                    gat_policy_emb,
                    policy_index,
                    policies_df,
                    node_maps,
                    int(policy_emb.shape[0]),
                    device,
                )
            print(
                f"加载GAT对比学习嵌入: 政策{gat_policy_emb.shape}, 企业{gat_company_emb.shape} "
                f"({gat_policy_rel}, {gat_company_rel})"
            )
        else:
            print("提示: GAT对比学习嵌入未找到，使用文本嵌入回退")

        # 加载政策重要性（PPR+衰减融合）分数
        policy_importance_scores: Dict[int, float] = {}
        importance_path = self.project_root / importance_rel
        if importance_path.exists():
            try:
                imp_df = pd.read_parquet(importance_path)
                score_col = "combined_decayed"
                if score_col in imp_df.columns and "policy_id" in imp_df.columns:
                    vals = imp_df[score_col].astype(float).values
                    if vals.max() - vals.min() > 1e-12:
                        vals = (vals - vals.min()) / (vals.max() - vals.min())
                    for pid, score in zip(imp_df["policy_id"].astype(int).tolist(), vals.tolist()):
                        policy_importance_scores[pid] = float(score)
                    print(f"加载政策重要性分数: {len(policy_importance_scores)} 条")
            except Exception as e:
                print(f"警告: 加载政策重要性分数失败: {e}")

        self.policy_query_matcher = PolicyQueryMatcher(
            policy_emb,
            policy_index,
            industry_mapping=industry_mapping,
            policy_industry_map=policy_industry_map,
            policy_structure_scores=policy_structure_scores,
            policy_importance_scores=policy_importance_scores,
            allowed_policy_ids=valid_policy_ids,
            device=device,
            gat_emb_by_idx=gat_emb_by_idx,
            gat_emb_valid=gat_emb_valid,
        )
        
        # 加载企业数据
        print("加载企业嵌入向量...")
        company_emb = np.load(self.project_root / company_emb_path)
        with open(self.project_root / company_index_path, 'r', encoding='utf-8') as f:
            company_index = json.load(f)
        
        # 加载图数据（node_maps 已在上方从 meta.json 读取）
        print("加载图数据...")
        graphs, _ = dgl.load_graphs(str(self.project_root / graph_path))
        graph = graphs[0]

        self._policies_df = policies_df
        self._node_maps = node_maps

        self.enterprise_retriever = PolicyToEnterpriseRetriever(
            graph,
            company_emb,
            company_index,
            node_maps=node_maps,
            policies_df=policies_df,
            gat_policy_emb=gat_policy_emb,
            gat_company_emb=gat_company_emb,
            device=device
        )

        print("双向匹配器初始化完成！")

    def refresh_e2p_structure_from_gat_np(self, gat_policy_emb: np.ndarray) -> None:
        """P→E 侧替换 GAT 矩阵后，同步 E→P 的范数结构分与按 embedding 行对齐的 GAT 张量（如 C2）。"""
        pr = self.policy_query_matcher
        policies_df = self._policies_df
        node_maps = self._node_maps
        n_emb = int(pr.policy_embeddings.shape[0])
        device = pr.device
        if policies_df is None or len(policies_df) == 0 or gat_policy_emb is None or len(gat_policy_emb) == 0:
            pr.policy_structure_scores = {}
            pr.gat_emb_by_idx = None
            pr.gat_emb_valid = None
            return
        pr.policy_structure_scores = _build_policy_structure_norm_by_pid(
            gat_policy_emb, policies_df, node_maps
        )
        t, v = _build_policy_gat_alignment(
            gat_policy_emb, pr.policy_index, policies_df, node_maps, n_emb, device
        )
        pr.gat_emb_by_idx = t
        pr.gat_emb_valid = v

    def query_policies_by_enterprise(
        self, 
        query_text: str, 
        top_k: int = 10,
        candidate_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        adaptive_quantile: Optional[float] = 0.75,
        relative_drop_threshold: Optional[float] = 0.15,
        max_output_cap: Optional[int] = 100,
        semantic_weight: float = 0.7,
        structure_weight: float = 0.2,
        importance_weight: float = 0.1,
        industry_boost: float = 0.05,
    ) -> List[Tuple[int, float]]:
        """
        企业/行业→政策查询
        
        Args:
            query_text: 查询文本（如"开饭店"）
            top_k: 返回前k个政策
            
        Returns:
            List of (policy_id, score) tuples
        """
        return self.policy_query_matcher.compute_similarity_scores(
            query_text,
            top_k=top_k,
            candidate_k=candidate_k,
            score_threshold=score_threshold,
            adaptive_quantile=adaptive_quantile,
            relative_drop_threshold=relative_drop_threshold,
            max_output_cap=max_output_cap,
            semantic_weight=semantic_weight,
            structure_weight=structure_weight,
            importance_weight=importance_weight,
            industry_boost=industry_boost,
        )
    
    def retrieve_enterprises_by_policy(
        self, 
        policy_id: int, 
        top_k: int = 50,
        k_hop: int = 2,
        score_threshold: Optional[float] = None,
        candidate_k: Optional[int] = None,
        adaptive_quantile: Optional[float] = None,
        relative_drop_threshold: Optional[float] = 0.15,
        max_output_cap: Optional[int] = 100,
        direct_support_boost: float = 0.2,
    ) -> List[Tuple[int, float]]:
        """
        政策→企业检索
        
        Args:
            policy_id: 政策ID
            top_k: 返回前k个企业
            k_hop: 子图采样跳数（2-hop 内无企业时可增大，如 4）
            
        Returns:
            List of (company_id, priority_score) tuples
        """
        return self.enterprise_retriever.retrieve_enterprises(
            policy_id,
            top_k=top_k,
            k_hop=k_hop,
            score_threshold=score_threshold,
            candidate_k=candidate_k,
            adaptive_quantile=adaptive_quantile,
            relative_drop_threshold=relative_drop_threshold,
            max_output_cap=max_output_cap,
            direct_support_boost=direct_support_boost,
        )


def main():
    """测试双向匹配功能"""
    import argparse
    
    parser = argparse.ArgumentParser(description="双向匹配测试")
    parser.add_argument("--query", type=str, default="开饭店", help="查询文本")
    parser.add_argument("--policy_id", type=int, default=0, help="政策ID")
    parser.add_argument("--top_k", type=int, default=10, help="返回前k个结果")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    matcher = BidirectionalMatcher(project_root)
    
    # 测试企业→政策查询
    print(f"\n=== 企业/行业→政策查询 ===")
    print(f"查询文本: {args.query}")
    results = matcher.query_policies_by_enterprise(args.query, top_k=args.top_k)
    print(f"找到 {len(results)} 个相关政策:")
    for i, (pid, score) in enumerate(results, 1):
        print(f"  {i}. 政策ID: {pid}, 相似度分数: {score:.4f}")
    
    # 测试政策→企业检索
    print(f"\n=== 政策→企业检索 ===")
    print(f"政策ID: {args.policy_id}")
    results = matcher.retrieve_enterprises_by_policy(args.policy_id, top_k=args.top_k)
    print(f"找到 {len(results)} 个相关企业:")
    for i, (cid, score) in enumerate(results, 1):
        print(f"  {i}. 企业ID: {cid}, 优先级分数: {score:.4f}")


if __name__ == "__main__":
    main()

