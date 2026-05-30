#!/usr/bin/env python3
"""Session 4 — Step 4: 抗泄露对比学习训练 (Leak-Proof Contrastive Training)

核心改进:
    1. 正样本建设 (Label Leakage Defense):
       - 仅使用 Train_supports (70% 切分) 作为直接正样本
       - 补充合法间接元路径: Policy → SubIndustry → Enterprise
       - 元路径正样本过滤: 排除 Val/Test supports 中的 (policy, enterprise) 对

    2. 负样本建设 (Hard Negative Mining):
       - 优先从同 SubIndustry 采样困难负样本
       - 严格 mask: 排除 ALL supports (Train/Val/Test) 中的企业
       - 防止 "假阴性" — 避免将真实资助关系标记为负样本

    3. 图前向传播:
       - 仅使用 message_graph.pt (含 ToUndirected 反向边)
       - 绝对不包含 supports 边
       - targetsSubIndustry 边携带 LLM confidence 作为 GAT edge_attr

    4. InfoNCE 对比损失 + 可学习温度参数
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import ToUndirected

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from layers import (
    ConfidenceAwareHeteroGAT,
    build_edge_attr_dict,
    load_node_features,
)

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Positive pair construction
# ---------------------------------------------------------------------------

def build_positive_pairs(
    train_supports: torch.Tensor,
    edge_index_dict: Dict[Tuple, torch.Tensor],
    num_enterprises: int,
    val_supports: torch.Tensor,
    test_supports: torch.Tensor,
    max_meta_per_policy: int = 30,
) -> Tuple[List[Tuple[int, int]], Dict[int, Set[int]]]:
    """构建正样本对 (policy_idx, enterprise_idx)。

    Source A — 直接监督: Train_supports 中的 (policy, enterprise) 对。
    Source B — 元路径推断: Policy → SubIndustry → Enterprise (间接)。

    元路径正样本会排除 Val/Test supports 中的对。
    """
    policy_to_pos: Dict[int, Set[int]] = defaultdict(set)

    # Source A: Train supports
    for i in range(train_supports.shape[1]):
        p = int(train_supports[0, i])
        e = int(train_supports[1, i])
        policy_to_pos[p].add(e)

    n_direct = sum(len(v) for v in policy_to_pos.values())

    # 构建 Val/Test 过滤集合 (避免元路径正样本泄露)
    val_test_pairs: Set[Tuple[int, int]] = set()
    for split in [val_supports, test_supports]:
        for i in range(split.shape[1]):
            val_test_pairs.add((int(split[0, i]), int(split[1, i])))

    # Source B: Meta-path Policy → SubIndustry → Enterprise
    # 从图边构建索引
    p2si: Dict[int, Set[int]] = defaultdict(set)  # policy → sub_industries
    si2ent: Dict[int, Set[int]] = defaultdict(set)  # sub_industry → enterprises

    for et, ei in edge_index_dict.items():
        if et[1] == "targetsSubIndustry":
            for j in range(ei.shape[1]):
                p2si[int(ei[0, j])].add(int(ei[1, j]))
        elif et[1] == "belongsTo":
            for j in range(ei.shape[1]):
                si2ent[int(ei[1, j])].add(int(ei[0, j]))

    n_meta_added = 0
    n_meta_skipped_val_test = 0

    for pid, si_set in p2si.items():
        candidates: Set[int] = set()
        for si in si_set:
            candidates.update(si2ent.get(si, set()))

        # 排除已有正样本
        existing = policy_to_pos.get(pid, set())
        candidates -= existing

        # 排除 Val/Test supports (防止泄露)
        for eid in list(candidates):
            if (pid, eid) in val_test_pairs:
                candidates.discard(eid)
                n_meta_skipped_val_test += 1

        candidates = list(candidates)
        if len(candidates) > max_meta_per_policy:
            candidates = random.sample(candidates, max_meta_per_policy)

        for eid in candidates:
            policy_to_pos[pid].add(eid)
            n_meta_added += 1

    positive_pairs = [(p, e) for p, es in policy_to_pos.items() for e in es]

    print(f"  正样本构建:")
    print(f"    Source A (Train_supports direct):    {n_direct:,}")
    print(f"    Source B (Meta-path Policy→SI→Ent):  {n_meta_added:,}")
    print(f"      └ Val/Test 过滤:                   {n_meta_skipped_val_test:,}")
    print(f"    Total positive pairs:                 {len(positive_pairs):,}")

    return positive_pairs, dict(policy_to_pos)


# ---------------------------------------------------------------------------
# Industry mapping for hard negatives
# ---------------------------------------------------------------------------

def build_industry_mapping(
    edge_index_dict: Dict[Tuple, torch.Tensor],
    num_enterprises: int,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """构建企业↔子行业映射，用于困难负样本挖掘。

    Returns:
        ent_to_si: {enterprise_idx: [sub_industry_idx, ...]}
        si_to_ent: {sub_industry_idx: [enterprise_idx, ...]}
    """
    ent_to_si: Dict[int, List[int]] = defaultdict(list)
    si_to_ent: Dict[int, List[int]] = defaultdict(list)

    for et, ei in edge_index_dict.items():
        if et[1] == "belongsTo":
            for j in range(ei.shape[1]):
                eid = int(ei[0, j])
                sid = int(ei[1, j])
                ent_to_si[eid].append(sid)
                si_to_ent[sid].append(eid)

    print(f"  Industry mapping: {len(ent_to_si)} enterprises → {len(si_to_ent)} sub-industries")
    return dict(ent_to_si), dict(si_to_ent)


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

def contrastive_loss(
    policy_emb: torch.Tensor,
    enterprise_emb: torch.Tensor,
    batch_pairs: torch.Tensor,
    temperature: torch.Tensor,
    num_negatives: int,
    ent_to_si: Dict[int, List[int]],
    si_to_ent: Dict[int, List[int]],
    policy_to_pos: Dict[int, Set[int]],
    all_supports_set: Set[Tuple[int, int]],
    num_enterprises: int,
    device: torch.device,
) -> torch.Tensor:
    """InfoNCE 对比损失，含困难负样本挖掘 + supports 掩码过滤。

    Args:
        batch_pairs: [B, 2] — (policy_idx, enterprise_idx) 正样本对。
        all_supports_set: ALL supports (Train+Val+Test) — 用于过滤假阴性。
    """
    B = batch_pairs.shape[0]
    p_idx = batch_pairs[:, 0]
    e_pos_idx = batch_pairs[:, 1]

    p_emb = policy_emb[p_idx]    # [B, D]
    e_pos_emb = enterprise_emb[e_pos_idx]  # [B, D]

    # 正样本相似度
    pos_sim = (p_emb * e_pos_emb).sum(dim=-1) / temperature  # [B]

    # —— 困难负样本采样 ——
    all_ent_ids = list(range(num_enterprises))
    neg_indices_list: List[List[int]] = []

    for i in range(B):
        pid = int(p_idx[i])
        eid_pos = int(e_pos_idx[i])

        # 已知正样本 (此 policy 的全部 supports)
        known_pos = policy_to_pos.get(pid, set()).copy()

        # 困难负样本候选: 同子行业企业
        hard_candidates: Set[int] = set()
        si_list = ent_to_si.get(eid_pos, [])
        for si in si_list:
            for e_cand in si_to_ent.get(si, []):
                if e_cand not in known_pos and (pid, e_cand) not in all_supports_set:
                    hard_candidates.add(e_cand)

        hard_candidates = list(hard_candidates)
        negs: List[int] = []

        if len(hard_candidates) >= num_negatives:
            negs = random.sample(hard_candidates, num_negatives)
        else:
            negs = list(hard_candidates)
            remaining = num_negatives - len(negs)
            # 全局候选 (排除所有正样本和 supports)
            global_candidates = [
                e for e in all_ent_ids
                if e not in known_pos and (pid, e) not in all_supports_set
            ]
            if len(global_candidates) >= remaining:
                negs += random.sample(global_candidates, remaining)
            else:
                negs += random.choices(global_candidates, k=remaining) if global_candidates else []

        neg_indices_list.append(negs)

    neg_indices = torch.tensor(neg_indices_list, dtype=torch.long, device=device)  # [B, K]
    e_neg_emb = enterprise_emb[neg_indices]  # [B, K, D]

    # 负样本相似度
    neg_sim = (p_emb.unsqueeze(1) * e_neg_emb).sum(dim=-1) / temperature  # [B, K]

    # InfoNCE
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+K]
    labels = torch.zeros(B, dtype=torch.long, device=device)

    loss = F.cross_entropy(logits, labels)
    return loss


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train_contrastive(
    message_graph_path: str,
    train_supports_path: str,
    val_supports_path: str,
    test_supports_path: str,
    full_supports_path: str,
    policy_text_emb_path: str,
    enterprise_temporal_emb_path: str,
    output_dir: str,
    hidden_dim: int = 128,
    out_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    num_epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    num_negatives: int = 20,
    max_meta_per_policy: int = 30,
    max_pairs_per_epoch: int = 200000,
    seed: int = 42,
    device_str: str = "cuda",
):
    project_root = Path(__file__).resolve().parents[2]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ================================================================
    # 1. 加载消息传递图 (零 supports)
    # ================================================================
    print("\n" + "=" * 60)
    print("[1/6] 加载消息传递图 (Message Graph)...")
    print("=" * 60)

    g = torch.load(message_graph_path, weights_only=False)
    edge_types_orig = list(g.edge_types)
    print(f"  原始边类型: {[et[1] for et in edge_types_orig]}")

    # 验证无 supports
    for et in edge_types_orig:
        assert et[1] not in ("supports", "supportedByPolicy", "supportedBy"), \
            f"LEAK: 消息传递图含禁止边 {et}"

    # 添加反向边 (ToUndirected)
    g = ToUndirected()(g)
    edge_types = list(g.edge_types)
    print(f"  加反向边后: {len(edge_types)} 种边类型")

    # 构建 edge_index_dict
    edge_index_dict: Dict[Tuple, torch.Tensor] = {}
    for et in edge_types:
        edge_index_dict[et] = g[et].edge_index.to(device)
        print(f"    {et}: {edge_index_dict[et].shape}")

    # 构建 edge_attr_dict (仅 targetsSubIndustry 有 confidence)
    edge_attr_dict = build_edge_attr_dict(g, edge_types_orig, device)
    # 反向边复制 confidence
    for et in list(edge_attr_dict.keys()):
        rev_et = (et[2], f"rev_{et[1]}", et[0])
        if rev_et in edge_index_dict and rev_et not in edge_attr_dict:
            edge_attr_dict[rev_et] = edge_attr_dict[et]

    # ================================================================
    # 2. 加载节点特征
    # ================================================================
    print("\n[2/6] 加载节点特征...")
    node_features = load_node_features(
        graph_path=message_graph_path,
        text_emb_path=policy_text_emb_path,
        enterprises_path=str(project_root / "data/processed/enterprises_final.json"),
        temporal_emb_path=enterprise_temporal_emb_path,
        device=device,
    )
    for nt, feat in node_features.items():
        print(f"  {nt}: {feat.shape}")

    in_channels = {nt: feat.shape[1] for nt, feat in node_features.items()}
    num_enterprises = g["Enterprise"].num_nodes

    # ================================================================
    # 3. 加载监督边
    # ================================================================
    print("\n[3/6] 加载监督边 (supports)...")
    train_supports = torch.load(train_supports_path, weights_only=False).to(device)
    val_supports = torch.load(val_supports_path, weights_only=False).to(device)
    test_supports = torch.load(test_supports_path, weights_only=False).to(device)
    full_supports = torch.load(full_supports_path, weights_only=False).to(device)

    # 构建全量 supports 集合 (用于负样本过滤)
    all_supports_set: Set[Tuple[int, int]] = set()
    for i in range(full_supports.shape[1]):
        all_supports_set.add((int(full_supports[0, i]), int(full_supports[1, i])))

    print(f"  Train: {train_supports.shape[1]:,}  Val: {val_supports.shape[1]:,}  "
          f"Test: {test_supports.shape[1]:,}  Full-mask: {len(all_supports_set):,}")

    # ================================================================
    # 4. 构建正样本 & 行业映射
    # ================================================================
    print("\n[4/6] 构建正样本对 & 行业映射...")
    positive_pairs, policy_to_pos = build_positive_pairs(
        train_supports, edge_index_dict, num_enterprises,
        val_supports, test_supports,
        max_meta_per_policy=max_meta_per_policy,
    )
    ent_to_si, si_to_ent = build_industry_mapping(edge_index_dict, num_enterprises)

    # ================================================================
    # 5. 创建模型
    # ================================================================
    print("\n[5/6] 创建置信度感知 HeteroGAT...")
    model = ConfidenceAwareHeteroGAT(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        edge_types=edge_types_orig,  # 不带反向边的原始类型列表
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ================================================================
    # 6. 训练
    # ================================================================
    print("\n[6/6] 开始训练...")
    random_baseline = math.log(num_negatives + 1)
    print(f"  理论随机基线 (InfoNCE): log(1+{num_negatives}) = {random_baseline:.4f}")

    best_loss = float("inf")
    train_start = time.time()

    for epoch in range(num_epochs):
        model.train()

        # 每 epoch 采样正样本子集 (控制计算量)
        if max_pairs_per_epoch > 0 and len(positive_pairs) > max_pairs_per_epoch:
            epoch_pairs = random.sample(positive_pairs, max_pairs_per_epoch)
        else:
            epoch_pairs = positive_pairs

        random.shuffle(epoch_pairs)
        total_loss = 0.0
        num_batches = 0

        # 前向传播 (整个图一次性)
        embeddings = model(node_features, edge_index_dict, edge_attr_dict)
        policy_emb = embeddings["Policy"]
        enterprise_emb = embeddings["Enterprise"]

        for b_start in range(0, len(epoch_pairs), batch_size):
            b_end = min(b_start + batch_size, len(epoch_pairs))
            batch_pairs_list = epoch_pairs[b_start:b_end]
            batch_pairs = torch.tensor(batch_pairs_list, dtype=torch.long, device=device)

            optimizer.zero_grad()

            loss = contrastive_loss(
                policy_emb=policy_emb,
                enterprise_emb=enterprise_emb,
                batch_pairs=batch_pairs,
                temperature=model.temperature if hasattr(model, 'temperature') else torch.tensor(0.07, device=device),
                num_negatives=num_negatives,
                ent_to_si=ent_to_si,
                si_to_ent=si_to_ent,
                policy_to_pos=policy_to_pos,
                all_supports_set=all_supports_set,
                num_enterprises=num_enterprises,
                device=device,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - train_start

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{num_epochs} | "
                f"loss={avg_loss:.4f} {'(*best)' if is_best else '':8s} | "
                f"lr={scheduler.get_last_lr()[0]:.6f} | "
                f"elapsed={_format_seconds(epoch_time)}"
            )

    # ================================================================
    # 保存
    # ================================================================
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        final_emb = model(node_features, edge_index_dict, edge_attr_dict)

    for ntype, emb in final_emb.items():
        fpath = output_path / f"gat_{ntype}_emb.pt"
        torch.save(emb.cpu(), fpath)
        print(f"  [SAVED] {fpath}  {list(emb.shape)}")

    ckpt_path = output_path / "gat_contrastive_best.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  [SAVED] {ckpt_path}")

    print(f"\n训练完成 — best_loss={best_loss:.4f}")
    return best_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Session 4 — 抗泄露对比学习训练")
    parser.add_argument("--message-graph", type=str,
                        default="data/processed/splits/message_graph.pt")
    parser.add_argument("--train-supports", type=str,
                        default="data/processed/splits/train_supports.pt")
    parser.add_argument("--val-supports", type=str,
                        default="data/processed/splits/val_supports.pt")
    parser.add_argument("--test-supports", type=str,
                        default="data/processed/splits/test_supports.pt")
    parser.add_argument("--full-supports", type=str,
                        default="data/processed/splits/full_supports.pt")
    parser.add_argument("--policy-text-emb", type=str,
                        default="data/processed/text_embeddings/policy_text_emb.pt")
    parser.add_argument("--enterprise-temporal-emb", type=str,
                        default="data/processed/time_embeddings/enterprise_temporal_emb.pt")
    parser.add_argument("--output-dir", type=str,
                        default="data/processed/gat_checkpoints")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-negatives", type=int, default=20)
    parser.add_argument("--max-meta-per-policy", type=int, default=30)
    parser.add_argument("--max-pairs-per-epoch", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]

    train_contrastive(
        message_graph_path=str(project_root / args.message_graph),
        train_supports_path=str(project_root / args.train_supports),
        val_supports_path=str(project_root / args.val_supports),
        test_supports_path=str(project_root / args.test_supports),
        full_supports_path=str(project_root / args.full_supports),
        policy_text_emb_path=str(project_root / args.policy_text_emb),
        enterprise_temporal_emb_path=str(project_root / args.enterprise_temporal_emb),
        output_dir=str(project_root / args.output_dir),
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_negatives=args.num_negatives,
        max_meta_per_policy=args.max_meta_per_policy,
        max_pairs_per_epoch=args.max_pairs_per_epoch,
        seed=args.seed,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
