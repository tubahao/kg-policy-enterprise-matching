#!/usr/bin/env python3
"""Session 4 — Step 3: 抗泄露图张量拆分管线 (Leak-Proof Graph Splitter)

核心防御 (Label Leakage Defense):
    1. 验证 Session 3 的 HeteroData 不含 supports 边
    2. 构建 Message_Passing 图 (仅含合法消息传递边)
    3. 从 triples_policy_entity.parquet 加载 supports 真值边
    4. 将 supports 按 7:1:2 切分为 Train/Val/Test
    5. 持久化切分索引 — 模型在任何阶段都"看不见" Test_supports

Message Edges (消息传递): transmitsTo, targetsSubIndustry, belongsTo, subClassOf
Supervision Edges (监督标签): supports → 仅用于对比学习正样本 & 评估

输入:
    data/processed/graph/hetero_graph.pt          — Session 3 图对象
    data/processed/graph/graph_meta.json           — 节点 ID 映射
    data/processed/policies_final.json             — 政策标题→ID 映射
    data/processed/enterprises_final.json           — 企业名→下标映射
    data/intermediate/triples_policy_entity.parquet — supports 三元组

输出:
    data/processed/splits/message_graph.pt          — 仅含 Message Edges 的图
    data/processed/splits/train_supports.pt          — [2, N_train]
    data/processed/splits/val_supports.pt            — [2, N_val]
    data/processed/splits/test_supports.pt           — [2, N_test]
    data/processed/splits/supports_split_meta.json   — 切分元数据
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MESSAGE_EDGE_TYPES = {"transmitsTo", "targetsSubIndustry", "belongsTo", "subClassOf"}
FORBIDDEN_EDGE_TYPES = {"supports", "supportedByPolicy", "supportedBy"}
SPLIT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}


# ---------------------------------------------------------------------------
# Load & verify
# ---------------------------------------------------------------------------

def load_hetero_graph(path: Path) -> object:
    """加载 PyG HeteroData 并验证无 supports 泄露."""
    g = torch.load(path, weights_only=False)
    edge_types = {et[1] for et in g.edge_types}

    # 验证: 消息传递边齐全
    for required in MESSAGE_EDGE_TYPES:
        assert required in edge_types, f"缺少消息传递边类型: {required}"

    # 验证: 绝对无泄露
    for forbidden in FORBIDDEN_EDGE_TYPES:
        assert forbidden not in edge_types, (
            f"数据泄露! 图中包含禁止边: {forbidden}"
        )

    print(f"  [PASS] 图包含 {len(edge_types)} 种边类型, 零 supports 泄露")
    for nt in g.node_types:
        print(f"    {nt}: {g[nt].num_nodes} 节点")

    return g


def build_title_to_policy_id(
    policies_path: Path,
) -> Dict[str, str]:
    """构建政策标题 → P_XXXX ID 的映射字典."""
    with open(policies_path, "r", encoding="utf-8") as f:
        policies = json.load(f)

    active = [p for p in policies if p.get("status") == "active"]
    title_to_id = {}
    for p in active:
        title_to_id[p["title"]] = p["policy_id"]

    print(f"  政策标题→ID 映射: {len(title_to_id)} 条 (active)")
    return title_to_id


def normalize_text(s: str) -> str:
    """去除空白和常见标点差异的文本规范化."""
    import re
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("　", "").replace("\xa0", "")
    return s


# ---------------------------------------------------------------------------
# Supports edge extraction
# ---------------------------------------------------------------------------

def extract_supports_edges(
    parquet_path: Path,
    title_to_pid: Dict[str, str],
    policy_id_to_idx: Dict[str, int],
    ent_name_to_idx: Dict[str, int],
) -> Tuple[torch.Tensor, List[dict]]:
    """从 parquet 提取 supports 边并映射到节点下标.

    Returns:
        edge_index: [2, N] tensor (policy_idx, enterprise_idx)
        unmapped: 未能映射的边列表 (用于审计)
    """
    df = pd.read_parquet(parquet_path)
    supp_df = df[df["predicate"] == "supports"].copy()

    print(f"  supports 原始三元组: {len(supp_df)} 条")

    # 构建规范化标题索引用于模糊匹配
    norm_title_to_pid = {normalize_text(t): pid for t, pid in title_to_pid.items()}
    norm_ent_to_idx = {normalize_text(n): idx for n, idx in ent_name_to_idx.items()}

    edges: List[Tuple[int, int]] = []
    unmapped: List[dict] = []

    for _, row in supp_df.iterrows():
        subject = str(row["subject"])
        obj = str(row["object"])

        # 1) 直接匹配
        pid = title_to_pid.get(subject)
        eidx = ent_name_to_idx.get(obj)

        # 2) 规范化匹配 (fallback)
        if pid is None:
            pid = norm_title_to_pid.get(normalize_text(subject))
        if eidx is None:
            eidx = norm_ent_to_idx.get(normalize_text(obj))

        if pid is None or eidx is None:
            unmapped.append({"subject": subject, "object": obj,
                             "pid_found": pid is not None,
                             "eidx_found": eidx is not None})
            continue

        policy_idx = policy_id_to_idx.get(pid)
        if policy_idx is None:
            unmapped.append({"subject": subject, "object": obj, "reason": "pid_not_in_graph"})
            continue

        edges.append((policy_idx, eidx))

    edge_index = torch.tensor(
        [[p, e] for p, e in edges], dtype=torch.long
    ).t().contiguous()  # [2, N]

    print(f"  映射成功: {edge_index.shape[1]} 条 supports 边")
    print(f"  映射失败: {len(unmapped)} 条")

    return edge_index, unmapped


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_edges(
    edge_index: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """随机切分边索引为 Train/Val/Test.

    策略: 按边随机排列后切分，确保所有 split 中的 (policy, enterprise) 对互斥。
    """
    N = edge_index.shape[1]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    n_test = N - n_train - n_val

    train_edges = edge_index[:, perm[:n_train]]
    val_edges = edge_index[:, perm[n_train:n_train + n_val]]
    test_edges = edge_index[:, perm[n_train + n_val:]]

    # 安全断言: 各 split 之间无交集
    train_set = set((int(train_edges[0, i]), int(train_edges[1, i])) for i in range(train_edges.shape[1]))
    val_set = set((int(val_edges[0, i]), int(val_edges[1, i])) for i in range(val_edges.shape[1]))
    test_set = set((int(test_edges[0, i]), int(test_edges[1, i])) for i in range(test_edges.shape[1]))

    assert len(train_set & val_set) == 0, "Train/Val 交集非空!"
    assert len(train_set & test_set) == 0, "Train/Test 交集非空!"
    assert len(val_set & test_set) == 0, "Val/Test 交集非空!"

    print(f"  Train supports: {train_edges.shape[1]}  ({train_edges.shape[1]/N:.1%})")
    print(f"  Val supports:   {val_edges.shape[1]}  ({val_edges.shape[1]/N:.1%})")
    print(f"  Test supports:  {test_edges.shape[1]}  ({test_edges.shape[1]/N:.1%})")
    print(f"  [PASS] Train/Val/Test 互斥验证通过")

    return train_edges, val_edges, test_edges


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_splits(
    output_dir: Path,
    message_graph: object,
    train_edges: torch.Tensor,
    val_edges: torch.Tensor,
    test_edges: torch.Tensor,
    full_supports: torch.Tensor,
    unmapped: List[dict],
    meta: dict,
) -> None:
    """持久化所有拆分产物."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 消息传递图 (PyG HeteroData, 原本就不含 supports)
    torch.save(message_graph, output_dir / "message_graph.pt")
    print(f"  [OK] message_graph.pt")

    # 监督边 splits
    torch.save(train_edges, output_dir / "train_supports.pt")
    torch.save(val_edges, output_dir / "val_supports.pt")
    torch.save(test_edges, output_dir / "test_supports.pt")
    print(f"  [OK] train/val/test_supports.pt")

    # 全量 supports (用于负样本过滤 mask)
    torch.save(full_supports, output_dir / "full_supports.pt")
    print(f"  [OK] full_supports.pt ({full_supports.shape[1]} edges)")

    # 元数据
    meta_path = output_dir / "supports_split_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [OK] supports_split_meta.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session 4 — 抗泄露图拆分管线"
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="data/processed/graph/hetero_graph.pt",
        help="Session 3 HeteroData 路径",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="data/processed/graph/graph_meta.json",
        help="图元数据路径 (含 node mappings)",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="data/processed/policies_final.json",
        help="policies_final.json 路径",
    )
    parser.add_argument(
        "--enterprises",
        type=str,
        default="data/processed/enterprises_final.json",
        help="enterprises_final.json 路径",
    )
    parser.add_argument(
        "--supports-parquet",
        type=str,
        default="data/intermediate/triples_policy_entity.parquet",
        help="supports 三元组 parquet 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/splits",
        help="拆分产物输出目录",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    graph_path = project_root / args.graph
    meta_path = project_root / args.meta
    policies_path = project_root / args.policies
    enterprises_path = project_root / args.enterprises
    parquet_path = project_root / args.supports_parquet
    output_dir = project_root / args.output_dir

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    assert test_ratio > 0, f"train+val ratio must be < 1.0, got {args.train_ratio}+{args.val_ratio}"

    print("=" * 60)
    print("Session 4 — Step 3: 抗泄露图拆分管线")
    print(f"  Message Edges: {sorted(MESSAGE_EDGE_TYPES)}")
    print(f"  Forbidden Edges: {sorted(FORBIDDEN_EDGE_TYPES)}")
    print(f"  Split: {args.train_ratio:.0%}/{args.val_ratio:.0%}/{test_ratio:.0%}")
    print("=" * 60)

    # ---- Phase 1: 加载 & 验证 ----
    print("\n[Phase 1] 加载 HeteroData & 验证无 supports 泄露...")
    g = load_hetero_graph(graph_path)

    print("\n[Phase 2] 加载节点映射...")
    with open(meta_path, "r", encoding="utf-8") as f:
        graph_meta = json.load(f)
    policy_id_to_idx = graph_meta["policy_id_to_idx"]
    ent_name_to_idx = graph_meta["ent_name_to_idx"]
    print(f"  policy_id_to_idx: {len(policy_id_to_idx)} 条")
    print(f"  ent_name_to_idx:  {len(ent_name_to_idx)} 条")

    print("\n[Phase 3] 加载政策标题→ID 映射...")
    title_to_pid = build_title_to_policy_id(policies_path)

    print("\n[Phase 4] 提取 supports 真值边...")
    full_supports, unmapped = extract_supports_edges(
        parquet_path, title_to_pid, policy_id_to_idx, ent_name_to_idx
    )

    if full_supports.shape[1] == 0:
        raise RuntimeError("未能提取任何 supports 边，请检查数据映射")

    # ---- Phase 5: 切分 ----
    print(f"\n[Phase 5] 随机切分 {full_supports.shape[1]} 条 supports 边...")
    train_edges, val_edges, test_edges = split_edges(
        full_supports,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
        seed=args.seed,
    )

    # ---- Phase 6: 持久化 ----
    print(f"\n[Phase 6] 保存拆分产物 → {output_dir}")

    meta = {
        "description": "Session 4 Step 3 — 抗泄露 supports 边切分",
        "generated": datetime.now().isoformat(),
        "message_edge_types": sorted(MESSAGE_EDGE_TYPES),
        "forbidden_edge_types": sorted(FORBIDDEN_EDGE_TYPES),
        "seed": args.seed,
        "total_supports": full_supports.shape[1],
        "unmapped_count": len(unmapped),
        "splits": {
            "train": {"count": train_edges.shape[1], "ratio": args.train_ratio},
            "val": {"count": val_edges.shape[1], "ratio": args.val_ratio},
            "test": {"count": test_edges.shape[1], "ratio": test_ratio},
        },
        "node_counts": graph_meta.get("node_counts", {}),
    }
    save_splits(output_dir, g, train_edges, val_edges, test_edges, full_supports, unmapped, meta)

    print("\n" + "=" * 60)
    print("[OK] 图拆分完成 — 模型前向传播图中零 supports 泄露")
    print(f"  Train: {train_edges.shape[1]} | Val: {val_edges.shape[1]} | Test: {test_edges.shape[1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
