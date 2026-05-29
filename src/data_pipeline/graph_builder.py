#!/usr/bin/env python3
"""Session 3 — Task 2+3: 靶向继承 + PyG 异构图构建 (Graph Builder)

Phase A — 基于行政依赖的靶向继承 (Target Inheritance):
  解决第 3 层"能量死胡同"问题。扫描所有接收 transmitsTo 入边
  但无 targetsSubIndustry 出边的政策，向上追溯父政策链，
  继承父政策的行业靶向边 (置信度 × 0.8^hops)。

Phase B — PyG HeteroData 图构建:
  节点类型: Policy (1,892), SubIndustry (~61), MajorIndustry (6), Enterprise (6,393)
  边类型: transmitsTo, targetsSubIndustry, belongsTo, subClassOf
  绝对不含 supports 边 (Session 4/5 的 Ground Truth 评估标签)。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

try:
    from torch_geometric.data import HeteroData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    HeteroData = None


# ---------------------------------------------------------------------------
# Phase A: 靶向继承 (Target Inheritance)
# ---------------------------------------------------------------------------

def build_parent_map(
    transmits_to_edges: List[dict],
) -> Dict[str, List[str]]:
    """
    从 transmitsTo 边构建 child→[parents] 映射。

    边方向: 上级Policy(subject) → transmitsTo → 下级Policy(object)
    因此通过 object(下级) 找 subject(上级) 来追溯父政策。
    """
    parent_map: Dict[str, List[str]] = defaultdict(list)
    for e in transmits_to_edges:
        child = e["object"]   # 下级政策 (接收传导)
        parent = e["subject"] # 上级政策 (发出传导)
        parent_map[child].append(parent)
    return dict(parent_map)


def build_target_map(
    targets_si_edges: List[dict],
) -> Dict[str, List[dict]]:
    """构建 policy_id → [targetsSubIndustry 边列表] 映射."""
    target_map: Dict[str, List[dict]] = defaultdict(list)
    for e in targets_si_edges:
        target_map[e["subject"]].append(e)
    return dict(target_map)


def find_targets_in_chain(
    policy_id: str,
    target_map: Dict[str, List[dict]],
    parent_map: Dict[str, List[str]],
    visited: Set[str],
    depth: int = 0,
    max_depth: int = 5,
) -> List[Tuple[dict, int]]:
    """
    递归向上追溯父政策链，找到第一个有 targetsSubIndustry 的祖先。

    返回: [(target_edge, hops), ...]  其中 hops 是距离
    """
    if policy_id in visited or depth >= max_depth:
        return []
    visited.add(policy_id)

    # 如果当前政策有靶向行业，直接返回
    if policy_id in target_map and target_map[policy_id]:
        return [(e, depth) for e in target_map[policy_id]]

    # 否则继续向上追溯
    results: List[Tuple[dict, int]] = []
    for parent_id in parent_map.get(policy_id, []):
        results.extend(
            find_targets_in_chain(
                parent_id, target_map, parent_map, visited, depth + 1, max_depth
            )
        )
    return results


def compute_target_inheritance(
    edges: Dict[str, List[dict]],
    all_policy_ids: Set[str],
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    为目标继承执行主逻辑。
    返回: (new_inherited_edges, inheritance_report)
    """
    transmits_to = edges.get("transmitsTo", [])
    original_targets = edges.get("targetsSubIndustry", [])

    parent_map = build_parent_map(transmits_to)
    target_map = build_target_map(original_targets)

    # 所有有入边的政策
    policies_with_incoming = set(parent_map.keys())

    # 所有有靶向行业的政策
    policies_with_targets = set(target_map.keys())

    # 死胡同: 有 transmitsTo 入边但无 targetsSubIndustry 出边
    dead_ends = policies_with_incoming - policies_with_targets
    dead_ends_sorted = sorted(dead_ends)

    print(f"\n  传导接收方 (有 transmitsTo 入边): {len(policies_with_incoming)}")
    print(f"  有靶向行业 (有 targetsSubIndustry 出边): {len(policies_with_targets)}")
    print(f"  死胡同政策 (有入边无出边): {len(dead_ends)}")

    # 对每个死胡同尝试继承
    inherited_edges: List[dict] = []
    activated: List[dict] = []
    still_dead: List[str] = []
    chain_depth_dist: Dict[int, int] = defaultdict(int)

    for pid in dead_ends_sorted:
        visited: Set[str] = set()
        ancestor_targets = find_targets_in_chain(
            pid, target_map, parent_map, visited, depth=0, max_depth=5
        )

        if not ancestor_targets:
            still_dead.append(pid)
            continue

        # 去重: 同一 (policy, sub_industry) 只保留最高置信度
        best_per_si: Dict[str, Tuple[dict, float]] = {}
        for target_edge, hops in ancestor_targets:
            si_id = target_edge["object"]
            decayed_conf = target_edge["confidence"] * (0.8 ** hops)
            key = (pid, si_id)
            if key not in best_per_si or decayed_conf > best_per_si[key][1]:
                best_per_si[key] = (target_edge, decayed_conf)

        activated_info = {
            "policy_id": pid,
            "num_inherited": len(best_per_si),
            "ancestors_consulted": len(ancestor_targets),
            "targets": [],
        }

        for (_, si_id), (target_edge, decayed_conf) in best_per_si.items():
            hops = next(
                h for e, h in ancestor_targets if e["object"] == si_id
            )
            chain_depth_dist[hops] += 1
            new_edge = {
                "subject": pid,
                "subject_type": target_edge.get("subject_type", "Policy"),
                "predicate": "targetsSubIndustry",
                "object": si_id,
                "object_type": "SubIndustry",
                "object_name": target_edge.get("object_name", ""),
                "confidence": round(decayed_conf, 6),
                "match_method": "inherited",
                "inherited_from": target_edge["subject"],
                "inheritance_hops": hops,
            }
            inherited_edges.append(new_edge)
            activated_info["targets"].append({
                "sub_industry_id": si_id,
                "sub_industry_name": target_edge.get("object_name", ""),
                "inherited_from": target_edge["subject"],
                "original_confidence": target_edge["confidence"],
                "decayed_confidence": round(decayed_conf, 6),
                "hops": hops,
            })

        activated.append(activated_info)

    # 继承报告
    report = {
        "title": "Session 3 — Target Inheritance Report",
        "generated": datetime.now().isoformat(),
        "summary": {
            "total_policies_with_incoming_transmitsTo": len(policies_with_incoming),
            "total_policies_with_original_targets": len(policies_with_targets),
            "total_dead_ends": len(dead_ends),
            "activated_by_inheritance": len(activated),
            "still_dead_no_ancestor_targets": len(still_dead),
            "total_inherited_edges": len(inherited_edges),
            "chain_depth_distribution": dict(chain_depth_dist),
            "avg_inherited_per_activated": (
                round(len(inherited_edges) / len(activated), 1) if activated else 0
            ),
        },
        "activated_samples": activated[:20],
        "still_dead_samples": still_dead[:20],
        "still_dead_all": still_dead if len(still_dead) <= 50 else None,
    }

    # 打印摘要
    print(f"  已激活 (通过继承): {len(activated)}")
    print(f"  仍死胡同 (整条祖先链无靶向): {len(still_dead)}")
    print(f"  继承边总数: {len(inherited_edges)}")
    print(f"  链路深度分布: {dict(sorted(chain_depth_dist.items()))}")

    if still_dead:
        print(f"  仍死胡同样例: {still_dead[:10]}")

    return inherited_edges, report


# ---------------------------------------------------------------------------
# Phase B: PyG 异构图构建
# ---------------------------------------------------------------------------

def build_node_mappings(
    policies: List[dict],
    enterprises: List[dict],
    subclass_edges: List[dict],
) -> Dict[str, Any]:
    """构建确定性的 name/id → integer index 映射."""
    # Policy 节点: 按 policy_id 排序
    policy_ids = sorted(
        p["policy_id"] for p in policies if p.get("status") == "active"
    )
    policy_id_to_idx = {pid: i for i, pid in enumerate(policy_ids)}

    # Enterprise 节点: 按企业名去重排序 (Session 1 数据中可能存在同名企业)
    ent_names = sorted(set(e["name"] for e in enterprises))
    ent_name_to_idx = {name: i for i, name in enumerate(ent_names)}

    # SubIndustry 节点: 从 subClassOf subject 提取, 排序
    si_ids = sorted(set(e["subject"] for e in subclass_edges))
    si_id_to_idx = {si: i for i, si in enumerate(si_ids)}

    # MajorIndustry 节点: 从 subClassOf object 提取, 排序
    mi_ids = sorted(set(e["object"] for e in subclass_edges))
    mi_id_to_idx = {mi: i for i, mi in enumerate(mi_ids)}

    return {
        "policy_id_to_idx": policy_id_to_idx,
        "ent_name_to_idx": ent_name_to_idx,
        "si_id_to_idx": si_id_to_idx,
        "mi_id_to_idx": mi_id_to_idx,
        "num_policies": len(policy_id_to_idx),
        "num_enterprises": len(ent_name_to_idx),
        "num_sub_industries": len(si_id_to_idx),
        "num_major_industries": len(mi_id_to_idx),
    }


def build_edge_index(
    edges: List[dict],
    src_key: str,
    dst_key: str,
    src_map: Dict[str, int],
    dst_map: Dict[str, int],
    edge_type_name: str,
) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
    """
    将边列表转换为 PyG edge_index [2, num_edges] 和可选的 confidence tensor.
    """
    src_indices: List[int] = []
    dst_indices: List[int] = []
    confidences: List[float] = []
    skipped = 0

    for e in edges:
        src_id = e[src_key]
        dst_id = e[dst_key]
        if src_id in src_map and dst_id in dst_map:
            src_indices.append(src_map[src_id])
            dst_indices.append(dst_map[dst_id])
            if "confidence" in e:
                confidences.append(e["confidence"])
        else:
            skipped += 1

    if skipped:
        print(f"  [WARN] {edge_type_name}: 跳过 {skipped} 条边 (节点未在映射中)")

    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    conf_tensor = (
        torch.tensor(confidences, dtype=torch.float)
        if confidences and len(confidences) == len(src_indices)
        else None
    )
    return edge_index, conf_tensor


def load_policy_features(
    policies: List[dict],
    policy_id_to_idx: Dict[str, int],
    text_emb_path: Path,
    emb_index_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """加载并对齐政策特征: text_emb (768-dim) + level (int)."""
    # 加载嵌入
    text_emb = torch.load(text_emb_path, weights_only=True)
    with open(emb_index_path, "r", encoding="utf-8") as f:
        emb_index = json.load(f)
    emb_id_to_idx = emb_index["policy_id_to_idx"]

    num_policies = len(policy_id_to_idx)
    emb_dim = text_emb.shape[1]

    # 按 policy_id 排序对齐
    aligned_emb = torch.zeros(num_policies, emb_dim)
    aligned_level = torch.zeros(num_policies, dtype=torch.long)

    for policy in policies:
        if policy.get("status") != "active":
            continue
        pid = policy["policy_id"]
        if pid not in policy_id_to_idx:
            continue
        idx = policy_id_to_idx[pid]

        # 文本嵌入
        if pid in emb_id_to_idx:
            emb_row = emb_id_to_idx[pid]
            aligned_emb[idx] = text_emb[emb_row]

        # 政策级别
        level_idx = policy.get("level", {}).get("level_index", 1)
        aligned_level[idx] = level_idx

    zero_emb = (aligned_emb.norm(dim=1) < 1e-8).sum().item()
    print(f"  政策文本嵌入: {aligned_emb.shape}, 零向量: {zero_emb}")
    print(f"  政策级别: Policy1={ (aligned_level == 1).sum().item()}, "
          f"Policy2={ (aligned_level == 2).sum().item()}, "
          f"Policy3={ (aligned_level == 3).sum().item()}")

    return aligned_emb, aligned_level


def load_enterprise_features(
    enterprises: List[dict],
    ent_name_to_idx: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """加载企业特征: static_feat (2-dim), temporal_series (8-dim), padding_mask (8-dim)."""
    num_ents = len(ent_name_to_idx)

    static_feat = torch.zeros(num_ents, 2)
    temporal_series = torch.zeros(num_ents, 8)
    padding_mask = torch.zeros(num_ents, 8, dtype=torch.long)

    for ent in enterprises:
        name = ent["name"]
        if name not in ent_name_to_idx:
            continue
        idx = ent_name_to_idx[name]

        # 静态特征
        static_feat[idx, 0] = ent.get("capital_log", 0.0)
        static_feat[idx, 1] = ent.get("scale_category", 1)

        # 时序特征
        its = ent.get("insurance_time_series", {})
        if its:
            log_vals = its.get("log_values", [])
            mask = its.get("padding_mask", [])
            for t in range(min(8, len(log_vals))):
                temporal_series[idx, t] = log_vals[t]
                padding_mask[idx, t] = int(mask[t]) if t < len(mask) else 0

    print(f"  企业静态特征: {static_feat.shape}")
    print(f"  企业时序特征: {temporal_series.shape}")
    valid_timesteps = padding_mask.sum().item()
    total_timesteps = padding_mask.numel()
    print(f"  时序 padding_mask 覆盖率: {valid_timesteps}/{total_timesteps} "
          f"({100*valid_timesteps/max(total_timesteps,1):.1f}%)")

    return static_feat, temporal_series, padding_mask


# ---------------------------------------------------------------------------
# 主流水线
# ---------------------------------------------------------------------------

def build_graph(
    graph_edges_path: Path,
    policies_path: Path,
    enterprises_path: Path,
    text_emb_path: Path,
    emb_index_path: Path,
    output_graph_path: Path,
    output_meta_path: Path,
    inheritance_report_path: Path,
) -> HeteroData:
    """主建图流水线."""

    # ---- 步骤 1: 加载数据 ----
    print("[1/6] 加载数据源...")
    with open(graph_edges_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    with open(policies_path, "r", encoding="utf-8") as f:
        policies = json.load(f)
    with open(enterprises_path, "r", encoding="utf-8") as f:
        enterprises = json.load(f)

    edges = graph_data["edges"]
    all_policy_ids = set(
        p["policy_id"] for p in policies if p.get("status") == "active"
    )

    print(f"  policies: {len(all_policy_ids)} active")
    print(f"  enterprises: {len(enterprises)}")
    print(f"  边: transmitsTo={len(edges.get('transmitsTo', []))}, "
          f"targetsSubIndustry={len(edges.get('targetsSubIndustry', []))}, "
          f"belongsTo={len(edges.get('belongsTo', []))}, "
          f"subClassOf={len(edges.get('subClassOf', []))}")

    # ---- 步骤 2: Phase A — 靶向继承 ----
    print("\n[2/6] Phase A: 靶向继承 (Target Inheritance)...")
    inherited_edges, inheritance_report = compute_target_inheritance(edges, all_policy_ids)

    # 合并原始 + 继承边
    all_targets_edges = edges["targetsSubIndustry"] + inherited_edges
    print(f"  合并后 targetsSubIndustry 边: {len(all_targets_edges)} "
          f"(原始 {len(edges['targetsSubIndustry'])} + 继承 {len(inherited_edges)})")

    # ---- 步骤 3: 构建节点映射 ----
    print("\n[3/6] 构建节点 ID → idx 映射...")
    mappings = build_node_mappings(policies, enterprises, edges["subClassOf"])
    for k, v in mappings.items():
        if k.startswith("num_"):
            print(f"  {k}: {v}")

    # ---- 步骤 4: 去重 belongsTo 边 + 构建边张量 ----
    print("\n[4/6] 去重 belongsTo 边 & 构建 edge_index 张量...")

    # 企业记录存在同名重复 (同一企业在多个源文件中出现), 去重 (enterprise, SI) 对
    raw_belongs = edges["belongsTo"]
    seen_belongs = set()
    belongs_deduped = []
    for e in raw_belongs:
        key = (e["subject"], e["object"])
        if key not in seen_belongs:
            seen_belongs.add(key)
            belongs_deduped.append(e)
    dup_count = len(raw_belongs) - len(belongs_deduped)
    if dup_count:
        print(f"  belongsTo 去重: {len(raw_belongs)} → {len(belongs_deduped)} (移除 {dup_count} 条重边)")

    # 4.1 transmitsTo: Policy → Policy
    transmits_ei, _ = build_edge_index(
        edges["transmitsTo"], "subject", "object",
        mappings["policy_id_to_idx"], mappings["policy_id_to_idx"],
        "transmitsTo",
    )
    print(f"  transmitsTo: {transmits_ei.shape}")

    # 4.2 targetsSubIndustry: Policy → SubIndustry (合并原始 + 继承)
    targets_ei, targets_conf = build_edge_index(
        all_targets_edges, "subject", "object",
        mappings["policy_id_to_idx"], mappings["si_id_to_idx"],
        "targetsSubIndustry",
    )
    print(f"  targetsSubIndustry: {targets_ei.shape}")

    # 4.3 belongsTo: Enterprise → SubIndustry (去重后)
    belongs_ei, _ = build_edge_index(
        belongs_deduped, "subject", "object",
        mappings["ent_name_to_idx"], mappings["si_id_to_idx"],
        "belongsTo",
    )
    print(f"  belongsTo: {belongs_ei.shape}")

    # 4.4 subClassOf: SubIndustry → MajorIndustry
    subclass_ei, _ = build_edge_index(
        edges["subClassOf"], "subject", "object",
        mappings["si_id_to_idx"], mappings["mi_id_to_idx"],
        "subClassOf",
    )
    print(f"  subClassOf: {subclass_ei.shape}")

    # ---- 步骤 5: 构建节点特征 ----
    print("\n[5/6] 构建节点特征张量...")

    # Policy 特征
    policy_emb, policy_level = load_policy_features(
        policies, mappings["policy_id_to_idx"], text_emb_path, emb_index_path
    )

    # Enterprise 特征
    ent_static, ent_temporal, ent_padding = load_enterprise_features(
        enterprises, mappings["ent_name_to_idx"]
    )

    # SubIndustry 特征: one-hot identity
    si_onehot = torch.eye(mappings["num_sub_industries"])
    print(f"  SubIndustry one-hot: {si_onehot.shape}")

    # MajorIndustry 特征: one-hot identity
    mi_onehot = torch.eye(mappings["num_major_industries"])
    print(f"  MajorIndustry one-hot: {mi_onehot.shape}")

    # ---- 步骤 6: 组装 HeteroData ----
    print("\n[6/6] 组装 PyG HeteroData...")

    if not PYG_AVAILABLE:
        print("[ERROR] torch_geometric 未安装! 请运行: pip install torch-geometric")
        sys.exit(1)

    data = HeteroData()

    # 节点数量
    data["Policy"].num_nodes = mappings["num_policies"]
    data["Enterprise"].num_nodes = mappings["num_enterprises"]
    data["SubIndustry"].num_nodes = mappings["num_sub_industries"]
    data["MajorIndustry"].num_nodes = mappings["num_major_industries"]

    # 节点特征
    data["Policy"].text_emb = policy_emb
    data["Policy"].level = policy_level
    data["Enterprise"].static_feat = ent_static
    data["Enterprise"].temporal_series = ent_temporal
    data["Enterprise"].padding_mask = ent_padding
    data["SubIndustry"].x = si_onehot
    data["MajorIndustry"].x = mi_onehot

    # 边 (仅单向, Session 4 用 ToUndirected())
    data["Policy", "transmitsTo", "Policy"].edge_index = transmits_ei
    data["Policy", "targetsSubIndustry", "SubIndustry"].edge_index = targets_ei
    data["Enterprise", "belongsTo", "SubIndustry"].edge_index = belongs_ei
    data["SubIndustry", "subClassOf", "MajorIndustry"].edge_index = subclass_ei

    # targetsSubIndustry 边的置信度
    if targets_conf is not None:
        data["Policy", "targetsSubIndustry", "SubIndustry"].confidence = targets_conf

    # 元数据: 保留在企业/子行业/大类名称
    data["Policy"].policy_ids = list(mappings["policy_id_to_idx"].keys())
    data["Enterprise"].names = list(mappings["ent_name_to_idx"].keys())
    data["SubIndustry"].si_ids = list(mappings["si_id_to_idx"].keys())
    data["MajorIndustry"].mi_ids = list(mappings["mi_id_to_idx"].keys())

    # 保存
    output_graph_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_graph_path)
    print(f"  [OK] HeteroData → {output_graph_path}")

    # 保存图元数据
    graph_meta = {
        "description": "Session 3 PyG HeteroData 元数据",
        "generated": datetime.now().isoformat(),
        "node_counts": {
            "Policy": mappings["num_policies"],
            "Enterprise": mappings["num_enterprises"],
            "SubIndustry": mappings["num_sub_industries"],
            "MajorIndustry": mappings["num_major_industries"],
        },
        "feature_dims": {
            "Policy_text_emb": 768,
            "Policy_level": "int (1|2|3)",
            "Enterprise_static_feat": 2,
            "Enterprise_temporal_series": 8,
            "Enterprise_padding_mask": "8 (binary)",
            "SubIndustry_x": mappings["num_sub_industries"],
            "MajorIndustry_x": mappings["num_major_industries"],
        },
        "edge_counts": {
            "transmitsTo": transmits_ei.shape[1],
            "targetsSubIndustry": targets_ei.shape[1],
            "belongsTo": belongs_ei.shape[1],
            "subClassOf": subclass_ei.shape[1],
            "belongsTo_before_dedup": len(raw_belongs),
        },
        "inheritance": {
            "inherited_edges": len(inherited_edges),
            "activated_policies": inheritance_report["summary"]["activated_by_inheritance"],
            "still_dead": inheritance_report["summary"]["still_dead_no_ancestor_targets"],
        },
        "policy_id_to_idx": mappings["policy_id_to_idx"],
        "ent_name_to_idx": mappings["ent_name_to_idx"],
        "si_id_to_idx": mappings["si_id_to_idx"],
        "mi_id_to_idx": mappings["mi_id_to_idx"],
    }
    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(graph_meta, f, ensure_ascii=False, indent=2)
    print(f"  [OK] graph_meta.json → {output_meta_path}")

    # 保存继承报告
    inheritance_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(inheritance_report_path, "w", encoding="utf-8") as f:
        json.dump(inheritance_report, f, ensure_ascii=False, indent=2)
    print(f"  [OK] inheritance_report.json → {inheritance_report_path}")

    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session 3 — 图构建: 靶向继承 + PyG HeteroData"
    )
    parser.add_argument(
        "--graph-edges",
        type=str,
        default="data/processed/graph_edges_corrected.json",
        help="修正后的图谱边 JSON 路径",
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
        default="data/processed/enterprises_corrected.json",
        help="修正后的企业数据路径",
    )
    parser.add_argument(
        "--text-emb",
        type=str,
        default="data/processed/text_embeddings/policy_text_emb.pt",
        help="政策文本嵌入 .pt 路径",
    )
    parser.add_argument(
        "--emb-index",
        type=str,
        default="data/processed/text_embeddings/policy_emb_index.json",
        help="嵌入 ID 索引 JSON 路径",
    )
    parser.add_argument(
        "--output-graph",
        type=str,
        default="data/processed/graph/hetero_graph.pt",
        help="输出 HeteroData .pt 路径",
    )
    parser.add_argument(
        "--output-meta",
        type=str,
        default="data/processed/graph/graph_meta.json",
        help="输出图元数据 JSON 路径",
    )
    parser.add_argument(
        "--inheritance-report",
        type=str,
        default="data/statistics/inheritance_report.json",
        help="靶向继承报告输出路径",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    print("=" * 60)
    print("Session 3 — Task 2+3: 图构建 (Graph Builder)")
    print("=" * 60)

    data = build_graph(
        graph_edges_path=project_root / args.graph_edges,
        policies_path=project_root / args.policies,
        enterprises_path=project_root / args.enterprises,
        text_emb_path=project_root / args.text_emb,
        emb_index_path=project_root / args.emb_index,
        output_graph_path=project_root / args.output_graph,
        output_meta_path=project_root / args.output_meta,
        inheritance_report_path=project_root / args.inheritance_report,
    )

    # 打印最终图概览
    print("\n" + "=" * 60)
    print("[OK] PyG HeteroData 图构建完成")
    print("=" * 60)
    print(f"\n  节点:")
    for ntype in ["Policy", "Enterprise", "SubIndustry", "MajorIndustry"]:
        print(f"    {ntype}: {data[ntype].num_nodes}")
    print(f"\n  边:")
    for etype in data.edge_types:
        ei = data[etype].edge_index
        print(f"    {etype}: {ei.shape[1]}")
    print(f"\n  节点特征:")
    print(f"    Policy.text_emb:     {list(data['Policy'].text_emb.shape)}")
    print(f"    Policy.level:        {list(data['Policy'].level.shape)}")
    print(f"    Enterprise.static_feat:   {list(data['Enterprise'].static_feat.shape)}")
    print(f"    Enterprise.temporal_series: {list(data['Enterprise'].temporal_series.shape)}")
    print(f"    Enterprise.padding_mask:    {list(data['Enterprise'].padding_mask.shape)}")
    print(f"    SubIndustry.x:       {list(data['SubIndustry'].x.shape)}")
    print(f"    MajorIndustry.x:     {list(data['MajorIndustry'].x.shape)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
