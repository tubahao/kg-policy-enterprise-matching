"""
Session 2 — Verifier: strict string alignment + edge merging.

1. Validates LLM classification outputs against the canonical sub-industry list.
2. Merges deterministic edges with validated LLM targetsSubIndustry edges.
3. Generates extraction quality report.
4. Generates ontology_v2.json.

Output:
  - data/processed/graph_edges_final.json  (merged edges for Session 3)
  - data/statistics/extraction_report.json (quality metrics)
  - src/extraction/ontology/ontology_v2.json (new unified ontology)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETERMINISTIC_PATH = PROJECT_ROOT / "data" / "processed" / "deterministic_graph_edges.json"
LLM_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "llm_classification_results.json"
GRAPH_EDGES_PATH = PROJECT_ROOT / "data" / "processed" / "graph_edges_final.json"
REPORT_PATH = PROJECT_ROOT / "data" / "statistics" / "extraction_report.json"
ONTOLOGY_V2_PATH = PROJECT_ROOT / "src" / "extraction" / "ontology" / "ontology_v2.json"
DEPRECATED_DIR = PROJECT_ROOT / "src" / "extraction" / "ontology" / "deprecated"


# ---------------------------------------------------------------------------
# String matching
# ---------------------------------------------------------------------------

def canonical_sub_industry_names(deterministic_data: dict) -> List[str]:
    """Extract the canonical sub-industry name list."""
    return deterministic_data["sub_industry_list"]


def match_sub_industry(
    llm_name: str,
    canonicals: List[str],
) -> Tuple[str | None, str]:
    """Try to match an LLM-returned industry name to the canonical list.

    Returns (canonical_name, match_method) or (None, reason).
    """
    # 1. Exact match
    if llm_name in canonicals:
        return llm_name, "exact"

    # 2. Case-insensitive / whitespace normalization
    llm_norm = llm_name.strip()
    for c in canonicals:
        if c.strip() == llm_norm:
            return c, "exact_normalized"

    # 3. Fuzzy match — edit distance via SequenceMatcher
    best_ratio = 0.0
    best_match = None
    for c in canonicals:
        ratio = SequenceMatcher(None, llm_norm, c).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = c

    if best_ratio >= 0.85 and best_match is not None:
        return best_match, f"fuzzy({best_ratio:.2f})"

    # 4. Containment match
    for c in canonicals:
        if llm_norm in c or c in llm_norm:
            return c, "containment"

    return None, f"no_match(best_ratio={best_ratio:.2f})"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_and_merge(
    llm_results: List[dict],
    canonicals: List[str],
    sub_id_map: Dict[str, str],
    deterministic_data: dict,
) -> Tuple[List[dict], List[dict], dict]:
    """Validate LLM outputs and produce merged edge set."""

    targets_edges: List[dict] = []
    rejection_log: List[dict] = []
    stats = {
        "total_policies_processed": len(llm_results),
        "total_llm_predictions": 0,
        "accepted": 0,
        "rejected": 0,
        "match_methods": defaultdict(int),
        "policies_with_targets": 0,
        "policies_without_targets": 0,
        "avg_confidence": 0.0,
    }

    confidences = []

    for entry in llm_results:
        policy_id = entry["policy_id"]
        predictions = entry.get("targetsSubIndustry", [])

        stats["total_llm_predictions"] += len(predictions)

        if not predictions:
            stats["policies_without_targets"] += 1
            continue

        stats["policies_with_targets"] += 1

        for pred in predictions:
            llm_name = pred["sub_industry"]
            confidence = pred["confidence"]
            matched_name, method = match_sub_industry(llm_name, canonicals)

            if matched_name is not None:
                si_id = sub_id_map.get(matched_name)
                if si_id is None:
                    rejection_log.append({
                        "policy_id": policy_id,
                        "llm_output": llm_name,
                        "reason": "canonical_name_not_in_id_map",
                    })
                    stats["rejected"] += 1
                    continue

                targets_edges.append({
                    "subject": policy_id,
                    "subject_type": entry.get("level", "Policy"),
                    "predicate": "targetsSubIndustry",
                    "object": si_id,
                    "object_type": "SubIndustry",
                    "object_name": matched_name,
                    "confidence": confidence,
                    "match_method": method,
                })
                stats["accepted"] += 1
                stats["match_methods"][method] += 1
                confidences.append(confidence)
            else:
                rejection_log.append({
                    "policy_id": policy_id,
                    "llm_output": llm_name,
                    "confidence": confidence,
                    "reason": method,
                })
                stats["rejected"] += 1

    if confidences:
        stats["avg_confidence"] = sum(confidences) / len(confidences)
    stats["match_methods"] = dict(stats["match_methods"])

    return targets_edges, rejection_log, stats


# ---------------------------------------------------------------------------
# Ontology v2 generation
# ---------------------------------------------------------------------------

def generate_ontology_v2(deterministic_data: dict, targets_edges: List[dict]):
    """Generate ontology_v2.json — the single authoritative ontology file."""

    stats = deterministic_data["statistics"]

    # Count unique sub-industries targeted by policies
    targeted_si = set(e["object"] for e in targets_edges)
    targeted_si_count = len(targeted_si)

    ontology = {
        "version": "2.0",
        "description": "TEI-Optimized Ontology — 四级传导异构图本体",
        "generated": datetime.now().isoformat(),
        "entity_types": {
            "Policy": {
                "subtypes": ["Policy1", "Policy2", "Policy3"],
                "description": "政策节点 — 国家级 / 自治区级 / 市级",
                "count": 1892,
            },
            "SubIndustry": {
                "description": "细分行业节点 (GB/T 4754-2017 小类/中类)",
                "count": stats["sub_industries"],
                "targeted_by_policy_count": targeted_si_count,
            },
            "MajorIndustry": {
                "description": "国民经济行业大类",
                "count": stats["major_industries"],
                "values": [
                    "制造业",
                    "科学研究和技术服务业",
                    "文化、体育和娱乐业",
                    "水利、环境和公共设施管理业",
                    "电力、热力、燃气及水生产和供应业",
                ],
            },
            "Enterprise": {
                "description": "企业实体节点",
                "count": stats["enterprises_total"],
                "attributes": ["name", "sub_industry", "major_industry", "is_high_tech"],
                "high_tech_count": stats["enterprises_high_tech"],
            },
        },
        "relationships": {
            "transmitsTo": {
                "subject_type": "Policy1|Policy2",
                "object_type": "Policy2|Policy3",
                "description": "行政纵向传导边 — 上级政策逐级传达给下级",
                "direction": "top_down",
                "build_method": "deterministic_regex",
                "count": stats["transmits_to_edges"],
                "role_in_tei": "空间层级衰减路径 (Hierarchical Decay)",
            },
            "targetsSubIndustry": {
                "subject_type": "Policy1|Policy2|Policy3",
                "object_type": "SubIndustry",
                "description": "政策行业靶向边 — 限定政策能量传播的行业边界",
                "direction": "policy_to_industry",
                "build_method": "llm_classification",
                "count": len(targets_edges),
                "has_confidence": True,
                "role_in_tei": "行业横向能量边界 (Energy Scope)",
            },
            "belongsTo": {
                "subject_type": "Enterprise",
                "object_type": "SubIndustry",
                "description": "企业归属边 — 企业实体到细分行业的 N:1 映射",
                "direction": "entity_to_industry",
                "build_method": "deterministic",
                "count": stats["belongs_to_edges"],
                "role_in_tei": "能量落点路径 (Entity Affiliation)",
            },
            "subClassOf": {
                "subject_type": "SubIndustry",
                "object_type": "MajorIndustry",
                "description": "行业层级边 — 细分行业到大类的树状归属",
                "direction": "fine_to_coarse",
                "build_method": "deterministic",
                "count": stats["subclass_of_edges"],
                "role_in_tei": "行业层级聚合 (Taxonomy Aggregation)",
            },
        },
        "ground_truth_label": {
            "supports": {
                "subject_type": "Policy2|Policy3",
                "object_type": "Enterprise",
                "description": "真实政策-企业资助/匹配记录 — 仅用于 Session 4/5 评估",
                "usage": "建图阶段不生成此边! 作为测试标签, 8:1:1 时序切分后评估 NDCG/Recall/MAP",
                "anti_leakage_note": "此边不可参与 PPR 传播和 GAT 邻居聚合",
            },
        },
        "removed_from_v1": {
            "executes": "与 transmitsTo 语义冗余 (逆边由 DGL 自动生成)",
            "implements": "与 supports 语义冗余 (逆边由 DGL 自动生成)",
            "6_industry_types": "重构为 MajorIndustry(5) + SubIndustry(63) 两级体系",
        },
    }

    ONTOLOGY_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ONTOLOGY_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(ontology, f, ensure_ascii=False, indent=2)

    print(f"  本体 v2: {ONTOLOGY_V2_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Session 2 — 校验与合并 (Verifier)")
    print("=" * 60)

    # Load deterministic data
    print("\n[1/4] 加载确定性边...")
    with open(DETERMINISTIC_PATH, "r", encoding="utf-8") as f:
        deterministic_data = json.load(f)
    print(f"  子行业: {len(deterministic_data['sub_industry_list'])}")
    print(f"  belongsTo: {len(deterministic_data['edges']['belongsTo'])} 条")
    print(f"  subClassOf: {len(deterministic_data['edges']['subClassOf'])} 条")
    print(f"  transmitsTo: {len(deterministic_data['edges']['transmitsTo'])} 条")

    # Load LLM results
    print("\n[2/4] 加载 LLM 分类结果...")
    if not LLM_RESULTS_PATH.exists():
        print(f"  WARNING: {LLM_RESULTS_PATH} 不存在! 将跳过 targetsSubIndustry 边。")
        print(f"  请先运行 llm_classifier.py")
        llm_results = []
    else:
        with open(LLM_RESULTS_PATH, "r", encoding="utf-8") as f:
            llm_results = json.load(f)
        print(f"  已处理政策: {len(llm_results)} 条")

    # Build canonical name list + ID map
    canonicals = canonical_sub_industry_names(deterministic_data)
    sub_entities = deterministic_data["entities"]["SubIndustry"]
    sub_id_map = {v["name"]: k for k, v in sub_entities.items()}

    # Validate and merge
    print("\n[3/4] 校验 LLM 输出并合并...")
    if llm_results:
        targets_edges, rejection_log, stats = validate_and_merge(
            llm_results, canonicals, sub_id_map, deterministic_data
        )
        print(f"  接受: {stats['accepted']} 条")
        print(f"  拒绝: {stats['rejected']} 条")
        print(f"  匹配方式: {stats['match_methods']}")
        print(f"  平均置信度: {stats['avg_confidence']:.3f}")

        if rejection_log:
            # Print first 5 rejections
            print(f"\n  拒绝样本 (前5条):")
            for r in rejection_log[:5]:
                print(f"    policy={r['policy_id']} llm_output='{r['llm_output']}' reason={r['reason']}")
    else:
        targets_edges = []
        rejection_log = []
        stats = {"accepted": 0, "rejected": 0, "total_policies_processed": 0,
                 "total_llm_predictions": 0, "match_methods": {}, "avg_confidence": 0,
                 "policies_with_targets": 0, "policies_without_targets": 0}

    # Assemble final output
    print("\n[4/4] 写入最终输出...")
    final_output = {
        "version": "2.0",
        "description": "Session 2 图谱边全集 — 确定性建边 + LLM 分类校验",
        "generated": datetime.now().isoformat(),
        "edges": {
            "belongsTo": deterministic_data["edges"]["belongsTo"],
            "subClassOf": deterministic_data["edges"]["subClassOf"],
            "transmitsTo": deterministic_data["edges"]["transmitsTo"],
            "targetsSubIndustry": targets_edges,
        },
        "statistics": {
            **deterministic_data["statistics"],
            "targets_sub_industry_edges": len(targets_edges),
            "llm_accept_rate": (
                stats["accepted"] / max(stats["total_llm_predictions"], 1)
                if stats["total_llm_predictions"] > 0 else 0
            ),
            "avg_confidence": stats["avg_confidence"],
            "rejected_predictions": stats["rejected"],
            "total_edges": (
                deterministic_data["statistics"]["belongs_to_edges"]
                + deterministic_data["statistics"]["subclass_of_edges"]
                + deterministic_data["statistics"]["transmits_to_edges"]
                + len(targets_edges)
            ),
        },
    }

    GRAPH_EDGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_EDGES_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    print(f"  图边全集: {GRAPH_EDGES_PATH}")

    # Extraction report
    report = {
        "generated": datetime.now().isoformat(),
        "deterministic": deterministic_data["statistics"],
        "llm_classification": stats,
        "rejection_log": rejection_log[:50],  # Cap log entries
        "rejection_total": len(rejection_log),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  抽取报告: {REPORT_PATH}")

    # Ontology v2
    generate_ontology_v2(deterministic_data, targets_edges)

    # Deprecate old ontology files
    DEPRECATED_DIR.mkdir(parents=True, exist_ok=True)
    old_ontology_dir = PROJECT_ROOT / "src" / "extraction" / "ontology"
    old_files = [
        "actual_data_schema.json",
        "actual_data_types.json",
        "actual_domain_concepts.json",
        "actual_entity_types.json",
        "actual_relationship_types.json",
    ]
    moved = 0
    for fname in old_files:
        src = old_ontology_dir / fname
        if src.exists():
            dst = DEPRECATED_DIR / fname
            src.rename(dst)
            moved += 1
    if moved > 0:
        print(f"  旧本体文件已移至: {DEPRECATED_DIR} ({moved} 个文件)")

    # Summary
    total_edges = final_output["statistics"]["total_edges"]
    print(f"\n{'='*60}")
    print(f"Session 2 完成!")
    print(f"  总边数: {total_edges}")
    print(f"    - belongsTo:          {deterministic_data['statistics']['belongs_to_edges']:>6}")
    print(f"    - subClassOf:         {deterministic_data['statistics']['subclass_of_edges']:>6}")
    print(f"    - transmitsTo:        {deterministic_data['statistics']['transmits_to_edges']:>6}")
    print(f"    - targetsSubIndustry: {len(targets_edges):>6}")
    print(f"  LLM 接受率: {final_output['statistics']['llm_accept_rate']:.1%}")
    print(f"  平均置信度: {stats['avg_confidence']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
