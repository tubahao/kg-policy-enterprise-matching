#!/usr/bin/env python3
"""Session 3 — Task 1: 子行业清洗字典 (Ontology Corrector)

读取 graph_edges_final.json，通过硬编码修正字典修复 subClassOf 错配：
- 专业技术服务业: 水利 → 科学研究和技术服务业
- "制造业" 作为子行业名 → 合并入 "其他制造业" (SI_07)
- 7 个跨域子行业 → 新建 MI_05 "其他跨域产业 (Cross-domain)"
- 修复企业数据中 "高新企业" 泄漏到 major_industry 的问题
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. 硬编码修正字典
# ---------------------------------------------------------------------------

# 子行业 subClassOf 修正: SI_ID -> 正确 MI_ID
SUBCLASS_CORRECTIONS: Dict[str, str] = {
    "SI_00": "MI_04",  # 专业技术服务业: 水利 → 科学研究和技术服务业
    "SI_09": "MI_05",  # 农业 → 其他跨域产业
    "SI_18": "MI_05",  # 土木工程建筑业 → 其他跨域产业
    "SI_23": "MI_05",  # 建筑安装业 → 其他跨域产业
    "SI_24": "MI_05",  # 建筑装饰、装修和其他建筑业 → 其他跨域产业
    "SI_26": "MI_05",  # 房屋建筑业 → 其他跨域产业
    "SI_27": "MI_05",  # 批发业 → 其他跨域产业
    "SI_59": "MI_05",  # 零售业 → 其他跨域产业
}

# 子行业重命名: SI_ID -> new_name
SI_RENAMES: Dict[str, str] = {
    "SI_11": "其他制造业",  # "制造业" 大类名当子类 → 合并入 SI_07
}

# SI_11 → SI_07 合并映射 (因为 "其他制造业" 已存在于 SI_07)
SI_MERGE_MAP: Dict[str, str] = {
    "SI_11": "SI_07",
}

# 5 个核心产业 (受保护的特征空间)
CORE_MAJOR_INDUSTRIES: List[str] = [
    "制造业",
    "科学研究和技术服务业",
    "文化、体育和娱乐业",
    "水利、环境和公共设施管理业",
    "电力、热力、燃气及水生产和供应业",
]

# 新建的跨域产业
CROSS_DOMAIN_NAME = "其他跨域产业 (Cross-domain)"
CROSS_DOMAIN_ID = "MI_05"


# ---------------------------------------------------------------------------
# 2. 工具函数
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_si_lookup(entities: dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    """从 deterministic_graph_edges.json 的 entities 构建 SI_ID→name 和 name→SI_ID 映射."""
    si_id_to_name: Dict[str, str] = {}
    si_name_to_id: Dict[str, str] = {}
    for si_id, si_info in entities.get("SubIndustry", {}).items():
        name = si_info["name"]
        si_id_to_name[si_id] = name
        si_name_to_id[name] = si_id
    return si_id_to_name, si_name_to_id


def build_mi_lookup(entities: dict) -> Dict[str, str]:
    """MI_ID → MI_name."""
    return {
        mi_id: mi_info["name"]
        for mi_id, mi_info in entities.get("MajorIndustry", {}).items()
    }


# ---------------------------------------------------------------------------
# 3. 核心修正逻辑
# ---------------------------------------------------------------------------

def apply_si_merge(
    edges: Dict[str, List[dict]],
    enterprises: List[dict],
    si_id_to_name: Dict[str, str],
) -> Tuple[Dict[str, List[dict]], List[dict], List[dict]]:
    """将 SI_11 合并入 SI_07，更新所有边和企业引用."""
    merge_log: List[dict] = []

    for old_id, new_id in SI_MERGE_MAP.items():
        old_name = si_id_to_name.get(old_id, old_id)
        new_name = si_id_to_name.get(new_id, new_id)

        # 更新 belongsTo 边
        belongs_updated = 0
        for e in edges["belongsTo"]:
            if e["object"] == old_id:
                e["object"] = new_id
                belongs_updated += 1
        if belongs_updated:
            merge_log.append({
                "action": "merge_belongsTo",
                "old_si_id": old_id, "old_si_name": old_name,
                "new_si_id": new_id, "new_si_name": new_name,
                "edges_updated": belongs_updated,
            })

        # 更新 targetsSubIndustry 边
        targets_updated = 0
        for e in edges["targetsSubIndustry"]:
            if e["object"] == old_id:
                e["object"] = new_id
                e["object_name"] = new_name
                targets_updated += 1
        if targets_updated:
            merge_log.append({
                "action": "merge_targetsSubIndustry",
                "old_si_id": old_id, "old_si_name": old_name,
                "new_si_id": new_id, "new_si_name": new_name,
                "edges_updated": targets_updated,
            })

        # 移除 SI_11 的 subClassOf 边
        edges["subClassOf"] = [
            e for e in edges["subClassOf"] if e["subject"] != old_id
        ]
        merge_log.append({
            "action": "remove_subClassOf",
            "si_id": old_id, "si_name": old_name,
        })

        # 更新企业 sub_industry 字段
        ent_updated = 0
        for ent in enterprises:
            if ent.get("sub_industry") == old_name:
                ent["sub_industry"] = new_name
                ent_updated += 1
        if ent_updated:
            merge_log.append({
                "action": "update_enterprise_sub_industry",
                "old_name": old_name, "new_name": new_name,
                "enterprises_updated": ent_updated,
            })

    return edges, enterprises, merge_log


def correct_subclass_edges(
    edges: Dict[str, List[dict]],
    mi_id_to_name: Dict[str, str],
) -> Tuple[List[dict], List[dict]]:
    """应用 SUBCLASS_CORRECTIONS 修正 subClassOf 边."""
    correction_log: List[dict] = []
    corrected_edges: List[dict] = []

    # 重建 subClassOf 边列表
    si_mi_map: Dict[str, str] = {}

    # 先从现有边读取映射
    for e in edges["subClassOf"]:
        si_id = e["subject"]
        mi_id = e["object"]
        si_mi_map[si_id] = mi_id

    # 应用修正
    for si_id, new_mi_id in SUBCLASS_CORRECTIONS.items():
        old_mi_id = si_mi_map.get(si_id, "?")
        old_mi_name = mi_id_to_name.get(old_mi_id, old_mi_id)
        new_mi_name = mi_id_to_name.get(new_mi_id, new_mi_id)
        si_mi_map[si_id] = new_mi_id
        correction_log.append({
            "action": "correct_subClassOf",
            "si_id": si_id,
            "old_mi_id": old_mi_id, "old_mi_name": old_mi_name,
            "new_mi_id": new_mi_id, "new_mi_name": new_mi_name,
        })

    # 重建全部 subClassOf 边
    for si_id, mi_id in sorted(si_mi_map.items()):
        corrected_edges.append({
            "subject": si_id,
            "subject_type": "SubIndustry",
            "predicate": "subClassOf",
            "object": mi_id,
            "object_type": "MajorIndustry",
        })

    return corrected_edges, correction_log


def create_cross_domain_major(mi_id_to_name: Dict[str, str]) -> Tuple[Dict[str, str], dict]:
    """创建 MI_05 其他跨域产业."""
    mi_id_to_name[CROSS_DOMAIN_ID] = CROSS_DOMAIN_NAME
    log_entry = {
        "action": "create_major_industry",
        "mi_id": CROSS_DOMAIN_ID,
        "mi_name": CROSS_DOMAIN_NAME,
    }
    return mi_id_to_name, log_entry


def fix_enterprise_major_industry(
    enterprises: List[dict],
    si_name_to_id: Dict[str, str],
    si_to_mi: Dict[str, str],
    mi_id_to_name: Dict[str, str],
) -> Tuple[List[dict], List[dict]]:
    """将所有 enterprise 的 major_industry='高新企业' 替换为正确的 major_industry."""
    fix_log: List[dict] = []
    fixed_count = 0

    for ent in enterprises:
        if ent.get("major_industry") == "高新企业":
            si_name = ent.get("sub_industry", "")
            si_id = si_name_to_id.get(si_name)
            if si_id and si_id in si_to_mi:
                correct_mi_id = si_to_mi[si_id]
                correct_mi_name = mi_id_to_name.get(correct_mi_id, correct_mi_id)
                old_mi = ent["major_industry"]
                ent["major_industry"] = correct_mi_name
                fixed_count += 1
                if fixed_count <= 20:
                    fix_log.append({
                        "enterprise": ent["name"],
                        "sub_industry": si_name,
                        "old_major": old_mi,
                        "new_major": correct_mi_name,
                    })

    summary = [{
        "action": "fix_enterprise_major_industry",
        "total_high_tech_enterprises": fixed_count,
        "detail_count": fixed_count,
    }] + fix_log

    return enterprises, summary


# ---------------------------------------------------------------------------
# 4. 验证
# ---------------------------------------------------------------------------

def validate_corrections(
    edges: Dict[str, List[dict]],
    enterprises: List[dict],
    si_id_to_name: Dict[str, str],
    mi_id_to_name: Dict[str, str],
) -> List[str]:
    """验证修正后的数据一致性."""
    errors: List[str] = []

    # 1. 所有 subClassOf subject 唯一
    si_ids_in_subclass = [e["subject"] for e in edges["subClassOf"]]
    si_counter = Counter(si_ids_in_subclass)
    for si_id, count in si_counter.items():
        if count > 1:
            errors.append(f"subClassOf 中 {si_id} 出现 {count} 次 (应唯一)")

    # 2. 所有边引用的 SI/MI ID 在 subClassOf 中定义
    valid_si_ids = set(si_counter.keys())
    valid_mi_ids = set(e["object"] for e in edges["subClassOf"])

    for e in edges["targetsSubIndustry"]:
        if e["object"] not in valid_si_ids:
            errors.append(f"targetsSubIndustry 引用未知 SI: {e['object']} (policy={e['subject']})")

    for e in edges["belongsTo"]:
        if e["object"] not in valid_si_ids:
            errors.append(f"belongsTo 引用未知 SI: {e['object']} (enterprise={e['subject']})")

    # 3. 企业 major_industry 不含 "高新企业"
    ht_enterprises = [e["name"] for e in enterprises if e.get("major_industry") == "高新企业"]
    if ht_enterprises:
        errors.append(f"仍有 {len(ht_enterprises)} 家企业 major_industry='高新企业': {ht_enterprises[:5]}...")

    # 4. SI_11 不应再出现在任何边中 (已合并)
    for edge_type in ["belongsTo", "targetsSubIndustry", "subClassOf"]:
        for e in edges[edge_type]:
            if e.get("object") == "SI_11" or e.get("subject") == "SI_11":
                errors.append(f"{edge_type} 中仍有 SI_11 引用: {e}")

    # 5. 企业 sub_industry 不应有 "制造业" (应为 "其他制造业")
    mfg_ents = [e["name"] for e in enterprises if e.get("sub_industry") == "制造业"]
    if mfg_ents:
        errors.append(f"仍有 {len(mfg_ents)} 家企业 sub_industry='制造业'")

    # 6. 核心 MI 数量应为 6 (5 + 跨域)
    if len(valid_mi_ids) != 6:
        errors.append(f"MajorIndustry 数量={len(valid_mi_ids)} (预期 6): {sorted(valid_mi_ids)}")

    return errors


# ---------------------------------------------------------------------------
# 5. 主流水线
# ---------------------------------------------------------------------------

def correct_ontology(
    graph_edges_path: Path,
    deterministic_path: Path,
    enterprises_path: Path,
    output_graph_path: Path,
    output_enterprises_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    """主修正流水线."""
    report_entries: List[dict] = []

    # ---- 步骤 1: 加载数据 ----
    print("[1/7] 加载数据...")
    graph_data = load_json(graph_edges_path)
    edges = graph_data["edges"]
    det_data = load_json(deterministic_path)
    enterprises = load_json(enterprises_path)

    print(f"  graph_edges_final.json: {sum(len(v) for v in edges.values())} 条边")
    print(f"  enterprises_final.json: {len(enterprises)} 家企业")

    # ---- 步骤 2: 构建查找映射 ----
    print("[2/7] 构建查找映射...")
    si_id_to_name, si_name_to_id = build_si_lookup(det_data["entities"])
    mi_id_to_name = build_mi_lookup(det_data["entities"])
    print(f"  子行业: {len(si_id_to_name)}, 大类行业: {len(mi_id_to_name)}")

    # ---- 步骤 3: 创建跨域产业 ----
    print("[3/7] 创建跨域产业 MI_05...")
    mi_id_to_name, log = create_cross_domain_major(mi_id_to_name)
    report_entries.append(log)
    print(f"  新增: {CROSS_DOMAIN_ID} = {CROSS_DOMAIN_NAME}")
    print(f"  核心产业数: 5 → 6")

    # ---- 步骤 4: SI_11 → SI_07 合并 ----
    print("[4/7] SI_11(制造业) → SI_07(其他制造业) 合并...")
    edges, enterprises, merge_log = apply_si_merge(edges, enterprises, si_id_to_name)
    report_entries.extend(merge_log)
    for entry in merge_log:
        print(f"  {entry['action']}: {entry.get('edges_updated', entry.get('enterprises_updated', '?'))} 条")

    # 更新 SI 查找映射 (移除 SI_11, 更新 SI_07 name)
    si_id_to_name.pop("SI_11", None)
    si_name_to_id.pop("制造业", None)

    # ---- 步骤 5: 修正 subClassOf ----
    print("[5/7] 修正 subClassOf 映射...")
    corrected_subclass, correction_log = correct_subclass_edges(edges, mi_id_to_name)
    old_subclass_count = len(edges["subClassOf"])
    edges["subClassOf"] = corrected_subclass
    report_entries.extend(correction_log)
    for entry in correction_log:
        print(f"  {entry['si_id']}: {entry['old_mi_name']} → {entry['new_mi_name']}")

    print(f"  subClassOf 边: {old_subclass_count} → {len(corrected_subclass)} (修正 {len(correction_log)} 条)")

    # ---- 步骤 6: 修正企业 major_industry ----
    print("[6/7] 修正企业 major_industry ('高新企业' 泄漏)...")

    # 构建当前 SI→MI 映射 (从已修正的 subClassOf)
    si_to_mi: Dict[str, str] = {}
    for e in edges["subClassOf"]:
        si_to_mi[e["subject"]] = e["object"]

    enterprises, fix_log = fix_enterprise_major_industry(
        enterprises, si_name_to_id, si_to_mi, mi_id_to_name
    )
    report_entries.extend(fix_log)
    ht_total = sum(1 for e in enterprises if e.get("major_industry") == "高新企业")
    print(f"  修正企业 major_industry: {fix_log[0].get('detail_count', 0) if fix_log else 0} 家")
    print(f"  残留 '高新企业': {ht_total} (应为 0)")

    # ---- 步骤 7: 验证 ----
    print("[7/7] 验证数据一致性...")
    errors = validate_corrections(edges, enterprises, si_id_to_name, mi_id_to_name)
    if errors:
        print(f"  [WARN] 发现 {len(errors)} 个问题:")
        for err in errors:
            print(f"    - {err}")
    else:
        print("  [OK] 所有验证通过")

    # ---- 保存输出 ----
    print("\n保存输出...")
    corrected_graph = {
        "version": "2.1",
        "description": "Session 3 本体修正 — 硬编码字典 + SI_11合并 + 跨域产业 MI_05",
        "generated": datetime.now().isoformat(),
        "corrections_applied": len(correction_log),
        "edges": edges,
        "statistics": {
            **graph_data.get("statistics", {}),
            "sub_industries_after_correction": len(set(e["subject"] for e in edges["subClassOf"])),
            "major_industries_after_correction": len(set(e["object"] for e in edges["subClassOf"])),
            "enterprises_with_fixed_major": fix_log[0].get("detail_count", 0) if fix_log else 0,
        },
    }
    save_json(corrected_graph, output_graph_path)
    print(f"  [OK] graph_edges_corrected.json → {output_graph_path}")

    save_json(enterprises, output_enterprises_path)
    print(f"  [OK] enterprises_corrected.json → {output_enterprises_path}")

    # 修正报告
    report = {
        "title": "Session 3 — Ontology Correction Report",
        "generated": datetime.now().isoformat(),
        "summary": {
            "subclass_corrections": len(correction_log),
            "si_merges": len(merge_log),
            "enterprise_major_fixes": fix_log[0].get("detail_count", 0) if fix_log else 0,
            "new_major_industries": 1,
            "cross_domain_name": CROSS_DOMAIN_NAME,
            "cross_domain_si_ids": [
                si_id for si_id, mi_id in SUBCLASS_CORRECTIONS.items()
                if mi_id == "MI_05"
            ],
            "validation_errors": len(errors),
        },
        "details": report_entries,
        "validation_errors": errors if errors else [],
    }
    save_json(report, report_path)
    print(f"  [OK] ontology_correction_report.json → {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session 3 — 本体修正: 子行业清洗字典 + 企业 major_industry 修复"
    )
    parser.add_argument(
        "--graph-edges",
        type=str,
        default="data/processed/graph_edges_final.json",
        help="输入的 graph_edges_final.json 路径",
    )
    parser.add_argument(
        "--deterministic",
        type=str,
        default="data/processed/deterministic_graph_edges.json",
        help="deterministic_graph_edges.json (含 entities 定义)",
    )
    parser.add_argument(
        "--enterprises",
        type=str,
        default="data/processed/enterprises_final.json",
        help="enterprises_final.json 路径",
    )
    parser.add_argument(
        "--output-graph",
        type=str,
        default="data/processed/graph_edges_corrected.json",
        help="修正后的图谱边输出路径",
    )
    parser.add_argument(
        "--output-enterprises",
        type=str,
        default="data/processed/enterprises_corrected.json",
        help="修正后的企业数据输出路径",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="data/statistics/ontology_correction_report.json",
        help="修正报告输出路径",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    graph_edges_path = project_root / args.graph_edges
    deterministic_path = project_root / args.deterministic
    enterprises_path = project_root / args.enterprises
    output_graph_path = project_root / args.output_graph
    output_enterprises_path = project_root / args.output_enterprises
    report_path = project_root / args.report

    print("=" * 60)
    print("Session 3 — Task 1: 本体修正 (Ontology Corrector)")
    print("=" * 60)

    report = correct_ontology(
        graph_edges_path,
        deterministic_path,
        enterprises_path,
        output_graph_path,
        output_enterprises_path,
        report_path,
    )

    print("\n" + "=" * 60)
    print("[OK] 本体修正完成")
    print(f"  subClassOf 修正: {report['summary']['subclass_corrections']} 条")
    print(f"  SI 合并: {report['summary']['si_merges']} 步")
    print(f"  企业 major 修复: {report['summary']['enterprise_major_fixes']} 家")
    print(f"  验证错误: {report['summary']['validation_errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
