"""
Session 2 — Deterministic edge generation (zero LLM).

Generates three edge types from structured data:
  1. Enterprise -> belongsTo -> SubIndustry   (from enterprise CSV fields)
  2. SubIndustry -> subClassOf -> MajorIndustry (from 5-class taxonomy)
  3. Policy(parent) -> transmitsTo -> Policy(child) (regex citation extraction)

Output: data/processed/deterministic_graph_edges.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTERPRISES_PATH = PROJECT_ROOT / "data" / "processed" / "enterprises_final.json"
POLICIES_PATH = PROJECT_ROOT / "data" / "processed" / "policies_final.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "deterministic_graph_edges.json"

# ---------------------------------------------------------------------------
# Step 1: Build authoritative SubIndustry -> MajorIndustry dictionary
#         from the 5 non-high-tech enterprise groups.
# ---------------------------------------------------------------------------

FIVE_MAJOR_INDUSTRIES = [
    "制造业",
    "科学研究和技术服务业",
    "文化、体育和娱乐业",
    "水利、环境和公共设施管理业",
    "电力、热力、燃气及水生产和供应业",
]

# Fallback major for orphan sub-industries (those only seen in 高新企业)
DEFAULT_MAJOR = "制造业"


def load_enterprises() -> List[dict]:
    with open(ENTERPRISES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sub_to_major(enterprises: List[dict]) -> Dict[str, str]:
    """Build SubIndustry->MajorIndustry map from non-high-tech enterprises only.

    For each sub_industry, collect the major_industry from enterprises whose
    major_industry is NOT '高新企业'.  Take the most common assignment.
    """
    sub_major_counter: Dict[str, Counter] = defaultdict(Counter)

    for e in enterprises:
        if e["major_industry"] == "高新企业":
            continue
        sub_major_counter[e["sub_industry"]][e["major_industry"]] += 1

    mapping: Dict[str, str] = {}
    for sub, counter in sub_major_counter.items():
        mapping[sub] = counter.most_common(1)[0][0]

    return mapping


def assign_major_and_high_tech(
    enterprises: List[dict],
    sub_to_major: Dict[str, str],
) -> Tuple[List[dict], Dict[str, str]]:
    """Assign every enterprise a correct major_industry and is_high_tech flag.

    For 高新企业 enterprises: look up sub_industry in sub_to_major dict.
    If missing (orphan), assign DEFAULT_MAJOR.
    Also fill in any sub_industries not yet in the mapping.
    """
    enriched = []
    new_sub_to_major = dict(sub_to_major)

    for e in enterprises:
        entry = {
            "name": e["name"],
            "sub_industry": e["sub_industry"],
            "major_industry": e["major_industry"],
            "original_major_industry": e["major_industry"],
            "is_high_tech": e["major_industry"] == "高新企业",
        }

        if e["major_industry"] == "高新企业":
            assigned = sub_to_major.get(e["sub_industry"])
            if assigned is None:
                assigned = DEFAULT_MAJOR
                print(f"  [孤儿] {e['sub_industry']} ({e['name']}) → 兜底为 {DEFAULT_MAJOR}")
            entry["major_industry"] = assigned
            # Also record the mapping for future use
            if e["sub_industry"] not in new_sub_to_major:
                new_sub_to_major[e["sub_industry"]] = assigned
        else:
            entry["major_industry"] = e["major_industry"]

        enriched.append(entry)

    return enriched, new_sub_to_major


# ---------------------------------------------------------------------------
# Step 2: Build edge lists
# ---------------------------------------------------------------------------

def build_entity_ids(sub_industries: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Assign stable IDs to SubIndustry and MajorIndustry nodes."""
    si_sorted = sorted(sub_industries)
    sub_id_map = {name: f"SI_{i:02d}" for i, name in enumerate(si_sorted)}

    major_sorted = sorted(FIVE_MAJOR_INDUSTRIES)
    major_id_map = {name: f"MI_{i:02d}" for i, name in enumerate(major_sorted)}

    return sub_id_map, major_id_map


def build_deterministic_edges(
    enriched_enterprises: List[dict],
    sub_id_map: Dict[str, str],
    major_id_map: Dict[str, str],
    sub_to_major: Dict[str, str],
) -> Tuple[List[dict], List[dict], List[dict], List[dict], List[dict]]:
    """Generate all deterministic edges and entity dictionaries."""

    belongs_to_edges: List[dict] = []
    subclass_edges: List[dict] = []
    enterprise_attrs: List[dict] = []
    sub_industry_entities: List[dict] = []

    # Track which sub-industries actually have enterprises
    sub_ent_count: Dict[str, int] = defaultdict(int)

    for e in enriched_enterprises:
        si_name = e["sub_industry"]
        si_id = sub_id_map[si_name]

        # Enterprise -> belongsTo -> SubIndustry
        belongs_to_edges.append({
            "subject": e["name"],
            "subject_type": "Enterprise",
            "predicate": "belongsTo",
            "object": si_id,
            "object_type": "SubIndustry",
        })

        # Enterprise attributes
        enterprise_attrs.append({
            "name": e["name"],
            "sub_industry": si_name,
            "sub_industry_id": si_id,
            "major_industry": e["major_industry"],
            "is_high_tech": e["is_high_tech"],
        })

        sub_ent_count[si_name] += 1

    # SubIndustry -> subClassOf -> MajorIndustry (unique per sub_industry)
    seen_si = set()
    for e in enriched_enterprises:
        si_name = e["sub_industry"]
        if si_name in seen_si:
            continue
        seen_si.add(si_name)

        si_id = sub_id_map[si_name]
        major = sub_to_major.get(si_name, DEFAULT_MAJOR)
        mi_id = major_id_map[major]

        subclass_edges.append({
            "subject": si_id,
            "subject_type": "SubIndustry",
            "predicate": "subClassOf",
            "object": mi_id,
            "object_type": "MajorIndustry",
        })

    # SubIndustry entities with metadata
    for si_name in sorted(sub_ent_count.keys()):
        sub_industry_entities.append({
            "id": sub_id_map[si_name],
            "name": si_name,
            "major_industry": sub_to_major.get(si_name, DEFAULT_MAJOR),
            "major_industry_id": major_id_map[sub_to_major.get(si_name, DEFAULT_MAJOR)],
            "enterprise_count": sub_ent_count[si_name],
        })

    # MajorIndustry entities
    major_entities = [
        {"id": mi_id, "name": name}
        for name, mi_id in sorted(major_id_map.items(), key=lambda x: x[1])
    ]

    return belongs_to_edges, subclass_edges, enterprise_attrs, sub_industry_entities, major_entities


# ---------------------------------------------------------------------------
# Step 3: Policy transmitsTo extraction via regex
# ---------------------------------------------------------------------------

# Regex: match 《...》 patterns that look like policy titles
BOOK_TITLE_RE = re.compile(r"《([^》]{4,80})》")
# Regex: match official document numbers like (桂政发〔2022〕1号)
DOC_NUMBER_RE = re.compile(r"[（(]([^）)]*?(?:发|字|函|通|办|规)[^）)]*?[〔\[][0-9]{4}[〕\]][^）)]*?号?)[）)]")


def load_policies() -> List[dict]:
    with open(POLICIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_title_index(policies: List[dict]) -> Dict[str, dict]:
    """Build a lookup index: policy title -> policy record.

    Also indexes by normalized (whitespace-collapsed) title.
    """
    index: Dict[str, dict] = {}
    for p in policies:
        if p.get("status") != "active":
            continue
        title = str(p.get("title", "")).strip()
        if title:
            index[title] = p
        # Also index without whitespace
        norm = re.sub(r"\s+", "", title)
        if norm and norm != title:
            index[norm] = p
    return index


def extract_transmits_to(
    policies: List[dict],
    title_index: Dict[str, dict],
) -> List[dict]:
    """For each policy, extract 《title》 citations from the first 500 chars
    of text_for_llm.  If the cited title exists in the policy database AND
    was published before the citing policy, create:

        cited_policy  --transmitsTo--> citing_policy
    """
    edges: List[dict] = []
    matched_count = 0
    checked_count = 0

    for p in policies:
        if p.get("status") != "active":
            continue

        policy_id = p["policy_id"]
        text = p.get("text_for_llm", "")
        pub_date = str(p.get("pub_date", ""))

        # Only scan first 500 characters (citation boilerplate is at the top)
        prefix = text[:500]

        # Extract all 《...》 matches
        titles_found = BOOK_TITLE_RE.findall(prefix)
        # Also try doc number extraction
        doc_numbers = DOC_NUMBER_RE.findall(prefix)

        # Unique set of candidate titles
        candidates = set(t.strip() for t in titles_found if len(t.strip()) >= 4)

        for cited_title in candidates:
            checked_count += 1

            # Exact match
            match = title_index.get(cited_title)
            if match is None:
                # Try normalized match
                norm = re.sub(r"\s+", "", cited_title)
                match = title_index.get(norm)

            if match is None:
                continue

            cited_id = match["policy_id"]
            cited_date = str(match.get("pub_date", ""))

            # Only establish edge if cited policy is published before citing policy
            if cited_date <= pub_date and cited_id != policy_id:
                edges.append({
                    "subject": cited_id,
                    "subject_type": f"Policy{match.get('level', {}).get('level_index', '')}",
                    "predicate": "transmitsTo",
                    "object": policy_id,
                    "object_type": f"Policy{p.get('level', {}).get('level_index', '')}",
                    "citation_text": cited_title,
                })
                matched_count += 1

    print(f"  transmitsTo: 检查了 {checked_count} 个书名号引用, "
          f"成功匹配 {matched_count} 条, 生成 {len(edges)} 条边")
    return edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Session 2 — 确定性建边 (Deterministic Edge Generation)")
    print("=" * 60)

    # ---- Load data ----
    print("\n[1/5] 加载企业数据...")
    enterprises = load_enterprises()
    print(f"  总计 {len(enterprises)} 家企业")
    high_tech_count = sum(1 for e in enterprises if e["major_industry"] == "高新企业")
    print(f"  其中高新企业: {high_tech_count}")

    # ---- Build SubIndustry->MajorIndustry dictionary ----
    print("\n[2/5] 构建 5大类 SubIndustry→MajorIndustry 映射字典...")
    sub_to_major = build_sub_to_major(enterprises)
    print(f"  从非高新企业提取了 {len(sub_to_major)} 个子行业映射")

    # ---- Assign majors and high-tech flag ----
    enriched, sub_to_major_full = assign_major_and_high_tech(enterprises, sub_to_major)

    # Collect all unique sub-industries
    all_sub_industries = sorted(set(e["sub_industry"] for e in enriched))
    print(f"  最终子行业总数: {len(all_sub_industries)}")
    print(f"  最终映射表大小: {len(sub_to_major_full)}")

    # ---- Build entity IDs and edges ----
    print("\n[3/5] 生成确定性边...")
    sub_id_map, major_id_map = build_entity_ids(all_sub_industries)

    belongs_to_edges, subclass_edges, enterprise_attrs, sub_entities, major_entities = \
        build_deterministic_edges(enriched, sub_id_map, major_id_map, sub_to_major_full)

    print(f"  belongsTo 边: {len(belongs_to_edges)}")
    print(f"  subClassOf 边: {len(subclass_edges)}")

    # ---- transmitsTo edges ----
    print("\n[4/5] 提取 transmitsTo (政策纵向传导) 边...")
    policies = load_policies()
    active_policies = [p for p in policies if p.get("status") == "active"]
    print(f"  加载 {len(active_policies)} 条 active 政策")
    title_index = build_title_index(policies)
    print(f"  标题索引: {len(title_index)} 条")
    transmits_to_edges = extract_transmits_to(policies, title_index)

    # ---- Assemble output ----
    print("\n[5/5] 写入输出文件...")

    output = {
        "version": "2.0",
        "description": "Session 2 确定性建边 — 零 LLM 参与",
        "entities": {
            "SubIndustry": {e["id"]: {
                "name": e["name"],
                "major_industry": e["major_industry"],
                "major_industry_id": e["major_industry_id"],
                "enterprise_count": e["enterprise_count"],
            } for e in sub_entities},
            "MajorIndustry": {e["id"]: {"name": e["name"]} for e in major_entities},
        },
        "enterprise_attributes": enterprise_attrs,
        "sub_industry_list": sorted(all_sub_industries),
        "edges": {
            "belongsTo": belongs_to_edges,
            "subClassOf": subclass_edges,
            "transmitsTo": transmits_to_edges,
        },
        "statistics": {
            "enterprises_total": len(enterprises),
            "enterprises_high_tech": high_tech_count,
            "sub_industries": len(all_sub_industries),
            "major_industries": len(FIVE_MAJOR_INDUSTRIES),
            "belongs_to_edges": len(belongs_to_edges),
            "subclass_of_edges": len(subclass_edges),
            "transmits_to_edges": len(transmits_to_edges),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  输出文件: {OUTPUT_PATH}")
    print(f"  子行业数: {len(all_sub_industries)}")
    print(f"  大类数: {len(FIVE_MAJOR_INDUSTRIES)}")
    print(f"  belongsTo 边: {len(belongs_to_edges)}")
    print(f"  subClassOf 边: {len(subclass_edges)}")
    print(f"  transmitsTo 边: {len(transmits_to_edges)}")
    print(f"  高新企业: {high_tech_count}")
    print(f"  孤儿映射: {sum(1 for e in enriched if e['original_major_industry'] == '高新企业' and e['sub_industry'] not in sub_to_major)}")
    print("\n确定性建边完成!")


if __name__ == "__main__":
    main()
