#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理政策相关三元组与原始政策/企业数据，生成中间结构化文件：
- policies_clean.parquet：政策节点（含标题/正文/发布时间/级别）
- triples_policy_policy.parquet：政策-政策三元组（含头/尾政策的 year/level/发布时间/raw_id/发布单位 等，由原始 JSON 对齐后写入）
- triples_policy_entity.parquet：政策-企业、企业-行业等三元组（原样保留并打标签）
- policies_filtered.parquet：与三元组对齐后的政策文本
- enterprises_filtered.parquet：与三元组对齐后的企业文本

默认输入：
- output/policy_policy_only.json（政策-政策）
- output/extracted_triples_cleaned.json（政策-企业、企业-行业）
- data/extracted_new_data.json（原始政策/企业内容）
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


STOP_WORDS = {"我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们"}
YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def load_industry_mapping(path: Path) -> Dict[str, List[str]]:
    """加载细分行业→行业大类映射表（基于GB/T 4754标准）。"""
    if not path.exists():
        print(f"警告: 行业映射文件 {path} 不存在，跳过映射")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: Dict[str, List[str]] = {}
    for fine_name, info in data.get("fine_industry_mapping", {}).items():
        mapping[fine_name] = info.get("majors", [])
    for major in data.get("major_industries", []):
        mapping[major] = [major]
    return mapping


def map_industry_to_majors(industry_name: str, mapping: Dict[str, List[str]]) -> List[str]:
    """将细分行业名称映射为行业大类列表。无法映射时保留原名。"""
    if not mapping:
        return [industry_name]
    majors = mapping.get(industry_name)
    if majors:
        return majors
    for key, vals in mapping.items():
        if key in industry_name or industry_name in key:
            return vals
    return [industry_name]


@dataclass
class PolicyNode:
    policy_id: int
    title: str
    content: str = ""
    year: Optional[int] = None
    publish_date: Optional[str] = None
    level: Optional[str] = None
    entity_type: str = "policy"
    extra: Dict[str, Optional[str]] = field(default_factory=dict)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    for w in STOP_WORDS:
        cleaned = cleaned.replace(w, "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def extract_year(text: str) -> Optional[int]:
    match = YEAR_PATTERN.search(text)
    if match:
        return int(match.group())
    return None


def load_json_list(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"文件 {path} 不是列表结构")
    return data


def load_raw_data(path: Path) -> Tuple[List[dict], List[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"文件 {path} 不是字典结构")
    policies = data.get("policies", [])
    enterprises = data.get("enterprises", [])
    if not isinstance(policies, list) or not isinstance(enterprises, list):
        raise ValueError("原始数据缺少 policies/enterprises 列表")
    return policies, enterprises


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_policy_nodes(policy_triples: List[dict]) -> Dict[str, PolicyNode]:
    policy_map: Dict[str, PolicyNode] = {}
    next_id = 0
    for rec in policy_triples:
        for key in ("subject", "object"):
            raw = normalize_title(rec.get(key, ""))
            if not raw:
                continue
            if raw not in policy_map:
                policy_map[raw] = PolicyNode(
                    policy_id=next_id,
                    title=raw,
                    content="",
                    year=extract_year(raw),
                    publish_date=None,
                    level=None,
                )
                next_id += 1
    return policy_map


def normalize_title(title: str) -> str:
    text = clean_text(title)
    text = text.replace("【已失效】", "").replace("（已失效）", "")
    text = re.sub(r"[《》“”]", "", text)
    return text.strip()


def _policy_edge_side_fields(node: PolicyNode, prefix: str) -> Dict[str, Any]:
    """
    政策-政策边头/尾侧标量字段（与 extracted_new_data 中 policies 项对齐）。
    正文仍在 policies_clean / policies_filtered，不在此重复，以免 Parquet 膨胀。
    """
    ex = node.extra or {}
    pub = ex.get("publisher")
    pub_s = str(pub).strip() if pub not in (None, "") else None
    return {
        f"{prefix}_year": node.year,
        f"{prefix}_level": node.level,
        f"{prefix}_publish_date": node.publish_date,
        f"{prefix}_raw_id": ex.get("raw_id"),
        f"{prefix}_publisher": pub_s,
        f"{prefix}_record_type": ex.get("record_type"),
        f"{prefix}_interpretation": ex.get("interpretation"),
    }


def enrich_policy_nodes(
    policy_map: Dict[str, PolicyNode], raw_policies: List[dict]
) -> pd.DataFrame:
    rows = []
    for rec in raw_policies:
        title = normalize_title(rec.get("标题", ""))
        if not title:
            continue
        if title not in policy_map:
            continue

        content = clean_text(rec.get("内容", ""))
        publish_date = rec.get("发布时间")
        level = rec.get("政策级别")
        year = extract_year(str(publish_date)) or extract_year(title)

        node = policy_map[title]
        node.content = content
        node.publish_date = publish_date
        node.level = level
        node.year = node.year or year
        node.extra["raw_id"] = rec.get("id")
        node.extra["publisher"] = rec.get("发布单位")
        node.extra["record_type"] = rec.get("type")
        node.extra["interpretation"] = rec.get("政策解读")

        rows.append(
            {
                "policy_id": node.policy_id,
                "title": node.title,
                "content": node.content,
                "publish_date": node.publish_date,
                "level": node.level,
                "year": node.year,
                "raw_id": node.extra.get("raw_id"),
                "publisher": node.extra.get("publisher"),
                "record_type": node.extra.get("record_type"),
                "interpretation": node.extra.get("interpretation"),
            }
        )
    return pd.DataFrame(rows)


def process_policy_policy(triples: List[dict], policy_map: Dict[str, PolicyNode]) -> pd.DataFrame:
    rows = []
    for rec in triples:
        subj = normalize_title(rec.get("subject", ""))
        obj = normalize_title(rec.get("object", ""))
        pred = rec.get("predicate", "")
        if not subj or not obj or not pred:
            continue
        if subj not in policy_map or obj not in policy_map:
            continue
        head_node = policy_map[subj]
        tail_node = policy_map[obj]
        row: Dict[str, Any] = {
            "head_id": head_node.policy_id,
            "head_name": subj,
            "relation": pred,
            "tail_id": tail_node.policy_id,
            "tail_name": obj,
            "source": rec.get("source"),
            "score": rec.get("tfidf_similarity"),
        }
        row.update(_policy_edge_side_fields(head_node, "head"))
        row.update(_policy_edge_side_fields(tail_node, "tail"))
        rows.append(row)
    return pd.DataFrame(rows)


def tag_entity(subject: str, obj: str, predicate: str) -> dict:
    """根据谓词打标签，方便后续融合。"""
    pred_lower = predicate.lower()
    if pred_lower.startswith("belongs"):
        return {"subject_type": "company", "object_type": "industry"}
    elif pred_lower == "targetsindustry":
        return {"subject_type": "policy", "object_type": "industry"}
    elif pred_lower == "supports":
        return {"subject_type": "policy", "object_type": "company"}
    elif "policy" in pred_lower:
        return {"subject_type": "policy", "object_type": "entity"}
    return {"subject_type": "entity", "object_type": "entity"}


def process_policy_entity(
    triples: List[dict],
    industry_mapping: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    seen = set()
    rows = []
    for rec in triples:
        subj = clean_text(rec.get("subject", ""))
        obj = clean_text(rec.get("object", ""))
        pred = rec.get("predicate", "")
        if not subj or not obj or not pred:
            continue
        tags = tag_entity(subj, obj, pred)

        if tags["object_type"] == "industry" and industry_mapping:
            original_industry = obj
            majors = map_industry_to_majors(obj, industry_mapping)
            for major in majors:
                dedup_key = (subj, pred, major)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                rows.append(
                    {
                        "subject": subj,
                        "predicate": pred,
                        "object": major,
                        "subject_type": tags["subject_type"],
                        "object_type": tags["object_type"],
                        "original_industry": original_industry,
                        "source": rec.get("source"),
                    }
                )
        else:
            dedup_key = (subj, pred, obj)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            rows.append(
                {
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "subject_type": tags["subject_type"],
                    "object_type": tags["object_type"],
                    "original_industry": "",
                    "source": rec.get("source"),
                }
            )
    return pd.DataFrame(rows)


def extract_company_set(triples: List[dict]) -> set:
    companies = set()
    for rec in triples:
        pred = str(rec.get("predicate", "")).lower()
        if pred.startswith("belongs"):
            subj = clean_text(rec.get("subject", ""))
            if subj:
                companies.add(subj)
    return companies


def extract_insurance_time_series(insurance_data: dict) -> dict:
    """
    从参保人员数据字典中提取时间序列。
    
    输入格式: {"2024年参保人数": 21, "2023年参保人数": 21, ...}
    输出格式: {2024: 21, 2023: 21, ...} (年份为整数key)
    """
    if not insurance_data or not isinstance(insurance_data, dict):
        return {}
    
    time_series = {}
    for key, value in insurance_data.items():
        # 提取年份：从"2024年参保人数"中提取2024
        if "年" in str(key):
            try:
                year_str = str(key).split("年")[0]
                year = int(year_str)
                # 确保年份在合理范围内（1990-2030）
                if 1990 <= year <= 2030:
                    # 处理value：可能是字符串或数字
                    if isinstance(value, str):
                        # 去除"人"等后缀，提取数字
                        value_clean = value.replace("人", "").strip()
                        try:
                            value_int = int(float(value_clean))
                        except:
                            value_int = 0
                    else:
                        value_int = int(value) if value else 0
                    time_series[year] = max(0, value_int)  # 确保非负
            except (ValueError, IndexError):
                continue
    
    return time_series


def export_policy_nodes(policy_map: Dict[str, PolicyNode]) -> pd.DataFrame:
    rows = []
    for node in policy_map.values():
        ex = node.extra or {}
        rows.append(
            {
                "policy_id": node.policy_id,
                "title": node.title,
                "content": node.content,
                "year": node.year,
                "publish_date": node.publish_date,
                "level": node.level,
                "entity_type": node.entity_type,
                "raw_id": ex.get("raw_id"),
                "publisher": ex.get("publisher"),
                "record_type": ex.get("record_type"),
                "interpretation": ex.get("interpretation"),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="预处理政策三元组，生成中间文件")
    parser.add_argument("--policy_policy_path", type=str, default="output/policy_policy_only.json")
    parser.add_argument("--policy_entity_path", type=str, default="output/extracted_triples_cleaned.json")
    parser.add_argument("--raw_data_path", type=str, default="data/extracted_new_data.json")
    parser.add_argument("--industry_mapping_path", type=str, default="industry_mapping_complete.json")
    parser.add_argument("--out_dir", type=str, default="data_intermediate")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    policy_policy_path = project_root / args.policy_policy_path
    policy_entity_path = project_root / args.policy_entity_path
    raw_data_path = project_root / args.raw_data_path
    out_dir = project_root / args.out_dir

    industry_mapping_path = project_root / args.industry_mapping_path

    ensure_dir(out_dir)

    print("加载行业映射表...")
    industry_mapping = load_industry_mapping(industry_mapping_path)
    print(f"行业映射条目数: {len(industry_mapping):,}")

    print("读取政策-政策三元组...")
    policy_policy_triples = load_json_list(policy_policy_path)
    print(f"政策-政策三元组数量: {len(policy_policy_triples):,}")

    print("构建政策节点表...")
    policy_map = build_policy_nodes(policy_policy_triples)
    print(f"政策节点数: {len(policy_map):,}")

    print("读取原始政策/企业数据...")
    raw_policies, raw_enterprises = load_raw_data(raw_data_path)
    print(f"原始政策数: {len(raw_policies):,}，原始企业数: {len(raw_enterprises):,}")

    print("对齐政策文本（写入 policy_map，供政策-政策边携带元数据）...")
    df_policy_filtered = enrich_policy_nodes(policy_map, raw_policies)

    print("处理政策-政策三元组...")
    df_policy_policy = process_policy_policy(policy_policy_triples, policy_map)

    print("读取政策-企业/行业三元组...")
    policy_entity_triples = load_json_list(policy_entity_path)
    print(f"政策-企业/行业三元组数量: {len(policy_entity_triples):,}")

    print("处理政策-企业/行业三元组（含行业大类映射）...")
    df_policy_entity = process_policy_entity(policy_entity_triples, industry_mapping)

    print("对齐企业文本...")
    company_set = extract_company_set(policy_entity_triples)
    enterprise_rows = []
    enterprise_time_series_rows = []
    
    for ent in raw_enterprises:
        name = clean_text(ent.get("企业名称", ""))
        if not name or name not in company_set:
            continue
        
        enterprise_id = ent.get("id")
        fine_industry = ent.get("所属行业", "") or ""
        raw_major = ent.get("行业大类", "") or ""

        mapped_majors = map_industry_to_majors(fine_industry, industry_mapping) if fine_industry else [raw_major]
        industry_major_str = "；".join(mapped_majors) if mapped_majors else raw_major

        industry_text_parts = []
        if fine_industry:
            industry_text_parts.append(f"细分行业：{fine_industry}")
        if industry_major_str:
            industry_text_parts.append(f"行业大类：{industry_major_str}")
        industry_text_supplement = "；".join(industry_text_parts)

        enterprise_rows.append(
            {
                "enterprise_id": enterprise_id,
                "name": name,
                "industry": fine_industry,
                "industry_major": industry_major_str,
                "scope": ent.get("经营范围"),
                "status": ent.get("经营状态"),
                "text_with_industry": f"{name} {industry_text_supplement}" if industry_text_supplement else name,
            }
        )
        
        # 提取参保人员时间序列数据
        insurance_data = ent.get("参保人员数据", {})
        time_series = extract_insurance_time_series(insurance_data)
        
        if time_series:
            # 保存时间序列数据（每个企业一条记录，时间序列作为字典）
            enterprise_time_series_rows.append(
                {
                    "enterprise_id": enterprise_id,
                    "name": name,
                    "time_series": time_series,  # 字典格式：{2024: 21, 2023: 21, ...}
                }
            )
    
    df_enterprises_filtered = pd.DataFrame(enterprise_rows)
    df_enterprises_time_series = pd.DataFrame(enterprise_time_series_rows)

    print("导出政策节点表...")
    df_policies = export_policy_nodes(policy_map)

    policies_path = out_dir / "policies_clean.parquet"
    p2p_path = out_dir / "triples_policy_policy.parquet"
    p2e_path = out_dir / "triples_policy_entity.parquet"
    policies_filtered_path = out_dir / "policies_filtered.parquet"
    enterprises_filtered_path = out_dir / "enterprises_filtered.parquet"
    enterprises_time_series_path = out_dir / "enterprises_time_series.parquet"

    df_policies.to_parquet(policies_path, index=False)
    df_policy_policy.to_parquet(p2p_path, index=False)
    df_policy_entity.to_parquet(p2e_path, index=False)
    df_policy_filtered.to_parquet(policies_filtered_path, index=False)
    df_enterprises_filtered.to_parquet(enterprises_filtered_path, index=False)
    
    # 保存企业时间序列数据（如果parquet不支持字典，可以保存为JSON字符串）
    if len(df_enterprises_time_series) > 0:
        # 将字典转换为JSON字符串保存（parquet可能不支持嵌套字典）
        df_enterprises_time_series["time_series_json"] = df_enterprises_time_series["time_series"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if x else "{}"
        )
        df_enterprises_time_series_export = df_enterprises_time_series[["enterprise_id", "name", "time_series_json"]].copy()
        df_enterprises_time_series_export.to_parquet(enterprises_time_series_path, index=False)
        print(f"- 企业时间序列数据: {enterprises_time_series_path} (共{len(df_enterprises_time_series)}条)")

    print("[OK] 预处理完成")
    print(f"- 政策节点: {policies_path}")
    print(f"- 政策-政策三元组: {p2p_path}")
    print(f"- 政策-企业/行业三元组: {p2e_path}")
    print(f"- 对齐政策文本: {policies_filtered_path}")
    print(f"- 对齐企业文本: {enterprises_filtered_path}")


if __name__ == "__main__":
    main()

