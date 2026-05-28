#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政策重要性评估与衰减模型：
- 结合PageRank重要性分数
- 实现层级衰减：V_policy^(l) = V_policy * e^(-β * Δl)，其中Δl = |l_policy - l_region|
- 实现时间衰减：V_policy(t) = V_policy * e^(-β * Δt)，其中Δt = |t_policy - t_region|
- 输出综合重要性评估结果
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from data_clean.preprocess_policies import clean_text, extract_year, normalize_title

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


# 层级映射：中央=0, 省=1, 市=2
LEVEL_MAPPING = {
    "国家": 0,      # 中央级别
    "中央": 0,
    "自治区": 1,    # 省级
    "省": 1,
    "柳州": 2,      # 市级
    "市": 2,
    "市级": 2,
}

# 衰减因子
BETA_LEVEL = 0.2   # 层级衰减因子
BETA_TIME = 0.05   # 时间衰减因子（年）


def map_level_to_numeric(level: str) -> int:
    """将层级字符串映射为数值"""
    if pd.isna(level) or not isinstance(level, str):
        return 1  # 默认为省级
    
    level = level.strip()
    for key, value in LEVEL_MAPPING.items():
        if key in level:
            return value
    
    return 1  # 默认为省级


def _attrs_from_raw_policies_json(project_root: Path, rel_path: str) -> Dict[str, Dict[str, Any]]:
    """从 extracted_new_data.json 的 policies[] 按 normalize_title(标题) 建 year/level 索引（与预处理一致）。"""
    if not (rel_path or "").strip():
        return {}
    p = project_root / rel_path.strip()
    if not p.is_file():
        print(f"   提示: 未找到原始政策 JSON，跳过按标题补全: {p}")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    policies = data.get("policies")
    if not isinstance(policies, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for rec in policies:
        t = normalize_title(rec.get("标题", ""))
        if not t:
            continue
        pd_raw = rec.get("发布时间")
        y = extract_year(str(pd_raw)) if pd_raw else None
        if y is None:
            y = extract_year(t)
        lv = rec.get("政策级别")
        cell: Dict[str, Any] = {}
        if y is not None:
            cell["year"] = int(y)
        if lv is not None and str(lv).strip():
            cell["level"] = str(lv).strip()
        if not cell:
            continue
        # 政策-实体三元组 subject 多为 clean_text，政策-政策为 normalize_title；双键索引减少漏配
        for key in {x for x in (t, clean_text(rec.get("标题", ""))) if x}:
            out[key] = dict(cell)
    print(f"   原始政策 JSON 按标题索引: {p} ({len(out):,} 条键，含 clean/normalize 别名)")
    return out


def _attrs_from_policy_policy_p2p(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """从 triples_policy_policy.parquet 头尾列聚合 year/level（与预处理 enrich 后写出的列一致）。"""
    if df is None or len(df) == 0:
        return {}
    cols = set(df.columns)
    if "head_name" not in cols or "tail_name" not in cols:
        return {}
    if not any(c in cols for c in ("head_year", "head_level", "tail_year", "tail_level")):
        print("   提示: triples_policy_policy 无 head_/tail_ year/level 列，跳过 P2P 补全")
        return {}
    out: Dict[str, Dict[str, Any]] = {}

    def take(name: Any, y: Any, lv: Any) -> None:
        if name is None or (isinstance(name, float) and pd.isna(name)):
            return
        k = str(name).strip()
        if not k:
            return
        if k not in out:
            out[k] = {}
        if y is not None and pd.notna(y) and "year" not in out[k]:
            try:
                out[k]["year"] = int(y)
            except (TypeError, ValueError):
                pass
        if lv is not None and pd.notna(lv) and str(lv).strip() and "level" not in out[k]:
            out[k]["level"] = str(lv).strip()

    for _, row in df.iterrows():
        if "head_year" in cols or "head_level" in cols:
            take(row.get("head_name"), row.get("head_year"), row.get("head_level"))
        if "tail_year" in cols or "tail_level" in cols:
            take(row.get("tail_name"), row.get("tail_year"), row.get("tail_level"))
    print(f"   政策-政策 Parquet 按标题补全: {len(out):,} 条")
    return out


def _supplement_record_for_title(supplement: Dict[str, Dict[str, Any]], title: str) -> Dict[str, Any]:
    """node_maps 标题与 clean_text / normalize_title 键不一致时依次尝试。"""
    seen = []
    for c in (str(title).strip(), normalize_title(title), clean_text(title)):
        if not c or c in seen:
            continue
        seen.append(c)
        if c in supplement:
            return supplement[c]
    return {}


def _merge_title_supplements_fill_missing(layers: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """后者仅填补前者缺失的 year/level 字段。"""
    merged: Dict[str, Dict[str, Any]] = {}
    for layer in layers:
        for title, rec in layer.items():
            k = str(title).strip()
            if not k:
                continue
            if k not in merged:
                merged[k] = {}
            for fld in ("year", "level"):
                if fld not in rec or rec[fld] is None:
                    continue
                if fld == "level" and isinstance(rec[fld], str) and not str(rec[fld]).strip():
                    continue
                if merged[k].get(fld) is None:
                    merged[k][fld] = rec[fld]
    return merged


def _load_policies_supplement(project_root: Path, rel_path: str) -> Dict[str, Dict[str, object]]:
    """可选：按标题合并 year/level（列名兼容 title/name, year, level）。"""
    p = project_root / rel_path.strip()
    if not rel_path.strip() or not p.is_file():
        return {}
    df = pd.read_parquet(p)
    title_col = None
    for c in ("title", "name", "policy_title", "标题"):
        if c in df.columns:
            title_col = c
            break
    if title_col is None:
        print(f"   警告: 补充表无 title 列，忽略: {p}")
        return {}
    out: Dict[str, Dict[str, object]] = {}
    ycol = "year" if "year" in df.columns else ("publish_year" if "publish_year" in df.columns else None)
    lcol = "level" if "level" in df.columns else None
    for _, row in df.iterrows():
        key = str(row[title_col]).strip()
        if not key:
            continue
        rec: Dict[str, object] = {}
        if ycol and pd.notna(row.get(ycol)):
            try:
                rec["year"] = int(row[ycol])
            except (TypeError, ValueError):
                pass
        if lcol and pd.notna(row.get(lcol)):
            rec["level"] = str(row[lcol]).strip()
        if rec:
            out[key] = rec
    print(f"   已加载政策属性补充表: {p} ({len(out)} 条按标题索引)")
    return out


def build_graph_only_policy_row(
    policy_id: int,
    title: str,
    target_year: int,
    supplement: Dict[str, Dict[str, object]],
) -> Tuple[Dict[str, object], Dict[str, bool]]:
    """
    图中独有节点（不在 policies_clean 行表里）的属性行。
    仅使用 supplement（原始 JSON / P2P / filtered / 用户表）中的 year、level；缺失时才用目标年与「自治区」占位。
    """
    flags = {"year_default": False, "level_default": False}
    sup = _supplement_record_for_title(supplement, title)
    year_v: Optional[int] = None
    level_v: Optional[str] = None
    if "year" in sup and sup["year"] is not None and not (isinstance(sup["year"], float) and pd.isna(sup["year"])):
        try:
            year_v = int(sup["year"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            year_v = None
    if "level" in sup and sup["level"] is not None and not (isinstance(sup["level"], float) and pd.isna(sup["level"])):
        ls = str(sup["level"]).strip()
        if ls:
            level_v = ls
    if year_v is None:
        year_v = target_year
        flags["year_default"] = True
    if level_v is None:
        level_v = "自治区"
        flags["level_default"] = True
    return (
        {
            "policy_id": policy_id,
            "title": title,
            "content": "",
            "year": year_v,
            "publish_date": None,
            "level": level_v,
            "entity_type": "policy",
        },
        flags,
    )


def calculate_level_decay(
    base_importance: np.ndarray,
    policy_levels: np.ndarray,
    target_level: int = 2,  # 默认目标层级为市级（2）
    beta: float = BETA_LEVEL
) -> np.ndarray:
    """
    计算层级衰减后的重要性
    
    公式：V_policy^(l) = V_policy * e^(-β * Δl)
    其中：Δl = |l_policy - l_region|
    
    Args:
        base_importance: 基础重要性分数（PageRank分数）
        policy_levels: 政策层级数组（数值：0=中央, 1=省, 2=市）
        target_level: 目标区域层级（默认2=市级）
        beta: 层级衰减因子（默认0.2）
    
    Returns:
        层级衰减后的重要性分数
    """
    delta_l = np.abs(policy_levels - target_level)
    decay_factor = np.exp(-beta * delta_l)
    decayed_importance = base_importance * decay_factor
    return decayed_importance, delta_l


def calculate_time_decay(
    base_importance: np.ndarray,
    policy_years: np.ndarray,
    target_year: int,
    beta: float = BETA_TIME
) -> np.ndarray:
    """
    计算时间衰减后的重要性
    
    公式：V_policy(t) = V_policy * e^(-β * Δt)
    其中：Δt = |t_policy - t_region|
    
    Args:
        base_importance: 基础重要性分数（可以是原始分数或层级衰减后的分数）
        policy_years: 政策发布年份数组
        target_year: 目标年份（当前年份或区域关注年份）
        beta: 时间衰减因子（默认0.05）
    
    Returns:
        时间衰减后的重要性分数
    """
    delta_t = np.abs(policy_years - target_year)
    decay_factor = np.exp(-beta * delta_t)
    decayed_importance = base_importance * decay_factor
    return decayed_importance, delta_t


def calculate_combined_decay(
    base_importance: np.ndarray,
    policy_levels: np.ndarray,
    policy_years: np.ndarray,
    target_level: int = 2,
    target_year: int = 2024,
    beta_level: float = BETA_LEVEL,
    beta_time: float = BETA_TIME,
    ppr_scores: Optional[np.ndarray] = None,
    alpha_ppr: float = 0.4,
    skip_level: bool = False,
    skip_time: bool = False,
) -> Dict[str, np.ndarray]:
    """
    计算综合衰减（层级衰减 + 时间衰减），并可选融合PPR结构权重。

    公式（含PPR时）:
        V_final = alpha * norm(PPR) + (1 - alpha) * norm( V_chain )
        其中 V_chain = base * e^(-beta_l*Dl) * e^(-beta_t*Dt)（先层级后时间）。

    ppr_scores=None（--no_ppr_fusion）:
        V_final = norm(V_chain)，即仅对衰减链做 min-max，不再与 PageRank/PPR 同源向量混合，
        使层级/时间对 combined_decayed 更可辨。

    skip_level=True（消融 B1：去掉层级衰减）:
        V_chain = base * e^(-beta_t*Dt)，即不在 base 上乘层级因子，再与 PPR 按同样规则融合。

    skip_time=True（消融 B2：去掉时间衰减）:
        V_chain = base * e^(-beta_l*Dl)（即层级衰减后的量），不再乘 e^(-beta_t*Dt)，再与 PPR 融合。
        delta_time 仍写入 |year - target_year| 便于核对，但链路上不施加时间因子。
    """
    if skip_level:
        level_decayed = base_importance.astype(np.float64).copy()
        delta_l = np.zeros(len(base_importance), dtype=np.float64)
    else:
        level_decayed, delta_l = calculate_level_decay(
            base_importance, policy_levels, target_level, beta_level
        )

    policy_years_f = policy_years.astype(np.float64)
    delta_t_abs = np.abs(policy_years_f - float(target_year))
    if skip_time:
        time_decayed = level_decayed.astype(np.float64).copy()
        delta_t = delta_t_abs
    else:
        time_decayed, delta_t = calculate_time_decay(
            level_decayed, policy_years, target_year, beta_time
        )
    
    time_first_decayed, _ = calculate_time_decay(
        base_importance, policy_years, target_year, beta_time
    )
    level_time_decayed, _ = calculate_level_decay(
        time_first_decayed, policy_levels, target_level, beta_level
    )

    combined = time_decayed.copy()

    if ppr_scores is not None:
        ppr_norm = ppr_scores.copy()
        ppr_range = ppr_norm.max() - ppr_norm.min()
        if ppr_range > 1e-12:
            ppr_norm = (ppr_norm - ppr_norm.min()) / ppr_range
        decay_norm = combined.copy()
        decay_range = decay_norm.max() - decay_norm.min()
        if decay_range > 1e-12:
            decay_norm = (decay_norm - decay_norm.min()) / decay_range
        combined = alpha_ppr * ppr_norm + (1 - alpha_ppr) * decay_norm
    else:
        decay_range = combined.max() - combined.min()
        if decay_range > 1e-12:
            combined = (combined - combined.min()) / decay_range

    return {
        "base_importance": base_importance,
        "level_decayed": level_decayed,
        "time_decayed": time_decayed,
        "combined_decayed": combined,
        "combined_decayed_alt": level_time_decayed,
        "delta_level": delta_l,
        "delta_time": delta_t,
    }


def main():
    parser = argparse.ArgumentParser(description="政策重要性评估与衰减模型")
    parser.add_argument("--policies", type=str, default="data_intermediate/policies_clean.parquet")
    parser.add_argument("--importance", type=str, default="graphrag/importance_scores.npy")
    parser.add_argument("--gat_emb", type=str, default="graph/gat_policy_emb.npy", help="GAT结构特征（可选）")
    parser.add_argument("--target_level", type=int, default=2, help="目标区域层级：0=中央, 1=省, 2=市")
    parser.add_argument("--target_year", type=int, default=2024, help="目标年份")
    parser.add_argument("--beta_level", type=float, default=BETA_LEVEL, help="层级衰减因子")
    parser.add_argument("--beta_time", type=float, default=BETA_TIME, help="时间衰减因子")
    parser.add_argument("--use_gat", action="store_true", help="是否使用GAT结构特征增强重要性")
    parser.add_argument("--alpha_ppr", type=float, default=0.4, help="PPR权重（0~1），0=纯衰减，1=纯PPR")
    parser.add_argument(
        "--output_tag",
        type=str,
        default="",
        help="非空时写入 policy_importance_with_decay_{tag}.csv/.parquet 与 policy_importance_stats_{tag}.json，避免覆盖主实验文件",
    )
    parser.add_argument(
        "--no_level_decay",
        action="store_true",
        help="消融 B1：跳过层级衰减，仅在原始 PageRank 上做时间衰减+PPR 融合（与完整管线公式一致，需单独跑本脚本生成 parquet）",
    )
    parser.add_argument(
        "--no_time_decay",
        action="store_true",
        help="消融 B2：跳过时间衰减，在层级衰减后进入 min-max 或（未加 --no_ppr_fusion 时）PPR 融合；delta_time 仍为 |year-target|",
    )
    parser.add_argument(
        "--no_ppr_fusion",
        action="store_true",
        help="不加载、不使用 PPR 支路：combined_decayed = min-max(衰减链)。避免与 PageRank 基线同源重复钉死排序。",
    )
    parser.add_argument(
        "--policies_supplement",
        type=str,
        default="",
        help="可选 parquet：按标题提供 year/level，补齐图中存在但不在 policies_clean 的节点（列含 title, year, level）",
    )
    parser.add_argument(
        "--raw_policy_json",
        type=str,
        default="data/extracted_new_data.json",
        help="原始政策库 JSON（policies 列表）；按 normalize_title(标题) 提供 year/level，覆盖图中扩展节点。置空字符串可关闭",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    
    print("="*60)
    print("政策重要性评估与衰减模型")
    print("="*60)
    
    # 加载数据
    print("\n1. 加载数据...")
    
    # 加载图元信息，获取所有政策节点
    meta_path = project_root / "graph" / "meta.json"
    if meta_path.exists():
        import json
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        node_maps = meta.get("node_maps", {}).get("policy", {})
        print(f"   图中政策节点数: {len(node_maps):,}")
    else:
        node_maps = {}
        print(f"   警告: 未找到meta.json，将只使用policies_clean.parquet中的政策")
    
    # 加载policies_clean.parquet中的政策（有完整属性信息）
    df_policies = pd.read_parquet(project_root / args.policies)
    print(f"   policies_clean.parquet中的政策数: {len(df_policies):,}")
    
    # 加载policy-entity三元组，提取所有政策节点
    p2e_path = project_root / "data_intermediate" / "triples_policy_entity.parquet"
    all_policy_titles = set()
    if p2e_path.exists():
        df_p2e = pd.read_parquet(p2e_path)
        policy_subjects = df_p2e[df_p2e["subject_type"] == "policy"]["subject"].unique()
        all_policy_titles.update(policy_subjects)
        print(f"   policy-entity三元组中的政策数: {len(policy_subjects):,}")
    
    # 合并所有政策标题
    all_policy_titles.update(df_policies["title"].tolist())
    print(f"   合并后唯一政策数: {len(all_policy_titles):,}")
    
    # 加载重要性分数
    importance = np.load(project_root / args.importance)
    print(f"   PageRank重要性分数: {len(importance):,} 个节点")

    # 按标题补全 year/level：原始 JSON → 政策-政策 Parquet → policies_filtered → 用户补充表（后者覆盖前者）
    raw_supp = _attrs_from_raw_policies_json(project_root, args.raw_policy_json)
    p2p_supp: Dict[str, Dict[str, Any]] = {}
    p2p_path_supp = project_root / "data_intermediate" / "triples_policy_policy.parquet"
    if p2p_path_supp.is_file():
        p2p_supp = _attrs_from_policy_policy_p2p(pd.read_parquet(p2p_path_supp))
    filtered_supp = _load_policies_supplement(
        project_root,
        "data_intermediate/policies_filtered.parquet"
        if (project_root / "data_intermediate" / "policies_filtered.parquet").is_file()
        else "",
    )
    user_supp = _load_policies_supplement(project_root, args.policies_supplement)
    title_supplement = _merge_title_supplements_fill_missing([raw_supp, p2p_supp, filtered_supp])
    for t, rec in user_supp.items():
        tt = str(t).strip()
        if not tt:
            continue
        title_supplement[tt] = {**title_supplement.get(tt, {}), **rec}
    print(f"   合并后按标题可查询属性条目: {len(title_supplement):,}")
    
    # 构建完整的政策数据框（包含所有图中的政策节点）
    policy_data = []
    policy_title_to_info = {row["title"]: row for _, row in df_policies.iterrows()}
    default_year_count = 0
    default_level_count = 0
    graph_only_count = 0
    
    # 从node_maps获取所有政策节点（按ID排序）
    if node_maps:
        # 反转映射：从ID到标题
        id_to_title = {v: k for k, v in node_maps.items()}
        sorted_ids = sorted(id_to_title.keys())
        
        for policy_id in sorted_ids:
            title = id_to_title[policy_id]
            if title in policy_title_to_info:
                # 使用policies_clean.parquet中的完整信息
                policy_data.append(policy_title_to_info[title].to_dict())
            else:
                graph_only_count += 1
                row, flags = build_graph_only_policy_row(
                    policy_id, title, args.target_year, title_supplement
                )
                if flags["year_default"]:
                    default_year_count += 1
                if flags["level_default"]:
                    default_level_count += 1
                policy_data.append(row)
        if graph_only_count:
            print(
                f"   图中不在 policies_clean 的扩展节点: {graph_only_count:,}；"
                f"其中仍缺年份而用目标年 {args.target_year}: {default_year_count:,}；"
                f"仍缺层级而用自治区: {default_level_count:,}（同一节点可同时缺两项）"
            )
    else:
        # 如果没有node_maps，使用policies_clean.parquet中的政策
        for _, row in df_policies.iterrows():
            policy_data.append(row.to_dict())
    
    df = pd.DataFrame(policy_data)
    print(f"   最终评估政策数: {len(df):,}")

    # graphrag/importance_scores.npy 与 GAT 导出向量均按「图中政策节点 ID」索引；主表 policy_id 与节点 ID 在本项目一致，
    # 仍显式用 graph_node_id 对齐，避免未来构图若改为非连续 ID 时 silent 错位。
    if "graph_node_id" not in df.columns:
        gn = []
        if node_maps:
            title_to_nid = {str(k): int(v) for k, v in node_maps.items()}
            for _, row in df.iterrows():
                t = str(row["title"])
                gn.append(title_to_nid.get(t, int(row["policy_id"])))
        else:
            gn = [int(r["policy_id"]) for _, r in df.iterrows()]
        df = df.copy()
        df["graph_node_id"] = gn

    # 对齐重要性分数（按图中节点 ID）
    imp_aligned = np.zeros(len(df), dtype=float)
    for idx, row in df.iterrows():
        nid = int(row["graph_node_id"])
        if 0 <= nid < len(importance):
            imp_aligned[idx] = importance[nid]
    
    # 处理层级和年份
    print("\n2. 处理政策属性...")
    policy_levels = df["level"].apply(map_level_to_numeric).values
    policy_years = df["year"].fillna(args.target_year).astype(int).values
    
    level_dist = pd.Series(policy_levels).value_counts().sort_index()
    print(f"   层级分布:")
    for level, count in level_dist.items():
        level_name = {0: "中央", 1: "省级", 2: "市级"}.get(level, f"未知({level})")
        print(f"     {level_name} (l={level}): {count:,} 个政策")
    
    print(f"\n   年份范围: {policy_years.min()} - {policy_years.max()}")
    print(f"   目标层级: {args.target_level} ({'中央' if args.target_level == 0 else '省级' if args.target_level == 1 else '市级'})")
    print(f"   目标年份: {args.target_year}")
    
    # 加载PPR分数（与 --no_ppr_fusion 互斥）
    ppr_aligned = None
    ppr_path = project_root / "graphrag" / "importance_scores.npy"
    if args.no_ppr_fusion:
        print("   【A1】--no_ppr_fusion：跳过 PPR 融合，combined_decayed 仅由衰减链 min-max 得到")
    elif ppr_path.exists():
        ppr_raw = np.load(ppr_path)
        ppr_aligned = np.zeros(len(df), dtype=float)
        for idx, row in df.iterrows():
            nid = int(row["graph_node_id"])
            if 0 <= nid < len(ppr_raw):
                ppr_aligned[idx] = ppr_raw[nid]
        print(f"   加载PPR分数: {ppr_path} (alpha_ppr={args.alpha_ppr})")
    else:
        print(f"   未找到PPR分数文件，仅使用传统衰减")

    # 计算衰减
    print("\n3. 计算衰减模型...")
    if args.no_level_decay:
        print(
            "   【B1】--no_level_decay：跳过层级衰减，在原始 PageRank（或 GAT 混合 base）上直接做时间衰减，"
            "再经 min-max 或（未加 --no_ppr_fusion 时）与 PPR 融合"
        )
    if args.no_time_decay:
        print(
            "   【B2】--no_time_decay：跳过时间衰减，在层级衰减后 min-max 或 PPR 融合（链路上不乘 e^(-β_t·Δt)）"
        )
    print(f"   层级衰减因子 β_level = {args.beta_level}")
    print(f"   时间衰减因子 β_time = {args.beta_time}")
    if ppr_aligned is not None and not args.no_ppr_fusion:
        print(f"   PPR融合权重 α = {args.alpha_ppr}")
    
    decay_results = calculate_combined_decay(
        imp_aligned,
        policy_levels,
        policy_years,
        args.target_level,
        args.target_year,
        args.beta_level,
        args.beta_time,
        ppr_scores=ppr_aligned,
        alpha_ppr=args.alpha_ppr,
        skip_level=bool(args.no_level_decay),
        skip_time=bool(args.no_time_decay),
    )
    
    # 可选：使用GAT结构特征增强重要性
    if args.use_gat and Path(project_root / args.gat_emb).exists():
        print("\n4. 使用GAT结构特征增强重要性...")
        gat_emb = np.load(project_root / args.gat_emb)
        # 按图中政策节点 ID 取行，禁止假定 df 行顺序与 gat_emb 前 len(df) 行一一对应
        gat_importance = np.zeros(len(df), dtype=np.float64)
        for i, row in df.iterrows():
            nid = int(row["graph_node_id"])
            if 0 <= nid < len(gat_emb):
                gat_importance[i] = float(np.linalg.norm(gat_emb[nid]))
        gat_importance_normalized = (gat_importance - gat_importance.min()) / (gat_importance.max() - gat_importance.min() + 1e-8)
        
        # 结合PageRank和GAT重要性（加权平均）
        combined_base = 0.7 * imp_aligned + 0.3 * gat_importance_normalized
        
        decay_results_gat = calculate_combined_decay(
            combined_base,
            policy_levels,
            policy_years,
            args.target_level,
            args.target_year,
            args.beta_level,
            args.beta_time,
            skip_level=bool(args.no_level_decay),
            skip_time=bool(args.no_time_decay),
        )
        decay_results["gat_enhanced_base"] = combined_base
        decay_results["gat_enhanced_decayed"] = decay_results_gat["combined_decayed"]
    else:
        print("\n4. 跳过GAT特征增强（未启用或文件不存在）")
    
    # 统计信息
    print("\n5. 衰减统计...")
    print(f"   原始重要性:")
    print(f"     均值: {np.mean(decay_results['base_importance']):.6f}")
    print(f"     最大值: {np.max(decay_results['base_importance']):.6f}")
    print(f"     最小值: {np.min(decay_results['base_importance']):.6f}")
    
    print(f"\n   层级衰减后:")
    print(f"     均值: {np.mean(decay_results['level_decayed']):.6f}")
    print(f"     最大值: {np.max(decay_results['level_decayed']):.6f}")
    print(f"     最小值: {np.min(decay_results['level_decayed']):.6f}")
    print(f"     平均衰减率: {(1 - np.mean(decay_results['level_decayed']) / np.mean(decay_results['base_importance'])) * 100:.2f}%")
    
    print(f"\n   时间衰减后:")
    time_only_decayed, _ = calculate_time_decay(
        decay_results['base_importance'], policy_years, args.target_year, args.beta_time
    )
    print(f"     均值: {np.mean(time_only_decayed):.6f}")
    print(f"     最大值: {np.max(time_only_decayed):.6f}")
    print(f"     最小值: {np.min(time_only_decayed):.6f}")
    print(f"     平均衰减率: {(1 - np.mean(time_only_decayed) / np.mean(decay_results['base_importance'])) * 100:.2f}%")
    
    if args.no_time_decay:
        _combo_label = "综合衰减后（仅层级+PPR，无时间）"
    elif args.no_level_decay:
        _combo_label = "综合衰减后（无层级，仅时间+PPR）"
    else:
        _combo_label = "综合衰减后（层级+时间）"
    print(f"\n   {_combo_label}:")
    print(f"     均值: {np.mean(decay_results['combined_decayed']):.6f}")
    print(f"     最大值: {np.max(decay_results['combined_decayed']):.6f}")
    print(f"     最小值: {np.min(decay_results['combined_decayed']):.6f}")
    print(f"     平均衰减率: {(1 - np.mean(decay_results['combined_decayed']) / np.mean(decay_results['base_importance'])) * 100:.2f}%")
    
    # 构建输出DataFrame
    print("\n6. 生成输出结果...")
    output_data = {
        "policy_id": df["policy_id"].values,
        "title": df["title"].values,
        "level": df["level"].values,
        "level_numeric": policy_levels,
        "year": policy_years,
        "base_importance": decay_results["base_importance"],
        "level_decayed": decay_results["level_decayed"],
        "time_decayed": time_only_decayed,
        "combined_decayed": decay_results["combined_decayed"],
        "delta_level": decay_results["delta_level"],
        "delta_time": decay_results["delta_time"],
    }
    
    if args.use_gat and "gat_enhanced_decayed" in decay_results:
        output_data["gat_enhanced_base"] = decay_results["gat_enhanced_base"]
        output_data["gat_enhanced_decayed"] = decay_results["gat_enhanced_decayed"]
    
    out_df = pd.DataFrame(output_data)
    
    # 排序（按综合衰减后的重要性降序）
    out_df = out_df.sort_values("combined_decayed", ascending=False).reset_index(drop=True)
    out_df["rank"] = range(1, len(out_df) + 1)
    
    # 保存结果
    out_dir = project_root / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    _tag = (args.output_tag or "").strip()
    _suf = f"_{_tag}" if _tag else ""
    
    # CSV格式（便于查看）
    csv_path = out_dir / f"policy_importance_with_decay{_suf}.csv"
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"   CSV结果: {csv_path}")
    
    # Parquet格式（便于后续处理）
    parquet_path = out_dir / f"policy_importance_with_decay{_suf}.parquet"
    out_df.to_parquet(parquet_path, index=False)
    print(f"   Parquet结果: {parquet_path}")
    
    # 保存统计信息
    stats = {
        "target_level": args.target_level,
        "target_year": args.target_year,
        "beta_level": args.beta_level,
        "beta_time": args.beta_time,
        "no_level_decay": bool(args.no_level_decay),
        "no_time_decay": bool(args.no_time_decay),
        "attribute_sources": {
            "raw_policy_json": (args.raw_policy_json or "").strip(),
            "title_supplement_keys": len(title_supplement),
            "graph_only_policy_nodes": graph_only_count,
            "fallback_year_to_target": default_year_count,
            "fallback_level_to_autonomous": default_level_count,
            "no_ppr_fusion": bool(args.no_ppr_fusion),
        },
        "statistics": {
            "base_importance": {
                "mean": float(np.mean(decay_results["base_importance"])),
                "std": float(np.std(decay_results["base_importance"])),
                "max": float(np.max(decay_results["base_importance"])),
                "min": float(np.min(decay_results["base_importance"])),
            },
            "level_decayed": {
                "mean": float(np.mean(decay_results["level_decayed"])),
                "std": float(np.std(decay_results["level_decayed"])),
                "max": float(np.max(decay_results["level_decayed"])),
                "min": float(np.min(decay_results["level_decayed"])),
            },
            "combined_decayed": {
                "mean": float(np.mean(decay_results["combined_decayed"])),
                "std": float(np.std(decay_results["combined_decayed"])),
                "max": float(np.max(decay_results["combined_decayed"])),
                "min": float(np.min(decay_results["combined_decayed"])),
            },
        },
        "top10_policies": out_df.head(10)[["policy_id", "title", "combined_decayed", "level", "year"]].to_dict("records"),
    }
    
    stats_path = out_dir / f"policy_importance_stats{_suf}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"   统计信息: {stats_path}")
    
    # 显示前10个重要政策
    print("\n7. 前10个重要政策（综合衰减后）:")
    print("-" * 100)
    for idx, row in out_df.head(10).iterrows():
        _lv = row["level"]
        _lv_s = _lv if isinstance(_lv, str) else ("?" if pd.isna(_lv) else str(_lv))
        _yr = int(row["year"]) if pd.notna(row["year"]) else int(args.target_year)
        print(f"{row['rank']:2d}. [{_lv_s:4s}] [{_yr:4d}] "
              f"重要性: {row['combined_decayed']:.6f} | "
              f"层级衰减: {row['level_decayed']:.6f} | "
              f"Δl={row['delta_level']:.0f}, Δt={row['delta_time']:.0f}")
        _tit = str(row["title"]) if pd.notna(row["title"]) else ""
        print(f"    {_tit[:80]}...")
    
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)


if __name__ == "__main__":
    main()

