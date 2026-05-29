#!/usr/bin/env python3
"""
Session 1 收尾: 企业结构化数据与时序对齐 (Enterprise Feature Engineering)

功能:
1. 双层行业拓扑: sub_industry (所属行业) + major_industry (文件名/6大类)
2. 静态特征: 注册资本解析 → 万 → log → 同子行业中位数填充
3. 规模标签: scale_category (基于2024年参保人数)
4. 时序张量 (2017-2024): 线性插值 + padding_mask + log_values

输入: data/raw/enterprises-source/*.xlsx
输出: data/processed/enterprises_final.json
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# -------------------------------------------------------
# 1. 常量
# -------------------------------------------------------

YEARS = list(range(2017, 2025))  # 2017-2024 (含)
SCALE_BINS = [
    (0, 5, 0),      # 微型: <5
    (5, 20, 1),     # 小型: 5-20
    (20, 100, 2),   # 中型: 21-100
    (100, float("inf"), 3),  # 大型: >100
]

# 6 个文件名 → major_industry
INDUSTRY_FILE_MAP = {
    "制造业": "制造业",
    "文化、体育和娱乐业": "文化、体育和娱乐业",
    "水利、环境和公共设施管理业": "水利、环境和公共设施管理业",
    "电力、热力、燃气及水生产和供应业": "电力、热力、燃气及水生产和供应业",
    "科学研究和技术服务业": "科学研究和技术服务业",
    "高新企业": "高新企业",
}

ENTERPRISE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "enterprises-source"

# -------------------------------------------------------
# 2. 解析工具
# -------------------------------------------------------

def parse_capital(s) -> float | None:
    """解析注册资本，返回万元为单位的浮点数。"""
    if pd.isna(s):
        return None
    raw = str(s).replace(",", "").replace("，", "").strip()
    if raw in ["", "-", "--", "0", "0万(元)", "0万元", "0万"]:
        return None
    # "100万(元)" → 100
    m = re.search(r"([\d.]+)\s*万", raw)
    if m:
        val = float(m.group(1))
        return val if val > 0 else None
    # 纯数字 (罕见: 仅"元")
    m = re.search(r"([\d.]+)\s*元", raw)
    if m:
        val = float(m.group(1))
        if val > 10000:
            return val / 10000
        return val if val > 0 else None
    return None


def parse_insurance(v) -> int | None:
    """解析参保人数，返回整数；NaN/空字符串返回 None。"""
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s in ["nan", "", "-", "--", "None"]:
        return None
    s = re.sub(r"[人,，]", "", s)
    try:
        val = float(s)
        if val < 0:
            return None
        return int(val)
    except ValueError:
        return None


def classify_scale(latest_insurance: int | None) -> int:
    """基于最新参保人数返回规模标签: 0=微型, 1=小型, 2=中型, 3=大型。"""
    if latest_insurance is None:
        return 0
    for lo, hi, label in SCALE_BINS:
        if lo <= latest_insurance < hi:
            return label
    return 3


# -------------------------------------------------------
# 3. 时序处理核心
# -------------------------------------------------------

def build_time_series(
    raw_vals: List[Optional[int]],
) -> Tuple[List[float], List[float], List[int]]:
    """
    输入: 原始参保人数列表 (8元素, 2017-2024), None = 缺失

    返回: (values, log_values, padding_mask)
      - values: 插值后的原始值 (float)
      - log_values: log(values+1)，mask=0 处为 0.0
      - padding_mask: 存续期=1, 未成立=0
    """
    n = len(raw_vals)
    # 1. 找出首个非零有效值的位置 → start_idx
    start_idx = n  # 默认全部未成立
    for i, v in enumerate(raw_vals):
        if v is not None and v > 0:
            start_idx = i
            break
    if start_idx == n:
        # 全部为 None/0: 全 mask=0
        return (
            [0.0] * n,
            [0.0] * n,
            [0] * n,
        )

    # 2. 构建 mask
    mask = [0] * n
    for i in range(start_idx, n):
        mask[i] = 1

    # 3. 线性插值 (仅 mask=1 区间内)
    #    先标记已知点
    known_idx = []
    known_val = []
    for i in range(n):
        if mask[i] == 1 and raw_vals[i] is not None:
            known_idx.append(i)
            known_val.append(float(raw_vals[i]))

    # 若 mask=1 区间内没有已知点 (罕见: start_idx 后全为 None)
    # 这种情况下我们无法确定值，用 0 填充
    if not known_idx:
        filled = [0.0] * n
        log_filled = [0.0] * n
        return filled, log_filled, mask

    # 对 mask=1 区间内每个位置进行插值
    filled = [0.0] * n
    for i in range(n):
        if mask[i] == 0:
            filled[i] = 0.0
            continue
        if raw_vals[i] is not None:
            filled[i] = float(raw_vals[i])
            continue
        # 线性插值: 找左右最近已知点
        left, right = None, None
        for j in range(i - 1, -1, -1):
            if mask[j] == 1 and raw_vals[j] is not None:
                left = (j, float(raw_vals[j]))
                break
        for j in range(i + 1, n):
            if mask[j] == 1 and raw_vals[j] is not None:
                right = (j, float(raw_vals[j]))
                break
        if left is not None and right is not None:
            # 线性插值
            ratio = (i - left[0]) / (right[0] - left[0])
            filled[i] = left[1] + ratio * (right[1] - left[1])
        elif left is not None:
            filled[i] = left[1]
        elif right is not None:
            filled[i] = right[1]
        else:
            filled[i] = 0.0

    # 4. log_values
    log_filled = [math.log(v + 1.0) if mask[i] == 1 else 0.0 for i, v in enumerate(filled)]

    return filled, log_filled, mask


# -------------------------------------------------------
# 4. 主流水线
# -------------------------------------------------------

def process_enterprises() -> List[dict]:
    files = sorted(ENTERPRISE_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"未找到企业数据文件: {ENTERPRISE_DIR}")

    all_records: List[dict] = []        # 原始解析结果
    capital_by_sub_industry: Dict[str, List[float]] = defaultdict(list)

    print(f"找到 {len(files)} 个行业文件")

    # ---- 第一轮: 读取 + 解析 ----
    for fp in files:
        major_name = INDUSTRY_FILE_MAP.get(fp.stem, fp.stem)
        df = pd.read_excel(fp)

        # 列名标准化
        col_map = {
            "企业名称": "name",
            "经营状态": "status",
            "注册资本": "capital_raw",
            "实缴资本": "paid_capital",
            "所属行业": "sub_industry",
            "经营范围": "scope",
        }
        for cn, en in col_map.items():
            if cn in df.columns:
                df.rename(columns={cn: en}, inplace=True)

        # 保险列统一命名
        ins_cols_raw = {str(c): int(re.search(r"(\d{4})", str(c)).group(1))
                        for c in df.columns if "年" in str(c) and "参保" in str(c)}
        ins_col_map = {c: f"ins_{yr}" for c, yr in ins_cols_raw.items() if 2017 <= yr <= 2024}
        df.rename(columns=ins_col_map, inplace=True)

        for _, row in df.iterrows():
            status = str(row.get("status", "")).strip()
            if status != "开业":
                continue

            name = str(row.get("name", "")).strip()
            if not name:
                continue

            sub_ind = str(row.get("sub_industry", "")).strip()
            if not sub_ind or sub_ind in ["-", "nan"]:
                sub_ind = "未分类"

            # 注册资本
            capital = parse_capital(row.get("capital_raw"))
            if capital is not None and capital > 0:
                capital_by_sub_industry[sub_ind].append(capital)

            # 参保序列 2017-2024
            raw_ins = []
            for yr in YEARS:
                col = f"ins_{yr}"
                raw_ins.append(parse_insurance(row.get(col)) if col in df.columns else None)

            all_records.append({
                "name": name,
                "major_industry": major_name,
                "sub_industry": sub_ind,
                "capital_raw_wan": capital,
                "capital_log": None,  # 待填充
                "scope": str(row.get("scope", "")) if not pd.isna(row.get("scope")) else "",
                "raw_insurance": raw_ins,
            })

    print(f"  开业企业: {len(all_records)}")
    print(f"  子行业数: {len(capital_by_sub_industry)}")

    # ---- 第二轮: 中位数填充 + log ----
    sub_industry_medians: Dict[str, float] = {}
    global_median = 0.0
    all_caps = [v for vals in capital_by_sub_industry.values() for v in vals]
    if all_caps:
        all_caps.sort()
        global_median = all_caps[len(all_caps) // 2]

    for sub_ind, caps in capital_by_sub_industry.items():
        if caps:
            caps.sort()
            sub_industry_medians[sub_ind] = caps[len(caps) // 2]
        else:
            sub_industry_medians[sub_ind] = global_median

    n_capital_filled = 0
    for rec in all_records:
        cap = rec["capital_raw_wan"]
        if cap is None or cap <= 0:
            cap = sub_industry_medians.get(rec["sub_industry"], global_median)
            n_capital_filled += 1
        rec["capital_log"] = round(math.log(cap + 1.0), 6)
        rec["capital_raw_wan"] = round(cap, 2)

    print(f"  注册资本中位数填充: {n_capital_filled} 条")

    # ---- 第三轮: 时序张量 + 规模标签 ----
    output = []
    for rec in all_records:
        raw_ins = rec["raw_insurance"]
        values, log_values, mask = build_time_series(raw_ins)

        # 规模标签 (基于2024年最新有效值)
        latest_valid = None
        for v in reversed(values):
            if v > 0:
                latest_valid = int(v)
                break
        scale = classify_scale(latest_valid)

        output.append({
            "name": rec["name"],
            "major_industry": rec["major_industry"],
            "sub_industry": rec["sub_industry"],
            "capital_wan": rec["capital_raw_wan"],
            "capital_log": rec["capital_log"],
            "scale_category": scale,
            "scope": rec["scope"],
            "insurance_time_series": {
                "years": YEARS,
                "values": [round(v, 1) for v in values],
                "log_values": [round(v, 6) for v in log_values],
                "padding_mask": mask,
            },
        })

    # 统计
    from collections import Counter
    scale_dist = Counter(o["scale_category"] for o in output)
    major_dist = Counter(o["major_industry"] for o in output)
    mask_coverage = sum(
        sum(o["insurance_time_series"]["padding_mask"]) for o in output
    )

    print(f"\n  最终输出: {len(output)} 条")
    print(f"  行业分布: {dict(major_dist)}")
    print(f"  规模分布: 微型={scale_dist[0]}, 小型={scale_dist[1]}, 中型={scale_dist[2]}, 大型={scale_dist[3]}")
    print(f"  总有效人年: {mask_coverage:,}")

    return output


def main():
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "processed" / "enterprises_final.json"

    print("=" * 60)
    print("企业结构化数据与时序对齐")
    print("=" * 60)

    records = process_enterprises()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n  输出: {output_path}")
    print("[OK] 企业特征工程完成")


if __name__ == "__main__":
    main()
