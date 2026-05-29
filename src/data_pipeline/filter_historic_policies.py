#!/usr/bin/env python3
"""
Session 1 收尾: 政策时间截断 (Left-Truncation)
筛选 pub_date >= 2018-01-01 的政策，去除早期稀疏数据。

输入: data/processed/policies_cleaned.json
输出: data/processed/policies_final.json
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

CUTOFF_DATE = date(2018, 1, 1)


def parse_date(d: str) -> date | None:
    try:
        parts = str(d).strip().split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return None


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "processed" / "policies_cleaned.json"
    output_path = project_root / "data" / "processed" / "policies_final.json"

    print(f"读取: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    active = [r for r in data if r.get("status") == "active"]
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for r in active:
        d = parse_date(r.get("pub_date", ""))
        if d is None:
            # 无有效日期 → 保留（保守策略）
            kept.append(r)
        elif d >= CUTOFF_DATE:
            kept.append(r)
        else:
            dropped.append(r)

    # 重新编号 policy_id
    for i, r in enumerate(kept):
        r["policy_id"] = f"P_{i:04d}"

    print(f"  Active (清洗后): {len(active)}")
    print(f"  截断后保留: {len(kept)}")
    print(f"  去除 (早于 {CUTOFF_DATE}): {len(dropped)}")
    if dropped:
        print(f"  去除样例:")
        for r in dropped:
            print(f"    {r['policy_id']}: {r['pub_date']} | {r['title'][:80]}")

    # 保留 pruned 记录不变，追加到输出
    pruned = [r for r in data if r.get("status") != "active"]
    output = kept + pruned

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  输出: {output_path}")
    print(f"  总计: {len(output)} 条 (active={len(kept)}, pruned={len(pruned)})")
    print("[OK] 政策时间截断完成")


if __name__ == "__main__":
    main()
