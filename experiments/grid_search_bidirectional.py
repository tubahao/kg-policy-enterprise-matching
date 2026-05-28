#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化网格搜索：企业->政策 与 政策->企业分方向独立调参。

特性：
1) 仅初始化一次 BidirectionalMatcher，避免重复加载模型带来的额外耗时
2) 支持自定义网格列表
3) 输出全量结果、Pareto前沿、推荐参数（按加权目标）
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 复用现有评估逻辑，保证口径一致
from matching.bidirectional_matching import BidirectionalMatcher  # noqa: E402
from matching.evaluate_matching import (  # noqa: E402
    build_test_queries_from_data,
    evaluate_enterprise_to_policy,
    evaluate_policy_to_enterprise,
)


def _parse_list(raw: str, cast_type):
    vals = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() in {"none", "null", "-1"}:
            vals.append(None)
        else:
            vals.append(cast_type(tok))
    return vals


def _pareto_front(rows: List[Dict[str, Any]], key_x: str, key_y: str) -> List[Dict[str, Any]]:
    """在最大化(key_x, key_y)意义下，返回非支配解集合。"""
    front = []
    for i, a in enumerate(rows):
        dominated = False
        ax, ay = a[key_x], a[key_y]
        for j, b in enumerate(rows):
            if i == j:
                continue
            bx, by = b[key_x], b[key_y]
            if (bx >= ax and by >= ay) and (bx > ax or by > ay):
                dominated = True
                break
        if not dominated:
            front.append(a)
    front.sort(key=lambda r: (r[key_x] + r[key_y], r[key_x], r[key_y]), reverse=True)
    return front


def main():
    parser = argparse.ArgumentParser(description="双向匹配自动化网格搜索")

    # 企业->政策网格（默认偏F1）
    parser.add_argument("--policy_adaptive_quantiles", type=str, default="0.72,0.75,0.78")
    parser.add_argument("--policy_relative_drop_thresholds", type=str, default="0.15")
    parser.add_argument("--policy_max_output_caps", type=str, default="100,120")

    # 政策->企业网格（默认偏Recall回补）
    parser.add_argument("--enterprise_adaptive_quantiles", type=str, default="0.58,0.60,0.62")
    parser.add_argument("--enterprise_relative_drop_thresholds", type=str, default="0.18,0.20")
    parser.add_argument("--enterprise_max_output_caps", type=str, default="120,150")
    parser.add_argument("--direct_support_boosts", type=str, default="0.30")

    parser.add_argument("--policy_candidate_k", type=int, default=1000)
    parser.add_argument("--enterprise_candidate_k", type=int, default=1000)
    parser.add_argument("--policy_score_threshold", type=float, default=-1.0)
    parser.add_argument("--enterprise_score_threshold", type=float, default=-1.0)
    parser.add_argument("--top_k_policy", type=int, default=-1)
    parser.add_argument("--top_k_enterprise", type=int, default=-1)

    # 综合目标：默认更关注双向F1
    parser.add_argument("--w_ep_f1", type=float, default=0.6, help="企业->政策 F1 权重")
    parser.add_argument("--w_pe_f1", type=float, default=0.4, help="政策->企业 F1 权重")

    parser.add_argument("--out_dir", type=str, default="matching/grid_search")
    parser.add_argument("--max_enterprise_queries", type=int, default=300)
    parser.add_argument("--max_industry_queries", type=int, default=30)
    parser.add_argument("--max_policy_queries", type=int, default=200)
    args = parser.parse_args()

    policy_score_threshold = None if args.policy_score_threshold < 0 else args.policy_score_threshold
    enterprise_score_threshold = None if args.enterprise_score_threshold < 0 else args.enterprise_score_threshold

    ep_qs = _parse_list(args.policy_adaptive_quantiles, float)
    ep_drops = _parse_list(args.policy_relative_drop_thresholds, float)
    ep_caps = _parse_list(args.policy_max_output_caps, int)

    pe_qs = _parse_list(args.enterprise_adaptive_quantiles, float)
    pe_drops = _parse_list(args.enterprise_relative_drop_thresholds, float)
    pe_caps = _parse_list(args.enterprise_max_output_caps, int)
    boosts = _parse_list(args.direct_support_boosts, float)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("双向匹配自动化网格搜索")
    print("=" * 60)
    print("初始化匹配器与查询集...")

    t0 = time.time()
    matcher = BidirectionalMatcher(PROJECT_ROOT)
    enterprise_queries, policy_queries = build_test_queries_from_data(
        max_enterprise_queries=args.max_enterprise_queries if args.max_enterprise_queries > 0 else None,
        max_industry_queries=args.max_industry_queries if args.max_industry_queries > 0 else None,
        max_policy_queries=args.max_policy_queries if args.max_policy_queries > 0 else None,
    )
    df_policies = pd.read_parquet(PROJECT_ROOT / "data_intermediate/policies_clean.parquet")
    valid_policy_ids = set(df_policies["policy_id"].astype(int).tolist())
    init_elapsed = time.time() - t0
    print(f"初始化完成，耗时 {init_elapsed:.2f}s")

    ep_cfgs: List[Tuple[Any, ...]] = []
    for ep_q in ep_qs:
        for ep_drop in ep_drops:
            for ep_cap in ep_caps:
                ep_cfgs.append((ep_q, ep_drop, ep_cap))

    pe_cfgs: List[Tuple[Any, ...]] = []
    for pe_q in pe_qs:
        for pe_drop in pe_drops:
            for pe_cap in pe_caps:
                for boost in boosts:
                    pe_cfgs.append((pe_q, pe_drop, pe_cap, boost))

    total_combos = len(ep_cfgs) * len(pe_cfgs)
    print(
        f"参数组合数: {total_combos} "
        f"(E->P配置={len(ep_cfgs)} × P->E配置={len(pe_cfgs)})"
    )

    # 第一阶段：E->P配置逐一评估并缓存
    ep_cache: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    ep_start = time.time()
    for i, (ep_q, ep_drop, ep_cap) in enumerate(ep_cfgs, start=1):
        with contextlib.redirect_stdout(io.StringIO()):
            ep_res = evaluate_enterprise_to_policy(
                matcher=matcher,
                queries=enterprise_queries,
                top_k=args.top_k_policy,
                candidate_k=args.policy_candidate_k,
                score_threshold=policy_score_threshold,
                adaptive_quantile=ep_q,
                relative_drop_threshold=ep_drop,
                max_output_cap=ep_cap,
                valid_policy_ids=valid_policy_ids,
            )
        ep_cache[(ep_q, ep_drop, ep_cap)] = ep_res["average"]
        if i % max(1, len(ep_cfgs) // 5) == 0 or i == len(ep_cfgs):
            elapsed = time.time() - ep_start
            eta = (elapsed / i) * (len(ep_cfgs) - i) if i > 0 else 0.0
            print(
                f"[E->P] 进度: {i}/{len(ep_cfgs)} ({100*i/len(ep_cfgs):.1f}%) | "
                f"elapsed={elapsed:.1f}s | eta={eta:.1f}s"
            )

    # 第二阶段：P->E配置逐一评估并缓存
    pe_cache: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    pe_start = time.time()
    for i, (pe_q, pe_drop, pe_cap, boost) in enumerate(pe_cfgs, start=1):
        with contextlib.redirect_stdout(io.StringIO()):
            pe_res = evaluate_policy_to_enterprise(
                matcher=matcher,
                queries=policy_queries,
                top_k=args.top_k_enterprise,
                candidate_k=args.enterprise_candidate_k,
                score_threshold=enterprise_score_threshold,
                adaptive_quantile=pe_q,
                relative_drop_threshold=pe_drop,
                max_output_cap=pe_cap,
                direct_support_boost=boost,
            )
        pe_cache[(pe_q, pe_drop, pe_cap, boost)] = pe_res["average"]
        if i % max(1, len(pe_cfgs) // 5) == 0 or i == len(pe_cfgs):
            elapsed = time.time() - pe_start
            eta = (elapsed / i) * (len(pe_cfgs) - i) if i > 0 else 0.0
            print(
                f"[P->E] 进度: {i}/{len(pe_cfgs)} ({100*i/len(pe_cfgs):.1f}%) | "
                f"elapsed={elapsed:.1f}s | eta={eta:.1f}s"
            )

    # 第三阶段：笛卡尔积组合目标（纯CPU拼表，无模型推理）
    rows: List[Dict[str, Any]] = []
    for ep_q, ep_drop, ep_cap in ep_cfgs:
        ep_avg = ep_cache[(ep_q, ep_drop, ep_cap)]
        for pe_q, pe_drop, pe_cap, boost in pe_cfgs:
            pe_avg = pe_cache[(pe_q, pe_drop, pe_cap, boost)]
            obj = args.w_ep_f1 * ep_avg["f1"] + args.w_pe_f1 * pe_avg["f1"]
            rows.append(
                {
                    "ep_q": ep_q,
                    "ep_drop": ep_drop,
                    "ep_cap": ep_cap,
                    "pe_q": pe_q,
                    "pe_drop": pe_drop,
                    "pe_cap": pe_cap,
                    "boost": boost,
                    "ep_precision": ep_avg["precision"],
                    "ep_recall": ep_avg["recall"],
                    "ep_f1": ep_avg["f1"],
                    "ep_map": ep_avg["map"],
                    "ep_ndcg": ep_avg["ndcg"],
                    "pe_precision": pe_avg["precision"],
                    "pe_recall": pe_avg["recall"],
                    "pe_f1": pe_avg["f1"],
                    "pe_map": pe_avg["map"],
                    "pe_ndcg": pe_avg["ndcg"],
                    "objective": obj,
                }
            )

    rows_sorted = sorted(rows, key=lambda r: r["objective"], reverse=True)
    pareto = _pareto_front(rows, "ep_f1", "pe_f1")

    best = rows_sorted[0]
    print("\n搜索完成")
    print(f"最佳组合 objective={best['objective']:.4f}")
    print(
        f"E->P: q={best['ep_q']}, drop={best['ep_drop']}, cap={best['ep_cap']} | "
        f"F1={best['ep_f1']:.4f}"
    )
    print(
        f"P->E: q={best['pe_q']}, drop={best['pe_drop']}, cap={best['pe_cap']}, boost={best['boost']} | "
        f"F1={best['pe_f1']:.4f}"
    )

    csv_path = out_dir / f"grid_search_results_{ts}.csv"
    json_path = out_dir / f"grid_search_results_{ts}.json"
    pareto_path = out_dir / f"grid_search_pareto_{ts}.json"

    pd.DataFrame(rows_sorted).to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": ts,
                "search_space": {
                    "ep_qs": ep_qs,
                    "ep_drops": ep_drops,
                    "ep_caps": ep_caps,
                    "pe_qs": pe_qs,
                    "pe_drops": pe_drops,
                    "pe_caps": pe_caps,
                    "boosts": boosts,
                },
                "weights": {"w_ep_f1": args.w_ep_f1, "w_pe_f1": args.w_pe_f1},
                "best": best,
                "top10": rows_sorted[:10],
                "results": rows_sorted,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(pareto_path, "w", encoding="utf-8") as f:
        json.dump({"pareto_front": pareto}, f, ensure_ascii=False, indent=2)

    print("\n输出文件:")
    print(f"- 全量CSV: {csv_path}")
    print(f"- 全量JSON: {json_path}")
    print(f"- Pareto前沿: {pareto_path}")


if __name__ == "__main__":
    main()

