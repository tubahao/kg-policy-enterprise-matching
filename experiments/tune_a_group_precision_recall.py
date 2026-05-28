#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A组模型下的定向调参：
1) 企业->政策：优先提升 Precision
2) 政策->企业：优先提升 Recall
"""

import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from matching.bidirectional_matching import BidirectionalMatcher
from matching.evaluate_matching import (  # noqa: E402
    build_test_queries_from_data,
    evaluate_enterprise_to_policy,
    evaluate_policy_to_enterprise,
)


def _score_ep(avg: Dict[str, float]) -> float:
    # 主目标：Precision；次目标：F1/NDCG 防止过度截断
    return (
        1.0 * float(avg["precision"])
        + 0.25 * float(avg["f1"])
        + 0.10 * float(avg["ndcg"])
    )


def _score_pe(avg: Dict[str, float]) -> float:
    # 主目标：Recall；次目标：F1 防止输出质量明显下滑
    return 1.0 * float(avg["recall"]) + 0.15 * float(avg["f1"])


def _run_ep(
    matcher: BidirectionalMatcher,
    enterprise_queries: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    valid_policy_ids,
) -> Dict[str, Any]:
    t0 = time.time()
    with redirect_stdout(io.StringIO()):
        out = evaluate_enterprise_to_policy(
            matcher=matcher,
            queries=enterprise_queries,
            top_k=-1,
            candidate_k=1000,
            score_threshold=None,
            adaptive_quantile=cfg["policy_adaptive_quantile"],
            relative_drop_threshold=cfg["policy_relative_drop_threshold"],
            max_output_cap=cfg["policy_max_output_cap"],
            semantic_weight=cfg["policy_semantic_weight"],
            structure_weight=cfg["policy_structure_weight"],
            importance_weight=cfg["policy_importance_weight"],
            industry_boost=cfg["policy_industry_boost"],
            industry_query_adaptive_quantile=cfg["policy_industry_query_adaptive_quantile"],
            industry_query_relative_drop_threshold=cfg["policy_industry_query_relative_drop_threshold"],
            industry_query_max_output_cap=cfg["policy_industry_query_max_output_cap"],
            valid_policy_ids=valid_policy_ids,
        )
    avg = out["average"]
    return {
        "cfg": cfg,
        "average": avg,
        "objective": _score_ep(avg),
        "elapsed_sec": time.time() - t0,
    }


def _run_pe(
    matcher: BidirectionalMatcher,
    policy_queries: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    t0 = time.time()
    with redirect_stdout(io.StringIO()):
        out = evaluate_policy_to_enterprise(
            matcher=matcher,
            queries=policy_queries,
            top_k=-1,
            candidate_k=1000,
            score_threshold=None,
            adaptive_quantile=cfg["enterprise_adaptive_quantile"],
            relative_drop_threshold=cfg["enterprise_relative_drop_threshold"],
            max_output_cap=cfg["enterprise_max_output_cap"],
            direct_support_boost=cfg["direct_support_boost"],
        )
    avg = out["average"]
    return {
        "cfg": cfg,
        "average": avg,
        "objective": _score_pe(avg),
        "elapsed_sec": time.time() - t0,
    }


def main() -> None:
    print("初始化匹配器与测试集...")
    matcher = BidirectionalMatcher(PROJECT_ROOT)
    enterprise_queries, policy_queries = build_test_queries_from_data(
        max_enterprise_queries=300,
        max_industry_queries=30,
        max_policy_queries=200,
    )
    valid_policy_ids = None
    try:
        df_pol = pd.read_parquet(PROJECT_ROOT / "data_intermediate/policies_clean.parquet")
        valid_policy_ids = set(df_pol["policy_id"].astype(int).tolist())
    except Exception:
        valid_policy_ids = None

    print(f"企业->政策查询数: {len(enterprise_queries)}")
    print(f"政策->企业查询数: {len(policy_queries)}")

    ep_cfgs: List[Dict[str, Any]] = [
        # baseline
        {
            "name": "ep_base",
            "policy_adaptive_quantile": 0.72,
            "policy_relative_drop_threshold": 0.15,
            "policy_max_output_cap": 120,
            "policy_semantic_weight": 0.60,
            "policy_structure_weight": 0.30,
            "policy_importance_weight": 0.10,
            "policy_industry_boost": 0.08,
            "policy_industry_query_adaptive_quantile": 0.82,
            "policy_industry_query_relative_drop_threshold": 0.12,
            "policy_industry_query_max_output_cap": 70,
        },
        {
            "name": "ep_p1",
            "policy_adaptive_quantile": 0.75,
            "policy_relative_drop_threshold": 0.12,
            "policy_max_output_cap": 100,
            "policy_semantic_weight": 0.60,
            "policy_structure_weight": 0.30,
            "policy_importance_weight": 0.10,
            "policy_industry_boost": 0.08,
            "policy_industry_query_adaptive_quantile": 0.85,
            "policy_industry_query_relative_drop_threshold": 0.10,
            "policy_industry_query_max_output_cap": 60,
        },
        {
            "name": "ep_p2",
            "policy_adaptive_quantile": 0.78,
            "policy_relative_drop_threshold": 0.12,
            "policy_max_output_cap": 90,
            "policy_semantic_weight": 0.60,
            "policy_structure_weight": 0.30,
            "policy_importance_weight": 0.10,
            "policy_industry_boost": 0.08,
            "policy_industry_query_adaptive_quantile": 0.88,
            "policy_industry_query_relative_drop_threshold": 0.10,
            "policy_industry_query_max_output_cap": 50,
        },
        {
            "name": "ep_p3",
            "policy_adaptive_quantile": 0.80,
            "policy_relative_drop_threshold": 0.12,
            "policy_max_output_cap": 80,
            "policy_semantic_weight": 0.60,
            "policy_structure_weight": 0.30,
            "policy_importance_weight": 0.10,
            "policy_industry_boost": 0.08,
            "policy_industry_query_adaptive_quantile": 0.90,
            "policy_industry_query_relative_drop_threshold": 0.08,
            "policy_industry_query_max_output_cap": 45,
        },
        # weight-focused
        {
            "name": "ep_w1",
            "policy_adaptive_quantile": 0.75,
            "policy_relative_drop_threshold": 0.12,
            "policy_max_output_cap": 100,
            "policy_semantic_weight": 0.70,
            "policy_structure_weight": 0.20,
            "policy_importance_weight": 0.10,
            "policy_industry_boost": 0.05,
            "policy_industry_query_adaptive_quantile": 0.85,
            "policy_industry_query_relative_drop_threshold": 0.10,
            "policy_industry_query_max_output_cap": 60,
        },
        {
            "name": "ep_w2",
            "policy_adaptive_quantile": 0.78,
            "policy_relative_drop_threshold": 0.12,
            "policy_max_output_cap": 90,
            "policy_semantic_weight": 0.72,
            "policy_structure_weight": 0.18,
            "policy_importance_weight": 0.10,
            "policy_industry_boost": 0.04,
            "policy_industry_query_adaptive_quantile": 0.88,
            "policy_industry_query_relative_drop_threshold": 0.10,
            "policy_industry_query_max_output_cap": 55,
        },
    ]

    pe_cfgs: List[Dict[str, Any]] = [
        {
            "name": "pe_base",
            "enterprise_adaptive_quantile": 0.58,
            "enterprise_relative_drop_threshold": 0.18,
            "enterprise_max_output_cap": 150,
            "direct_support_boost": 0.30,
        },
        {
            "name": "pe_r1",
            "enterprise_adaptive_quantile": 0.55,
            "enterprise_relative_drop_threshold": 0.22,
            "enterprise_max_output_cap": 200,
            "direct_support_boost": 0.25,
        },
        {
            "name": "pe_r2",
            "enterprise_adaptive_quantile": 0.52,
            "enterprise_relative_drop_threshold": 0.25,
            "enterprise_max_output_cap": 220,
            "direct_support_boost": 0.20,
        },
        {
            "name": "pe_r3",
            "enterprise_adaptive_quantile": 0.50,
            "enterprise_relative_drop_threshold": 0.28,
            "enterprise_max_output_cap": 260,
            "direct_support_boost": 0.20,
        },
        {
            "name": "pe_r4",
            "enterprise_adaptive_quantile": 0.48,
            "enterprise_relative_drop_threshold": 0.30,
            "enterprise_max_output_cap": 300,
            "direct_support_boost": 0.15,
        },
    ]

    ep_rows: List[Dict[str, Any]] = []
    print("\n搜索 E->P（目标: 提升 Precision）...")
    for i, cfg in enumerate(ep_cfgs, start=1):
        row = _run_ep(matcher, enterprise_queries, cfg, valid_policy_ids)
        ep_rows.append(row)
        avg = row["average"]
        print(
            f"[E {i}/{len(ep_cfgs)}] {cfg['name']} | "
            f"P={avg['precision']:.4f}, R={avg['recall']:.4f}, F1={avg['f1']:.4f}, "
            f"MAP={avg['map']:.4f}, NDCG={avg['ndcg']:.4f}, obj={row['objective']:.4f}, "
            f"time={row['elapsed_sec']:.1f}s"
        )

    ep_best = max(ep_rows, key=lambda x: x["objective"])
    print(f"\nE->P 最优: {ep_best['cfg']['name']} | obj={ep_best['objective']:.4f}")

    pe_rows: List[Dict[str, Any]] = []
    print("\n搜索 P->E（目标: 提升 Recall）...")
    for i, cfg in enumerate(pe_cfgs, start=1):
        row = _run_pe(matcher, policy_queries, cfg)
        pe_rows.append(row)
        avg = row["average"]
        print(
            f"[P {i}/{len(pe_cfgs)}] {cfg['name']} | "
            f"P={avg['precision']:.4f}, R={avg['recall']:.4f}, F1={avg['f1']:.4f}, "
            f"MAP={avg['map']:.4f}, NDCG={avg['ndcg']:.4f}, obj={row['objective']:.4f}, "
            f"time={row['elapsed_sec']:.1f}s"
        )

    pe_best = max(pe_rows, key=lambda x: x["objective"])
    print(f"\nP->E 最优: {pe_best['cfg']['name']} | obj={pe_best['objective']:.4f}")

    out = {
        "ep_results": ep_rows,
        "pe_results": pe_rows,
        "ep_best": ep_best,
        "pe_best": pe_best,
    }
    out_path = PROJECT_ROOT / "matching" / "tuning_a_precision_recall_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存搜索结果: {out_path}")


if __name__ == "__main__":
    main()

