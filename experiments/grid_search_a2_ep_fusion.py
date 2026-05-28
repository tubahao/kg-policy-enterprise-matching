#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2 主实验（joint BERT + gat_artifact_tag=a2_joint + 默认重要性 parquet）下，
对企业→政策（E→P）做小规模网格搜索：融合权重 × policy_adaptive_quantile。

- 单次初始化 BidirectionalMatcher，复用 evaluate_enterprise_to_policy。
- 默认 10 组 (semantic, structure, importance) × 5 个分位数 = 50 轮。
- 勿设置 KGE_POLICY_IMPORTANCE_PARQUET，以免偏离 A2 主 parquet。

用法:
  python scripts/grid_search_a2_ep_fusion.py
  python scripts/grid_search_a2_ep_fusion.py --out_dir matching/grid_search_a2_fusion
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from matching.torch_metadata_fix import apply_torch_metadata_fix  # noqa: E402

apply_torch_metadata_fix()

from matching.bidirectional_matching import BidirectionalMatcher  # noqa: E402
from matching.evaluate_matching import (  # noqa: E402
    build_test_queries_from_data,
    evaluate_enterprise_to_policy,
)
from matching.policy_embedding_defaults import JOINT_EMB, JOINT_IDX  # noqa: E402


# 与 evaluation_results_a2_joint_full_pipeline.json 对齐的固定项
A2_FIXED = {
    "top_k": -1,
    "candidate_k": 1000,
    "score_threshold": None,
    "relative_drop_threshold": 0.15,
    "max_output_cap": 120,
    "industry_boost": 0.12,
    "industry_query_adaptive_quantile": 0.82,
    "industry_query_relative_drop_threshold": 0.12,
    "industry_query_max_output_cap": 70,
}

# 粗网格：和为 1 的三元组（语义 / 结构 / 重要性）
DEFAULT_WEIGHT_TRIPLES: List[Tuple[float, float, float]] = [
    (0.45, 0.45, 0.10),
    (0.425, 0.425, 0.15),
    (0.40, 0.40, 0.20),
    (0.375, 0.375, 0.25),
    (0.35, 0.35, 0.30),
    (0.50, 0.30, 0.20),
    (0.30, 0.50, 0.20),
    (0.48, 0.32, 0.20),
    (0.32, 0.48, 0.20),
    (0.52, 0.33, 0.15),
]

DEFAULT_EP_QUANTILES = [0.66, 0.68, 0.72, 0.74, 0.76]


def _objective_ndcg_f1(ep_avg: Dict[str, float], w_ndcg: float, w_f1: float) -> float:
    return w_ndcg * float(ep_avg["ndcg"]) + w_f1 * float(ep_avg["f1"])


def main() -> None:
    parser = argparse.ArgumentParser(description="A2 E→P 融合权重 × quantile 网格（约50轮）")
    parser.add_argument("--out_dir", type=str, default="matching/grid_search_a2_fusion")
    parser.add_argument("--max_enterprise_queries", type=int, default=300)
    parser.add_argument("--max_industry_queries", type=int, default=30)
    parser.add_argument("--max_policy_queries", type=int, default=200)
    parser.add_argument("--w_ndcg", type=float, default=0.55, help="目标函数中 E→P NDCG 权重")
    parser.add_argument("--w_f1", type=float, default=0.45, help="目标函数中 E→P F1 权重")
    args = parser.parse_args()

    os.environ.pop("KGE_POLICY_IMPORTANCE_PARQUET", None)

    triples = DEFAULT_WEIGHT_TRIPLES
    ep_qs = DEFAULT_EP_QUANTILES
    n_runs = len(triples) * len(ep_qs)

    for ws, wt, wi in triples:
        s = ws + wt + wi
        if abs(s - 1.0) > 1e-6:
            raise SystemExit(f"权重和须为 1，收到 {ws}+{wt}+{wi}={s}")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("A2 主实验 E→P 网格搜索（joint + a2_joint）")
    print("=" * 60)
    print(f"组合数: {n_runs} = {len(triples)} 组权重 × {len(ep_qs)} 个 policy_adaptive_quantile")
    print(f"固定: rel_drop={A2_FIXED['relative_drop_threshold']}, cap={A2_FIXED['max_output_cap']}, "
          f"industry_boost={A2_FIXED['industry_boost']}")
    print()

    from matching.ensure_joint_policy_embeddings import ensure_joint_policy_embeddings

    ensure_joint_policy_embeddings(PROJECT_ROOT)

    t0 = time.time()
    matcher = BidirectionalMatcher(
        PROJECT_ROOT,
        policy_emb_path=JOINT_EMB,
        policy_index_path=JOINT_IDX,
        gat_artifact_tag="a2_joint",
    )
    enterprise_queries, _ = build_test_queries_from_data(
        max_enterprise_queries=args.max_enterprise_queries if args.max_enterprise_queries > 0 else None,
        max_industry_queries=args.max_industry_queries if args.max_industry_queries > 0 else None,
        max_policy_queries=args.max_policy_queries if args.max_policy_queries > 0 else None,
    )
    df_policies = pd.read_parquet(PROJECT_ROOT / "data_intermediate" / "policies_clean.parquet")
    valid_policy_ids = set(df_policies["policy_id"].astype(int).tolist())
    print(f"初始化 matcher + 查询集 耗时 {time.time() - t0:.1f}s，有效政策掩码 {len(valid_policy_ids)} 条\n")

    rows: List[Dict[str, Any]] = []
    run_i = 0
    t1 = time.time()
    for ws, wt, wi in triples:
        for ep_q in ep_qs:
            run_i += 1
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ep_res = evaluate_enterprise_to_policy(
                    matcher=matcher,
                    queries=enterprise_queries,
                    top_k=A2_FIXED["top_k"],
                    candidate_k=A2_FIXED["candidate_k"],
                    score_threshold=A2_FIXED["score_threshold"],
                    adaptive_quantile=ep_q,
                    relative_drop_threshold=A2_FIXED["relative_drop_threshold"],
                    max_output_cap=A2_FIXED["max_output_cap"],
                    semantic_weight=ws,
                    structure_weight=wt,
                    importance_weight=wi,
                    industry_boost=A2_FIXED["industry_boost"],
                    industry_query_adaptive_quantile=A2_FIXED["industry_query_adaptive_quantile"],
                    industry_query_relative_drop_threshold=A2_FIXED["industry_query_relative_drop_threshold"],
                    industry_query_max_output_cap=A2_FIXED["industry_query_max_output_cap"],
                    valid_policy_ids=valid_policy_ids,
                )
            avg = ep_res["average"]
            obj = _objective_ndcg_f1(avg, args.w_ndcg, args.w_f1)
            rows.append(
                {
                    "run": run_i,
                    "policy_semantic_weight": ws,
                    "policy_structure_weight": wt,
                    "policy_importance_weight": wi,
                    "policy_adaptive_quantile": ep_q,
                    "ep_precision": avg["precision"],
                    "ep_recall": avg["recall"],
                    "ep_f1": avg["f1"],
                    "ep_map": avg["map"],
                    "ep_ndcg": avg["ndcg"],
                    "objective": obj,
                }
            )
            if run_i % 10 == 0 or run_i == n_runs:
                elapsed = time.time() - t1
                eta = (elapsed / run_i) * (n_runs - run_i) if run_i else 0.0
                print(
                    f"[{run_i}/{n_runs}] "
                    f"w={ws:.3f}/{wt:.3f}/{wi:.3f} q={ep_q:.2f} -> "
                    f"NDCG={avg['ndcg']:.4f} F1={avg['f1']:.4f} obj={obj:.4f} | "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
                )

    rows_sorted = sorted(rows, key=lambda r: r["objective"], reverse=True)
    best = rows_sorted[0]
    best_ndcg = max(rows, key=lambda r: r["ep_ndcg"])

    csv_path = out_dir / f"a2_ep_fusion_grid_{ts}.csv"
    pd.DataFrame(rows_sorted).to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": ts,
        "matcher": "joint + gat_artifact_tag=a2_joint + policy_importance_with_decay_a2_joint.parquet",
        "n_runs": n_runs,
        "fixed": A2_FIXED,
        "weight_triples": [list(t) for t in triples],
        "policy_adaptive_quantiles": ep_qs,
        "objective": f"{args.w_ndcg}*ep_ndcg + {args.w_f1}*ep_f1",
        "best_by_objective": best,
        "best_by_ep_ndcg_only": best_ndcg,
        "top10_by_objective": rows_sorted[:10],
    }
    json_path = out_dir / f"a2_ep_fusion_grid_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("按目标函数最优:")
    print(
        f"  w_s/w_t/w_i={best['policy_semantic_weight']}/{best['policy_structure_weight']}/"
        f"{best['policy_importance_weight']}, q={best['policy_adaptive_quantile']}"
    )
    print(
        f"  E→P NDCG={best['ep_ndcg']:.4f} F1={best['ep_f1']:.4f} "
        f"P={best['ep_precision']:.4f} R={best['ep_recall']:.4f} MAP={best['ep_map']:.4f}"
    )
    print("纯 NDCG 最优:")
    print(
        f"  w_s/w_t/w_i={best_ndcg['policy_semantic_weight']}/{best_ndcg['policy_structure_weight']}/"
        f"{best_ndcg['policy_importance_weight']}, q={best_ndcg['policy_adaptive_quantile']}"
    )
    print(
        f"  E→P NDCG={best_ndcg['ep_ndcg']:.4f} F1={best_ndcg['ep_f1']:.4f}"
    )
    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
