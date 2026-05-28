#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2 主实验：企业→政策（E→P）NDCG 网格搜索。

- 基准超参与 `matching/evaluation_results_a2_joint_full_pipeline.json` 的 parameters 一致；
- 查询规模与 `run_a2_joint_full_pipeline.py` 一致：300 / 30 / 200；
- Matcher 只初始化一次，各组合仅重复 E→P 评估（stdout 重定向以降噪）。

默认网格：**以三通道权重锚点 0.5 / 0.3 / 0.2（语义/结构/重要性）** 为中心做局部扰动，再 × 分位 × 断崖。
目标 NDCG（如 0.38）仅作对照打印；**仅靠推理超参未必能到**，若差距大需改表征/GAT/训练或数据。

用法：
  python scripts/grid_search_a2_ep_ndcg.py              # 一阶段：0.5/0.3/0.2 锚点 ×210
  python scripts/grid_search_a2_ep_ndcg.py --refine    # 二阶段：围绕 ~0.54/0.26/0.2 与高 quantile，× cap×boost
  python scripts/grid_search_a2_ep_ndcg.py --target-ndcg 0.38
  python scripts/grid_search_a2_ep_ndcg.py --quick|--medium|--full
  python scripts/grid_search_a2_ep_ndcg.py --write-best
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from contextlib import redirect_stdout
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from matching.torch_metadata_fix import apply_torch_metadata_fix  # noqa: E402

apply_torch_metadata_fix()

from matching.bidirectional_matching import BidirectionalMatcher  # noqa: E402
from matching.evaluate_matching import (  # noqa: E402
    build_test_queries_from_data,
    evaluate_enterprise_to_policy,
    evaluate_policy_to_enterprise,
)
from matching.gat_importance_defaults import resolve_gat_importance_paths  # noqa: E402
from matching.policy_embedding_defaults import JOINT_EMB, JOINT_IDX  # noqa: E402
from matching.ensure_joint_policy_embeddings import ensure_joint_policy_embeddings  # noqa: E402


def _load_frozen_parameters() -> Dict[str, Any]:
    p = PROJECT_ROOT / "matching" / "evaluation_results_a2_joint_full_pipeline.json"
    if not p.is_file():
        raise FileNotFoundError(f"缺少冻结结果 JSON: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    return dict(obj.get("parameters") or {})


def _ep_kwargs_from_params(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "top_k": p.get("top_k_policy", -1),
        "candidate_k": p.get("policy_candidate_k", 1000),
        "score_threshold": p.get("policy_score_threshold"),
        "adaptive_quantile": p.get("policy_adaptive_quantile"),
        "relative_drop_threshold": p.get("policy_relative_drop_threshold"),
        "max_output_cap": p.get("policy_max_output_cap"),
        "semantic_weight": p.get("policy_semantic_weight"),
        "structure_weight": p.get("policy_structure_weight"),
        "importance_weight": p.get("policy_importance_weight"),
        "industry_boost": p.get("policy_industry_boost"),
        "industry_query_adaptive_quantile": p.get("policy_industry_query_adaptive_quantile"),
        "industry_query_relative_drop_threshold": p.get("policy_industry_query_relative_drop_threshold"),
        "industry_query_max_output_cap": p.get("policy_industry_query_max_output_cap"),
    }


def _pe_kwargs_from_params(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "top_k": p.get("top_k_enterprise", -1),
        "candidate_k": p.get("enterprise_candidate_k", 1000),
        "score_threshold": p.get("enterprise_score_threshold"),
        "adaptive_quantile": p.get("enterprise_adaptive_quantile"),
        "relative_drop_threshold": p.get("enterprise_relative_drop_threshold"),
        "max_output_cap": p.get("enterprise_max_output_cap"),
        "direct_support_boost": p.get("direct_support_boost", 0.3),
    }


def _weight_triples() -> List[Tuple[float, float, float]]:
    """(semantic, structure, importance)，和为 1。"""
    return [
        (0.45, 0.45, 0.10),
        (0.40, 0.40, 0.20),
        (0.42, 0.42, 0.16),
        (0.50, 0.40, 0.10),
        (0.40, 0.50, 0.10),
        (0.38, 0.42, 0.20),
        (0.44, 0.41, 0.15),
    ]


def _weight_triples_anchor_503020() -> List[Tuple[float, float, float]]:
    """以 (0.5, 0.3, 0.2) 为锚点的局部网格，和恒为 1。"""
    return [
        (0.50, 0.30, 0.20),
        (0.52, 0.28, 0.20),
        (0.48, 0.32, 0.20),
        (0.50, 0.28, 0.22),
        (0.50, 0.32, 0.18),
        (0.54, 0.26, 0.20),
        (0.46, 0.34, 0.20),
        (0.48, 0.30, 0.22),
        (0.52, 0.30, 0.18),
        (0.50, 0.25, 0.25),
        (0.47, 0.28, 0.25),
        (0.55, 0.30, 0.15),
        (0.45, 0.30, 0.25),
        (0.53, 0.27, 0.20),
    ]


def _grid_small() -> List[Dict[str, Any]]:
    """默认：锚点权重 × 分位 × 断崖。14×5×3=210 组。"""
    quantiles = [0.70, 0.73, 0.76, 0.78, 0.80]
    drops = [0.10, 0.14, 0.18]
    triples = _weight_triples_anchor_503020()
    rows: List[Dict[str, Any]] = []
    for q, d in product(quantiles, drops):
        for ws, wt, wi in triples:
            rows.append(
                {
                    "adaptive_quantile": q,
                    "relative_drop_threshold": d,
                    "semantic_weight": ws,
                    "structure_weight": wt,
                    "importance_weight": wi,
                }
            )
    return rows


def _grid_quick() -> List[Dict[str, Any]]:
    """键名与 evaluate_enterprise_to_policy 一致，便于 {**ep_base, **row}。"""
    rows = []
    for q, d, (ws, wt, wi) in product(
        [0.69, 0.72, 0.75],
        [0.12, 0.15, 0.18],
        [(0.45, 0.45, 0.10), (0.40, 0.40, 0.20)],
    ):
        rows.append(
            {
                "adaptive_quantile": q,
                "relative_drop_threshold": d,
                "semantic_weight": ws,
                "structure_weight": wt,
                "importance_weight": wi,
            }
        )
    return rows


def _grid_medium() -> List[Dict[str, Any]]:
    """默认：约 5×4×3×5=300 组（cap 固定为冻结基准 120）。"""
    quantiles = [0.66, 0.69, 0.72, 0.75, 0.78]
    drops = [0.10, 0.14, 0.18, 0.22]
    caps = [120]
    boosts = [0.10, 0.12, 0.15]
    triples = _weight_triples()[:5]
    rows: List[Dict[str, Any]] = []
    for q, d, cap, ib in product(quantiles, drops, caps, boosts):
        for ws, wt, wi in triples:
            rows.append(
                {
                    "adaptive_quantile": q,
                    "relative_drop_threshold": d,
                    "max_output_cap": cap,
                    "industry_boost": ib,
                    "semantic_weight": ws,
                    "structure_weight": wt,
                    "importance_weight": wi,
                }
            )
    return rows


def _weight_triples_refine_high_sem() -> List[Tuple[float, float, float]]:
    """
    二阶段：以首轮最优附近 (0.54, 0.26, 0.20) 为核，细调语义/结构/重要性（和为 1）。
    """
    return [
        (0.54, 0.26, 0.20),
        (0.55, 0.25, 0.20),
        (0.56, 0.24, 0.20),
        (0.53, 0.27, 0.20),
        (0.52, 0.28, 0.20),
        (0.57, 0.23, 0.20),
        (0.55, 0.28, 0.17),
        (0.54, 0.22, 0.24),
        (0.54, 0.30, 0.16),
        (0.58, 0.24, 0.18),
        (0.50, 0.28, 0.22),
        (0.56, 0.27, 0.17),
    ]


def _grid_refine() -> List[Dict[str, Any]]:
    """
    二阶段网格：5×2×2×2×12=480 组（在 ~0.54/0.26/0.2 附近细调权重，提高分位并扫 cap / industry_boost）。
    """
    quantiles = [0.77, 0.80, 0.83, 0.86, 0.88]
    drops = [0.08, 0.11]
    caps = [100, 130]
    boosts = [0.08, 0.14]
    triples = _weight_triples_refine_high_sem()
    rows: List[Dict[str, Any]] = []
    for q, d, cap, ib in product(quantiles, drops, caps, boosts):
        for ws, wt, wi in triples:
            rows.append(
                {
                    "adaptive_quantile": q,
                    "relative_drop_threshold": d,
                    "max_output_cap": cap,
                    "industry_boost": ib,
                    "semantic_weight": ws,
                    "structure_weight": wt,
                    "importance_weight": wi,
                }
            )
    return rows


def _grid_full() -> List[Dict[str, Any]]:
    """完整网格（数千组，耗时可数小时）。"""
    quantiles = [0.64, 0.67, 0.70, 0.72, 0.74, 0.76, 0.78]
    drops = [0.08, 0.11, 0.14, 0.18, 0.22]
    caps = [100, 120, 140]
    boosts = [0.08, 0.10, 0.12, 0.15]
    rows: List[Dict[str, Any]] = []
    for q, d, cap, ib in product(quantiles, drops, caps, boosts):
        for ws, wt, wi in _weight_triples():
            rows.append(
                {
                    "adaptive_quantile": q,
                    "relative_drop_threshold": d,
                    "max_output_cap": cap,
                    "industry_boost": ib,
                    "semantic_weight": ws,
                    "structure_weight": wt,
                    "importance_weight": wi,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="A2 E→P NDCG 网格搜索")
    parser.add_argument("--quick", action="store_true", help="约 18 组")
    parser.add_argument(
        "--medium",
        action="store_true",
        help="约 300 组（分位×断崖×行业 boost×多组权重）",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="完整网格（数千组，很慢）",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="二阶段：围绕 ~0.54/0.26/0.2 + 更高分位 + cap/industry_boost（约 480 组）",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="reports/grid_search_a2_ep_ndcg.csv",
        help="结果 CSV（相对项目根）",
    )
    parser.add_argument("--write-best", action="store_true", help="将最优全参 + 完整双向评测写入 reports/grid_search_a2_best.json")
    parser.add_argument("--max-runs", type=int, default=0, help=">0 时仅跑前 N 组（调试）")
    parser.add_argument(
        "--target-ndcg",
        type=float,
        default=0.38,
        help="目标 E→P NDCG（仅打印与最优的差距，用于对照）",
    )
    args = parser.parse_args()

    base_p = _load_frozen_parameters()
    baseline_ndcg = float(
        json.loads((PROJECT_ROOT / "matching" / "evaluation_results_a2_joint_full_pipeline.json").read_text(encoding="utf-8"))[
            "enterprise_to_policy"
        ]["average"]["ndcg"]
    )

    if args.quick:
        grid = _grid_quick()
    elif args.full:
        grid = _grid_full()
    elif args.medium:
        grid = _grid_medium()
    elif args.refine:
        grid = _grid_refine()
    else:
        grid = _grid_small()
    if args.max_runs and args.max_runs > 0:
        grid = grid[: args.max_runs]

    mode = (
        "quick"
        if args.quick
        else (
            "full"
            if args.full
            else ("medium" if args.medium else ("refine" if args.refine else "small"))
        )
    )
    print(
        f"[grid_search_a2] 模式={mode} 组合数={len(grid)} | 冻结基准 E→P NDCG={baseline_ndcg:.4f} | "
        f"目标 NDCG={args.target_ndcg:.4f}",
        flush=True,
    )

    ensure_joint_policy_embeddings(PROJECT_ROOT)
    gat_p, gat_c, imp_pq = resolve_gat_importance_paths(
        base_p.get("gat_artifact_tag") or "a2_joint",
        importance_parquet=None,
        ignore_env_importance_override=True,
    )
    t0 = time.time()
    matcher = BidirectionalMatcher(
        PROJECT_ROOT,
        policy_emb_path=JOINT_EMB,
        policy_index_path=JOINT_IDX,
        gat_artifact_tag=base_p.get("gat_artifact_tag") or "a2_joint",
        policy_importance_parquet=None,
        ignore_env_importance_override=True,
    )
    print(f"[grid_search_a2] Matcher 初始化 {time.time() - t0:.1f}s | GAT={gat_p}", flush=True)

    enterprise_queries, policy_queries = build_test_queries_from_data(
        max_enterprise_queries=300,
        max_industry_queries=30,
        max_policy_queries=200,
    )
    df_policies = pd.read_parquet(PROJECT_ROOT / "data_intermediate" / "policies_clean.parquet")
    valid_policy_ids = set(df_policies["policy_id"].astype(int).tolist())

    ep_base = _ep_kwargs_from_params(base_p)
    results: List[Dict[str, Any]] = []

    for i, overrides in enumerate(grid, 1):
        kw = {**ep_base, **overrides}

        buf = io.StringIO()
        t1 = time.time()
        with redirect_stdout(buf):
            er = evaluate_enterprise_to_policy(
                matcher,
                enterprise_queries,
                valid_policy_ids=valid_policy_ids,
                **kw,
            )
        dt = time.time() - t1
        av = er["average"]
        row = {
            "run": i,
            "ep_ndcg": float(av["ndcg"]),
            "ep_f1": float(av["f1"]),
            "ep_map": float(av["map"]),
            "ep_precision": float(av["precision"]),
            "ep_recall": float(av["recall"]),
            "seconds": round(dt, 2),
            "adaptive_quantile": kw.get("adaptive_quantile"),
            "relative_drop_threshold": kw.get("relative_drop_threshold"),
            "max_output_cap": kw.get("max_output_cap"),
            "industry_boost": kw.get("industry_boost"),
            "semantic_weight": kw.get("semantic_weight"),
            "structure_weight": kw.get("structure_weight"),
            "importance_weight": kw.get("importance_weight"),
        }
        results.append(row)
        if i == 1 or i % max(1, len(grid) // 10) == 0 or i == len(grid):
            print(
                f"  [{i}/{len(grid)}] NDCG={av['ndcg']:.4f} F1={av['f1']:.4f} "
                f"q={kw.get('adaptive_quantile')} drop={kw.get('relative_drop_threshold')} "
                f"w={kw.get('semantic_weight')}/{kw.get('structure_weight')}/{kw.get('importance_weight')}",
                flush=True,
            )

    df = pd.DataFrame(results)
    out_path = PROJECT_ROOT / args.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[grid_search_a2] CSV -> {out_path}", flush=True)

    df_sorted = df.sort_values("ep_ndcg", ascending=False)
    best = df_sorted.iloc[0].to_dict()
    print("\n[Top 5 E→P NDCG]", flush=True)
    for _, r in df_sorted.head(5).iterrows():
        print(
            f"  NDCG={r['ep_ndcg']:.4f} Δ={r['ep_ndcg'] - baseline_ndcg:+.4f} | "
            f"q={r.get('adaptive_quantile')} drop={r.get('relative_drop_threshold')} "
            f"cap={r.get('max_output_cap')} ib={r.get('industry_boost')} "
            f"w={r.get('semantic_weight')}/{r.get('structure_weight')}/{r.get('importance_weight')}",
            flush=True,
        )

    best_ndcg = float(df_sorted.iloc[0]["ep_ndcg"])
    gap_target = args.target_ndcg - best_ndcg
    if gap_target <= 0:
        print(
            f"\n[目标对照] 最优 E→P NDCG={best_ndcg:.4f} | 目标={args.target_ndcg:.4f} | 已达标（高出 {-gap_target:.4f}）",
            flush=True,
        )
    else:
        print(
            f"\n[目标对照] 最优 E→P NDCG={best_ndcg:.4f} | 目标={args.target_ndcg:.4f} | 距目标还差 {gap_target:.4f}",
            flush=True,
        )
    if gap_target > 0.02:
        print(
            "[提示] 差距仍较大时，单靠网格调 policy 阈值/权重通常不够；可试 --medium/--full 或改进 GAT/文本编码与训练。",
            flush=True,
        )

    if args.write_best:
        merged = dict(base_p)
        key_map = [
            ("adaptive_quantile", "policy_adaptive_quantile"),
            ("relative_drop_threshold", "policy_relative_drop_threshold"),
            ("max_output_cap", "policy_max_output_cap"),
            ("industry_boost", "policy_industry_boost"),
            ("semantic_weight", "policy_semantic_weight"),
            ("structure_weight", "policy_structure_weight"),
            ("importance_weight", "policy_importance_weight"),
        ]
        for eval_key, json_key in key_map:
            v = best.get(eval_key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                merged[json_key] = v

        ep_kw = _ep_kwargs_from_params(merged)
        pe_kw = _pe_kwargs_from_params(merged)
        buf = io.StringIO()
        with redirect_stdout(buf):
            ep_full = evaluate_enterprise_to_policy(
                matcher,
                enterprise_queries,
                valid_policy_ids=valid_policy_ids,
                **ep_kw,
            )
            pe_full = evaluate_policy_to_enterprise(matcher, policy_queries, **pe_kw)

        out_best = PROJECT_ROOT / "reports" / "grid_search_a2_best.json"
        payload = {
            "baseline_frozen_ep_ndcg": baseline_ndcg,
            "best_ep_ndcg": float(ep_full["average"]["ndcg"]),
            "improvement_ep_ndcg": float(ep_full["average"]["ndcg"]) - baseline_ndcg,
            "parameters": merged,
            "enterprise_to_policy": ep_full,
            "policy_to_enterprise": pe_full,
            "gat_policy_emb_path": str(gat_p),
            "gat_company_emb_path": str(gat_c),
            "policy_importance_parquet_path": str(imp_pq),
        }
        out_best.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[grid_search_a2] 最优完整评测 -> {out_best}", flush=True)


if __name__ == "__main__":
    main()
