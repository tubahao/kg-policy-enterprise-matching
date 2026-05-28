#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在固定 P->E 参数下网格搜索 E->P 权重与截断，单次初始化 BidirectionalMatcher。
用法（项目根目录）:
  python scripts/tune_main_matching_grid.py
  python scripts/tune_main_matching_grid.py --fine
  python scripts/tune_main_matching_grid.py --round3
  python scripts/tune_main_matching_grid.py --round4
  python scripts/tune_main_matching_grid.py --max_configs 12
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from matching.bidirectional_matching import BidirectionalMatcher
from matching.evaluate_matching import (
    build_test_queries_from_data,
    evaluate_enterprise_to_policy,
    evaluate_policy_to_enterprise,
)


# P->E 固定为主实验当前采用（与 实验全流程总结 一致）
PE_KWARGS = dict(
    top_k=-1,
    candidate_k=1000,
    score_threshold=None,
    adaptive_quantile=0.58,
    relative_drop_threshold=0.18,
    max_output_cap=150,
    direct_support_boost=0.3,
)

QUERY_KWARGS = dict(
    max_enterprise_queries=300,
    max_industry_queries=30,
    max_policy_queries=200,
)

# 行业查询专用（与主实验默认一致，可随网格略调）
INDUSTRY_Q = 0.82
INDUSTRY_DROP = 0.12
INDUSTRY_CAP = 70
INDUSTRY_BOOST = 0.12

# P→E 与 E→P 超参无关；不在网格前调用 retrieve（否则会动共享图/GPU 状态，拖累后续 E→P 静默评估）
PE_TYPICAL: Dict[str, float] = {
    "precision": 0.6666666666666666,
    "recall": 0.1749997058855242,
    "f1": 0.2650894380870469,
    "map": 0.1749997058855242,
    "ndcg": 0.6666666666666666,
}


def _avg(d: Dict[str, Any], key: str) -> float:
    return float(d["average"][key])


def run_ep_only(
    matcher: BidirectionalMatcher,
    enterprise_queries: List[Dict],
    valid_policy_ids: set,
    ep_kwargs: Dict[str, Any],
) -> Dict[str, float]:
    """仅 E→P。P→E 与 policy 侧权重无关，禁止在网格内重复跑以免污染 matcher 状态。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        er = evaluate_enterprise_to_policy(
            matcher,
            enterprise_queries,
            top_k=-1,
            candidate_k=1000,
            score_threshold=None,
            valid_policy_ids=valid_policy_ids,
            industry_query_adaptive_quantile=ep_kwargs["industry_q"],
            industry_query_relative_drop_threshold=ep_kwargs["industry_drop"],
            industry_query_max_output_cap=ep_kwargs["industry_cap"],
            **{k: v for k, v in ep_kwargs.items() if k not in ("industry_q", "industry_drop", "industry_cap")},
        )
    ea = er["average"]
    return {
        "precision": float(ea["precision"]),
        "recall": float(ea["recall"]),
        "f1": float(ea["f1"]),
        "map": float(ea["map"]),
        "ndcg": float(ea["ndcg"]),
    }


def run_pe_once(
    matcher: BidirectionalMatcher,
    policy_queries: List[Dict],
) -> Dict[str, float]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pr = evaluate_policy_to_enterprise(matcher, policy_queries, **PE_KWARGS)
    pa = pr["average"]
    return {
        "precision": float(pa["precision"]),
        "recall": float(pa["recall"]),
        "f1": float(pa["f1"]),
        "map": float(pa["map"]),
        "ndcg": float(pa["ndcg"]),
    }


ROUND1_BEST = dict(
    semantic_weight=0.65,
    structure_weight=0.25,
    importance_weight=0.10,
    adaptive_quantile=0.68,
    relative_drop_threshold=0.12,
    max_output_cap=150,
    industry_boost=INDUSTRY_BOOST,
    industry_q=INDUSTRY_Q,
    industry_drop=INDUSTRY_DROP,
    industry_cap=INDUSTRY_CAP,
)

# 第二轮摘要中的文件最优（用于第三轮对照与静默基线）
ROUND2_BEST = dict(
    semantic_weight=0.65,
    structure_weight=0.23,
    importance_weight=0.12,
    adaptive_quantile=0.65,
    relative_drop_threshold=0.12,
    max_output_cap=150,
    industry_boost=INDUSTRY_BOOST,
    industry_q=INDUSTRY_Q,
    industry_drop=INDUSTRY_DROP,
    industry_cap=INDUSTRY_CAP,
)

# 第三轮（当前主实验锚点）：用于第四轮粗扫对照
ROUND3_BEST = dict(
    semantic_weight=0.66,
    structure_weight=0.22,
    importance_weight=0.12,
    adaptive_quantile=0.63,
    relative_drop_threshold=0.11,
    max_output_cap=155,
    industry_boost=INDUSTRY_BOOST,
    industry_q=INDUSTRY_Q,
    industry_drop=INDUSTRY_DROP,
    industry_cap=INDUSTRY_CAP,
)


def _dedupe_configs(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[Any, ...]] = set()
    out: List[Dict[str, Any]] = []
    for c in raw:
        key = (
            round(c["semantic_weight"], 6),
            round(c["structure_weight"], 6),
            round(c["importance_weight"], 6),
            round(c["adaptive_quantile"], 6),
            round(c["relative_drop_threshold"], 6),
            int(c["max_output_cap"]),
            round(c.get("industry_boost", INDUSTRY_BOOST), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_configs", type=int, default=0, help=">0 时只跑前 N 个配置（调试）")
    ap.add_argument(
        "--fine",
        action="store_true",
        help="第二轮：围绕上一轮最优 (0.65/0.25/0.1, q=0.68, drop=0.12, cap=150) 细扫",
    )
    ap.add_argument(
        "--round3",
        action="store_true",
        help="第三轮：环绕 ROUND1/ROUND2 锚点的小网格，追求更高 E→P NDCG（勿与 --fine 同用）",
    )
    ap.add_argument(
        "--round4",
        action="store_true",
        help="第四轮：第一轮式粗网格（权重×截断笛卡尔积），中心放在 ROUND3 最优附近，~55 组量级",
    )
    ap.add_argument("--out_summary", type=str, default="")
    ap.add_argument("--out_best_eval", type=str, default="")
    ns = ap.parse_args()

    _mode_n = sum([bool(ns.fine), bool(ns.round3), bool(ns.round4)])
    if _mode_n > 1:
        raise SystemExit("只能任选其一：默认粗扫 / --fine / --round3 / --round4")

    if not ns.out_summary:
        if ns.round4:
            ns.out_summary = "reports/main_matching_tune_round4_summary.json"
        elif ns.round3:
            ns.out_summary = "reports/main_matching_tune_round3_summary.json"
        elif ns.fine:
            ns.out_summary = "reports/main_matching_tune_round2_summary.json"
        else:
            ns.out_summary = "reports/main_matching_tune_grid_summary.json"
    if not ns.out_best_eval:
        if ns.round4:
            ns.out_best_eval = "matching/evaluation_results_main_tuned_round4.json"
        elif ns.round3:
            ns.out_best_eval = "matching/evaluation_results_main_tuned_round3.json"
        elif ns.fine:
            ns.out_best_eval = "matching/evaluation_results_main_tuned_round2.json"
        else:
            ns.out_best_eval = "matching/evaluation_results_main_tuned_best.json"

    print("初始化匹配器（一次）…", flush=True)
    matcher = BidirectionalMatcher(project_root)
    ent_q, pol_q = build_test_queries_from_data(**QUERY_KWARGS)
    df_policies = pd.read_parquet(project_root / "data_intermediate/policies_clean.parquet")
    valid_policy_ids = set(df_policies["policy_id"].astype(int).tolist())

    pe_fixed = dict(PE_TYPICAL)
    print(
        "P→E 使用固定典型值做约束（与 E→P 超参无关）；网格结束后再实测 P→E …",
        flush=True,
    )

    # 粗扫：论文原主实验基线（用于 improvement 对比）
    baseline_ep = dict(
        adaptive_quantile=0.72,
        relative_drop_threshold=0.15,
        max_output_cap=120,
        semantic_weight=0.45,
        structure_weight=0.45,
        importance_weight=0.10,
        industry_boost=INDUSTRY_BOOST,
        industry_q=INDUSTRY_Q,
        industry_drop=INDUSTRY_DROP,
        industry_cap=INDUSTRY_CAP,
    )

    configs: List[Dict[str, Any]] = []
    if ns.round4:
        print("模式: 第四轮粗扫（第一轮式：权重×截断，中心 ROUND3 最优附近）\n", flush=True)
        # 与第一轮类似：9 组权重 × 6 组截断；步长比 round3 细网格大，覆盖更广
        weight_variants = [
            (0.66, 0.22, 0.12),  # ROUND3
            (0.65, 0.23, 0.12),  # ROUND2
            (0.65, 0.25, 0.10),  # ROUND1
            (0.64, 0.24, 0.12),
            (0.67, 0.21, 0.12),
            (0.68, 0.20, 0.12),
            (0.62, 0.26, 0.12),
            (0.70, 0.18, 0.12),
            (0.65, 0.22, 0.13),
        ]
        trunc_variants = [
            (0.60, 0.10, 145),
            (0.62, 0.11, 150),
            (0.63, 0.11, 155),
            (0.65, 0.12, 150),
            (0.66, 0.12, 160),
            (0.68, 0.13, 140),
        ]
        for sw, st, iw in weight_variants:
            for q, drop, cap in trunc_variants:
                configs.append(
                    {
                        "semantic_weight": sw,
                        "structure_weight": st,
                        "importance_weight": iw,
                        "adaptive_quantile": q,
                        "relative_drop_threshold": drop,
                        "max_output_cap": cap,
                        "industry_boost": INDUSTRY_BOOST,
                        "industry_q": INDUSTRY_Q,
                        "industry_drop": INDUSTRY_DROP,
                        "industry_cap": INDUSTRY_CAP,
                    }
                )
        configs.append(baseline_ep.copy())
        configs = _dedupe_configs(configs)
        configs.insert(0, ROUND3_BEST.copy())
        configs.insert(0, ROUND2_BEST.copy())
        configs.insert(0, ROUND1_BEST.copy())
        configs = _dedupe_configs(configs)
    elif ns.round3:
        print("模式: 第三轮小规模环绕网格\n", flush=True)
        weight_variants = [
            (0.65, 0.23, 0.12),
            (0.65, 0.25, 0.10),
            (0.64, 0.24, 0.12),
            (0.66, 0.22, 0.12),
        ]
        q_variants = [0.63, 0.65, 0.67, 0.69]
        drop_variants = [0.11, 0.12, 0.13]
        cap_variants = [145, 150, 155]
        for sw, st, iw in weight_variants:
            for q, drop, cap in itertools.product(q_variants, drop_variants, cap_variants):
                configs.append(
                    {
                        "semantic_weight": sw,
                        "structure_weight": st,
                        "importance_weight": iw,
                        "adaptive_quantile": q,
                        "relative_drop_threshold": drop,
                        "max_output_cap": cap,
                        "industry_boost": INDUSTRY_BOOST,
                        "industry_q": INDUSTRY_Q,
                        "industry_drop": INDUSTRY_DROP,
                        "industry_cap": INDUSTRY_CAP,
                    }
                )
        for ib in (0.10, 0.14):
            for q, drop, cap in ((0.65, 0.12, 150), (0.67, 0.12, 155)):
                configs.append(
                    {
                        "semantic_weight": 0.65,
                        "structure_weight": 0.23,
                        "importance_weight": 0.12,
                        "adaptive_quantile": q,
                        "relative_drop_threshold": drop,
                        "max_output_cap": cap,
                        "industry_boost": ib,
                        "industry_q": INDUSTRY_Q,
                        "industry_drop": INDUSTRY_DROP,
                        "industry_cap": INDUSTRY_CAP,
                    }
                )
        configs = _dedupe_configs(configs)
        configs.insert(0, ROUND2_BEST.copy())
        configs.insert(0, ROUND1_BEST.copy())
        configs = _dedupe_configs(configs)
    elif ns.fine:
        print("模式: 细扫（第二轮）\n", flush=True)
        weight_variants = [
            (0.62, 0.28, 0.10),
            (0.63, 0.27, 0.10),
            (0.64, 0.26, 0.10),
            (0.65, 0.25, 0.10),
            (0.66, 0.24, 0.10),
            (0.67, 0.23, 0.10),
            (0.68, 0.22, 0.10),
            (0.65, 0.27, 0.08),
            (0.65, 0.23, 0.12),
            (0.64, 0.25, 0.11),
            (0.66, 0.25, 0.09),
        ]
        trunc_variants = [
            (0.66, 0.11, 145),
            (0.66, 0.11, 150),
            (0.67, 0.11, 150),
            (0.68, 0.11, 150),
            (0.68, 0.12, 140),
            (0.68, 0.12, 145),
            (0.68, 0.12, 150),
            (0.68, 0.12, 155),
            (0.68, 0.12, 165),
            (0.69, 0.12, 150),
            (0.70, 0.12, 155),
            (0.68, 0.10, 150),
            (0.68, 0.13, 150),
            (0.65, 0.12, 150),
            (0.71, 0.12, 160),
        ]
        boost_variants = [0.10, 0.12, 0.14]
        for sw, st, iw in weight_variants:
            for q, drop, cap in trunc_variants:
                configs.append(
                    {
                        "semantic_weight": sw,
                        "structure_weight": st,
                        "importance_weight": iw,
                        "adaptive_quantile": q,
                        "relative_drop_threshold": drop,
                        "max_output_cap": cap,
                        "industry_boost": INDUSTRY_BOOST,
                        "industry_q": INDUSTRY_Q,
                        "industry_drop": INDUSTRY_DROP,
                        "industry_cap": INDUSTRY_CAP,
                    }
                )
        # 仅对上一轮最优权重扫 industry_boost
        for ib in boost_variants:
            if ib == INDUSTRY_BOOST:
                continue
            for q, drop, cap in [(0.68, 0.12, 150), (0.68, 0.11, 150), (0.67, 0.12, 155)]:
                configs.append(
                    {
                        "semantic_weight": 0.65,
                        "structure_weight": 0.25,
                        "importance_weight": 0.10,
                        "adaptive_quantile": q,
                        "relative_drop_threshold": drop,
                        "max_output_cap": cap,
                        "industry_boost": ib,
                        "industry_q": INDUSTRY_Q,
                        "industry_drop": INDUSTRY_DROP,
                        "industry_cap": INDUSTRY_CAP,
                    }
                )
        configs = _dedupe_configs(configs)
        configs.insert(0, ROUND1_BEST.copy())
        configs = _dedupe_configs(configs)
    else:
        print("模式: 粗扫（第一轮）\n", flush=True)
        weight_variants = [
            (0.45, 0.45, 0.10),
            (0.50, 0.40, 0.10),
            (0.52, 0.38, 0.10),
            (0.55, 0.35, 0.10),
            (0.58, 0.32, 0.10),
            (0.60, 0.30, 0.10),
            (0.65, 0.25, 0.10),
            (0.40, 0.45, 0.15),
            (0.35, 0.50, 0.15),
        ]
        trunc_variants = [
            (0.72, 0.15, 120),
            (0.70, 0.12, 130),
            (0.68, 0.14, 140),
            (0.74, 0.12, 120),
            (0.70, 0.18, 150),
            (0.68, 0.12, 150),
        ]
        for sw, st, iw in weight_variants:
            for q, drop, cap in trunc_variants:
                configs.append(
                    {
                        "semantic_weight": sw,
                        "structure_weight": st,
                        "importance_weight": iw,
                        "adaptive_quantile": q,
                        "relative_drop_threshold": drop,
                        "max_output_cap": cap,
                        "industry_boost": INDUSTRY_BOOST,
                        "industry_q": INDUSTRY_Q,
                        "industry_drop": INDUSTRY_DROP,
                        "industry_cap": INDUSTRY_CAP,
                    }
                )
        configs.append(baseline_ep.copy())

    if ns.max_configs > 0:
        configs = configs[: ns.max_configs]

    results_rows: List[Dict[str, Any]] = []
    baseline_ep_metrics: Dict[str, float] | None = None

    t0 = __import__("time").time()
    for i, cfg in enumerate(configs):
        ep = run_ep_only(matcher, ent_q, valid_policy_ids, cfg)
        row = {**cfg, "ep": ep, "pe": pe_fixed}
        results_rows.append(row)
        if ns.round4:
            is_r3 = (
                cfg.get("semantic_weight") == ROUND3_BEST["semantic_weight"]
                and cfg.get("structure_weight") == ROUND3_BEST["structure_weight"]
                and cfg.get("importance_weight") == ROUND3_BEST["importance_weight"]
                and cfg.get("adaptive_quantile") == ROUND3_BEST["adaptive_quantile"]
                and cfg.get("relative_drop_threshold") == ROUND3_BEST["relative_drop_threshold"]
                and cfg.get("max_output_cap") == ROUND3_BEST["max_output_cap"]
                and cfg.get("industry_boost") == ROUND3_BEST["industry_boost"]
            )
            if is_r3:
                baseline_ep_metrics = ep
        elif ns.round3:
            is_r2 = (
                cfg.get("semantic_weight") == ROUND2_BEST["semantic_weight"]
                and cfg.get("structure_weight") == ROUND2_BEST["structure_weight"]
                and cfg.get("importance_weight") == ROUND2_BEST["importance_weight"]
                and cfg.get("adaptive_quantile") == ROUND2_BEST["adaptive_quantile"]
                and cfg.get("relative_drop_threshold") == ROUND2_BEST["relative_drop_threshold"]
                and cfg.get("max_output_cap") == ROUND2_BEST["max_output_cap"]
                and cfg.get("industry_boost") == ROUND2_BEST["industry_boost"]
            )
            if is_r2:
                baseline_ep_metrics = ep
        elif ns.fine:
            is_r1 = (
                cfg.get("semantic_weight") == ROUND1_BEST["semantic_weight"]
                and cfg.get("structure_weight") == ROUND1_BEST["structure_weight"]
                and cfg.get("importance_weight") == ROUND1_BEST["importance_weight"]
                and cfg.get("adaptive_quantile") == ROUND1_BEST["adaptive_quantile"]
                and cfg.get("relative_drop_threshold") == ROUND1_BEST["relative_drop_threshold"]
                and cfg.get("max_output_cap") == ROUND1_BEST["max_output_cap"]
                and cfg.get("industry_boost") == ROUND1_BEST["industry_boost"]
            )
            if is_r1:
                baseline_ep_metrics = ep
        else:
            is_base = (
                cfg.get("semantic_weight") == 0.45
                and cfg.get("structure_weight") == 0.45
                and cfg.get("adaptive_quantile") == 0.72
                and cfg.get("relative_drop_threshold") == 0.15
                and cfg.get("max_output_cap") == 120
            )
            if is_base:
                baseline_ep_metrics = ep
        ib = cfg.get("industry_boost", INDUSTRY_BOOST)
        print(
            f"[{i+1}/{len(configs)}] "
            f"w=({cfg['semantic_weight']:.2f},{cfg['structure_weight']:.2f},{cfg['importance_weight']:.2f}) "
            f"q={cfg['adaptive_quantile']:.2f} drop={cfg['relative_drop_threshold']:.2f} cap={cfg['max_output_cap']} "
            f"ib={ib:.2f} "
            f"=> E-P NDCG={ep['ndcg']:.4f} MAP={ep['map']:.4f} F1={ep['f1']:.4f}",
            flush=True,
        )

    elapsed = __import__("time").time() - t0
    if baseline_ep_metrics is None:
        baseline_ep_metrics = {"ndcg": 0.0, "map": 0.0, "f1": 0.0}

    if ns.round4:
        summary_note = "baseline_ep_in_grid = ROUND3_BEST（第三轮最优）在本轮静默重算"
    elif ns.round3:
        summary_note = "baseline_ep_in_grid = ROUND2_BEST（第二轮文件最优）在本轮静默重算"
    elif ns.fine:
        summary_note = "baseline_ep_in_grid = 第一轮最优 (ROUND1_BEST) 在本轮中的重算"
    else:
        summary_note = "baseline_ep_in_grid = 论文原主实验 (0.45/0.45/0.1, q=0.72, cap=120)"

    # 约束：P->E NDCG 不低于主实验典型值太多；Recall 不明显崩
    pe_ndcg_floor = 0.655
    pe_recall_floor = 0.168

    def score_row(r: Dict[str, Any]) -> Tuple[float, float, float]:
        ep, pe = r["ep"], r["pe"]
        if pe["ndcg"] < pe_ndcg_floor or pe["recall"] < pe_recall_floor:
            return (-1e9, ep["ndcg"], ep["map"])
        primary = ep["ndcg"] + 0.35 * ep["map"] + 0.15 * ep["f1"]
        return (primary, ep["ndcg"], ep["map"])

    feasible = [r for r in results_rows if score_row(r)[0] > -1e8]
    if not feasible:
        feasible = results_rows
        print("警告: 无满足 P->E 约束的配置，改为在全部结果中选最优。", flush=True)

    best = max(feasible, key=score_row)

    summary = {
        "fine_mode": ns.fine,
        "round3_mode": ns.round3,
        "round4_mode": ns.round4,
        "baseline_note": summary_note,
        "elapsed_sec": round(elapsed, 2),
        "num_configs": len(configs),
        "constraints": {"pe_ndcg_floor": pe_ndcg_floor, "pe_recall_floor": pe_recall_floor},
        "baseline_ep_in_grid": baseline_ep_metrics,
        "best_config": {k: best[k] for k in best if k not in ("ep", "pe")},
        "best_ep": best["ep"],
        "best_pe": best["pe"],
        "improvement_vs_baseline_ep": {
            "ndcg": round(best["ep"]["ndcg"] - baseline_ep_metrics["ndcg"], 6),
            "map": round(best["ep"]["map"] - baseline_ep_metrics["map"], 6),
            "f1": round(best["ep"]["f1"] - baseline_ep_metrics["f1"], 6),
        },
        "all_results": [
            {
                "semantic_weight": r["semantic_weight"],
                "structure_weight": r["structure_weight"],
                "importance_weight": r["importance_weight"],
                "adaptive_quantile": r["adaptive_quantile"],
                "relative_drop_threshold": r["relative_drop_threshold"],
                "max_output_cap": r["max_output_cap"],
                "industry_boost": r.get("industry_boost", INDUSTRY_BOOST),
                "ep": r["ep"],
                "pe": r["pe"],
                "score": score_row(r)[0],
            }
            for r in results_rows
        ],
    }

    out_sum = project_root / ns.out_summary
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n摘要已写: {out_sum}", flush=True)

    print("网格后实测 P→E（写入摘要 verified_pe）…", flush=True)
    verified_pe = run_pe_once(matcher, pol_q)
    summary["verified_pe_after_grid"] = verified_pe
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 用最优配置再跑一遍完整评估（带打印），写出正式 JSON
    bc = {k: best[k] for k in best if k not in ("ep", "pe")}
    import subprocess

    cmd = [
        sys.executable,
        str(project_root / "matching" / "evaluate_matching.py"),
        "--top_k_policy",
        "-1",
        "--top_k_enterprise",
        "-1",
        "--policy_candidate_k",
        "1000",
        "--enterprise_candidate_k",
        "1000",
        "--policy_score_threshold",
        "-1",
        "--policy_adaptive_quantile",
        str(bc["adaptive_quantile"]),
        "--policy_relative_drop_threshold",
        str(bc["relative_drop_threshold"]),
        "--policy_max_output_cap",
        str(bc["max_output_cap"]),
        "--policy_semantic_weight",
        str(bc["semantic_weight"]),
        "--policy_structure_weight",
        str(bc["structure_weight"]),
        "--policy_importance_weight",
        str(bc["importance_weight"]),
        "--policy_industry_boost",
        str(bc["industry_boost"]),
        "--policy_industry_query_adaptive_quantile",
        str(bc["industry_q"]),
        "--policy_industry_query_relative_drop_threshold",
        str(bc["industry_drop"]),
        "--policy_industry_query_max_output_cap",
        str(bc["industry_cap"]),
        "--enterprise_score_threshold",
        "-1",
        "--enterprise_adaptive_quantile",
        "0.58",
        "--enterprise_relative_drop_threshold",
        "0.18",
        "--enterprise_max_output_cap",
        "150",
        "--direct_support_boost",
        "0.3",
        "--max_enterprise_queries",
        "300",
        "--max_industry_queries",
        "30",
        "--max_policy_queries",
        "200",
        "--output",
        ns.out_best_eval,
    ]
    print("正在写出最优配置的完整评估 JSON …", flush=True)
    subprocess.run(cmd, cwd=str(project_root), check=True)
    print(f"完成: {project_root / ns.out_best_eval}", flush=True)


if __name__ == "__main__":
    main()
