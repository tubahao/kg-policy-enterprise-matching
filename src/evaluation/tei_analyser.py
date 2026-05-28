#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政策传导效能评估：
1) 传导广度：企业覆盖率 Coverage
2) 传导深度：Hop 深度 + PPR 能量衰减深度
3) 综合指标：TEI
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from matching.bidirectional_matching import BidirectionalMatcher  # noqa: E402
from matching.evaluate_matching import build_test_queries_from_data  # noqa: E402


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_global_graph_structures(matcher: BidirectionalMatcher):
    g = matcher.enterprise_retriever.graph
    offsets: Dict[str, int] = {}
    cur = 0
    for ntype in g.ntypes:
        offsets[ntype] = cur
        cur += g.number_of_nodes(ntype)
    total_nodes = cur

    # 稀疏转移矩阵（列随机）：p_{t+1} = alpha*e + (1-alpha)*P^T*p_t
    rows = []
    cols = []
    vals = []
    adjacency: List[List[int]] = [[] for _ in range(total_nodes)]

    out_deg = np.zeros(total_nodes, dtype=np.int64)

    for stype, _, dtype in g.canonical_etypes:
        src, dst = g.edges(etype=(stype, _, dtype))
        src_np = src.numpy()
        dst_np = dst.numpy()
        s_off = offsets[stype]
        d_off = offsets[dtype]
        g_src = src_np + s_off
        g_dst = dst_np + d_off
        for s, d in zip(g_src.tolist(), g_dst.tolist()):
            adjacency[s].append(d)
            out_deg[s] += 1

    # 构建列随机转移矩阵 P^T
    for s in range(total_nodes):
        deg = int(out_deg[s])
        if deg > 0:
            w = 1.0 / deg
            for d in adjacency[s]:
                rows.append(d)  # P^T 的行是目标节点
                cols.append(s)  # 列是源节点
                vals.append(w)
        else:
            # dangling 节点：加自环
            rows.append(s)
            cols.append(s)
            vals.append(1.0)

    pt = sparse.csr_matrix((vals, (rows, cols)), shape=(total_nodes, total_nodes), dtype=np.float64)
    return g, offsets, adjacency, pt


def bfs_shortest_dist(adjacency: List[List[int]], source: int) -> np.ndarray:
    n = len(adjacency)
    dist = np.full(n, -1, dtype=np.int32)
    q = deque([source])
    dist[source] = 0
    while q:
        u = q.popleft()
        du = int(dist[u])
        for v in adjacency[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                q.append(v)
    return dist


def personalized_pagerank_vector(
    pt: sparse.csr_matrix,
    seed: int,
    alpha: float = 0.15,
    max_iter: int = 40,
    tol: float = 1e-8,
) -> np.ndarray:
    n = pt.shape[0]
    e = np.zeros(n, dtype=np.float64)
    e[seed] = 1.0
    p = e.copy()
    for _ in range(max_iter):
        nxt = alpha * e + (1.0 - alpha) * (pt @ p)
        if np.linalg.norm(nxt - p, ord=1) < tol:
            p = nxt
            break
        p = nxt
    return p


def main():
    import argparse

    parser = argparse.ArgumentParser(description="政策传导效能评估")
    parser.add_argument("--top_k_enterprise", type=int, default=-1)
    parser.add_argument("--enterprise_candidate_k", type=int, default=1000)
    parser.add_argument("--enterprise_score_threshold", type=float, default=-1.0)
    parser.add_argument("--enterprise_adaptive_quantile", type=float, default=0.56)
    parser.add_argument("--enterprise_relative_drop_threshold", type=float, default=0.16)
    parser.add_argument("--enterprise_max_output_cap", type=int, default=150)
    parser.add_argument("--direct_support_boost", type=float, default=0.25)
    parser.add_argument("--max_enterprise_queries", type=int, default=300)
    parser.add_argument("--max_industry_queries", type=int, default=30)
    parser.add_argument("--max_policy_queries", type=int, default=200)
    parser.add_argument("--ppr_alpha", type=float, default=0.15)
    parser.add_argument("--ppr_max_iter", type=int, default=40)
    parser.add_argument("--tei_alpha", type=float, default=0.4)
    parser.add_argument("--tei_beta", type=float, default=0.3)
    parser.add_argument("--tei_gamma", type=float, default=0.3)
    parser.add_argument("--output_json", type=str, default="evaluation/transmission_efficiency.json")
    parser.add_argument("--output_csv", type=str, default="evaluation/transmission_efficiency.csv")
    args = parser.parse_args()

    enterprise_score_threshold = None if args.enterprise_score_threshold < 0 else args.enterprise_score_threshold

    print("=" * 60)
    print("政策传导效能评估")
    print("=" * 60)
    print("初始化匹配器与测试查询...")

    t0 = time.time()
    matcher = BidirectionalMatcher(project_root)
    _, policy_queries = build_test_queries_from_data(
        max_enterprise_queries=args.max_enterprise_queries if args.max_enterprise_queries > 0 else None,
        max_industry_queries=args.max_industry_queries if args.max_industry_queries > 0 else None,
        max_policy_queries=args.max_policy_queries if args.max_policy_queries > 0 else None,
    )
    print(f"政策查询数: {len(policy_queries)}")
    print(f"初始化耗时: {_format_seconds(time.time() - t0)}")

    g, offsets, adjacency, pt = build_global_graph_structures(matcher)
    company_count = g.number_of_nodes("company")
    policy_off = offsets["policy"]
    company_off = offsets["company"]

    all_rows = []
    total = len(policy_queries)
    progress_interval = max(1, total // 10)
    start = time.time()

    for i, q in enumerate(policy_queries, start=1):
        policy_id = int(q["policy_id"])
        policy_title = str(q["policy_title"])

        policy_node = matcher.enterprise_retriever._resolve_policy_node_id(policy_id)
        if policy_node is None:
            continue

        results = matcher.retrieve_enterprises_by_policy(
            policy_id=policy_id,
            top_k=args.top_k_enterprise,
            score_threshold=enterprise_score_threshold,
            candidate_k=args.enterprise_candidate_k,
            adaptive_quantile=args.enterprise_adaptive_quantile,
            relative_drop_threshold=args.enterprise_relative_drop_threshold,
            max_output_cap=args.enterprise_max_output_cap,
            direct_support_boost=args.direct_support_boost,
        )

        covered = {int(cid) for cid, _ in results}
        direct_set = matcher.enterprise_retriever.policy_direct_support_companies.get(int(policy_node), set())
        direct_set = {int(x) for x in direct_set}
        indirect_set = covered - direct_set
        covered_union = covered | direct_set

        coverage = (len(covered_union) / company_count) if company_count > 0 else 0.0

        # Hop深度
        src_gid = policy_off + int(policy_node)
        dist = bfs_shortest_dist(adjacency, src_gid)
        covered_gids = [company_off + cid for cid in covered_union if 0 <= cid < company_count]
        hop_vals = [int(dist[gidx]) for gidx in covered_gids if dist[gidx] >= 0]
        depth_hops = float(np.mean(hop_vals)) if hop_vals else 0.0

        # PPR能量深度
        ppr = personalized_pagerank_vector(
            pt=pt,
            seed=src_gid,
            alpha=args.ppr_alpha,
            max_iter=args.ppr_max_iter,
            tol=1e-8,
        )
        if covered_gids:
            ppr_vals = np.array([float(ppr[gidx]) for gidx in covered_gids], dtype=np.float64)
            depth_energy = float(1.0 - np.mean(ppr_vals))
        else:
            depth_energy = 1.0

        inv_depth = (1.0 / depth_hops) if depth_hops > 0 else 0.0
        tei = (
            args.tei_alpha * coverage
            + args.tei_beta * inv_depth
            + args.tei_gamma * depth_energy
        )

        all_rows.append(
            {
                "policy_id": policy_id,
                "policy_title": policy_title,
                "direct_count": len(direct_set),
                "indirect_count": len(indirect_set),
                "covered_count": len(covered_union),
                "coverage": coverage,
                "depth_hops": depth_hops,
                "depth_energy": depth_energy,
                "tei": tei,
            }
        )

        if i % progress_interval == 0 or i == total:
            elapsed = time.time() - start
            eta = (elapsed / i) * (total - i) if i > 0 else 0.0
            print(
                f"进度: {i}/{total} ({100*i/total:.1f}%) | "
                f"elapsed={_format_seconds(elapsed)} | eta={_format_seconds(eta)}"
            )

    if not all_rows:
        print("未生成任何评估结果，请检查输入配置。")
        return

    df = pd.DataFrame(all_rows).sort_values("tei", ascending=False).reset_index(drop=True)
    summary = {
        "policies_evaluated": int(len(df)),
        "coverage_mean": float(df["coverage"].mean()),
        "coverage_median": float(df["coverage"].median()),
        "depth_hops_mean": float(df["depth_hops"].mean()),
        "depth_energy_mean": float(df["depth_energy"].mean()),
        "tei_mean": float(df["tei"].mean()),
        "tei_median": float(df["tei"].median()),
        "top10_policy_ids_by_tei": df.head(10)["policy_id"].astype(int).tolist(),
    }

    out_json = project_root / args.output_json
    out_csv = project_root / args.output_csv
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "params": {
                    "enterprise_adaptive_quantile": args.enterprise_adaptive_quantile,
                    "enterprise_relative_drop_threshold": args.enterprise_relative_drop_threshold,
                    "enterprise_max_output_cap": args.enterprise_max_output_cap,
                    "direct_support_boost": args.direct_support_boost,
                    "ppr_alpha": args.ppr_alpha,
                    "ppr_max_iter": args.ppr_max_iter,
                    "tei_alpha": args.tei_alpha,
                    "tei_beta": args.tei_beta,
                    "tei_gamma": args.tei_gamma,
                },
                "summary": summary,
                "details": df.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n评估完成")
    print(f"- 输出JSON: {out_json}")
    print(f"- 输出CSV: {out_csv}")
    print("总体统计:")
    print(f"  覆盖率均值: {summary['coverage_mean']:.4f}")
    print(f"  平均传导深度(hops): {summary['depth_hops_mean']:.4f}")
    print(f"  平均能量深度: {summary['depth_energy_mean']:.4f}")
    print(f"  TEI均值: {summary['tei_mean']:.4f}")


if __name__ == "__main__":
    main()

