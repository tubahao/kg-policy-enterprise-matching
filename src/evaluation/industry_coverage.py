#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业条件覆盖率（探索脚本）：
- I(p)：图中 (policy, targetsIndustry, industry) 的客体行业名，且仅保留 industry_mapping_complete.json 的 major_industries（六大类）。
- 相关行业企业全集 E_rel(p)：图中属于 I(p) 任一行业的 company 节点（belongsTo）的并集。
- 分子：(检索返回企业 ∪ supports 直连企业) ∩ E_rel(p)；分母：|E_rel(p)|。

policy_scope:
- test：与 transmission_efficiency 相同的 P→E 测试查询子集；无六大类 targets 或 E_rel 空时回退全图分母（与旧脚本一致）。
- all_industry_only：遍历 policies_clean；仅评测图上有六大类 targetsIndustry 且 E_rel 非空者，其余跳过（不做全图回退）。
"""

from __future__ import annotations

import argparse
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

from matching.bidirectional_matching import load_policy_to_enterprise_retriever  # noqa: E402
from matching.evaluate_matching import build_test_queries_from_data  # noqa: E402


def load_major_industries(root: Path) -> List[str]:
    path = root / "industry_mapping_complete.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    majors = list(data.get("major_industries", []))
    if not majors:
        raise RuntimeError("industry_mapping_complete.json 缺少 major_industries")
    return majors


def build_industry_nid_to_name(node_maps: dict) -> Dict[int, str]:
    ind_map = node_maps.get("industry") or {}
    return {int(v): str(k) for k, v in ind_map.items()}


def build_major_company_sets(
    graph,
    majors: Set[str],
    industry_nid_to_name: Dict[int, str],
) -> Dict[str, Set[int]]:
    """行业名（六大类之一）-> 图中 company 节点 id 集合（belongsTo）。"""
    et = ("company", "belongsTo", "industry")
    if et not in graph.canonical_etypes:
        return {m: set() for m in majors}
    src, dst = graph.edges(etype=et)
    src_np = src.numpy()
    dst_np = dst.numpy()
    out: Dict[str, Set[int]] = {m: set() for m in majors}
    for c, i in zip(src_np.tolist(), dst_np.tolist()):
        name = industry_nid_to_name.get(int(i))
        if name in majors:
            out[name].add(int(c))
    return out


def policy_targets_major_industries(
    graph,
    policy_node_id: int,
    majors: Set[str],
    industry_nid_to_name: Dict[int, str],
) -> Set[str]:
    """图中该政策节点 targetsIndustry 指向的行业名 ∩ majors。"""
    et = ("policy", "targetsIndustry", "industry")
    if et not in graph.canonical_etypes:
        return set()
    src, dst = graph.edges(etype=et)
    src_np = src.numpy()
    dst_np = dst.numpy()
    mask = src_np == int(policy_node_id)
    names: Set[str] = set()
    for i in dst_np[mask].tolist():
        nm = industry_nid_to_name.get(int(i))
        if nm is not None and nm in majors:
            names.add(nm)
    return names


def union_companies_for_industries(
    industry_names: Set[str],
    major_to_companies: Dict[str, Set[int]],
) -> Set[int]:
    s: Set[int] = set()
    for m in industry_names:
        s |= major_to_companies.get(m, set())
    return s


def build_global_graph_structures_from_graph(g):
    """与 transmission_efficiency.build_global_graph_structures 相同逻辑，仅输入 DGL 图。"""
    offsets: Dict[str, int] = {}
    cur = 0
    for ntype in g.ntypes:
        offsets[ntype] = cur
        cur += g.number_of_nodes(ntype)
    total_nodes = cur

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
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

    for s in range(total_nodes):
        deg = int(out_deg[s])
        if deg > 0:
            w = 1.0 / deg
            for d in adjacency[s]:
                rows.append(d)
                cols.append(s)
                vals.append(w)
        else:
            rows.append(s)
            cols.append(s)
            vals.append(1.0)

    pt = sparse.csr_matrix((vals, (rows, cols)), shape=(total_nodes, total_nodes), dtype=np.float64)
    return offsets, adjacency, pt


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


DEFAULT_JSON = "evaluation/industry_conditioned_coverage.json"
DEFAULT_CSV = "evaluation/industry_conditioned_coverage.csv"
DEFAULT_JSON_ALL = "evaluation/industry_conditioned_coverage_all_policies.json"
DEFAULT_CSV_ALL = "evaluation/industry_conditioned_coverage_all_policies.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="行业条件覆盖率（六大类；test 可回退全图 / all 仅行业）")
    parser.add_argument("--top_k_enterprise", type=int, default=-1)
    parser.add_argument("--enterprise_candidate_k", type=int, default=1000)
    parser.add_argument("--enterprise_score_threshold", type=float, default=-1.0)
    parser.add_argument("--enterprise_adaptive_quantile", type=float, default=0.58)
    parser.add_argument("--enterprise_relative_drop_threshold", type=float, default=0.18)
    parser.add_argument("--enterprise_max_output_cap", type=int, default=150)
    parser.add_argument("--direct_support_boost", type=float, default=0.3)
    parser.add_argument("--max_enterprise_queries", type=int, default=300)
    parser.add_argument("--max_industry_queries", type=int, default=30)
    parser.add_argument("--max_policy_queries", type=int, default=200)
    parser.add_argument(
        "--policy_scope",
        type=str,
        choices=["test", "all_industry_only"],
        default="test",
        help="test=主实验 P→E 查询集（可全图回退）；all_industry_only=policies_clean 全量，仅行业口径，无行业不评",
    )
    parser.add_argument("--output_json", type=str, default=DEFAULT_JSON)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_CSV)
    parser.add_argument("--ppr_alpha", type=float, default=0.15)
    parser.add_argument("--ppr_max_iter", type=int, default=40)
    parser.add_argument("--tei_alpha", type=float, default=0.4)
    parser.add_argument("--tei_beta", type=float, default=0.3)
    parser.add_argument("--tei_gamma", type=float, default=0.3)
    args = parser.parse_args()

    if args.policy_scope == "all_industry_only":
        if args.output_json == DEFAULT_JSON:
            args.output_json = DEFAULT_JSON_ALL
        if args.output_csv == DEFAULT_CSV:
            args.output_csv = DEFAULT_CSV_ALL

    enterprise_score_threshold = None if args.enterprise_score_threshold < 0 else args.enterprise_score_threshold

    majors_list = load_major_industries(project_root)
    majors_set = set(majors_list)
    print(f"六大类行业 ({len(majors_set)}): {majors_list}")

    print("初始化 PolicyToEnterpriseRetriever（无 BERT）…")
    t0 = time.time()
    retriever = load_policy_to_enterprise_retriever(project_root)

    if args.policy_scope == "test":
        _, policy_queries = build_test_queries_from_data(
            max_enterprise_queries=args.max_enterprise_queries if args.max_enterprise_queries > 0 else None,
            max_industry_queries=args.max_industry_queries if args.max_industry_queries > 0 else None,
            max_policy_queries=args.max_policy_queries if args.max_policy_queries > 0 else None,
        )
        policy_jobs = [(int(q["policy_id"]), str(q["policy_title"])) for q in policy_queries]
        print(f"政策查询数 (test): {len(policy_jobs)} | 初始化耗时: {time.time() - t0:.1f}s")
    else:
        pdf = retriever.policies_df
        if pdf is None:
            pdf = pd.read_parquet(project_root / "data_intermediate/policies_clean.parquet")
        policy_jobs = [(int(r["policy_id"]), str(r["title"])) for _, r in pdf.iterrows()]
        print(f"政策总数 (policies_clean): {len(policy_jobs)} | 初始化耗时: {time.time() - t0:.1f}s")

    g = retriever.graph
    node_maps = retriever.node_maps or {}
    industry_nid_to_name = build_industry_nid_to_name(node_maps)
    major_to_companies = build_major_company_sets(g, majors_set, industry_nid_to_name)
    company_count = g.number_of_nodes("company")

    n_companies_per_major = {m: len(major_to_companies[m]) for m in majors_list}
    print("各行业图中企业数（belongsTo）:", n_companies_per_major)

    offsets, adjacency, pt = build_global_graph_structures_from_graph(g)
    policy_off = offsets["policy"]
    company_off = offsets["company"]

    rows: List[dict] = []
    skipped_no_node = 0
    skipped_no_industry = 0
    total_jobs = len(policy_jobs)

    for policy_id, policy_title in policy_jobs:
        policy_node = retriever._resolve_policy_node_id(policy_id)
        if policy_node is None:
            skipped_no_node += 1
            continue

        ip = policy_targets_major_industries(g, int(policy_node), majors_set, industry_nid_to_name)
        erel = union_companies_for_industries(ip, major_to_companies)

        if args.policy_scope == "all_industry_only":
            if len(ip) == 0 or len(erel) == 0:
                skipped_no_industry += 1
                continue

        results = retriever.retrieve_enterprises(
            policy_id,
            top_k=args.top_k_enterprise,
            score_threshold=enterprise_score_threshold,
            candidate_k=args.enterprise_candidate_k,
            adaptive_quantile=args.enterprise_adaptive_quantile,
            relative_drop_threshold=args.enterprise_relative_drop_threshold,
            max_output_cap=args.enterprise_max_output_cap,
            direct_support_boost=args.direct_support_boost,
        )
        covered = {int(cid) for cid, _ in results}
        direct_set = retriever.policy_direct_support_companies.get(int(policy_node), set())
        direct_set = {int(x) for x in direct_set}
        covered_union = covered | direct_set

        coverage_global = (len(covered_union) / company_count) if company_count > 0 else 0.0

        if args.policy_scope == "test":
            fallback = len(ip) == 0 or len(erel) == 0
            if fallback:
                denom = company_count
                numer = len(covered_union)
                coverage_industry = coverage_global
                mode = "fallback_full_graph" if len(ip) == 0 else "fallback_erel_empty"
            else:
                denom = len(erel)
                numer = len(covered_union & erel)
                coverage_industry = numer / max(denom, 1)
                mode = "industry_conditioned"
        else:
            denom = len(erel)
            numer = len(covered_union & erel)
            coverage_industry = numer / max(denom, 1)
            mode = "industry_conditioned"
            fallback = False

        if not fallback:
            output_precision_in_erel = len(covered & erel) / max(len(covered), 1)
        else:
            output_precision_in_erel = float("nan")

        src_gid = policy_off + int(policy_node)
        dist = bfs_shortest_dist(adjacency, src_gid)
        covered_gids = [company_off + cid for cid in covered_union if 0 <= cid < company_count]
        hop_vals = [int(dist[gidx]) for gidx in covered_gids if dist[gidx] >= 0]
        depth_hops = float(np.mean(hop_vals)) if hop_vals else 0.0

        ppr = personalized_pagerank_vector(
            pt, src_gid, alpha=args.ppr_alpha, max_iter=args.ppr_max_iter
        )
        if covered_gids:
            ppr_vals = np.array([float(ppr[gidx]) for gidx in covered_gids], dtype=np.float64)
            depth_energy = float(1.0 - np.mean(ppr_vals))
        else:
            depth_energy = 1.0

        inv_depth = (1.0 / depth_hops) if depth_hops > 0 else 0.0
        tei_global = (
            args.tei_alpha * coverage_global
            + args.tei_beta * inv_depth
            + args.tei_gamma * depth_energy
        )
        tei_industry = (
            args.tei_alpha * coverage_industry
            + args.tei_beta * inv_depth
            + args.tei_gamma * depth_energy
        )

        rows.append(
            {
                "policy_id": policy_id,
                "policy_title": policy_title,
                "policy_node_id": int(policy_node),
                "targets_industry_majors": sorted(ip),
                "n_targets_majors": len(ip),
                "n_erel_union": len(erel),
                "n_covered_retrieval": len(covered),
                "n_direct_support": len(direct_set),
                "indirect_count": len(covered - direct_set),
                "n_covered_union": len(covered_union),
                "numerator_intersection": numer,
                "denominator": denom,
                "coverage_industry": float(coverage_industry),
                "coverage_global": float(coverage_global),
                "fallback_full_graph": bool(fallback),
                "mode": mode,
                "output_precision_in_erel": float(output_precision_in_erel),
                "depth_hops": float(depth_hops),
                "depth_energy": float(depth_energy),
                "inv_depth": float(inv_depth),
                "tei_global": float(tei_global),
                "tei_industry": float(tei_industry),
            }
        )

    df = pd.DataFrame(rows)
    ind_only = df[~df["fallback_full_graph"]] if len(df) and "fallback_full_graph" in df.columns else df

    summary = {
        "policy_scope": args.policy_scope,
        "company_count_graph": int(company_count),
        "majors": majors_list,
        "policies_evaluated": int(len(df)),
        "coverage_industry_mean": float(df["coverage_industry"].mean()) if len(df) else 0.0,
        "coverage_industry_median": float(df["coverage_industry"].median()) if len(df) else 0.0,
        "coverage_global_mean": float(df["coverage_global"].mean()) if len(df) else 0.0,
        "coverage_global_median": float(df["coverage_global"].median()) if len(df) else 0.0,
        "depth_hops_mean": float(df["depth_hops"].mean()) if len(df) else 0.0,
        "depth_hops_median": float(df["depth_hops"].median()) if len(df) else 0.0,
        "depth_energy_mean": float(df["depth_energy"].mean()) if len(df) else 0.0,
        "depth_energy_median": float(df["depth_energy"].median()) if len(df) else 0.0,
        "tei_global_mean": float(df["tei_global"].mean()) if len(df) else 0.0,
        "tei_global_median": float(df["tei_global"].median()) if len(df) else 0.0,
        "tei_industry_mean": float(df["tei_industry"].mean()) if len(df) else 0.0,
        "tei_industry_median": float(df["tei_industry"].median()) if len(df) else 0.0,
        "output_precision_in_erel_mean": float(np.nanmean(df["output_precision_in_erel"].to_numpy())) if len(df) else 0.0,
    }
    if len(df):
        dfi = df.sort_values("tei_industry", ascending=False).reset_index(drop=True)
        summary["top10_policy_ids_by_tei_industry"] = dfi.head(10)["policy_id"].astype(int).tolist()

    if args.policy_scope == "test":
        summary["fallback_count"] = int(df["fallback_full_graph"].sum()) if len(df) else 0
        summary["industry_conditioned_count"] = int(len(ind_only))
        summary["coverage_industry_mean_all_rows"] = float(df["coverage_industry"].mean()) if len(df) else 0.0
        summary["coverage_industry_mean_conditioned_only"] = (
            float(ind_only["coverage_industry"].mean()) if len(ind_only) else None
        )
        summary["coverage_industry_median_conditioned_only"] = (
            float(ind_only["coverage_industry"].median()) if len(ind_only) else None
        )
        if len(ind_only):
            summary["tei_industry_mean_conditioned_only"] = float(ind_only["tei_industry"].mean())
            summary["tei_industry_median_conditioned_only"] = float(ind_only["tei_industry"].median())
            summary["depth_hops_mean_conditioned_only"] = float(ind_only["depth_hops"].mean())
            summary["depth_energy_mean_conditioned_only"] = float(ind_only["depth_energy"].mean())
    else:
        summary["policies_in_table"] = int(total_jobs)
        summary["skipped_no_graph_node"] = int(skipped_no_node)
        summary["skipped_no_industry_or_empty_erel"] = int(skipped_no_industry)

    params_out = {
        "enterprise_adaptive_quantile": args.enterprise_adaptive_quantile,
        "enterprise_relative_drop_threshold": args.enterprise_relative_drop_threshold,
        "enterprise_max_output_cap": args.enterprise_max_output_cap,
        "direct_support_boost": args.direct_support_boost,
        "ppr_alpha": args.ppr_alpha,
        "ppr_max_iter": args.ppr_max_iter,
        "tei_alpha": args.tei_alpha,
        "tei_beta": args.tei_beta,
        "tei_gamma": args.tei_gamma,
    }

    out_j = project_root / args.output_json
    out_c = project_root / args.output_csv
    out_j.parent.mkdir(parents=True, exist_ok=True)
    with open(out_j, "w", encoding="utf-8") as f:
        json.dump({"params": params_out, "summary": summary, "details": rows}, f, ensure_ascii=False, indent=2)
    df.to_csv(out_c, index=False, encoding="utf-8-sig")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {out_j}")
    print(f"Wrote {out_c}")


if __name__ == "__main__":
    main()
