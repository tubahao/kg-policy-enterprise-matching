#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
子图 GAT 嵌入在主对比协议下的匹配评测（与 KG-BERT / OpenKE 同口径）。

打分：训练得到的 policy / company 各 64 维 GAT 嵌入的 **L2 归一化后余弦相似度**
（与对比学习目标一致）。候选池 = 子图内全部政策 / 企业；GT = OpenKE test split supports，
查询集 = build_test_queries_from_data；截断 = quantile + relative_drop + cap。
**E→P 的 max_output_cap** 与 `real_comparison_openke_matching_eval.py` 子图默认一致：按查询类型对 **mask 后 GT**
（政策须在 `valid_policy_titles` 内）长度取均值，`cap = max(1, ceil(mean × multiplier))`，可选 `ceiling`；
显式传入正数 `--policy_max_output_cap` / `--policy_industry_query_max_output_cap` 时覆盖对应分组。

说明：与「全系统 BERT + 多路加权 + 全图 GAT」的数值不可直接比，但协议（查询、GT、截断、指标）一致。
行业并集子图建议加 `--eval_query_scope subgraph_entities`，使 E→P / P→E 查询条数
仅含「查询端落在子图实体上」的项，随子图规模变化。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from subgraph_main_protocol_utils import (  # type: ignore
    SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2,
    SUBGRAPH_EVAL_PROTOCOL_LEGACY,
    build_subgraph_induced_eval_queries,
)

from matching.evaluate_matching import (  # type: ignore
    build_test_queries_from_data,
    calculate_metrics,
    calculate_ranking_metrics,
)


def _read_id_map(path: Path) -> Dict[str, int]:
    mp: Dict[str, int] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        parts = ln.strip().split("\t")
        if len(parts) != 2:
            continue
        mp[parts[0]] = int(parts[1])
    return mp


def _read_openke_triples(path: Path) -> List[Tuple[int, int, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[Tuple[int, int, int]] = []
    for ln in lines[1:]:
        parts = ln.strip().split()
        if len(parts) != 3:
            continue
        out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return out


def _build_support_maps_from_test2id(
    openke_data: Path,
    supports_rid: int,
    token_to_eid: Dict[str, int],
    openke_raw_to_tok: Dict[str, str],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    id_to_token_entity = {v: k for k, v in token_to_eid.items()}
    token_to_raw_entity = {v: k for k, v in openke_raw_to_tok.items()}
    test_triples = _read_openke_triples(openke_data / "test2id.txt")
    e2p: Dict[str, Set[str]] = {}
    p2e: Dict[str, Set[str]] = {}
    for h_id, t_id, r_id in test_triples:
        if r_id != supports_rid:
            continue
        h_tok = id_to_token_entity.get(h_id)
        t_tok = id_to_token_entity.get(t_id)
        if not h_tok or not t_tok:
            continue
        h_raw = token_to_raw_entity.get(h_tok)
        t_raw = token_to_raw_entity.get(t_tok)
        if not h_raw or not t_raw:
            continue
        e2p.setdefault(t_raw, set()).add(h_raw)
        p2e.setdefault(h_raw, set()).add(t_raw)
    return e2p, p2e


def _apply_rank_cutoff(
    ranked_pairs: List[Tuple[int, float]],
    top_k: Optional[int],
    score_threshold: Optional[float],
    adaptive_quantile: Optional[float],
    relative_drop_threshold: Optional[float],
    max_output_cap: Optional[int],
) -> List[Tuple[int, float]]:
    if not ranked_pairs:
        return []
    pairs = sorted(ranked_pairs, key=lambda x: x[1], reverse=True)
    eff_threshold = score_threshold
    if (
        eff_threshold is None
        and adaptive_quantile is not None
        and 0.0 < adaptive_quantile < 1.0
        and pairs
    ):
        score_arr = np.array([s for _, s in pairs], dtype=float)
        eff_threshold = float(np.quantile(score_arr, adaptive_quantile))
    if eff_threshold is not None:
        pairs = [(a, s) for a, s in pairs if s >= eff_threshold]
    if relative_drop_threshold is not None and 0.0 < relative_drop_threshold < 1.0 and len(pairs) > 1:
        kept = [pairs[0]]
        for a, s in pairs[1:]:
            prev_s = kept[-1][1]
            if prev_s > 0:
                dr = (prev_s - s) / max(prev_s, 1e-12)
                if dr > relative_drop_threshold:
                    break
            kept.append((a, s))
        pairs = kept
    if top_k is not None and top_k > 0:
        pairs = pairs[:top_k]
    if max_output_cap is not None and max_output_cap > 0:
        pairs = pairs[:max_output_cap]
    return pairs


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _resolve_output_caps_from_gt(
    enterprise_queries: List[Dict],
    policy_queries: List[Dict],
    valid_policy_titles: Set[str],
    valid_company_names: Set[str],
    multiplier: float,
    ceiling: int,
) -> Tuple[Optional[int], Optional[int], Dict]:
    """与 OpenKE 子图 GT 自适应一致：E→P 分组 cap = max(1, ceil(mean_masked_gt × multiplier))。"""
    company_sizes: List[int] = []
    industry_sizes: List[int] = []
    for q in enterprise_queries:
        gt = [x for x in q["ground_truth"] if x in valid_policy_titles]
        q_type = q.get("type", "company_name")
        if q_type == "industry":
            industry_sizes.append(len(gt))
        else:
            company_sizes.append(len(gt))
    pe_sizes = [
        len([x for x in q["ground_truth"] if x in valid_company_names])
        for q in policy_queries
    ]

    def _one(sizes: List[int]) -> Optional[int]:
        if not sizes:
            return None
        m = float(np.mean(sizes))
        c = max(1, int(math.ceil(m * float(multiplier))))
        if ceiling > 0:
            c = min(c, ceiling)
        return c

    cap_c = _one(company_sizes)
    cap_i = _one(industry_sizes)
    stats = {
        "mean_gt_company_ep": float(np.mean(company_sizes)) if company_sizes else None,
        "mean_gt_industry_ep": float(np.mean(industry_sizes)) if industry_sizes else None,
        "mean_gt_pe": float(np.mean(pe_sizes)) if pe_sizes else None,
        "n_company_ep_queries": len(company_sizes),
        "n_industry_ep_queries": len(industry_sizes),
        "n_pe_queries": len(pe_sizes),
    }
    return cap_c, cap_i, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale_dir",
        type=str,
        required=True,
        help="如 data_intermediate/data_scale_subgraphs/frac_0_1",
    )
    parser.add_argument("--top_k_policy", type=int, default=-1)
    parser.add_argument("--top_k_enterprise", type=int, default=-1)
    parser.add_argument("--policy_candidate_k", type=int, default=1000)
    parser.add_argument("--enterprise_candidate_k", type=int, default=1000)
    parser.add_argument("--policy_score_threshold", type=float, default=-1.0)
    parser.add_argument("--enterprise_score_threshold", type=float, default=-1.0)
    parser.add_argument("--policy_adaptive_quantile", type=float, default=0.72)
    parser.add_argument("--policy_relative_drop_threshold", type=float, default=0.15)
    parser.add_argument(
        "--policy_max_output_cap",
        type=int,
        default=-1,
        help=">0 时强制企业类 E→P 的 cap；≤0 时按 OpenKE 规则用 mask 后 GT 均值（无企业类查询时回退 120）",
    )
    parser.add_argument("--policy_industry_query_adaptive_quantile", type=float, default=0.82)
    parser.add_argument("--policy_industry_query_relative_drop_threshold", type=float, default=0.12)
    parser.add_argument(
        "--policy_industry_query_max_output_cap",
        type=int,
        default=-1,
        help=">0 时强制行业类 E→P 的 cap；≤0 时按 OpenKE 规则（无行业查询时回退 70）",
    )
    parser.add_argument(
        "--adaptive_output_cap_gt_multiplier",
        type=float,
        default=1.0,
        help="与 OpenKE 一致：cap = max(1, ceil(mean_gt_masked × multiplier))",
    )
    parser.add_argument(
        "--adaptive_output_cap_ceiling",
        type=int,
        default=0,
        help=">0 时对自适应 cap 上限截断；0 表示不截断",
    )
    parser.add_argument("--enterprise_adaptive_quantile", type=float, default=0.58)
    parser.add_argument("--enterprise_relative_drop_threshold", type=float, default=0.18)
    parser.add_argument("--enterprise_max_output_cap", type=int, default=150)
    parser.add_argument("--max_enterprise_queries", type=int, default=300)
    parser.add_argument("--max_industry_queries", type=int, default=30)
    parser.add_argument("--max_policy_queries", type=int, default=200)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="默认 reports/real_comparison_results/data_scale_gat_matching_<tag>.json",
    )
    parser.add_argument(
        "--eval_query_scope",
        type=str,
        choices=["full", "subgraph_entities"],
        default="full",
        help="full=与主协议相同条数的查询集；subgraph_entities=仅保留查询端在子图内的查询，"
        "评测条数随子图规模变化（适合行业并集子集）",
    )
    parser.add_argument(
        "--subgraph_eval_protocol",
        type=str,
        default=SUBGRAPH_EVAL_PROTOCOL_LEGACY,
        choices=[SUBGRAPH_EVAL_PROTOCOL_LEGACY, SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2],
        help="legacy_filter=主协议查询+eval_query_scope 过滤；induced_v2=test supports 在子图内构造查询集",
    )
    parser.add_argument("--min_company_ep_queries", type=int, default=40)
    parser.add_argument("--min_industry_ep_queries", type=int, default=12)
    parser.add_argument("--min_pe_queries", type=int, default=25)
    args = parser.parse_args()

    scale_dir = (PROJECT_ROOT / args.scale_dir).resolve()
    if not scale_dir.is_dir():
        raise FileNotFoundError(scale_dir)

    tag = scale_dir.name
    pol_emb_p = scale_dir / "gat_policy_emb_contrastive.npy"
    com_emb_p = scale_dir / "gat_company_emb_contrastive.npy"
    meta_p = scale_dir / "graph_meta.json"
    if not pol_emb_p.is_file() or not com_emb_p.is_file():
        raise FileNotFoundError("请先完成该比例下的 GAT 训练（存在 gat_*_emb_contrastive.npy）")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    node_maps = meta.get("node_maps", {})
    pol_map: Dict[str, int] = {str(k): int(v) for k, v in node_maps.get("policy", {}).items()}
    com_map: Dict[str, int] = {str(k): int(v) for k, v in node_maps.get("company", {}).items()}

    df_pol = pd.read_parquet(scale_dir / "policies_clean.parquet")
    policy_titles = [str(x) for x in df_pol["title"].astype(str).tolist()]
    n_pol = len(pol_map)
    n_com = len(com_map)
    pol_emb = np.load(pol_emb_p).astype(np.float32)
    com_emb = np.load(com_emb_p).astype(np.float32)
    if pol_emb.shape[0] != n_pol or com_emb.shape[0] != n_com:
        raise ValueError(
            f"嵌入行数与 meta 不一致: pol_emb {pol_emb.shape[0]} vs n_pol {n_pol}, "
            f"com {com_emb.shape[0]} vs n_com {n_com}"
        )

    pol_emb_n = _l2n(pol_emb)
    com_emb_n = _l2n(com_emb)
    # sim[i,j] = cos(policy_i, company_j) -> shape (n_pol, n_com)
    sim_pc = pol_emb_n @ com_emb_n.T

    title_to_pid = {str(r.title): int(r.policy_id) for r in df_pol.itertuples(index=False)}
    for t in policy_titles:
        if t not in pol_map:
            raise KeyError(f"政策不在 meta: {t[:50]}...")

    # 列 j 对应的企业名（与 com_emb 行、sim 的列一致）
    sorted_companies: List[str] = [""] * n_com
    for name, idx in com_map.items():
        sorted_companies[int(idx)] = str(name)
    name_to_cidx = {str(n): int(com_map[n]) for n in com_map.keys()}

    data_root = PROJECT_ROOT / "reports" / "real_comparison_data"
    openke_data = data_root / "openke_policykg"
    entity_token_map = json.loads((data_root / "entity_token_map.json").read_text(encoding="utf-8"))
    relation_token_map = json.loads((data_root / "relation_token_map.json").read_text(encoding="utf-8"))
    openke_ent2id = _read_id_map(openke_data / "entity2id.txt")
    openke_rel2id = _read_id_map(openke_data / "relation2id.txt")
    supports_rid_ok = openke_rel2id[relation_token_map["openke"]["supports"]]

    policy_score_threshold = None if args.policy_score_threshold < 0 else float(args.policy_score_threshold)
    enterprise_score_threshold = None if args.enterprise_score_threshold < 0 else float(args.enterprise_score_threshold)

    valid_policy_titles = set(policy_titles)
    valid_company_names = set(com_map.keys())
    policy_id_to_title = {int(r.policy_id): str(r.title) for r in df_pol.itertuples(index=False)}
    policy_name_to_id = {str(r.title): int(r.policy_id) for r in df_pol.itertuples(index=False)}

    df_ent_sub = pd.read_parquet(scale_dir / "enterprises_filtered.parquet")
    industry_to_company_names: Dict[str, List[str]] = {}
    if "industry" in df_ent_sub.columns:
        for ind, grp in df_ent_sub.groupby("industry"):
            industry_to_company_names[str(ind)] = [str(x) for x in grp["name"].astype(str).tolist()]

    use_induced_v2 = (
        str(getattr(args, "subgraph_eval_protocol", SUBGRAPH_EVAL_PROTOCOL_LEGACY)).strip()
        == SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2
    )
    query_set_meta: Optional[Dict] = None

    if use_induced_v2:
        enterprise_queries, policy_queries, query_set_meta = build_subgraph_induced_eval_queries(
            openke_data=openke_data,
            supports_rid=supports_rid_ok,
            token_to_eid=openke_ent2id,
            openke_raw_to_tok=entity_token_map["openke"],
            df_policies=df_pol,
            df_enterprises=df_ent_sub,
            max_enterprise_queries=args.max_enterprise_queries,
            max_industry_queries=args.max_industry_queries,
            max_policy_queries=args.max_policy_queries,
            min_company_ep_queries=int(args.min_company_ep_queries),
            min_industry_ep_queries=int(args.min_industry_ep_queries),
            min_pe_queries=int(args.min_pe_queries),
        )
        print(
            f"[data_scale_gat_eval] induced_v2: company_ep={query_set_meta['n_company_ep_queries']} "
            f"industry_ep={query_set_meta['n_industry_ep_queries']} pe={query_set_meta['n_pe_queries']}",
            flush=True,
        )
    else:
        enterprise_queries, policy_queries = build_test_queries_from_data(
            max_enterprise_queries=args.max_enterprise_queries,
            max_industry_queries=args.max_industry_queries,
            max_policy_queries=args.max_policy_queries,
        )

        e2p_test, p2e_test = _build_support_maps_from_test2id(
            openke_data=openke_data,
            supports_rid=supports_rid_ok,
            token_to_eid=openke_ent2id,
            openke_raw_to_tok=entity_token_map["openke"],
        )
        df_ent_full = pd.read_parquet(PROJECT_ROOT / "data_intermediate" / "enterprises_filtered.parquet")
        industry_to_companies = (
            df_ent_full.groupby("industry")["name"].apply(list).to_dict()
            if "industry" in df_ent_full.columns
            else {}
        )
        for q in enterprise_queries:
            qt = str(q["query"])
            if q.get("type", "company_name") == "industry":
                pols: Set[str] = set()
                for cname in industry_to_companies.get(qt, []):
                    pols.update(e2p_test.get(str(cname), set()))
                q["ground_truth"] = sorted(list(pols))
            else:
                q["ground_truth"] = sorted(list(e2p_test.get(qt, set())))
        for q in policy_queries:
            t = str(q["policy_title"])
            q["ground_truth"] = sorted(list(p2e_test.get(t, set())))
            q["policy_id"] = int(policy_name_to_id.get(t, -1))

        if args.eval_query_scope == "subgraph_entities":
            def _keep_enterprise_query(q: Dict) -> bool:
                qt = str(q["query"])
                if q.get("type", "company_name") == "industry":
                    return any(str(n) in valid_company_names for n in industry_to_companies.get(qt, []))
                return qt in valid_company_names

            enterprise_queries = [q for q in enterprise_queries if _keep_enterprise_query(q)]
            policy_queries = [q for q in policy_queries if str(q["policy_title"]) in valid_policy_titles]

    cap_c, cap_i, cap_gt_stats = _resolve_output_caps_from_gt(
        enterprise_queries,
        policy_queries,
        valid_policy_titles,
        valid_company_names,
        multiplier=float(args.adaptive_output_cap_gt_multiplier),
        ceiling=int(args.adaptive_output_cap_ceiling),
    )
    # 与 OpenKE：cap_c/cap_i 为 None 时用 CLI 默认 120 / 70
    openke_default_policy_cap = 120
    openke_default_industry_cap = 70
    policy_cap = (
        int(args.policy_max_output_cap)
        if args.policy_max_output_cap > 0
        else (cap_c if cap_c is not None else openke_default_policy_cap)
    )
    policy_industry_cap = (
        int(args.policy_industry_query_max_output_cap)
        if args.policy_industry_query_max_output_cap > 0
        else (cap_i if cap_i is not None else openke_default_industry_cap)
    )

    _scope_log = "subgraph_induced_v2" if use_induced_v2 else args.eval_query_scope
    print(
        f"[data_scale_gat_eval] scale={tag} scope={_scope_log} n_pol={n_pol} n_com={n_com} "
        f"E->P queries={len(enterprise_queries)} P->E queries={len(policy_queries)}",
        flush=True,
    )

    ep_metrics: List[Dict] = []
    ep_masked = 0
    for i, q in enumerate(enterprise_queries, start=1):
        q_text = str(q["query"])
        q_type = q.get("type", "company_name")
        gt_raw = [x for x in q["ground_truth"] if x in valid_policy_titles]
        if not gt_raw:
            ep_masked += 1

        if q_type == "industry" and q_text in industry_to_company_names:
            stacks = []
            for cname in industry_to_company_names[q_text]:
                ci = name_to_cidx.get(cname)
                if ci is None:
                    continue
                stacks.append(sim_pc[:, ci])
            score = np.max(np.stack(stacks, axis=0), axis=0) if stacks else np.array([])
        else:
            ci = name_to_cidx.get(q_text)
            if ci is not None:
                score = sim_pc[:, ci]
            else:
                score = np.array([])

        if score.size > 0:
            order = np.argsort(score)[::-1]
            cands = order[: min(args.policy_candidate_k, len(order))]
            ranked = [(int(j), float(score[j])) for j in cands]
            if q_type == "industry":
                ranked = _apply_rank_cutoff(
                    ranked,
                    args.top_k_policy,
                    policy_score_threshold,
                    args.policy_industry_query_adaptive_quantile,
                    args.policy_industry_query_relative_drop_threshold,
                    policy_industry_cap,
                )
            else:
                ranked = _apply_rank_cutoff(
                    ranked,
                    args.top_k_policy,
                    policy_score_threshold,
                    args.policy_adaptive_quantile,
                    args.policy_relative_drop_threshold,
                    policy_cap,
                )
            pred_titles = [policy_id_to_title[pid] for pid, _ in ranked if pid in policy_id_to_title]
        else:
            pred_titles = []
        pred_titles = _dedup_keep_order(pred_titles)
        m = calculate_metrics(pred_titles, gt_raw)
        rm = calculate_ranking_metrics(pred_titles, gt_raw)
        ep_metrics.append(
            {**m, **rm, "query": q_text, "query_type": q_type, "gt_size_after_mask": len(gt_raw)}
        )
        if i % max(1, len(enterprise_queries) // 10) == 0 or i == len(enterprise_queries):
            print(f"[E->P] {i}/{len(enterprise_queries)}", flush=True)

    pe_metrics: List[Dict] = []
    pe_masked = 0
    for i, q in enumerate(policy_queries, start=1):
        title = str(q["policy_title"])
        gt_raw = [x for x in q["ground_truth"] if x in valid_company_names]
        if not gt_raw:
            pe_masked += 1
        pid = title_to_pid.get(title)
        if pid is None or pid >= n_pol:
            pred_names = []
        else:
            score = sim_pc[pid, :]
            order = np.argsort(score)[::-1]
            cands = order[: min(args.enterprise_candidate_k, len(order))]
            ranked_c = [(int(j), float(score[j])) for j in cands]
            ranked_c = _apply_rank_cutoff(
                ranked_c,
                args.top_k_enterprise,
                enterprise_score_threshold,
                args.enterprise_adaptive_quantile,
                args.enterprise_relative_drop_threshold,
                args.enterprise_max_output_cap,
            )
            pred_names = [sorted_companies[j] for j, _ in ranked_c if 0 <= j < len(sorted_companies)]
        pred_names = _dedup_keep_order(pred_names)
        m = calculate_metrics(pred_names, gt_raw)
        rm = calculate_ranking_metrics(pred_names, gt_raw)
        pe_metrics.append(
            {
                **m,
                **rm,
                "policy_id": int(q["policy_id"]),
                "policy_title": title,
                "gt_size_after_mask": len(gt_raw),
            }
        )
        if i % max(1, len(policy_queries) // 10) == 0 or i == len(policy_queries):
            print(f"[P->E] {i}/{len(policy_queries)}", flush=True)

    def _avg_block(rows: List[Dict]) -> Dict[str, float]:
        if not rows:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "map": 0.0,
                "ndcg": 0.0,
            }
        return {
            "precision": float(np.mean([x["precision"] for x in rows])),
            "recall": float(np.mean([x["recall"] for x in rows])),
            "f1": float(np.mean([x["f1"] for x in rows])),
            "map": float(np.mean([x["ap"] for x in rows])),
            "ndcg": float(np.mean([x["ndcg"] for x in rows])),
        }

    ep_avg = {
        **_avg_block(ep_metrics),
        "masked_gt_empty": ep_masked,
    }
    ep_nonempty = [x for x in ep_metrics if x.get("gt_size_after_mask", 0) > 0]
    ep_avg_nonempty = {**_avg_block(ep_nonempty), "num_queries": len(ep_nonempty)}

    pe_avg = {
        **_avg_block(pe_metrics),
        "masked_gt_empty": pe_masked,
    }
    pe_nonempty = [x for x in pe_metrics if x.get("gt_size_after_mask", 0) > 0]
    pe_avg_nonempty = {**_avg_block(pe_nonempty), "num_queries": len(pe_nonempty)}

    default_out = f"reports/real_comparison_results/data_scale_gat_matching_{tag}.json"
    out_path = PROJECT_ROOT / (args.output or default_out)
    _eval_scope = "subgraph_induced_v2" if use_induced_v2 else args.eval_query_scope
    _eval_query_set = (
        "induced_v2_test_supports"
        if use_induced_v2
        else (
            "legacy_main_protocol_subgraph_filtered"
            if args.eval_query_scope == "subgraph_entities"
            else "legacy_main_protocol_full"
        )
    )
    result = {
        "model": "HeteroGATContrastive-embedding-cosine",
        "scale_dir": str(scale_dir.relative_to(PROJECT_ROOT)),
        "eval_query_scope": _eval_scope,
        "eval_query_set": _eval_query_set,
        "evaluation_protocol": "main_queries + test_split_gt + unified_cutoff",
        "scoring": "L2-normalized cosine between subgraph GAT policy and company embeddings",
        "timestamp": datetime.now().isoformat(),
        "output_cap_mode": "openke_gt_masked_ceil",
        "parameters": {
            "subgraph_eval_protocol": args.subgraph_eval_protocol,
            "min_company_ep_queries": int(args.min_company_ep_queries),
            "min_industry_ep_queries": int(args.min_industry_ep_queries),
            "min_pe_queries": int(args.min_pe_queries),
            "eval_query_scope_cli": args.eval_query_scope,
            "adaptive_output_cap_gt_multiplier": float(args.adaptive_output_cap_gt_multiplier),
            "adaptive_output_cap_ceiling": int(args.adaptive_output_cap_ceiling),
            "effective_policy_max_output_cap": policy_cap,
            "effective_policy_industry_query_max_output_cap": policy_industry_cap,
            "policy_max_output_cap_cli": int(args.policy_max_output_cap),
            "policy_industry_query_max_output_cap_cli": int(args.policy_industry_query_max_output_cap),
            "adaptive_output_cap_gt_stats": cap_gt_stats,
        },
        "subgraph_stats": {"num_policies": n_pol, "num_companies": n_com},
        "enterprise_to_policy": {
            "num_queries": len(ep_metrics),
            "average": ep_avg,
            "average_gt_nonempty_only": ep_avg_nonempty,
        },
        "policy_to_enterprise": {
            "num_queries": len(pe_metrics),
            "average": pe_avg,
            "average_gt_nonempty_only": pe_avg_nonempty,
        },
    }
    if query_set_meta is not None:
        result["query_set_meta"] = query_set_meta
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPO_ROOT / "report" / out_path.name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
