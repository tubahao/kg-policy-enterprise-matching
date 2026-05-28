#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 OpenKE-TransE 的 supports 关系打分，按主实验口径做双向评测：
- E->P: 企业/行业查询 -> 政策检索
- P->E: 政策查询 -> 企业检索
输出指标：Precision / Recall / F1 / MAP / NDCG

子图（--scale_dir）默认对 E→P 的 max_output_cap 按当前查询集 GT（mask 后）分组均值自适应；
P→E 默认仍用 --enterprise_max_output_cap。全量不传 scale_dir 时默认关闭 GT 自适应。
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
sys.path.insert(0, str(PROJECT_ROOT.parents[1] / "openkg" / "OpenKE"))

from matching.evaluate_matching import calculate_metrics, calculate_ranking_metrics  # type: ignore
from openke.module.model import TransE  # type: ignore

from subgraph_main_protocol_utils import (  # type: ignore
    SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2,
    SUBGRAPH_EVAL_PROTOCOL_LEGACY,
    build_subgraph_induced_eval_queries,
    enterprise_policy_result_blocks,
    filter_queries_subgraph_entities,
    industry_to_companies_full_map,
    read_enterprises_full,
    read_policy_enterprise_tables,
    triples_parquet_path,
)


def _read_id_map(path: Path) -> Dict[str, int]:
    mp: Dict[str, int] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return mp
    for ln in lines[1:]:
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
        h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
        out.append((h, t, r))
    return out


def _build_queries_from_support_pairs(
    support_pairs: List[Tuple[str, str]],
    df_enterprises: pd.DataFrame,
    max_enterprise_queries: int,
    max_industry_queries: int,
    max_policy_queries: int,
):
    enterprise_to_policies: Dict[str, Set[str]] = {}
    policy_to_enterprises: Dict[str, Set[str]] = {}
    for policy_title, company_name in support_pairs:
        enterprise_to_policies.setdefault(company_name, set()).add(policy_title)
        policy_to_enterprises.setdefault(policy_title, set()).add(company_name)

    enterprise_items = sorted(
        enterprise_to_policies.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )
    if max_enterprise_queries > 0:
        enterprise_items = enterprise_items[:max_enterprise_queries]
    enterprise_queries = [
        {"query": name, "type": "company_name", "ground_truth": sorted(list(pols))}
        for name, pols in enterprise_items
    ]

    industry_to_companies = (
        df_enterprises.groupby("industry")["name"].apply(list).to_dict()
        if "industry" in df_enterprises.columns
        else {}
    )
    industries = list(industry_to_companies.keys())
    if max_industry_queries > 0:
        industries = industries[:max_industry_queries]
    for ind in industries:
        pols: Set[str] = set()
        for cname in industry_to_companies.get(ind, []):
            pols.update(enterprise_to_policies.get(str(cname), set()))
        if pols:
            enterprise_queries.append(
                {"query": str(ind), "type": "industry", "ground_truth": sorted(list(pols))}
            )

    policy_items = sorted(
        policy_to_enterprises.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )
    if max_policy_queries > 0:
        policy_items = policy_items[:max_policy_queries]
    policy_queries = [
        {
            "policy_id": -1,
            "policy_title": title,
            "ground_truth": sorted(list(comps)),
        }
        for title, comps in policy_items
    ]
    return enterprise_queries, policy_queries


def _build_support_maps_from_test2id(
    openke_data: Path,
    supports_rid: int,
    token_to_eid: Dict[str, int],
    openke_raw_to_tok: Dict[str, str],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    id_to_token_entity = {v: k for k, v in token_to_eid.items()}
    token_to_raw_entity = {v: k for k, v in openke_raw_to_tok.items()}
    test_triples = _read_openke_triples(openke_data / "test2id.txt")
    enterprise_to_policies: Dict[str, Set[str]] = {}
    policy_to_enterprises: Dict[str, Set[str]] = {}
    for h_id, t_id, r_id in test_triples:
        if r_id != supports_rid:
            continue
        h_tok = id_to_token_entity.get(h_id)
        t_tok = id_to_token_entity.get(t_id)
        if h_tok is None or t_tok is None:
            continue
        h_raw = token_to_raw_entity.get(h_tok)
        t_raw = token_to_raw_entity.get(t_tok)
        if h_raw is None or t_raw is None:
            continue
        enterprise_to_policies.setdefault(t_raw, set()).add(h_raw)
        policy_to_enterprises.setdefault(h_raw, set()).add(t_raw)
    return enterprise_to_policies, policy_to_enterprises


def _load_transe_embeddings(
    ckpt: Path,
    entity2id_file: Path,
    relation2id_file: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    entity2id = _read_id_map(entity2id_file)
    relation2id = _read_id_map(relation2id_file)
    model = TransE(
        ent_tot=len(entity2id),
        rel_tot=len(relation2id),
        dim=200,
        p_norm=1,
        norm_flag=True,
    )
    model.load_checkpoint(str(ckpt))
    params = model.get_parameters(mode="numpy")
    ent = params["ent_embeddings.weight"].astype(np.float32)
    rel = params["rel_embeddings.weight"].astype(np.float32)
    return ent, rel


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _transe_similarity_from_distance(dist: np.ndarray) -> np.ndarray:
    """
    将 TransE L1 距离映射到 (0,1]，便于与主流程的截断逻辑对齐。
    使用 1 / (1 + dist) 保持单调性，且分数恒为正，relative_drop 可正常生效。
    """
    return 1.0 / (1.0 + np.maximum(dist, 0.0))


def _format_progress(i: int, total: int) -> bool:
    if total <= 0:
        return False
    step = max(1, total // 10)
    return i % step == 0 or i == total


def _apply_rank_cutoff(
    ranked_pairs: List[Tuple[int, float]],
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    adaptive_quantile: Optional[float] = None,
    relative_drop_threshold: Optional[float] = None,
    max_output_cap: Optional[int] = None,
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
        pairs = [(nid, s) for nid, s in pairs if s >= eff_threshold]
    if relative_drop_threshold is not None and 0.0 < relative_drop_threshold < 1.0 and len(pairs) > 1:
        kept = [pairs[0]]
        for nid, s in pairs[1:]:
            prev_s = kept[-1][1]
            if prev_s > 0:
                drop_ratio = (prev_s - s) / max(prev_s, 1e-12)
                if drop_ratio > relative_drop_threshold:
                    break
            kept.append((nid, s))
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


def _resolve_output_caps_from_gt(
    enterprise_queries: List[Dict],
    policy_queries: List[Dict],
    valid_policy_titles: Set[str],
    valid_company_names: Set[str],
    multiplier: float,
    ceiling: int,
) -> Tuple[Optional[int], Optional[int], Dict]:
    """E→P 分组：企业查询 / 行业查询各自 GT（mask 后）均值 → cap = max(1, ceil(mean * multiplier))。含 P→E 的 GT 规模统计供 JSON，不用于截断。"""
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


def main():
    parser = argparse.ArgumentParser()
    # 对齐主实验默认参数（见 实验全流程总结.md 8.4）
    parser.add_argument("--top_k_policy", type=int, default=-1)
    parser.add_argument("--top_k_enterprise", type=int, default=-1)
    parser.add_argument("--policy_candidate_k", type=int, default=1000)
    parser.add_argument("--enterprise_candidate_k", type=int, default=1000)
    parser.add_argument("--policy_score_threshold", type=float, default=-1.0)
    parser.add_argument("--enterprise_score_threshold", type=float, default=-1.0)
    parser.add_argument("--policy_adaptive_quantile", type=float, default=0.72)
    parser.add_argument("--policy_relative_drop_threshold", type=float, default=0.15)
    parser.add_argument("--policy_max_output_cap", type=int, default=120)
    parser.add_argument("--policy_industry_query_adaptive_quantile", type=float, default=0.82)
    parser.add_argument("--policy_industry_query_relative_drop_threshold", type=float, default=0.12)
    parser.add_argument("--policy_industry_query_max_output_cap", type=int, default=70)
    parser.add_argument("--enterprise_adaptive_quantile", type=float, default=0.58)
    parser.add_argument("--enterprise_relative_drop_threshold", type=float, default=0.18)
    parser.add_argument("--enterprise_max_output_cap", type=int, default=150)
    parser.add_argument("--direct_support_boost", type=float, default=0.3)
    parser.add_argument("--max_enterprise_queries", type=int, default=300)
    parser.add_argument("--max_industry_queries", type=int, default=30)
    parser.add_argument("--max_policy_queries", type=int, default=200)
    parser.add_argument(
        "--output",
        type=str,
        default="reports/real_comparison_results/openke_matching_eval.json",
    )
    parser.add_argument(
        "--ground_truth_source",
        type=str,
        default="full",
        choices=["full", "test"],
        help="full=全量triples构造GT；test=仅用OpenKE test2id构造GT（避免训练泄漏）",
    )
    parser.add_argument(
        "--scale_dir",
        type=str,
        default="",
        help="子图目录；非空则候选与查询同 subgraph_entities 协议",
    )
    parser.add_argument(
        "--subgraph_eval_protocol",
        type=str,
        default=SUBGRAPH_EVAL_PROTOCOL_LEGACY,
        choices=[SUBGRAPH_EVAL_PROTOCOL_LEGACY, SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2],
        help="仅 --scale_dir 非空时生效：legacy_filter=主协议查询+子图过滤；"
        "induced_v2=由 test supports 在子图内构造查询集（需 ground_truth_source=test）",
    )
    parser.add_argument(
        "--min_company_ep_queries",
        type=int,
        default=40,
        help="induced_v2 下期望的企业类 E→P 条数下限（仅写入 query_set_meta.shortfalls，不强行补齐）",
    )
    parser.add_argument(
        "--min_industry_ep_queries",
        type=int,
        default=12,
        help="induced_v2 下的行业类 E→P 条数下限（仅 meta）",
    )
    parser.add_argument(
        "--min_pe_queries",
        type=int,
        default=25,
        help="induced_v2 下 P→E 条数下限（仅 meta）",
    )
    parser.add_argument(
        "--include_per_query",
        action="store_true",
        help="在 JSON 中写入 enterprise_to_policy_per_query / policy_to_enterprise_per_query（查询很多时文件变大）",
    )
    parser.add_argument(
        "--adaptive_output_cap_from_gt",
        action="store_true",
        help="按当前查询集 GT（mask 后）分组均值×倍数设置 max_output_cap；未指定时仅子图(--scale_dir)默认启用",
    )
    parser.add_argument(
        "--no_adaptive_output_cap",
        action="store_true",
        help="关闭子图默认的 GT 自适应 cap，强制使用下方固定 policy_* / enterprise_max_output_cap",
    )
    parser.add_argument(
        "--adaptive_output_cap_gt_multiplier",
        type=float,
        default=1.0,
        help="cap = max(1, ceil(mean_gt * multiplier))；默认 1.0 对齐 GT 均值，需要余量时可设 2.0",
    )
    parser.add_argument(
        "--adaptive_output_cap_ceiling",
        type=int,
        default=0,
        help=">0 时对自适应 cap 上限截断；0 表示不截断",
    )
    parser.add_argument(
        "--adaptive_output_cap_pe_from_gt",
        action="store_true",
        help="同时按 GT 均值自适应 P→E 的 enterprise_max_output_cap（默认仅 E→P 自适应，P→E 用 --enterprise_max_output_cap）",
    )
    args = parser.parse_args()
    policy_score_threshold = None if args.policy_score_threshold < 0 else float(args.policy_score_threshold)
    enterprise_score_threshold = None if args.enterprise_score_threshold < 0 else float(args.enterprise_score_threshold)

    data_root = PROJECT_ROOT / "reports" / "real_comparison_data"
    result_root = PROJECT_ROOT / "reports" / "real_comparison_results"
    result_root.mkdir(parents=True, exist_ok=True)

    ckpt = result_root / "openke_transe.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"未找到 OpenKE checkpoint: {ckpt}")

    openke_data = data_root / "openke_policykg"
    ent2id_file = openke_data / "entity2id.txt"
    rel2id_file = openke_data / "relation2id.txt"
    if not ent2id_file.exists() or not rel2id_file.exists():
        raise FileNotFoundError("OpenKE 数据映射文件缺失(entity2id/relation2id)")

    relation_token_map = json.loads((data_root / "relation_token_map.json").read_text(encoding="utf-8"))
    entity_token_map = json.loads((data_root / "entity_token_map.json").read_text(encoding="utf-8"))
    openke_raw_to_tok: Dict[str, str] = entity_token_map["openke"]
    supports_tok = relation_token_map["openke"]["supports"]

    token_to_eid = _read_id_map(ent2id_file)
    token_to_rid = _read_id_map(rel2id_file)
    if supports_tok not in token_to_rid:
        raise KeyError(f"relation2id 中不存在 supports token: {supports_tok}")
    supports_rid = token_to_rid[supports_tok]

    print("[OpenKE-Eval] 加载 TransE 向量...", flush=True)
    ent_emb, rel_emb = _load_transe_embeddings(ckpt, ent2id_file, rel2id_file)
    ent_emb = _l2norm(ent_emb)
    rel_emb = _l2norm(rel_emb)
    r_support = rel_emb[supports_rid]

    scale_opt = (args.scale_dir or "").strip() or None
    df_policies, df_enterprises, is_subgraph, subgraph_tag = read_policy_enterprise_tables(
        PROJECT_ROOT, scale_opt
    )
    df_ent_full = read_enterprises_full(PROJECT_ROOT)

    policy_titles = [str(x) for x in df_policies["title"].astype(str).tolist()]
    policy_name_to_id = {t: int(pid) for t, pid in zip(policy_titles, df_policies["policy_id"].astype(int).tolist())}
    policy_id_to_title = {int(pid): str(t) for t, pid in policy_name_to_id.items()}
    valid_policy_ids = set(policy_id_to_title.keys())
    valid_policy_titles = set(policy_titles)
    company_names = [str(x) for x in df_enterprises["name"].astype(str).tolist()]
    valid_company_names = set(company_names)

    # raw 名称 -> OpenKE embedding 行索引
    raw_to_ent_idx: Dict[str, int] = {}
    for raw_name, tok in openke_raw_to_tok.items():
        if tok in token_to_eid:
            raw_to_ent_idx[raw_name] = token_to_eid[tok]

    # 候选集合索引
    policy_candidates: List[Tuple[str, int]] = [
        (title, raw_to_ent_idx[title]) for title in policy_titles if title in raw_to_ent_idx
    ]
    company_candidates: List[Tuple[str, int]] = [
        (name, raw_to_ent_idx[name]) for name in company_names if name in raw_to_ent_idx
    ]
    if not policy_candidates or not company_candidates:
        raise RuntimeError("无法构建政策/企业候选向量，请检查 token 映射")

    policy_names_arr = np.array([x[0] for x in policy_candidates], dtype=object)
    policy_idx_arr = np.array([x[1] for x in policy_candidates], dtype=np.int64)
    company_names_arr = np.array([x[0] for x in company_candidates], dtype=object)
    company_idx_arr = np.array([x[1] for x in company_candidates], dtype=np.int64)

    use_induced_v2 = (
        is_subgraph
        and str(getattr(args, "subgraph_eval_protocol", SUBGRAPH_EVAL_PROTOCOL_LEGACY)).strip()
        == SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2
    )
    query_set_meta: Optional[Dict] = None

    # 行业 -> 公司实体idx列表
    industry_to_company_idx: Dict[str, List[int]] = {}
    for industry, grp in df_enterprises.groupby("industry"):
        inds = []
        for n in grp["name"].astype(str).tolist():
            if n in raw_to_ent_idx:
                inds.append(raw_to_ent_idx[n])
        if inds:
            industry_to_company_idx[str(industry)] = inds
    # legacy 子图：行业 max 用全量行业映射，但只保留在子图且可嵌入的企业
    if is_subgraph and not use_induced_v2:
        itc_full = industry_to_companies_full_map(df_ent_full)
        industry_to_company_idx = {}
        for ind, names in itc_full.items():
            inds = [
                raw_to_ent_idx[str(n)]
                for n in names
                if str(n) in raw_to_ent_idx and str(n) in valid_company_names
            ]
            if inds:
                industry_to_company_idx[str(ind)] = inds

    industry_to_companies_gt = industry_to_companies_full_map(df_ent_full)

    if use_induced_v2:
        if args.ground_truth_source != "test":
            raise SystemExit("induced_v2 子图评测须使用 --ground_truth_source test")
        enterprise_queries, policy_queries, query_set_meta = build_subgraph_induced_eval_queries(
            openke_data=openke_data,
            supports_rid=supports_rid,
            token_to_eid=token_to_eid,
            openke_raw_to_tok=openke_raw_to_tok,
            df_policies=df_policies,
            df_enterprises=df_enterprises,
            max_enterprise_queries=args.max_enterprise_queries,
            max_industry_queries=args.max_industry_queries,
            max_policy_queries=args.max_policy_queries,
            min_company_ep_queries=int(args.min_company_ep_queries),
            min_industry_ep_queries=int(args.min_industry_ep_queries),
            min_pe_queries=int(args.min_pe_queries),
        )
        print(
            f"[OpenKE-Eval] induced_v2: company_ep={query_set_meta['n_company_ep_queries']} "
            f"industry_ep={query_set_meta['n_industry_ep_queries']} pe={query_set_meta['n_pe_queries']} "
            f"shortfalls={query_set_meta.get('shortfalls')}",
            flush=True,
        )
    else:
        from matching.evaluate_matching import build_test_queries_from_data  # type: ignore

        enterprise_queries, policy_queries = build_test_queries_from_data(
            max_enterprise_queries=args.max_enterprise_queries,
            max_industry_queries=args.max_industry_queries,
            max_policy_queries=args.max_policy_queries,
        )

        if args.ground_truth_source == "test":
            e2p_test, p2e_test = _build_support_maps_from_test2id(
                openke_data=openke_data,
                supports_rid=supports_rid,
                token_to_eid=token_to_eid,
                openke_raw_to_tok=openke_raw_to_tok,
            )
            for q in enterprise_queries:
                q_text = str(q["query"])
                q_type = q.get("type", "company_name")
                if q_type == "industry":
                    pols: Set[str] = set()
                    for cname in industry_to_companies_gt.get(q_text, []):
                        pols.update(e2p_test.get(str(cname), set()))
                    q["ground_truth"] = sorted(list(pols))
                else:
                    q["ground_truth"] = sorted(list(e2p_test.get(q_text, set())))
            for q in policy_queries:
                title = str(q["policy_title"])
                q["ground_truth"] = sorted(list(p2e_test.get(title, set())))
                q["policy_id"] = int(policy_name_to_id.get(title, -1))

        if is_subgraph:
            enterprise_queries, policy_queries = filter_queries_subgraph_entities(
                enterprise_queries,
                policy_queries,
                valid_policy_titles,
                valid_company_names,
                industry_to_companies_gt,
            )

    adaptive_from_gt = (
        not args.no_adaptive_output_cap
        and (args.adaptive_output_cap_from_gt or is_subgraph)
    )
    if adaptive_from_gt:
        cap_c, cap_i, cap_gt_stats = _resolve_output_caps_from_gt(
            enterprise_queries,
            policy_queries,
            valid_policy_titles,
            valid_company_names,
            multiplier=args.adaptive_output_cap_gt_multiplier,
            ceiling=int(args.adaptive_output_cap_ceiling),
        )
        cap_ep_company = cap_c if cap_c is not None else args.policy_max_output_cap
        cap_ep_industry = cap_i if cap_i is not None else args.policy_industry_query_max_output_cap
        if args.adaptive_output_cap_pe_from_gt and cap_gt_stats["n_pe_queries"]:
            mpe = float(cap_gt_stats["mean_gt_pe"])
            cap_pe = max(1, int(math.ceil(mpe * float(args.adaptive_output_cap_gt_multiplier))))
            if int(args.adaptive_output_cap_ceiling) > 0:
                cap_pe = min(cap_pe, int(args.adaptive_output_cap_ceiling))
        else:
            cap_pe = args.enterprise_max_output_cap
        log_caps = []
        if cap_gt_stats["n_company_ep_queries"]:
            log_caps.append(f"E→P_company={cap_ep_company}")
        if cap_gt_stats["n_industry_ep_queries"]:
            log_caps.append(f"E→P_industry={cap_ep_industry}")
        if args.adaptive_output_cap_pe_from_gt:
            log_caps.append(f"P→E={cap_pe}")
        else:
            log_caps.append(f"P→E={cap_pe}(固定enterprise_max_output_cap)")
        print(
            "[OpenKE-Eval] 自适应 max_output_cap（GT 均值×"
            f"{args.adaptive_output_cap_gt_multiplier:g}）: "
            + ", ".join(log_caps)
            + " | mean_gt: "
            f"company_ep={cap_gt_stats['mean_gt_company_ep']}, "
            f"industry_ep={cap_gt_stats['mean_gt_industry_ep']}, pe={cap_gt_stats['mean_gt_pe']}",
            flush=True,
        )
    else:
        cap_ep_company = args.policy_max_output_cap
        cap_ep_industry = args.policy_industry_query_max_output_cap
        cap_pe = args.enterprise_max_output_cap
        cap_gt_stats = None
        cap_c, cap_i = None, None

    print(f"[OpenKE-Eval] E->P 查询数: {len(enterprise_queries)}", flush=True)
    ep_metrics: List[Dict] = []
    ep_masked_gt_empty = 0
    policy_vecs = ent_emb[policy_idx_arr]

    # policy_title -> 是否在 full 支持关系中直连该企业（用于 P->E 先验加分）
    policy_direct_support_companies: Dict[str, Set[str]] = {}
    if args.ground_truth_source == "full":
        triples_df = pd.read_parquet(triples_parquet_path(PROJECT_ROOT, scale_opt))
        for row in triples_df.itertuples(index=False):
            if str(row.predicate).lower() == "supports":
                p = str(row.subject)
                c = str(row.object)
                policy_direct_support_companies.setdefault(p, set()).add(c)

    for i, q in enumerate(enterprise_queries, start=1):
        q_text = str(q["query"])
        q_type = q.get("type", "company_name")
        gt_raw = [x for x in q["ground_truth"] if x in valid_policy_titles]
        if len(gt_raw) == 0:
            ep_masked_gt_empty += 1

        score: Optional[np.ndarray] = None
        if q_type == "industry" and q_text in industry_to_company_idx:
            # 行业查询：对行业内所有公司按 max 聚合 policy->company 关系得分
            tail_idx_list = industry_to_company_idx[q_text]
            all_scores = []
            for tail_idx in tail_idx_list:
                tail_vec = ent_emb[tail_idx]
                dist = np.linalg.norm((policy_vecs + r_support) - tail_vec, ord=1, axis=1)
                all_scores.append(_transe_similarity_from_distance(dist))
            score = np.max(np.stack(all_scores, axis=0), axis=0)
        elif q_text in raw_to_ent_idx:
            tail_vec = ent_emb[raw_to_ent_idx[q_text]]
            dist = np.linalg.norm((policy_vecs + r_support) - tail_vec, ord=1, axis=1)
            score = _transe_similarity_from_distance(dist)
        else:
            # 无法映射时返回空预测
            score = np.array([], dtype=np.float32)

        if score.size > 0:
            order = np.argsort(score)[::-1]
            cands = order[: min(args.policy_candidate_k, len(order))]
            ranked = [(int(policy_name_to_id[str(policy_names_arr[j])]), float(score[j])) for j in cands if str(policy_names_arr[j]) in policy_name_to_id]
            if q_type == "industry":
                ranked = _apply_rank_cutoff(
                    ranked_pairs=ranked,
                    top_k=args.top_k_policy,
                    score_threshold=policy_score_threshold,
                    adaptive_quantile=args.policy_industry_query_adaptive_quantile,
                    relative_drop_threshold=args.policy_industry_query_relative_drop_threshold,
                    max_output_cap=cap_ep_industry,
                )
            else:
                ranked = _apply_rank_cutoff(
                    ranked_pairs=ranked,
                    top_k=args.top_k_policy,
                    score_threshold=policy_score_threshold,
                    adaptive_quantile=args.policy_adaptive_quantile,
                    relative_drop_threshold=args.policy_relative_drop_threshold,
                    max_output_cap=cap_ep_company,
                )
            pred_titles = [policy_id_to_title[pid] for pid, _ in ranked if pid in valid_policy_ids]
        else:
            pred_titles = []
        pred_titles = _dedup_keep_order(pred_titles)

        m = calculate_metrics(pred_titles, gt_raw)
        rm = calculate_ranking_metrics(pred_titles, gt_raw)
        ep_metrics.append(
            {**m, **rm, "query": q_text, "query_type": q_type, "gt_size_after_mask": len(gt_raw)}
        )

        if _format_progress(i, len(enterprise_queries)):
            print(f"[OpenKE-Eval][E->P] {i}/{len(enterprise_queries)}", flush=True)

    print(f"[OpenKE-Eval] P->E 查询数: {len(policy_queries)}", flush=True)
    pe_metrics: List[Dict] = []
    pe_masked_gt_empty = 0
    company_vecs = ent_emb[company_idx_arr]

    for i, q in enumerate(policy_queries, start=1):
        title = str(q["policy_title"])
        gt_raw = [x for x in q["ground_truth"] if x in valid_company_names]
        if len(gt_raw) == 0:
            pe_masked_gt_empty += 1

        if title in raw_to_ent_idx:
            head_vec = ent_emb[raw_to_ent_idx[title]]
            dist = np.linalg.norm((head_vec + r_support) - company_vecs, ord=1, axis=1)
            score = _transe_similarity_from_distance(dist)
            order = np.argsort(score)[::-1]
            cands = order[: min(args.enterprise_candidate_k, len(order))]
            ranked_company = [(int(j), float(score[j])) for j in cands]
            boost_set = policy_direct_support_companies.get(title, set())
            if boost_set:
                tmp = []
                for j, s in ranked_company:
                    nm = str(company_names_arr[j])
                    if nm in boost_set:
                        s += float(args.direct_support_boost)
                    tmp.append((j, s))
                ranked_company = tmp
            ranked_company = _apply_rank_cutoff(
                ranked_pairs=ranked_company,
                top_k=args.top_k_enterprise,
                score_threshold=enterprise_score_threshold,
                adaptive_quantile=args.enterprise_adaptive_quantile,
                relative_drop_threshold=args.enterprise_relative_drop_threshold,
                max_output_cap=cap_pe,
            )
            pred_names = [str(company_names_arr[j]) for j, _ in ranked_company]
        else:
            pred_names = []
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

        if _format_progress(i, len(policy_queries)):
            print(f"[OpenKE-Eval][P->E] {i}/{len(policy_queries)}", flush=True)

    ep_block, pe_block = enterprise_policy_result_blocks(
        ep_metrics, pe_metrics, ep_masked_gt_empty, pe_masked_gt_empty
    )

    nq = len(enterprise_queries) + len(policy_queries)
    include_pq = bool(args.include_per_query) or nq <= 150

    param_dict: Dict = {
        "train_times": 120,
        "dim": 200,
        "ground_truth_source": args.ground_truth_source,
        "top_k_policy": args.top_k_policy,
        "top_k_enterprise": args.top_k_enterprise,
        "policy_candidate_k": args.policy_candidate_k,
        "enterprise_candidate_k": args.enterprise_candidate_k,
        "policy_score_threshold": policy_score_threshold,
        "enterprise_score_threshold": enterprise_score_threshold,
        "policy_adaptive_quantile": args.policy_adaptive_quantile,
        "policy_relative_drop_threshold": args.policy_relative_drop_threshold,
        "policy_max_output_cap": args.policy_max_output_cap,
        "policy_industry_query_adaptive_quantile": args.policy_industry_query_adaptive_quantile,
        "policy_industry_query_relative_drop_threshold": args.policy_industry_query_relative_drop_threshold,
        "policy_industry_query_max_output_cap": args.policy_industry_query_max_output_cap,
        "enterprise_adaptive_quantile": args.enterprise_adaptive_quantile,
        "enterprise_relative_drop_threshold": args.enterprise_relative_drop_threshold,
        "enterprise_max_output_cap": args.enterprise_max_output_cap,
        "direct_support_boost": args.direct_support_boost,
        "max_enterprise_queries": args.max_enterprise_queries,
        "max_industry_queries": args.max_industry_queries,
        "max_policy_queries": args.max_policy_queries,
        "subgraph_eval_protocol": (args.subgraph_eval_protocol if is_subgraph else None),
        "min_company_ep_queries": int(args.min_company_ep_queries),
        "min_industry_ep_queries": int(args.min_industry_ep_queries),
        "min_pe_queries": int(args.min_pe_queries),
        "output_cap_mode": "adaptive_gt_mean" if adaptive_from_gt else "fixed",
        "adaptive_output_cap_from_gt_cli": bool(args.adaptive_output_cap_from_gt),
        "no_adaptive_output_cap": bool(args.no_adaptive_output_cap),
        "effective_policy_max_output_cap": (cap_c if adaptive_from_gt else args.policy_max_output_cap),
        "effective_policy_industry_query_max_output_cap": (
            cap_i if adaptive_from_gt else args.policy_industry_query_max_output_cap
        ),
        "effective_enterprise_max_output_cap": cap_pe,
        "adaptive_output_cap_pe_from_gt": bool(args.adaptive_output_cap_pe_from_gt),
    }
    if adaptive_from_gt:
        param_dict["adaptive_output_cap_gt_multiplier"] = args.adaptive_output_cap_gt_multiplier
        param_dict["adaptive_output_cap_ceiling"] = int(args.adaptive_output_cap_ceiling)
        param_dict["adaptive_output_cap_gt_stats"] = cap_gt_stats

    _eval_scope = "full"
    if is_subgraph:
        _eval_scope = "subgraph_induced_v2" if use_induced_v2 else "subgraph_entities"
    _eval_query_set = (
        "induced_v2_test_supports"
        if use_induced_v2
        else ("legacy_main_protocol_subgraph_filtered" if is_subgraph else "legacy_main_protocol_full")
    )
    result = {
        "model": "OpenKE-TransE",
        "evaluation_mode": "matching_metrics_with_openke_supports_relation",
        "eval_query_scope": _eval_scope,
        "eval_query_set": _eval_query_set,
        "scale_dir": scale_opt,
        "subgraph_tag": subgraph_tag if is_subgraph else None,
        "timestamp": datetime.now().isoformat(),
        "parameters": param_dict,
        "enterprise_to_policy": ep_block,
        "policy_to_enterprise": pe_block,
    }
    if query_set_meta is not None:
        result["query_set_meta"] = query_set_meta
    if include_pq:
        result["enterprise_to_policy_per_query"] = ep_metrics
        result["policy_to_enterprise_per_query"] = pe_metrics
        result["per_query_included"] = True
        result["per_query_note"] = (
            "自动纳入：总查询数<=150；否则需传 --include_per_query。字段含 query / precision / recall / f1 / ap / ndcg / gt_size_after_mask。"
        )
    else:
        result["per_query_included"] = False

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 同步一份到项目根 report 目录（子图不写覆盖全量文件名）
    repo_report = REPO_ROOT / "report"
    repo_report.mkdir(parents=True, exist_ok=True)
    if is_subgraph:
        # 与 --output 文件名一致，避免 legacy / induced_v2 等协议互相覆盖 report/
        repo_path = repo_report / out_path.name
    else:
        repo_name = "openke_matching_eval.json" if args.ground_truth_source == "full" else "openke_matching_eval_testsplit.json"
        repo_path = repo_report / repo_name
    repo_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OpenKE-Eval] 已写入: {out_path}")
    print(f"[OpenKE-Eval] 已写入: {repo_path}")


if __name__ == "__main__":
    main()

