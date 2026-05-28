#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATISE 时序嵌入在主实验口径下的双向匹配评测（与 real_comparison_openke_matching_eval 对齐）：
- 查询与截断：build_test_queries_from_data + 与 HippoRAG 一致的 E→P 自适应 cap
- GT：默认 test split（OpenKE test2id 中 supports）
- 指标：Precision / Recall / F1 / MAP(AP) / NDCG
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
ATISE_SRC = REPO_ROOT / "atise" / "ATISE"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(ATISE_SRC))

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

from matching.evaluate_matching import (  # type: ignore
    build_test_queries_from_data,
    calculate_metrics,
    calculate_ranking_metrics,
)

import model as KGE  # type: ignore
from Dataset import KnowledgeGraph  # type: ignore


def _read_id_map(path: Path) -> Dict[str, int]:
    mp: Dict[str, int] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return mp
    for ln in lines:
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


def _date_to_day(date_str: str, kg: KnowledgeGraph) -> float:
    end_sec = time.mktime(time.strptime(str(date_str).strip()[:10], "%Y-%m-%d"))
    return float((end_sec - kg.start_sec) / (kg.gran * 24 * 60 * 60))


def _atise_energy_to_sim(energy: np.ndarray) -> np.ndarray:
    """ATISE 能量越低越好，转为越大越好的相似度，与 OpenKE 脚本中 1/(1+dist) 同构。"""
    e = np.asarray(energy, dtype=np.float64)
    return 1.0 / (1.0 + np.maximum(e, 0.0))


def _batched_forward_scores(
    model: torch.nn.Module,
    h: np.ndarray,
    t: np.ndarray,
    r: int,
    d: np.ndarray,
    batch_size: int,
    use_cuda: bool,
) -> np.ndarray:
    n = h.shape[0]
    out_list: List[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            X = np.stack([h[s:e], t[s:e], np.full(e - s, r, dtype=np.int64), d[s:e]], axis=1)
            y = model.forward(X)
            if torch.is_tensor(y):
                y = y.detach().cpu().numpy()
            out_list.append(np.asarray(y).reshape(-1))
    return np.concatenate(out_list, axis=0)


def _find_latest_params(data_dir: Path) -> Path:
    cands = sorted(data_dir.rglob("params.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"未在 {data_dir} 下找到 params.pkl，请先完成 ATISE 训练")
    return cands[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k_policy", type=int, default=-1)
    parser.add_argument("--top_k_enterprise", type=int, default=-1)
    parser.add_argument("--policy_candidate_k", type=int, default=1000)
    parser.add_argument("--enterprise_candidate_k", type=int, default=1000)
    parser.add_argument("--policy_score_threshold", type=float, default=-1.0)
    parser.add_argument("--enterprise_score_threshold", type=float, default=-1.0)
    parser.add_argument("--policy_adaptive_quantile", type=float, default=0.72)
    parser.add_argument("--policy_relative_drop_threshold", type=float, default=0.15)
    parser.add_argument("--policy_max_output_cap", type=int, default=-1)
    parser.add_argument("--policy_industry_query_adaptive_quantile", type=float, default=0.82)
    parser.add_argument("--policy_industry_query_relative_drop_threshold", type=float, default=0.12)
    parser.add_argument("--policy_industry_query_max_output_cap", type=int, default=-1)
    parser.add_argument("--enterprise_adaptive_quantile", type=float, default=0.58)
    parser.add_argument("--enterprise_relative_drop_threshold", type=float, default=0.18)
    parser.add_argument("--enterprise_max_output_cap", type=int, default=150)
    parser.add_argument("--direct_support_boost", type=float, default=0.3)
    parser.add_argument("--max_enterprise_queries", type=int, default=300)
    parser.add_argument("--max_industry_queries", type=int, default=30)
    parser.add_argument("--max_policy_queries", type=int, default=200)
    parser.add_argument("--ground_truth_source", type=str, default="test", choices=["full", "test"])
    parser.add_argument("--checkpoint", type=str, default="", help="params.pkl 路径，默认取数据目录下最新")
    parser.add_argument("--dim", type=int, default=200, help="须与训练 ATISE 的 dim 一致")
    parser.add_argument("--batch_score", type=int, default=2048)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="reports/real_comparison_results/atise_matching_eval_testsplit.json",
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
        help="仅子图：legacy_filter 或 induced_v2（须 ground_truth_source=test）",
    )
    parser.add_argument("--min_company_ep_queries", type=int, default=40)
    parser.add_argument("--min_industry_ep_queries", type=int, default=12)
    parser.add_argument("--min_pe_queries", type=int, default=25)
    args = parser.parse_args()

    policy_score_threshold = None if args.policy_score_threshold < 0 else float(args.policy_score_threshold)
    enterprise_score_threshold = None if args.enterprise_score_threshold < 0 else float(args.enterprise_score_threshold)

    data_root = PROJECT_ROOT / "reports" / "real_comparison_data"
    atise_data = data_root / "atise_policykg"
    openke_data = data_root / "openke_policykg"
    result_root = PROJECT_ROOT / "reports" / "real_comparison_results"
    result_root.mkdir(parents=True, exist_ok=True)

    entity_token_map = json.loads((data_root / "entity_token_map.json").read_text(encoding="utf-8"))
    relation_token_map = json.loads((data_root / "relation_token_map.json").read_text(encoding="utf-8"))
    atise_raw_to_tok: Dict[str, str] = entity_token_map["atise"]
    supports_tok = relation_token_map["atise"]["supports"]

    token_to_eid = _read_id_map(atise_data / "entity2id.txt")
    token_to_rid = _read_id_map(atise_data / "relation2id.txt")
    if supports_tok not in token_to_rid:
        raise KeyError(f"relation2id 中不存在 supports token: {supports_tok}")
    supports_rid = token_to_rid[supports_tok]

    ckpt = Path(args.checkpoint) if args.checkpoint.strip() else _find_latest_params(atise_data)
    print(f"[ATISE-Eval] 使用 checkpoint: {ckpt}", flush=True)

    use_cuda = bool(args.cuda and torch.cuda.is_available())
    kg = KnowledgeGraph(data_dir=str(atise_data), gran=1, rev_set=1)
    model = KGE.ATISE(
        kg,
        embedding_dim=args.dim,
        batch_size=min(512, args.batch_score),
        learning_rate=1e-4,
        gamma=120.0,
        cmin=0.003,
        cmax=0.3,
        gpu=use_cuda,
    )
    state = torch.load(str(ckpt), map_location="cuda" if use_cuda else "cpu")
    model.load_state_dict(state)
    model.eval()

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

    policy_title_to_date = {
        str(r["title"]): str(r.get("publish_date", "2024-01-01"))[:10]
        for _, r in df_policies.iterrows()
    }

    raw_to_ent_idx: Dict[str, int] = {}
    for raw_name, tok in atise_raw_to_tok.items():
        if tok in token_to_eid:
            raw_to_ent_idx[raw_name] = token_to_eid[tok]

    policy_candidates: List[Tuple[str, int]] = [
        (title, raw_to_ent_idx[title]) for title in policy_titles if title in raw_to_ent_idx
    ]
    company_candidates: List[Tuple[str, int]] = [
        (name, raw_to_ent_idx[name]) for name in company_names if name in raw_to_ent_idx
    ]
    if not policy_candidates or not company_candidates:
        raise RuntimeError("无法构建政策/企业候选（检查 entity_token_map 与 atise entity2id）")

    policy_names_arr = np.array([x[0] for x in policy_candidates], dtype=object)
    policy_idx_arr = np.array([x[1] for x in policy_candidates], dtype=np.int64)
    company_names_arr = np.array([x[0] for x in company_candidates], dtype=object)
    company_idx_arr = np.array([x[1] for x in company_candidates], dtype=np.int64)

    openke_tok = entity_token_map["openke"]
    openke_ent2id = _read_id_map(openke_data / "entity2id.txt")
    openke_rel2id = _read_id_map(openke_data / "relation2id.txt")
    supports_tok_ok = relation_token_map["openke"]["supports"]
    supports_rid_ok = openke_rel2id[supports_tok_ok]

    use_induced_v2 = (
        is_subgraph
        and str(getattr(args, "subgraph_eval_protocol", SUBGRAPH_EVAL_PROTOCOL_LEGACY)).strip()
        == SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2
    )
    query_set_meta: Optional[Dict] = None

    industry_to_company_idx: Dict[str, List[int]] = {}
    for industry, grp in df_enterprises.groupby("industry"):
        inds = []
        for n in grp["name"].astype(str).tolist():
            if n in raw_to_ent_idx:
                inds.append(raw_to_ent_idx[n])
        if inds:
            industry_to_company_idx[str(industry)] = inds
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

    if use_induced_v2:
        if args.ground_truth_source != "test":
            raise SystemExit("induced_v2 须 --ground_truth_source test")
        enterprise_queries, policy_queries, query_set_meta = build_subgraph_induced_eval_queries(
            openke_data=openke_data,
            supports_rid=supports_rid_ok,
            token_to_eid=openke_ent2id,
            openke_raw_to_tok=openke_tok,
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
            f"[ATISE-Eval] induced_v2: company_ep={query_set_meta['n_company_ep_queries']} "
            f"industry_ep={query_set_meta['n_industry_ep_queries']} pe={query_set_meta['n_pe_queries']}",
            flush=True,
        )
    else:
        enterprise_queries, policy_queries = build_test_queries_from_data(
            max_enterprise_queries=args.max_enterprise_queries,
            max_industry_queries=args.max_industry_queries,
            max_policy_queries=args.max_policy_queries,
        )

        if args.ground_truth_source == "test":
            e2p_test, p2e_test = _build_support_maps_from_test2id(
                openke_data=openke_data,
                supports_rid=supports_rid_ok,
                token_to_eid=openke_ent2id,
                openke_raw_to_tok=openke_tok,
            )
            industry_to_companies = industry_to_companies_full_map(df_ent_full)
            for q in enterprise_queries:
                q_text = str(q["query"])
                q_type = q.get("type", "company_name")
                if q_type == "industry":
                    pols: Set[str] = set()
                    for cname in industry_to_companies.get(q_text, []):
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
                industry_to_companies_full_map(df_ent_full),
            )

    company_gt_sizes = [
        len(q["ground_truth"]) for q in enterprise_queries if q.get("type", "company_name") != "industry"
    ]
    industry_gt_sizes = [
        len(q["ground_truth"]) for q in enterprise_queries if q.get("type", "company_name") == "industry"
    ]
    inferred_policy_cap = max(1, int(round(float(np.mean(company_gt_sizes))))) if company_gt_sizes else 25
    inferred_policy_industry_cap = (
        max(1, int(round(float(np.mean(industry_gt_sizes))))) if industry_gt_sizes else inferred_policy_cap
    )
    policy_cap = args.policy_max_output_cap if args.policy_max_output_cap > 0 else inferred_policy_cap
    policy_industry_cap = (
        args.policy_industry_query_max_output_cap
        if args.policy_industry_query_max_output_cap > 0
        else inferred_policy_industry_cap
    )

    policy_direct_support_companies: Dict[str, Set[str]] = {}
    if args.ground_truth_source == "full":
        triples_df = pd.read_parquet(triples_parquet_path(PROJECT_ROOT, scale_opt))
        for row in triples_df.itertuples(index=False):
            if str(row.predicate).lower() == "supports":
                policy_direct_support_companies.setdefault(str(row.subject), set()).add(str(row.object))

    n_pol = len(policy_idx_arr)
    n_com = len(company_idx_arr)

    print(f"[ATISE-Eval] E->P 查询数: {len(enterprise_queries)}", flush=True)
    ep_metrics: List[Dict] = []
    ep_masked_gt_empty = 0

    for i, q in enumerate(enterprise_queries, start=1):
        q_text = str(q["query"])
        q_type = q.get("type", "company_name")
        gt_raw = [x for x in q["ground_truth"] if x in valid_policy_titles]
        if len(gt_raw) == 0:
            ep_masked_gt_empty += 1

        if q_type == "industry" and q_text in industry_to_company_idx:
            tail_idx_list = industry_to_company_idx[q_text]
            stacks = []
            for tail_idx in tail_idx_list:
                h_arr = policy_idx_arr.copy()
                t_arr = np.full(n_pol, tail_idx, dtype=np.int64)
                d_arr = np.array(
                    [_date_to_day(policy_title_to_date.get(str(policy_names_arr[j]), "2024-01-01"), kg) for j in range(n_pol)],
                    dtype=np.float64,
                )
                energy = _batched_forward_scores(
                    model, h_arr, t_arr, supports_rid, d_arr, args.batch_score, use_cuda
                )
                stacks.append(_atise_energy_to_sim(energy))
            score = np.max(np.stack(stacks, axis=0), axis=0)
        elif q_text in raw_to_ent_idx:
            tail_idx = raw_to_ent_idx[q_text]
            h_arr = policy_idx_arr.copy()
            t_arr = np.full(n_pol, tail_idx, dtype=np.int64)
            d_arr = np.array(
                [_date_to_day(policy_title_to_date.get(str(policy_names_arr[j]), "2024-01-01"), kg) for j in range(n_pol)],
                dtype=np.float64,
            )
            energy = _batched_forward_scores(
                model, h_arr, t_arr, supports_rid, d_arr, args.batch_score, use_cuda
            )
            score = _atise_energy_to_sim(energy)
        else:
            score = np.array([], dtype=np.float32)

        if score.size > 0:
            order = np.argsort(score)[::-1]
            cands = order[: min(args.policy_candidate_k, len(order))]
            ranked = [
                (int(policy_name_to_id[str(policy_names_arr[j])]), float(score[j]))
                for j in cands
                if str(policy_names_arr[j]) in policy_name_to_id
            ]
            if q_type == "industry":
                ranked = _apply_rank_cutoff(
                    ranked_pairs=ranked,
                    top_k=args.top_k_policy,
                    score_threshold=policy_score_threshold,
                    adaptive_quantile=args.policy_industry_query_adaptive_quantile,
                    relative_drop_threshold=args.policy_industry_query_relative_drop_threshold,
                    max_output_cap=policy_industry_cap,
                )
            else:
                ranked = _apply_rank_cutoff(
                    ranked_pairs=ranked,
                    top_k=args.top_k_policy,
                    score_threshold=policy_score_threshold,
                    adaptive_quantile=args.policy_adaptive_quantile,
                    relative_drop_threshold=args.policy_relative_drop_threshold,
                    max_output_cap=policy_cap,
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

        step = max(1, len(enterprise_queries) // 10)
        if i % step == 0 or i == len(enterprise_queries):
            print(f"[ATISE-Eval][E->P] {i}/{len(enterprise_queries)}", flush=True)

    print(f"[ATISE-Eval] P->E 查询数: {len(policy_queries)}", flush=True)
    pe_metrics: List[Dict] = []
    pe_masked_gt_empty = 0

    for i, q in enumerate(policy_queries, start=1):
        title = str(q["policy_title"])
        gt_raw = [x for x in q["ground_truth"] if x in valid_company_names]
        if len(gt_raw) == 0:
            pe_masked_gt_empty += 1

        if title in raw_to_ent_idx:
            head_idx = raw_to_ent_idx[title]
            d0 = _date_to_day(policy_title_to_date.get(title, "2024-01-01"), kg)
            h_arr = np.full(n_com, head_idx, dtype=np.int64)
            t_arr = company_idx_arr.copy()
            d_arr = np.full(n_com, d0, dtype=np.float64)
            energy = _batched_forward_scores(
                model, h_arr, t_arr, supports_rid, d_arr, args.batch_score, use_cuda
            )
            score = _atise_energy_to_sim(energy)
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
                max_output_cap=args.enterprise_max_output_cap,
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

        step = max(1, len(policy_queries) // 10)
        if i % step == 0 or i == len(policy_queries):
            print(f"[ATISE-Eval][P->E] {i}/{len(policy_queries)}", flush=True)

    ep_block, pe_block = enterprise_policy_result_blocks(
        ep_metrics, pe_metrics, ep_masked_gt_empty, pe_masked_gt_empty
    )

    _eval_scope = "full"
    if is_subgraph:
        _eval_scope = "subgraph_induced_v2" if use_induced_v2 else "subgraph_entities"
    _eval_query_set = (
        "induced_v2_test_supports"
        if use_induced_v2
        else ("legacy_main_protocol_subgraph_filtered" if is_subgraph else "legacy_main_protocol_full")
    )
    result = {
        "model": "ATISE",
        "evaluation_protocol": "main_queries + test_split_gt + unified_cutoff",
        "eval_query_scope": _eval_scope,
        "eval_query_set": _eval_query_set,
        "scale_dir": scale_opt,
        "subgraph_tag": subgraph_tag if is_subgraph else None,
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(ckpt),
        "parameters": {
            "dim": args.dim,
            "ground_truth_source": args.ground_truth_source,
            "subgraph_eval_protocol": (args.subgraph_eval_protocol if is_subgraph else None),
            "min_company_ep_queries": int(args.min_company_ep_queries),
            "min_industry_ep_queries": int(args.min_industry_ep_queries),
            "min_pe_queries": int(args.min_pe_queries),
            "policy_candidate_k": args.policy_candidate_k,
            "enterprise_candidate_k": args.enterprise_candidate_k,
            "policy_max_output_cap": policy_cap,
            "policy_industry_query_max_output_cap": policy_industry_cap,
            "enterprise_max_output_cap": args.enterprise_max_output_cap,
            "policy_max_output_cap_source": "arg" if args.policy_max_output_cap > 0 else "avg_gt_company",
            "policy_industry_query_max_output_cap_source": "arg"
            if args.policy_industry_query_max_output_cap > 0
            else "avg_gt_industry",
            "direct_support_boost": args.direct_support_boost if args.ground_truth_source == "full" else 0.0,
        },
        "enterprise_to_policy": ep_block,
        "policy_to_enterprise": pe_block,
    }
    if query_set_meta is not None:
        result["query_set_meta"] = query_set_meta

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    repo_report = REPO_ROOT / "report"
    repo_report.mkdir(parents=True, exist_ok=True)
    repo_name = out_path.name
    if is_subgraph and not repo_name.startswith("atise_matching_eval_subgraph_"):
        repo_name = f"atise_matching_eval_subgraph_{subgraph_tag}.json"
    (repo_report / repo_name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ATISE-Eval] 已写入: {out_path}", flush=True)


if __name__ == "__main__":
    main()
