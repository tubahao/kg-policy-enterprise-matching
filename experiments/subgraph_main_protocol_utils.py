#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""行业子图上的主协议评测共用：读子图 parquet、过滤查询、双宏平均。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# 子图评测协议：legacy=主协议查询 + subgraph_entities 过滤；induced_v2=仅子图内实体从 test supports 构造查询集
SUBGRAPH_EVAL_PROTOCOL_LEGACY = "legacy_filter"
SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2 = "induced_v2"


def read_policy_enterprise_tables(
    project_root: Path, scale_dir: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, bool, str]:
    """
    Returns:
        df_policies, df_enterprises, is_subgraph, tag (e.g. nodes_0_10 or "full")
    """
    if scale_dir and str(scale_dir).strip():
        sd = (project_root / scale_dir).resolve()
        pol_p = sd / "policies_clean.parquet"
        ent_p = sd / "enterprises_filtered.parquet"
        if not pol_p.is_file() or not ent_p.is_file():
            raise FileNotFoundError(f"子图目录缺少 parquet: {sd}")
        df_pol = pd.read_parquet(pol_p)
        df_ent = pd.read_parquet(ent_p)
        return df_pol, df_ent, True, sd.name
    mid = project_root / "data_intermediate"
    df_pol = pd.read_parquet(mid / "policies_clean.parquet")
    df_ent = pd.read_parquet(mid / "enterprises_filtered.parquet")
    return df_pol, df_ent, False, "full"


def read_enterprises_full(project_root: Path) -> pd.DataFrame:
    return pd.read_parquet(project_root / "data_intermediate" / "enterprises_filtered.parquet")


def industry_to_companies_full_map(df_ent_full: pd.DataFrame) -> Dict[str, List[str]]:
    if "industry" not in df_ent_full.columns:
        return {}
    return df_ent_full.groupby("industry")["name"].apply(list).to_dict()


def filter_queries_subgraph_entities(
    enterprise_queries: List[dict],
    policy_queries: List[dict],
    valid_policy_titles: Set[str],
    valid_company_names: Set[str],
    industry_to_companies_full: Dict[str, List[str]],
) -> Tuple[List[dict], List[dict]]:
    def keep_ent(q: dict) -> bool:
        qt = str(q["query"])
        if q.get("type", "company_name") == "industry":
            return any(str(n) in valid_company_names for n in industry_to_companies_full.get(qt, []))
        return qt in valid_company_names

    eq = [q for q in enterprise_queries if keep_ent(q)]
    pq = [q for q in policy_queries if str(q["policy_title"]) in valid_policy_titles]
    return eq, pq


def _avg_block(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "map": 0.0, "ndcg": 0.0}
    return {
        "precision": float(np.mean([x["precision"] for x in rows])),
        "recall": float(np.mean([x["recall"] for x in rows])),
        "f1": float(np.mean([x["f1"] for x in rows])),
        "map": float(np.mean([x["ap"] for x in rows])),
        "ndcg": float(np.mean([x["ndcg"] for x in rows])),
    }


def enterprise_policy_result_blocks(
    ep_metrics: List[Dict[str, Any]],
    pe_metrics: List[Dict[str, Any]],
    ep_masked: int,
    pe_masked: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """与 data_scale_gat_matching_eval 一致：average + average_gt_nonempty_only。"""
    ep_avg = {**_avg_block(ep_metrics), "masked_gt_empty": int(ep_masked)}
    ep_nonempty = [x for x in ep_metrics if int(x.get("gt_size_after_mask", 0)) > 0]
    ep_avg_ne = {**_avg_block(ep_nonempty), "num_queries": len(ep_nonempty)}

    pe_avg = {**_avg_block(pe_metrics), "masked_gt_empty": int(pe_masked)}
    pe_nonempty = [x for x in pe_metrics if int(x.get("gt_size_after_mask", 0)) > 0]
    pe_avg_ne = {**_avg_block(pe_nonempty), "num_queries": len(pe_nonempty)}

    ep_block = {
        "num_queries": len(ep_metrics),
        "average": ep_avg,
        "average_gt_nonempty_only": ep_avg_ne,
    }
    pe_block = {
        "num_queries": len(pe_metrics),
        "average": pe_avg,
        "average_gt_nonempty_only": pe_avg_ne,
    }
    return ep_block, pe_block


def triples_parquet_path(project_root: Path, scale_dir: Optional[str]) -> Path:
    if scale_dir and str(scale_dir).strip():
        return (project_root / scale_dir).resolve() / "triples_policy_entity.parquet"
    return project_root / "data_intermediate" / "triples_policy_entity.parquet"


def _read_openke_triples_file(path: Path) -> List[Tuple[int, int, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[Tuple[int, int, int]] = []
    for ln in lines[1:]:
        parts = ln.strip().split()
        if len(parts) != 3:
            continue
        out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return out


def build_support_maps_from_test2id(
    openke_data: Path,
    supports_rid: int,
    token_to_eid: Dict[str, int],
    openke_raw_to_tok: Dict[str, str],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """OpenKE test2id 中 supports 三元组 -> 企业->政策 / 政策->企业（原始实体名字符串）。"""
    id_to_token_entity = {v: k for k, v in token_to_eid.items()}
    token_to_raw_entity = {v: k for k, v in openke_raw_to_tok.items()}
    test_triples = _read_openke_triples_file(openke_data / "test2id.txt")
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


def build_subgraph_induced_eval_queries(
    *,
    openke_data: Path,
    supports_rid: int,
    token_to_eid: Dict[str, int],
    openke_raw_to_tok: Dict[str, str],
    df_policies: pd.DataFrame,
    df_enterprises: pd.DataFrame,
    max_enterprise_queries: int = 300,
    max_industry_queries: int = 30,
    max_policy_queries: int = 200,
    min_company_ep_queries: int = 40,
    min_industry_ep_queries: int = 12,
    min_pe_queries: int = 25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    子图诱导评测集：查询与 GT 均只来自子图内实体；E→P 的 GT 为 test supports 与当前子图政策的交集；
    行业题 GT 为子图内该行业企业 test supports 政策并集后再 mask 到子图政策。
    """
    e2p_test, p2e_test = build_support_maps_from_test2id(
        openke_data=openke_data,
        supports_rid=supports_rid,
        token_to_eid=token_to_eid,
        openke_raw_to_tok=openke_raw_to_tok,
    )
    valid_policy_titles: Set[str] = {str(x) for x in df_policies["title"].astype(str).tolist()}
    valid_company_names: Set[str] = {str(x) for x in df_enterprises["name"].astype(str).tolist()}
    policy_name_to_id = {
        str(t): int(pid) for t, pid in zip(df_policies["title"].astype(str), df_policies["policy_id"].astype(int))
    }

    company_pool: List[Tuple[str, Set[str]]] = []
    for name in valid_company_names:
        pols = {p for p in e2p_test.get(str(name), set()) if p in valid_policy_titles}
        if pols:
            company_pool.append((str(name), pols))
    company_pool.sort(key=lambda x: len(x[1]), reverse=True)
    pool_n_company = len(company_pool)
    n_company_take = min(max(0, max_enterprise_queries), pool_n_company)
    company_sel = company_pool[:n_company_take]
    enterprise_queries: List[Dict[str, Any]] = [
        {"query": n, "type": "company_name", "ground_truth": sorted(list(pols))} for n, pols in company_sel
    ]

    industry_pool: List[Tuple[str, Set[str], int]] = []
    if "industry" in df_enterprises.columns:
        for ind, grp in df_enterprises.groupby("industry"):
            ind_s = str(ind)
            pols: Set[str] = set()
            for cname in grp["name"].astype(str).tolist():
                for p in e2p_test.get(str(cname), set()):
                    if p in valid_policy_titles:
                        pols.add(p)
            if pols:
                industry_pool.append((ind_s, pols, int(len(grp))))
    industry_pool.sort(key=lambda x: len(x[1]), reverse=True)
    pool_n_industry = len(industry_pool)
    n_industry_take = min(max(0, max_industry_queries), pool_n_industry)
    for ind_s, pols, _ in industry_pool[:n_industry_take]:
        enterprise_queries.append(
            {"query": ind_s, "type": "industry", "ground_truth": sorted(list(pols))}
        )

    pe_pool: List[Tuple[str, Set[str]]] = []
    for title in valid_policy_titles:
        ents = {e for e in p2e_test.get(str(title), set()) if e in valid_company_names}
        if ents:
            pe_pool.append((str(title), ents))
    pe_pool.sort(key=lambda x: len(x[1]), reverse=True)
    pool_n_pe = len(pe_pool)
    n_pe_take = min(max(0, max_policy_queries), pool_n_pe)
    policy_queries: List[Dict[str, Any]] = [
        {
            "policy_id": int(policy_name_to_id.get(title, -1)),
            "policy_title": title,
            "ground_truth": sorted(list(ents)),
        }
        for title, ents in pe_pool[:n_pe_take]
    ]

    n_co = n_company_take
    n_ind = n_industry_take
    shortfalls: List[Dict[str, Any]] = []
    if n_co < min_company_ep_queries:
        shortfalls.append(
            {
                "axis": "company_ep",
                "selected": n_co,
                "min_target": min_company_ep_queries,
                "pool": pool_n_company,
            }
        )
    if n_ind < min_industry_ep_queries:
        shortfalls.append(
            {
                "axis": "industry_ep",
                "selected": n_ind,
                "min_target": min_industry_ep_queries,
                "pool": pool_n_industry,
            }
        )
    if n_pe_take < min_pe_queries:
        shortfalls.append(
            {
                "axis": "policy_to_enterprise",
                "selected": n_pe_take,
                "min_target": min_pe_queries,
                "pool": pool_n_pe,
            }
        )

    meta: Dict[str, Any] = {
        "protocol": SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2,
        "max_enterprise_queries": max_enterprise_queries,
        "max_industry_queries": max_industry_queries,
        "max_policy_queries": max_policy_queries,
        "min_company_ep_queries": min_company_ep_queries,
        "min_industry_ep_queries": min_industry_ep_queries,
        "min_pe_queries": min_pe_queries,
        "pool_n_company_ep": pool_n_company,
        "pool_n_industry_ep": pool_n_industry,
        "pool_n_pe": pool_n_pe,
        "n_company_ep_queries": n_co,
        "n_industry_ep_queries": n_ind,
        "n_pe_queries": n_pe_take,
        "shortfalls": shortfalls,
    }
    return enterprise_queries, policy_queries, meta
