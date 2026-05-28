#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主对比协议下的纯文本检索基线：Naive（TF-IDF + 余弦）与 Vector（DashScope Embedding + 余弦）。
与 KG-BERT / OpenKE 一致：build_test_queries_from_data、test split GT、统一截断与 matching 指标。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from matching.evaluate_matching import (  # type: ignore
    build_test_queries_from_data,
    calculate_metrics,
    calculate_ranking_metrics,
)

from subgraph_main_protocol_utils import (  # type: ignore
    SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2,
    SUBGRAPH_EVAL_PROTOCOL_LEGACY,
    build_subgraph_induced_eval_queries,
    enterprise_policy_result_blocks,
    filter_queries_subgraph_entities,
    industry_to_companies_full_map,
    read_enterprises_full,
    read_policy_enterprise_tables,
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


def _truncate(s: str, max_chars: int) -> str:
    s = s or ""
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars]


def _policy_doc(title: str, content: str, max_chars: int) -> str:
    return f"{title}\n{_truncate(content, max_chars)}"


def _enterprise_doc(name: str, text_with_industry: str, scope: str, max_chars: int) -> str:
    body = f"{name}\n{text_with_industry or ''}\n{scope or ''}"
    return _truncate(body, max_chars)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def _embed_batches_openai(
    texts: List[str],
    *,
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    chunks: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        for j, item in enumerate(resp.data):
            chunks.append(list(item.embedding))
        if (i // batch_size + 1) % 10 == 0:
            print(f"  [embed] {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.asarray(chunks, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["naive", "vector"], required=True)
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
    parser.add_argument("--max_enterprise_queries", type=int, default=300)
    parser.add_argument("--max_industry_queries", type=int, default=30)
    parser.add_argument("--max_policy_queries", type=int, default=200)
    parser.add_argument("--ground_truth_source", type=str, default="test", choices=["full", "test"])
    parser.add_argument("--policy_content_chars", type=int, default=2048)
    parser.add_argument("--enterprise_text_chars", type=int, default=2048)
    parser.add_argument("--tfidf_max_features", type=int, default=100_000)
    parser.add_argument("--embedding_model", type=str, default="text-embedding-v1")
    parser.add_argument(
        "--embedding_base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--embedding_batch_size", type=int, default=16)
    parser.add_argument("--api_key", type=str, default="", help="默认 DASHSCOPE_API_KEY / QWEN_API_KEY")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="默认按 mode 写入 reports/real_comparison_results/",
    )
    parser.add_argument(
        "--scale_dir",
        type=str,
        default="",
        help="子图目录（含 policies_clean / enterprises_filtered）；非空时候选仅限子图且 eval_query_scope=subgraph_entities",
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

    data_root = PROJECT_ROOT / "reports" / "real_comparison_data"
    openke_data = data_root / "openke_policykg"
    entity_token_map = json.loads((data_root / "entity_token_map.json").read_text(encoding="utf-8"))
    relation_token_map = json.loads((data_root / "relation_token_map.json").read_text(encoding="utf-8"))
    openke_ent2id = _read_id_map(openke_data / "entity2id.txt")
    openke_rel2id = _read_id_map(openke_data / "relation2id.txt")
    supports_rid_ok = openke_rel2id[relation_token_map["openke"]["supports"]]

    policy_score_threshold = None if args.policy_score_threshold < 0 else float(args.policy_score_threshold)
    enterprise_score_threshold = None if args.enterprise_score_threshold < 0 else float(args.enterprise_score_threshold)

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
    name_to_row = {str(r["name"]): r for _, r in df_enterprises.iterrows()}

    title_to_content: Dict[str, str] = {}
    for _, row in df_policies.iterrows():
        title_to_content[str(row["title"])] = str(row.get("content", "") or "")

    policy_docs: List[str] = [
        _policy_doc(t, title_to_content.get(t, ""), args.policy_content_chars) for t in policy_titles
    ]
    company_docs: List[str] = []
    for n in company_names:
        r = name_to_row.get(n)
        if r is None:
            company_docs.append(_enterprise_doc(n, "", "", args.enterprise_text_chars))
        else:
            company_docs.append(
                _enterprise_doc(
                    n,
                    str(r.get("text_with_industry", "") or ""),
                    str(r.get("scope", "") or ""),
                    args.enterprise_text_chars,
                )
            )

    industry_to_company_names: Dict[str, List[str]] = {}
    if "industry" in df_enterprises.columns:
        for ind, grp in df_enterprises.groupby("industry"):
            industry_to_company_names[str(ind)] = [str(x) for x in grp["name"].astype(str).tolist()]

    use_induced_v2 = (
        is_subgraph
        and str(getattr(args, "subgraph_eval_protocol", SUBGRAPH_EVAL_PROTOCOL_LEGACY)).strip()
        == SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2
    )
    query_set_meta: Optional[Dict] = None

    if use_induced_v2:
        if args.ground_truth_source != "test":
            raise SystemExit("induced_v2 须 --ground_truth_source test")
        enterprise_queries, policy_queries, query_set_meta = build_subgraph_induced_eval_queries(
            openke_data=openke_data,
            supports_rid=supports_rid_ok,
            token_to_eid=openke_ent2id,
            openke_raw_to_tok=entity_token_map["openke"],
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
            f"[TextRAG] induced_v2: company_ep={query_set_meta['n_company_ep_queries']} "
            f"industry_ep={query_set_meta['n_industry_ep_queries']} pe={query_set_meta['n_pe_queries']}",
            flush=True,
        )
    else:
        enterprise_queries, policy_queries = build_test_queries_from_data(
            max_enterprise_queries=args.max_enterprise_queries,
            max_industry_queries=args.max_industry_queries,
            max_policy_queries=args.max_policy_queries,
        )

        industry_to_companies = industry_to_companies_full_map(df_ent_full)
        if args.ground_truth_source == "test":
            e2p_test, p2e_test = _build_support_maps_from_test2id(
                openke_data=openke_data,
                supports_rid=supports_rid_ok,
                token_to_eid=openke_ent2id,
                openke_raw_to_tok=entity_token_map["openke"],
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

        if is_subgraph:
            enterprise_queries, policy_queries = filter_queries_subgraph_entities(
                enterprise_queries,
                policy_queries,
                valid_policy_titles,
                valid_company_names,
                industry_to_companies,
            )

    company_gt_sizes = [len(q["ground_truth"]) for q in enterprise_queries if q.get("type", "company_name") != "industry"]
    industry_gt_sizes = [len(q["ground_truth"]) for q in enterprise_queries if q.get("type", "company_name") == "industry"]
    inferred_policy_cap = max(1, int(round(float(np.mean(company_gt_sizes))))) if company_gt_sizes else 25
    inferred_ind_cap = max(1, int(round(float(np.mean(industry_gt_sizes))))) if industry_gt_sizes else inferred_policy_cap
    policy_cap = args.policy_max_output_cap if args.policy_max_output_cap > 0 else inferred_policy_cap
    policy_industry_cap = (
        args.policy_industry_query_max_output_cap if args.policy_industry_query_max_output_cap > 0 else inferred_ind_cap
    )

    title_to_idx = {t: i for i, t in enumerate(policy_titles)}
    name_to_idx = {n: i for i, n in enumerate(company_names)}

    if args.mode == "naive":
        corpus_for_vocab = policy_docs + company_docs
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 2),
            max_features=args.tfidf_max_features,
            min_df=1,
        )
        vectorizer.fit(corpus_for_vocab)
        X_pol = vectorizer.transform(policy_docs)
        X_com = vectorizer.transform(company_docs)
        model_label = "Naive-TFIDF"
    else:
        api_key = (args.api_key or "").strip() or os.getenv("DASHSCOPE_API_KEY", "") or os.getenv("QWEN_API_KEY", "")
        if not api_key:
            raise RuntimeError("Vector 模式需要 API Key：设置 DASHSCOPE_API_KEY 或使用 --api_key")
        print("[Vector] Embedding policies...", flush=True)
        E_pol = _embed_batches_openai(
            policy_docs,
            api_key=api_key,
            base_url=args.embedding_base_url,
            model=args.embedding_model,
            batch_size=args.embedding_batch_size,
        )
        print("[Vector] Embedding enterprises...", flush=True)
        E_com = _embed_batches_openai(
            company_docs,
            api_key=api_key,
            base_url=args.embedding_base_url,
            model=args.embedding_model,
            batch_size=args.embedding_batch_size,
        )
        E_pol = _l2_normalize(E_pol)
        E_com = _l2_normalize(E_com)
        model_label = f"VectorRAG-{args.embedding_model}"
        X_pol = E_pol
        X_com = E_com

    def scores_ep_company(company_name: str) -> np.ndarray:
        idx = name_to_idx.get(company_name)
        if idx is None:
            return np.array([])
        if args.mode == "naive":
            qv = vectorizer.transform([company_docs[idx]])
            return cosine_similarity(qv, X_pol)[0]
        return np.asarray(X_com[idx] @ X_pol.T, dtype=np.float32)

    def scores_pe_policy(policy_title: str) -> np.ndarray:
        idx = title_to_idx.get(policy_title)
        if idx is None:
            return np.array([])
        if args.mode == "naive":
            qv = vectorizer.transform([policy_docs[idx]])
            return cosine_similarity(qv, X_com)[0]
        return np.asarray(X_pol[idx] @ X_com.T, dtype=np.float32)

    print(f"[{model_label}] E->P queries={len(enterprise_queries)}", flush=True)
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
                s = scores_ep_company(cname)
                if s.size > 0:
                    stacks.append(s)
            score = np.max(np.stack(stacks, axis=0), axis=0) if stacks else np.array([])
        else:
            score = scores_ep_company(q_text)

        if score.size > 0:
            order = np.argsort(score)[::-1]
            cands = order[: min(args.policy_candidate_k, len(order))]
            ranked = [
                (int(policy_name_to_id[str(policy_titles[j])]), float(score[j]))
                for j in cands
                if str(policy_titles[j]) in policy_name_to_id
            ]
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
            pred_titles = [policy_id_to_title[pid] for pid, _ in ranked if pid in valid_policy_ids]
        else:
            pred_titles = []
        pred_titles = _dedup_keep_order(pred_titles)
        m = calculate_metrics(pred_titles, gt_raw)
        rm = calculate_ranking_metrics(pred_titles, gt_raw)
        ep_metrics.append(
            {**m, **rm, "query": q_text, "query_type": q_type, "gt_size_after_mask": len(gt_raw)}
        )
        if i % max(1, len(enterprise_queries) // 10) == 0 or i == len(enterprise_queries):
            print(f"[{model_label}][E->P] {i}/{len(enterprise_queries)}", flush=True)

    print(f"[{model_label}] P->E queries={len(policy_queries)}", flush=True)
    pe_metrics: List[Dict] = []
    pe_masked = 0

    for i, q in enumerate(policy_queries, start=1):
        title = str(q["policy_title"])
        gt_raw = [x for x in q["ground_truth"] if x in valid_company_names]
        if not gt_raw:
            pe_masked += 1
        score = scores_pe_policy(title)
        if score.size > 0:
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
            pred_names = [str(company_names[j]) for j, _ in ranked_c]
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
        if i % max(1, len(policy_queries) // 10) == 0 or i == len(policy_queries):
            print(f"[{model_label}][P->E] {i}/{len(policy_queries)}", flush=True)

    ep_block, pe_block = enterprise_policy_result_blocks(ep_metrics, pe_metrics, ep_masked, pe_masked)

    default_out = (
        "reports/real_comparison_results/naive_tfidf_matching_eval_testsplit.json"
        if args.mode == "naive"
        else "reports/real_comparison_results/vector_rag_matching_eval_testsplit.json"
    )
    if is_subgraph:
        suf = f"_subgraph_{subgraph_tag}"
        default_out = default_out.replace(".json", f"{suf}.json")
    out_path = PROJECT_ROOT / (args.output or default_out)

    _eval_scope = "full"
    if is_subgraph:
        _eval_scope = "subgraph_induced_v2" if use_induced_v2 else "subgraph_entities"
    _eval_query_set = (
        "induced_v2_test_supports"
        if use_induced_v2
        else ("legacy_main_protocol_subgraph_filtered" if is_subgraph else "legacy_main_protocol_full")
    )
    result = {
        "model": model_label,
        "mode": args.mode,
        "evaluation_protocol": "main_queries + test_split_gt + unified_cutoff",
        "eval_query_scope": _eval_scope,
        "eval_query_set": _eval_query_set,
        "scale_dir": str(scale_opt) if is_subgraph else None,
        "subgraph_tag": subgraph_tag if is_subgraph else None,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "ground_truth_source": args.ground_truth_source,
            "subgraph_eval_protocol": (args.subgraph_eval_protocol if is_subgraph else None),
            "min_company_ep_queries": int(args.min_company_ep_queries),
            "min_industry_ep_queries": int(args.min_industry_ep_queries),
            "min_pe_queries": int(args.min_pe_queries),
            "policy_content_chars": args.policy_content_chars,
            "enterprise_text_chars": args.enterprise_text_chars,
            "policy_max_output_cap": policy_cap,
            "policy_industry_query_max_output_cap": policy_industry_cap,
            "enterprise_max_output_cap": args.enterprise_max_output_cap,
            "embedding_model": args.embedding_model if args.mode == "vector" else None,
        },
        "enterprise_to_policy": ep_block,
        "policy_to_enterprise": pe_block,
    }
    if query_set_meta is not None:
        result["query_set_meta"] = query_set_meta

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPO_ROOT / "report" / out_path.name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
