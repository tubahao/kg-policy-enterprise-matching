#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实纯文本检索基线（Naive/Vector RAG）：
- 不使用图结构、GNN、PPR、衰减
- 仅使用文本向量相似度检索
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from matching.evaluate_matching import (  # type: ignore
    build_test_queries_from_data,
    calculate_metrics,
    calculate_ranking_metrics,
)


def _avg_metrics(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "map": 0.0, "ndcg": 0.0}
    keys = ["precision", "recall", "f1", "map", "ndcg"]
    return {k: float(np.mean([x.get(k, 0.0) for x in items])) for k in keys}


def _rank_docs(query: str, vectorizer: TfidfVectorizer, mat, ids: List) -> List:
    q = vectorizer.transform([query])
    sim = cosine_similarity(q, mat).flatten()
    order = np.argsort(sim)[::-1]
    return [ids[i] for i in order if sim[i] > 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_enterprise_queries", type=int, default=300)
    parser.add_argument("--max_industry_queries", type=int, default=30)
    parser.add_argument("--max_policy_queries", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=200)
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    out_dir = project_root / "reports" / "real_comparison_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pol = pd.read_parquet(project_root / "data_intermediate" / "policies_clean.parquet")
    df_ent = pd.read_parquet(project_root / "data_intermediate" / "enterprises_filtered.parquet")

    policy_titles = df_pol["title"].astype(str).tolist()
    policy_ids = df_pol["policy_id"].astype(int).tolist()
    policy_texts = (df_pol["title"].astype(str) + " " + df_pol["content"].fillna("").astype(str)).tolist()

    ent_names = df_ent["name"].astype(str).tolist()
    ent_texts = (
        df_ent["name"].astype(str)
        + " "
        + df_ent["industry"].fillna("").astype(str)
        + " "
        + df_ent["text_with_industry"].fillna("").astype(str)
    ).tolist()

    policy_vec = TfidfVectorizer(max_features=50000)
    policy_mat = policy_vec.fit_transform(policy_texts)
    ent_vec = TfidfVectorizer(max_features=50000)
    ent_mat = ent_vec.fit_transform(ent_texts)

    enterprise_queries, policy_queries = build_test_queries_from_data(
        max_enterprise_queries=args.max_enterprise_queries,
        max_industry_queries=args.max_industry_queries,
        max_policy_queries=args.max_policy_queries,
    )

    ep_results: List[Dict[str, float]] = []
    for q in enterprise_queries:
        ranked_titles = _rank_docs(q["query"], policy_vec, policy_mat, policy_titles)[: args.top_k]
        m = calculate_metrics(ranked_titles, q["ground_truth"])
        r = calculate_ranking_metrics(ranked_titles, q["ground_truth"])
        ep_results.append({**m, **r})

    pid_to_title = {int(r["policy_id"]): str(r["title"]) for _, r in df_pol.iterrows()}
    pe_results: List[Dict[str, float]] = []
    for q in policy_queries:
        pid = int(q["policy_id"])
        qtext = pid_to_title.get(pid, "")
        ranked_names = _rank_docs(qtext, ent_vec, ent_mat, ent_names)[: args.top_k]
        m = calculate_metrics(ranked_names, q["ground_truth"])
        r = calculate_ranking_metrics(ranked_names, q["ground_truth"])
        pe_results.append({**m, **r})

    result = {
        "model": "Text-Only-VectorRAG",
        "settings": {
            "vectorizer": "TF-IDF",
            "top_k": args.top_k,
            "max_enterprise_queries": args.max_enterprise_queries,
            "max_industry_queries": args.max_industry_queries,
            "max_policy_queries": args.max_policy_queries,
        },
        "enterprise_to_policy": {
            "num_queries": len(ep_results),
            "average": _avg_metrics(ep_results),
        },
        "policy_to_enterprise": {
            "num_queries": len(pe_results),
            "average": _avg_metrics(pe_results),
        },
    }
    out_json = out_dir / "text_rag_results.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

