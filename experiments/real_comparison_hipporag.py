#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HippoRAG 真实对比实验（主对比协议）
- 固定主实验查询集合（build_test_queries_from_data）
- 使用 OpenKE test split 的 supports 关系作为 GT（避免训练泄漏）
- 评测指标：Precision / Recall / F1 / MAP / NDCG

默认（与 LightRAG 主实验对齐）：DashScope 兼容 OpenAI 接口 — text-embedding-v1 + qwen-plus，
索引阶段仍使用预写 OpenIE 占位 JSON（不调 LLM 做三元组抽取）；检索阶段默认开启事实重排（DSPyFilter）。
本地 MiniLM 对照请加 --local_baseline。
**--scale_dir**：子图目录（与主协议其它基线一致）；缓存子目录带 `_subgraph_<tag>`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import multiprocessing
from hashlib import md5
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
HIPPO_SRC = REPO_ROOT / "HippoRAG" / "HippoRAG" / "src"
if not HIPPO_SRC.is_dir():
    _hip_alt = REPO_ROOT.parent / "HippoRAG" / "HippoRAG" / "src"
    if _hip_alt.is_dir():
        HIPPO_SRC = _hip_alt
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(HIPPO_SRC))


def _patch_importlib_metadata_torch_version() -> None:
    """部分 Anaconda 环境中 torch 的 dist-info 不完整，metadata.version('torch') 为 None，会导致 transformers 导入崩溃。"""
    import importlib.metadata as im

    _orig = im.version

    def _version(name: str) -> str:
        try:
            out = _orig(name)
        except im.PackageNotFoundError:
            out = None
        if out is None and name == "torch":
            import torch

            return str(torch.__version__)
        if out is None:
            raise im.PackageNotFoundError(name)
        return out

    im.version = _version  # type: ignore[method-assign]


_patch_importlib_metadata_torch_version()

from subgraph_main_protocol_utils import (  # type: ignore
    SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2,
    SUBGRAPH_EVAL_PROTOCOL_LEGACY,
    build_subgraph_induced_eval_queries,
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

HippoRAG = None  # lazy import for Windows multiprocessing safety


class _PassThroughRerankFilter:
    """关闭 HippoRAG 事实重排（DSPyFilter LLM）时的占位，与原版脚本逻辑一致。"""

    def __call__(self, query, candidate_items, candidate_indices, len_after_rerank=None):
        k = len_after_rerank if len_after_rerank is not None else len(candidate_indices)
        return candidate_indices[:k], candidate_items[:k], {"confidence": None}


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
    out: List[Tuple[int, int, int]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for ln in lines[1:]:
        p = ln.strip().split()
        if len(p) != 3:
            continue
        out.append((int(p[0]), int(p[1]), int(p[2])))
    return out


def _apply_rank_cutoff(
    ranked_pairs: List[Tuple[str, float]],
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    adaptive_quantile: Optional[float] = None,
    relative_drop_threshold: Optional[float] = None,
    max_output_cap: Optional[int] = None,
) -> List[Tuple[str, float]]:
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


def _build_test_gt_maps() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    data_root = PROJECT_ROOT / "reports" / "real_comparison_data"
    openke_data = data_root / "openke_policykg"
    relation_token_map = json.loads((data_root / "relation_token_map.json").read_text(encoding="utf-8"))
    entity_token_map = json.loads((data_root / "entity_token_map.json").read_text(encoding="utf-8"))
    openke_raw_to_tok: Dict[str, str] = entity_token_map["openke"]

    token_to_eid = _read_id_map(openke_data / "entity2id.txt")
    token_to_rid = _read_id_map(openke_data / "relation2id.txt")
    supports_tok = relation_token_map["openke"]["supports"]
    supports_rid = token_to_rid[supports_tok]

    id_to_token = {v: k for k, v in token_to_eid.items()}
    tok_to_raw = {v: k for k, v in openke_raw_to_tok.items()}
    test_triples = _read_openke_triples(openke_data / "test2id.txt")

    e2p: Dict[str, Set[str]] = {}
    p2e: Dict[str, Set[str]] = {}
    for h, t, r in test_triples:
        if r != supports_rid:
            continue
        ht = id_to_token.get(h)
        tt = id_to_token.get(t)
        if ht is None or tt is None:
            continue
        ph = tok_to_raw.get(ht)
        ce = tok_to_raw.get(tt)
        if ph is None or ce is None:
            continue
        e2p.setdefault(ce, set()).add(ph)
        p2e.setdefault(ph, set()).add(ce)
    return e2p, p2e


def _prepare_queries_with_test_gt(
    max_enterprise_queries: int,
    max_industry_queries: int,
    max_policy_queries: int,
) -> Tuple[List[Dict], List[Dict]]:
    enterprise_queries, policy_queries = build_test_queries_from_data(
        max_enterprise_queries=max_enterprise_queries,
        max_industry_queries=max_industry_queries,
        max_policy_queries=max_policy_queries,
    )
    e2p_test, p2e_test = _build_test_gt_maps()

    industry_to_companies = industry_to_companies_full_map(read_enterprises_full(PROJECT_ROOT))

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
    policy_name_to_id: Dict[str, int] = {}
    df_pol_full = pd.read_parquet(PROJECT_ROOT / "data_intermediate" / "policies_clean.parquet")
    for _, row in df_pol_full.iterrows():
        policy_name_to_id[str(row["title"])] = int(row["policy_id"])

    for q in policy_queries:
        title = str(q["policy_title"])
        q["ground_truth"] = sorted(list(p2e_test.get(title, set())))
        q["policy_id"] = int(policy_name_to_id.get(title, -1))
    return enterprise_queries, policy_queries


def _build_openie_docs(docs: List[str], titles: List[str]) -> List[Dict]:
    openie_docs: List[Dict] = []
    for doc, title in zip(docs, titles):
        chunk_id = "chunk-" + md5(doc.encode()).hexdigest()
        openie_docs.append(
            {
                "idx": chunk_id,
                "passage": doc,
                "extracted_entities": [title],
                "extracted_triples": [[title, "related_to", title]],
            }
        )
    return openie_docs


def _init_hipporag_with_cached_openie(
    save_dir: Path,
    docs: List[str],
    doc_titles: List[str],
    *,
    use_dashscope_api: bool,
    api_key: str,
    llm_model: str,
    llm_base_url: str,
    embedding_model: str,
    embedding_base_url: str,
    embedding_batch_size: int,
    enable_fact_rerank: bool,
    rerank_llm_name: str,
    rerank_base_url: str,
    reuse_cached_index: bool = False,
) -> HippoRAG:
    """
    初始化 HippoRAG：OpenIE 仍使用预写 JSON（占位三元组 title-related_to-title），索引阶段不调 LLM 做抽取。

    - use_dashscope_api=True（默认）：chunk/entity/fact 向量走 OpenAI 兼容 Embeddings API；LLM 走兼容 Chat API；
      与 LightRAG 主实验对齐（text-embedding-v1 + qwen-plus）。可选开启事实重排（DSPyFilter）。
    - use_dashscope_api=False：原「本地向量 + 假 LLM 端点」对照管线。
    """
    global HippoRAG
    if HippoRAG is None:
        from hipporag import HippoRAG as _HippoRAG  # type: ignore

        HippoRAG = _HippoRAG

    save_dir.mkdir(parents=True, exist_ok=True)
    _scratch = not bool(reuse_cached_index)

    if use_dashscope_api:
        llm_name = llm_model
        if not (api_key or "").strip():
            raise RuntimeError(
                "DashScope API 模式需要密钥：请设置环境变量 DASHSCOPE_API_KEY（或 QWEN_API_KEY）或传入 --api_key。"
            )
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        from hipporag.utils.config_utils import BaseConfig  # type: ignore

        cfg = BaseConfig()
        cfg.save_dir = str(save_dir)
        cfg.llm_name = llm_name
        cfg.llm_base_url = llm_base_url
        cfg.embedding_model_name = embedding_model
        cfg.embedding_base_url = embedding_base_url
        cfg.embedding_batch_size = int(embedding_batch_size)
        cfg.force_index_from_scratch = _scratch
        cfg.force_openie_from_scratch = False
        hipporag = HippoRAG(global_config=cfg)
        if not enable_fact_rerank:
            hipporag.rerank_filter = _PassThroughRerankFilter()
    else:
        # 本地对照：与历史脚本一致（OpenIE JSON 在下方统一写入）
        llm_name = rerank_llm_name if enable_fact_rerank else "gpt-4o-mini"
        if enable_fact_rerank:
            use_api_key = api_key.strip() or os.getenv("DASHSCOPE_API_KEY", "") or os.getenv("QWEN_API_KEY", "")
            if not use_api_key:
                raise RuntimeError("本地模式下开启事实重排时未提供 API Key，请设置 DASHSCOPE_API_KEY 或 --api_key。")
            os.environ["OPENAI_API_KEY"] = use_api_key
            hipporag = HippoRAG(
                save_dir=str(save_dir),
                llm_model_name=llm_name,
                llm_base_url=rerank_base_url,
                embedding_model_name="Transformers/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
        else:
            os.environ.setdefault("OPENAI_API_KEY", "sk-")
            hipporag = HippoRAG(
                save_dir=str(save_dir),
                llm_model_name=llm_name,
                llm_base_url="http://localhost:1/v1",
                embedding_model_name="Transformers/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
        hipporag.global_config.force_index_from_scratch = _scratch
        hipporag.global_config.force_openie_from_scratch = False
        if not enable_fact_rerank:
            hipporag.rerank_filter = _PassThroughRerankFilter()

    # OpenIE 文件名必须与 global_config.llm_name 一致（HippoRAG 内部拼接路径）
    llm_label = hipporag.global_config.llm_name.replace("/", "_")
    openie_path = save_dir / f"openie_results_ner_{llm_label}.json"
    openie_payload = {"docs": _build_openie_docs(docs, doc_titles), "avg_ent_chars": 0.0, "avg_ent_words": 0.0}
    openie_path.write_text(json.dumps(openie_payload, ensure_ascii=True), encoding="utf-8")

    hipporag.global_config.force_index_from_scratch = _scratch
    hipporag.global_config.force_openie_from_scratch = False

    hipporag.index(docs=docs)
    return hipporag


def main():
    multiprocessing.freeze_support()
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
    parser.add_argument("--max_enterprise_queries", type=int, default=300)
    parser.add_argument("--max_industry_queries", type=int, default=30)
    parser.add_argument("--max_policy_queries", type=int, default=200)
    parser.add_argument(
        "--local_baseline",
        action="store_true",
        help="使用本地 sentence-transformers 向量 + localhost 假 LLM（历史对照）；默认改为 DashScope Embedding+LLM",
    )
    parser.add_argument(
        "--enable_fact_rerank",
        action="store_true",
        help="仅在与 --local_baseline 联用时生效：事实重排走 Qwen API",
    )
    parser.add_argument(
        "--disable_fact_rerank",
        action="store_true",
        help="DashScope 默认管线中关闭 DSPy 事实重排（仍走 API 向量与检索）",
    )
    parser.add_argument("--api_key", type=str, default="", help="DashScope Key，默认读 DASHSCOPE_API_KEY / QWEN_API_KEY")
    parser.add_argument("--llm_model", type=str, default="qwen-plus")
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-v1",
        help="须含 text-embedding 以使用 HippoRAG OpenAIEmbeddingModel（DashScope 兼容）",
    )
    parser.add_argument(
        "--embedding_base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--embedding_batch_size", type=int, default=16)
    parser.add_argument("--rerank_llm_name", type=str, default="qwen-plus")
    parser.add_argument("--rerank_base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument(
        "--rerank_api_key",
        type=str,
        default="",
        help="已弃用：请统一使用 --api_key 或 DASHSCOPE_API_KEY",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/real_comparison_results/hipporag_results_qwen_api.json",
    )
    parser.add_argument(
        "--scale_dir",
        type=str,
        default="",
        help="子图目录（含 policies_clean / enterprises_filtered）；非空时 eval_query_scope=subgraph_entities，缓存目录与全量隔离",
    )
    parser.add_argument(
        "--subgraph_eval_protocol",
        type=str,
        default=SUBGRAPH_EVAL_PROTOCOL_LEGACY,
        choices=[SUBGRAPH_EVAL_PROTOCOL_LEGACY, SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2],
        help="仅子图：induced_v2 用 test supports 在子图内构造查询集",
    )
    parser.add_argument("--min_company_ep_queries", type=int, default=40)
    parser.add_argument("--min_industry_ep_queries", type=int, default=12)
    parser.add_argument("--min_pe_queries", type=int, default=25)
    parser.add_argument(
        "--reuse_cached_index",
        action="store_true",
        help="子图/全量复跑评测时：force_index_from_scratch=False，复用 save_dir 下已有向量与图；"
        "语料未变时不再批量调用 Embedding API（仍可能因检索走 LLM 产生少量调用）。",
    )
    args = parser.parse_args()

    use_api = not args.local_baseline
    api_key = (args.api_key or args.rerank_api_key or os.getenv("DASHSCOPE_API_KEY", "") or os.getenv("QWEN_API_KEY", "")).strip()
    if use_api:
        enable_fact_rerank = not args.disable_fact_rerank
    else:
        enable_fact_rerank = bool(args.enable_fact_rerank)

    if use_api and not api_key:
        print(
            "[HippoRAG] 未配置 DashScope API Key（需 DASHSCOPE_API_KEY / QWEN_API_KEY 或 --api_key）。"
            "若仅跑本地对照，请加 --local_baseline。",
            flush=True,
        )
        raise SystemExit(1)

    policy_score_threshold = None if args.policy_score_threshold < 0 else float(args.policy_score_threshold)
    enterprise_score_threshold = None if args.enterprise_score_threshold < 0 else float(args.enterprise_score_threshold)

    scale_opt = (getattr(args, "scale_dir", "") or "").strip() or None
    df_policies, df_enterprises, is_subgraph, subgraph_tag = read_policy_enterprise_tables(
        PROJECT_ROOT, scale_opt
    )
    policy_titles = [str(x) for x in df_policies["title"].astype(str).tolist()]
    company_names = [str(x) for x in df_enterprises["name"].astype(str).tolist()]

    policy_docs = [f"{t}\n{str(c)[:1200]}" for t, c in zip(policy_titles, df_policies["content"].fillna("").astype(str).tolist())]
    company_docs = [f"{n}\n{str(c)[:1200]}" for n, c in zip(company_names, df_enterprises["text_with_industry"].fillna("").astype(str).tolist())]

    _hip_sub = f"_subgraph_{subgraph_tag}" if is_subgraph else ""
    if use_api:
        pol_dir = PROJECT_ROOT / "reports" / "real_comparison_results" / f"hipporag_policy_cache_dashscope{_hip_sub}"
        com_dir = PROJECT_ROOT / "reports" / "real_comparison_results" / f"hipporag_company_cache_dashscope{_hip_sub}"
        print(
            f"[HippoRAG] DashScope API 管线：llm={args.llm_model} embed={args.embedding_model} fact_rerank={enable_fact_rerank}",
            flush=True,
        )
    else:
        pol_dir = PROJECT_ROOT / "reports" / "real_comparison_results" / f"hipporag_policy_cache{_hip_sub}"
        com_dir = PROJECT_ROOT / "reports" / "real_comparison_results" / f"hipporag_company_cache{_hip_sub}"
        print("[HippoRAG] 本地 baseline 管线（Transformers 向量）...", flush=True)

    print(
        f"[HippoRAG] 初始化并索引政策语料... reuse_cached_index={bool(args.reuse_cached_index)}",
        flush=True,
    )
    policy_hippo = _init_hipporag_with_cached_openie(
        save_dir=pol_dir,
        docs=policy_docs,
        doc_titles=policy_titles,
        use_dashscope_api=use_api,
        api_key=api_key,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        embedding_batch_size=args.embedding_batch_size,
        enable_fact_rerank=enable_fact_rerank,
        rerank_llm_name=args.rerank_llm_name,
        rerank_base_url=args.rerank_base_url,
        reuse_cached_index=bool(args.reuse_cached_index),
    )
    print("[HippoRAG] 初始化并索引企业语料...", flush=True)
    company_hippo = _init_hipporag_with_cached_openie(
        save_dir=com_dir,
        docs=company_docs,
        doc_titles=company_names,
        use_dashscope_api=use_api,
        api_key=api_key,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        embedding_batch_size=args.embedding_batch_size,
        enable_fact_rerank=enable_fact_rerank,
        rerank_llm_name=args.rerank_llm_name,
        rerank_base_url=args.rerank_base_url,
        reuse_cached_index=bool(args.reuse_cached_index),
    )

    use_induced_v2 = (
        is_subgraph
        and str(getattr(args, "subgraph_eval_protocol", SUBGRAPH_EVAL_PROTOCOL_LEGACY)).strip()
        == SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2
    )
    query_set_meta: Optional[Dict] = None

    if use_induced_v2:
        data_root_h = PROJECT_ROOT / "reports" / "real_comparison_data"
        openke_data_h = data_root_h / "openke_policykg"
        entity_token_map_h = json.loads((data_root_h / "entity_token_map.json").read_text(encoding="utf-8"))
        relation_token_map_h = json.loads((data_root_h / "relation_token_map.json").read_text(encoding="utf-8"))
        tok_eid_h = _read_id_map(openke_data_h / "entity2id.txt")
        tok_rid_h = _read_id_map(openke_data_h / "relation2id.txt")
        sr_h = tok_rid_h[relation_token_map_h["openke"]["supports"]]
        enterprise_queries, policy_queries, query_set_meta = build_subgraph_induced_eval_queries(
            openke_data=openke_data_h,
            supports_rid=sr_h,
            token_to_eid=tok_eid_h,
            openke_raw_to_tok=entity_token_map_h["openke"],
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
            f"[HippoRAG] induced_v2 tag={subgraph_tag} E->P {len(enterprise_queries)} P->E {len(policy_queries)}",
            flush=True,
        )
    else:
        enterprise_queries, policy_queries = _prepare_queries_with_test_gt(
            max_enterprise_queries=args.max_enterprise_queries,
            max_industry_queries=args.max_industry_queries,
            max_policy_queries=args.max_policy_queries,
        )
        if is_subgraph:
            enterprise_queries, policy_queries = filter_queries_subgraph_entities(
                enterprise_queries,
                policy_queries,
                set(policy_titles),
                set(company_names),
                industry_to_companies_full_map(read_enterprises_full(PROJECT_ROOT)),
            )
            pid_map = {str(r["title"]): int(r["policy_id"]) for _, r in df_policies.iterrows()}
            for q in policy_queries:
                q["policy_id"] = int(pid_map.get(str(q["policy_title"]), -1))
            print(
                f"[HippoRAG] subgraph_entities tag={subgraph_tag} E->P {len(enterprise_queries)} P->E {len(policy_queries)}",
                flush=True,
            )

    # E->P cap 按平均 GT 自适应（如果用户未显式给定）
    company_gt_sizes = [len(q["ground_truth"]) for q in enterprise_queries if q.get("type", "company_name") != "industry"]
    industry_gt_sizes = [len(q["ground_truth"]) for q in enterprise_queries if q.get("type", "company_name") == "industry"]
    inferred_policy_cap = max(1, int(round(float(np.mean(company_gt_sizes))))) if company_gt_sizes else 25
    inferred_policy_industry_cap = max(1, int(round(float(np.mean(industry_gt_sizes))))) if industry_gt_sizes else inferred_policy_cap

    policy_cap = args.policy_max_output_cap if args.policy_max_output_cap > 0 else inferred_policy_cap
    policy_industry_cap = (
        args.policy_industry_query_max_output_cap
        if args.policy_industry_query_max_output_cap > 0
        else inferred_policy_industry_cap
    )
    print(
        f"[HippoRAG] E->P cap设置: company={policy_cap}, industry={policy_industry_cap} "
        f"(company_avg_gt={float(np.mean(company_gt_sizes)) if company_gt_sizes else 0:.2f}, "
        f"industry_avg_gt={float(np.mean(industry_gt_sizes)) if industry_gt_sizes else 0:.2f})",
        flush=True,
    )

    print(f"[HippoRAG] E->P 查询数: {len(enterprise_queries)}", flush=True)
    ep_metrics: List[Dict] = []
    masked_ep = 0
    policy_title_set = set(policy_titles)

    for i, q in enumerate(enterprise_queries, start=1):
        q_text = str(q["query"])
        q_type = q.get("type", "company_name")
        gt = [x for x in q["ground_truth"] if x in policy_title_set]
        if len(gt) == 0:
            masked_ep += 1
        sol = policy_hippo.retrieve([q_text], num_to_retrieve=args.policy_candidate_k)[0]
        ranked = []
        for d, s in zip(sol.docs, sol.doc_scores):
            title = d.split("\n", 1)[0].strip()
            ranked.append((title, float(s)))
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
        pred = []
        seen = set()
        for t, _ in ranked:
            if t in policy_title_set and t not in seen:
                pred.append(t)
                seen.add(t)
        m = calculate_metrics(pred, gt)
        rm = calculate_ranking_metrics(pred, gt)
        ep_metrics.append({**m, **rm, "query": q_text, "query_type": q_type})
        if i % max(1, len(enterprise_queries) // 10) == 0 or i == len(enterprise_queries):
            print(f"[HippoRAG][E->P] {i}/{len(enterprise_queries)}", flush=True)

    print(f"[HippoRAG] P->E 查询数: {len(policy_queries)}", flush=True)
    pe_metrics: List[Dict] = []
    masked_pe = 0
    company_name_set = set(company_names)
    policy_to_companies = (
        df_policies[["policy_id", "title"]]
        .set_index("policy_id")["title"]
        .to_dict()
    )
    direct_support_boost_map: Dict[str, Set[str]] = {}
    triples_df = pd.read_parquet(triples_parquet_path(PROJECT_ROOT, scale_opt))
    for row in triples_df.itertuples(index=False):
        if str(row.predicate).lower() == "supports":
            direct_support_boost_map.setdefault(str(row.subject), set()).add(str(row.object))

    for i, q in enumerate(policy_queries, start=1):
        title = str(q["policy_title"])
        gt = [x for x in q["ground_truth"] if x in company_name_set]
        if len(gt) == 0:
            masked_pe += 1
        sol = company_hippo.retrieve([title], num_to_retrieve=args.enterprise_candidate_k)[0]
        ranked = []
        direct_set = direct_support_boost_map.get(title, set())
        for d, s in zip(sol.docs, sol.doc_scores):
            cname = d.split("\n", 1)[0].strip()
            score = float(s) + (0.3 if cname in direct_set else 0.0)
            ranked.append((cname, score))
        ranked = _apply_rank_cutoff(
            ranked_pairs=ranked,
            top_k=args.top_k_enterprise,
            score_threshold=enterprise_score_threshold,
            adaptive_quantile=args.enterprise_adaptive_quantile,
            relative_drop_threshold=args.enterprise_relative_drop_threshold,
            max_output_cap=args.enterprise_max_output_cap,
        )
        pred = []
        seen = set()
        for c, _ in ranked:
            if c in company_name_set and c not in seen:
                pred.append(c)
                seen.add(c)
        m = calculate_metrics(pred, gt)
        rm = calculate_ranking_metrics(pred, gt)
        pe_metrics.append({**m, **rm, "policy_id": int(q["policy_id"]), "policy_title": title})
        if i % max(1, len(policy_queries) // 10) == 0 or i == len(policy_queries):
            print(f"[HippoRAG][P->E] {i}/{len(policy_queries)}", flush=True)

    ep_avg = {
        "precision": float(np.mean([m["precision"] for m in ep_metrics])) if ep_metrics else 0.0,
        "recall": float(np.mean([m["recall"] for m in ep_metrics])) if ep_metrics else 0.0,
        "f1": float(np.mean([m["f1"] for m in ep_metrics])) if ep_metrics else 0.0,
        "map": float(np.mean([m["ap"] for m in ep_metrics])) if ep_metrics else 0.0,
        "ndcg": float(np.mean([m["ndcg"] for m in ep_metrics])) if ep_metrics else 0.0,
        "masked_gt_empty": int(masked_ep),
    }
    pe_avg = {
        "precision": float(np.mean([m["precision"] for m in pe_metrics])) if pe_metrics else 0.0,
        "recall": float(np.mean([m["recall"] for m in pe_metrics])) if pe_metrics else 0.0,
        "f1": float(np.mean([m["f1"] for m in pe_metrics])) if pe_metrics else 0.0,
        "map": float(np.mean([m["ap"] for m in pe_metrics])) if pe_metrics else 0.0,
        "ndcg": float(np.mean([m["ndcg"] for m in pe_metrics])) if pe_metrics else 0.0,
        "masked_gt_empty": int(masked_pe),
    }

    _eval_scope = "full"
    if is_subgraph:
        _eval_scope = "subgraph_induced_v2" if use_induced_v2 else "subgraph_entities"
    _eval_query_set = (
        "induced_v2_test_supports"
        if use_induced_v2
        else ("legacy_main_protocol_subgraph_filtered" if is_subgraph else "legacy_main_protocol_full")
    )
    result = {
        "model": "HippoRAG",
        "timestamp": datetime.now().isoformat(),
        "evaluation_protocol": "main_queries + test_split_gt + unified_cutoff",
        "eval_query_scope": _eval_scope,
        "eval_query_set": _eval_query_set,
        "scale_dir": scale_opt,
        "subgraph_tag": subgraph_tag if is_subgraph else None,
        "hipporag_notes": (
            "openie=cached_placeholder(title,related_to,title); no LLM OpenIE extraction; "
            f"pipeline={'dashscope_api' if use_api else 'local_transformers'}; "
            f"fact_rerank_dsp={'on' if enable_fact_rerank else 'off'}."
        ),
        "parameters": {
            "subgraph_eval_protocol": (args.subgraph_eval_protocol if is_subgraph else None),
            "min_company_ep_queries": int(args.min_company_ep_queries),
            "min_industry_ep_queries": int(args.min_industry_ep_queries),
            "min_pe_queries": int(args.min_pe_queries),
            "local_baseline": bool(args.local_baseline),
            "use_dashscope_api": use_api,
            "llm_model": args.llm_model if use_api else (args.rerank_llm_name if enable_fact_rerank else "gpt-4o-mini"),
            "llm_base_url": args.llm_base_url if use_api else (args.rerank_base_url if enable_fact_rerank else "http://localhost:1/v1"),
            "embedding_model": args.embedding_model if use_api else "Transformers/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_base_url": args.embedding_base_url if use_api else None,
            "embedding_batch_size": args.embedding_batch_size if use_api else None,
            "enable_fact_rerank": enable_fact_rerank,
            "policy_cache_dir": str(pol_dir),
            "company_cache_dir": str(com_dir),
            "policy_max_output_cap": policy_cap,
            "policy_industry_query_max_output_cap": policy_industry_cap,
            "enterprise_max_output_cap": args.enterprise_max_output_cap,
            "policy_max_output_cap_source": "arg" if args.policy_max_output_cap > 0 else "avg_gt_company",
            "policy_industry_query_max_output_cap_source": "arg"
            if args.policy_industry_query_max_output_cap > 0
            else "avg_gt_industry",
        },
        "enterprise_to_policy": {"num_queries": len(ep_metrics), "average": ep_avg},
        "policy_to_enterprise": {"num_queries": len(pe_metrics), "average": pe_avg},
    }
    if query_set_meta is not None:
        result["query_set_meta"] = query_set_meta

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt = json.dumps(result, ensure_ascii=False, indent=2)
    out_path.write_text(out_txt, encoding="utf-8")
    print(out_txt)

    report_dir = REPO_ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        (report_dir / Path(args.output).name).write_text(out_txt, encoding="utf-8")
    except OSError as e:
        print(f"[HippoRAG] report/ 同步失败（主输出已写入）: {e}", flush=True)


if __name__ == "__main__":
    main()

