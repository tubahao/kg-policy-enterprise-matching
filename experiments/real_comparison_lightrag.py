#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightRAG 真实对比实验（与主实验 / HippoRAG 对齐的主对比协议）

默认 index_mode=triples（需 DASHSCOPE_API_KEY），与 reports/实验全流程总结.md、reports/三元组提取.md 对齐：

**政策–实体/行业（主表，清洗后）**
- 优先：`--triples_parquet`（默认 data_intermediate/triples_policy_entity.parquet）
- 对应全流程总结中的预处理三元组输出；不另做 JSON 回退，以保证与主实验口径一致。

**政策–政策（传导边，如 transmitsTo）**
- 优先：`--triples_policy_policy_parquet`（默认 data_intermediate/triples_policy_policy.parquet）
- 若 parquet 不存在，自动回退 `output/policy_policy_only.json`（与三元组提取.md 5.1 一致）。
- `--no_policy_policy_triples` 可关闭合并。

索引经 ainsert_custom_kg 写入图与向量库，**不调用** LightRAG 索引阶段 LLM 抽取。
向量默认 DashScope openai_embed；加 --local_gpu_embedding 时改为本机 sentence-transformers（CUDA 优先）。
查询 / 重排仍按 LightRAG：hybrid + LLM 关键词、可选 ali_rerank（DashScope）。

index_mode=full_llm：LightRAG.ainsert 全量 LLM 抽取（极慢，「原版满血」对照）。
--minimal_index：等同 index_mode=minimal（本地向量 + 占位图 + 手写关键词，不调 DashScope）。

**自动检查点**（reports/real_comparison_results/ 与 report/ 各写一份）：
after_policy_ingest、after_company_ingest、ep_eval_progress、after_enterprise_to_policy_eval、
pe_eval_progress、after_policy_to_enterprise_eval、complete；并维护 lightrag_checkpoint_latest.json。
**--scale_dir**：子图评测时工作目录与检查点文件名带 `_subgraph_<tag>`，与全量隔离。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import multiprocessing
import numbers
import os
import shutil
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def _make_json_safe(obj: Any) -> Any:
    """numpy / Path / 非有限 float 等转为 JSON 可序列化类型，避免 dumps 或检查点写入抛错中断流程。"""
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _make_json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _make_json_safe(obj.item())
    if isinstance(obj, numbers.Integral) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, numbers.Real) and not isinstance(obj, bool):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    return str(obj)


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(
        _make_json_safe(obj),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
LIGHTRAG_SRC = REPO_ROOT / "LigntRAG" / "LightRAG"
if not LIGHTRAG_SRC.is_dir():
    _alt = REPO_ROOT.parent / "LigntRAG" / "LightRAG"
    if _alt.is_dir():
        LIGHTRAG_SRC = _alt
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(LIGHTRAG_SRC))

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
    for q in policy_queries:
        title = str(q["policy_title"])
        q["ground_truth"] = sorted(list(p2e_test.get(title, set())))
    return enterprise_queries, policy_queries


def _chunk_title_from_lightrag_chunk(ch: Dict[str, Any]) -> str:
    fp = (ch.get("file_path") or "").strip()
    if fp and fp != "custom_kg":
        return fp
    content = ch.get("content") or ""
    line = content.split("\n", 1)[0].strip()
    return line


def _ranked_from_query_data(
    result: Dict[str, Any],
    *,
    base_score: float = 1.0,
) -> List[Tuple[str, float]]:
    if not result or result.get("status") != "success":
        return []
    data = result.get("data") or {}
    chunks = data.get("chunks") or []
    ranked: List[Tuple[str, float]] = []
    seen: Set[str] = set()
    for i, ch in enumerate(chunks):
        title = _chunk_title_from_lightrag_chunk(ch)
        if not title or title in seen:
            continue
        seen.add(title)
        # 若有 rerank_score，用其作主要排序依据（与列表顺序一致时仍保持区分度）
        rs = ch.get("rerank_score")
        if rs is not None:
            sc = float(rs)
        else:
            sc = base_score / float(i + 1)
        ranked.append((title, sc))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _resolve_local_embed_device(pref: str) -> str:
    """cuda / gpu / auto → 有 CUDA 则用 cuda，否则 cpu；支持 cuda:0。"""
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("本地 GPU/CPU 向量需要安装 PyTorch（torch）。") from e
    p = (pref or "cuda").strip().lower()
    if p in ("cuda", "gpu", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        print("[LightRAG] CUDA 不可用，sentence-transformers 使用 CPU。", flush=True)
        return "cpu"
    if p == "cpu":
        return "cpu"
    if p.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[LightRAG] 指定了 CUDA 但当前不可用，回退 CPU。", flush=True)
            return "cpu"
        return p
    return p


def _make_local_sentence_transformer_embedding(
    model_name: str,
    *,
    device: str | None = None,
    batch_size: int = 64,
):
    _model = None
    dev = device or "cpu"

    async def _embed(texts: List[str]) -> np.ndarray:
        nonlocal _model
        if _model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            _model = SentenceTransformer(model_name, device=dev)
            print(
                f"[LightRAG] SentenceTransformer 已加载: {model_name!r} device={dev} batch_size={batch_size}",
                flush=True,
            )
        loop = asyncio.get_running_loop()
        texts_list = list(texts)

        def _encode() -> np.ndarray:
            return np.asarray(
                _model.encode(
                    texts_list,
                    convert_to_numpy=True,
                    batch_size=batch_size,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )

        return await loop.run_in_executor(None, _encode)

    async def _probe_dim() -> int:
        v = await _embed(["ping"])
        return int(v.shape[1])

    return _embed, _probe_dim


async def _build_lightrag_minimal(
    working_dir: Path,
    workspace: str,
    st_model_name: str,
    clean: bool,
) -> Any:
    from lightrag import LightRAG  # type: ignore
    from lightrag.utils import EmbeddingFunc  # type: ignore

    if clean and working_dir.exists():
        shutil.rmtree(working_dir, ignore_errors=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    embed_inner, probe = _make_local_sentence_transformer_embedding(
        st_model_name, device=_resolve_local_embed_device("auto"), batch_size=32
    )
    dim = await probe()
    embedding_func = EmbeddingFunc(embedding_dim=dim, func=embed_inner, max_token_size=8192, model_name=st_model_name)

    async def _noop_llm(*_a: Any, **_kw: Any) -> str:
        return ""

    rag = LightRAG(
        working_dir=str(working_dir),
        workspace=workspace,
        embedding_func=embedding_func,
        llm_model_func=_noop_llm,
        llm_model_name="noop",
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        llm_model_max_async=1,
        embedding_func_max_async=2,
        addon_params={"language": "Chinese"},
    )
    await rag.initialize_storages()
    return rag


async def _build_lightrag_full_api(
    working_dir: Path,
    workspace: str,
    args: argparse.Namespace,
    api_key: str,
    clean: bool,
) -> Any:
    from lightrag import LightRAG  # type: ignore
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed  # type: ignore
    from lightrag.rerank import ali_rerank  # type: ignore
    from lightrag.utils import EmbeddingFunc  # type: ignore

    if clean and working_dir.exists():
        shutil.rmtree(working_dir, ignore_errors=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    os.environ["OPENAI_API_KEY"] = api_key

    async def llm_model_func(
        prompt: str,
        system_prompt: Any = None,
        history_messages: Any = None,
        keyword_extraction: bool = False,
        **kwargs: Any,
    ) -> str:
        return await openai_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            keyword_extraction=keyword_extraction,
            base_url=args.llm_base_url,
            api_key=api_key,
            **kwargs,
        )

    if getattr(args, "local_gpu_embedding", False):
        dev = _resolve_local_embed_device(args.local_embedding_device)
        embed_inner, probe = _make_local_sentence_transformer_embedding(
            args.local_embedding_model,
            device=dev,
            batch_size=int(args.local_embedding_batch_size),
        )
        dim = await probe()
        embedding_func = EmbeddingFunc(
            embedding_dim=dim,
            func=embed_inner,
            max_token_size=8192,
            model_name=f"{args.local_embedding_model}|{dev}",
        )
    else:
        embed_partial = partial(
            openai_embed.func,
            model=args.embedding_api_model,
            base_url=args.embedding_base_url,
            api_key=api_key,
        )
        embedding_func = EmbeddingFunc(
            embedding_dim=args.embedding_dim,
            func=embed_partial,
            max_token_size=8192,
            model_name=args.embedding_api_model,
        )

    rerank_model_func = None
    if not args.no_chunk_rerank:

        async def rerank_model_func(
            query: str,
            documents: list,
            top_n: int | None = None,
            extra_body: dict | None = None,
        ):
            return await ali_rerank(
                query=query,
                documents=documents,
                top_n=top_n,
                api_key=api_key,
                model=args.rerank_model,
                base_url=args.rerank_base_url,
                extra_body=extra_body,
            )

    rag = LightRAG(
        working_dir=str(working_dir),
        workspace=workspace,
        llm_model_func=llm_model_func,
        llm_model_name=args.llm_model,
        embedding_func=embedding_func,
        rerank_model_func=rerank_model_func,
        addon_params={"language": "Chinese"},
        llm_model_max_async=args.llm_max_async,
        max_parallel_insert=args.max_parallel_insert,
        embedding_func_max_async=args.embedding_max_async,
        default_embedding_timeout=(
            max(int(args.embedding_timeout), 600)
            if getattr(args, "local_gpu_embedding", False)
            else int(args.embedding_timeout)
        ),
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
    )
    await rag.initialize_storages()
    return rag


def _predicate_mask(df: pd.DataFrame, *needles: str) -> pd.Series:
    pl = df["predicate"].astype(str).str.strip().str.lower()
    nset = {n.lower() for n in needles}
    return pl.isin(nset)


def _custom_kg_policies_from_triples(
    policy_docs: List[str],
    policy_titles: List[str],
    triples_df: pd.DataFrame,
    triples_policy_policy: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """政策库：triples_policy_entity（supports/targetsIndustry）+ 可选 triples_policy_policy（如 transmitsTo）。"""
    from lightrag.utils import compute_mdhash_id  # type: ignore

    sup = triples_df[_predicate_mask(triples_df, "supports")]
    tgt = triples_df[_predicate_mask(triples_df, "targetsIndustry")]

    pol_to_co: Dict[str, List[str]] = {}
    if len(sup) > 0:
        for subj, g in sup.groupby(sup["subject"].astype(str), sort=False):
            pol_to_co[str(subj)] = list(dict.fromkeys(g["object"].astype(str)))
    pol_to_ind: Dict[str, List[str]] = {}
    if len(tgt) > 0:
        for subj, g in tgt.groupby(tgt["subject"].astype(str), sort=False):
            pol_to_ind[str(subj)] = list(dict.fromkeys(g["object"].astype(str)))

    # 政策—政策：subject -> 下游政策标题（及谓词、可选相似度权重）
    pol_to_pp: Dict[str, List[Tuple[str, str, float]]] = {}
    if triples_policy_policy is not None and len(triples_policy_policy) > 0:
        pp = triples_policy_policy
        for row in pp.itertuples(index=False):
            s = str(getattr(row, "subject", "") or "")
            o = str(getattr(row, "object", "") or "")
            pr = str(getattr(row, "predicate", "transmitsTo") or "transmitsTo")
            if not s or not o:
                continue
            w = 1.0
            if hasattr(row, "tfidf_similarity"):
                try:
                    w = float(getattr(row, "tfidf_similarity") or 1.0)
                except (TypeError, ValueError):
                    w = 1.0
            pol_to_pp.setdefault(s, []).append((o, pr, max(0.1, min(2.0, w))))

    chunks: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []

    for i, (doc, title) in enumerate(zip(policy_docs, policy_titles)):
        t = str(title)
        sid = compute_mdhash_id(f"pol|{t}|{i}", prefix="src-")
        chunks.append({"content": doc, "source_id": sid, "file_path": t, "chunk_order_index": 0})
        entities.append(
            {
                "entity_name": t,
                "entity_type": "policy",
                "description": doc[:1200],
                "source_id": sid,
                "file_path": t,
            }
        )
        for co in pol_to_co.get(t, []):
            co = str(co)
            entities.append(
                {
                    "entity_name": co,
                    "entity_type": "company",
                    "description": f"企业：{co}",
                    "source_id": sid,
                    "file_path": t,
                }
            )
            relationships.append(
                {
                    "src_id": t,
                    "tgt_id": co,
                    "description": f"政策《{t[:120]}》 supports 企业 {co}",
                    "keywords": "supports,扶持,政策企业",
                    "source_id": sid,
                    "weight": 1.0,
                    "file_path": t,
                }
            )
        for ind in pol_to_ind.get(t, []):
            ind = str(ind)
            entities.append(
                {
                    "entity_name": ind,
                    "entity_type": "industry",
                    "description": f"行业：{ind}",
                    "source_id": sid,
                    "file_path": t,
                }
            )
            relationships.append(
                {
                    "src_id": t,
                    "tgt_id": ind,
                    "description": f"政策《{t[:120]}》 targetsIndustry {ind}",
                    "keywords": "targetsIndustry,行业,政策",
                    "source_id": sid,
                    "weight": 1.0,
                    "file_path": t,
                }
            )
        for obj_pol, pred, wt in pol_to_pp.get(t, []):
            entities.append(
                {
                    "entity_name": obj_pol,
                    "entity_type": "policy",
                    "description": f"关联政策：{obj_pol}",
                    "source_id": sid,
                    "file_path": t,
                }
            )
            relationships.append(
                {
                    "src_id": t,
                    "tgt_id": obj_pol,
                    "description": f"《{t[:80]}》 {pred} 《{obj_pol[:80]}》",
                    "keywords": f"{pred},政策传导,层级",
                    "source_id": sid,
                    "weight": float(wt),
                    "file_path": t,
                }
            )

    return {"chunks": chunks, "entities": entities, "relationships": relationships}


def _custom_kg_companies_from_triples(
    df_enterprises: pd.DataFrame,
    triples_df: pd.DataFrame,
) -> Dict[str, Any]:
    """企业库：由 supports 反查政策，belongsTo 接行业；实体名与企业名一致以便 P->E 评测对齐。"""
    from lightrag.utils import compute_mdhash_id  # type: ignore

    sup = triples_df[_predicate_mask(triples_df, "supports")]
    bel = triples_df[_predicate_mask(triples_df, "belongsTo", "belongsto")]

    co_to_pol: Dict[str, List[str]] = {}
    if len(sup) > 0:
        for obj, g in sup.groupby(sup["object"].astype(str), sort=False):
            co_to_pol[str(obj)] = list(dict.fromkeys(g["subject"].astype(str)))
    co_to_ind: Dict[str, List[str]] = {}
    if len(bel) > 0:
        for subj, g in bel.groupby(bel["subject"].astype(str), sort=False):
            co_to_ind[str(subj)] = list(dict.fromkeys(g["object"].astype(str)))

    chunks: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []

    names = df_enterprises["name"].astype(str).tolist()
    texts = df_enterprises["text_with_industry"].fillna("").astype(str).tolist()
    eids = df_enterprises["enterprise_id"].tolist()

    for i, (eid, name, text) in enumerate(zip(eids, names, texts)):
        name = str(name)
        doc = f"{name}\n{text[:1200]}"
        sid = compute_mdhash_id(f"ent|{eid}|{i}", prefix="src-")
        chunks.append({"content": doc, "source_id": sid, "file_path": name, "chunk_order_index": 0})
        entities.append(
            {
                "entity_name": name,
                "entity_type": "enterprise",
                "description": f"{name}\n{text[:1000]}",
                "source_id": sid,
                "file_path": name,
            }
        )
        for pol in co_to_pol.get(name, []):
            pol = str(pol)
            entities.append(
                {
                    "entity_name": pol,
                    "entity_type": "policy",
                    "description": f"政策：{pol}",
                    "source_id": sid,
                    "file_path": name,
                }
            )
            relationships.append(
                {
                    "src_id": pol,
                    "tgt_id": name,
                    "description": f"{pol} supports {name}",
                    "keywords": "supports,扶持",
                    "source_id": sid,
                    "weight": 1.0,
                    "file_path": name,
                }
            )
        for ind in co_to_ind.get(name, []):
            ind = str(ind)
            entities.append(
                {
                    "entity_name": ind,
                    "entity_type": "industry",
                    "description": f"行业：{ind}",
                    "source_id": sid,
                    "file_path": name,
                }
            )
            relationships.append(
                {
                    "src_id": name,
                    "tgt_id": ind,
                    "description": f"{name} belongsTo {ind}",
                    "keywords": "belongsTo,行业",
                    "source_id": sid,
                    "weight": 1.0,
                    "file_path": name,
                }
            )

    return {"chunks": chunks, "entities": entities, "relationships": relationships}


def _custom_kg_for_policies(policy_docs: List[str], titles: List[str]) -> Dict[str, Any]:
    from lightrag.utils import compute_mdhash_id  # type: ignore

    chunks: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    for i, (doc, title) in enumerate(zip(policy_docs, titles)):
        sid = compute_mdhash_id(f"pol|{title}|{i}", prefix="src-")
        chunks.append({"content": doc, "source_id": sid, "file_path": title, "chunk_order_index": 0})
        entities.append(
            {
                "entity_name": title,
                "entity_type": "policy",
                "description": doc[:1200],
                "source_id": sid,
                "file_path": title,
            }
        )
        relationships.append(
            {
                "src_id": title,
                "tgt_id": title,
                "description": "policy",
                "keywords": title[:200].replace(" ", ","),
                "source_id": sid,
                "weight": 1.0,
                "file_path": title,
            }
        )
    return {"chunks": chunks, "entities": entities, "relationships": relationships}


def _custom_kg_for_companies(df: pd.DataFrame) -> Dict[str, Any]:
    from lightrag.utils import compute_mdhash_id  # type: ignore

    chunks: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    names = df["name"].astype(str).tolist()
    texts = df["text_with_industry"].fillna("").astype(str).tolist()
    eids = df["enterprise_id"].tolist()
    for i, (eid, name, text) in enumerate(zip(eids, names, texts)):
        doc = f"{name}\n{text[:1200]}"
        sid = compute_mdhash_id(f"ent|{eid}|{i}", prefix="src-")
        ent_key = f"__ent__{eid}"
        chunks.append({"content": doc, "source_id": sid, "file_path": name, "chunk_order_index": 0})
        entities.append(
            {
                "entity_name": ent_key,
                "entity_type": "enterprise",
                "description": f"{name}\n{text[:1000]}",
                "source_id": sid,
                "file_path": name,
            }
        )
        relationships.append(
            {
                "src_id": ent_key,
                "tgt_id": ent_key,
                "description": "enterprise",
                "keywords": name[:200].replace(" ", ","),
                "source_id": sid,
                "weight": 1.0,
                "file_path": name,
            }
        )
    return {"chunks": chunks, "entities": entities, "relationships": relationships}


async def _ingest_custom_kg(rag: Any, kg: Dict[str, Any], label: str) -> None:
    n = len(kg.get("chunks", []))
    print(f"[LightRAG] {label} ainsert_custom_kg chunks={n} ...", flush=True)
    t0 = time.perf_counter()

    async def _heartbeat() -> None:
        while True:
            await asyncio.sleep(60.0)
            print(
                f"[LightRAG] {label} 仍在向量化/写入中… 已耗时 {time.perf_counter() - t0:.0f}s",
                flush=True,
            )

    hb = asyncio.create_task(_heartbeat())
    try:
        await rag.ainsert_custom_kg(kg)
    finally:
        hb.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await hb
    print(f"[LightRAG] {label} ingest done.", flush=True)


async def _ingest_ainsert(rag: Any, docs: List[str], titles: List[str], label: str) -> None:
    print(f"[LightRAG] {label} ainsert LLM pipeline docs={len(docs)} ...", flush=True)
    await rag.ainsert(docs, file_paths=titles)
    print(f"[LightRAG] {label} ainsert done.", flush=True)


def _resolve_project_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_policy_policy_triples(
    args: argparse.Namespace, scale_opt: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """与《三元组提取.md》一致：子图优先 scale_dir/triples_policy_policy.parquet；全量优先 data_intermediate/...；否则回退 JSON。"""
    if getattr(args, "no_policy_policy_triples", False):
        return None
    if scale_opt and str(scale_opt).strip():
        pp_sub = (PROJECT_ROOT / scale_opt).resolve() / "triples_policy_policy.parquet"
        if pp_sub.is_file():
            df = pd.read_parquet(pp_sub)
            print(f"[LightRAG] 已加载政策-政策三元组(子图): {pp_sub} ({len(df)} 行)", flush=True)
            return df
    pp_path = _resolve_project_path(str(args.triples_policy_policy_parquet))
    if pp_path.is_file():
        df = pd.read_parquet(pp_path)
        print(f"[LightRAG] 已加载政策-政策三元组: {pp_path} ({len(df)} 行)", flush=True)
        return df
    json_fb = PROJECT_ROOT / "output" / "policy_policy_only.json"
    if json_fb.is_file():
        raw = json.loads(json_fb.read_text(encoding="utf-8"))
        df = pd.DataFrame(raw)
        print(
            f"[LightRAG] 未找到 {pp_path}，回退使用 {json_fb} ({len(df)} 行)",
            flush=True,
        )
        return df
    print(
        f"[LightRAG] 未找到政策-政策三元组（{pp_path} 与 output/policy_policy_only.json 均不存在），跳过 transmitsTo 等边",
        flush=True,
    )
    return None


def _build_lightrag_notes(
    args: argparse.Namespace,
    index_mode: str,
    triples_pp_df: Optional[pd.DataFrame],
) -> str:
    if index_mode == "minimal":
        return (
            "minimal: ainsert_custom_kg(placeholder) + local sentence-transformers + manual hl/ll; "
            "no DashScope."
        )
    if index_mode == "triples":
        _em = (
            f"local_ST_GPU|{args.local_embedding_model}"
            if getattr(args, "local_gpu_embedding", False)
            else str(args.embedding_api_model)
        )
        return (
            f"triples_index: PE={args.triples_parquet}; PP_file={args.triples_policy_policy_parquet}; "
            f"pp_edges_loaded={bool(triples_pp_df is not None and len(triples_pp_df) > 0)}; "
            f"no LLM on insert; embed={_em}; LLM={args.llm_model} for query keywords; "
            f"rerank={'ali '+args.rerank_model if not args.no_chunk_rerank else 'off'}."
        )
    return (
        f"full_llm: ainsert extract/summarize ({args.llm_model}); "
        f"embedding {args.embedding_api_model}; hybrid + keyword LLM; "
        f"rerank={'ali '+args.rerank_model if not args.no_chunk_rerank else 'off'}."
    )


def _build_parameters_payload(
    args: argparse.Namespace,
    index_mode: str,
    policy_dir: Path,
    company_dir: Path,
    triples_pp_df: Optional[pd.DataFrame],
    policy_rag: Any,
    policy_cap: int,
    policy_industry_cap: int,
) -> Dict[str, Any]:
    return {
        "index_mode": index_mode,
        "policy_working_dir": str(policy_dir),
        "company_working_dir": str(company_dir),
        "triples_parquet": str(args.triples_parquet),
        "triples_policy_policy_parquet": str(args.triples_policy_policy_parquet),
        "policy_policy_triples_loaded": triples_pp_df is not None and len(triples_pp_df) > 0,
        "minimal_index_flag": bool(args.minimal_index),
        "query_mode": "hybrid",
        "llm_model": None if index_mode == "minimal" else args.llm_model,
        "llm_base_url": None if index_mode == "minimal" else args.llm_base_url,
        "embedding_api_model": None if index_mode == "minimal" else args.embedding_api_model,
        "embedding_dim": (
            None
            if index_mode == "minimal"
            else (
                policy_rag.embedding_func.embedding_dim
                if getattr(args, "local_gpu_embedding", False)
                else args.embedding_dim
            )
        ),
        "local_gpu_embedding": bool(getattr(args, "local_gpu_embedding", False)),
        "local_embedding_model": (
            args.local_embedding_model
            if index_mode == "minimal" or getattr(args, "local_gpu_embedding", False)
            else None
        ),
        "local_embedding_device_requested": (
            args.local_embedding_device
            if index_mode != "minimal" and getattr(args, "local_gpu_embedding", False)
            else None
        ),
        "embedding_timeout_sec": None if index_mode == "minimal" else int(args.embedding_timeout),
        "embedding_base_url": None if index_mode == "minimal" else args.embedding_base_url,
        "chunk_rerank": (not args.no_chunk_rerank) if index_mode != "minimal" else False,
        "rerank_model": None if index_mode == "minimal" or args.no_chunk_rerank else args.rerank_model,
        "lightrag_top_k": args.lightrag_top_k,
        "lightrag_chunk_top_k": args.lightrag_chunk_top_k,
        "policy_max_output_cap": policy_cap,
        "policy_industry_query_max_output_cap": policy_industry_cap,
        "enterprise_max_output_cap": args.enterprise_max_output_cap,
        "policy_max_output_cap_source": "arg" if args.policy_max_output_cap > 0 else "avg_gt_company",
        "policy_industry_query_max_output_cap_source": (
            "arg" if args.policy_industry_query_max_output_cap > 0 else "avg_gt_industry"
        ),
        "max_policy_docs": args.max_policy_docs,
        "max_company_docs": args.max_company_docs,
        "scale_dir": (getattr(args, "scale_dir", "") or "").strip() or None,
        "subgraph_eval_protocol": (
            getattr(args, "subgraph_eval_protocol", SUBGRAPH_EVAL_PROTOCOL_LEGACY)
            if (getattr(args, "scale_dir", "") or "").strip()
            else None
        ),
        "min_company_ep_queries": int(getattr(args, "min_company_ep_queries", 40)),
        "min_industry_ep_queries": int(getattr(args, "min_industry_ep_queries", 12)),
        "min_pe_queries": int(getattr(args, "min_pe_queries", 25)),
        "eval_query_scope": (
            "subgraph_induced_v2"
            if (getattr(args, "scale_dir", "") or "").strip()
            and str(getattr(args, "subgraph_eval_protocol", SUBGRAPH_EVAL_PROTOCOL_LEGACY)).strip()
            == SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2
            else ("subgraph_entities" if (getattr(args, "scale_dir", "") or "").strip() else "full")
        ),
    }


def _average_from_metric_rows(rows: List[Dict[str, Any]], masked: int) -> Dict[str, Any]:
    if not rows:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "map": 0.0,
            "ndcg": 0.0,
            "masked_gt_empty": int(masked),
        }
    return {
        "precision": float(np.mean([m["precision"] for m in rows])),
        "recall": float(np.mean([m["recall"] for m in rows])),
        "f1": float(np.mean([m["f1"] for m in rows])),
        "map": float(np.mean([m["ap"] for m in rows])),
        "ndcg": float(np.mean([m["ndcg"] for m in rows])),
        "masked_gt_empty": int(masked),
    }


def _save_lightrag_checkpoint(
    args: argparse.Namespace,
    *,
    stage: str,
    partial: bool,
    index_mode: str,
    policy_dir: Path,
    company_dir: Path,
    triples_pp_df: Optional[pd.DataFrame],
    policy_rag: Any,
    policy_cap: int,
    policy_industry_cap: int,
    enterprise_to_policy: Dict[str, Any],
    policy_to_enterprise: Dict[str, Any],
) -> None:
    """写入检查点 JSON（results + report）；失败仅打日志，不抛出，以免中断后续索引/评测。"""
    try:
        ck_suffix = getattr(args, "_lightrag_checkpoint_suffix", "") or ""
        notes = _build_lightrag_notes(args, index_mode, triples_pp_df)
        body: Dict[str, Any] = {
            "model": "LightRAG",
            "checkpoint_stage": stage,
            "partial": partial,
            "timestamp": datetime.now().isoformat(),
            "evaluation_protocol": "main_queries + test_split_gt + unified_cutoff",
            "lightrag_notes": notes,
            "parameters": _build_parameters_payload(
                args,
                index_mode,
                policy_dir,
                company_dir,
                triples_pp_df,
                policy_rag,
                policy_cap,
                policy_industry_cap,
            ),
            "enterprise_to_policy": enterprise_to_policy,
            "policy_to_enterprise": policy_to_enterprise,
        }
        cache_root = PROJECT_ROOT / "reports" / "real_comparison_results"
        cache_root.mkdir(parents=True, exist_ok=True)
        stem = f"lightrag_checkpoint_{stage}{ck_suffix}"
        latest_name = f"lightrag_checkpoint_latest{ck_suffix}.json"
        text = _safe_json_dumps(body)
        (cache_root / f"{stem}.json").write_text(text, encoding="utf-8")
        (cache_root / latest_name).write_text(text, encoding="utf-8")
        report_dir = REPO_ROOT / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{stem}.json").write_text(text, encoding="utf-8")
        (report_dir / latest_name).write_text(text, encoding="utf-8")
        print(f"[LightRAG] 已自动保存检查点: {stem}.json（及 {latest_name}）", flush=True)
    except Exception as e:
        print(
            f"[LightRAG] 检查点保存失败（已忽略，评测继续）stage={stage!r}: {e}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()


async def _run_eval_async(args: argparse.Namespace) -> Dict[str, Any]:
    from lightrag.base import QueryParam  # type: ignore

    index_mode: str = "minimal" if args.minimal_index else str(args.index_mode)
    api_key = (args.api_key or os.getenv("DASHSCOPE_API_KEY", "") or os.getenv("QWEN_API_KEY", "")).strip()
    if index_mode == "full_llm":
        print(
            "[LightRAG] index_mode=full_llm：将使用 ainsert 全量 LLM 抽取，耗时长、费用高；"
            "主对比默认请用 triples（数据集三元组 + ainsert_custom_kg）。",
            flush=True,
        )
    skip_ingest = bool(getattr(args, "skip_ingest", False))
    clean_wd = bool(args.clean_index) and not skip_ingest
    if args.clean_index and skip_ingest:
        print("[LightRAG] --skip_ingest：已忽略 --clean_index。", flush=True)
    print(
        f"[LightRAG] index_mode={index_mode} clean_index={bool(args.clean_index)} "
        f"skip_ingest={skip_ingest} effective_clean_workdir={clean_wd}",
        flush=True,
    )
    if getattr(args, "local_gpu_embedding", False) and index_mode != "minimal":
        print(
            "[LightRAG] --local_gpu_embedding：向量由本机 sentence-transformers 计算（优先 CUDA），"
            "LLM/重排仍用 DashScope。显存不足可调小 --local_embedding_batch_size 或 --embedding_max_async。",
            flush=True,
        )
    if index_mode in ("triples", "full_llm") and not api_key:
        raise RuntimeError(
            "triples / full_llm 模式需要 API Key：请设置 DASHSCOPE_API_KEY 或传入 --api_key。"
        )

    scale_opt = (getattr(args, "scale_dir", "") or "").strip() or None
    df_policies, df_enterprises, is_subgraph, subgraph_tag = read_policy_enterprise_tables(
        PROJECT_ROOT, scale_opt
    )
    setattr(args, "_lightrag_checkpoint_suffix", f"_subgraph_{subgraph_tag}" if is_subgraph else "")

    triples_path = (
        triples_parquet_path(PROJECT_ROOT, scale_opt)
        if scale_opt
        else _resolve_project_path(str(args.triples_parquet))
    )
    if not triples_path.is_file():
        raise FileNotFoundError(f"三元组文件不存在: {triples_path}")
    triples_df_full = pd.read_parquet(triples_path)
    triples_pp_df = _load_policy_policy_triples(args, scale_opt)
    if is_subgraph:
        print(
            f"[LightRAG] eval_query_scope=subgraph_entities tag={subgraph_tag} triples={triples_path}",
            flush=True,
        )

    policy_titles = [str(x) for x in df_policies["title"].astype(str).tolist()]
    policy_docs = [
        f"{t}\n{str(c)[:1200]}"
        for t, c in zip(policy_titles, df_policies["content"].fillna("").astype(str).tolist())
    ]
    company_names = [str(x) for x in df_enterprises["name"].astype(str).tolist()]

    if args.max_policy_docs and args.max_policy_docs > 0:
        policy_docs = policy_docs[: args.max_policy_docs]
        policy_titles = policy_titles[: args.max_policy_docs]
    if args.max_company_docs and args.max_company_docs > 0:
        df_enterprises = df_enterprises.iloc[: args.max_company_docs].copy()
        company_names = [str(x) for x in df_enterprises["name"].astype(str).tolist()]

    policy_name_to_id = {
        t: int(pid) for t, pid in zip(policy_titles, df_policies["policy_id"].astype(int).tolist())
    }

    cache_root = PROJECT_ROOT / "reports" / "real_comparison_results"
    # triples/full_llm：DashScope 向量与本地 ST 向量维度不同，用 api / stgpu 子目录隔离 vdb
    _sg = f"_subgraph_{subgraph_tag}" if is_subgraph else ""
    if index_mode == "minimal":
        policy_dir = cache_root / f"lightrag_policy_workspace_{index_mode}{_sg}"
        company_dir = cache_root / f"lightrag_company_workspace_{index_mode}{_sg}"
    else:
        _emb_tag = "stgpu" if getattr(args, "local_gpu_embedding", False) else "api"
        policy_dir = cache_root / f"lightrag_policy_workspace_{index_mode}_{_emb_tag}{_sg}"
        company_dir = cache_root / f"lightrag_company_workspace_{index_mode}_{_emb_tag}{_sg}"
        print(f"[LightRAG] embedding_backend={_emb_tag}", flush=True)
    print(f"[LightRAG] policy working_dir: {policy_dir}", flush=True)
    print(f"[LightRAG] company working_dir: {company_dir}", flush=True)

    if skip_ingest:
        print(
            "[LightRAG] --skip_ingest：跳过索引写入，仅 initialize_storages 加载已有向量/图。",
            flush=True,
        )
        for nm, root, ws in (
            ("policy", policy_dir, "lr_policy"),
            ("company", company_dir, "lr_company"),
        ):
            vdb = root / ws / "vdb_chunks.json"
            if not vdb.is_file():
                raise FileNotFoundError(
                    f"--skip_ingest 需要已有索引目录，缺少 {nm} 向量库: {vdb}"
                )

    use_induced_v2 = (
        is_subgraph
        and str(getattr(args, "subgraph_eval_protocol", SUBGRAPH_EVAL_PROTOCOL_LEGACY)).strip()
        == SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2
    )
    query_set_meta: Optional[Dict[str, Any]] = None

    if use_induced_v2:
        data_root_lr = PROJECT_ROOT / "reports" / "real_comparison_data"
        openke_data_lr = data_root_lr / "openke_policykg"
        entity_token_map_lr = json.loads((data_root_lr / "entity_token_map.json").read_text(encoding="utf-8"))
        relation_token_map_lr = json.loads((data_root_lr / "relation_token_map.json").read_text(encoding="utf-8"))
        tok_eid_lr = _read_id_map(openke_data_lr / "entity2id.txt")
        tok_rid_lr = _read_id_map(openke_data_lr / "relation2id.txt")
        sr_lr = tok_rid_lr[relation_token_map_lr["openke"]["supports"]]
        enterprise_queries, policy_queries, query_set_meta = build_subgraph_induced_eval_queries(
            openke_data=openke_data_lr,
            supports_rid=sr_lr,
            token_to_eid=tok_eid_lr,
            openke_raw_to_tok=entity_token_map_lr["openke"],
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
            f"[LightRAG] induced_v2: E->P {len(enterprise_queries)} P->E {len(policy_queries)}",
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
            for q in policy_queries:
                t = str(q["policy_title"])
                q["policy_id"] = int(policy_name_to_id.get(t, -1))
            print(
                f"[LightRAG] 子图过滤后: E->P {len(enterprise_queries)} P->E {len(policy_queries)}",
                flush=True,
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
    print(
        f"[LightRAG] E->P cap: company={policy_cap}, industry={policy_industry_cap}",
        flush=True,
    )

    def _pending_ep() -> Dict[str, Any]:
        return {
            "num_queries": 0,
            "average": _average_from_metric_rows([], 0),
            "per_query_metrics": [],
            "checkpoint_note": "enterprise_to_policy_not_run",
        }

    def _pending_pe() -> Dict[str, Any]:
        return {
            "num_queries": 0,
            "average": _average_from_metric_rows([], 0),
            "per_query_metrics": [],
            "checkpoint_note": "policy_to_enterprise_not_run",
        }

    if index_mode == "minimal":
        policy_rag = await _build_lightrag_minimal(
            policy_dir, "lr_policy", args.local_embedding_model, clean=clean_wd
        )
        company_rag = await _build_lightrag_minimal(
            company_dir, "lr_company", args.local_embedding_model, clean=clean_wd
        )
        if not skip_ingest:
            await _ingest_custom_kg(policy_rag, _custom_kg_for_policies(policy_docs, policy_titles), "policy")
            _save_lightrag_checkpoint(
                args,
                stage="after_policy_ingest",
                partial=True,
                index_mode=index_mode,
                policy_dir=policy_dir,
                company_dir=company_dir,
                triples_pp_df=triples_pp_df,
                policy_rag=policy_rag,
                policy_cap=policy_cap,
                policy_industry_cap=policy_industry_cap,
                enterprise_to_policy=_pending_ep(),
                policy_to_enterprise=_pending_pe(),
            )
            await _ingest_custom_kg(company_rag, _custom_kg_for_companies(df_enterprises), "company")
            _save_lightrag_checkpoint(
                args,
                stage="after_company_ingest",
                partial=True,
                index_mode=index_mode,
                policy_dir=policy_dir,
                company_dir=company_dir,
                triples_pp_df=triples_pp_df,
                policy_rag=policy_rag,
                policy_cap=policy_cap,
                policy_industry_cap=policy_industry_cap,
                enterprise_to_policy=_pending_ep(),
                policy_to_enterprise=_pending_pe(),
            )
    elif index_mode == "triples":
        policy_rag = await _build_lightrag_full_api(policy_dir, "lr_policy", args, api_key, clean=clean_wd)
        company_rag = await _build_lightrag_full_api(company_dir, "lr_company", args, api_key, clean=clean_wd)
        if not skip_ingest:
            kg_pol = _custom_kg_policies_from_triples(
                policy_docs, policy_titles, triples_df_full, triples_pp_df
            )
            kg_co = _custom_kg_companies_from_triples(df_enterprises, triples_df_full)
            await _ingest_custom_kg(policy_rag, kg_pol, "policy(triples→custom_kg)")
            _save_lightrag_checkpoint(
                args,
                stage="after_policy_ingest",
                partial=True,
                index_mode=index_mode,
                policy_dir=policy_dir,
                company_dir=company_dir,
                triples_pp_df=triples_pp_df,
                policy_rag=policy_rag,
                policy_cap=policy_cap,
                policy_industry_cap=policy_industry_cap,
                enterprise_to_policy=_pending_ep(),
                policy_to_enterprise=_pending_pe(),
            )
            await _ingest_custom_kg(company_rag, kg_co, "company(triples→custom_kg)")
            _save_lightrag_checkpoint(
                args,
                stage="after_company_ingest",
                partial=True,
                index_mode=index_mode,
                policy_dir=policy_dir,
                company_dir=company_dir,
                triples_pp_df=triples_pp_df,
                policy_rag=policy_rag,
                policy_cap=policy_cap,
                policy_industry_cap=policy_industry_cap,
                enterprise_to_policy=_pending_ep(),
                policy_to_enterprise=_pending_pe(),
            )
    else:
        policy_rag = await _build_lightrag_full_api(policy_dir, "lr_policy", args, api_key, clean=clean_wd)
        company_rag = await _build_lightrag_full_api(company_dir, "lr_company", args, api_key, clean=clean_wd)
        if not skip_ingest:
            await _ingest_ainsert(policy_rag, policy_docs, policy_titles, "policy")
            _save_lightrag_checkpoint(
                args,
                stage="after_policy_ingest",
                partial=True,
                index_mode=index_mode,
                policy_dir=policy_dir,
                company_dir=company_dir,
                triples_pp_df=triples_pp_df,
                policy_rag=policy_rag,
                policy_cap=policy_cap,
                policy_industry_cap=policy_industry_cap,
                enterprise_to_policy=_pending_ep(),
                policy_to_enterprise=_pending_pe(),
            )
            company_docs = [
                f"{n}\n{str(t)[:1200]}"
                for n, t in zip(
                    df_enterprises["name"].astype(str).tolist(),
                    df_enterprises["text_with_industry"].fillna("").astype(str).tolist(),
                )
            ]
            company_fps = [str(x) for x in df_enterprises["name"].astype(str).tolist()]
            await _ingest_ainsert(company_rag, company_docs, company_fps, "company")
            _save_lightrag_checkpoint(
                args,
                stage="after_company_ingest",
                partial=True,
                index_mode=index_mode,
                policy_dir=policy_dir,
                company_dir=company_dir,
                triples_pp_df=triples_pp_df,
                policy_rag=policy_rag,
                policy_cap=policy_cap,
                policy_industry_cap=policy_industry_cap,
                enterprise_to_policy=_pending_ep(),
                policy_to_enterprise=_pending_pe(),
            )

    policy_score_threshold = None if args.policy_score_threshold < 0 else float(args.policy_score_threshold)
    enterprise_score_threshold = None if args.enterprise_score_threshold < 0 else float(args.enterprise_score_threshold)

    use_manual_kw = index_mode == "minimal"
    qp_base = QueryParam(
        mode="hybrid",
        top_k=args.lightrag_top_k,
        chunk_top_k=args.lightrag_chunk_top_k,
        enable_rerank=(False if index_mode == "minimal" else not args.no_chunk_rerank),
        hl_keywords=[] if not use_manual_kw else [],
        ll_keywords=[] if not use_manual_kw else [],
    )

    policy_title_set = set(policy_titles)
    ep_metrics: List[Dict] = []
    masked_ep = 0
    print(f"[LightRAG] E->P queries={len(enterprise_queries)} (manual_kw={use_manual_kw})", flush=True)
    for i, q in enumerate(enterprise_queries, start=1):
        q_text = str(q["query"])
        q_type = q.get("type", "company_name")
        gt = [x for x in q["ground_truth"] if x in policy_title_set]
        if len(gt) == 0:
            masked_ep += 1
        if use_manual_kw:
            qparam = replace(qp_base, hl_keywords=[q_text], ll_keywords=[q_text])
        else:
            qparam = qp_base
        raw = await policy_rag.aquery_data(q_text, qparam)
        ranked = _ranked_from_query_data(raw)
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
        pred: List[str] = []
        seen: Set[str] = set()
        for t, _ in ranked:
            if t in policy_title_set and t not in seen:
                pred.append(t)
                seen.add(t)
        m = calculate_metrics(pred, gt)
        rm = calculate_ranking_metrics(pred, gt)
        ep_metrics.append({**m, **rm, "query": q_text, "query_type": q_type})
        if i % max(1, len(enterprise_queries) // 10) == 0 or i == len(enterprise_queries):
            print(f"[LightRAG][E->P] {i}/{len(enterprise_queries)}", flush=True)
            _save_lightrag_checkpoint(
                args,
                stage="ep_eval_progress",
                partial=True,
                index_mode=index_mode,
                policy_dir=policy_dir,
                company_dir=company_dir,
                triples_pp_df=triples_pp_df,
                policy_rag=policy_rag,
                policy_cap=policy_cap,
                policy_industry_cap=policy_industry_cap,
                enterprise_to_policy={
                    "num_queries": len(ep_metrics),
                    "average": _average_from_metric_rows(ep_metrics, masked_ep),
                    "per_query_metrics": list(ep_metrics),
                    "checkpoint_note": f"ep_progress_{i}_of_{len(enterprise_queries)}",
                },
                policy_to_enterprise=_pending_pe(),
            )

    _save_lightrag_checkpoint(
        args,
        stage="after_enterprise_to_policy_eval",
        partial=True,
        index_mode=index_mode,
        policy_dir=policy_dir,
        company_dir=company_dir,
        triples_pp_df=triples_pp_df,
        policy_rag=policy_rag,
        policy_cap=policy_cap,
        policy_industry_cap=policy_industry_cap,
        enterprise_to_policy={
            "num_queries": len(ep_metrics),
            "average": _average_from_metric_rows(ep_metrics, masked_ep),
            "per_query_metrics": list(ep_metrics),
            "checkpoint_note": "enterprise_to_policy_complete",
        },
        policy_to_enterprise=_pending_pe(),
    )

    company_name_set = set(company_names)
    direct_support_boost_map: Dict[str, Set[str]] = {}
    for row in triples_df_full.itertuples(index=False):
        if str(row.predicate).lower() == "supports":
            direct_support_boost_map.setdefault(str(row.subject), set()).add(str(row.object))

    pe_metrics: List[Dict] = []
    masked_pe = 0
    print(f"[LightRAG] P->E queries={len(policy_queries)}", flush=True)
    for i, q in enumerate(policy_queries, start=1):
        title = str(q["policy_title"])
        gt = [x for x in q["ground_truth"] if x in company_name_set]
        if len(gt) == 0:
            masked_pe += 1
        if use_manual_kw:
            qparam = replace(qp_base, hl_keywords=[title], ll_keywords=[title])
        else:
            qparam = qp_base
        raw = await company_rag.aquery_data(title, qparam)
        ranked = _ranked_from_query_data(raw)
        direct_set = direct_support_boost_map.get(title, set())
        boosted: List[Tuple[str, float]] = []
        for name, sc in ranked:
            boosted.append((name, float(sc) + (0.3 if name in direct_set else 0.0)))
        ranked = _apply_rank_cutoff(
            ranked_pairs=boosted,
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
            print(f"[LightRAG][P->E] {i}/{len(policy_queries)}", flush=True)
            _save_lightrag_checkpoint(
                args,
                stage="pe_eval_progress",
                partial=True,
                index_mode=index_mode,
                policy_dir=policy_dir,
                company_dir=company_dir,
                triples_pp_df=triples_pp_df,
                policy_rag=policy_rag,
                policy_cap=policy_cap,
                policy_industry_cap=policy_industry_cap,
                enterprise_to_policy={
                    "num_queries": len(ep_metrics),
                    "average": _average_from_metric_rows(ep_metrics, masked_ep),
                    "per_query_metrics": list(ep_metrics),
                    "checkpoint_note": "enterprise_to_policy_complete",
                },
                policy_to_enterprise={
                    "num_queries": len(pe_metrics),
                    "average": _average_from_metric_rows(pe_metrics, masked_pe),
                    "per_query_metrics": list(pe_metrics),
                    "checkpoint_note": f"pe_progress_{i}_of_{len(policy_queries)}",
                },
            )

    _save_lightrag_checkpoint(
        args,
        stage="after_policy_to_enterprise_eval",
        partial=True,
        index_mode=index_mode,
        policy_dir=policy_dir,
        company_dir=company_dir,
        triples_pp_df=triples_pp_df,
        policy_rag=policy_rag,
        policy_cap=policy_cap,
        policy_industry_cap=policy_industry_cap,
        enterprise_to_policy={
            "num_queries": len(ep_metrics),
            "average": _average_from_metric_rows(ep_metrics, masked_ep),
            "per_query_metrics": list(ep_metrics),
            "checkpoint_note": "enterprise_to_policy_complete",
        },
        policy_to_enterprise={
            "num_queries": len(pe_metrics),
            "average": _average_from_metric_rows(pe_metrics, masked_pe),
            "per_query_metrics": list(pe_metrics),
            "checkpoint_note": "policy_to_enterprise_complete",
        },
    )

    await policy_rag.finalize_storages()
    await company_rag.finalize_storages()

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

    notes = _build_lightrag_notes(args, index_mode, triples_pp_df)
    params = _build_parameters_payload(
        args,
        index_mode,
        policy_dir,
        company_dir,
        triples_pp_df,
        policy_rag,
        policy_cap,
        policy_industry_cap,
    )
    _lr_eval_scope = "full"
    if is_subgraph:
        _lr_eval_scope = "subgraph_induced_v2" if use_induced_v2 else "subgraph_entities"
    _lr_eval_query_set = (
        "induced_v2_test_supports"
        if use_induced_v2
        else ("legacy_main_protocol_subgraph_filtered" if is_subgraph else "legacy_main_protocol_full")
    )
    final_body: Dict[str, Any] = {
        "model": "LightRAG",
        "checkpoint_stage": "complete",
        "partial": False,
        "timestamp": datetime.now().isoformat(),
        "evaluation_protocol": "main_queries + test_split_gt + unified_cutoff",
        "eval_query_scope": _lr_eval_scope,
        "eval_query_set": _lr_eval_query_set,
        "scale_dir": scale_opt,
        "subgraph_tag": subgraph_tag if is_subgraph else None,
        "lightrag_notes": notes,
        "parameters": params,
        "enterprise_to_policy": {"num_queries": len(ep_metrics), "average": ep_avg},
        "policy_to_enterprise": {"num_queries": len(pe_metrics), "average": pe_avg},
    }
    if query_set_meta is not None:
        final_body["query_set_meta"] = query_set_meta
    try:
        cache_root = PROJECT_ROOT / "reports" / "real_comparison_results"
        cache_root.mkdir(parents=True, exist_ok=True)
        complete_txt = _safe_json_dumps(final_body)
        ck_suffix = getattr(args, "_lightrag_checkpoint_suffix", "") or ""
        complete_name = f"lightrag_checkpoint_complete{ck_suffix}.json"
        latest_name = f"lightrag_checkpoint_latest{ck_suffix}.json"
        (cache_root / complete_name).write_text(complete_txt, encoding="utf-8")
        (cache_root / latest_name).write_text(complete_txt, encoding="utf-8")
        rd = REPO_ROOT / "report"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / complete_name).write_text(complete_txt, encoding="utf-8")
        (rd / latest_name).write_text(complete_txt, encoding="utf-8")
        print(
            f"[LightRAG] 完整结果已同步为 {complete_name} / {latest_name}",
            flush=True,
        )
    except Exception as e:
        print(
            f"[LightRAG] 完整检查点写入失败（已忽略，仍将返回结果 dict）: {e}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
    return final_body


def main() -> None:
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_mode",
        choices=["triples", "full_llm", "minimal"],
        default="triples",
        help="triples=数据集三元组注入；full_llm=LightRAG 原生 ainsert 抽取；minimal=本地向量+占位图",
    )
    parser.add_argument(
        "--triples_parquet",
        type=str,
        default="data_intermediate/triples_policy_entity.parquet",
        help="政策-实体/行业三元组（supports、targetsIndustry、belongsTo 等）",
    )
    parser.add_argument(
        "--triples_policy_policy_parquet",
        type=str,
        default="data_intermediate/triples_policy_policy.parquet",
        help="政策-政策三元组（如 transmitsTo）；不存在时尝试 output/policy_policy_only.json",
    )
    parser.add_argument(
        "--no_policy_policy_triples",
        action="store_true",
        help="不合并政策-政策边（仅用 triples_policy_entity）",
    )
    parser.add_argument("--minimal_index", action="store_true", help="跳过 API：custom_kg + 本地向量 + 手写关键词")
    parser.add_argument("--top_k_policy", type=int, default=-1)
    parser.add_argument("--top_k_enterprise", type=int, default=-1)
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
    parser.add_argument("--api_key", type=str, default="", help="默认读 DASHSCOPE_API_KEY")
    parser.add_argument(
        "--llm_model",
        type=str,
        default="qwen-plus",
        help="LightRAG 全链路 LLM（抽取/摘要/关键词等）",
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument(
        "--embedding_api_model",
        type=str,
        default="text-embedding-v1",
        help="DashScope 兼容 Embeddings 模型名（与文档一致即可）",
    )
    parser.add_argument(
        "--embedding_base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=1536,
        help="须与 DashScope text-embedding-v1 等模型输出维一致（兼容接口常见为 1536）",
    )
    parser.add_argument(
        "--local_embedding_model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="--minimal_index 或 --local_gpu_embedding 时的 sentence-transformers 模型名",
    )
    parser.add_argument(
        "--local_gpu_embedding",
        action="store_true",
        help="索引与查询中的向量改用本机 sentence-transformers（优先 CUDA）；LLM 关键词与 ali_rerank 仍走 DashScope",
    )
    parser.add_argument(
        "--local_embedding_device",
        type=str,
        default="cuda",
        help="cuda / cpu / cuda:0；与 --local_gpu_embedding 联用",
    )
    parser.add_argument(
        "--local_embedding_batch_size",
        type=int,
        default=64,
        help="本地 encode 批大小；显存不足可改为 16～32",
    )
    parser.add_argument("--rerank_model", type=str, default="gte-rerank-v2")
    parser.add_argument(
        "--rerank_base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    )
    parser.add_argument("--no_chunk_rerank", action="store_true", help="关闭 DashScope 文本重排")
    parser.add_argument("--llm_max_async", type=int, default=2)
    parser.add_argument("--embedding_max_async", type=int, default=4)
    parser.add_argument(
        "--embedding_timeout",
        type=int,
        default=120,
        help="LightRAG 对单次 Embedding 调用的基础超时(秒)；内部 worker≈2×、健康检查≈2×+15。"
        " DashScope 较慢或网络重试时请加大（默认 30 会触发 >75s 强制终止）。也可用环境变量 EMBEDDING_TIMEOUT。",
    )
    parser.add_argument("--max_parallel_insert", type=int, default=2)
    parser.add_argument("--lightrag_top_k", type=int, default=80)
    parser.add_argument("--lightrag_chunk_top_k", type=int, default=800)
    parser.add_argument(
        "--clean_index",
        action="store_true",
        help="删除当前 index_mode 对应的 policy/company working_dir 后重建索引",
    )
    parser.add_argument(
        "--skip_ingest",
        action="store_true",
        help="复用已有 working_dir（vdb_chunks.json 等），不执行 ainsert_custom_kg / ainsert；与 --clean_index 互斥（指定时忽略 clean）",
    )
    parser.add_argument("--max_policy_docs", type=int, default=0, help=">0 时只索引前 N 条政策（调试）")
    parser.add_argument("--max_company_docs", type=int, default=0, help=">0 时只索引前 N 条企业（调试）")
    parser.add_argument(
        "--output",
        type=str,
        default="reports/real_comparison_results/lightrag_results_main_protocol.json",
        help="默认文件名含 main_protocol；full_llm 对照可另存如 lightrag_results_full_llm.json",
    )
    parser.add_argument(
        "--scale_dir",
        type=str,
        default="",
        help="子图目录（含 policies_clean / enterprises_filtered / triples_*.parquet）；非空时 eval_query_scope=subgraph_entities，索引与检查点与全量隔离",
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
    args = parser.parse_args()

    print(f"[LightRAG] Python 解释器: {sys.executable}", flush=True)

    if not LIGHTRAG_SRC.is_dir():
        raise RuntimeError(f"LightRAG 源码目录不存在: {LIGHTRAG_SRC}")

    result = asyncio.run(_run_eval_async(args))

    try:
        out_txt = _safe_json_dumps(result)
    except Exception as e:
        print(f"[LightRAG] 主结果 JSON 序列化失败: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_txt, encoding="utf-8")
    print(out_txt)

    report_dir = REPO_ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        (report_dir / Path(args.output).name).write_text(out_txt, encoding="utf-8")
    except Exception as e:
        print(f"[LightRAG] report/ 同步失败（主输出已写入）: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
