#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KG-BERT（三元组分类 BERT）在主实验口径下的双向匹配评测，与 OpenKE / ATISE 对齐。
使用微调后 checkpoint 对 (头文本, 关系文本, 尾文本) 打分，取正类 softmax 概率为相似度。
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
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
KG_BERT_ROOT = REPO_ROOT / "KG-BERT" / "kg-bert"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(KG_BERT_ROOT))

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

from pytorch_pretrained_bert.modeling import BertForSequenceClassification  # type: ignore
from pytorch_pretrained_bert.tokenization import BertTokenizer  # type: ignore

from run_bert_link_prediction import (  # type: ignore
    InputExample,
    convert_examples_to_features,
    KGProcessor,
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


def _bert_prob_positive(
    model: torch.nn.Module,
    device: torch.device,
    tokenizer,
    label_list: List[str],
    max_seq_length: int,
    examples: List,
    batch_size: int,
) -> np.ndarray:
    features = convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, print_info=False
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    loader = DataLoader(data, sampler=SequentialSampler(data), batch_size=batch_size)
    out_chunks: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids, input_mask, segment_ids, _ = tuple(t.to(device) for t in batch)
            logits = model(input_ids, segment_ids, input_mask, labels=None)
            prob = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
            out_chunks.append(prob.detach().cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kgbert_data_dir", type=str, default="", help="含 train.tsv / entity2text.txt 的目录，默认 real_comparison_data/kgbert_policykg")
    parser.add_argument("--output_dir", type=str, default="", help="微调输出目录（含 pytorch_model.bin），默认 reports/.../kgbert_policykg_out")
    parser.add_argument("--bert_model", type=str, default="bert-base-chinese")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
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
    parser.add_argument("--output", type=str, default="reports/real_comparison_results/kgbert_matching_eval_testsplit.json")
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

    data_root = PROJECT_ROOT / "reports" / "real_comparison_data"
    kgbert_data = Path(args.kgbert_data_dir) if args.kgbert_data_dir else data_root / "kgbert_policykg"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "reports" / "real_comparison_results" / "kgbert_policykg_out"
    from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME  # type: ignore

    if not (output_dir / WEIGHTS_NAME).is_file():
        raise FileNotFoundError(f"未找到 {WEIGHTS_NAME}，请先完成 KG-BERT 微调: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    processor = KGProcessor()
    label_list = processor.get_labels(str(kgbert_data))
    num_labels = len(label_list)
    tokenizer = BertTokenizer.from_pretrained(str(output_dir), do_lower_case=args.do_lower_case)
    model = BertForSequenceClassification.from_pretrained(str(output_dir), num_labels=num_labels)
    model.to(device)

    entity_token_map = json.loads((data_root / "entity_token_map.json").read_text(encoding="utf-8"))
    relation_token_map = json.loads((data_root / "relation_token_map.json").read_text(encoding="utf-8"))
    kgbert_tok = entity_token_map["kgbert"]
    supports_tok = relation_token_map["kgbert"]["supports"]

    openke_data = data_root / "openke_policykg"
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

    raw_to_kgbert_tok: Dict[str, str] = kgbert_tok
    ent2text_lines = (kgbert_data / "entity2text.txt").read_text(encoding="utf-8").splitlines()
    ent2text: Dict[str, str] = {}
    for ln in ent2text_lines:
        p = ln.strip().split("\t")
        if len(p) >= 2:
            ent2text[p[0]] = p[1]
    rel2text_lines = (kgbert_data / "relation2text.txt").read_text(encoding="utf-8").splitlines()
    rel2text: Dict[str, str] = {}
    for ln in rel2text_lines:
        p = ln.strip().split("\t")
        if len(p) >= 2:
            rel2text[p[0]] = p[1]

    supports_rel_text = rel2text.get(supports_tok, "supports")

    policy_candidates: List[Tuple[str, str]] = [(t, raw_to_kgbert_tok[t]) for t in policy_titles if t in raw_to_kgbert_tok]
    company_candidates: List[Tuple[str, str]] = [(n, raw_to_kgbert_tok[n]) for n in company_names if n in raw_to_kgbert_tok]
    policy_names_arr = np.array([x[0] for x in policy_candidates], dtype=object)
    policy_tok_arr = np.array([x[1] for x in policy_candidates], dtype=object)
    company_names_arr = np.array([x[0] for x in company_candidates], dtype=object)
    company_tok_arr = np.array([x[1] for x in company_candidates], dtype=object)

    itc_full = industry_to_companies_full_map(df_ent_full)
    use_induced_v2 = (
        is_subgraph
        and str(getattr(args, "subgraph_eval_protocol", SUBGRAPH_EVAL_PROTOCOL_LEGACY)).strip()
        == SUBGRAPH_EVAL_PROTOCOL_INDUCED_V2
    )
    query_set_meta: Optional[Dict] = None

    industry_to_company_toks: Dict[str, List[str]] = {}
    if use_induced_v2:
        if "industry" in df_enterprises.columns:
            for ind, grp in df_enterprises.groupby("industry"):
                toks = [
                    raw_to_kgbert_tok[str(n)]
                    for n in grp["name"].astype(str).tolist()
                    if str(n) in raw_to_kgbert_tok
                ]
                if toks:
                    industry_to_company_toks[str(ind)] = toks
    else:
        for ind, names in itc_full.items():
            toks = [
                raw_to_kgbert_tok[str(n)]
                for n in names
                if str(n) in valid_company_names and str(n) in raw_to_kgbert_tok
            ]
            if toks:
                industry_to_company_toks[str(ind)] = toks

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
            f"[KG-BERT-Eval] induced_v2: company_ep={query_set_meta['n_company_ep_queries']} "
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
                openke_raw_to_tok=entity_token_map["openke"],
            )
            industry_to_companies = itc_full
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
                itc_full,
            )

    company_gt_sizes = [len(q["ground_truth"]) for q in enterprise_queries if q.get("type", "company_name") != "industry"]
    industry_gt_sizes = [len(q["ground_truth"]) for q in enterprise_queries if q.get("type", "company_name") == "industry"]
    inferred_policy_cap = max(1, int(round(float(np.mean(company_gt_sizes))))) if company_gt_sizes else 25
    inferred_ind_cap = max(1, int(round(float(np.mean(industry_gt_sizes))))) if industry_gt_sizes else inferred_policy_cap
    policy_cap = args.policy_max_output_cap if args.policy_max_output_cap > 0 else inferred_policy_cap
    policy_industry_cap = (
        args.policy_industry_query_max_output_cap if args.policy_industry_query_max_output_cap > 0 else inferred_ind_cap
    )

    policy_direct_support: Dict[str, Set[str]] = {}
    if args.ground_truth_source == "full":
        tdf = pd.read_parquet(triples_parquet_path(PROJECT_ROOT, scale_opt))
        for row in tdf.itertuples(index=False):
            if str(row.predicate).lower() == "supports":
                policy_direct_support.setdefault(str(row.subject), set()).add(str(row.object))

    n_pol = len(policy_tok_arr)
    n_com = len(company_tok_arr)

    def text_for_tok(tok: str) -> str:
        return ent2text.get(tok, tok)

    print(f"[KG-BERT-Eval] E->P queries={len(enterprise_queries)}", flush=True)
    ep_metrics: List[Dict] = []
    ep_masked = 0

    for i, q in enumerate(enterprise_queries, start=1):
        q_text = str(q["query"])
        q_type = q.get("type", "company_name")
        gt_raw = [x for x in q["ground_truth"] if x in valid_policy_titles]
        if not gt_raw:
            ep_masked += 1

        if q_type == "industry" and q_text in industry_to_company_toks:
            ctoks = industry_to_company_toks[q_text]
            stacks = []
            for ct in ctoks:
                ex = [
                    InputExample(
                        guid=str(j),
                        text_a=text_for_tok(policy_tok_arr[j]),
                        text_b=supports_rel_text,
                        text_c=text_for_tok(ct),
                        label="1",
                    )
                    for j in range(n_pol)
                ]
                prob = _bert_prob_positive(model, device, tokenizer, label_list, args.max_seq_length, ex, args.eval_batch_size)
                stacks.append(prob)
            score = np.max(np.stack(stacks, axis=0), axis=0)
        elif q_text in raw_to_kgbert_tok:
            ct = raw_to_kgbert_tok[q_text]
            ex = [
                InputExample(
                    guid=str(j),
                    text_a=text_for_tok(policy_tok_arr[j]),
                    text_b=supports_rel_text,
                    text_c=text_for_tok(ct),
                    label="1",
                )
                for j in range(n_pol)
            ]
            score = _bert_prob_positive(model, device, tokenizer, label_list, args.max_seq_length, ex, args.eval_batch_size)
        else:
            score = np.array([])

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
            print(f"[KG-BERT-Eval][E->P] {i}/{len(enterprise_queries)}", flush=True)

    print(f"[KG-BERT-Eval] P->E queries={len(policy_queries)}", flush=True)
    pe_metrics: List[Dict] = []
    pe_masked = 0

    for i, q in enumerate(policy_queries, start=1):
        title = str(q["policy_title"])
        gt_raw = [x for x in q["ground_truth"] if x in valid_company_names]
        if not gt_raw:
            pe_masked += 1
        if title not in raw_to_kgbert_tok:
            pred_names = []
        else:
            pt = raw_to_kgbert_tok[title]
            ex = [
                InputExample(
                    guid=str(j),
                    text_a=text_for_tok(pt),
                    text_b=supports_rel_text,
                    text_c=text_for_tok(company_tok_arr[j]),
                    label="1",
                )
                for j in range(n_com)
            ]
            score = _bert_prob_positive(model, device, tokenizer, label_list, args.max_seq_length, ex, args.eval_batch_size)
            order = np.argsort(score)[::-1]
            cands = order[: min(args.enterprise_candidate_k, len(order))]
            ranked_c = [(int(j), float(score[j])) for j in cands]
            boost = policy_direct_support.get(title, set()) if args.ground_truth_source == "full" else set()
            if boost:
                ranked_c = [(j, s + (args.direct_support_boost if str(company_names_arr[j]) in boost else 0.0)) for j, s in ranked_c]
            ranked_c = _apply_rank_cutoff(
                ranked_c,
                args.top_k_enterprise,
                enterprise_score_threshold,
                args.enterprise_adaptive_quantile,
                args.enterprise_relative_drop_threshold,
                args.enterprise_max_output_cap,
            )
            pred_names = [str(company_names_arr[j]) for j, _ in ranked_c]
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
            print(f"[KG-BERT-Eval][P->E] {i}/{len(policy_queries)}", flush=True)

    ep_block, pe_block = enterprise_policy_result_blocks(ep_metrics, pe_metrics, ep_masked, pe_masked)

    _eval_scope = "full"
    if is_subgraph:
        _eval_scope = "subgraph_induced_v2" if use_induced_v2 else "subgraph_entities"
    _eval_query_set = (
        "induced_v2_test_supports"
        if use_induced_v2
        else ("legacy_main_protocol_subgraph_filtered" if is_subgraph else "legacy_main_protocol_full")
    )
    result = {
        "model": "KG-BERT",
        "evaluation_protocol": "main_queries + test_split_gt + unified_cutoff",
        "eval_query_scope": _eval_scope,
        "eval_query_set": _eval_query_set,
        "scale_dir": scale_opt,
        "subgraph_tag": subgraph_tag if is_subgraph else None,
        "timestamp": datetime.now().isoformat(),
        "checkpoint_dir": str(output_dir),
        "kgbert_data_dir": str(kgbert_data),
        "parameters": {
            "bert_model": args.bert_model,
            "max_seq_length": args.max_seq_length,
            "ground_truth_source": args.ground_truth_source,
            "subgraph_eval_protocol": (args.subgraph_eval_protocol if is_subgraph else None),
            "min_company_ep_queries": int(args.min_company_ep_queries),
            "min_industry_ep_queries": int(args.min_industry_ep_queries),
            "min_pe_queries": int(args.min_pe_queries),
            "policy_max_output_cap": policy_cap,
            "policy_industry_query_max_output_cap": policy_industry_cap,
            "enterprise_max_output_cap": args.enterprise_max_output_cap,
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
    repo_name = out_path.name
    if is_subgraph and not repo_name.startswith("kgbert_matching_eval_subgraph_"):
        repo_name = f"kgbert_matching_eval_subgraph_{subgraph_tag}.json"
    (REPO_ROOT / "report" / repo_name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
