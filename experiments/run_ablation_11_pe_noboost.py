#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`run_ablation_11.py` 的副本：逻辑相同，但 P→E 评估固定 `direct_support_boost=0`，
去掉 supports 先验加分，使 A/B/D 等仅改 E→P 管线时 P→E 指标也能反映排序变化（打破锁榜）。

不修改原脚本；输出独立文件：
- reports/ablation_results_11_pe_noboost.csv
- reports/ablation_results_11_pe_noboost.md
- reports/ablation_beta_sensitivity_pe_noboost.csv
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from matching.torch_metadata_fix import apply_torch_metadata_fix  # noqa: E402

apply_torch_metadata_fix()

import torch
import torch.nn.functional as F

# 重新生成 ablation_results_11.md 时自动附加（勿删：解释 P→E 不变与 C1 的 NDCG 现象）
ABLATION_RESULTS_MD_APPENDIX = r"""

## 关键结论速览（便于汇报）

| 组别 | 代表实验 | 主要现象 | 对基线含义 |
| --- | --- | --- | --- |
| BERT嵌入策略 | A1/A2/A3 | E→P 指标均低于 BASE | 当前基线的文本表征组合更稳健 |
| 多模态融合 | B1/B2/B3 | 去掉层级/时间衰减后明显下降，B3 跌幅最大 | 衰减与多源融合均必要 |
| 图结构表征 | C1/C2 | **旧版 C1 因未重归一融合权导致 NDCG 不可比**；已改为 semantic=0.9/0/0.1 后须重跑表；C2 整体低于 BASE | 以重跑后 C1 为准 |
| 衰减机制 | D1/D2/D3 | no decay / linear decay 均低于 BASE | 当前指数衰减与参数更优 |

## delta 列解释

- `delta_ep_ndcg`：该实验 **E→P 的 NDCG** 减 BASE 的 NDCG。
- `delta_pe_recall`：该实验 **P→E 的 Recall** 减 BASE 的 Recall。
- `>0` 优于基线，`<0` 劣于基线。

## 质疑 1：为何只有 C1、C2 改变 P→E，其余 P→E 完全不变？这合理吗？

从**“希望一次消融撼动整条系统”**的直觉上看，会显得不合理；但从**当前实现**看，这是**接线方式导致的必然结果**，不是 CSV 算错或评估 bug。

P→E 评估走 `retrieve_enterprises_by_policy` → `encode_subgraph_with_gnn`：**主信号是 GAT（对比或替换后的 GCN）政策–企业嵌入 + `direct_support_boost`**；GAT 清空时才回退企业文本嵌入。**该路径不读取**政策 BERT 向量，也**不读取** `policy_importance_scores`（B/D 组改动点），也**不接收** E→P 的 `policy_semantic_weight` 等融合参数（B3）。

因此 A/B/D 组只改 E→P 侧对象时，P→E 的候选与排序与 BASE **完全一致**，各 P→E 指标会逐字段相同。只有 **C1（清空 GAT）**、**C2（替换 GAT 向量）** 会改变 P→E。

**论文中可写的表述**：本批 11 项主要针对 **E→P 管线**；对 **P→E** 的系统消融需另设实验（例如 `direct_support_boost`、`enterprise_adaptive_quantile`、`candidate_k`、子图 `k_hop`、企业嵌入路径等）。

## 质疑 2：旧版 C1 的 NDCG 为何高于主实验？（已修正）

旧版仅设 `policy_structure_weight=0` 而保留 semantic=0.45、importance=0.1，融合总权仅 **0.55**，与 BASE 的 **1.0** 标尺不一致，分位数截断行为改变，**NDCG/MAP 与主实验不可比**，易出现虚高。

**修正**：去掉图结构时重归一为 **semantic=0.9、structure=0、importance=0.1**。见脚本中 C1 的 `params`；可用 `python scripts/run_ablation_11.py --quick-c1` 快速对比 JSON BASE。预期公平对比下 C1 不优于 BASE。

## 附：P→E 代码路径核对（摘要）

| 组别 | 脚本改动 | 为何多数情况下 P→E 不变 |
| --- | --- | --- |
| A1–A3 | 仅政策 BERT 路径 | P→E 不用政策 BERT |
| B1–B2, D1–D3 | `policy_importance_scores` | 只参与 E→P |
| B3 | E→P 融合权重 | 评估 P→E 时不传这些参数 |
| C1 | 清空 GAT；E→P 须 **0.9/0/0.1** 重归一 | P→E 与 E→P 图信号均变 |
| C2 | 替换为非对比 GAT 向量 | P→E 排序变 |

> 说明：本附录由 `run_ablation_11_pe_noboost.py` 生成；数据表见 `ablation_results_11_pe_noboost.csv`。本批 **P→E 已设 `direct_support_boost=0`**，与原版锁榜现象不同，应对照阅读。
"""

from matching.bidirectional_matching import BidirectionalMatcher  # noqa: E402
from matching.evaluate_matching import (  # noqa: E402
    build_test_queries_from_data,
    evaluate_enterprise_to_policy,
    evaluate_policy_to_enterprise,
)


def _norm01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    mx = float(arr.max())
    mn = float(arr.min())
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mn) / (mx - mn)


def _load_base_importance_df() -> pd.DataFrame:
    p = PROJECT_ROOT / "evaluation" / "policy_importance_with_decay.parquet"
    df = pd.read_parquet(p)
    need_cols = ["policy_id", "base_importance", "delta_level", "delta_time", "combined_decayed"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"缺少字段: {c} in {p}")
    return df.copy()


def _importance_map_from_mode(mode: str, beta_time: float = 0.05, beta_level: float = 0.2) -> Dict[int, float]:
    df = _load_base_importance_df()
    base = df["base_importance"].to_numpy(dtype=np.float64)
    dl = df["delta_level"].to_numpy(dtype=np.float64)
    dt = df["delta_time"].to_numpy(dtype=np.float64)

    if mode == "default_combined":
        score = df["combined_decayed"].to_numpy(dtype=np.float64)
    elif mode == "no_hierarchy":
        score = base * np.exp(-beta_time * dt)
    elif mode == "no_time":
        score = base * np.exp(-beta_level * dl)
    elif mode == "no_decay":
        score = base
    elif mode == "linear_decay":
        # 线性衰减替代指数衰减
        lam_t = beta_time
        lam_l = beta_level
        t_term = np.maximum(0.0, 1.0 - lam_t * dt)
        l_term = np.maximum(0.0, 1.0 - lam_l * dl)
        score = base * t_term * l_term
    elif mode == "beta_time_exp":
        score = base * np.exp(-beta_level * dl) * np.exp(-beta_time * dt)
    else:
        raise ValueError(f"未知importance模式: {mode}")

    score = _norm01(score)
    out = {}
    for pid, s in zip(df["policy_id"].astype(int).tolist(), score.tolist()):
        out[int(pid)] = float(s)
    return out


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def _encode_texts(texts: List[str], tokenizer, model, device, batch_size: int = 16, max_length: int = 256) -> np.ndarray:
    embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = model(**encoded)
            pooled = _mean_pooling(out, encoded["attention_mask"])
            embs.append(pooled.cpu())
    return torch.cat(embs, dim=0).numpy()


def ensure_joint_bert_policy_embeddings() -> Dict[str, str]:
    emb_path = PROJECT_ROOT / "embeddings" / "policy_text_joint_bert_emb.npy"
    idx_path = PROJECT_ROOT / "embeddings" / "policy_text_joint_bert_index.json"
    if emb_path.exists() and idx_path.exists():
        return {"policy_emb_path": "embeddings/policy_text_joint_bert_emb.npy", "policy_index_path": "embeddings/policy_text_joint_bert_index.json"}

    from transformers import AutoModel, AutoTokenizer

    print("生成变体B文本级拼接后编码向量（policy_text_joint_bert_emb.npy）...")
    df = pd.read_parquet(PROJECT_ROOT / "data_intermediate" / "policies_clean.parquet")
    titles = df["title"].fillna("").astype(str).tolist()
    contents = df["content"].fillna("").astype(str).tolist()
    merged = [f"{t} [SEP] {c if c.strip() else t}" for t, c in zip(titles, contents)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", local_files_only=False)
    model = AutoModel.from_pretrained("bert-base-chinese", local_files_only=False).to(device)
    embs = _encode_texts(merged, tokenizer, model, device)

    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embs)
    idx_map = {int(pid): int(i) for i, pid in enumerate(df["policy_id"].astype(int).tolist())}
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(idx_map, f, ensure_ascii=False, indent=2)

    return {"policy_emb_path": "embeddings/policy_text_joint_bert_emb.npy", "policy_index_path": "embeddings/policy_text_joint_bert_index.json"}


def _load_current_best_metrics() -> Dict[str, float]:
    p = PROJECT_ROOT / "matching" / "evaluation_results_a_ndcg_best_current.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    ep = obj["enterprise_to_policy"]["average"]
    pe = obj["policy_to_enterprise"]["average"]
    return {
        "ep_precision": float(ep["precision"]),
        "ep_recall": float(ep["recall"]),
        "ep_f1": float(ep["f1"]),
        "ep_map": float(ep["map"]),
        "ep_ndcg": float(ep["ndcg"]),
        "pe_precision": float(pe["precision"]),
        "pe_recall": float(pe["recall"]),
        "pe_f1": float(pe["f1"]),
        "pe_map": float(pe["map"]),
        "pe_ndcg": float(pe["ndcg"]),
    }


def _md_escape(v: Any) -> str:
    s = str(v)
    return s.replace("|", "\\|")


def _df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals: List[str] = []
        for c in cols:
            x = row[c]
            if isinstance(x, (float, np.floating)):
                vals.append(format(float(x), floatfmt))
            else:
                vals.append(_md_escape(x))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _base_eval_params() -> Dict[str, Any]:
    # E→P 与 run_ablation_11.py / round4 主实验一致；P→E 仍 direct_support_boost=0
    return {
        # E->P
        "top_k_policy": -1,
        "policy_candidate_k": 1000,
        "policy_score_threshold": None,
        "policy_adaptive_quantile": 0.66,
        "policy_relative_drop_threshold": 0.12,
        "policy_max_output_cap": 160,
        "policy_semantic_weight": 0.7,
        "policy_structure_weight": 0.18,
        "policy_importance_weight": 0.12,
        "policy_industry_boost": 0.12,
        "policy_industry_query_adaptive_quantile": 0.82,
        "policy_industry_query_relative_drop_threshold": 0.12,
        "policy_industry_query_max_output_cap": 70,
        # P->E
        "top_k_enterprise": -1,
        "enterprise_candidate_k": 1000,
        "enterprise_score_threshold": None,
        "enterprise_adaptive_quantile": 0.58,
        "enterprise_relative_drop_threshold": 0.18,
        "enterprise_max_output_cap": 150,
        "direct_support_boost": 0.0,
    }


def _evaluate(
    matcher: BidirectionalMatcher,
    enterprise_queries: List[Dict[str, Any]],
    policy_queries: List[Dict[str, Any]],
    valid_policy_ids,
    params: Dict[str, Any],
) -> Dict[str, float]:
    with redirect_stdout(io.StringIO()):
        ep = evaluate_enterprise_to_policy(
            matcher=matcher,
            queries=enterprise_queries,
            top_k=params["top_k_policy"],
            candidate_k=params["policy_candidate_k"],
            score_threshold=params["policy_score_threshold"],
            adaptive_quantile=params["policy_adaptive_quantile"],
            relative_drop_threshold=params["policy_relative_drop_threshold"],
            max_output_cap=params["policy_max_output_cap"],
            semantic_weight=params["policy_semantic_weight"],
            structure_weight=params["policy_structure_weight"],
            importance_weight=params["policy_importance_weight"],
            industry_boost=params["policy_industry_boost"],
            industry_query_adaptive_quantile=params["policy_industry_query_adaptive_quantile"],
            industry_query_relative_drop_threshold=params["policy_industry_query_relative_drop_threshold"],
            industry_query_max_output_cap=params["policy_industry_query_max_output_cap"],
            valid_policy_ids=valid_policy_ids,
        )
        pe = evaluate_policy_to_enterprise(
            matcher=matcher,
            queries=policy_queries,
            top_k=params["top_k_enterprise"],
            candidate_k=params["enterprise_candidate_k"],
            score_threshold=params["enterprise_score_threshold"],
            adaptive_quantile=params["enterprise_adaptive_quantile"],
            relative_drop_threshold=params["enterprise_relative_drop_threshold"],
            max_output_cap=params["enterprise_max_output_cap"],
            direct_support_boost=params["direct_support_boost"],
        )
    ep_avg = ep["average"]
    pe_avg = pe["average"]
    return {
        "ep_precision": float(ep_avg["precision"]),
        "ep_recall": float(ep_avg["recall"]),
        "ep_f1": float(ep_avg["f1"]),
        "ep_map": float(ep_avg["map"]),
        "ep_ndcg": float(ep_avg["ndcg"]),
        "pe_precision": float(pe_avg["precision"]),
        "pe_recall": float(pe_avg["recall"]),
        "pe_f1": float(pe_avg["f1"]),
        "pe_map": float(pe_avg["map"]),
        "pe_ndcg": float(pe_avg["ndcg"]),
    }


def _build_matcher(policy_emb_path: str, policy_index_path: str, company_emb_path: str, company_index_path: str) -> BidirectionalMatcher:
    return BidirectionalMatcher(
        PROJECT_ROOT,
        policy_emb_path=policy_emb_path,
        policy_index_path=policy_index_path,
        company_emb_path=company_emb_path,
        company_index_path=company_index_path,
    )


def _apply_importance_map(matcher: BidirectionalMatcher, imp_map: Dict[int, float]) -> None:
    matcher.policy_query_matcher.policy_importance_scores = imp_map


def _disable_gnn(matcher: BidirectionalMatcher) -> None:
    matcher.policy_query_matcher.policy_structure_scores = {}
    matcher.enterprise_retriever.gat_policy_emb = None
    matcher.enterprise_retriever.gat_company_emb = None


def _replace_with_non_contrastive_graph_emb(matcher: BidirectionalMatcher) -> None:
    p_path = PROJECT_ROOT / "graph" / "gat_policy_emb.npy"
    c_path = PROJECT_ROOT / "graph" / "gat_company_emb.npy"
    if not p_path.exists() or not c_path.exists():
        return
    p = np.load(p_path)
    c = np.load(c_path)
    device = matcher.enterprise_retriever.device
    tp = torch.tensor(p, dtype=torch.float32, device=device)
    tc = torch.tensor(c, dtype=torch.float32, device=device)
    matcher.enterprise_retriever.gat_policy_emb = F.normalize(tp, p=2, dim=1)
    matcher.enterprise_retriever.gat_company_emb = F.normalize(tc, p=2, dim=1)
    # 更新E->P结构分数
    pn = np.linalg.norm(p, axis=1)
    pn = _norm01(pn)
    matcher.policy_query_matcher.policy_structure_scores = {int(i): float(v) for i, v in enumerate(pn.tolist())}


def _snapshot_matcher_state(matcher: BidirectionalMatcher) -> Dict[str, Any]:
    pr = matcher.policy_query_matcher
    er = matcher.enterprise_retriever
    snap = {
        "policy_structure_scores": dict(pr.policy_structure_scores),
        "policy_importance_scores": dict(pr.policy_importance_scores),
        "gat_policy_emb": er.gat_policy_emb.clone() if er.gat_policy_emb is not None else None,
        "gat_company_emb": er.gat_company_emb.clone() if er.gat_company_emb is not None else None,
    }
    return snap


def _restore_matcher_state(matcher: BidirectionalMatcher, snap: Dict[str, Any]) -> None:
    pr = matcher.policy_query_matcher
    er = matcher.enterprise_retriever
    pr.policy_structure_scores = dict(snap["policy_structure_scores"])
    pr.policy_importance_scores = dict(snap["policy_importance_scores"])
    er.gat_policy_emb = snap["gat_policy_emb"].clone() if snap["gat_policy_emb"] is not None else None
    er.gat_company_emb = snap["gat_company_emb"].clone() if snap["gat_company_emb"] is not None else None


def _set_eval_seeds(seed: int = 42, *, cudnn_deterministic: bool = False) -> None:
    """
    固定 Python/NumPy/CPU 随机种子。
    默认 **不** 开启 cuDNN deterministic：在 CUDA + BERT 推理下会显著改变数值路径，
    曾导致「现场 BASE」远低于 `evaluation_results_*_best_current.json` 的历史结果，误判主实验变差。
    若需可复现实验再显式传 cudnn_deterministic=True（可能仍与旧 JSON 不完全一致）。
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
        torch.backends.cudnn.benchmark = not bool(cudnn_deterministic)


def run_quick_align_base_c1(seed: int = 42, *, cudnn_deterministic: bool = False) -> None:
    """
    同批、同查询、同 matcher 快照下重算「主实验 BASE」与「C1」，避免与历史 JSON 混比。
    历史表中 BASE 来自 evaluation_results_a_ndcg_best_current.json，与现场 _evaluate 可能因
    版本/随机性不一致，导致 C1 看起来「远高于主实验」。
    """
    print(
        f"对齐评估: seed={seed}, cudnn_deterministic={cudnn_deterministic}，"
        f"同批重算 BASE(全图) 与 C1(w/o GNN, 0.9/0/0.1)…"
    )
    _set_eval_seeds(seed, cudnn_deterministic=cudnn_deterministic)

    enterprise_queries, policy_queries = build_test_queries_from_data(
        max_enterprise_queries=300, max_industry_queries=30, max_policy_queries=200
    )
    df_pol = pd.read_parquet(PROJECT_ROOT / "data_intermediate" / "policies_clean.parquet")
    valid_policy_ids = set(df_pol["policy_id"].astype(int).tolist())
    base_params = _base_eval_params()
    frozen = _load_current_best_metrics()

    default_paths = {
        "policy_emb_path": "embeddings/policy_text_concat_emb.npy",
        "policy_index_path": "embeddings/policy_index.json",
        "company_emb_path": "embeddings/enterprise_text_emb.npy",
        "company_index_path": "embeddings/enterprise_index.json",
    }
    matcher = _build_matcher(**default_paths)
    snap = _snapshot_matcher_state(matcher)

    _set_eval_seeds(seed, cudnn_deterministic=cudnn_deterministic)
    _restore_matcher_state(matcher, snap)
    m_base = _evaluate(matcher, enterprise_queries, policy_queries, valid_policy_ids, dict(base_params))

    c1_params = dict(base_params)
    c1_params.update(
        {
            "policy_semantic_weight": 0.9,
            "policy_structure_weight": 0.0,
            "policy_importance_weight": 0.1,
        }
    )
    _set_eval_seeds(seed, cudnn_deterministic=cudnn_deterministic)
    _restore_matcher_state(matcher, snap)
    _disable_gnn(matcher)
    m_c1 = _evaluate(matcher, enterprise_queries, policy_queries, valid_policy_ids, c1_params)

    def _sub(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
        keys = [
            "ep_precision",
            "ep_recall",
            "ep_f1",
            "ep_map",
            "ep_ndcg",
            "pe_precision",
            "pe_recall",
            "pe_f1",
            "pe_map",
            "pe_ndcg",
        ]
        return {k: float(a[k]) - float(b[k]) for k in keys}

    delta_c1_vs_live = _sub(m_c1, m_base)
    out = {
        "seed": seed,
        "frozen_json_base": frozen,
        "live_base_same_script": m_base,
        "live_c1": m_c1,
        "delta_live_c1_minus_live_base": delta_c1_vs_live,
    }
    out_path = PROJECT_ROOT / "reports" / "ablation_base_c1_aligned_pe_noboost.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n--- 冻结 JSON 中的 BASE（旧报告，非本次运行）---")
    print(
        f"E->P P={frozen['ep_precision']:.4f} R={frozen['ep_recall']:.4f} F1={frozen['ep_f1']:.4f} "
        f"MAP={frozen['ep_map']:.4f} NDCG={frozen['ep_ndcg']:.4f}"
    )
    print("\n--- 本次同脚本现场 BASE（主实验全图）---")
    print(
        f"E->P P={m_base['ep_precision']:.4f} R={m_base['ep_recall']:.4f} F1={m_base['ep_f1']:.4f} "
        f"MAP={m_base['ep_map']:.4f} NDCG={m_base['ep_ndcg']:.4f}"
    )
    print("\n--- 本次同脚本 C1 ---")
    print(
        f"E->P P={m_c1['ep_precision']:.4f} R={m_c1['ep_recall']:.4f} F1={m_c1['ep_f1']:.4f} "
        f"MAP={m_c1['ep_map']:.4f} NDCG={m_c1['ep_ndcg']:.4f}"
    )
    print("\n--- C1 − 现场 BASE（应以此判断消融，而非减 JSON）---")
    print(
        f"ΔE->P P={delta_c1_vs_live['ep_precision']:.4f} R={delta_c1_vs_live['ep_recall']:.4f} "
        f"F1={delta_c1_vs_live['ep_f1']:.4f} MAP={delta_c1_vs_live['ep_map']:.4f} "
        f"NDCG={delta_c1_vs_live['ep_ndcg']:.4f}"
    )
    print(f"\n已写入: {out_path}")


def run_quick_c1() -> None:
    """仅评估修正后的 C1，与 JSON 中 BASE 对比（用于快速验收，不写全表）。"""
    print("构建测试查询与评估掩码...")
    enterprise_queries, policy_queries = build_test_queries_from_data(
        max_enterprise_queries=300, max_industry_queries=30, max_policy_queries=200
    )
    df_pol = pd.read_parquet(PROJECT_ROOT / "data_intermediate" / "policies_clean.parquet")
    valid_policy_ids = set(df_pol["policy_id"].astype(int).tolist())
    base_metrics = _load_current_best_metrics()
    base_params = _base_eval_params()
    default_paths = {
        "policy_emb_path": "embeddings/policy_text_concat_emb.npy",
        "policy_index_path": "embeddings/policy_index.json",
        "company_emb_path": "embeddings/enterprise_text_emb.npy",
        "company_index_path": "embeddings/enterprise_index.json",
    }
    matcher = _build_matcher(**default_paths)
    snap = _snapshot_matcher_state(matcher)
    _restore_matcher_state(matcher, snap)
    _disable_gnn(matcher)
    c1_params = dict(base_params)
    c1_params.update(
        {
            "policy_semantic_weight": 0.9,
            "policy_structure_weight": 0.0,
            "policy_importance_weight": 0.1,
        }
    )
    m = _evaluate(matcher, enterprise_queries, policy_queries, valid_policy_ids, c1_params)
    print("JSON BASE (主实验冻结行) E->P: "
          f"P={base_metrics['ep_precision']:.4f} R={base_metrics['ep_recall']:.4f} "
          f"F1={base_metrics['ep_f1']:.4f} MAP={base_metrics['ep_map']:.4f} NDCG={base_metrics['ep_ndcg']:.4f}")
    print("C1 (w/o GNN, 权重 0.9/0/0.1) E->P: "
          f"P={m['ep_precision']:.4f} R={m['ep_recall']:.4f} "
          f"F1={m['ep_f1']:.4f} MAP={m['ep_map']:.4f} NDCG={m['ep_ndcg']:.4f}")
    print(
        "C1 P->E: "
        f"P={m['pe_precision']:.4f} R={m['pe_recall']:.4f} F1={m['pe_f1']:.4f} "
        f"MAP={m['pe_map']:.4f} NDCG={m['pe_ndcg']:.4f}"
    )
    print(f"delta NDCG (C1 - BASE) = {m['ep_ndcg'] - base_metrics['ep_ndcg']:.4f}")
    out_j = PROJECT_ROOT / "reports" / "ablation_c1_quick_metrics_pe_noboost.json"
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入: {out_j}")


def run_smoke() -> None:
    """仅评估 BASE（P→E 使用 direct_support_boost=0），用于快速验收副本可运行。"""
    print("run_ablation_11_pe_noboost.py --smoke：构建查询与 BASE 单次评估…")
    enterprise_queries, policy_queries = build_test_queries_from_data(
        max_enterprise_queries=300, max_industry_queries=30, max_policy_queries=200
    )
    df_pol = pd.read_parquet(PROJECT_ROOT / "data_intermediate" / "policies_clean.parquet")
    valid_policy_ids = set(df_pol["policy_id"].astype(int).tolist())
    base_params = _base_eval_params()
    assert base_params["direct_support_boost"] == 0.0
    default_paths = {
        "policy_emb_path": "embeddings/policy_text_concat_emb.npy",
        "policy_index_path": "embeddings/policy_index.json",
        "company_emb_path": "embeddings/enterprise_text_emb.npy",
        "company_index_path": "embeddings/enterprise_index.json",
    }
    matcher = _build_matcher(**default_paths)
    m = _evaluate(matcher, enterprise_queries, policy_queries, valid_policy_ids, dict(base_params))
    print(
        f"Smoke OK | P→E direct_support_boost=0 | "
        f"E→P NDCG={m['ep_ndcg']:.4f} | P→E NDCG={m['pe_ndcg']:.4f} Recall={m['pe_recall']:.4f}"
    )


def run():
    print("构建测试查询与评估掩码...")
    enterprise_queries, policy_queries = build_test_queries_from_data(
        max_enterprise_queries=300, max_industry_queries=30, max_policy_queries=200
    )
    df_pol = pd.read_parquet(PROJECT_ROOT / "data_intermediate" / "policies_clean.parquet")
    valid_policy_ids = set(df_pol["policy_id"].astype(int).tolist())

    frozen_json_ref = _load_current_best_metrics()
    base_params = _base_eval_params()

    rows: List[Dict[str, Any]] = []

    default_paths = {
        "policy_emb_path": "embeddings/policy_text_concat_emb.npy",
        "policy_index_path": "embeddings/policy_index.json",
        "company_emb_path": "embeddings/enterprise_text_emb.npy",
        "company_index_path": "embeddings/enterprise_index.json",
    }

    joint_paths = ensure_joint_bert_policy_embeddings() | {
        "company_emb_path": "embeddings/enterprise_text_emb.npy",
        "company_index_path": "embeddings/enterprise_index.json",
    }
    title_only_paths = {
        "policy_emb_path": "embeddings/policy_title_emb.npy",
        "policy_index_path": "embeddings/policy_index.json",
        "company_emb_path": "embeddings/enterprise_text_emb.npy",
        "company_index_path": "embeddings/enterprise_index.json",
    }

    print("初始化matcher缓存（base/joint/title）...")
    matchers = {
        "base": _build_matcher(**default_paths),
        "joint": _build_matcher(**joint_paths),
        "title": _build_matcher(**title_only_paths),
    }
    matcher_snaps = {k: _snapshot_matcher_state(v) for k, v in matchers.items()}

    print("本批现场评估 BASE（与下方消融共用同一 _evaluate，避免 JSON 冻结值与现场不一致）…")
    _restore_matcher_state(matchers["base"], matcher_snaps["base"])
    live_base_metrics = _evaluate(
        matchers["base"],
        enterprise_queries,
        policy_queries,
        valid_policy_ids,
        dict(base_params),
    )
    rows.append(
        {
            "id": "BASE",
            "name": "主实验最优(第四轮粗扫E→P)",
            "category": "Baseline",
            **live_base_metrics,
        }
    )
    print(
        f"  现场 BASE E->P NDCG={live_base_metrics['ep_ndcg']:.4f} F1={live_base_metrics['ep_f1']:.4f} | "
        f"evaluation_results JSON 参考 NDCG={frozen_json_ref['ep_ndcg']:.4f}（仅对比，主表以现场为准）"
    )

    experiments: List[Dict[str, Any]] = [
        {"id": "A1", "name": "独立编码后拼接(BERT title+content concat)", "category": "BERT嵌入策略", "matcher_key": "base", "params": {}},
        {"id": "A2", "name": "文本级拼接后编码(单BERT)", "category": "BERT嵌入策略", "matcher_key": "joint", "params": {}},
        {"id": "A3", "name": "仅单实体嵌入(仅政策标题向量)", "category": "BERT嵌入策略", "matcher_key": "title", "params": {}},
        {"id": "B1", "name": "w/o Hierarchy(移除层级衰减)", "category": "多模态融合", "matcher_key": "base", "params": {}, "importance_mode": "no_hierarchy"},
        {"id": "B2", "name": "w/o Time(移除时间衰减)", "category": "多模态融合", "matcher_key": "base", "params": {}, "importance_mode": "no_time"},
        {"id": "B3", "name": "Shallow Fusion(语义单流线性)", "category": "多模态融合", "matcher_key": "base", "params": {"policy_semantic_weight": 1.0, "policy_structure_weight": 0.0, "policy_importance_weight": 0.0, "policy_industry_boost": 0.0}},
        {
            "id": "C1",
            # 去掉图结构分项时，必须把原 structure 权重并入语义，保持 w_sem+w_str+w_imp=1，
            # 否则只剩 0.45+0.1=0.55 的融合标尺，自适应分位数截断与 BASE 不可比，NDCG 会异常偏高。
            "name": "w/o GNN(移除图结构聚合,融合权重重归一)",
            "category": "图结构表征",
            "matcher_key": "base",
            "params": {
                "policy_semantic_weight": 0.9,
                "policy_structure_weight": 0.0,
                "policy_importance_weight": 0.1,
            },
            "disable_gnn": True,
        },
        {"id": "C2", "name": "GCN替代GAT(无对比图嵌入)", "category": "图结构表征", "matcher_key": "base", "params": {}, "replace_graph_emb": True},
        {"id": "D1", "name": "w/o Decay(β=0)", "category": "衰减机制", "matcher_key": "base", "params": {}, "importance_mode": "no_decay"},
        {"id": "D2", "name": "Linear Decay替代Exponential", "category": "衰减机制", "matcher_key": "base", "params": {}, "importance_mode": "linear_decay"},
    ]

    print("开始执行 10 项单次消融...")
    for i, exp in enumerate(experiments, start=1):
        print(f"[{i}/10] {exp['id']} {exp['name']}")
        mkey = exp["matcher_key"]
        matcher = matchers[mkey]
        _restore_matcher_state(matcher, matcher_snaps[mkey])

        if exp.get("disable_gnn"):
            _disable_gnn(matcher)
        if exp.get("replace_graph_emb"):
            _replace_with_non_contrastive_graph_emb(matcher)
        if exp.get("importance_mode"):
            imp = _importance_map_from_mode(exp["importance_mode"])
            _apply_importance_map(matcher, imp)

        params = dict(base_params)
        params.update(exp.get("params", {}))
        metrics = _evaluate(matcher, enterprise_queries, policy_queries, valid_policy_ids, params)
        rows.append({"id": exp["id"], "name": exp["name"], "category": exp["category"], **metrics})
        print(
            f"  -> E->P F1={metrics['ep_f1']:.4f}, NDCG={metrics['ep_ndcg']:.4f}; "
            f"P->E F1={metrics['pe_f1']:.4f}, Recall={metrics['pe_recall']:.4f}"
        )

    # D3: beta敏感性（单独一组多beta）
    print("执行 D3 β 参数敏感性分析...")
    beta_rows: List[Dict[str, Any]] = []
    for beta in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]:
        matcher = matchers["base"]
        _restore_matcher_state(matcher, matcher_snaps["base"])
        imp = _importance_map_from_mode("beta_time_exp", beta_time=beta, beta_level=0.2)
        _apply_importance_map(matcher, imp)
        metrics = _evaluate(matcher, enterprise_queries, policy_queries, valid_policy_ids, dict(base_params))
        beta_rows.append({"beta_time": beta, **metrics})
        print(f"  beta={beta:.2f} -> E->P NDCG={metrics['ep_ndcg']:.4f}, P->E Recall={metrics['pe_recall']:.4f}")

    beta_df = pd.DataFrame(beta_rows).sort_values("ep_ndcg", ascending=False).reset_index(drop=True)
    best_beta = beta_df.iloc[0].to_dict()
    rows.append(
        {
            "id": "D3",
            "name": f"β敏感性最优(beta={best_beta['beta_time']:.2f})",
            "category": "衰减机制",
            "ep_precision": best_beta["ep_precision"],
            "ep_recall": best_beta["ep_recall"],
            "ep_f1": best_beta["ep_f1"],
            "ep_map": best_beta["ep_map"],
            "ep_ndcg": best_beta["ep_ndcg"],
            "pe_precision": best_beta["pe_precision"],
            "pe_recall": best_beta["pe_recall"],
            "pe_f1": best_beta["pe_f1"],
            "pe_map": best_beta["pe_map"],
            "pe_ndcg": best_beta["pe_ndcg"],
        }
    )

    out_reports = PROJECT_ROOT / "reports"
    out_reports.mkdir(parents=True, exist_ok=True)
    out_beta = out_reports / "ablation_beta_sensitivity_pe_noboost.csv"
    beta_df.to_csv(out_beta, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    base_row = df[df["id"] == "BASE"].iloc[0]
    for c in ["ep_precision", "ep_recall", "ep_f1", "ep_map", "ep_ndcg", "pe_precision", "pe_recall", "pe_f1", "pe_map", "pe_ndcg"]:
        df[f"delta_{c}"] = df[c] - float(base_row[c])

    # BASE 置顶，其余按 A1…D3 实验编号便于阅读（避免按字符串序 BASE 落在 B3/C1 之间）
    id_order = [
        "BASE",
        "A1",
        "A2",
        "A3",
        "B1",
        "B2",
        "B3",
        "C1",
        "C2",
        "D1",
        "D2",
        "D3",
    ]
    df["_sort"] = df["id"].map({x: i for i, x in enumerate(id_order)})
    df = df.sort_values(by=["_sort", "id"]).drop(columns=["_sort"]).reset_index(drop=True)
    out_csv = out_reports / "ablation_results_11_pe_noboost.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # markdown表
    show_cols = [
        "id", "name", "category",
        "ep_precision", "ep_recall", "ep_f1", "ep_map", "ep_ndcg",
        "pe_precision", "pe_recall", "pe_f1", "pe_map", "pe_ndcg",
        "delta_ep_ndcg", "delta_pe_recall",
    ]
    md = _df_to_markdown(df[show_cols], floatfmt=".4f")
    out_md = out_reports / "ablation_results_11_pe_noboost.md"
    out_md.write_text(
        "# 11项消融实验结果（P→E：`direct_support_boost=0`）\n\n"
        "协议：**P→E 关闭先验加分**（`direct_support_boost=0`），便于观测各模块对政策→企业排序的真实影响；"
        "E→P 与原版 `run_ablation_11.py` 一致。\n\n"
        f"基线：`{base_row['name']}`（本表 BASE 为同脚本现场评估，非历史 JSON）\n\n"
        + md
        + "\n\n"
        + f"- β敏感性明细：`{out_beta}`\n"
        + ABLATION_RESULTS_MD_APPENDIX,
        encoding="utf-8",
    )

    print("\n全部完成。")
    print(f"- 总表CSV: {out_csv}")
    print(f"- 总表Markdown: {out_md}")
    print(f"- β敏感性CSV: {out_beta}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="11 项消融（P→E direct_support_boost=0 副本），或 --smoke / --quick-c1 / --quick-align-base-c1"
    )
    ap.add_argument(
        "--quick-c1",
        action="store_true",
        help="仅重评修正后的 C1，与 evaluation_results JSON 中的 BASE 对比",
    )
    ap.add_argument(
        "--quick-align-base-c1",
        action="store_true",
        help="同批、固定种子重算现场 BASE 与 C1，写入 reports/ablation_base_c1_aligned.json",
    )
    ap.add_argument("--align-seed", type=int, default=42, help="--quick-align-base-c1 使用的随机种子")
    ap.add_argument(
        "--align-cudnn-deterministic",
        action="store_true",
        help="对齐评估时启用 cuDNN deterministic（会改变数值，一般不建议）",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="仅跑 BASE 一次（P→E boost=0）后退出，验收脚本可用性",
    )
    ns = ap.parse_args()
    if ns.smoke:
        run_smoke()
    elif ns.quick_align_base_c1:
        run_quick_align_base_c1(
            seed=ns.align_seed,
            cudnn_deterministic=ns.align_cudnn_deterministic,
        )
    elif ns.quick_c1:
        run_quick_c1()
    else:
        run()

