# -*- coding: utf-8 -*-
"""
GAT 对比嵌入与政策重要性 Parquet 的路径解析（与主实验 concat 流水线并存）。

- 默认：与历史主实验一致（graph/gat_*_contrastive.npy、evaluation/policy_importance_with_decay.parquet）。
- 环境变量 KGE_GAT_ARTIFACT_TAG：例如 a2_joint，则使用带后缀的 GAT 文件名（与 A2 全流程产物一致）。
- 若同时设置 KGE_POLICY_IMPORTANCE_PARQUET，则 **覆盖** 默认的 `policy_importance_with_decay_{tag}.parquet`（用于 B1 等单独生成的消融 parquet）。
- `evaluate_matching --experiment_profile a2_base|a3_title` 时默认 **忽略** 该环境变量，严格使用 `policy_importance_with_decay_{tag}.parquet`（BASE）；消融 B1/B2 等请用 `--policy_importance_parquet` 显式指定。
- 也可显式传入 `resolve_gat_importance_paths(..., importance_parquet=...)` 或 BidirectionalMatcher(..., policy_importance_parquet=...) 覆盖。
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

DEFAULT_GAT_POLICY = "graph/gat_policy_emb_contrastive.npy"
DEFAULT_GAT_COMPANY = "graph/gat_company_emb_contrastive.npy"
DEFAULT_IMPORTANCE_PARQUET = "evaluation/policy_importance_with_decay.parquet"


def resolve_gat_importance_paths(
    gat_artifact_tag: Optional[str] = None,
    *,
    importance_parquet: Optional[str] = None,
    ignore_env_importance_override: bool = False,
) -> Tuple[str, str, str]:
    """
    返回 (gat_policy_rel, gat_company_rel, importance_parquet_rel)，均为相对项目根的路径字符串。

    importance_parquet: 非空时直接使用（最高优先级）。
    ignore_env_importance_override: 为 True 且未传 importance_parquet 时，不用 KGE_POLICY_IMPORTANCE_PARQUET，
        仅用带 tag 的默认 parquet（主实验 BASE / a3_title 严格口径）。
    """
    tag = (gat_artifact_tag if gat_artifact_tag is not None else os.environ.get("KGE_GAT_ARTIFACT_TAG", "")).strip()
    if tag:
        explicit = (importance_parquet or "").strip()
        if explicit:
            imp_path = explicit
        elif ignore_env_importance_override:
            imp_path = f"evaluation/policy_importance_with_decay_{tag}.parquet"
        else:
            imp_override = os.environ.get("KGE_POLICY_IMPORTANCE_PARQUET", "").strip()
            imp_path = (
                imp_override
                if imp_override
                else f"evaluation/policy_importance_with_decay_{tag}.parquet"
            )
        return (
            f"graph/gat_policy_emb_contrastive_{tag}.npy",
            f"graph/gat_company_emb_contrastive_{tag}.npy",
            imp_path,
        )
    gp = os.environ.get("KGE_GAT_POLICY_EMB", DEFAULT_GAT_POLICY).strip() or DEFAULT_GAT_POLICY
    gc = os.environ.get("KGE_GAT_COMPANY_EMB", DEFAULT_GAT_COMPANY).strip() or DEFAULT_GAT_COMPANY
    explicit = (importance_parquet or "").strip()
    if explicit:
        ip = explicit
    else:
        ip = os.environ.get("KGE_POLICY_IMPORTANCE_PARQUET", DEFAULT_IMPORTANCE_PARQUET).strip() or DEFAULT_IMPORTANCE_PARQUET
    return gp, gc, ip
