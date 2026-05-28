# -*- coding: utf-8 -*-
"""
主实验政策文本向量默认路径（可切换 A2 / 恢复原 concat）。

- 默认（或未设置环境变量）：与历史主实验一致 —— concat 两路 BERT 再拼接（1536 维），对应消融里的「BASE/A1」。
- 环境变量 KGE_POLICY_TEXT_MODE=joint：A2 —— 标题+正文拼成一句后单次 BERT（768 维），文件见 embeddings/policy_text_joint_bert_*.npy/json。
- 环境变量 KGE_POLICY_TEXT_MODE=title：仅标题 BERT，见 embeddings/policy_title_emb.npy + policy_index.json（常与 a3_title GAT 线联用）。

恢复原文本编码：取消该环境变量或设为 concat。

显式传入 BidirectionalMatcher(policy_emb_path=..., policy_index_path=...) 时不受环境变量影响。
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

CONCAT_EMB = "embeddings/policy_text_concat_emb.npy"
CONCAT_IDX = "embeddings/policy_index.json"
JOINT_EMB = "embeddings/policy_text_joint_bert_emb.npy"
JOINT_IDX = "embeddings/policy_text_joint_bert_index.json"
TITLE_EMB = "embeddings/policy_title_emb.npy"
TITLE_IDX = "embeddings/policy_index.json"


def resolve_policy_embedding_paths(
    policy_emb_path: Optional[str],
    policy_index_path: Optional[str],
) -> Tuple[str, str]:
    if policy_emb_path is not None and policy_index_path is not None:
        return policy_emb_path, policy_index_path
    if policy_emb_path is not None or policy_index_path is not None:
        raise ValueError("policy_emb_path 与 policy_index_path 必须同时传入或同时为 None")
    mode = os.environ.get("KGE_POLICY_TEXT_MODE", "concat").strip().lower()
    if mode == "joint":
        return JOINT_EMB, JOINT_IDX
    if mode == "title":
        return TITLE_EMB, TITLE_IDX
    return CONCAT_EMB, CONCAT_IDX
