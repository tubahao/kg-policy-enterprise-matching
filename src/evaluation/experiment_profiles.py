# -*- coding: utf-8 -*-
"""
主实验三档配置：legacy（原 concat+主图）、a2_base（当前 BASE：joint 全文线）、a3_title（仅标题向量+独立 GAT 线）。

用于 evaluate_matching 一键对齐超参与路径；消融脚本可复用超参字典。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROFILE_CHOICES = ("legacy", "a2_base", "a3_title")

_JSON_PATHS = {
    "legacy": PROJECT_ROOT / "matching" / "evaluation_results_a_ndcg_best_current.json",
    "a2_base": PROJECT_ROOT / "matching" / "evaluation_results_a2_joint_full_pipeline.json",
    # 由 scripts/run_a3_title_full_pipeline.py 末步写出，或手动：
    # python matching/evaluate_matching.py --experiment_profile a3_title --output matching/evaluation_results_a3_title_full_pipeline.json
    "a3_title": PROJECT_ROOT / "matching" / "evaluation_results_a3_title_full_pipeline.json",
}

# 超参与 a2 主实验对齐（公平对比文本策略）；a3_title 在结果 JSON 尚未生成时也使用该文件中的数值字段
_HP_JSON_KEY = {
    "legacy": "legacy",
    "a2_base": "a2_base",
    "a3_title": "a2_base",
}


def _load_parameters_from_json(rel_key: str) -> Dict[str, Any]:
    key = _HP_JSON_KEY.get(rel_key, rel_key)
    path = _JSON_PATHS[key]
    if not path.is_file():
        raise FileNotFoundError(f"缺少结果 JSON，无法加载 profile 超参 key={key}: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    return dict(obj.get("parameters") or {})


def get_profile_eval_parameter_dict(profile: str) -> Dict[str, Any]:
    """
    返回与 evaluate_matching 的 argparse 字段对齐的参数字典（不含 output / max_*_queries）。
    """
    if profile not in PROFILE_CHOICES:
        raise ValueError(f"profile 须为 {PROFILE_CHOICES}，收到 {profile!r}")
    p = _load_parameters_from_json(profile)
    out: Dict[str, Any] = {
        "top_k_policy": p.get("top_k_policy", -1),
        "top_k_enterprise": p.get("top_k_enterprise", -1),
        "policy_candidate_k": p.get("policy_candidate_k", 1000),
        "enterprise_candidate_k": p.get("enterprise_candidate_k", 1000),
        "policy_score_threshold": p.get("policy_score_threshold"),
        "policy_adaptive_quantile": p.get("policy_adaptive_quantile"),
        "policy_relative_drop_threshold": p.get("policy_relative_drop_threshold"),
        "policy_max_output_cap": p.get("policy_max_output_cap"),
        "policy_semantic_weight": p.get("policy_semantic_weight"),
        "policy_structure_weight": p.get("policy_structure_weight"),
        "policy_importance_weight": p.get("policy_importance_weight"),
        "policy_industry_boost": p.get("policy_industry_boost"),
        "policy_industry_query_adaptive_quantile": p.get("policy_industry_query_adaptive_quantile"),
        "policy_industry_query_relative_drop_threshold": p.get("policy_industry_query_relative_drop_threshold"),
        "policy_industry_query_max_output_cap": p.get("policy_industry_query_max_output_cap"),
        "enterprise_score_threshold": p.get("enterprise_score_threshold"),
        "enterprise_adaptive_quantile": p.get("enterprise_adaptive_quantile"),
        "enterprise_relative_drop_threshold": p.get("enterprise_relative_drop_threshold"),
        "enterprise_max_output_cap": p.get("enterprise_max_output_cap"),
        "direct_support_boost": p.get("direct_support_boost", 0.3),
    }
    # a2_base：与冻结主表 JSON 内 parameters 一致（0.45/0.45/0.1），便于 evaluate_matching 复现 ~0.342 NDCG。
    # 消融脚本若需其它权重，应在脚本内显式覆盖字典，勿在此静默改写。
    return out


def get_profile_matcher_bindings(profile: str) -> Tuple[str, str, Optional[str]]:
    """
    返回 (policy_text_mode, policy_emb_path_or_sentinel, gat_artifact_tag)。
    policy_text_mode in concat | joint | title；title 时用显式路径，由调用方解析。
    """
    if profile == "legacy":
        return "concat", "__default_concat__", None
    if profile == "a2_base":
        return "joint", "__default_joint__", "a2_joint"
    if profile == "a3_title":
        return "title", "embeddings/policy_title_emb.npy", "a3_title"
    raise ValueError(profile)


def default_output_path_for_profile(profile: str) -> str:
    if profile == "a3_title":
        return "matching/evaluation_results_a3_title_full_pipeline.json"
    return f"matching/evaluation_results_profile_{profile}.json"


def load_frozen_average_metrics(profile: str) -> Dict[str, float]:
    """从对应 profile 的 JSON 读取 enterprise/policy average 指标（作 BASE 冻结行）。"""
    path = _JSON_PATHS[profile]
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
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
