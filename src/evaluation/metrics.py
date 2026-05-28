#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双向匹配效果评估
使用精确度、召回率、F1分数等指标评估匹配效果
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import time
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Set
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from matching.bidirectional_matching import BidirectionalMatcher

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def load_ground_truth_from_triples():
    """
    从三元组数据构建ground truth
    1. 企业→政策：从policy-entity三元组中提取 (企业, 政策) 关系
    2. 政策→企业：从policy-entity三元组中提取 (政策, 企业) 关系
    """
    print("从三元组数据构建ground truth...")
    
    # 加载policy-entity三元组
    df_p2e = pd.read_parquet(project_root / "data_intermediate/triples_policy_entity.parquet")
    
    # 构建企业→政策的ground truth
    # 如果predicate是supports，表示政策支持企业，那么企业应该能查询到该政策
    enterprise_to_policies = defaultdict(set)
    policy_to_enterprises = defaultdict(set)
    
    for _, row in df_p2e.iterrows():
        subject = str(row["subject"])
        obj = str(row["object"])
        predicate = str(row["predicate"]).lower()
        sub_type = row["subject_type"]
        obj_type = row["object_type"]
        
        if sub_type == "policy" and obj_type == "company" and predicate == "supports":
            enterprise_to_policies[obj].add(subject)
            policy_to_enterprises[subject].add(obj)
    
    print(f"  企业→政策关系数: {sum(len(v) for v in enterprise_to_policies.values())}")
    print(f"  政策→企业关系数: {sum(len(v) for v in policy_to_enterprises.values())}")
    
    return enterprise_to_policies, policy_to_enterprises


def build_test_queries_from_data(
    max_enterprise_queries: int = 200,
    max_industry_queries: int = 20,
    max_policy_queries: int = 200,
):
    """
    从实际数据构建测试查询
    1. 企业→政策：使用企业名称或行业作为查询
    2. 政策→企业：使用policy_id作为查询
    """
    print("构建测试查询...")
    
    # 加载企业数据
    df_enterprises = pd.read_parquet(project_root / "data_intermediate/enterprises_filtered.parquet")
    
    # 加载政策数据
    df_policies = pd.read_parquet(project_root / "data_intermediate/policies_clean.parquet")
    
    # 加载policy-entity三元组
    df_p2e = pd.read_parquet(project_root / "data_intermediate/triples_policy_entity.parquet")
    
    # 构建企业→政策的测试查询
    # 使用企业名称、行业等作为查询文本
    enterprise_queries = []
    enterprise_ground_truth = defaultdict(set)
    
    # 从三元组中提取企业-政策关系
    policy_titles = set(df_policies["title"].astype(str))
    company_names = set(df_enterprises["name"].astype(str))
    total_rows = len(df_p2e)
    progress_interval = max(1, total_rows // 10)
    print(f"  解析supports关系（企业→政策）: 共 {total_rows} 条三元组", flush=True)
    for idx, row in enumerate(df_p2e.itertuples(index=False), start=1):
        predicate = str(row.predicate).lower()
        subject = str(row.subject)
        obj = str(row.object)
        sub_type = str(row.subject_type).lower()
        obj_type = str(row.object_type).lower()
        
        # supports关系：policy -> company
        # 由于分类可能不准确，我们检查predicate和实际内容
        if predicate == "supports":
            if subject in policy_titles and obj in company_names:
                enterprise_ground_truth[obj].add(subject)
            elif sub_type == "policy" or (sub_type == "entity" and subject in policy_titles):
                # 如果subject是政策（即使被标记为entity）
                if obj in company_names or obj_type == "company":
                    enterprise_ground_truth[obj].add(subject)
        if idx % progress_interval == 0 or idx == total_rows:
            print(f"    进度: {idx}/{total_rows} ({100*idx/total_rows:.1f}%)", flush=True)
    
    # 选择有ground truth的企业作为测试查询（先按标签量排序，再截断）
    enterprise_items = sorted(
        enterprise_ground_truth.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )
    if max_enterprise_queries is not None and max_enterprise_queries > 0:
        enterprise_items = enterprise_items[:max_enterprise_queries]
    for company_name, policies in enterprise_items:
        enterprise_queries.append({
            "query": company_name,
            "type": "company_name",
            "ground_truth": list(policies)
        })
    
    # 添加一些行业查询
    industries = df_enterprises["industry"].dropna().value_counts().index.tolist()
    if max_industry_queries is not None and max_industry_queries > 0:
        industries = industries[:max_industry_queries]
    for industry in industries:
        # 找到该行业的企业
        industry_companies = df_enterprises[df_enterprises["industry"] == industry]["name"].tolist()
        # 找到这些企业关联的政策
        industry_policies = set()
        for company in industry_companies:
            industry_policies.update(enterprise_ground_truth.get(company, set()))
        
        if industry_policies:
            enterprise_queries.append({
                "query": industry,
                "type": "industry",
                "ground_truth": list(industry_policies)
            })
    
    # 构建政策→企业的测试查询
    policy_queries = []
    policy_ground_truth = defaultdict(set)
    
    # 从三元组中提取政策-企业关系
    print(f"  解析supports关系（政策→企业）: 共 {total_rows} 条三元组", flush=True)
    for idx, row in enumerate(df_p2e.itertuples(index=False), start=1):
        predicate = str(row.predicate).lower()
        subject = str(row.subject)
        obj = str(row.object)
        sub_type = str(row.subject_type).lower()
        obj_type = str(row.object_type).lower()
        
        # supports关系：policy -> company
        if predicate == "supports":
            # 验证subject是政策，object是企业
            if subject in policy_titles and obj in company_names:
                policy_ground_truth[subject].add(obj)
            elif (sub_type == "policy" or (sub_type == "entity" and subject in policy_titles)):
                # 如果subject是政策（即使被标记为entity）
                if obj in company_names or obj_type == "company":
                    policy_ground_truth[subject].add(obj)
        if idx % progress_interval == 0 or idx == total_rows:
            print(f"    进度: {idx}/{total_rows} ({100*idx/total_rows:.1f}%)", flush=True)
    
    # 选择有ground truth的政策作为测试查询
    # 需要将政策标题映射到policy_id
    policy_title_to_id = {}
    for _, row in df_policies.iterrows():
        policy_title_to_id[str(row["title"])] = int(row["policy_id"])
    
    # 关键修复：先过滤到可映射 policy_id，再按标签量排序后截断
    mapped_policy_items = [
        (policy_title, companies)
        for policy_title, companies in policy_ground_truth.items()
        if policy_title in policy_title_to_id
    ]
    mapped_policy_items.sort(key=lambda x: len(x[1]), reverse=True)
    if max_policy_queries is not None and max_policy_queries > 0:
        mapped_policy_items = mapped_policy_items[:max_policy_queries]

    for policy_title, companies in mapped_policy_items:
        if policy_title in policy_title_to_id:
            policy_queries.append({
                "policy_id": policy_title_to_id[policy_title],
                "policy_title": policy_title,
                "ground_truth": list(companies)
            })
    
    print(f"  企业→政策测试查询数: {len(enterprise_queries)}")
    print(f"  政策→企业测试查询数: {len(policy_queries)}")
    
    return enterprise_queries, policy_queries


def calculate_metrics(predicted: List, ground_truth: List) -> Dict[str, float]:
    """
    计算精确度、召回率、F1分数
    
    公式：
    - Precision = TP / (TP + FP) = |predicted ∩ ground_truth| / |predicted|
    - Recall = TP / (TP + FN) = |predicted ∩ ground_truth| / |ground_truth|
    - F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        predicted: 预测结果列表
        ground_truth: 真实结果列表
    
    Returns:
        包含precision, recall, f1的字典
    """
    predicted_set = set(predicted)
    ground_truth_set = set(ground_truth)
    
    # 计算交集
    intersection = predicted_set & ground_truth_set
    tp = len(intersection)  # True Positives
    
    # 计算精确度
    if len(predicted_set) == 0:
        precision = 0.0
    else:
        precision = tp / len(predicted_set)
    
    # 计算召回率
    if len(ground_truth_set) == 0:
        recall = 0.0
    else:
        recall = tp / len(ground_truth_set)
    
    # 计算F1分数
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": len(predicted_set) - tp,
        "fn": len(ground_truth_set) - tp
    }


def calculate_ranking_metrics(predicted: List, ground_truth: List) -> Dict[str, float]:
    """计算排名指标：AP / NDCG。"""
    gt = set(ground_truth)
    if not predicted or not gt:
        return {"ap": 0.0, "ndcg": 0.0}

    # Average Precision
    hit = 0
    ap_sum = 0.0
    for i, item in enumerate(predicted, start=1):
        if item in gt:
            hit += 1
            ap_sum += hit / i
    ap = ap_sum / max(len(gt), 1)

    # NDCG (binary relevance)
    dcg = 0.0
    for i, item in enumerate(predicted, start=1):
        rel = 1.0 if item in gt else 0.0
        dcg += rel / np.log2(i + 1)
    ideal_hits = min(len(gt), len(predicted))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"ap": float(ap), "ndcg": float(ndcg)}


def evaluate_enterprise_to_policy(
    matcher: BidirectionalMatcher,
    queries: List[Dict],
    top_k: int = 10,
    candidate_k: int = 1000,
    score_threshold: Optional[float] = None,
    adaptive_quantile: Optional[float] = 0.75,
    relative_drop_threshold: Optional[float] = 0.15,
    max_output_cap: Optional[int] = 100,
    semantic_weight: float = 0.7,
    structure_weight: float = 0.2,
    importance_weight: float = 0.1,
    industry_boost: float = 0.05,
    industry_query_adaptive_quantile: Optional[float] = None,
    industry_query_relative_drop_threshold: Optional[float] = None,
    industry_query_max_output_cap: Optional[int] = None,
    valid_policy_ids: Optional[Set[int]] = None,
):
    """
    评估企业→政策查询效果
    
    Args:
        matcher: 双向匹配器
        queries: 测试查询列表
        top_k: 返回前k个结果
    """
    print(f"\n{'='*60}")
    print("评估企业→政策查询效果")
    print(f"{'='*60}")
    print(f"测试查询数: {len(queries)}")
    print(f"Top-K: {top_k}（<=0 表示不截断）")
    print(
        f"Candidate-K: {candidate_k} | Score threshold: {score_threshold} | "
        f"adaptive_quantile: {adaptive_quantile} | relative_drop: {relative_drop_threshold} | "
        f"max_output_cap: {max_output_cap}\n"
    )
    if any(v is not None for v in [industry_query_adaptive_quantile, industry_query_relative_drop_threshold, industry_query_max_output_cap]):
        print(
            "Industry查询自适应覆盖: "
            f"adaptive_quantile={industry_query_adaptive_quantile}, "
            f"relative_drop={industry_query_relative_drop_threshold}, "
            f"max_output_cap={industry_query_max_output_cap}\n"
        )
    
    all_metrics = []
    total = len(queries)
    start = time.time()
    progress_interval = max(1, total // 10)
    
    # 显式评估掩码：仅保留在 policies_clean.parquet 中存在的 policy_id
    df_policies = pd.read_parquet(project_root / "data_intermediate/policies_clean.parquet")
    policy_id_to_title = {int(row["policy_id"]): str(row["title"]) for _, row in df_policies.iterrows()}
    if valid_policy_ids is None:
        valid_policy_ids = set(policy_id_to_title.keys())

    valid_policy_titles = set(policy_id_to_title.values())
    masked_gt_empty = 0

    for i, query_info in enumerate(queries, 1):
        query_text = query_info["query"]
        ground_truth_raw = query_info["ground_truth"]
        ground_truth = [t for t in ground_truth_raw if t in valid_policy_titles]
        if len(ground_truth) == 0:
            masked_gt_empty += 1
        
        # 执行查询（可按查询类型做参数自适应）
        q_type = query_info.get("type", "")
        local_adaptive_q = adaptive_quantile
        local_drop = relative_drop_threshold
        local_cap = max_output_cap
        if q_type == "industry":
            if industry_query_adaptive_quantile is not None:
                local_adaptive_q = industry_query_adaptive_quantile
            if industry_query_relative_drop_threshold is not None:
                local_drop = industry_query_relative_drop_threshold
            if industry_query_max_output_cap is not None:
                local_cap = industry_query_max_output_cap

        results = matcher.query_policies_by_enterprise(
            query_text,
            top_k=top_k,
            candidate_k=candidate_k,
            score_threshold=score_threshold,
            adaptive_quantile=local_adaptive_q,
            relative_drop_threshold=local_drop,
            max_output_cap=local_cap,
            semantic_weight=semantic_weight,
            structure_weight=structure_weight,
            importance_weight=importance_weight,
            industry_boost=industry_boost,
        )
        predicted_policy_ids = [pid for pid, _ in results if pid in valid_policy_ids]
        predicted_titles = [policy_id_to_title.get(pid, "") for pid in predicted_policy_ids if pid in policy_id_to_title]
        
        # 计算指标
        metrics = calculate_metrics(predicted_titles, ground_truth)
        ranking_metrics = calculate_ranking_metrics(predicted_titles, ground_truth)
        metrics.update(ranking_metrics)
        metrics["query"] = query_text
        metrics["query_type"] = query_info["type"]
        all_metrics.append(metrics)
        
        if i <= 5:  # 显示前5个查询的详细信息
            print(f"查询 {i}: {query_text}")
            print(f"  预测数: {len(predicted_titles)}, 真实数: {len(ground_truth)}")
            print(
                f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}, AP: {metrics['ap']:.4f}, NDCG: {metrics['ndcg']:.4f}"
            )
            print()

        if i % progress_interval == 0 or i == total:
            elapsed = time.time() - start
            eta = (elapsed / i) * (total - i) if i > 0 else 0
            print(
                f"进度: {i}/{total} ({100*i/total:.1f}%) | "
                f"elapsed={_format_seconds(elapsed)} | eta={_format_seconds(eta)}",
                flush=True,
            )
    
    # 计算平均指标
    avg_precision = np.mean([m["precision"] for m in all_metrics])
    avg_recall = np.mean([m["recall"] for m in all_metrics])
    avg_f1 = np.mean([m["f1"] for m in all_metrics])
    avg_ap = np.mean([m["ap"] for m in all_metrics])
    avg_ndcg = np.mean([m["ndcg"] for m in all_metrics])
    
    print(f"\n平均指标:")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall: {avg_recall:.4f}")
    print(f"  F1: {avg_f1:.4f}")
    print(f"  MAP(AP均值): {avg_ap:.4f}")
    print(f"  NDCG: {avg_ndcg:.4f}")
    print(f"  评估掩码后GT为空查询数: {masked_gt_empty}")
    
    return {
        "metrics": all_metrics,
        "average": {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "map": avg_ap,
            "ndcg": avg_ndcg,
            "masked_gt_empty": masked_gt_empty,
        }
    }


def evaluate_policy_to_enterprise(
    matcher: BidirectionalMatcher,
    queries: List[Dict],
    top_k: int = 20,
    candidate_k: int = 1000,
    score_threshold: Optional[float] = None,
    adaptive_quantile: Optional[float] = None,
    relative_drop_threshold: Optional[float] = 0.15,
    max_output_cap: Optional[int] = 100,
    direct_support_boost: float = 0.2,
):
    """
    评估政策→企业检索效果
    
    Args:
        matcher: 双向匹配器
        queries: 测试查询列表
        top_k: 返回前k个结果
    """
    print(f"\n{'='*60}")
    print("评估政策→企业检索效果")
    print(f"{'='*60}")
    print(f"测试查询数: {len(queries)}")
    print(f"Top-K: {top_k}（<=0 表示不截断）")
    print(
        f"Candidate-K: {candidate_k} | Score threshold: {score_threshold} | "
        f"adaptive_quantile: {adaptive_quantile} | relative_drop: {relative_drop_threshold} | "
        f"max_output_cap: {max_output_cap} | direct_support_boost: {direct_support_boost}\n"
    )
    
    all_metrics = []
    total = len(queries)
    start = time.time()
    progress_interval = max(1, total // 10)

    # 优先使用图节点ID -> 企业名称映射（检索输出的cid是company节点ID）
    graph_nodeid_to_name = {}
    if hasattr(matcher, "enterprise_retriever"):
        node_maps = getattr(matcher.enterprise_retriever, "node_maps", {}) or {}
        company_map = node_maps.get("company", {})
        # company_map: {company_name: node_id}
        graph_nodeid_to_name = {int(v): str(k) for k, v in company_map.items()}
    
    valid_company_names = set(graph_nodeid_to_name.values())
    masked_gt_empty = 0

    for i, query_info in enumerate(queries, 1):
        policy_id = query_info["policy_id"]
        policy_title = query_info["policy_title"]
        ground_truth_raw = query_info["ground_truth"]
        ground_truth = [n for n in ground_truth_raw if n in valid_company_names]
        if len(ground_truth) == 0:
            masked_gt_empty += 1
        
        # 执行检索
        results = matcher.retrieve_enterprises_by_policy(
            policy_id,
            top_k=top_k,
            score_threshold=score_threshold,
            candidate_k=candidate_k,
            adaptive_quantile=adaptive_quantile,
            relative_drop_threshold=relative_drop_threshold,
            max_output_cap=max_output_cap,
            direct_support_boost=direct_support_boost,
        )
        predicted_company_ids = [cid for cid, _ in results]
        
        # 需要将company_id转换为企业名称进行比较
        # 加载企业数据（避免重复加载），作为enterprise_id回退映射
        if not hasattr(evaluate_policy_to_enterprise, '_company_id_to_name'):
            df_enterprises = pd.read_parquet(project_root / "data_intermediate/enterprises_filtered.parquet")
            company_id_to_name = {}
            # 构建多种格式的映射
            for _, row in df_enterprises.iterrows():
                company_id = row.get("enterprise_id")
                company_name = str(row["name"])
                
                if pd.notna(company_id):
                    # 尝试转换为整数
                    try:
                        company_id_to_name[int(company_id)] = company_name
                    except (ValueError, TypeError):
                        # 如果是字符串格式，也保存
                        company_id_to_name[str(company_id)] = company_name
                    # 也保存enterprise_X格式
                    if isinstance(company_id, (int, float)):
                        company_id_to_name[f"enterprise_{int(company_id)}"] = company_name
            
            evaluate_policy_to_enterprise._company_id_to_name = company_id_to_name
        
        company_id_to_name = evaluate_policy_to_enterprise._company_id_to_name
        
        # 转换预测的企业ID为企业名称
        predicted_names = []
        for cid in predicted_company_ids:
            # 尝试多种格式查找
            name = None
            # 1) 首先按图节点ID映射（主路径）
            if isinstance(cid, int) and cid in graph_nodeid_to_name:
                name = graph_nodeid_to_name[cid]
            # 2) 回退到enterprise_id映射（兼容旧逻辑）
            elif cid in company_id_to_name:
                name = company_id_to_name[cid]
            elif isinstance(cid, int) and f"enterprise_{cid}" in company_id_to_name:
                name = company_id_to_name[f"enterprise_{cid}"]
            elif str(cid) in company_id_to_name:
                name = company_id_to_name[str(cid)]
            
            if name:
                predicted_names.append(name)
            else:
                # 如果找不到，使用ID作为名称（用于比较）
                predicted_names.append(f"enterprise_{cid}" if isinstance(cid, int) else str(cid))
        
        # 计算指标
        metrics = calculate_metrics(predicted_names, ground_truth)
        ranking_metrics = calculate_ranking_metrics(predicted_names, ground_truth)
        metrics.update(ranking_metrics)
        metrics["policy_id"] = policy_id
        metrics["policy_title"] = policy_title
        all_metrics.append(metrics)
        
        if i <= 5:  # 显示前5个查询的详细信息
            print(f"查询 {i}: Policy ID {policy_id} - {policy_title[:50]}...")
            print(f"  预测数: {len(predicted_names)}, 真实数: {len(ground_truth)}")
            print(
                f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}, AP: {metrics['ap']:.4f}, NDCG: {metrics['ndcg']:.4f}"
            )
            print()

        if i % progress_interval == 0 or i == total:
            elapsed = time.time() - start
            eta = (elapsed / i) * (total - i) if i > 0 else 0
            print(
                f"进度: {i}/{total} ({100*i/total:.1f}%) | "
                f"elapsed={_format_seconds(elapsed)} | eta={_format_seconds(eta)}",
                flush=True,
            )
    
    # 计算平均指标
    avg_precision = np.mean([m["precision"] for m in all_metrics])
    avg_recall = np.mean([m["recall"] for m in all_metrics])
    avg_f1 = np.mean([m["f1"] for m in all_metrics])
    avg_ap = np.mean([m["ap"] for m in all_metrics])
    avg_ndcg = np.mean([m["ndcg"] for m in all_metrics])
    
    print(f"\n平均指标:")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall: {avg_recall:.4f}")
    print(f"  F1: {avg_f1:.4f}")
    print(f"  MAP(AP均值): {avg_ap:.4f}")
    print(f"  NDCG: {avg_ndcg:.4f}")
    print(f"  评估掩码后GT为空查询数: {masked_gt_empty}")
    
    return {
        "metrics": all_metrics,
        "average": {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "map": avg_ap,
            "ndcg": avg_ndcg,
            "masked_gt_empty": masked_gt_empty,
        }
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="双向匹配效果评估")
    parser.add_argument("--top_k_policy", type=int, default=10, help="企业→政策查询返回前k个结果")
    parser.add_argument("--top_k_enterprise", type=int, default=20, help="政策→企业检索返回前k个结果")
    parser.add_argument("--policy_candidate_k", type=int, default=1000, help="企业→政策候选数量（阈值截断前）")
    parser.add_argument("--enterprise_candidate_k", type=int, default=1000, help="政策→企业候选数量（阈值截断前）")
    parser.add_argument("--policy_score_threshold", type=float, default=None, help="企业→政策分数阈值")
    parser.add_argument("--policy_adaptive_quantile", type=float, default=0.75, help="企业→政策自适应分位阈值(0,1)，显式阈值为空时生效")
    parser.add_argument("--policy_relative_drop_threshold", type=float, default=0.15, help="企业→政策动态断崖截断阈值（相邻分数相对跌幅）")
    parser.add_argument("--policy_max_output_cap", type=int, default=100, help="企业→政策最大输出数量上限")
    parser.add_argument("--policy_semantic_weight", type=float, default=0.7, help="企业→政策语义相似度权重")
    parser.add_argument("--policy_structure_weight", type=float, default=0.2, help="企业→政策结构分数权重")
    parser.add_argument("--policy_importance_weight", type=float, default=0.1, help="企业→政策重要性权重")
    parser.add_argument("--policy_industry_boost", type=float, default=0.05, help="企业→政策行业重排加分")
    parser.add_argument("--policy_industry_query_adaptive_quantile", type=float, default=None, help="industry查询专用自适应分位阈值")
    parser.add_argument("--policy_industry_query_relative_drop_threshold", type=float, default=None, help="industry查询专用断崖阈值")
    parser.add_argument("--policy_industry_query_max_output_cap", type=int, default=None, help="industry查询专用输出上限")
    parser.add_argument("--enterprise_score_threshold", type=float, default=0.55, help="政策→企业分数阈值")
    parser.add_argument("--enterprise_adaptive_quantile", type=float, default=None, help="政策→企业自适应分位阈值(0,1)，当显式阈值为空时生效")
    parser.add_argument("--enterprise_relative_drop_threshold", type=float, default=0.15, help="政策→企业动态断崖截断阈值（相邻分数相对跌幅）")
    parser.add_argument("--enterprise_max_output_cap", type=int, default=100, help="政策→企业最大输出数量上限")
    parser.add_argument("--direct_support_boost", type=float, default=0.2, help="政策→企业直接supports先验加分")
    parser.add_argument("--output", type=str, default="matching/evaluation_results.json", help="评估结果输出文件")
    parser.add_argument(
        "--experiment_profile",
        type=str,
        choices=("legacy", "a2_base", "a3_title"),
        default=None,
        help="一键主实验三档：legacy=原concat+默认GAT；a2_base=当前BASE(joint+GAT/衰减 a2_joint)；a3_title=仅标题向量+GAT/衰减 a3_title。会覆盖超参默认值，可用 --output 仍覆盖输出路径",
    )
    parser.add_argument(
        "--policy_text_mode",
        type=str,
        choices=("concat", "joint", "title"),
        default=None,
        help="政策文本向量：concat|joint|title(仅标题BERT)。默认读环境变量 KGE_POLICY_TEXT_MODE 或未设置时用 concat",
    )
    parser.add_argument(
        "--gat_artifact_tag",
        type=str,
        default=None,
        help="与独立 GAT/衰减产物一致的后缀，例如 a2_joint（加载 gat_*_contrastive_{tag}.npy 与 policy_importance_with_decay_{tag}.parquet）。未传时读环境变量 KGE_GAT_ARTIFACT_TAG",
    )
    parser.add_argument(
        "--policy_importance_parquet",
        type=str,
        default=None,
        help="相对项目根的政策重要性 parquet；非空时最高优先级。a2_base/a3_title 未指定时默认忽略 KGE_POLICY_IMPORTANCE_PARQUET，仅用 policy_importance_with_decay_{tag}.parquet",
    )
    parser.add_argument("--max_enterprise_queries", type=int, default=200, help="企业→政策测试查询上限（<=0表示不过滤）")
    parser.add_argument("--max_industry_queries", type=int, default=20, help="行业查询上限（<=0表示不过滤）")
    parser.add_argument("--max_policy_queries", type=int, default=200, help="政策→企业测试查询上限（<=0表示不过滤）")
    args = parser.parse_args()
    _default_output = "matching/evaluation_results.json"
    if args.experiment_profile:
        from matching.experiment_profiles import (
            default_output_path_for_profile,
            get_profile_eval_parameter_dict,
            get_profile_matcher_bindings,
        )

        hp = get_profile_eval_parameter_dict(args.experiment_profile)
        for k, v in hp.items():
            if hasattr(args, k):
                setattr(args, k, v)
        mode, _, gat_tag = get_profile_matcher_bindings(args.experiment_profile)
        args.policy_text_mode = mode
        args.gat_artifact_tag = gat_tag
        if args.output == _default_output:
            args.output = default_output_path_for_profile(args.experiment_profile)

    if args.policy_score_threshold is not None and args.policy_score_threshold < 0:
        args.policy_score_threshold = None
    if args.enterprise_score_threshold is not None and args.enterprise_score_threshold < 0:
        args.enterprise_score_threshold = None
    if args.policy_adaptive_quantile is not None and not (0 < args.policy_adaptive_quantile < 1):
        args.policy_adaptive_quantile = None
    if args.enterprise_adaptive_quantile is not None and not (0 < args.enterprise_adaptive_quantile < 1):
        args.enterprise_adaptive_quantile = None
    if args.policy_industry_query_adaptive_quantile is not None and not (0 < args.policy_industry_query_adaptive_quantile < 1):
        args.policy_industry_query_adaptive_quantile = None

    import os

    policy_text_mode = args.policy_text_mode
    if policy_text_mode is None:
        policy_text_mode = os.environ.get("KGE_POLICY_TEXT_MODE", "concat").strip().lower()
    if policy_text_mode not in ("concat", "joint", "title"):
        policy_text_mode = "concat"
    
    print("="*60)
    print("双向匹配效果评估")
    print("="*60)
    print(f"\n评估参数:")
    print(f"  企业→政策 Top-K: {args.top_k_policy}")
    print(f"  政策→企业 Top-K: {args.top_k_enterprise}")
    print(f"  企业→政策 candidate_k: {args.policy_candidate_k}, threshold: {args.policy_score_threshold}")
    print(
        f"  企业→政策 adaptive_quantile: {args.policy_adaptive_quantile}, "
        f"relative_drop: {args.policy_relative_drop_threshold}, max_output_cap: {args.policy_max_output_cap}"
    )
    print(
        f"  企业→政策权重 semantic/structure/importance: "
        f"{args.policy_semantic_weight}/{args.policy_structure_weight}/{args.policy_importance_weight}, "
        f"industry_boost: {args.policy_industry_boost}"
    )
    print(
        f"  企业→政策 industry查询覆盖参数: q={args.policy_industry_query_adaptive_quantile}, "
        f"drop={args.policy_industry_query_relative_drop_threshold}, cap={args.policy_industry_query_max_output_cap}"
    )
    print(f"  政策→企业 candidate_k: {args.enterprise_candidate_k}, threshold: {args.enterprise_score_threshold}")
    print(
        f"  政策→企业 adaptive_quantile: {args.enterprise_adaptive_quantile}, "
        f"relative_drop: {args.enterprise_relative_drop_threshold}, max_output_cap: {args.enterprise_max_output_cap}, "
        f"direct_support_boost: {args.direct_support_boost}"
    )
    if args.experiment_profile:
        print(f"  实验档位 experiment_profile: {args.experiment_profile}")
    print(
        f"  政策文本编码 policy_text_mode: {policy_text_mode} "
        f"(环境变量 KGE_POLICY_TEXT_MODE=concat|joint|title)"
    )
    from matching.gat_importance_defaults import resolve_gat_importance_paths

    _strict_base_imp = args.experiment_profile in ("a2_base", "a3_title")
    _imp_arg = (args.policy_importance_parquet or "").strip() or None
    _gat_p, _gat_c, _imp_pq = resolve_gat_importance_paths(
        args.gat_artifact_tag,
        importance_parquet=_imp_arg,
        ignore_env_importance_override=_strict_base_imp,
    )
    print(f"  GAT/重要性产物: policy={_gat_p}, company={_gat_c}, parquet={_imp_pq}")
    if _strict_base_imp and not _imp_arg:
        print("  （a2_base/a3_title：已忽略环境变量 KGE_POLICY_IMPORTANCE_PARQUET，使用上表 BASE 衰减 parquet）")
    print()

    from matching.bidirectional_matching import BidirectionalMatcher
    from matching.policy_embedding_defaults import CONCAT_EMB, CONCAT_IDX, JOINT_EMB, JOINT_IDX, TITLE_EMB, TITLE_IDX

    if policy_text_mode == "joint":
        from matching.ensure_joint_policy_embeddings import ensure_joint_policy_embeddings

        ensure_joint_policy_embeddings(project_root)
        _pe, _pi = JOINT_EMB, JOINT_IDX
    elif policy_text_mode == "title":
        _pe, _pi = TITLE_EMB, TITLE_IDX
    else:
        _pe, _pi = CONCAT_EMB, CONCAT_IDX

    t0 = time.time()
    print("初始化双向匹配器...", flush=True)
    matcher = BidirectionalMatcher(
        project_root,
        policy_emb_path=_pe,
        policy_index_path=_pi,
        gat_artifact_tag=args.gat_artifact_tag,
        policy_importance_parquet=_imp_arg,
        ignore_env_importance_override=_strict_base_imp,
    )
    print(f"初始化完成！耗时 {_format_seconds(time.time()-t0)}\n", flush=True)
    
    # 构建测试查询
    t1 = time.time()
    enterprise_queries, policy_queries = build_test_queries_from_data(
        max_enterprise_queries=args.max_enterprise_queries if args.max_enterprise_queries > 0 else None,
        max_industry_queries=args.max_industry_queries if args.max_industry_queries > 0 else None,
        max_policy_queries=args.max_policy_queries if args.max_policy_queries > 0 else None,
    )
    print(f"构建测试查询完成，耗时 {_format_seconds(time.time()-t1)}\n", flush=True)

    # 显式评估掩码：仅保留在 policies_clean 中有标签的 policy_id（避免图扩展节点污染评估）
    df_policies = pd.read_parquet(project_root / "data_intermediate/policies_clean.parquet")
    valid_policy_ids = set(df_policies["policy_id"].astype(int).tolist())
    print(f"评估掩码: 仅对 {len(valid_policy_ids)} 个 policies_clean 政策评估\n", flush=True)
    
    # 评估企业→政策查询
    t2 = time.time()
    enterprise_results = evaluate_enterprise_to_policy(
        matcher,
        enterprise_queries,
        top_k=args.top_k_policy,
        candidate_k=args.policy_candidate_k,
        score_threshold=args.policy_score_threshold,
        adaptive_quantile=args.policy_adaptive_quantile,
        relative_drop_threshold=args.policy_relative_drop_threshold,
        max_output_cap=args.policy_max_output_cap,
        semantic_weight=args.policy_semantic_weight,
        structure_weight=args.policy_structure_weight,
        importance_weight=args.policy_importance_weight,
        industry_boost=args.policy_industry_boost,
        industry_query_adaptive_quantile=args.policy_industry_query_adaptive_quantile,
        industry_query_relative_drop_threshold=args.policy_industry_query_relative_drop_threshold,
        industry_query_max_output_cap=args.policy_industry_query_max_output_cap,
        valid_policy_ids=valid_policy_ids,
    )
    print(f"企业→政策评估耗时 {_format_seconds(time.time()-t2)}\n", flush=True)
    
    # 评估政策→企业检索
    t3 = time.time()
    policy_results = evaluate_policy_to_enterprise(
        matcher,
        policy_queries,
        top_k=args.top_k_enterprise,
        candidate_k=args.enterprise_candidate_k,
        score_threshold=args.enterprise_score_threshold,
        adaptive_quantile=args.enterprise_adaptive_quantile,
        relative_drop_threshold=args.enterprise_relative_drop_threshold,
        max_output_cap=args.enterprise_max_output_cap,
        direct_support_boost=args.direct_support_boost,
    )
    print(f"政策→企业评估耗时 {_format_seconds(time.time()-t3)}\n", flush=True)
    
    # 保存结果
    results = {
        "model": {
            "name": "BidirectionalMatcher",
            "query_encoder": "BERT-base-chinese",
            "similarity_method": "Cosine Similarity",
            "retrieval_method": "GraphRAG + GNN + Threshold"
        },
        "parameters": {
            "top_k_policy": args.top_k_policy,
            "top_k_enterprise": args.top_k_enterprise,
            "policy_candidate_k": args.policy_candidate_k,
            "enterprise_candidate_k": args.enterprise_candidate_k,
            "policy_score_threshold": args.policy_score_threshold,
            "policy_adaptive_quantile": args.policy_adaptive_quantile,
            "policy_relative_drop_threshold": args.policy_relative_drop_threshold,
            "policy_max_output_cap": args.policy_max_output_cap,
            "policy_semantic_weight": args.policy_semantic_weight,
            "policy_structure_weight": args.policy_structure_weight,
            "policy_importance_weight": args.policy_importance_weight,
            "policy_industry_boost": args.policy_industry_boost,
            "policy_industry_query_adaptive_quantile": args.policy_industry_query_adaptive_quantile,
            "policy_industry_query_relative_drop_threshold": args.policy_industry_query_relative_drop_threshold,
            "policy_industry_query_max_output_cap": args.policy_industry_query_max_output_cap,
            "enterprise_score_threshold": args.enterprise_score_threshold,
            "enterprise_adaptive_quantile": args.enterprise_adaptive_quantile,
            "enterprise_relative_drop_threshold": args.enterprise_relative_drop_threshold,
            "enterprise_max_output_cap": args.enterprise_max_output_cap,
            "direct_support_boost": args.direct_support_boost,
            "k_hop": 2,
            "experiment_profile": args.experiment_profile,
            "policy_text_mode": policy_text_mode,
            "policy_emb_path": _pe,
            "policy_index_path": _pi,
            "gat_artifact_tag": args.gat_artifact_tag,
            "gat_policy_emb_path": _gat_p,
            "gat_company_emb_path": _gat_c,
            "policy_importance_parquet_path": _imp_pq,
            "policy_importance_parquet_cli": args.policy_importance_parquet,
            "policy_importance_ignore_env_for_profile": _strict_base_imp,
        },
        "formulas": {
            "precision": "TP / (TP + FP) = |predicted ∩ ground_truth| / |predicted|",
            "recall": "TP / (TP + FN) = |predicted ∩ ground_truth| / |ground_truth|",
            "f1": "2 * (Precision * Recall) / (Precision + Recall)"
        },
        "enterprise_to_policy": enterprise_results,
        "policy_to_enterprise": policy_results
    }
    
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("评估完成！")
    print(f"{'='*60}")
    print(f"\n结果已保存到: {output_path}")
    print(f"\n总体评估结果:")
    print(f"  企业→政策:")
    print(f"    Precision: {enterprise_results['average']['precision']:.4f}")
    print(f"    Recall: {enterprise_results['average']['recall']:.4f}")
    print(f"    F1: {enterprise_results['average']['f1']:.4f}")
    print(f"    MAP: {enterprise_results['average']['map']:.4f}")
    print(f"    NDCG: {enterprise_results['average']['ndcg']:.4f}")
    print(f"  政策→企业:")
    print(f"    Precision: {policy_results['average']['precision']:.4f}")
    print(f"    Recall: {policy_results['average']['recall']:.4f}")
    print(f"    F1: {policy_results['average']['f1']:.4f}")
    print(f"    MAP: {policy_results['average']['map']:.4f}")
    print(f"    NDCG: {policy_results['average']['ndcg']:.4f}")


if __name__ == "__main__":
    main()

