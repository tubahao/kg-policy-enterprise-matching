#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双向匹配API使用示例
展示如何在Python代码中使用双向匹配功能
"""

import sys
import os
from pathlib import Path

# 检查是否在虚拟环境中（仅提示，不强制退出）
if sys.platform == 'win32':
    venv_python = Path(__file__).resolve().parents[1] / "venv_graph" / "Scripts" / "python.exe"
    if venv_python.exists() and "venv_graph" not in sys.executable:
        print("="*60)
        print("⚠️  提示：建议使用虚拟环境运行")
        print("="*60)
        print(f"当前Python: {sys.executable}")
        print(f"\n直接运行: {venv_python} matching/example_usage.py")
        print("或双击: matching/run_example.bat")
        print("="*60)
        print()

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from matching.bidirectional_matching import BidirectionalMatcher
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def example_1_query_policies():
    """示例1: 企业/行业 → 政策查询"""
    print("="*60)
    print("示例1: 企业/行业 → 政策查询")
    print("="*60)
    
    # 初始化匹配器
    matcher = BidirectionalMatcher(project_root)
    
    # 查询文本
    query_text = "开饭店"
    top_k = 10
    
    # 执行查询
    results = matcher.query_policies_by_enterprise(query_text, top_k=top_k)
    
    # 显示结果
    print(f"\n查询文本: {query_text}")
    print(f"找到 {len(results)} 个相关政策:\n")
    
    # 加载政策信息
    df_policies = pd.read_parquet(project_root / "data_intermediate/policies_clean.parquet")
    policy_dict = df_policies.set_index("policy_id")["title"].to_dict()
    
    for i, (policy_id, score) in enumerate(results, 1):
        policy_title = policy_dict.get(policy_id, f"政策ID: {policy_id}")
        print(f"{i:2d}. [相似度: {score:.6f}] {policy_title}")


def example_2_retrieve_enterprises():
    """示例2: 政策 → 企业检索"""
    print("\n" + "="*60)
    print("示例2: 政策 → 企业检索")
    print("="*60)
    
    # 初始化匹配器
    matcher = BidirectionalMatcher(project_root)
    
    # 政策ID
    policy_id = 0
    top_k = 20
    
    # 执行检索
    results = matcher.retrieve_enterprises_by_policy(policy_id, top_k=top_k)
    
    # 显示结果
    print(f"\n政策ID: {policy_id}")
    print(f"找到 {len(results)} 个相关企业:\n")
    
    # 加载企业信息
    df_enterprises = pd.read_parquet(project_root / "data_intermediate/enterprises_filtered.parquet")
    company_dict = df_enterprises.set_index("enterprise_id")["name"].to_dict()
    
    for i, (company_id, score) in enumerate(results, 1):
        company_name = company_dict.get(company_id, f"企业ID: {company_id}")
        print(f"{i:2d}. [优先级: {score:.4f}] {company_name}")


def example_3_batch_query():
    """示例3: 批量查询"""
    print("\n" + "="*60)
    print("示例3: 批量查询")
    print("="*60)
    
    # 初始化匹配器
    matcher = BidirectionalMatcher(project_root)
    
    # 多个查询文本
    queries = [
        "开饭店",
        "餐饮业",
        "制造业",
        "科技创新",
        "中小企业"
    ]
    
    # 加载政策信息
    df_policies = pd.read_parquet(project_root / "data_intermediate/policies_clean.parquet")
    policy_dict = df_policies.set_index("policy_id")["title"].to_dict()
    
    for query in queries:
        print(f"\n查询: {query}")
        results = matcher.query_policies_by_enterprise(query, top_k=5)
        
        if results:
            print(f"  找到 {len(results)} 个相关政策:")
            for policy_id, score in results[:3]:  # 只显示前3个
                policy_title = policy_dict.get(policy_id, f"政策ID: {policy_id}")
                print(f"    - [{score:.6f}] {policy_title[:50]}...")
        else:
            print("  未找到相关政策")


def example_4_custom_query():
    """示例4: 自定义查询（用户输入）"""
    print("\n" + "="*60)
    print("示例4: 自定义查询")
    print("="*60)
    
    # 初始化匹配器
    matcher = BidirectionalMatcher(project_root)
    
    # 用户输入
    print("\n请输入查询文本（如：开饭店、餐饮、制造业等）:")
    query_text = input("> ").strip()
    
    if not query_text:
        print("查询文本不能为空")
        return
    
    # 执行查询
    results = matcher.query_policies_by_enterprise(query_text, top_k=10)
    
    # 显示结果
    print(f"\n查询结果: {query_text}")
    print(f"找到 {len(results)} 个相关政策:\n")
    
    # 加载政策信息
    df_policies = pd.read_parquet(project_root / "data_intermediate/policies_clean.parquet")
    policy_dict = df_policies.set_index("policy_id")["title"].to_dict()
    
    for i, (policy_id, score) in enumerate(results, 1):
        policy_title = policy_dict.get(policy_id, f"政策ID: {policy_id}")
        print(f"{i:2d}. [相似度: {score:.6f}]")
        print(f"    政策ID: {policy_id}")
        print(f"    标题: {policy_title}")
        print()


def main():
    """主函数"""
    print("双向匹配API使用示例\n")
    print("请选择要运行的示例:")
    print("  1. 企业/行业 → 政策查询")
    print("  2. 政策 → 企业检索")
    print("  3. 批量查询")
    print("  4. 自定义查询（交互式）")
    print("  0. 运行所有示例")
    
    choice = input("\n请输入选项 (0-4): ").strip()
    
    if choice == '0':
        example_1_query_policies()
        example_2_retrieve_enterprises()
        example_3_batch_query()
    elif choice == '1':
        example_1_query_policies()
    elif choice == '2':
        example_2_retrieve_enterprises()
    elif choice == '3':
        example_3_batch_query()
    elif choice == '4':
        example_4_custom_query()
    else:
        print("无效的选项")


if __name__ == "__main__":
    main()

