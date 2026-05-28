#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式双向匹配工具
支持用户输入查询文本或政策ID进行匹配
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
        print(f"\n直接运行: {venv_python} matching/interactive_matching.py")
        print("或双击: matching/run_matching.bat")
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


def load_policy_info():
    """加载政策信息，用于显示政策标题"""
    try:
        df = pd.read_parquet(project_root / "data_intermediate/policies_clean.parquet")
        return df.set_index("policy_id")["title"].to_dict()
    except Exception as e:
        print(f"警告: 无法加载政策信息: {e}")
        return {}


def load_company_info():
    """加载企业信息，用于显示企业名称"""
    try:
        df = pd.read_parquet(project_root / "data_intermediate/enterprises_filtered.parquet")
        return df.set_index("enterprise_id")["name"].to_dict()
    except Exception as e:
        print(f"警告: 无法加载企业信息: {e}")
        return {}


def query_policies(matcher, query_text, top_k=10):
    """查询政策"""
    print(f"\n{'='*60}")
    print(f"🔍 查询文本: {query_text}")
    print(f"{'='*60}\n")
    
    results = matcher.query_policies_by_enterprise(query_text, top_k=top_k)
    
    if not results:
        print("❌ 未找到相关政策")
        return
    
    print(f"✅ 找到 {len(results)} 个相关政策:\n")
    
    # 加载政策信息
    policy_info = load_policy_info()
    
    for i, (policy_id, score) in enumerate(results, 1):
        policy_title = policy_info.get(policy_id, f"政策ID: {policy_id}")
        print(f"{i:2d}. [相似度: {score:.6f}]")
        print(f"    政策ID: {policy_id}")
        print(f"    政策标题: {policy_title}")
        print()


def retrieve_enterprises(matcher, policy_id, top_k=20):
    """检索企业"""
    print(f"\n{'='*60}")
    print(f"📋 政策ID: {policy_id}")
    print(f"{'='*60}\n")
    
    results = matcher.retrieve_enterprises_by_policy(policy_id, top_k=top_k)
    
    if not results:
        print("❌ 未找到相关企业")
        print("   可能原因：")
        print("   1. 政策ID不存在于图中")
        print("   2. 该政策没有关联的企业")
        print("   3. 节点映射需要完善")
        return
    
    print(f"✅ 找到 {len(results)} 个相关企业:\n")
    
    # 加载企业信息
    company_info = load_company_info()
    
    for i, (company_id, score) in enumerate(results, 1):
        # 尝试多种格式查找企业名称
        company_name = None
        if isinstance(company_id, int):
            company_name = company_info.get(company_id)
        if not company_name:
            company_name = company_info.get(f"enterprise_{company_id}")
        if not company_name:
            company_name = f"企业ID: {company_id}"
        
        print(f"{i:2d}. [优先级: {score:.4f}]")
        print(f"    企业ID: {company_id}")
        print(f"    企业名称: {company_name}")
        print()


def show_menu():
    """显示菜单"""
    print("\n" + "="*60)
    print("📊 双向匹配工具")
    print("="*60)
    print("\n请选择操作:")
    print("  1. 企业/行业 → 政策查询（输入查询文本，如：开饭店）")
    print("  2. 政策 → 企业检索（输入政策ID）")
    print("  3. 批量查询（从文件读取查询）")
    print("  0. 退出")
    print("="*60)


def batch_query_from_file(matcher, file_path, top_k=10):
    """从文件批量查询"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"\n📁 从文件读取 {len(queries)} 个查询\n")
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] 查询: {query}")
            query_policies(matcher, query, top_k=top_k)
            
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")


def main():
    """主函数"""
    print("正在初始化双向匹配器...")
    print("（首次运行可能需要加载BERT模型，请稍候...）\n")
    
    try:
        matcher = BidirectionalMatcher(project_root)
        print("✅ 初始化完成！\n")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("\n请检查：")
        print("  1. 是否已运行预处理脚本生成嵌入向量")
        print("  2. 是否已构建图数据")
        print("  3. BERT模型是否已下载")
        return
    
    while True:
        show_menu()
        choice = input("\n请输入选项 (0-3): ").strip()
        
        if choice == '0':
            print("\n👋 再见！")
            break
        elif choice == '1':
            query_text = input("\n请输入查询文本（如：开饭店、餐饮、制造业等）: ").strip()
            if query_text:
                try:
                    top_k = int(input("返回前几个结果（默认10）: ").strip() or "10")
                except ValueError:
                    top_k = 10
                query_policies(matcher, query_text, top_k=top_k)
            else:
                print("❌ 查询文本不能为空")
        elif choice == '2':
            try:
                policy_id = int(input("\n请输入政策ID（如：0, 1, 100等）: ").strip())
                top_k = int(input("返回前几个结果（默认20）: ").strip() or "20")
                retrieve_enterprises(matcher, policy_id, top_k=top_k)
            except ValueError:
                print("❌ 请输入有效的政策ID（整数）")
        elif choice == '3':
            file_path = input("\n请输入查询文件路径（每行一个查询）: ").strip()
            if file_path:
                try:
                    top_k = int(input("每个查询返回前几个结果（默认10）: ").strip() or "10")
                except ValueError:
                    top_k = 10
                batch_query_from_file(matcher, file_path, top_k=top_k)
            else:
                print("❌ 文件路径不能为空")
        else:
            print("❌ 无效的选项，请重新选择")
        
        input("\n按回车键继续...")


if __name__ == "__main__":
    main()

