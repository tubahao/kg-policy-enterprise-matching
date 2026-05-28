#!/usr/bin/env python3
"""
清理三元组数据脚本
功能：
1. 删除所有结构性的反关系（implements, executes）
2. 删除所有政策-政策之间的关系（transmitsTo, executes）
只保留正向关系：supports, belongsTo, targetsIndustry
"""

import json
import os
from pathlib import Path


def clean_triples(input_file, output_file):
    """
    清理三元组数据
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print("=" * 60)
    print("开始清理三元组数据")
    print("=" * 60)
    
    # 需要删除的关系类型
    relations_to_remove = {
        'implements',  # supports的反向关系
        'executes',    # transmitsTo的反向关系
        'transmitsTo'  # 政策-政策关系
    }
    
    # 统计信息
    stats = {
        'total': 0,
        'removed': {
            'implements': 0,
            'executes': 0,
            'transmitsTo': 0
        },
        'kept': {
            'supports': 0,
            'belongsTo': 0,
            'targetsIndustry': 0,
            'other': 0
        }
    }
    
    print(f"\n读取输入文件: {input_file}")
    
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"总三元组数: {len(data)}")
        stats['total'] = len(data)
        
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return False
    
    # 清理数据
    print("\n开始清理...")
    cleaned_data = []
    
    for triple in data:
        predicate = triple.get('predicate', '')
        
        # 检查是否需要删除
        if predicate in relations_to_remove:
            stats['removed'][predicate] = stats['removed'].get(predicate, 0) + 1
            continue
        
        # 保留的关系
        cleaned_data.append(triple)
        if predicate in stats['kept']:
            stats['kept'][predicate] = stats['kept'][predicate] + 1
        else:
            stats['kept']['other'] = stats['kept']['other'] + 1
    
    # 保存清理后的数据
    print(f"\n保存清理后的数据到: {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 数据已保存")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
        return False
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("清理统计")
    print("=" * 60)
    print(f"总三元组数: {stats['total']:,}")
    print(f"\n删除的关系:")
    total_removed = 0
    for rel, count in stats['removed'].items():
        if count > 0:
            print(f"  {rel}: {count:,}")
            total_removed += count
    print(f"  总计删除: {total_removed:,}")
    
    print(f"\n保留的关系:")
    total_kept = 0
    for rel, count in stats['kept'].items():
        if count > 0:
            print(f"  {rel}: {count:,}")
            total_kept += count
    print(f"  总计保留: {total_kept:,}")
    
    print(f"\n清理完成！")
    print(f"  删除率: {total_removed / stats['total'] * 100:.2f}%")
    print(f"  保留率: {total_kept / stats['total'] * 100:.2f}%")
    
    return True


def main():
    """主函数"""
    # 文件路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    input_file = project_root / "output" / "extracted_triples_raw.json"
    output_file = project_root / "output" / "extracted_triples_cleaned.json"
    
    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    # 执行清理
    success = clean_triples(str(input_file), str(output_file))
    
    if success:
        print(f"\n✅ 清理完成！")
        print(f"  输入文件: {input_file}")
        print(f"  输出文件: {output_file}")
    else:
        print(f"\n❌ 清理失败！")


if __name__ == "__main__":
    main()


