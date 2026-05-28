#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
企业数据处理脚本
功能：
1. 读取企业数据Excel文件
2. 识别并统一参保人数列名格式
3. 筛选有效数据（以最新年份参保人数为标准）
4. 保存处理后的文件
"""

import pandas as pd
import os
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置pandas选项以避免NumPy兼容性问题
pd.set_option('mode.chained_assignment', None)

def identify_insurance_columns(df):
    """
    识别参保人数相关的列
    支持格式：
    - uiNum_2024
    - 2017年参保人数
    - 参保人员2024
    """
    insurance_columns = {}
    
    for col in df.columns:
        col_str = str(col)
        
        # 匹配 uiNum_年份 格式
        match1 = re.search(r'uiNum_(\d{4})', col_str)
        if match1:
            year = match1.group(1)
            insurance_columns[col] = f"{year}年参保人数"
            continue
            
        # 匹配 年份年参保人数 格式
        match2 = re.search(r'(\d{4})年参保人数', col_str)
        if match2:
            year = match2.group(1)
            insurance_columns[col] = f"{year}年参保人数"
            continue
            
        # 匹配 参保人员年份 格式
        match3 = re.search(r'参保人员(\d{4})', col_str)
        if match3:
            year = match3.group(1)
            insurance_columns[col] = f"{year}年参保人数"
            continue
    
    return insurance_columns

def process_excel_file(file_path, output_dir):
    """
    处理单个Excel文件
    """
    print(f"正在处理文件: {os.path.basename(file_path)}")
    
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        print(f"  原始数据行数: {len(df)}")
        
        # 识别参保人数列
        insurance_columns = identify_insurance_columns(df)
        print(f"  识别到参保人数列: {list(insurance_columns.keys())}")
        
        if not insurance_columns:
            print(f"  警告: 未找到参保人数相关列，跳过此文件")
            return
        
        # 选择需要的列
        required_columns = ['企业名称', '经营状态', '法定代表人', '注册资本', '实缴资本', '所属行业', '经营范围']
        selected_columns = []
        
        # 添加基础信息列（如果存在）
        for col in required_columns:
            if col in df.columns:
                selected_columns.append(col)
            else:
                print(f"  警告: 未找到列 '{col}'")
        
        # 添加参保人数列
        selected_columns.extend(insurance_columns.keys())
        
        # 筛选存在的列
        available_columns = [col for col in selected_columns if col in df.columns]
        df_selected = df[available_columns].copy()
        
        # 重命名参保人数列
        rename_dict = {old_col: new_col for old_col, new_col in insurance_columns.items()}
        df_selected = df_selected.rename(columns=rename_dict)
        
        # 找到最新年份的参保人数列
        insurance_year_cols = [col for col in df_selected.columns if '年参保人数' in col]
        if not insurance_year_cols:
            print(f"  警告: 重命名后未找到参保人数列，跳过此文件")
            return
        
        # 提取年份并找到最新年份
        years = []
        for col in insurance_year_cols:
            match = re.search(r'(\d{4})年参保人数', col)
            if match:
                years.append(int(match.group(1)))
        
        if not years:
            print(f"  警告: 无法提取年份信息，跳过此文件")
            return
        
        latest_year = max(years)
        latest_year_col = f"{latest_year}年参保人数"
        
        print(f"  最新年份: {latest_year}, 对应列: {latest_year_col}")
        
        # 筛选数据：以最新年份参保人数为标准，排除空值或0
        if latest_year_col in df_selected.columns:
            # 处理参保人数数据：提取数字部分
            def extract_number(value):
                if pd.isna(value):
                    return 0
                if isinstance(value, (int, float)):
                    return value
                # 处理字符串格式，如"6人"、"0人"
                if isinstance(value, str):
                    # 提取数字部分
                    import re
                    match = re.search(r'(\d+)', str(value))
                    if match:
                        return int(match.group(1))
                return 0
            
            # 应用数字提取函数
            df_selected[latest_year_col] = df_selected[latest_year_col].apply(extract_number)
            
            # 筛选条件：最新年份参保人数大于0
            valid_mask = df_selected[latest_year_col] > 0
            df_filtered = df_selected[valid_mask].copy()
            
            print(f"  筛选后数据行数: {len(df_filtered)}")
            print(f"  筛选掉的行数: {len(df_selected) - len(df_filtered)}")
        else:
            print(f"  警告: 未找到最新年份列 {latest_year_col}")
            df_filtered = df_selected
        
        # 保存处理后的文件
        output_filename = f"processed_{os.path.basename(file_path)}"
        output_path = os.path.join(output_dir, output_filename)
        
        df_filtered.to_excel(output_path, index=False)
        print(f"  已保存到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"  错误: 处理文件时出错 - {str(e)}")
        return False

def main():
    """
    主函数
    """
    # 设置路径
    input_dir = r"E:\论文\知识图谱\数据\企业数据"
    output_dir = r"E:\论文\知识图谱\数据\企业数据\数据"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有Excel文件
    excel_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.xlsx') and not file.startswith('processed_'):
            excel_files.append(os.path.join(input_dir, file))
    
    print(f"找到 {len(excel_files)} 个Excel文件需要处理:")
    for file in excel_files:
        print(f"  - {os.path.basename(file)}")
    
    print("\n开始处理文件...")
    print("=" * 50)
    
    success_count = 0
    for file_path in excel_files:
        if process_excel_file(file_path, output_dir):
            success_count += 1
        print("-" * 30)
    
    print(f"\n处理完成！成功处理 {success_count}/{len(excel_files)} 个文件")
    print(f"处理后的文件保存在: {output_dir}")

if __name__ == "__main__":
    main()
