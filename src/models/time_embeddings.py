#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成企业时间嵌入（使用PyTorch MLP编码时间序列）：
- 输入：data_intermediate/enterprises_time_series.parquet
- 输出：
    features/enterprise_time_emb.npy
    features/enterprise_time_meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class TimeSeriesMLP(nn.Module):
    """时间序列MLP编码器"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # 添加dropout防止过拟合
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        
        # 初始化权重
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_insurance_values(values: np.ndarray) -> np.ndarray:
    """归一化参保人数"""
    values = np.asarray(values, dtype=float)
    # 使用log1p处理，因为数值范围很大（0-25300）
    values_log = np.log1p(values)
    mean = np.mean(values_log)
    std = np.std(values_log) or 1.0
    return (values_log - mean) / std


def build_time_series_vector(
    time_series: dict,
    year_range: tuple,
    fill_method: str = "zero"
) -> np.ndarray:
    """
    构建固定长度的时间序列向量
    
    Args:
        time_series: 字典格式 {年份: 参保人数}
        year_range: (起始年份, 结束年份)
        fill_method: 缺失值填充方法 ("zero", "forward", "mean")
    
    Returns:
        归一化后的时间序列向量
    """
    start_year, end_year = year_range
    num_years = end_year - start_year + 1
    vector = np.zeros(num_years, dtype=float)
    
    # 填充已知值
    for year in range(start_year, end_year + 1):
        idx = year - start_year
        year_str = str(year)
        if year_str in time_series:
            vector[idx] = float(time_series[year_str])
        elif str(year) in time_series:  # 也支持整数key
            vector[idx] = float(time_series[year])
    
    # 处理缺失值
    if fill_method == "forward":
        # 前向填充：用前一年的值填充
        last_value = 0.0
        for i in range(num_years):
            if vector[i] == 0:
                vector[i] = last_value
            else:
                last_value = vector[i]
    elif fill_method == "mean":
        # 用非零值的均值填充
        non_zero_values = vector[vector > 0]
        if len(non_zero_values) > 0:
            mean_value = np.mean(non_zero_values)
            vector[vector == 0] = mean_value
    # else: "zero" - 保持0值
    
    # 归一化
    vector_norm = normalize_insurance_values(vector)
    
    return vector_norm


def main():
    parser = argparse.ArgumentParser(description="生成企业时间嵌入")
    parser.add_argument("--input", type=str, default="data_intermediate/enterprises_time_series.parquet")
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--hidden_dims", type=str, default="128,64", help="MLP隐藏层维度，逗号分隔")
    parser.add_argument("--year_start", type=int, default=2016, help="时间序列起始年份")
    parser.add_argument("--year_end", type=int, default=2024, help="时间序列结束年份")
    parser.add_argument("--fill_method", type=str, default="forward", choices=["zero", "forward", "mean"],
                       help="缺失值填充方法: zero(0填充), forward(前向填充), mean(均值填充)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / args.input
    out_dir = project_root / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    
    year_range = (args.year_start, args.year_end)
    num_years = args.year_end - args.year_start + 1
    
    print(f"读取企业时间序列数据: {len(df)}条")
    print(f"时间范围: {args.year_start}-{args.year_end} ({num_years}年)")
    print(f"缺失值填充方法: {args.fill_method}")

    # 构建时间序列向量
    time_series_vectors = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        ts_json = row['time_series_json']
        if not ts_json or ts_json == "{}":
            continue
        
        try:
            ts = json.loads(ts_json)
            if not ts:
                continue
            
            vector = build_time_series_vector(ts, year_range, args.fill_method)
            time_series_vectors.append(vector)
            valid_indices.append(idx)
        except Exception as e:
            print(f"警告: 处理企业 {row.get('name', 'unknown')} 时出错: {e}")
            continue
    
    if len(time_series_vectors) == 0:
        raise ValueError("没有有效的时间序列数据")
    
    time_series_array = np.array(time_series_vectors)
    print(f"有效时间序列数量: {len(time_series_array)}")
    print(f"时间序列向量形状: {time_series_array.shape}")

    # 使用MLP编码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    hidden_dims = [int(d) for d in args.hidden_dims.split(",")]
    model = TimeSeriesMLP(input_dim=num_years, hidden_dims=hidden_dims, output_dim=args.time_dim).to(device)
    model.eval()
    
    # 批处理编码
    batch_size = 64
    embeddings = []
    
    with torch.no_grad():
        for start_idx in range(0, len(time_series_array), batch_size):
            batch = time_series_array[start_idx:start_idx + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            batch_emb = model(batch_tensor).cpu().numpy()
            embeddings.append(batch_emb)
    
    enterprise_time_emb = np.concatenate(embeddings, axis=0)
    print(f"企业时间向量形状: {enterprise_time_emb.shape}")

    # 保存结果
    np.save(out_dir / "enterprise_time_emb.npy", enterprise_time_emb)
    
    # 保存元信息
    meta = {
        "num_enterprises": len(enterprise_time_emb),
        "time_dim": args.time_dim,
        "year_range": year_range,
        "num_years": num_years,
        "fill_method": args.fill_method,
        "hidden_dims": hidden_dims,
        "valid_indices": valid_indices[:10],  # 只保存前10个作为示例
    }
    
    meta_path = out_dir / "enterprise_time_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print("[OK] 企业时间嵌入已生成")
    print(f"- 时间向量: {out_dir / 'enterprise_time_emb.npy'}")
    print(f"- 元信息: {meta_path}")
    print(f"- 向量维度: {enterprise_time_emb.shape}")


if __name__ == "__main__":
    main()

