#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成政策的层级与时间嵌入（使用PyTorch MLP）：
- 输入：data_intermediate/policies_clean.parquet
- 输出：
    features/policy_attributes.parquet
    features/level_emb.npy
    features/time_emb.npy
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def normalize_year(years: np.ndarray) -> np.ndarray:
    years = np.asarray(years, dtype=float)
    mean = np.nanmean(years)
    std = np.nanstd(years) or 1.0
    years = np.nan_to_num(years, nan=mean)
    return (years - mean) / std


def parse_year(col_year: pd.Series, col_date: pd.Series) -> np.ndarray:
    years: List[float] = []
    for y, d in zip(col_year, col_date):
        val = None
        if pd.notna(y):
            try:
                val = float(y)
            except Exception:
                val = None
        if val is None and pd.notna(d):
            try:
                val = dt.datetime.fromisoformat(str(d)).year
            except Exception:
                val = None
        years.append(val if val is not None else np.nan)
    return np.array(years, dtype=float)


class LevelEmbedding(nn.Module):
    """层级嵌入层（类似词嵌入）"""
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, level_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(level_ids)


class TimeMLP(nn.Module):
    """时间嵌入MLP"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
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


def build_level_embedding(levels: pd.Series, dim: int, seed: int) -> Tuple[np.ndarray, dict]:
    """构建层级嵌入（使用PyTorch Embedding层）"""
    unique_levels = ["<UNK>"] + sorted(
        {str(lv).strip() for lv in levels.tolist() if pd.notna(lv) and str(lv).strip()}
    )
    vocab = {lv: idx for idx, lv in enumerate(unique_levels)}
    
    # 使用PyTorch Embedding层
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LevelEmbedding(len(vocab), dim).to(device)
    model.eval()
    
    level_ids = torch.tensor(
        [vocab.get(str(lv).strip(), 0) if pd.notna(lv) else 0 for lv in levels.tolist()],
        dtype=torch.long
    ).to(device)
    
    with torch.no_grad():
        emb = model(level_ids).cpu().numpy()
    
    return emb, vocab


def build_time_embedding(years_norm: np.ndarray, dim: int, seed: int, hidden_dims: List[int] = [64, 32]) -> np.ndarray:
    """构建时间嵌入（使用PyTorch MLP）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    model = TimeMLP(input_dim=1, hidden_dims=hidden_dims, output_dim=dim).to(device)
    model.eval()
    
    years_tensor = torch.tensor(years_norm.reshape(-1, 1), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        emb = model(years_tensor).cpu().numpy()
    
    return emb


def main():
    parser = argparse.ArgumentParser(description="生成层级/时间嵌入")
    parser.add_argument("--input", type=str, default="data_intermediate/policies_clean.parquet")
    parser.add_argument("--level_dim", type=int, default=32)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--time_hidden_dims", type=str, default="64,32", help="时间MLP隐藏层维度，逗号分隔")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / args.input
    out_dir = project_root / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    years = parse_year(df.get("year"), df.get("publish_date"))
    years_norm = normalize_year(years)

    # 解析隐藏层维度
    hidden_dims = [int(d) for d in args.time_hidden_dims.split(",")]

    print("构建层级嵌入...")
    level_emb, vocab = build_level_embedding(df.get("level"), args.level_dim, args.seed)
    print(f"层级向量形状: {level_emb.shape}")

    print("构建时间嵌入...")
    time_emb = build_time_embedding(years_norm, args.time_dim, args.seed, hidden_dims)
    print(f"时间向量形状: {time_emb.shape}")

    # 保存属性表
    attr_df = pd.DataFrame(
        {
            "policy_id": df["policy_id"],
            "level": df.get("level"),
            "year": years,
            "publish_date": df.get("publish_date"),
        }
    )
    attr_df.to_parquet(out_dir / "policy_attributes.parquet", index=False)

    np.save(out_dir / "level_emb.npy", level_emb)
    np.save(out_dir / "time_emb.npy", time_emb)

    vocab_path = out_dir / "level_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("[OK] 层级/时间嵌入已生成")
    print(f"- 属性表: {out_dir / 'policy_attributes.parquet'}")
    print(f"- 层级向量: {out_dir / 'level_emb.npy'}")
    print(f"- 时间向量: {out_dir / 'time_emb.npy'}")
    print(f"- 层级词典: {vocab_path}")


if __name__ == "__main__":
    main()
