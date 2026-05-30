#!/usr/bin/env python3
"""Session 4 — Step 2: 政策层级与时间属性编码器 (Hierarchical-Temporal Attribute Encoder)

核心模块:
    LevelEmbedding     — 离散层级 (Policy1/2/3) 的可学习 Embedding
    TimeMLP            — 连续年份标量的 MLP 编码
    DeltaEncoder       — 标量差异值 (delta_l_ref, delta_t_ref) 的通用 MLP 编码器
    HierarchicalEncoder— 层级差异 (delta_l_ref) 专用编码器 (64-dim)
    TemporalEncoder    — 时间差异 (delta_t_ref) 专用编码器 (64-dim)

输入: policies_final.json (或 policies_cleaned.parquet, 向后兼容)
输出:
    features/level_emb.npy
    features/time_emb.npy
    features/policy_attributes.parquet
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


# ---------------------------------------------------------------------------
# Encoder Modules
# ---------------------------------------------------------------------------

class LevelEmbedding(nn.Module):
    """离散层级 (Policy1/Policy2/Policy3) 的可学习 Embedding 层."""

    def __init__(self, vocab_size: int, embedding_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, level_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(level_ids)


class TimeMLP(nn.Module):
    """连续年份标量的 MLP 编码器 (输出低维嵌入)."""

    def __init__(self, input_dim: int = 1, hidden_dims: List[int] = [64, 32], output_dim: int = 32):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeltaEncoder(nn.Module):
    """标量差异值 (delta_l_ref / delta_t_ref) 的通用 MLP 编码器。

    将单个标量 (如层级差 1, 时间差 3) 映射到 output_dim 维嵌入空间。
    用于在向量空间中表达政策-企业之间的层级距离和时间距离。
    """

    def __init__(
        self,
        output_dim: int = 64,
        hidden_dims: List[int] = [16, 32, 48],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = 1  # scalar input
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            delta: [B] 或 [B, 1] 标量差异值。

        Returns:
            [B, output_dim]
        """
        if delta.dim() == 1:
            delta = delta.unsqueeze(-1)
        return self.net(delta)


class HierarchicalEncoder(DeltaEncoder):
    """层级差异 (delta_l_ref) 专用编码器 — 策略与企业的行政层级距离."""

    def __init__(self, output_dim: int = 64):
        super().__init__(output_dim=output_dim)


class TemporalEncoder(DeltaEncoder):
    """时间差异 (delta_t_ref) 专用编码器 — 策略发布与企业成立/评估的时间距离."""

    def __init__(self, output_dim: int = 64):
        super().__init__(output_dim=output_dim)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

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


def build_level_embedding(levels: pd.Series, dim: int, seed: int) -> Tuple[np.ndarray, dict]:
    """构建层级嵌入 (PyTorch Embedding 层)."""
    unique_levels = ["<UNK>"] + sorted(
        {str(lv).strip() for lv in levels.tolist() if pd.notna(lv) and str(lv).strip()}
    )
    vocab = {lv: idx for idx, lv in enumerate(unique_levels)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LevelEmbedding(len(vocab), dim).to(device)
    model.eval()

    level_ids = torch.tensor(
        [vocab.get(str(lv).strip(), 0) if pd.notna(lv) else 0 for lv in levels.tolist()],
        dtype=torch.long,
    ).to(device)

    with torch.no_grad():
        emb = model(level_ids).cpu().numpy()

    return emb, vocab


def build_time_embedding(
    years_norm: np.ndarray, dim: int, seed: int, hidden_dims: List[int] = [64, 32]
) -> np.ndarray:
    """构建时间嵌入 (MLP)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = TimeMLP(input_dim=1, hidden_dims=hidden_dims, output_dim=dim).to(device)
    model.eval()

    years_tensor = torch.tensor(years_norm.reshape(-1, 1), dtype=torch.float32).to(device)

    with torch.no_grad():
        emb = model(years_tensor).cpu().numpy()

    return emb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="生成层级/时间嵌入")
    parser.add_argument("--input", type=str, default="data_intermediate/policies_clean.parquet")
    parser.add_argument("--level_dim", type=int, default=32)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--time_hidden_dims", type=str, default="64,32", help="时间MLP隐藏层维度，逗号分隔")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / args.input
    out_dir = project_root / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    years = parse_year(df.get("year"), df.get("publish_date"))
    years_norm = normalize_year(years)

    hidden_dims = [int(d) for d in args.time_hidden_dims.split(",")]

    print("构建层级嵌入...")
    level_emb, vocab = build_level_embedding(df.get("level"), args.level_dim, args.seed)
    print(f"层级向量形状: {level_emb.shape}")

    print("构建时间嵌入...")
    time_emb = build_time_embedding(years_norm, args.time_dim, args.seed, hidden_dims)
    print(f"时间向量形状: {time_emb.shape}")

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
