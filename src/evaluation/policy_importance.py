#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于节点向量与重要性分数的政策重要性评估：
- 输入：
    data_intermediate/policies_clean.parquet
    graph/policy_node_emb.npy
    graphrag/importance_scores.npy
- 输出：
    evaluation/policy_importance.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="计算政策重要性评分")
    parser.add_argument("--policies", type=str, default="data_intermediate/policies_clean.parquet")
    parser.add_argument("--node_emb", type=str, default="graph/policy_node_emb.npy")
    parser.add_argument("--importance", type=str, default="graphrag/importance_scores.npy")
    parser.add_argument("--beta", type=float, default=0.05, help="时间衰减系数")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_parquet(project_root / args.policies)

    node_emb = np.load(project_root / args.node_emb)
    importance = np.load(project_root / args.importance)

    current_year = dt.datetime.now().year
    years = df["year"].fillna(current_year).astype(float)
    delta_t = np.maximum(0.0, current_year - years)

    # 若 importance 长度不足，进行对齐填充
    imp_aligned = np.zeros(len(df), dtype=float)
    limit = min(len(importance), len(df))
    imp_aligned[:limit] = importance[:limit]

    # 简易衰减与差异度指标
    importance_decay = imp_aligned * np.exp(-args.beta * delta_t)
    baseline = np.mean(imp_aligned) if len(imp_aligned) > 0 else 0.0
    delta_I = np.abs(imp_aligned - baseline)

    out_df = pd.DataFrame(
        {
            "policy_id": df["policy_id"],
            "title": df["title"],
            "raw_importance": imp_aligned,
            "decayed_importance": importance_decay,
            "delta_I": delta_I,
        }
    )

    out_dir = project_root / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "policy_importance.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("✅ 政策重要性评估完成")
    print(f"- 结果: {out_path}")
    print(f"- 节点向量形状: {node_emb.shape}")
    print(f"- 原始重要性均值: {baseline:.6f}")


if __name__ == "__main__":
    main()

