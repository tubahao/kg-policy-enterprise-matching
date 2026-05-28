#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融合企业文本和时间向量，生成统一的企业特征。

根据流程图：
- 企业：文本向量 + 时间戳向量

输入：
- embeddings/enterprise_text_emb.npy
- features/enterprise_time_emb.npy（如果企业有时间信息）

输出：
- features/enterprise_feature_fused.npy
- features/enterprise_feature_meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def align_enterprise_vectors(
    text_emb: np.ndarray,
    text_index: dict,
    time_emb: np.ndarray,
    enterprise_ids: list
) -> tuple:
    """
    根据enterprise_id对齐向量
    
    Args:
        text_emb: 文本向量数组
        text_index: enterprise_id到文本向量索引的映射
        time_emb: 时间向量数组
        enterprise_ids: 需要对齐的enterprise_id列表
    
    Returns:
        对齐后的文本、时间向量和有效的enterprise_id列表
    """
    aligned_text = []
    aligned_time = []
    valid_enterprise_ids = []
    
    # 时间向量按顺序对应enterprises_time_series中的企业
    # 需要找到文本向量中对应的索引
    for i, ent_id in enumerate(enterprise_ids):
        if i >= len(time_emb):
            break
        
        time_vec = time_emb[i]
        
        # 文本向量通过索引映射查找
        text_idx = text_index.get(str(ent_id)) or text_index.get(ent_id)
        if text_idx is not None and text_idx < len(text_emb):
            text_vec = text_emb[text_idx]
        else:
            # 如果找不到，尝试使用索引i
            if i < len(text_emb):
                text_vec = text_emb[i]
            else:
                continue
        
        aligned_text.append(text_vec)
        aligned_time.append(time_vec)
        valid_enterprise_ids.append(ent_id)
    
    return (
        np.array(aligned_text),
        np.array(aligned_time),
        valid_enterprise_ids
    )


def main():
    parser = argparse.ArgumentParser(description="融合企业多模态特征")
    parser.add_argument("--text_emb", type=str, default="embeddings/enterprise_text_emb.npy")
    parser.add_argument("--text_index", type=str, default="embeddings/enterprise_index.json")
    parser.add_argument("--time_emb", type=str, default="features/enterprise_time_emb.npy")
    parser.add_argument("--enterprises_time_series", type=str, default="data_intermediate/enterprises_time_series.parquet")
    parser.add_argument("--text_weight", type=float, default=1.0)
    parser.add_argument("--time_weight", type=float, default=1.0)
    parser.add_argument("--normalize", action="store_true", help="对各模态单独L2归一化")
    parser.add_argument("--use_time", action="store_true", default=True, help="是否使用时间向量")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "features" / "enterprise_feature_fused.npy"
    meta_path = project_root / "features" / "enterprise_feature_meta.json"

    text_emb = np.load(project_root / args.text_emb)
    
    if args.use_time:
        time_emb_path = project_root / args.time_emb
        if not time_emb_path.exists():
            print(f"警告: 时间向量文件不存在: {time_emb_path}")
            print("将只使用文本向量")
            args.use_time = False
    
    if args.use_time:
        time_emb = np.load(project_root / args.time_emb)
        
        # 加载企业ID列表（使用enterprises_time_series作为基准，因为它有完整的时间序列）
        df_enterprises = pd.read_parquet(project_root / args.enterprises_time_series)
        enterprise_ids = df_enterprises["enterprise_id"].tolist()
        
        # 加载文本向量索引
        with open(project_root / args.text_index, "r", encoding="utf-8") as f:
            text_index = json.load(f)
        
        print(f"文本向量数量: {len(text_emb)}")
        print(f"时间向量数量: {len(time_emb)}")
        print(f"企业ID数量: {len(enterprise_ids)}")
        
        # 对齐向量
        print("对齐向量...")
        aligned_text, aligned_time, valid_enterprise_ids = align_enterprise_vectors(
            text_emb, text_index, time_emb, enterprise_ids
        )
        
        print(f"对齐后向量数量: {len(aligned_text)}")
        
        if args.normalize:
            aligned_text = l2_normalize(aligned_text)
            aligned_time = l2_normalize(aligned_time)
        
        # 按照流程图拼接：文本向量 + 时间向量
        text_part = args.text_weight * aligned_text
        time_part = args.time_weight * aligned_time
        fused = np.concatenate([text_part, time_part], axis=1)
    else:
        # 如果企业没有时间信息，只使用文本向量
        if args.normalize:
            text_emb = l2_normalize(text_emb)
        fused = args.text_weight * text_emb
        aligned_text = text_emb
        aligned_time = None
        valid_enterprise_ids = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, fused)

    meta = {
        "text_emb": str(project_root / args.text_emb),
        "time_emb": str(project_root / args.time_emb) if args.use_time else None,
        "fused_path": str(out_path),
        "num_enterprises": len(fused),
        "dims": {
            "text": aligned_text.shape[1],
            "time": aligned_time.shape[1] if aligned_time is not None else 0,
            "fused": fused.shape[1],
        },
        "normalize": args.normalize,
        "text_weight": args.text_weight,
        "time_weight": args.time_weight if args.use_time else 0,
        "use_time": args.use_time,
        "fusion_method": "concatenate",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] 企业特征融合完成")
    print(f"- 融合向量: {out_path}")
    print(f"- 向量维度: {fused.shape}")
    print(f"- 元信息: {meta_path}")


if __name__ == "__main__":
    main()

