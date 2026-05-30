#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融合文本、层级、时间向量，生成统一的政策特征。

根据流程图：
- 政策：文本向量 + 层级结构向量 + 时间戳向量

输入：
- embeddings/policy_text_concat_emb.npy（标题+内容拼接向量，1536维）
- features/level_emb.npy（层级向量，32维）
- features/time_emb.npy（时间向量，32维）

输出：
- features/policy_feature_fused.npy
- features/policy_feature_meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


class AlignmentMLP(nn.Module):
    """将异构维度特征投影到统一的低维潜空间。"""

    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        mid_dim = (in_dim + out_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def project_to_aligned_space(
    features: np.ndarray, in_dim: int, out_dim: int = 512, batch_size: int = 1024
) -> np.ndarray:
    """使用 AlignmentMLP 将特征投影到统一维度。"""
    model = AlignmentMLP(in_dim, out_dim)
    model.eval()
    x = torch.tensor(features, dtype=torch.float32)
    aligned_parts = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            aligned_parts.append(model(x[i : i + batch_size]))
    aligned = torch.cat(aligned_parts, dim=0).numpy()
    return aligned


def align_vectors_by_policy_id(
    text_emb: np.ndarray,
    text_index: dict,
    level_emb: np.ndarray,
    time_emb: np.ndarray,
    policy_ids: list
) -> tuple:
    """
    根据policy_id对齐向量
    
    Args:
        text_emb: 文本向量数组
        text_index: policy_id到文本向量索引的映射
        level_emb: 层级向量数组
        time_emb: 时间向量数组
        policy_ids: 需要对齐的policy_id列表
    
    Returns:
        对齐后的文本、层级、时间向量
    """
    aligned_text = []
    aligned_level = []
    aligned_time = []
    valid_policy_ids = []
    
    for pid in policy_ids:
        # 层级和时间向量按顺序对应policies_clean中的policy_id
        # 需要找到对应的索引
        if pid < len(level_emb):
            level_vec = level_emb[pid]
            time_vec = time_emb[pid]
        else:
            continue
        
        # 文本向量通过索引映射查找
        text_idx = text_index.get(str(pid))
        if text_idx is not None and isinstance(text_idx, int) and text_idx < len(text_emb):
            text_vec = text_emb[text_idx]
        else:
            continue
        
        aligned_text.append(text_vec)
        aligned_level.append(level_vec)
        aligned_time.append(time_vec)
        valid_policy_ids.append(pid)
    
    return (
        np.array(aligned_text),
        np.array(aligned_level),
        np.array(aligned_time),
        valid_policy_ids
    )


def main():
    parser = argparse.ArgumentParser(description="融合政策多模态特征")
    parser.add_argument("--text_emb", type=str, default="embeddings/policy_text_concat_emb.npy")
    parser.add_argument("--text_index", type=str, default="embeddings/policy_index.json")
    parser.add_argument("--level_emb", type=str, default="features/level_emb.npy")
    parser.add_argument("--time_emb", type=str, default="features/time_emb.npy")
    parser.add_argument("--policies_clean", type=str, default="data_intermediate/policies_clean.parquet")
    parser.add_argument("--text_weight", type=float, default=1.0)
    parser.add_argument("--attr_weight", type=float, default=1.0)
    parser.add_argument("--normalize", action="store_true", help="对各模态单独L2归一化")
    parser.add_argument(
        "--policy_output_tag",
        type=str,
        default="",
        help="若非空（如 a2_joint），则写入 policy_feature_fused_{tag}.npy / aligned / meta，且不覆盖 enterprise_feature_aligned",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    tag = (args.policy_output_tag or "").strip()
    suffix = f"_{tag}" if tag else ""
    out_path = project_root / "features" / f"policy_feature_fused{suffix}.npy"
    meta_path = project_root / "features" / f"policy_feature_meta{suffix}.json"

    # 加载向量
    text_emb = np.load(project_root / args.text_emb)
    level_emb = np.load(project_root / args.level_emb)
    time_emb = np.load(project_root / args.time_emb)
    
    # 加载索引映射
    with open(project_root / args.text_index, "r", encoding="utf-8") as f:
        text_index_raw = json.load(f)
        # 反转索引：从policy_id到向量索引（确保key是字符串）
        text_index = {str(v): int(k) for k, v in text_index_raw.items()}
    
    # 加载政策ID列表（使用policies_clean作为基准）
    df_policies = pd.read_parquet(project_root / args.policies_clean)
    policy_ids = df_policies["policy_id"].tolist()
    
    print(f"文本向量数量: {len(text_emb)}")
    print(f"层级向量数量: {len(level_emb)}")
    print(f"时间向量数量: {len(time_emb)}")
    print(f"政策ID数量: {len(policy_ids)}")
    
    # 对齐向量
    print("对齐向量...")
    aligned_text, aligned_level, aligned_time, valid_policy_ids = align_vectors_by_policy_id(
        text_emb, text_index, level_emb, time_emb, policy_ids
    )
    
    print(f"对齐后向量数量: {len(aligned_text)}")
    
    # Session 4 Step 1b — 强制 L2 归一化文本嵌入，解决 L2 范数失衡
    # 768 维 text embedding 的 L2 norm (~14) 远大于 32 维 level/time (~1-5)，
    # 未归一化拼接会导致高维文本特征在训练中吞噬低维属性特征。
    aligned_text = l2_normalize(aligned_text)

    if args.normalize:
        aligned_level = l2_normalize(aligned_level)
        aligned_time = l2_normalize(aligned_time)

    # 按照流程图拼接：文本向量 + 层级向量 + 时间向量
    text_part = args.text_weight * aligned_text
    attr_part = args.attr_weight * np.concatenate([aligned_level, aligned_time], axis=1)
    fused = np.concatenate([text_part, attr_part], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, fused)

    # ---- 特征对齐：投影到512维统一空间 ----
    aligned_dim = 512
    print(f"\n投影政策特征到 {aligned_dim}D 对齐空间...")
    policy_aligned = project_to_aligned_space(fused, fused.shape[1], aligned_dim)
    policy_aligned_path = project_root / "features" / f"policy_feature_aligned{suffix}.npy"
    np.save(policy_aligned_path, policy_aligned)
    print(f"  政策对齐向量: {policy_aligned.shape} -> {policy_aligned_path}")

    if tag:
        ent_ref = project_root / "features" / "enterprise_feature_aligned.npy"
        print(f"  已指定 --policy_output_tag={tag!r}：跳过企业侧重投影，GAT 仍使用现有 {ent_ref.name}")
    else:
        enterprise_fused_path = project_root / "features" / "enterprise_feature_fused.npy"
        if enterprise_fused_path.exists():
            ent_fused = np.load(enterprise_fused_path)
            print(f"投影企业特征到 {aligned_dim}D 对齐空间...")
            ent_aligned = project_to_aligned_space(ent_fused, ent_fused.shape[1], aligned_dim)
            ent_aligned_path = project_root / "features" / "enterprise_feature_aligned.npy"
            np.save(ent_aligned_path, ent_aligned)
            print(f"  企业对齐向量: {ent_aligned.shape} -> {ent_aligned_path}")
        else:
            enterprise_text_path = project_root / "embeddings" / "enterprise_text_emb.npy"
            if enterprise_text_path.exists():
                ent_text = np.load(enterprise_text_path)
                print(f"投影企业文本嵌入到 {aligned_dim}D 对齐空间...")
                ent_aligned = project_to_aligned_space(ent_text, ent_text.shape[1], aligned_dim)
                ent_aligned_path = project_root / "features" / "enterprise_feature_aligned.npy"
                np.save(ent_aligned_path, ent_aligned)
                print(f"  企业对齐向量: {ent_aligned.shape} -> {ent_aligned_path}")

    meta = {
        "text_emb": str(project_root / args.text_emb),
        "level_emb": str(project_root / args.level_emb),
        "time_emb": str(project_root / args.time_emb),
        "fused_path": str(out_path),
        "aligned_path": str(policy_aligned_path),
        "aligned_dim": aligned_dim,
        "num_policies": len(fused),
        "dims": {
            "text": text_part.shape[1],
            "level": aligned_level.shape[1],
            "time": aligned_time.shape[1],
            "attr": attr_part.shape[1],
            "fused": fused.shape[1],
            "aligned": aligned_dim,
        },
        "normalize": args.normalize,
        "text_l2_normalized": True,  # Session 4: 强制 L2 归一化文本嵌入
        "text_weight": args.text_weight,
        "attr_weight": args.attr_weight,
        "fusion_method": "concatenate(L2_norm_text, [level|time]) + alignment_mlp",
        "policy_output_tag": tag or None,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[OK] 政策特征融合与对齐完成")
    print(f"- 融合向量: {out_path} ({fused.shape})")
    print(f"- 对齐向量: {policy_aligned_path} ({policy_aligned.shape})")
    print(f"- 元信息: {meta_path}")


if __name__ == "__main__":
    main()

