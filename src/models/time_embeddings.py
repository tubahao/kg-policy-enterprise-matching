#!/usr/bin/env python3
"""Session 4 — Step 2: 企业时间序列编码器 (Temporal Encoder with GRU)

针对 enterprises_final.json 中企业的 insurance_time_series (8 年 values + padding_mask)，
使用双向 GRU 提取企业动态生存状态的低维表征。

输入: data/processed/enterprises_final.json
输出:
    data/processed/time_embeddings/enterprise_temporal_emb.pt   [N, 64]
    data/processed/time_embeddings/enterprise_temporal_meta.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Temporal GRU Encoder
# ---------------------------------------------------------------------------

class TemporalGRUEncoder(nn.Module):
    """双向 GRU 编码器，处理含 padding_mask 的企业参保时间序列。

    序列: 8 年 (2017-2024) log_values + padding_mask (1=存续, 0=未成立)。
    输出: 固定维度的企业动态生存状态表征。
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_proj = nn.Sequential(
            nn.Linear(gru_out_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )

        for m in self.output_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, values: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """前向传播。

        Args:
            values: [batch, seq_len] 企业年参保人数 (log 值)。
            padding_mask: [batch, seq_len] 1=有效年份, 0=企业未成立。

        Returns:
            [batch, output_dim] 企业存活状态嵌入。
        """
        x = values.unsqueeze(-1)  # [batch, seq_len, 1]

        lengths = padding_mask.sum(dim=1).long().clamp(min=1)

        # pack_padded_sequence 要求 lengths 按降序排列
        lengths_sorted, sort_idx = lengths.sort(descending=True)
        _, unsort_idx = sort_idx.sort()
        x_sorted = x[sort_idx]

        packed = nn.utils.rnn.pack_padded_sequence(
            x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
        )

        _, h_n = self.gru(packed)

        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h = torch.cat([h_forward, h_backward], dim=-1)
        else:
            h = h_n[-1, :, :]

        h = h[unsort_idx]  # 恢复原始顺序
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_enterprise_time_series(
    enterprises_path: Path,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """从 enterprises_final.json 加载时间序列数据。

    Returns:
        (enterprise_ids, values_array [N, 8], padding_mask [N, 8])
    """
    with open(enterprises_path, "r", encoding="utf-8") as f:
        enterprises = json.load(f)

    enterprises.sort(key=lambda e: e["name"])

    enterprise_ids = []
    all_values = []
    all_masks = []

    for e in enterprises:
        ts = e.get("insurance_time_series")
        if ts is None:
            continue

        enterprise_ids.append(e["name"])
        all_values.append(ts["log_values"])
        all_masks.append(ts["padding_mask"])

    values_arr = np.array(all_values, dtype=np.float32)
    mask_arr = np.array(all_masks, dtype=np.float32)

    return enterprise_ids, values_arr, mask_arr


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  使用 CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  使用 MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("  使用 CPU")
    return device


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_batched(
    model: TemporalGRUEncoder,
    values: np.ndarray,
    padding_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """批量编码时间序列，返回 [N, output_dim] 张量."""
    model.eval()
    all_embeddings: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            v_batch = torch.from_numpy(values[start: start + batch_size]).to(device)
            m_batch = torch.from_numpy(padding_mask[start: start + batch_size]).to(device)
            emb = model(v_batch, m_batch)
            all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session 4 — 企业时间序列 GRU 编码"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/enterprises_final.json",
        help="enterprises_final.json 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/time_embeddings/enterprise_temporal_emb.pt",
        help="输出张量路径 (.pt)",
    )
    parser.add_argument(
        "--output-meta",
        type=str,
        default="data/processed/time_embeddings/enterprise_temporal_meta.json",
        help="输出元信息路径",
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="GRU 隐藏维度")
    parser.add_argument("--output-dim", type=int, default=64, help="输出维度")
    parser.add_argument("--num-layers", type=int, default=2, help="GRU 层数")
    parser.add_argument("--batch-size", type=int, default=128, help="编码批量大小")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / args.input
    output_path = project_root / args.output
    meta_path = project_root / args.output_meta

    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Session 4 — Step 2: 企业时间序列 GRU 编码")
    print("=" * 60)

    # 1. 加载数据
    print("[1/3] 加载企业时间序列...")
    enterprise_ids, values, masks = load_enterprise_time_series(input_path)
    num_enterprises = len(enterprise_ids)
    seq_len = values.shape[1]
    mask_coverage = masks.mean(axis=0)
    print(f"  企业数量: {num_enterprises}")
    print(f"  序列长度: {seq_len} 年")
    print(f"  padding_mask 年度覆盖率: {[f'{c:.1%}' for c in mask_coverage]}")

    # 2. 初始化模型
    print("[2/3] 初始化 GRU 编码器...")
    device = get_device()

    model = TemporalGRUEncoder(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        bidirectional=True,
    ).to(device)
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 编码
    print("[3/3] 批量编码...")
    embeddings = encode_batched(model, values, masks, args.batch_size, device)
    print(f"  输出形状: {list(embeddings.shape)}")

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"  [OK] 嵌入张量 → {output_path}")

    meta = {
        "description": "企业动态生存状态 GRU 编码 (Session 4 Step 2)",
        "generated": datetime.now().isoformat(),
        "num_enterprises": num_enterprises,
        "seq_len": seq_len,
        "output_dim": args.output_dim,
        "model": "TemporalGRUEncoder (bidirectional GRU)",
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "enterprise_ids": enterprise_ids[:20],
        "mask_coverage_per_year": [float(c) for c in mask_coverage],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [OK] 元信息 → {meta_path}")

    # 统计
    norms = embeddings.norm(dim=1)
    print(f"\n  嵌入统计:")
    print(f"    L2 范数 — min={norms.min():.4f}, mean={norms.mean():.4f}, max={norms.max():.4f}")
    zero_vecs = (norms < 1e-8).sum().item()
    if zero_vecs:
        print(f"    零向量数: {zero_vecs}")

    print("\n" + "=" * 60)
    print(f"[OK] 企业时间序列编码完成 — {num_enterprises} 家企业 × {args.output_dim} 维")
    print("=" * 60)


if __name__ == "__main__":
    main()
