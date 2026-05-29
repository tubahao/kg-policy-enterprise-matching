#!/usr/bin/env python3
"""Session 3 — Task 4: 离线政策文本嵌入 (Text Embedder)

使用 HuggingFace shibing624/text2vec-base-chinese 模型，
将 1,892 条活跃政策的 text_for_llm 编码为 768 维向量。
输出供 graph_builder.py 挂载到 Policy 节点特征。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm


def load_policies(policies_path: Path) -> tuple[List[str], List[str], List[dict]]:
    """加载政策数据，返回排序后的 (policy_ids, texts, policies) 三元组."""
    with open(policies_path, "r", encoding="utf-8") as f:
        policies = json.load(f)

    # 仅保留 active 政策，按 policy_id 排序 (确定性顺序)
    active = [p for p in policies if p.get("status") == "active"]
    active.sort(key=lambda p: p["policy_id"])

    policy_ids = [p["policy_id"] for p in active]
    texts = [p.get("text_for_llm", "") or "" for p in active]

    empty_count = sum(1 for t in texts if not t.strip())
    if empty_count:
        print(f"  [WARN] {empty_count} 条政策 text_for_llm 为空，将使用零向量")

    return policy_ids, texts, active


def get_device() -> torch.device:
    """检测最佳可用设备: CUDA > MPS > CPU."""
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


def embed_policies(
    policy_ids: List[str],
    texts: List[str],
    model_name: str,
    batch_size: int,
    max_seq_length: int,
    device: torch.device,
) -> torch.Tensor:
    """批量编码政策文本为 768 维向量."""
    from sentence_transformers import SentenceTransformer

    print(f"  加载模型: {model_name}")
    model = SentenceTransformer(model_name, device=str(device))
    model.max_seq_length = max_seq_length

    print(f"  最大序列长度: {max_seq_length}")
    print(f"  批量大小: {batch_size}")
    print(f"  政策数量: {len(texts)}")

    # 批量编码, tqdm 进度条
    embeddings_np: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    embeddings = torch.from_numpy(embeddings_np).float()
    return embeddings


def save_outputs(
    embeddings: torch.Tensor,
    policy_ids: List[str],
    output_path: Path,
    index_path: Path,
) -> None:
    """保存嵌入张量和 ID 索引."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存 tensor
    torch.save(embeddings, output_path)
    print(f"  [OK] 嵌入张量 → {output_path}")
    print(f"       形状: {list(embeddings.shape)} (N_policies × 768)")

    # 保存索引映射
    index = {
        "description": "policy_id → row_index 映射, 与 policy_text_emb.pt 对齐",
        "generated": datetime.now().isoformat(),
        "model": "shibing624/text2vec-base-chinese",
        "embedding_dim": 768,
        "num_policies": len(policy_ids),
        "policy_id_to_idx": {pid: i for i, pid in enumerate(policy_ids)},
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"  [OK] 索引映射 → {index_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session 3 — 离线政策文本嵌入 (Text2Vec)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/policies_final.json",
        help="policies_final.json 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/text_embeddings/policy_text_emb.pt",
        help="嵌入张量输出路径 (.pt)",
    )
    parser.add_argument(
        "--output-index",
        type=str,
        default="data/processed/text_embeddings/policy_emb_index.json",
        help="ID→行索引映射输出路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="shibing624/text2vec-base-chinese",
        help="HuggingFace 模型名",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="编码批量大小",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="最大序列长度 (text2vec 上限 512)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / args.input
    output_path = project_root / args.output
    index_path = project_root / args.output_index

    print("=" * 60)
    print("Session 3 — Task 4: 离线政策文本嵌入")
    print("=" * 60)

    # 1. 加载数据
    print("[1/3] 加载政策数据...")
    policy_ids, texts, _ = load_policies(input_path)
    print(f"  活跃政策: {len(policy_ids)} 条")
    avg_len = sum(len(t) for t in texts) / max(len(texts), 1)
    print(f"  平均 text_for_llm 长度: {avg_len:.0f} 字符")

    # 2. 选择设备
    print("[2/3] 初始化编码器...")
    device = get_device()

    # 3. 编码
    print("[3/3] 批量文本嵌入...")
    embeddings = embed_policies(
        policy_ids,
        texts,
        model_name=args.model,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        device=device,
    )

    # 保存
    save_outputs(embeddings, policy_ids, output_path, index_path)

    # 统计
    norms = embeddings.norm(dim=1)
    print(f"\n  嵌入统计:")
    print(f"    L2 范数 — min={norms.min():.4f}, mean={norms.mean():.4f}, max={norms.max():.4f}")
    zero_vecs = (norms < 1e-8).sum().item()
    if zero_vecs:
        print(f"    零向量数: {zero_vecs}")

    print("\n" + "=" * 60)
    print(f"[OK] 文本嵌入完成 — {len(policy_ids)} 条政策 × 768 维")
    print("=" * 60)


if __name__ == "__main__":
    main()
