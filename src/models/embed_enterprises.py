#!/usr/bin/env python3
"""Session 4 — Step 1a: 企业文本离线嵌入 (Enterprise Text Embedder)

使用 HuggingFace shibing624/text2vec-base-chinese 模型，
对 enterprises_final.json 中的 6,393 家企业生成 768 维文本向量。
文本拼接策略: "{name}，所属行业：{major_industry}-{sub_industry}。主营业务：{scope}"
输出供 graph_builder.py 挂载到 Enterprise 节点特征。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm


def load_enterprises(enterprises_path: Path) -> tuple[List[str], List[str], List[dict]]:
    """加载企业数据，返回排序后的 (enterprise_ids, texts, enterprises)."""
    with open(enterprises_path, "r", encoding="utf-8") as f:
        enterprises = json.load(f)

    enterprises.sort(key=lambda e: e["name"])

    enterprise_ids = [e["name"] for e in enterprises]
    texts = [
        f"{e['name']}，所属行业：{e['major_industry']}-{e['sub_industry']}。主营业务：{e['scope']}"
        for e in enterprises
    ]

    empty_count = sum(1 for t in texts if not t.strip())
    if empty_count:
        print(f"  [WARN] {empty_count} 家企业文本为空，将使用零向量")

    return enterprise_ids, texts, enterprises


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


def embed_enterprises(
    enterprise_ids: List[str],
    texts: List[str],
    model_name: str,
    batch_size: int,
    max_seq_length: int,
    device: torch.device,
) -> torch.Tensor:
    """批量编码企业文本为 768 维向量."""
    from sentence_transformers import SentenceTransformer

    print(f"  加载模型: {model_name}")
    model = SentenceTransformer(model_name, device=str(device))
    model.max_seq_length = max_seq_length

    print(f"  最大序列长度: {max_seq_length}")
    print(f"  批量大小: {batch_size}")
    print(f"  企业数量: {len(texts)}")

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
    enterprise_ids: List[str],
    output_path: Path,
    index_path: Path,
) -> None:
    """保存嵌入张量和 ID 索引."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(embeddings, output_path)
    print(f"  [OK] 嵌入张量 → {output_path}")
    print(f"       形状: {list(embeddings.shape)} (N_enterprises × 768)")

    index = {
        "description": "enterprise_name → row_index 映射, 与 enterprise_text_emb.pt 对齐",
        "generated": datetime.now().isoformat(),
        "model": "shibing624/text2vec-base-chinese",
        "embedding_dim": 768,
        "num_enterprises": len(enterprise_ids),
        "enterprise_id_to_idx": {eid: i for i, eid in enumerate(enterprise_ids)},
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"  [OK] 索引映射 → {index_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session 4 — 企业文本离线嵌入 (Text2Vec)"
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
        default="data/processed/text_embeddings/enterprise_text_emb.pt",
        help="嵌入张量输出路径 (.pt)",
    )
    parser.add_argument(
        "--output-index",
        type=str,
        default="data/processed/text_embeddings/enterprise_emb_index.json",
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
    print("Session 4 — Step 1a: 企业文本离线嵌入")
    print("=" * 60)

    # 1. 加载数据
    print("[1/3] 加载企业数据...")
    enterprise_ids, texts, _ = load_enterprises(input_path)
    print(f"  企业数量: {len(enterprise_ids)}")
    avg_len = sum(len(t) for t in texts) / max(len(texts), 1)
    print(f"  平均文本长度: {avg_len:.0f} 字符")

    # 2. 选择设备
    print("[2/3] 初始化编码器...")
    device = get_device()

    # 3. 编码
    print("[3/3] 批量文本嵌入...")
    embeddings = embed_enterprises(
        enterprise_ids,
        texts,
        model_name=args.model,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        device=device,
    )

    # 保存
    save_outputs(embeddings, enterprise_ids, output_path, index_path)

    # 统计
    norms = embeddings.norm(dim=1)
    print(f"\n  嵌入统计:")
    print(f"    L2 范数 — min={norms.min():.4f}, mean={norms.mean():.4f}, max={norms.max():.4f}")
    zero_vecs = (norms < 1e-8).sum().item()
    if zero_vecs:
        print(f"    零向量数: {zero_vecs}")

    print("\n" + "=" * 60)
    print(f"[OK] 企业文本嵌入完成 — {len(enterprise_ids)} 家企业 × 768 维")
    print("=" * 60)


if __name__ == "__main__":
    main()
