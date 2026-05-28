#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用中文BERT对政策标题/内容生成向量：
- 输入：data_intermediate/policies_filtered.parquet
- 输出：
    embeddings/policy_title_emb.npy
    embeddings/policy_content_emb.npy
    embeddings/policy_text_concat_emb.npy
    embeddings/policy_index.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

# 设置Hugging Face镜像源（国内加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(texts: List[str], tokenizer, model, device, batch_size: int, max_length: int) -> np.ndarray:
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_text = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded)
            pooled = mean_pooling(output, encoded["attention_mask"])
            embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser(description="生成政策文本向量")
    parser.add_argument("--input", type=str, default="data_intermediate/policies_filtered.parquet")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / args.input
    out_dir = project_root / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    titles = df["title"].fillna("").astype(str).tolist()
    contents = df["content"].fillna("").astype(str).tolist()

    # 若内容为空，退化为使用标题
    contents_filled = [c if c.strip() else t for c, t in zip(contents, titles)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    print(f"设备: {device}")
    print("编码标题向量...")
    title_emb = encode_texts(titles, tokenizer, model, device, args.batch_size, args.max_length)

    print("编码内容向量...")
    content_emb = encode_texts(contents_filled, tokenizer, model, device, args.batch_size, args.max_length)

    np.save(out_dir / "policy_title_emb.npy", title_emb)
    np.save(out_dir / "policy_content_emb.npy", content_emb)
    np.save(out_dir / "policy_text_concat_emb.npy", np.concatenate([title_emb, content_emb], axis=1))

    index_map = {int(pid): idx for idx, pid in enumerate(df["policy_id"].tolist())}
    with open(out_dir / "policy_index.json", "w", encoding="utf-8") as f:
        json.dump(index_map, f, ensure_ascii=False, indent=2)

    print("[OK] 向量生成完成")
    print(f"- 标题向量: {out_dir / 'policy_title_emb.npy'}")
    print(f"- 内容向量: {out_dir / 'policy_content_emb.npy'}")
    print(f"- 文本拼接向量: {out_dir / 'policy_text_concat_emb.npy'}")
    print(f"- 索引映射: {out_dir / 'policy_index.json'}")


if __name__ == "__main__":
    main()

