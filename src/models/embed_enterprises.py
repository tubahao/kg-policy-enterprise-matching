#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用中文BERT对企业文本生成向量：
- 输入：data_intermediate/enterprises_filtered.parquet
- 输出：
    embeddings/enterprise_text_emb.npy
    embeddings/enterprise_index.json

企业文本构建：企业名称 + 所属行业 + 行业大类 + 经营范围
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


def build_enterprise_text(name: str, industry: str, industry_major: str, scope: str) -> str:
    """构建企业描述文本：企业名称 + 所属行业 + 行业大类 + 经营范围"""
    parts = []
    if pd.notna(name) and str(name).strip():
        parts.append(str(name).strip())
    if pd.notna(industry) and str(industry).strip():
        parts.append(f"所属行业：{str(industry).strip()}")
    if pd.notna(industry_major) and str(industry_major).strip():
        parts.append(f"行业大类：{str(industry_major).strip()}")
    if pd.notna(scope) and str(scope).strip():
        parts.append(f"经营范围：{str(scope).strip()}")
    
    text = "，".join(parts)
    return text if text else "企业"


def main():
    parser = argparse.ArgumentParser(description="生成企业文本向量")
    parser.add_argument("--input", type=str, default="data_intermediate/enterprises_filtered.parquet")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / args.input
    out_dir = project_root / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    
    # 构建企业描述文本
    enterprise_texts = []
    for _, row in df.iterrows():
        text = build_enterprise_text(
            row.get("name", ""),
            row.get("industry", ""),
            row.get("industry_major", ""),
            row.get("scope", "")
        )
        enterprise_texts.append(text)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    print(f"设备: {device}")
    print(f"企业数量: {len(enterprise_texts)}")
    print("编码企业文本向量...")
    enterprise_emb = encode_texts(enterprise_texts, tokenizer, model, device, args.batch_size, args.max_length)

    np.save(out_dir / "enterprise_text_emb.npy", enterprise_emb)

    # 创建索引映射（使用enterprise_id或name作为key）
    index_map = {}
    for idx, row in df.iterrows():
        ent_id = row.get("enterprise_id") or row.get("name", f"enterprise_{idx}")
        index_map[str(ent_id)] = idx
    
    with open(out_dir / "enterprise_index.json", "w", encoding="utf-8") as f:
        json.dump(index_map, f, ensure_ascii=False, indent=2)

    print("[OK] 企业向量生成完成")
    print(f"- 企业文本向量: {out_dir / 'enterprise_text_emb.npy'}")
    print(f"- 向量维度: {enterprise_emb.shape}")
    print(f"- 索引映射: {out_dir / 'enterprise_index.json'}")


if __name__ == "__main__":
    main()

