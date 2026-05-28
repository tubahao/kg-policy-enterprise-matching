# -*- coding: utf-8 -*-
"""若不存在则生成 A2 用的 policy_text_joint_bert_emb.npy / policy_text_joint_bert_index.json。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def _encode_texts(texts, tokenizer, model, device, batch_size: int = 16, max_length: int = 256):
    embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = model(**encoded)
            pooled = _mean_pooling(out, encoded["attention_mask"])
            embs.append(pooled.cpu())
    return torch.cat(embs, dim=0).numpy()


def ensure_joint_policy_embeddings(project_root: Path) -> None:
    root = Path(project_root)
    emb_path = root / "embeddings" / "policy_text_joint_bert_emb.npy"
    idx_path = root / "embeddings" / "policy_text_joint_bert_index.json"
    if emb_path.exists() and idx_path.exists():
        return
    print("生成 A2 文本级拼接后编码向量 policy_text_joint_bert_emb.npy …", flush=True)
    df = pd.read_parquet(root / "data_intermediate" / "policies_clean.parquet")
    titles = df["title"].fillna("").astype(str).tolist()
    contents = df["content"].fillna("").astype(str).tolist()
    merged = [f"{t} [SEP] {c if c.strip() else t}" for t, c in zip(titles, contents)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", local_files_only=False)
    model = AutoModel.from_pretrained("bert-base-chinese", local_files_only=False).to(device)
    embs = _encode_texts(merged, tokenizer, model, device)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embs)
    idx_map = {int(pid): int(i) for i, pid in enumerate(df["policy_id"].astype(int).tolist())}
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(idx_map, f, ensure_ascii=False, indent=2)
    print(f"已写入: {emb_path} , {idx_path}", flush=True)
