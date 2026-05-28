#!/usr/bin/env python3
"""根据映射后的行业大类重新生成行业嵌入向量。"""
import json, os, sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from matching.torch_metadata_fix import apply_torch_metadata_fix

apply_torch_metadata_fix()

import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoTokenizer

df = pd.read_parquet(project_root / "data_intermediate/triples_policy_entity.parquet")
industries = sorted(df[df["object_type"] == "industry"]["object"].unique())
print(f"行业节点数: {len(industries)}")
for i, ind in enumerate(industries):
    print(f"  {i}: {ind}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese").to(device)
model.eval()

embs = []
with torch.no_grad():
    for ind in industries:
        enc = tokenizer(ind, padding=True, truncation=True, max_length=64, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).expand(out[0].size()).float()
        pooled = (out[0] * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        embs.append(pooled.cpu())

emb_arr = torch.cat(embs, dim=0).numpy()
out_path = project_root / "embeddings/industry_text_emb.npy"
np.save(out_path, emb_arr)

idx_map = {ind: i for i, ind in enumerate(industries)}
with open(project_root / "embeddings/industry_index.json", "w", encoding="utf-8") as f:
    json.dump(idx_map, f, ensure_ascii=False, indent=2)

print(f"行业嵌入: {emb_arr.shape} -> {out_path}")
