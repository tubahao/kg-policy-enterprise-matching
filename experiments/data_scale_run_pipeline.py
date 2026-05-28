#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 `data_scale_connected_subgraph.py` 或 `data_scale_industry_union_subgraph.py` 生成的子目录，一键：
1. 从全量对齐特征中切片出子图节点特征（policy / company / industry，行序与 build_graph 节点 id 一致）
2. 调用 `graph/build_graph.py` 生成 `graph_data.bin` + `meta.json`（写入子目录）
3. 调用 `train_gat_contrastive` 训练并保存该比例下的 checkpoint 与 GAT 嵌入

用法示例
--------
cd <project-root>
python scripts/data_scale_connected_subgraph.py --fractions 0.1 0.2 0.5 --seed 42
python scripts/data_scale_industry_union_subgraph.py
python scripts/data_scale_run_pipeline.py --scale_dir data_intermediate/data_scale_subgraphs/frac_0_1 --num_epochs 30
python scripts/data_scale_run_pipeline.py --all_under data_intermediate/data_scale_by_industry --num_epochs 30
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _prepare_features(scale_dir: Path, project_root: Path) -> None:
    from graph.build_graph import build_node_maps

    df_pol = pd.read_parquet(scale_dir / "policies_clean.parquet")
    df_p2e = pd.read_parquet(scale_dir / "triples_policy_entity.parquet")
    nm = build_node_maps(df_pol, df_p2e)

    feat_out = scale_dir / "features"
    feat_out.mkdir(parents=True, exist_ok=True)

    full_pol_df = pd.read_parquet(project_root / "data_intermediate/policies_clean.parquet")
    pid_to_row_full = {int(r.policy_id): i for i, r in enumerate(full_pol_df.itertuples(index=False))}
    full_pol_emb = np.load(project_root / "features/policy_feature_aligned.npy")

    if "orig_policy_id" not in df_pol.columns:
        raise ValueError("policies_clean.parquet 缺少 orig_policy_id，请重新运行 data_scale_connected_subgraph.py")
    n_pol = int(df_pol["policy_id"].max()) + 1 if len(df_pol) else 0
    pol_feat = np.zeros((n_pol, full_pol_emb.shape[1]), dtype=np.float32)
    for r in df_pol.itertuples(index=False):
        j = int(r.policy_id)
        op = int(r.orig_policy_id)
        ri = pid_to_row_full.get(op)
        if ri is None:
            raise KeyError(f"orig_policy_id={op} 不在全量 policies_clean 中")
        pol_feat[j] = full_pol_emb[ri]

    full_ent_df = pd.read_parquet(project_root / "data_intermediate/enterprises_filtered.parquet")
    name_to_row = {str(r.name): i for i, r in enumerate(full_ent_df.itertuples(index=False))}
    full_ent_emb = np.load(project_root / "features/enterprise_feature_aligned.npy")
    n_com = max(nm["company"].values(), default=-1) + 1
    com_feat = np.zeros((n_com, full_ent_emb.shape[1]), dtype=np.float32)
    for name, cid in nm["company"].items():
        ri = name_to_row.get(str(name))
        if ri is None:
            raise KeyError(f"企业 {name!r} 不在全量 enterprises_filtered 中")
        com_feat[int(cid)] = full_ent_emb[ri]

    ind_path = project_root / "embeddings/industry_index.json"
    full_ind_emb = np.load(project_root / "embeddings/industry_text_emb.npy")
    with open(ind_path, "r", encoding="utf-8") as f:
        ind_index = json.load(f)
    n_ind = max(nm["industry"].values(), default=-1) + 1
    ind_feat = np.zeros((n_ind, full_ind_emb.shape[1]), dtype=np.float32)
    for name, iid in nm["industry"].items():
        ix = ind_index.get(str(name))
        if ix is None:
            raise KeyError(f"行业 {name!r} 不在 industry_index.json 中")
        ind_feat[int(iid)] = full_ind_emb[int(ix)]

    np.save(feat_out / "policy_feature_aligned.npy", pol_feat)
    np.save(feat_out / "enterprise_feature_aligned.npy", com_feat)
    if n_ind > 0:
        np.save(feat_out / "industry_text_emb.npy", ind_feat)
    # build_graph 默认读 fused；此处用对齐特征维数一致即可
    np.save(feat_out / "policy_feature_fused.npy", pol_feat)
    print(f"[data_scale] 特征已写入 {feat_out}", flush=True)


def _run_build_graph(scale_dir: Path, project_root: Path) -> None:
    py = sys.executable
    rel_scale = scale_dir.relative_to(project_root)
    feat_rel = rel_scale / "features" / "policy_feature_fused.npy"
    cmd = [
        py,
        str(project_root / "graph" / "build_graph.py"),
        "--policies",
        str(rel_scale / "policies_clean.parquet"),
        "--p2p",
        str(rel_scale / "triples_policy_policy.parquet"),
        "--p2e",
        str(rel_scale / "triples_policy_entity.parquet"),
        "--feat",
        str(feat_rel),
        "--out",
        str(rel_scale / "graph_data.bin"),
        "--meta",
        str(rel_scale / "graph_meta.json"),
    ]
    print("[data_scale] build_graph:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(project_root))


def _run_train_gat(scale_dir: Path, project_root: Path, train_kwargs: dict) -> None:
    from graph.train_gat_contrastive import train_gat_contrastive

    rel = scale_dir.relative_to(project_root)
    pol_rel = str(rel / "policies_clean.parquet")
    feat_pol = str(rel / "features" / "policy_feature_aligned.npy")
    feat_com = str(rel / "features" / "enterprise_feature_aligned.npy")
    fo: Dict[str, str] = {"policy": feat_pol, "company": feat_com}
    ind_npy = scale_dir / "features" / "industry_text_emb.npy"
    if ind_npy.is_file() and np.load(ind_npy).shape[0] > 0:
        fo["industry"] = str(rel / "features" / "industry_text_emb.npy")
    train_gat_contrastive(
        project_root,
        graph_bin=str(rel / "graph_data.bin"),
        meta_json=str(rel / "graph_meta.json"),
        policies_clean_path=pol_rel,
        feature_override=fo,
        checkpoint_best=str(rel / "checkpoints" / "gat_contrastive_best.pt"),
        checkpoint_final=str(rel / "checkpoints" / "gat_contrastive_final.pt"),
        policy_emb_out=str(rel / "gat_policy_emb_contrastive.npy"),
        company_emb_out=str(rel / "gat_company_emb_contrastive.npy"),
        industry_emb_out=str(rel / "gat_industry_emb_contrastive.npy"),
        **train_kwargs,
    )


def _discover_scale_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "scale_meta.json").is_file()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale_dir", type=str, default="", help="单个子图目录，如 data_intermediate/data_scale_subgraphs/frac_0_1")
    ap.add_argument("--all_under", type=str, default="", help="对该目录下所有含 scale_meta.json 的子目录依次运行")
    ap.add_argument("--skip_build_graph", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--num_epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    project_root = PROJECT_ROOT
    dirs: List[Path] = []
    if args.scale_dir:
        dirs.append(project_root / args.scale_dir)
    if args.all_under:
        dirs.extend(_discover_scale_dirs(project_root / args.all_under))
    if not dirs:
        ap.error("请指定 --scale_dir 或 --all_under")

    train_kw = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "device": args.device,
    }

    for d in dirs:
        d = d.resolve()
        if not (d / "scale_meta.json").is_file():
            print(f"[skip] 非子图目录（无 scale_meta.json）: {d}", flush=True)
            continue
        print(f"\n========== 数据规模管线: {d.relative_to(project_root)} ==========", flush=True)
        _prepare_features(d, project_root)
        if not args.skip_build_graph:
            _run_build_graph(d, project_root)
        if not args.skip_train:
            (d / "checkpoints").mkdir(parents=True, exist_ok=True)
            _run_train_gat(d, project_root, train_kw)

    print("\n[data_scale] 全部完成。", flush=True)


if __name__ == "__main__":
    main()
