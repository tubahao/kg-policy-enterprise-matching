#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A3（论文中「仅标题 BERT」策略）全流程：融合 → GAT(对比) → 重要性衰减 → 双向匹配评测。
层级/时间向量沿用 features/level_emb.npy、time_emb.npy（不重算）。

产物后缀均为 a3_title，与 A2 BASE（a2_joint）并存。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("→", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    run(
        [
            PY,
            str(ROOT / "features" / "fuse_features.py"),
            "--text_emb",
            "embeddings/policy_title_emb.npy",
            "--text_index",
            "embeddings/policy_index.json",
            "--policy_output_tag",
            "a3_title",
        ]
    )
    run(
        [
            PY,
            str(ROOT / "graph" / "train_gat_contrastive.py"),
            "--device",
            "cuda",
            "--num_epochs",
            "30",
            "--max_industry_pairs_per_policy",
            "20",
            "--max_pairs_per_epoch",
            "120000",
            "--transmit_drop_rate",
            "0.4",
            "--reverse_neighbor_cap",
            "0",
            "--feature_policy_aligned",
            "features/policy_feature_aligned_a3_title.npy",
            "--policy_emb_out",
            "graph/gat_policy_emb_contrastive_a3_title.npy",
            "--company_emb_out",
            "graph/gat_company_emb_contrastive_a3_title.npy",
            "--checkpoint_best",
            "graph/checkpoints/gat_contrastive_best_a3_title.pt",
            "--checkpoint_final",
            "graph/checkpoints/gat_contrastive_final_a3_title.pt",
        ]
    )
    run(
        [
            PY,
            str(ROOT / "evaluation" / "policy_importance_with_decay.py"),
            "--target_level",
            "2",
            "--target_year",
            "2024",
            "--beta_level",
            "0.2",
            "--beta_time",
            "0.05",
            "--no_ppr_fusion",
            "--use_gat",
            "--gat_emb",
            "graph/gat_policy_emb_contrastive_a3_title.npy",
            "--output_tag",
            "a3_title",
        ]
    )
    run(
        [
            PY,
            str(ROOT / "matching" / "evaluate_matching.py"),
            "--experiment_profile",
            "a3_title",
            "--max_enterprise_queries",
            "300",
            "--max_industry_queries",
            "30",
            "--max_policy_queries",
            "200",
            "--output",
            "matching/evaluation_results_a3_title_full_pipeline.json",
        ]
    )
    print("A3 title 全流程完成。", flush=True)


if __name__ == "__main__":
    main()
