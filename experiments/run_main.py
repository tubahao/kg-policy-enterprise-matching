#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2 主实验（BASE / a2_joint）一键复跑：建图 → GraphRAG → GAT → 层级/时间衰减 → 双向匹配评测。

前提：`data_clean/preprocess_policies.py` 已生成最新 `data_intermediate/*`；
政策特征已存在 `features/policy_feature_aligned_a2_joint.npy`（joint 文本编码线）。

产物覆盖：
- graph/graph_data.bin, graph/meta.json
- graphrag/importance_scores.npy
- graph/gat_*_contrastive_a2_joint.npy
- evaluation/policy_importance_with_decay_a2_joint.parquet（及 csv/stats；默认 --no_ppr_fusion）
- matching/evaluation_results_a2_joint_full_pipeline_rerun.json（默认不覆盖冻结榜）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _project_venv_python(root: Path) -> Path:
    """与消融日志一致：优先使用项目内 venv_graph（含 CUDA 版 DGL），避免误用 Anaconda CPU 版 DGL。"""
    win = root / "venv_graph" / "Scripts" / "python.exe"
    if win.is_file():
        return win
    nix = root / "venv_graph" / "bin" / "python"
    if nix.is_file():
        return nix
    return Path(sys.executable)


# 本管线目标为 GPU；若本机无 CUDA，请显式传 --device cpu（且需 CPU 版 DGL）。
_DEFAULT_DEVICE = "cuda"


def run_step(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 72, flush=True)
    print(">>", " ".join(cmd), flush=True)
    print("=" * 72, flush=True)
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    p = argparse.ArgumentParser(description="A2 joint 全流程复跑")
    p.add_argument(
        "--device",
        type=str,
        default=_DEFAULT_DEVICE,
        help=f"GAT 设备（默认 {_DEFAULT_DEVICE}）",
    )
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument(
        "--skip_gat",
        action="store_true",
        help="跳过 GAT 重训（仅当图节点数与现有 gat_policy_emb_contrastive_a2_joint.npy 行数一致时使用）",
    )
    p.add_argument(
        "--eval_output",
        type=str,
        default="matching/evaluation_results_a2_joint_full_pipeline_rerun.json",
        help="评测写出路径，避免覆盖 evaluation_results_a2_joint_full_pipeline.json",
    )
    args = p.parse_args()

    py = str(_project_venv_python(ROOT))
    run_step(
        [
            py,
            "graph/build_graph.py",
            "--feat",
            "features/policy_feature_aligned_a2_joint.npy",
        ],
        ROOT,
    )
    run_step(
        [
            py,
            "graph/graphrag_pipeline.py",
            "--graph_path",
            "graph/graph_data.bin",
            "--top_k",
            "50",
            "--k_hop",
            "2",
        ],
        ROOT,
    )
    if not args.skip_gat:
        run_step(
            [
                py,
                "graph/train_gat_contrastive.py",
                "--device",
                args.device,
                "--num_epochs",
                str(args.num_epochs),
                "--max_industry_pairs_per_policy",
                "20",
                "--max_pairs_per_epoch",
                "120000",
                "--transmit_drop_rate",
                "0.4",
                "--reverse_neighbor_cap",
                "0",
                "--feature_policy_aligned",
                "features/policy_feature_aligned_a2_joint.npy",
                "--policy_emb_out",
                "graph/gat_policy_emb_contrastive_a2_joint.npy",
                "--company_emb_out",
                "graph/gat_company_emb_contrastive_a2_joint.npy",
                "--industry_emb_out",
                "graph/gat_industry_emb_contrastive_a2_joint.npy",
            ],
            ROOT,
        )
    run_step(
        [
            py,
            "evaluation/policy_importance_with_decay.py",
            "--output_tag",
            "a2_joint",
            "--no_ppr_fusion",
            "--use_gat",
            "--gat_emb",
            "graph/gat_policy_emb_contrastive_a2_joint.npy",
        ],
        ROOT,
    )
    run_step(
        [
            py,
            "matching/evaluate_matching.py",
            "--experiment_profile",
            "a2_base",
            "--max_enterprise_queries",
            "300",
            "--max_industry_queries",
            "30",
            "--max_policy_queries",
            "200",
            "--output",
            args.eval_output,
        ],
        ROOT,
    )
    print("\n[OK] A2 joint 全流程结束。", flush=True)


if __name__ == "__main__":
    main()
