#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅跑消融 B1：用与主脚本相同代码路径生成「无层级衰减」的重要性 parquet。

说明：
- 子图 PageRank/PPR（graphrag/importance_scores.npy）不随 B1 改变，无需重跑 GraphRAG。
- 必须与生成 `policy_importance_with_decay_a2_joint.*` 时相同的
  `--no_ppr_fusion`、`--use_gat`、`--gat_emb` 等，否则与 BASE 不可比。

第二步评测（推荐，不依赖环境变量）：
  .\\venv_graph\\Scripts\\python.exe matching/evaluate_matching.py --experiment_profile a2_base \\
    --policy_importance_parquet evaluation/policy_importance_with_decay_a2_joint_b1.parquet \\
    --max_enterprise_queries 300 --max_industry_queries 30 --max_policy_queries 200 \\
    --output matching/evaluation_results_b1_no_level_a2_joint.json
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _project_venv_python(root: Path) -> Path:
    win = root / "venv_graph" / "Scripts" / "python.exe"
    if win.is_file():
        return win
    nix = root / "venv_graph" / "bin" / "python"
    if nix.is_file():
        return nix
    return Path(sys.executable)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    py = _project_venv_python(root)

    p = argparse.ArgumentParser(description="单独生成 B1（无层级衰减）重要性 parquet")
    p.add_argument(
        "--output_tag",
        default="a2_joint_b1",
        help="写入 evaluation/policy_importance_with_decay_{tag}.parquet",
    )
    p.add_argument(
        "--gat_emb",
        default=os.environ.get(
            "KGE_B1_GAT_EMB", "graph/gat_policy_emb_contrastive_a2_joint.npy"
        ),
        help="须与当前 A2 BASE 一致（默认 a2_joint GAT）；可用 KGE_B1_GAT_EMB 覆盖",
    )
    p.add_argument(
        "--legacy_ppr_fusion",
        action="store_true",
        help="与旧版一致：参与 PPR 融合（不加 --no_ppr_fusion）；默认与主流程一致为无 PPR 融合",
    )
    p.add_argument("--alpha_ppr", type=float, default=0.4)
    p.add_argument(
        "--no_use_gat",
        action="store_true",
        help="若 BASE 未开 GAT，可加此开关与 BASE 对齐",
    )
    args = p.parse_args()

    cmd = [
        str(py),
        str(root / "evaluation" / "policy_importance_with_decay.py"),
        "--output_tag",
        args.output_tag,
        "--alpha_ppr",
        str(args.alpha_ppr),
        "--no_level_decay",
    ]
    if not args.no_use_gat:
        cmd.extend(["--use_gat", "--gat_emb", args.gat_emb])
    if not args.legacy_ppr_fusion:
        cmd.append("--no_ppr_fusion")

    print("运行:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(root))
    if r.returncode != 0:
        return r.returncode

    rel_parquet = f"evaluation/policy_importance_with_decay_{args.output_tag}.parquet"
    print("\n下一步：在保持 KGE_GAT_ARTIFACT_TAG=a2_joint 的前提下，指向 B1 parquet，例如：")
    print(f'  $env:KGE_POLICY_IMPORTANCE_PARQUET="{rel_parquet}"')
    print("  python matching/evaluate_matching.py ... --output matching/evaluation_results_b1_no_level.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
