#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融 B2：生成「无时间衰减」的重要性 parquet（默认与主脚本一致：无 PPR 融合 + GAT）。

评测示例：--policy_importance_parquet evaluation/policy_importance_with_decay_a2_joint_b2.parquet
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

    p = argparse.ArgumentParser(description="单独生成 B2（无时间衰减）重要性 parquet")
    p.add_argument("--output_tag", default="a2_joint_b2")
    p.add_argument(
        "--gat_emb",
        default=os.environ.get("KGE_B2_GAT_EMB", "graph/gat_policy_emb_contrastive_a2_joint.npy"),
    )
    p.add_argument("--alpha_ppr", type=float, default=0.4)
    p.add_argument("--no_use_gat", action="store_true")
    p.add_argument(
        "--legacy_ppr_fusion",
        action="store_true",
        help="旧版：启用 PPR 融合（不加 --no_ppr_fusion）",
    )
    args = p.parse_args()

    py = _project_venv_python(root)
    cmd = [
        str(py),
        str(root / "evaluation" / "policy_importance_with_decay.py"),
        "--output_tag",
        args.output_tag,
        "--alpha_ppr",
        str(args.alpha_ppr),
        "--no_time_decay",
    ]
    if not args.no_use_gat:
        cmd.extend(["--use_gat", "--gat_emb", args.gat_emb])
    if not args.legacy_ppr_fusion:
        cmd.append("--no_ppr_fusion")

    print("运行:", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(root)).returncode


if __name__ == "__main__":
    sys.exit(main())
