# -*- coding: utf-8 -*-
"""从完整 atise_policykg 抽样生成小规模子集，便于在 CPU 上完成一次完整 filtered 评估。"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "qwen-kge"
        / "ai-knowledge-graph-main"
        / "reports"
        / "real_comparison_data"
        / "atise_policykg",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=Path(__file__).resolve().parent / "atise_policykg_mini",
    )
    p.add_argument("--train_lines", type=int, default=6000)
    p.add_argument("--valid_lines", type=int, default=300)
    p.add_argument("--test_lines", type=int, default=50)
    args = p.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    if not (src / "entity2id.txt").is_file():
        raise SystemExit(f"缺少源数据: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("entity2id.txt", "relation2id.txt"):
        shutil.copy(src / name, dst / name)

    def head_lines(fname: str, n: int) -> None:
        lines = (src / fname).read_text(encoding="utf-8").splitlines(keepends=True)
        (dst / fname).write_text("".join(lines[:n]), encoding="utf-8")

    head_lines("train.txt", args.train_lines)
    head_lines("valid.txt", args.valid_lines)
    head_lines("test.txt", args.test_lines)
    print(f"已写入 {dst}  train={args.train_lines} valid={args.valid_lines} test={args.test_lines}")


if __name__ == "__main__":
    main()
