#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 KG-BERT 官方脚本进行真实链路预测实验。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def _copy_dataset(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _truncate_lines(path: Path, keep_lines: int):
    if keep_lines <= 0 or not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    orig = len(lines)
    if orig <= keep_lines:
        return {"original": orig, "kept": orig}
    with path.open("w", encoding="utf-8") as f:
        f.writelines(lines[:keep_lines])
    return {"original": orig, "kept": keep_lines}


def _parse_ranks(path: Path):
    ranks = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            a, b = ln.split("\t")
            ranks.append(int(a) + 1)
            ranks.append(int(b) + 1)
    if not ranks:
        return {"MRR": 0.0, "MR": 0.0, "Hits@10": 0.0, "Hits@3": 0.0, "Hits@1": 0.0}
    arr = np.array(ranks, dtype=np.float64)
    return {
        "MRR": float(np.mean(1.0 / arr)),
        "MR": float(np.mean(arr)),
        "Hits@10": float(np.mean(arr <= 10)),
        "Hits@3": float(np.mean(arr <= 3)),
        "Hits@1": float(np.mean(arr <= 1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", type=str, default="bert-base-chinese")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--max_train_lines", type=int, default=0, help=">0 时截断 train.tsv 行数用于快速实验")
    parser.add_argument("--max_dev_lines", type=int, default=0, help=">0 时截断 dev.tsv 行数用于快速实验")
    parser.add_argument("--max_test_lines", type=int, default=0, help=">0 时截断 test.tsv 行数用于快速实验")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    kgbert_root = project_root.parents[1] / "KG-BERT" / "kg-bert"
    src_data = project_root / "reports" / "real_comparison_data" / "kgbert_policykg"
    dst_data = kgbert_root / "data" / "policykg_real"
    out_dir = project_root / "reports" / "real_comparison_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out = kgbert_root / f"output_policykg_real_{run_tag}"
    run_out.mkdir(parents=True, exist_ok=True)

    _copy_dataset(src_data, dst_data)
    trunc_info = {
        "train.tsv": _truncate_lines(dst_data / "train.tsv", args.max_train_lines),
        "dev.tsv": _truncate_lines(dst_data / "dev.tsv", args.max_dev_lines),
        "test.tsv": _truncate_lines(dst_data / "test.tsv", args.max_test_lines),
    }

    cmd = [
        str(project_root / "venv_graph" / "Scripts" / "python.exe"),
        "-u",
        "run_bert_link_prediction.py",
        "--task_name", "kg",
        "--do_train",
        "--do_eval",
        "--do_predict",
        "--data_dir", "./data/policykg_real",
        "--bert_model", args.bert_model,
        "--max_seq_length", str(args.max_seq_length),
        "--train_batch_size", str(args.train_batch_size),
        "--learning_rate", str(args.learning_rate),
        "--num_train_epochs", str(args.num_train_epochs),
        "--output_dir", f"./{run_out.name}/",
        "--gradient_accumulation_steps", "1",
        "--eval_batch_size", str(args.eval_batch_size),
    ]
    log_path = out_dir / "kgbert_stdout.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if args.no_cuda:
        cmd.append("--no_cuda")
    print(f"[KG-BERT] 启动命令: {' '.join(cmd)}", flush=True)
    print(f"[KG-BERT] 日志文件: {log_path}", flush=True)

    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(kgbert_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="ignore",
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            line_stripped = line.rstrip("\n")
            # 训练关键进度直接回显到终端
            if (
                "Epoch" in line_stripped
                or "Writing example" in line_stripped
                or "Running training" in line_stripped
                or "Iteration" in line_stripped
                or "Training loss" in line_stripped
                or "hit@10" in line_stripped.lower()
                or "Mean rank" in line_stripped
                or "Mean reciprocal rank" in line_stripped
            ):
                print(f"[KG-BERT] {line_stripped}", flush=True)
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"KG-BERT 运行失败，见日志: {log_path}")

    # 依据官方脚本文件名规则定位 ranks 文件
    prefix = f"policykg_real_{args.train_batch_size}_{args.learning_rate}_{args.max_seq_length}_{args.num_train_epochs}"
    rank_file = kgbert_root / f"{prefix}_ranks.txt"
    if not rank_file.exists():
        # 回退：扫描 *_ranks.txt 的最新文件
        cands = sorted(kgbert_root.glob("*_ranks.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise FileNotFoundError("KG-BERT 未生成 ranks 文件")
        rank_file = cands[0]

    metrics = _parse_ranks(rank_file)
    result = {
        "model": "KG-BERT",
        "data_dir": str(dst_data),
        "rank_file": str(rank_file),
        "metrics": metrics,
        "params": {
            "bert_model": args.bert_model,
            "max_seq_length": args.max_seq_length,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "max_train_lines": args.max_train_lines,
            "max_dev_lines": args.max_dev_lines,
            "max_test_lines": args.max_test_lines,
        },
        "data_truncation": trunc_info,
    }
    out_json = out_dir / "kgbert_results.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

