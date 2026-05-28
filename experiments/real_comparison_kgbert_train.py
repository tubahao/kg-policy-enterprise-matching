#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KG-BERT 三元组分类微调（政策-企业全量 kgbert_policykg），跳过官方脚本中极慢的 filtered 链接预测循环。
训练结束后可运行 real_comparison_kgbert_matching_eval.py 得到与主实验一致的 P/R/F1/MAP/NDCG。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bert_model", type=str, default="bert-base-chinese")
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument(
        "--no_amp",
        action="store_true",
        help="关闭 torch.cuda.amp 混合精度（默认开启）",
    )
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--fresh", action="store_true", help="删除 output_dir 后重训")
    p.add_argument(
        "--run_matching_eval",
        action="store_true",
        help="训练结束后运行 real_comparison_kgbert_matching_eval.py",
    )
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parents[1]
    kgbert_code = repo_root / "KG-BERT" / "kg-bert"
    data_dir = project_root / "reports" / "real_comparison_data" / "kgbert_policykg"
    out_dir = project_root / "reports" / "real_comparison_results" / "kgbert_policykg_out"
    py = project_root / "venv_graph" / "Scripts" / "python.exe"
    log_path = project_root / "reports" / "real_comparison_results" / "kgbert_train.log"

    if not data_dir.is_dir():
        raise FileNotFoundError(f"缺少数据目录，请先运行 real_comparison_prepare_data.py: {data_dir}")

    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(py),
        str(kgbert_code / "run_bert_link_prediction.py"),
        "--task_name",
        "kg",
        "--do_train",
        "--do_eval",
        "--do_predict",
        "--skip_link_prediction",
        "--data_dir",
        str(data_dir.resolve()),
        "--bert_model",
        args.bert_model,
        "--output_dir",
        str(out_dir.resolve()),
        "--max_seq_length",
        str(args.max_seq_length),
        "--train_batch_size",
        str(args.train_batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
    ]
    if args.no_cuda:
        cmd.append("--no_cuda")
    if not args.no_amp and not args.no_cuda:
        cmd.append("--use_amp")

    print("[KG-BERT] 启动训练...", flush=True)
    print(" ".join(cmd), flush=True)
    print(f"[KG-BERT] 日志: {log_path}", flush=True)

    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(kgbert_code), stdout=logf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"KG-BERT 训练失败，见 {log_path}")

    print("[KG-BERT] 训练完成。", flush=True)

    if args.run_matching_eval:
        ev = [
            str(py),
            str(project_root / "scripts" / "real_comparison_kgbert_matching_eval.py"),
            "--output_dir",
            str(out_dir),
            "--output",
            "reports/real_comparison_results/kgbert_matching_eval_testsplit.json",
        ]
        if args.no_cuda:
            ev.append("--no_cuda")
        print("[KG-BERT] 运行匹配评测...", flush=True)
        subprocess.run(ev, cwd=str(project_root), check=True)


if __name__ == "__main__":
    main()
