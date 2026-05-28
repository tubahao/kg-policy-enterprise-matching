#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATISE 全量数据训练 + 主实验口径匹配评测（P/R/F1/MAP/NDCG，与 OpenKE/HippoRAG 脚本对齐）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _parse_metric_file(path: Path) -> dict:
    out = {}
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epoch", type=int, default=120, help="全量训练轮数（建议为 min_epoch 的整数倍以便最后一轮跑 test）")
    parser.add_argument("--min_epoch", type=int, default=10, help="每隔多少 epoch 做验证/早停判断")
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=120.0)
    parser.add_argument("--eta", type=int, default=10)
    parser.add_argument("--gran", type=int, default=1)
    parser.add_argument("--cmin", type=float, default=0.003)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="跳过训练，仅使用已有 params.pkl 跑匹配评测",
    )
    parser.add_argument(
        "--fresh_train",
        action="store_true",
        help="训练前删除数据目录下既有 ATISE/ 产物，避免混入旧 checkpoint",
    )
    args = parser.parse_args()

    if args.max_epoch % args.min_epoch != 0:
        print(
            "[ATISE] 警告: max_epoch 应为 min_epoch 整数倍，否则官方脚本可能在最后一轮不触发 filtered test；"
            "匹配评测不依赖该 test，但建议调整参数。",
            flush=True,
        )

    project_root = Path(__file__).resolve().parents[1]
    atise_root = project_root.parents[1] / "atise" / "ATISE"
    data_dir = project_root / "reports" / "real_comparison_data" / "atise_policykg"
    out_dir = project_root / "reports" / "real_comparison_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    py = str(project_root / "venv_graph" / "Scripts" / "python.exe")

    if not data_dir.is_dir():
        raise FileNotFoundError(f"缺少 ATISE 数据目录: {data_dir}")

    if args.fresh_train and not args.skip_train:
        atise_artifact_dir = data_dir / "ATISE"
        if atise_artifact_dir.exists():
            shutil.rmtree(atise_artifact_dir, ignore_errors=True)
            print(f"[ATISE] 已清理旧训练目录: {atise_artifact_dir}", flush=True)

    log_path = out_dir / "atise_train_full.log"

    if not args.skip_train:
        cmd = [
            py,
            "Main.py",
            "--model",
            "ATISE",
            "--dataset",
            str(data_dir.resolve()),
            "--max_epoch",
            str(args.max_epoch),
            "--min_epoch",
            str(args.min_epoch),
            "--dim",
            str(args.dim),
            "--batch",
            str(args.batch),
            "--lr",
            str(args.lr),
            "--gamma",
            str(args.gamma),
            "--eta",
            str(args.eta),
            "--loss",
            "logloss",
            "--timedisc",
            "0",
            "--gran",
            str(args.gran),
            "--cmin",
            str(args.cmin),
            "--cuda",
            str(bool(args.cuda)),
        ]
        print(f"[ATISE] 训练命令: {' '.join(cmd)}", flush=True)
        print(f"[ATISE] 训练日志: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(atise_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                lf.write(line)
                line_s = line.rstrip("\n")
                if any(
                    x in line_s
                    for x in (
                        "Epoch-",
                        "Iter-",
                        "validation results",
                        "test result",
                        "Mean Rank",
                        "Mean RR",
                        "Hit@",
                        "Custom time axis",
                        "#training triple",
                    )
                ):
                    print(f"[ATISE] {line_s}", flush=True)
            proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ATISE 训练失败，见: {log_path}")

        cand = sorted(data_dir.rglob("test_result*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        link_pred_metrics = _parse_metric_file(cand[0]) if cand else {}
        (out_dir / "atise_link_prediction_test.txt").write_text(
            json.dumps(link_pred_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 主实验口径：test split GT + 与 HippoRAG 一致的自适应 cap
    eval_cmd = [
        py,
        str(project_root / "scripts" / "real_comparison_atise_matching_eval.py"),
        "--dim",
        str(args.dim),
        "--ground_truth_source",
        "test",
        "--output",
        "reports/real_comparison_results/atise_matching_eval_testsplit.json",
    ]
    if args.cuda:
        eval_cmd.append("--cuda")
    print(f"[ATISE] 匹配评测: {' '.join(eval_cmd)}", flush=True)
    r = subprocess.run(eval_cmd, cwd=str(project_root))
    if r.returncode != 0:
        raise RuntimeError("ATISE 匹配评测失败")

    print("[ATISE] 全部完成。", flush=True)


if __name__ == "__main__":
    main()
