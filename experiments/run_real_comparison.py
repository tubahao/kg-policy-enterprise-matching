#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实对比实验总控脚本：
- 准备多框架数据
- 分模型独立运行（文本、OpenKE、KG-BERT、ATISE）
- 汇总结果与失败信息
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run(cmd, cwd: Path):
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, encoding="utf-8", errors="ignore")
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def main():
    project_root = Path(__file__).resolve().parents[1]
    py = project_root / "venv_graph" / "Scripts" / "python.exe"
    out_dir = project_root / "reports" / "real_comparison_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "name": "prepare_data",
            "cmd": [str(py), "scripts/real_comparison_prepare_data.py"],
        },
        {
            "name": "text_rag",
            "cmd": [str(py), "scripts/real_comparison_text_rag.py"],
        },
        {
            "name": "openke_transe",
            "cmd": [str(py), "scripts/real_comparison_openke_transe.py", "--train_times", "200", "--dim", "200"],
        },
        {
            "name": "kg_bert",
            "cmd": [
                str(py), "scripts/real_comparison_kgbert.py",
                "--bert_model", "bert-base-chinese",
                "--num_train_epochs", "1.0",
                "--train_batch_size", "16",
                "--eval_batch_size", "128",
            ],
        },
        {
            "name": "atise",
            "cmd": [
                str(py), "scripts/real_comparison_atise.py",
                "--max_epoch", "120",
                "--dim", "200",
                "--batch", "512",
                "--lr", "0.00003",
            ],
        },
        {
            "name": "export_to_report",
            "cmd": [str(py), "scripts/export_real_comparison_to_report.py"],
        },
    ]

    summary = {"success": [], "failed": []}
    for t in tasks:
        print(f"[RUN] {t['name']} ...")
        res = _run(t["cmd"], project_root)
        log_file = out_dir / f"{t['name']}.log"
        log_file.write_text(
            "[stdout]\n" + res["stdout"] + "\n\n[stderr]\n" + res["stderr"],
            encoding="utf-8",
        )
        if res["returncode"] == 0:
            summary["success"].append({"task": t["name"], "log": str(log_file)})
            print(f"[OK] {t['name']}")
        else:
            summary["failed"].append({"task": t["name"], "log": str(log_file), "returncode": res["returncode"]})
            print(f"[FAIL] {t['name']} (code={res['returncode']})")

    out_summary = out_dir / "real_comparison_run_summary.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

