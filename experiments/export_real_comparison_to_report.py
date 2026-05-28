#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将真实对比实验结果统一导出到项目根目录 report 目录。
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _safe_read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_metric_block(item: Dict) -> Dict:
    if not item:
        return {}
    if "metrics" in item:
        return item["metrics"]
    out = {}
    ep = (((item.get("enterprise_to_policy") or {}).get("average")) or {})
    pe = (((item.get("policy_to_enterprise") or {}).get("average")) or {})
    if ep:
        out["E->P"] = ep
    if pe:
        out["P->E"] = pe
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report_root",
        type=str,
        default="",
        help="不传则默认输出到项目外层目录的 report",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parents[1]
    src_dir = project_root / "reports" / "real_comparison_results"
    if not src_dir.exists():
        raise FileNotFoundError(f"结果目录不存在: {src_dir}")

    report_root = Path(args.report_root).resolve() if args.report_root else (repo_root / "report")
    report_root.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_dir = report_root / f"real_comparison_{run_tag}"
    dst_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = sorted(src_dir.glob("*.json")) + sorted(src_dir.glob("*.log")) + sorted(src_dir.glob("*.ckpt"))
    copied: List[str] = []
    for fp in files_to_copy:
        target = dst_dir / fp.name
        shutil.copy2(fp, target)
        copied.append(str(target))

    model_files = {
        "Text-Only-VectorRAG": dst_dir / "text_rag_results.json",
        "OpenKE-TransE": dst_dir / "openke_transe_results.json",
        "KG-BERT": dst_dir / "kgbert_results.json",
        "ATISE": dst_dir / "atise_results.json",
    }

    summary = {
        "run_tag": run_tag,
        "report_dir": str(dst_dir),
        "models": {},
        "module_replacement_plan": {
            "基础纯文本检索": {
                "models": ["Text-Only-VectorRAG"],
                "replace": "跳过图构建/GNN/GraphRAG，仅保留文本向量检索",
                "keep": "数据清洗与统一评估",
            },
            "三元组特征嵌入": {
                "models": ["OpenKE-TransE", "KG-BERT"],
                "replace": "替换表征学习与多模态融合",
                "keep": "三元组抽取与评估任务",
            },
            "时序动态图谱": {
                "models": ["ATISE"],
                "replace": "替换时间建模与衰减模块",
                "keep": "图谱关系与评估口径",
            },
        },
        "copied_files": copied,
    }

    for name, fp in model_files.items():
        data = _safe_read_json(fp)
        summary["models"][name] = {
            "file": str(fp),
            "exists": bool(data is not None),
            "metrics": _extract_metric_block(data or {}),
        }

    summary_path = dst_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# 对比实验结果汇总",
        "",
        f"- 运行批次: `{run_tag}`",
        f"- 输出目录: `{dst_dir}`",
        "",
        "## 模型结果文件",
    ]
    for n, m in summary["models"].items():
        flag = "OK" if m["exists"] else "MISSING"
        md_lines.append(f"- {n}: `{m['file']}` ({flag})")
    md_lines.extend(
        [
            "",
            "## 四类模块替换说明",
            "- 基础纯文本检索：替换图检索链路，仅保留文本向量匹配。",
            "- 三元组特征嵌入：使用 TransE / KG-BERT 替换表征融合模块。",
            "- 时序动态图谱：使用 ATISE 替换时间嵌入与衰减建模。",
            "- 端到端 GraphRAG：当前批次仅收录已完成真实运行的模型结果。",
            "",
            f"- 结构化汇总文件：`{summary_path}`",
        ]
    )
    md_path = dst_dir / "summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({"report_dir": str(dst_dir), "summary_json": str(summary_path), "summary_md": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

