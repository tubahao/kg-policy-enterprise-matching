#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据企业名单清理三元组数据。

要求：
1. 读取指定的三元组 JSON 文件（默认：output/extracted_triples_cleaned.json）
2. 从 CSV 名单（默认：E:\\论文\\知识图谱\\数据\\企业数据\\参保人数小于等于10或记录少于2年的企业名单.csv）
   中读取企业名称
3. 删除所有 subject 或 object 中包含上述企业名称的三元组
4. 覆盖写回原文件（写入前自动备份）
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Set


def load_company_names(csv_path: Path, column_name: str = "企业名称") -> List[str]:
    """读取企业名单."""
    if not csv_path.exists():
        raise FileNotFoundError(f"企业名单不存在: {csv_path}")

    names: List[str] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("企业名单没有表头，无法解析")

        target_key = None
        for field in reader.fieldnames:
            if field.strip("\ufeff") == column_name:
                target_key = field
                break

        if not target_key:
            raise ValueError(
                f"企业名单中缺少列 '{column_name}'，实际列: {reader.fieldnames}"
            )

        for row in reader:
            raw = (row.get(target_key) or "").strip()
            if raw:
                names.append(raw)
    return names


def should_remove(subject: str, obj: str, company_names: Sequence[str]) -> bool:
    """判断 subject 或 object 是否包含任何一个企业名称."""
    # 检查 subject
    if isinstance(subject, str):
        subject = subject.strip()
        if subject:
            for name in company_names:
                if name and name in subject:
                    return True
    
    # 检查 object
    if isinstance(obj, str):
        obj = obj.strip()
        if obj:
            for name in company_names:
                if name and name in obj:
                    return True
    
    return False


def filter_triples(triples_path: Path, company_names: Sequence[str]) -> List[dict]:
    """过滤需要删除的三元组，返回保留列表."""
    if not triples_path.exists():
        raise FileNotFoundError(f"三元组文件不存在: {triples_path}")

    with open(triples_path, "r", encoding="utf-8") as f:
        triples = json.load(f)

    if not isinstance(triples, list):
        raise ValueError("三元组文件不是列表结构")

    filtered = []
    removed = 0
    for triple in triples:
        subject = triple.get("subject", "")
        obj = triple.get("object", "")
        if should_remove(subject, obj, company_names):
            removed += 1
            continue
        filtered.append(triple)

    print(f"总三元组: {len(triples):,}")
    print(f"删除三元组: {removed:,}")
    print(f"保留三元组: {len(filtered):,}")
    return filtered


def backup_file(src: Path) -> Path:
    """生成备份文件."""
    backup_path = src.with_name(f"{src.stem}_backup_before_company_filter{src.suffix}")
    shutil.copy2(src, backup_path)
    print(f"已创建备份: {backup_path}")
    return backup_path


def main():
    project_root = Path(__file__).resolve().parents[1]
    triples_path = project_root / "output" / "extracted_triples_cleaned.json"
    company_csv = Path(
        r"E:\论文\知识图谱\数据\企业数据\参保人数小于等于10或记录少于2年的企业名单.csv"
    )

    print("=" * 60)
    print("根据企业名单清理三元组")
    print("=" * 60)
    print(f"三元组文件: {triples_path}")
    print(f"企业名单: {company_csv}")

    company_names = load_company_names(company_csv)
    if not company_names:
        raise ValueError("企业名单为空，无法继续")
    print(f"名单企业数: {len(company_names):,}")

    filtered_triples = filter_triples(triples_path, company_names)

    # 备份并写回
    backup_file(triples_path)
    with open(triples_path, "w", encoding="utf-8") as f:
        json.dump(filtered_triples, f, ensure_ascii=False, indent=2)

    print("\n✅ 已根据企业名单完成清理并写回原文件")


if __name__ == "__main__":
    main()


