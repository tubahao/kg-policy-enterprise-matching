#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实对比实验数据准备：
- OpenKE(TransE) 数据格式
- KG-BERT 数据格式
- ATISE 数据格式
- 统一查询与评估基准
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _safe_date(x: object) -> str:
    if x is None:
        return "2024-01-01"
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return "2024-01-01"
    try:
        # 尝试标准化 yyyy-mm-dd
        dt = datetime.fromisoformat(s.replace("/", "-"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        if len(s) >= 10 and s[4] in "-/" and s[7] in "-/":
            return s[:10].replace("/", "-")
        return "2024-01-01"


def _split_rows(rows: List[Tuple[str, str, str, str]], seed: int, train_ratio: float, valid_ratio: float):
    rnd = random.Random(seed)
    rows = list(rows)
    rnd.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = rows[:n_train]
    valid = rows[n_train : n_train + n_valid]
    test = rows[n_train + n_valid :]
    return train, valid, test


def _write_openke(
    out_dir: Path,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    train,
    valid,
    test,
    openke_entity_token: Dict[str, str],
    openke_relation_token: Dict[str, str],
):
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_id_file(path: Path, mp: Dict[str, int]):
        with path.open("w", encoding="utf-8") as f:
            f.write(f"{len(mp)}\n")
            for k, v in sorted(mp.items(), key=lambda x: x[1]):
                f.write(f"{k}\t{v}\n")

    def _write_triple2id(path: Path, triples):
        with path.open("w", encoding="utf-8") as f:
            f.write(f"{len(triples)}\n")
            for h, r, t, _ in triples:
                f.write(f"{entity2id[h]} {entity2id[t]} {relation2id[r]}\n")

    # OpenKE 对实体字符串更敏感，使用无空白 token 作为名称
    _write_id_file(
        out_dir / "entity2id.txt",
        {openke_entity_token[e]: i for e, i in entity2id.items()},
    )
    _write_id_file(
        out_dir / "relation2id.txt",
        {openke_relation_token[r]: i for r, i in relation2id.items()},
    )
    _write_triple2id(out_dir / "train2id.txt", train)
    _write_triple2id(out_dir / "valid2id.txt", valid)
    _write_triple2id(out_dir / "test2id.txt", test)

    # type_constrain
    head_sets = defaultdict(set)
    tail_sets = defaultdict(set)
    for h, r, t, _ in (train + valid + test):
        rid = relation2id[r]
        head_sets[rid].add(entity2id[h])
        tail_sets[rid].add(entity2id[t])
    with (out_dir / "type_constrain.txt").open("w", encoding="utf-8") as f:
        f.write(f"{len(relation2id)}\n")
        for rid in range(len(relation2id)):
            hs = sorted(head_sets[rid])
            ts = sorted(tail_sets[rid])
            f.write(f"{rid}\t{len(hs)}" + ("" if not hs else "\t" + "\t".join(map(str, hs))) + "\n")
            f.write(f"{rid}\t{len(ts)}" + ("" if not ts else "\t" + "\t".join(map(str, ts))) + "\n")


def _write_kgbert(
    out_dir: Path,
    train,
    valid,
    test,
    entity_text: Dict[str, str],
    relation_text: Dict[str, str],
    kgbert_entity_token: Dict[str, str],
    kgbert_relation_token: Dict[str, str],
):
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_tsv(path: Path, triples):
        with path.open("w", encoding="utf-8") as f:
            for h, r, t, _ in triples:
                f.write(f"{kgbert_entity_token[h]}\t{kgbert_relation_token[r]}\t{kgbert_entity_token[t]}\n")

    _write_tsv(out_dir / "train.tsv", train)
    _write_tsv(out_dir / "dev.tsv", valid)
    _write_tsv(out_dir / "test.tsv", test)

    entities = sorted(kgbert_entity_token.values(), key=lambda x: int(x[1:]))
    relations = sorted(kgbert_relation_token.values(), key=lambda x: int(x[1:]))
    (out_dir / "entities.txt").write_text("\n".join(entities) + "\n", encoding="utf-8")
    (out_dir / "relations.txt").write_text("\n".join(relations) + "\n", encoding="utf-8")

    with (out_dir / "entity2text.txt").open("w", encoding="utf-8") as f:
        for raw_e, tok_e in sorted(kgbert_entity_token.items(), key=lambda x: int(x[1][1:])):
            # KG-BERT 官方脚本在 Windows 下按系统编码读取文件，这里保持 ASCII 文本避免解码失败
            txt = entity_text.get(raw_e, "")
            txt_ascii = "".join(ch if ord(ch) < 128 else " " for ch in txt)
            txt_ascii = " ".join(txt_ascii.split())
            if not txt_ascii:
                txt_ascii = f"entity {tok_e}"
            f.write(f"{tok_e}\t{txt_ascii}\n")
    with (out_dir / "relation2text.txt").open("w", encoding="utf-8") as f:
        for raw_r, tok_r in sorted(kgbert_relation_token.items(), key=lambda x: int(x[1][1:])):
            rtxt = relation_text.get(raw_r, raw_r)
            rtxt_ascii = "".join(ch if ord(ch) < 128 else " " for ch in rtxt)
            rtxt_ascii = " ".join(rtxt_ascii.split()) or f"relation {tok_r}"
            f.write(f"{tok_r}\t{rtxt_ascii}\n")


def _write_atise(
    out_dir: Path,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    train,
    valid,
    test,
    atise_entity_token: Dict[str, str],
    atise_relation_token: Dict[str, str],
):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "entity2id.txt").open("w", encoding="utf-8") as f:
        for e, i in sorted(entity2id.items(), key=lambda x: x[1]):
            f.write(f"{atise_entity_token[e]}\t{i}\n")
    with (out_dir / "relation2id.txt").open("w", encoding="utf-8") as f:
        for r, i in sorted(relation2id.items(), key=lambda x: x[1]):
            f.write(f"{atise_relation_token[r]}\t{i}\n")

    def _write_quad(path: Path, quads):
        with path.open("w", encoding="utf-8") as f:
            for h, r, t, d in quads:
                f.write(f"{atise_entity_token[h]}\t{atise_relation_token[r]}\t{atise_entity_token[t]}\t{d}\n")

    _write_quad(out_dir / "train.txt", train)
    _write_quad(out_dir / "valid.txt", valid)
    _write_quad(out_dir / "test.txt", test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--max_triples", type=int, default=0, help="<=0 表示全部")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_root = project_root / "reports" / "real_comparison_data"
    out_root.mkdir(parents=True, exist_ok=True)

    triples_path = project_root / "data_intermediate" / "triples_policy_entity.parquet"
    pol_path = project_root / "data_intermediate" / "policies_clean.parquet"
    ent_path = project_root / "data_intermediate" / "enterprises_filtered.parquet"
    ind_map_path = project_root / "industry_mapping_complete.json"

    df_t = pd.read_parquet(triples_path)
    df_p = pd.read_parquet(pol_path)
    df_e = pd.read_parquet(ent_path)
    ind_map = json.loads(ind_map_path.read_text(encoding="utf-8")) if ind_map_path.exists() else {}

    policy_title_to_date = {
        str(r["title"]): _safe_date(r.get("publish_date", None))
        for _, r in df_p.iterrows()
    }
    policy_title_to_content = {
        str(r["title"]): str(r.get("content", ""))
        for _, r in df_p.iterrows()
    }
    enterprise_name_to_text = {
        str(r["name"]): str(r.get("text_with_industry", ""))
        for _, r in df_e.iterrows()
    }

    majors = set(ind_map.get("major_industries", []))
    fine_map = ind_map.get("fine_industry_mapping", {})

    # 选取核心可比关系
    allow_preds = {"supports", "targetsIndustry", "belongsTo", "transmitsTo"}
    rows: List[Tuple[str, str, str, str]] = []
    for _, r in df_t.iterrows():
        h = str(r["subject"])
        p = str(r["predicate"])
        t = str(r["object"])
        if p not in allow_preds:
            continue
        date = "2024-01-01"
        if h in policy_title_to_date:
            date = policy_title_to_date[h]
        elif t in policy_title_to_date:
            date = policy_title_to_date[t]
        rows.append((h, p, t, date))

    # 可选截断（调试用）
    if args.max_triples and args.max_triples > 0:
        rows = rows[: args.max_triples]

    # 构建实体/关系字典
    entities = sorted({h for h, _, _, _ in rows} | {t for _, _, t, _ in rows})
    relations = sorted({p for _, p, _, _ in rows})
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    openke_entity_token = {e: f"E{i}" for e, i in entity2id.items()}
    openke_relation_token = {r: f"R{i}" for r, i in relation2id.items()}
    atise_entity_token = dict(openke_entity_token)
    atise_relation_token = dict(openke_relation_token)
    kgbert_entity_token = dict(openke_entity_token)
    kgbert_relation_token = dict(openke_relation_token)

    # 文本描述映射
    relation_text = {
        "supports": "policy supports company",
        "targetsIndustry": "policy targets industry",
        "belongsTo": "company belongs to industry",
        "transmitsTo": "policy transmits to policy",
    }
    entity_text: Dict[str, str] = {}
    for e in entities:
        if e in policy_title_to_content:
            c = policy_title_to_content.get(e, "")
            entity_text[e] = f"{e}。{c[:500]}"
        elif e in enterprise_name_to_text:
            entity_text[e] = enterprise_name_to_text[e][:500]
        elif e in majors:
            entity_text[e] = f"行业大类：{e}"
        elif e in fine_map:
            ms = ",".join(fine_map[e].get("majors", []))
            entity_text[e] = f"细分行业：{e}；对应大类：{ms}"
        else:
            entity_text[e] = e

    train, valid, test = _split_rows(rows, seed=args.seed, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)

    _write_openke(
        out_root / "openke_policykg",
        entity2id,
        relation2id,
        train,
        valid,
        test,
        openke_entity_token,
        openke_relation_token,
    )
    _write_kgbert(
        out_root / "kgbert_policykg",
        train,
        valid,
        test,
        entity_text,
        relation_text,
        kgbert_entity_token,
        kgbert_relation_token,
    )
    _write_atise(
        out_root / "atise_policykg",
        entity2id,
        relation2id,
        train,
        valid,
        test,
        atise_entity_token,
        atise_relation_token,
    )

    meta = {
        "n_triples_total": len(rows),
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "n_entities": len(entity2id),
        "n_relations": len(relation2id),
        "relations": relations,
    }
    (out_root / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "entity_token_map.json").write_text(
        json.dumps(
            {"openke": openke_entity_token, "atise": atise_entity_token, "kgbert": kgbert_entity_token},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_root / "relation_token_map.json").write_text(
        json.dumps(
            {"openke": openke_relation_token, "atise": atise_relation_token, "kgbert": kgbert_relation_token},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"数据已输出到: {out_root}")


if __name__ == "__main__":
    main()

