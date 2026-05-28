#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据规模实验（按六大行业并集）：在 LCC 内选取 **若干完整行业**，使
「政策数 + 企业数 + 所含行业节点数」相对 **LCC 内全六大行业并集** 的比例
分别最接近 1/10、1/5、1/2；且三档子集 **嵌套**（小 ⊂ 中 ⊂ 大）。

与 `data_scale_connected_subgraph.py`（BFS 截断 LCC）不同：此处行业来自
`industry_mapping_complete.json` 的 `major_industries`，图中 `belongsto` /
`targetsindustry` 的 industry 对象仅限这六类。

诱导子图规则（逻辑闭合）
------------------------
- 企业：LCC 内且 `belongsto` 目标行业 ∈ 选中集合
- 政策：LCC 内且 `targetsindustry` 目标行业 ∈ 选中集合
- 行业节点：选中集合（有实体支撑）
- 三元组：`supports` / `targetsindustry` / `belongsto` 保留两端均在上述集合内的行；
  `triples_policy_policy` 两端政策均在政策子集内

输出目录（默认）
----------------
`data_intermediate/data_scale_by_industry/nodes_0_10|nodes_0_20|nodes_0_50/`
各含与 BFS 子图同结构的 parquet + `scale_meta.json`（含 `orig_policy_id`），
可接 `data_scale_run_pipeline.py`。

评测建议：对匹配脚本使用 `--eval_query_scope subgraph_entities`，使查询条数
随子图内实体变化（仅保留查询端在子图内的测试查询）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import data_scale_connected_subgraph as dsc  # noqa: E402


def _policies_companies_by_major_lcc(
    df_p2e: pd.DataFrame,
    valid_policy_titles: Set[str],
    ent_names: Set[str],
    majors: List[str],
    lcc: Set[str],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    pe = df_p2e.copy()
    pe["_pred"] = pe["predicate"].astype(str).str.lower()
    pol_by: Dict[str, Set[str]] = {m: set() for m in majors}
    com_by: Dict[str, Set[str]] = {m: set() for m in majors}
    majors_set = set(majors)
    for pred, st, ot, subj, obj in zip(
        pe["_pred"].tolist(),
        pe["subject_type"].astype(str).tolist(),
        pe["object_type"].astype(str).tolist(),
        pe["subject"].astype(str).tolist(),
        pe["object"].astype(str).tolist(),
    ):
        pred = str(pred).lower()
        st, ot = str(st), str(ot)
        subj, obj = str(subj), str(obj)
        if st == "policy" and ot == "industry" and pred == "targetsindustry":
            if subj in valid_policy_titles and obj in majors_set:
                if dsc._node("policy", subj) in lcc:
                    pol_by[obj].add(subj)
        if st == "company" and ot == "industry" and pred == "belongsto":
            if subj in ent_names and obj in majors_set:
                if dsc._node("company", subj) in lcc:
                    com_by[obj].add(subj)
    return pol_by, com_by


def _metric_for_mask(
    majors: List[str],
    pol_by: Dict[str, Set[str]],
    com_by: Dict[str, Set[str]],
    mask: int,
) -> Tuple[int, Set[str], Set[str], List[str]]:
    sel = [majors[i] for i in range(len(majors)) if mask & (1 << i)]
    p: Set[str] = set()
    c: Set[str] = set()
    for m in sel:
        p |= pol_by[m]
        c |= com_by[m]
    return len(p) + len(c) + len(sel), p, c, sel


def _choose_nested_masks(
    majors: List[str],
    pol_by: Dict[str, Set[str]],
    com_by: Dict[str, Set[str]],
    targets: List[float],
) -> List[int]:
    """依次在上一档 mask 的超集上，选与 targets[k] 最接近的节点规模比。"""
    full_mask = (1 << len(majors)) - 1
    tot, _, _, _ = _metric_for_mask(majors, pol_by, com_by, full_mask)
    if tot <= 0:
        raise RuntimeError("LCC 内全行业节点规模为 0")

    def rel(m: int) -> float:
        return _metric_for_mask(majors, pol_by, com_by, m)[0] / tot

    prev = 0
    out: List[int] = []
    for t in targets:
        best_m, best_err = full_mask, 1e9
        for m in range(1, full_mask + 1):
            if (m & prev) != prev:
                continue
            err = abs(rel(m) - t)
            if err < best_err or (err == best_err and m < best_m):
                best_err = err
                best_m = m
        out.append(best_m)
        prev = best_m
    return out


def _count_supports(
    df_p2e: pd.DataFrame,
    pol_titles: Set[str],
    comp_names: Set[str],
) -> int:
    pe = df_p2e.copy()
    pe["_pred"] = pe["predicate"].astype(str).str.lower()
    pe["_st"] = pe["subject_type"].astype(str)
    pe["_ot"] = pe["object_type"].astype(str)
    m = (
        pe["_st"].eq("policy")
        & pe["_ot"].eq("company")
        & pe["_pred"].eq("supports")
        & pe["subject"].astype(str).isin(pol_titles)
        & pe["object"].astype(str).isin(comp_names)
    )
    return int(m.sum())


def _export_scale_subset(
    project_root: Path,
    out_subdir: Path,
    pol_titles: Set[str],
    comp_names: Set[str],
    ind_names: Set[str],
    df_pol: pd.DataFrame,
    df_ent: pd.DataFrame,
    df_p2e: pd.DataFrame,
    df_p2p: pd.DataFrame,
    ent_names: Set[str],
    meta: dict,
) -> None:
    sorted_policies = sorted(pol_titles)
    new_pid = {t: i for i, t in enumerate(sorted_policies)}
    df_pol_sub = df_pol[df_pol["title"].astype(str).isin(pol_titles)].copy()
    df_pol_sub["orig_policy_id"] = df_pol_sub["policy_id"].astype(int)
    df_pol_sub["policy_id"] = df_pol_sub["title"].astype(str).map(new_pid)

    df_ent_sub = df_ent[df_ent["name"].astype(str).isin(comp_names & ent_names)].copy()

    pe = df_p2e.copy()
    pe["_pred"] = pe["predicate"].astype(str).str.lower()
    pe["_subj"] = pe["subject"].astype(str)
    pe["_obj"] = pe["object"].astype(str)
    pe["_st"] = pe["subject_type"].astype(str)
    pe["_ot"] = pe["object_type"].astype(str)
    mask_ent = ~(pe["_st"].eq("entity") | pe["_ot"].eq("entity"))
    m_sup = (
        pe["_st"].eq("policy")
        & pe["_ot"].eq("company")
        & pe["_pred"].eq("supports")
        & pe["_subj"].isin(pol_titles)
        & pe["_obj"].isin(comp_names)
    )
    m_ti = (
        pe["_st"].eq("policy")
        & pe["_ot"].eq("industry")
        & pe["_pred"].eq("targetsindustry")
        & pe["_subj"].isin(pol_titles)
        & pe["_obj"].isin(ind_names)
    )
    m_bt = (
        pe["_st"].eq("company")
        & pe["_ot"].eq("industry")
        & pe["_pred"].eq("belongsto")
        & pe["_subj"].isin(comp_names)
        & pe["_obj"].isin(ind_names)
    )
    df_pe_sub = pe.loc[mask_ent & (m_sup | m_ti | m_bt), df_p2e.columns].copy()

    pp = df_p2p.copy()
    pp["_hn"] = pp["head_name"].astype(str)
    pp["_tn"] = pp["tail_name"].astype(str)
    ppm = pp["_hn"].isin(pol_titles) & pp["_tn"].isin(pol_titles)
    df_pp_sub = pp.loc[ppm, df_p2p.columns].copy()
    df_pp_sub["head_id"] = df_pp_sub["head_name"].astype(str).map(new_pid)
    df_pp_sub["tail_id"] = df_pp_sub["tail_name"].astype(str).map(new_pid)

    out_subdir.mkdir(parents=True, exist_ok=True)
    df_pol_sub.to_parquet(out_subdir / "policies_clean.parquet", index=False)
    df_ent_sub.to_parquet(out_subdir / "enterprises_filtered.parquet", index=False)
    df_pe_sub.to_parquet(out_subdir / "triples_policy_entity.parquet", index=False)
    df_pp_sub.to_parquet(out_subdir / "triples_policy_policy.parquet", index=False)

    meta_out = {
        **meta,
        "num_policies": len(pol_titles),
        "num_companies": len(comp_names & ent_names),
        "num_industries": len(ind_names),
        "num_p2e_edges": len(df_pe_sub),
        "num_p2p_edges": len(df_pp_sub),
        "policy_titles": sorted(pol_titles),
        "company_names": sorted(comp_names & ent_names),
        "industry_names": sorted(ind_names),
    }
    (out_subdir / "scale_meta.json").write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.5],
        help="相对 LCC 内「六行业并集」的节点规模目标比例（嵌套选取最接近者）",
    )
    ap.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=["nodes_0_10", "nodes_0_20", "nodes_0_50"],
        help="与 --targets 等长的输出子目录名",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="data_intermediate/data_scale_by_industry",
        help="输出根目录",
    )
    ap.add_argument(
        "--industry_map",
        type=str,
        default="industry_mapping_complete.json",
        help="含 major_industries 的 JSON",
    )
    args = ap.parse_args()
    if len(args.targets) != len(args.tags):
        ap.error("--targets 与 --tags 长度须一致")

    project_root = Path(__file__).resolve().parents[1]
    with open(project_root / args.industry_map, "r", encoding="utf-8") as f:
        ind_map = json.load(f)
    majors: List[str] = list(ind_map["major_industries"])
    if len(majors) != 6:
        print(f"[warn] major_industries 数量为 {len(majors)}，嵌套枚举按位宽={len(majors)}")

    df_pol = pd.read_parquet(project_root / "data_intermediate/policies_clean.parquet")
    df_ent = pd.read_parquet(project_root / "data_intermediate/enterprises_filtered.parquet")
    df_p2e = pd.read_parquet(project_root / "data_intermediate/triples_policy_entity.parquet")
    df_p2p = pd.read_parquet(project_root / "data_intermediate/triples_policy_policy.parquet")
    valid_policy_titles = set(str(x) for x in df_pol["title"].astype(str))
    ent_names = set(str(x) for x in df_ent["name"].astype(str))

    adj, nodes = dsc.build_untyped_adjacency(df_p2e, df_p2p, valid_policy_titles, ent_names)
    lcc = dsc.largest_connected_component(adj, nodes)
    pol_by, com_by = _policies_companies_by_major_lcc(
        df_p2e, valid_policy_titles, ent_names, majors, lcc
    )

    full_mask = (1 << len(majors)) - 1
    tot_metric, p_all, c_all, _ = _metric_for_mask(majors, pol_by, com_by, full_mask)
    num_pol_lcc = len([n for n in lcc if n.startswith("policy::")])
    num_com_lcc = len([n for n in lcc if n.startswith("company::")])
    print(
        f"[industry_scale] LCC 节点={len(lcc)} (policy≈{num_pol_lcc}, company≈{num_com_lcc}) | "
        f"六行业并集规模指标={tot_metric} (P={len(p_all)} C={len(c_all)} I={len(majors)})",
        flush=True,
    )

    masks = _choose_nested_masks(majors, pol_by, com_by, args.targets)
    out_root = project_root / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    for tag, target, mask in zip(args.tags, args.targets, masks, strict=True):
        _, pol_titles, comp_names, sel = _metric_for_mask(majors, pol_by, com_by, mask)
        ind_names = set(sel)
        n_sup = _count_supports(df_p2e, pol_titles, comp_names)
        achieved = (len(pol_titles) + len(comp_names) + len(ind_names)) / tot_metric if tot_metric else 0.0
        meta = {
            "sampling": "industry_union_nested",
            "target_fraction": float(target),
            "achieved_node_metric_fraction": round(achieved, 6),
            "denominator": "lcc_six_majors_union_policies_plus_companies_plus_industry_nodes",
            "lcc_size": len(lcc),
            "reference_node_metric_total": tot_metric,
            "selected_industries": sel,
            "supports_triples_in_subgraph": n_sup,
            "industry_mask_bits": mask,
        }
        sub_dir = out_root / tag
        _export_scale_subset(
            project_root,
            sub_dir,
            pol_titles,
            comp_names,
            ind_names,
            df_pol,
            df_ent,
            df_p2e,
            df_p2p,
            ent_names,
            meta,
        )
        print(
            f"[industry_scale] 写入 {sub_dir.name}: target={target:.2f} achieved≈{achieved:.3f} "
            f"P={len(pol_titles)} C={len(comp_names & ent_names)} I={len(ind_names)} "
            f"supports={n_sup} 行业={sel}",
            flush=True,
        )


if __name__ == "__main__":
    main()
