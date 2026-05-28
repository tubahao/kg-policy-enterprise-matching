#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据规模实验：从政策-企业异构图中抽取 **连通子图**（非孤立节点采样）。

做法概述
--------
1. 用 policy / company / industry 三类节点、supports 等与 build_graph 一致的边类型，构造 **无向邻接图**。
2. 取 **最大连通分量 (LCC)**，避免多岛导致子图语义破碎。
3. 在 LCC 内从随机（可复现）政策节点出发 **BFS**，按发现顺序扩张，直到节点数 ≥ ceil(fraction * |LCC|)，
   保证导出集合在原始图上 **诱导子图仍连通**（包含 BFS 树，且保留端点均在集合内的全部边）。
4. 对选中集合重编号 policy_id / company / industry，写出子集 parquet + 元数据；特征按 **原 policy_id / 企业行号 / industry_index** 对齐切片。

输出目录（默认）
----------------
`data_intermediate/data_scale_subgraphs/{tag}/`
  - policies_clean.parquet
  - enterprises_filtered.parquet
  - triples_policy_entity.parquet
  - triples_policy_policy.parquet
  - scale_meta.json

后续：对该目录运行 `data_scale_run_pipeline.py` 构建图、训练 GAT。
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def _norm_pred(s: str) -> str:
    return str(s).strip().lower()


def _node(t: str, name: str) -> str:
    return f"{t}::{name}"


def _parse(n: str) -> Tuple[str, str]:
    t, name = n.split("::", 1)
    return t, name


def build_untyped_adjacency(
    df_p2e: pd.DataFrame,
    df_p2p: pd.DataFrame,
    valid_policy_titles: Set[str],
    valid_company_names: Set[str],
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """无向邻接表；节点键为 policy::title / company::name / industry::name。

    仅保留在 policies_clean / enterprises_filtered 中存在的政策与企业，
    避免三元组里「无特征、无 policy_id」的孤立政策标题进入子图采样（否则 build_graph 会分配额外节点导致 ID 越界）。
    """
    adj: Dict[str, Set[str]] = {}

    def add_undirected(a: str, b: str) -> None:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    for row in df_p2e.itertuples(index=False):
        pred = _norm_pred(getattr(row, "predicate"))
        subj, obj = str(row.subject), str(row.object)
        st, ot = str(row.subject_type), str(row.object_type)
        if st == "entity" or ot == "entity":
            continue
        if st == "policy" and ot == "company" and pred == "supports":
            if subj not in valid_policy_titles or obj not in valid_company_names:
                continue
            add_undirected(_node("policy", subj), _node("company", obj))
        elif st == "policy" and ot == "industry" and pred == "targetsindustry":
            if subj not in valid_policy_titles:
                continue
            add_undirected(_node("policy", subj), _node("industry", obj))
        elif st == "company" and ot == "industry" and pred == "belongsto":
            if subj not in valid_company_names:
                continue
            add_undirected(_node("company", subj), _node("industry", obj))

    for row in df_p2p.itertuples(index=False):
        hn, tn = str(row.head_name), str(row.tail_name)
        if hn not in valid_policy_titles or tn not in valid_policy_titles:
            continue
        add_undirected(_node("policy", hn), _node("policy", tn))

    nodes = set(adj.keys())
    for n in list(nodes):
        for m in adj.get(n, ()):
            adj.setdefault(m, set()).add(n)

    return adj, nodes


def largest_connected_component(adj: Dict[str, Set[str]], nodes: Set[str]) -> Set[str]:
    unseen = set(nodes)
    best: Set[str] = set()
    while unseen:
        start = next(iter(unseen))
        comp = set()
        dq = deque([start])
        while dq:
            u = dq.popleft()
            if u in comp:
                continue
            comp.add(u)
            unseen.discard(u)
            for v in adj.get(u, ()):
                if v not in comp:
                    dq.append(v)
        if len(comp) > len(best):
            best = comp
    return best


def bfs_connected_subset(
    adj: Dict[str, Set[str]],
    lcc: Set[str],
    target_size: int,
    seed: int,
    seed_mode: str = "high_degree_policy",
) -> Set[str]:
    """在 LCC 内 BFS，**严格**最多 ``target_size`` 个节点（连通、诱导子图端点均在集合内）。

    邻居扩展顺序：同层内 **政策节点优先于企业与行业**，在固定节点预算下尽量多纳入政策，减轻「单政策+企业海」偏置。
    """
    policies = [n for n in lcc if n.startswith("policy::")]
    if not policies:
        raise RuntimeError("LCC 内无政策节点，无法 BFS 种子")
    rng = random.Random(seed)
    if seed_mode == "random_policy":
        start = policies[rng.randrange(len(policies))]
    else:
        degs = [(n, len(adj.get(n, set()) & lcc)) for n in policies]
        mx = max(d for _, d in degs)
        top = [n for n, d in degs if d == mx]
        start = top[rng.randrange(len(top))]
    target_size = min(target_size, len(lcc))
    visited: Set[str] = set()
    dq = deque([start])

    def neigh_order(nodes: List[str]) -> List[str]:
        pol = [x for x in nodes if x.startswith("policy::")]
        com = [x for x in nodes if x.startswith("company::")]
        ind = [x for x in nodes if x.startswith("industry::")]
        rng.shuffle(pol)
        rng.shuffle(com)
        rng.shuffle(ind)
        return pol + com + ind

    while dq and len(visited) < target_size:
        u = dq.popleft()
        if u in visited:
            continue
        visited.add(u)
        raw = [v for v in adj.get(u, set()) & lcc if v not in visited]
        for v in neigh_order(raw):
            if v not in visited:
                dq.append(v)

    if len(visited) < target_size:
        frontier = list(visited)
        while frontier and len(visited) < target_size:
            u = frontier.pop(0)
            for v in neigh_order([w for w in adj.get(u, set()) & lcc if w not in visited]):
                if v not in visited:
                    visited.add(v)
                    frontier.append(v)
                    if len(visited) >= target_size:
                        break

    return visited


def split_typed_nodes(subset: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    ps, cs, ins = set(), set(), set()
    for n in subset:
        t, name = _parse(n)
        if t == "policy":
            ps.add(name)
        elif t == "company":
            cs.add(name)
        elif t == "industry":
            ins.add(name)
    return ps, cs, ins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.5],
        help="相对最大连通分量节点数的比例：如 0.1=十分之一, 0.2=五分之一, 0.5=二分之一",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out_root",
        type=str,
        default="data_intermediate/data_scale_subgraphs",
        help="各比例子图数据根目录",
    )
    ap.add_argument("--min_nodes", type=int, default=20, help="子图至少节点数下限（防止过小）")
    ap.add_argument(
        "--seed_mode",
        type=str,
        choices=["high_degree_policy", "random_policy"],
        default="high_degree_policy",
        help="BFS 起点：高度政策(默认) 或 随机政策",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    df_pol = pd.read_parquet(project_root / "data_intermediate/policies_clean.parquet")
    df_ent = pd.read_parquet(project_root / "data_intermediate/enterprises_filtered.parquet")
    df_p2e = pd.read_parquet(project_root / "data_intermediate/triples_policy_entity.parquet")
    df_p2p = pd.read_parquet(project_root / "data_intermediate/triples_policy_policy.parquet")

    valid_policy_titles = set(str(x) for x in df_pol["title"].astype(str))
    ent_names = set(str(x) for x in df_ent["name"].astype(str))

    adj, nodes = build_untyped_adjacency(df_p2e, df_p2p, valid_policy_titles, ent_names)
    lcc = largest_connected_component(adj, nodes)
    num_policies_lcc = len([n for n in lcc if n.startswith("policy::")])
    print(
        f"[data_scale] 全图节点数(有边): {len(nodes)} | LCC 节点数: {len(lcc)} | LCC 内政策数: {num_policies_lcc}",
        flush=True,
    )

    out_root = project_root / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    for frac in args.fractions:
        tag = f"frac_{frac:.4f}".rstrip("0").rstrip(".").replace(".", "_")
        target = max(args.min_nodes, int(math.ceil(frac * len(lcc))))
        subset_nodes = bfs_connected_subset(
            adj,
            lcc,
            target,
            seed=args.seed + int(frac * 10000),
            seed_mode=args.seed_mode,
        )
        pol_titles, comp_names, ind_names = split_typed_nodes(subset_nodes)

        # --- 重编号 policy ---
        sorted_policies = sorted(pol_titles)
        new_pid = {t: i for i, t in enumerate(sorted_policies)}
        df_pol_sub = df_pol[df_pol["title"].astype(str).isin(pol_titles)].copy()
        df_pol_sub["orig_policy_id"] = df_pol_sub["policy_id"].astype(int)
        df_pol_sub["policy_id"] = df_pol_sub["title"].astype(str).map(new_pid)

        # --- 企业子集（仅保留图中的公司）---
        df_ent_sub = df_ent[df_ent["name"].astype(str).isin(comp_names & ent_names)].copy()

        # --- triples_policy_entity：两端都在子图内（按名称）---
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

        # --- triples_policy_policy：两端政策均在子集 ---
        pp = df_p2p.copy()
        pp["_hn"] = pp["head_name"].astype(str)
        pp["_tn"] = pp["tail_name"].astype(str)
        ppm = pp["_hn"].isin(pol_titles) & pp["_tn"].isin(pol_titles)
        df_pp_sub = pp.loc[ppm, df_p2p.columns].copy()
        df_pp_sub["head_id"] = df_pp_sub["head_name"].astype(str).map(new_pid)
        df_pp_sub["tail_id"] = df_pp_sub["tail_name"].astype(str).map(new_pid)

        sub_dir = out_root / tag
        sub_dir.mkdir(parents=True, exist_ok=True)
        df_pol_sub.to_parquet(sub_dir / "policies_clean.parquet", index=False)
        df_ent_sub.to_parquet(sub_dir / "enterprises_filtered.parquet", index=False)
        df_pe_sub.to_parquet(sub_dir / "triples_policy_entity.parquet", index=False)
        df_pp_sub.to_parquet(sub_dir / "triples_policy_policy.parquet", index=False)

        meta = {
            "fraction_of_lcc": frac,
            "lcc_size": len(lcc),
            "lcc_policy_count": num_policies_lcc,
            "target_nodes": target,
            "selected_nodes": len(subset_nodes),
            "num_policies": len(pol_titles),
            "num_companies": len(comp_names & ent_names),
            "num_industries": len(ind_names),
            "num_p2e_edges": len(df_pe_sub),
            "num_p2p_edges": len(df_pp_sub),
            "seed": args.seed,
            "policy_titles": sorted(pol_titles),
            "company_names": sorted(comp_names & ent_names),
            "industry_names": sorted(ind_names),
        }
        (sub_dir / "scale_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[data_scale] 写入 {sub_dir.name}: nodes={len(subset_nodes)} "
            f"P={len(pol_titles)} C={len(comp_names & ent_names)} I={len(ind_names)} "
            f"p2e={len(df_pe_sub)} p2p={len(df_pp_sub)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
