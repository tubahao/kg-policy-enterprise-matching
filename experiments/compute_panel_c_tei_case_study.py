#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel C（TEI 案例）：与插图 Panel B 使用同一 policy_id 与 PE_HP，计算
行业条件覆盖率、跳数深度、PPR 能量深度与 TEI；写 JSON 并更新 reports/figure_case_study_ep_pe.md。

默认政策 535：图上有六大类 targetsIndustry（行业条件分母 |E_rel|），传导评测中 TEI_ind 与 C_ind 表现较好。
policy 46 在图上无六大类 targetsIndustry，行业口径回退全图，不宜作「行业覆盖率」插图。

用法：
  python scripts/compute_panel_c_tei_case_study.py
  python scripts/compute_panel_c_tei_case_study.py --policy-id 652
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from matching.torch_metadata_fix import apply_torch_metadata_fix  # noqa: E402

apply_torch_metadata_fix()

from evaluation.industry_conditioned_coverage import (  # noqa: E402
    build_global_graph_structures_from_graph,
    build_industry_nid_to_name,
    build_major_company_sets,
    bfs_shortest_dist,
    load_major_industries,
    personalized_pagerank_vector,
    policy_targets_major_industries,
    union_companies_for_industries,
)
from matching.bidirectional_matching import BidirectionalMatcher  # noqa: E402
from matching.ensure_joint_policy_embeddings import ensure_joint_policy_embeddings  # noqa: E402
from matching.policy_embedding_defaults import JOINT_EMB, JOINT_IDX  # noqa: E402

# 与 scripts/export_figure_case_study_samples.py 中 Panel B 保持一致
DEFAULT_CASE_STUDY_POLICY_ID = 535
DEFAULT_CASE_STUDY_POLICY_TITLE_FALLBACK = (
    "关于印发广西整市推进产业工人队伍建设支持企业用工行动方案的通知"
)

PE_HP = {
    "top_k": -1,
    "candidate_k": 1000,
    "k_hop": 2,
    "score_threshold": None,
    "adaptive_quantile": 0.58,
    "relative_drop_threshold": 0.18,
    "max_output_cap": 150,
    "direct_support_boost": 0.3,
}

PANEL_C_FIGURE_BASENAME = "panel_c_tei_case_study.png"

PPR_ALPHA = 0.15
PPR_MAX_ITER = 40
TEI_ALPHA = 0.4
TEI_BETA = 0.3
TEI_GAMMA = 0.3

PANEL_C_MARKERS = (
    "<!-- PANEL_C_TEI_START -->",
    "<!-- PANEL_C_TEI_END -->",
)


def _company_id_to_name(matcher: BidirectionalMatcher) -> Dict[int, str]:
    nm = getattr(matcher.enterprise_retriever, "node_maps", {}) or {}
    cmap = nm.get("company", {}) or {}
    return {int(v): str(k) for k, v in cmap.items()}


def build_panel_c_payload(matcher: BidirectionalMatcher, policy_id: int) -> Dict[str, Any]:
    er = matcher.enterprise_retriever
    g = er.graph
    node_maps = er.node_maps or {}

    majors_list = load_major_industries(PROJECT_ROOT)
    majors_set = set(majors_list)
    industry_nid_to_name = build_industry_nid_to_name(node_maps)
    major_to_companies = build_major_company_sets(g, majors_set, industry_nid_to_name)
    company_count = g.number_of_nodes("company")

    policy_node = er._resolve_policy_node_id(policy_id)
    if policy_node is None:
        raise RuntimeError(f"无法解析 policy_id={policy_id} 的图节点")

    ip = policy_targets_major_industries(g, int(policy_node), majors_set, industry_nid_to_name)
    erel = union_companies_for_industries(ip, major_to_companies)

    results = matcher.retrieve_enterprises_by_policy(
        policy_id,
        top_k=PE_HP["top_k"],
        k_hop=int(PE_HP["k_hop"]),
        candidate_k=PE_HP["candidate_k"],
        score_threshold=PE_HP["score_threshold"],
        adaptive_quantile=PE_HP["adaptive_quantile"],
        relative_drop_threshold=PE_HP["relative_drop_threshold"],
        max_output_cap=PE_HP["max_output_cap"],
        direct_support_boost=PE_HP["direct_support_boost"],
    )
    covered: Set[int] = {int(cid) for cid, _ in results}
    direct_set: Set[int] = {
        int(x) for x in (er.policy_direct_support_companies.get(int(policy_node), set()) or set())
    }
    covered_union = covered | direct_set

    coverage_global = (len(covered_union) / company_count) if company_count > 0 else 0.0
    fallback = len(ip) == 0 or len(erel) == 0
    if fallback:
        denom = company_count
        numer = len(covered_union)
        coverage_industry = coverage_global
        mode = "fallback_full_graph" if len(ip) == 0 else "fallback_erel_empty"
    else:
        denom = len(erel)
        numer = len(covered_union & erel)
        coverage_industry = numer / max(denom, 1)
        mode = "industry_conditioned"

    if not fallback:
        output_precision_in_erel = len(covered & erel) / max(len(covered), 1)
    else:
        output_precision_in_erel = float("nan")

    offsets, adjacency, pt = build_global_graph_structures_from_graph(g)
    policy_off = offsets["policy"]
    company_off = offsets["company"]

    src_gid = policy_off + int(policy_node)
    dist = bfs_shortest_dist(adjacency, src_gid)
    covered_gids = [company_off + cid for cid in covered_union if 0 <= cid < company_count]
    hop_vals = [int(dist[gidx]) for gidx in covered_gids if dist[gidx] >= 0]
    depth_hops = float(np.mean(hop_vals)) if hop_vals else 0.0

    ppr = personalized_pagerank_vector(pt, src_gid, alpha=PPR_ALPHA, max_iter=PPR_MAX_ITER)
    if covered_gids:
        ppr_vals = np.array([float(ppr[gidx]) for gidx in covered_gids], dtype=np.float64)
        depth_energy = float(1.0 - np.mean(ppr_vals))
    else:
        depth_energy = 1.0

    inv_depth = (1.0 / depth_hops) if depth_hops > 0 else 0.0
    tei_global = TEI_ALPHA * coverage_global + TEI_BETA * inv_depth + TEI_GAMMA * depth_energy
    tei_industry = TEI_ALPHA * coverage_industry + TEI_BETA * inv_depth + TEI_GAMMA * depth_energy

    hop_counter = Counter(hop_vals)
    hop_distribution = {str(k): int(v) for k, v in sorted(hop_counter.items())}

    cid_to_name = _company_id_to_name(matcher)
    company_nodes: List[Dict[str, Any]] = []
    for cid in sorted(covered_union):
        if cid < 0 or cid >= company_count:
            continue
        gidx = company_off + cid
        h = int(dist[gidx]) if dist[gidx] >= 0 else -1
        pv = float(ppr[gidx]) if gidx < len(ppr) else 0.0
        company_nodes.append(
            {
                "company_graph_node_id": int(cid),
                "name": cid_to_name.get(int(cid), ""),
                "hop": h,
                "ppr_mass": pv,
                "in_retrieval": bool(cid in covered),
                "direct_support": bool(cid in direct_set),
            }
        )

    policies_df = pd.read_parquet(PROJECT_ROOT / "data_intermediate" / "policies_clean.parquet")
    title_row = policies_df[policies_df["policy_id"].astype(int) == int(policy_id)]
    policy_title = (
        str(title_row.iloc[0]["title"])
        if len(title_row)
        else DEFAULT_CASE_STUDY_POLICY_TITLE_FALLBACK
    )

    return {
        "panel": "C",
        "metric_family_zh": "传导效能指数 TEI（与 evaluation/industry_conditioned_coverage 公式一致）",
        "policy_id": int(policy_id),
        "policy_title": policy_title,
        "policy_node_id": int(policy_node),
        "matcher": {
            "policy_text_mode": "joint",
            "gat_artifact_tag": "a2_joint",
        },
        "hyperparameters_pe": PE_HP,
        "ppr_alpha": PPR_ALPHA,
        "ppr_max_iter": PPR_MAX_ITER,
        "tei_weights": {"alpha": TEI_ALPHA, "beta": TEI_BETA, "gamma": TEI_GAMMA},
        "targets_industry_majors": sorted(ip),
        "n_targets_majors": len(ip),
        "n_erel_union": len(erel),
        "n_covered_retrieval": len(covered),
        "n_direct_support": len(direct_set),
        "n_covered_union": len(covered_union),
        "numerator_intersection_erel": int(numer),
        "denominator_coverage": int(denom),
        "coverage_industry": float(coverage_industry),
        "coverage_global": float(coverage_global),
        "coverage_industry_pct": round(100.0 * coverage_industry, 2),
        "coverage_global_pct": round(100.0 * coverage_global, 2),
        "fallback_full_graph": bool(fallback),
        "mode": mode,
        "output_precision_in_erel": float(output_precision_in_erel)
        if output_precision_in_erel == output_precision_in_erel
        else None,
        "depth_hops_mean": float(depth_hops),
        "depth_energy": float(depth_energy),
        "inv_depth": float(inv_depth),
        "tei_global": float(tei_global),
        "tei_industry": float(tei_industry),
        "hop_distribution": hop_distribution,
        "company_count_graph": int(company_count),
        "company_nodes": company_nodes,
        "figure_output_basename": PANEL_C_FIGURE_BASENAME,
    }


def render_panel_c_markdown(data: Dict[str, Any]) -> str:
    pid = data["policy_id"]
    kh = int((data.get("hyperparameters_pe") or {}).get("k_hop", 2))
    cov_i = data["coverage_industry_pct"]
    cov_g = data["coverage_global_pct"]
    hops = data["depth_hops_mean"]
    de = data["depth_energy"]
    tei_g = data["tei_global"]
    tei_i = data["tei_industry"]
    hop_dist = data["hop_distribution"]
    targets = ", ".join(data["targets_industry_majors"]) or "（无六大类 targetsIndustry，已回退全图分母）"
    mode_note = (
        "本政策在图上具备六大类 **targetsIndustry**，覆盖率采用 **行业条件口径**（分母 \\(|E_{\\mathrm{rel}}|\\)）。"
        if not data.get("fallback_full_graph")
        else "**注意**：当前政策无行业锚或 \\(|E_{\\mathrm{rel}}|\\) 为空，行业覆盖率已回退为全图口径。"
    )
    fig_name = data.get("figure_output_basename", PANEL_C_FIGURE_BASENAME)
    lines = [
        "## Panel C — TEI（宏观传播效果）",
        "",
        f"承接 Panel B：在 **同一检索设置**（`a2_joint` GAT、`k_hop={kh}`、与 Panel B 相同的 adaptive 截断与 `direct_support_boost`）下，对 **policy_id={pid}** 计算传导效能相关量；",
        mode_note,
        "定义与 `reports/传导效能评估.md` / `evaluation/industry_conditioned_coverage.py` 一致：",
        "**Coverage** 取行业条件覆盖率 \\(C_{\\mathrm{ind}}\\)（分母 \\(|E_{\\mathrm{rel}}|\\)，分子 \\(|C(p)\\cap E_{\\mathrm{rel}}|\\)，\\(C=R\\cup D\\)）；",
        "**Hop** 为 \\(C(p)\\) 上相对政策节点的平均最短路跳数；**Energy** 为 \\(1-\\mathrm{mean}_{e\\in C}\\pi(e)\\)（PPR \\(\\pi\\)，\\(\\alpha=0.15\\)）。",
        "",
        f"- **政策目标行业（六大类，图）**：{targets}",
        f"- **\\(|E_{{\\mathrm{{rel}}}}|\\)**：{data['n_erel_union']}；**\\(|R|\\)**：{data['n_covered_retrieval']}；**\\(|D|\\)**：{data['n_direct_support']}；**\\(|C|=|R\\cup D|\\)**：{data['n_covered_union']}",
        f"- **Coverage（行业）**：**{cov_i}%**（全局口径约 {cov_g}%）",
        f"- **Hop depth（平均）**：**{hops:.4f}**",
        f"- **Energy depth（PPR）**：**{de:.6f}**（越大表示覆盖集合上平均 PPR 质量越低、传播路径更长/更分散）",
        f"- **TEI**：\\(\\mathrm{{TEI}}_{{\\mathrm{{ind}}}}\\)=**{tei_i:.6f}**，\\(\\mathrm{{TEI}}_{{\\mathrm{{glob}}}}\\)=**{tei_g:.6f}**（权重 \\(\\alpha,\\beta,\\gamma\\)=0.4/0.3/0.3）",
        "",
        "### 跳数分布（\\(C(p)\\) 内最短路径 hop，频数）",
        "",
        "| hop | count |",
        "|-----|-------|",
    ]
    for hk, cnt in sorted(hop_dist.items(), key=lambda x: int(x[0])):
        lines.append(f"| {hk} | {cnt} |")
    lines.extend(
        [
            "",
            "作图时可用上表作 **同心圆层级**（按 hop 分环），节点 **颜色** 映射 `ppr_mass`；**环形进度条** 映射行业覆盖率百分比。",
            "",
            f"结构化结果：`reports/figure_case_study_panel_c.json`；图：`python scripts/visualize_panel_c_tei.py` → `reports/figures/{fig_name}`。",
            "",
        ]
    )
    return "\n".join(lines)


def patch_md(md_path: Path, block: str) -> None:
    text = md_path.read_text(encoding="utf-8")
    wrapped = f"{PANEL_C_MARKERS[0]}\n{block}\n{PANEL_C_MARKERS[1]}"
    start_m, end_m = PANEL_C_MARKERS
    if start_m in text and end_m in text:
        i = text.index(start_m)
        j = text.index(end_m) + len(end_m)
        text_new = text[:i] + wrapped + text[j:]
    else:
        text_new = text.rstrip() + "\n\n" + wrapped + "\n"
    md_path.write_text(text_new, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Panel C TEI 案例（与 Panel B 同政策）")
    parser.add_argument(
        "--policy-id",
        type=int,
        default=DEFAULT_CASE_STUDY_POLICY_ID,
        help=f"policies_clean.policy_id，默认 {DEFAULT_CASE_STUDY_POLICY_ID}",
    )
    args = parser.parse_args()

    ensure_joint_policy_embeddings(PROJECT_ROOT)
    matcher = BidirectionalMatcher(
        PROJECT_ROOT,
        policy_emb_path=JOINT_EMB,
        policy_index_path=JOINT_IDX,
        gat_artifact_tag="a2_joint",
        policy_importance_parquet=None,
        ignore_env_importance_override=True,
    )
    data = build_panel_c_payload(matcher, int(args.policy_id))

    out_json = PROJECT_ROOT / "reports" / "figure_case_study_panel_c.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写 {out_json}")

    md_block = render_panel_c_markdown(data)
    md_path = PROJECT_ROOT / "reports" / "figure_case_study_ep_pe.md"
    patch_md(md_path, md_block)
    print(f"已更新 {md_path}")


if __name__ == "__main__":
    main()
