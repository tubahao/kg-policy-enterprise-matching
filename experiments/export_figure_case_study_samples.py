#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从主实验双向匹配中导出论文插图用 E→P / P→E 样例（UTF-8 JSON + 简短 MD）。

Panel A：在测试企业查询中自动选取「Top-1=柳州、Top-2=自治区」等最易用于论文叙事的案例（与历史版本一致，如柳工）。
Panel B：固定 policy_id（默认 535，图上有六大类 targetsIndustry，便于与 Panel C 行业口径 TEI 对齐）跑 P→E。

用法：
  python scripts/export_figure_case_study_samples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from matching.torch_metadata_fix import apply_torch_metadata_fix  # noqa: E402

apply_torch_metadata_fix()

from matching.bidirectional_matching import BidirectionalMatcher  # noqa: E402
from matching.evaluate_matching import build_test_queries_from_data  # noqa: E402
from matching.policy_embedding_defaults import JOINT_EMB, JOINT_IDX  # noqa: E402
from matching.ensure_joint_policy_embeddings import ensure_joint_policy_embeddings  # noqa: E402


# 网格搜索最优 E→P 超参（与 evaluation_results_a2_joint_grid_best 一致）
EP_HP = {
    "top_k": -1,
    "candidate_k": 1000,
    "score_threshold": None,
    "adaptive_quantile": 0.80,
    "relative_drop_threshold": 0.08,
    "max_output_cap": 130,
    "semantic_weight": 0.54,
    "structure_weight": 0.22,
    "importance_weight": 0.24,
    "industry_boost": 0.14,
}

INDUSTRY_Q_HP = {
    "adaptive_quantile": 0.82,
    "relative_drop_threshold": 0.12,
    "max_output_cap": 70,
}

# Panel B / Panel C 案例政策（与 compute_panel_c_tei_case_study.py 默认一致）
PANEL_B_POLICY_ID = 535
PANEL_B_POLICY_TITLE_EXPECTED = (
    "关于印发广西整市推进产业工人队伍建设支持企业用工行动方案的通知"
)

PANEL_C_MD_MARKERS = ("<!-- PANEL_C_TEI_START -->", "<!-- PANEL_C_TEI_END -->")

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


def _policy_meta(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        pid = int(r["policy_id"])
        y = r.get("year")
        out[pid] = {
            "policy_id": pid,
            "title": str(r["title"]),
            "year": int(y) if pd.notna(y) else None,
            "level": str(r["level"]) if pd.notna(r.get("level")) else "",
        }
    return out


def _enterprise_row_by_name(df_ent: pd.DataFrame, name: str) -> Optional[pd.Series]:
    m = df_ent[df_ent["name"].astype(str) == str(name)]
    if len(m) == 0:
        return None
    return m.iloc[0]


def _score_ep_case(m1: Dict[str, Any], m2: Dict[str, Any]) -> int:
    """越高越适合 Panel A（优先 Top-1 柳州 + Top-2 国家，其次柳州+自治区）。"""
    l1, l2 = m1.get("level") or "", m2.get("level") or ""
    y1, y2 = m1.get("year"), m2.get("year")
    s = 0
    if l1 == "柳州" and l2 == "国家":
        s += 200
    elif l1 == "柳州" and l2 == "自治区":
        s += 80
    if y1 is not None and y2 is not None and y1 >= y2:
        s += 25
    if l1 == "柳州":
        s += 5
    return s


def _first_national_rank(results: List[Tuple[int, float]], meta: Dict[int, Dict[str, Any]]) -> Optional[int]:
    for i, (pid, _) in enumerate(results[:20], start=1):
        m = meta.get(pid)
        if m and m.get("level") == "国家":
            return i
    return None


def _graph_company_id_to_name(matcher: BidirectionalMatcher) -> Dict[int, str]:
    nm = getattr(matcher.enterprise_retriever, "node_maps", {}) or {}
    cmap = nm.get("company", {}) or {}
    return {int(v): str(k) for k, v in cmap.items()}


def _extract_panel_c_md_block(text: str) -> Optional[str]:
    """重写 MD 时保留 Panel C（由 compute_panel_c_tei_case_study.py 维护）。"""
    start, end = PANEL_C_MD_MARKERS
    if start not in text or end not in text:
        return None
    i = text.index(start)
    j = text.index(end) + len(end)
    return text[i:j].strip()


def main() -> None:
    ensure_joint_policy_embeddings(PROJECT_ROOT)
    policies_df = pd.read_parquet(PROJECT_ROOT / "data_intermediate" / "policies_clean.parquet")
    meta = _policy_meta(policies_df)
    df_ent = pd.read_parquet(PROJECT_ROOT / "data_intermediate" / "enterprises_filtered.parquet")

    matcher = BidirectionalMatcher(
        PROJECT_ROOT,
        policy_emb_path=JOINT_EMB,
        policy_index_path=JOINT_IDX,
        gat_artifact_tag="a2_joint",
        policy_importance_parquet=None,
        ignore_env_importance_override=True,
    )
    cid_to_name = _graph_company_id_to_name(matcher)

    enterprise_queries, _policy_queries = build_test_queries_from_data(
        max_enterprise_queries=300,
        max_industry_queries=30,
        max_policy_queries=200,
    )

    best: Optional[Tuple[int, List[Tuple[int, float]], Dict[str, Any]]] = None
    best_score = -1
    runners_up: List[Dict[str, Any]] = []

    company_queries = [q for q in enterprise_queries if q.get("type") == "company_name"]
    for q in company_queries:
        text = str(q["query"])
        results = matcher.query_policies_by_enterprise(
            text,
            top_k=EP_HP["top_k"],
            candidate_k=EP_HP["candidate_k"],
            score_threshold=EP_HP["score_threshold"],
            adaptive_quantile=EP_HP["adaptive_quantile"],
            relative_drop_threshold=EP_HP["relative_drop_threshold"],
            max_output_cap=EP_HP["max_output_cap"],
            semantic_weight=EP_HP["semantic_weight"],
            structure_weight=EP_HP["structure_weight"],
            importance_weight=EP_HP["importance_weight"],
            industry_boost=EP_HP["industry_boost"],
        )
        if len(results) < 2:
            continue
        p1_id, s1 = results[0]
        p2_id, s2 = results[1]
        m1, m2 = meta.get(p1_id), meta.get(p2_id)
        if not m1 or not m2:
            continue
        sc = _score_ep_case(m1, m2)
        rec = {
            "enterprise_name": text,
            "query_type": "company_name",
            "pattern_score": sc,
            "top1": {**m1, "match_score": float(s1)},
            "top2": {**m2, "match_score": float(s2)},
        }
        if sc > best_score:
            if best is not None:
                _, _, prev = best
                runners_up.append(
                    {
                        "enterprise_name": prev["enterprise_name"],
                        "pattern_score": best_score,
                        "top1_level": prev["top1"]["level"],
                        "top2_level": prev["top2"]["level"],
                    }
                )
            best_score = sc
            best = (sc, results, rec)
        elif sc >= 40 and len(runners_up) < 8:
            runners_up.append(
                {
                    "enterprise_name": text,
                    "pattern_score": sc,
                    "top1_level": m1["level"],
                    "top2_level": m2["level"],
                }
            )

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "figure_case_study_ep_pe.json"
    md_path = out_dir / "figure_case_study_ep_pe.md"

    if best is None:
        payload = {
            "error": "未找到可用的企业→政策样例（请检查测试查询与 policies_clean 层级字段）。",
            "runners_up": runners_up[:20],
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"未找到理想样例，已写 {json_path}")
        return

    _, ep_results, rec = best
    ent_name = rec["enterprise_name"]
    erow = _enterprise_row_by_name(df_ent, ent_name)
    industry = str(erow["industry"]) if erow is not None and pd.notna(erow.get("industry")) else ""
    ind_major = str(erow["industry_major"]) if erow is not None and pd.notna(erow.get("industry_major")) else ""

    top_policies = []
    for rank, (pid, sc) in enumerate(ep_results[:5], start=1):
        m = meta.get(pid, {})
        top_policies.append(
            {
                "rank": rank,
                "policy_id": int(pid),
                "title": m.get("title", ""),
                "year": m.get("year"),
                "level": m.get("level", ""),
                "match_score": float(sc),
            }
        )

    pe_policy_id = int(PANEL_B_POLICY_ID)
    meta_pe = meta.get(pe_policy_id, {})

    pe_raw = matcher.retrieve_enterprises_by_policy(
        pe_policy_id,
        top_k=PE_HP["top_k"],
        k_hop=int(PE_HP.get("k_hop", 2)),
        candidate_k=PE_HP["candidate_k"],
        score_threshold=PE_HP["score_threshold"],
        adaptive_quantile=PE_HP["adaptive_quantile"],
        relative_drop_threshold=PE_HP["relative_drop_threshold"],
        max_output_cap=PE_HP["max_output_cap"],
        direct_support_boost=PE_HP["direct_support_boost"],
    )
    pe_node = matcher.enterprise_retriever._resolve_policy_node_id(pe_policy_id)
    ds_map = getattr(matcher.enterprise_retriever, "policy_direct_support_companies", {}) or {}
    ds_set = set(ds_map.get(int(pe_node), set())) if pe_node is not None else set()
    top_enterprises = []
    for rank, (cid, sc) in enumerate(pe_raw[:12], start=1):
        top_enterprises.append(
            {
                "rank": rank,
                "company_graph_node_id": int(cid),
                "name": cid_to_name.get(int(cid), ""),
                "priority_score": float(sc),
                "direct_support_edge": bool(int(cid) in ds_set),
            }
        )

    nat_rank = _first_national_rank(ep_results, meta)
    if rec["top1"]["level"] == "柳州" and rec["top2"]["level"] == "国家":
        nar_tail = "符合「市级更靠前、国家级宏观政策相对靠后」的示意图叙事。"
    elif rec["top1"]["level"] == "柳州" and rec["top2"]["level"] == "自治区":
        nar_tail = (
            "体现「柳州市政策优先于自治区层面政策」的空间层级；"
            + (
                f"同列表中第 {nat_rank} 位起可见国家级政策，插图若需「Top-2=国家」可改用该条并注明。"
                if nat_rank is not None
                else "若需「Top-2=国家级」，可在同列表中另选国家层级条目（见 JSON national_policy_rank_in_list）。"
            )
        )
    else:
        nar_tail = "详见层级与年份字段。"

    narrative_a = (
        f"企业名查询「{ent_name}」下，排序第 1 为{rec['top1']['level']}层级政策（{rec['top1']['year']} 年），"
        f"第 2 为{rec['top2']['level']}层级政策（{rec['top2']['year']} 年）。{nar_tail}"
    )
    kh = int(PE_HP.get("k_hop", 2))
    narrative_b = (
        f"固定以 policy_id={pe_policy_id}（{meta_pe.get('title', PANEL_B_POLICY_TITLE_EXPECTED)}）为 P→E 查询；"
        f"该政策在图上有六大类 targetsIndustry，可与 Panel C 的 **行业条件覆盖率** 对齐。"
        f"子图采样 k_hop={kh}（与传导主评测 `industry_conditioned_coverage` 默认一致），"
        f"GAT 子图编码 + direct_support_boost={PE_HP['direct_support_boost']}；"
        f"direct_support_edge=true 表示图中 policy supports company 直接边先验加分生效。"
    )

    payload: Dict[str, Any] = {
        "matcher": {
            "policy_text_mode": "joint",
            "gat_artifact_tag": "a2_joint",
            "importance_parquet": "evaluation/policy_importance_with_decay_a2_joint.parquet",
        },
        "panel_a_ep": {
            "selection_rule_zh": "测试集企业名查询中 pattern_score 最高（柳州优先、其次柳州+自治区等）",
            "enterprise_query_name": ent_name,
            "query_type": "company_name",
            "enterprise_industry": industry,
            "enterprise_industry_major": ind_major,
            "figure_note_zh": "插图中「注册年份、初创期」等若无 KG 字段可自行作为示意图标签；行业类信息来自 enterprises_filtered。",
            "hyperparameters": {**EP_HP, "industry_query_overrides": INDUSTRY_Q_HP},
            "top_policies": top_policies,
            "narrative_zh": narrative_a,
        },
        "panel_b_pe": {
            "query_policy_id": pe_policy_id,
            "query_policy_title": str(meta_pe.get("title", PANEL_B_POLICY_TITLE_EXPECTED)),
            "hyperparameters": PE_HP,
            "top_enterprises": top_enterprises,
            "narrative_zh": narrative_b,
        },
        "alternative_ep_patterns": sorted(runners_up, key=lambda x: -x.get("pattern_score", 0))[:12],
        "national_policy_rank_in_list": nat_rank,
    }

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = f"""# 论文插图样例：E→P / P→E（主实验 joint + a2_joint）

数据文件：`reports/figure_case_study_ep_pe.json`

## Panel A — 企业 → 政策

- **选取方式**：企业名测试查询中自动选 pattern 最优（与历史柳工案例一致）
- **查询企业**：{ent_name}
- **行业（数据）**：{industry or "（无）"} / 大类：{ind_major or "（无）"}

### Top 政策（前 5）

| Rank | policy_id | level | year | score | title |
|------|-----------|-------|------|-------|-------|
"""
    for row in top_policies:
        t = str(row["title"]).replace("|", "\\|")
        md += f"| {row['rank']} | {row['policy_id']} | {row['level']} | {row['year']} | {row['match_score']:.4f} | {t} |\n"
    md += f"""
{narrative_a}

## Panel B — 政策 → 企业（固定政策）

- **政策 ID**：{pe_policy_id}
- **标题**：{meta_pe.get("title", PANEL_B_POLICY_TITLE_EXPECTED)}

### Top 企业（前 12）

| Rank | graph_node_id | direct_support | score | name |
|------|---------------|----------------|-------|------|
"""
    for row in top_enterprises:
        nm = str(row["name"]).replace("|", "\\|")
        md += f"| {row['rank']} | {row['company_graph_node_id']} | {row['direct_support_edge']} | {row['priority_score']:.4f} | {nm} |\n"
    md += f"\n{narrative_b}\n"
    preserved = (
        _extract_panel_c_md_block(md_path.read_text(encoding="utf-8"))
        if md_path.exists()
        else None
    )
    if preserved:
        md = md.rstrip() + "\n\n" + preserved + "\n"
    md_path.write_text(md, encoding="utf-8")

    print(f"已写入:\n  {json_path}\n  {md_path}")
    print(narrative_a)
    print(narrative_b)


if __name__ == "__main__":
    main()
