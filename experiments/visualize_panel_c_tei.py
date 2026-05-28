#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel C 插图：读取 reports/figure_case_study_panel_c.json，绘制
- 行业覆盖率环形进度示意
- 跳数分布柱状图（同心圆可用此分布近似表达）
- 以政策为中心的放射状散点（半径∝hop，颜色∝PPR）

用法：
  python scripts/visualize_panel_c_tei.py
  python scripts/visualize_panel_c_tei.py --json reports/figure_case_study_panel_c.json --out reports/figures/panel_c_tei_case_study.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _configure_matplotlib_cjk() -> None:
    """避免中文标题/标签在 DejaVu Sans 下缺字。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {
        f.name for f in font_manager.fontManager.ttflist
    }
    for candidate in (
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
    ):
        if candidate in names:
            plt.rcParams["font.sans-serif"] = [candidate] + list(
                plt.rcParams.get("font.sans-serif", [])
            )
            break
    plt.rcParams["axes.unicode_minus"] = False


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ring_coverage_ax(ax, pct: float, title: str) -> None:
    """简易环形进度（matplotlib wedge）。"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    ax.set_aspect("equal")
    ax.axis("off")
    theta1 = 90
    theta2 = 90 - (pct / 100.0) * 360.0
    # 背景环
    bg = Wedge((0, 0), 1.0, 0, 360, width=0.25, facecolor="#e8e8e8", edgecolor="none")
    ax.add_patch(bg)
    if pct > 0:
        fg = Wedge(
            (0, 0),
            1.0,
            theta2,
            theta1,
            width=0.25,
            facecolor="#2ca02c",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(fg)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.text(0, 0.05, f"{pct:.1f}%", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0, -0.35, title, ha="center", va="center", fontsize=10)


def _hop_bars_ax(ax, hop_dist: Dict[str, int], title: str) -> None:
    import matplotlib.pyplot as plt

    keys = sorted(hop_dist.keys(), key=lambda x: int(x))
    vals = [hop_dist[k] for k in keys]
    xs = np.arange(len(keys))
    ax.bar(xs, vals, color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}-hop" for k in keys])
    ax.set_ylabel("count")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.25)


def _radial_scatter_ax(ax, nodes: List[Dict[str, Any]], policy_title_short: str) -> None:
    import matplotlib.pyplot as plt

    usable = [n for n in nodes if int(n.get("hop", -1)) >= 0]
    if not usable:
        ax.text(0.5, 0.5, "无可用 hop 数据", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    hops = np.array([int(n["hop"]) for n in usable], dtype=np.float64)
    pprs = np.array([max(float(n.get("ppr_mass", 0.0)), 1e-12) for n in usable], dtype=np.float64)
    ppr_log = np.log10(pprs)
    ppr_norm = (ppr_log - ppr_log.min()) / max(ppr_log.max() - ppr_log.min(), 1e-9)

    n = len(usable)
    angles = np.array([2 * math.pi * i / max(n, 1) for i in range(n)], dtype=np.float64)
    # 半径：hop + 小抖动避免重叠
    rng = np.random.default_rng(42)
    r = hops + 0.08 * rng.standard_normal(n)
    r = np.clip(r, 0.15, None)
    x = r * np.cos(angles)
    y = r * np.sin(angles)

    sc = ax.scatter(
        x,
        y,
        c=ppr_norm,
        cmap="YlOrRd",
        s=28,
        alpha=0.85,
        edgecolors="#333333",
        linewidths=0.2,
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("PPR (norm, log-scale)", fontsize=8)

    ax.scatter([0], [0], c="#d62728", s=120, zorder=5, edgecolors="white", linewidths=1)
    ax.text(0, 0, "P", ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=6)
    short = policy_title_short[:22] + ("…" if len(policy_title_short) > 22 else "")
    ax.set_title(f"放射状布局（半径∝hop）\n{short}", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    # 参考同心圆
    for h in sorted(set(int(h) for h in hops.tolist())):
        circle = plt.Circle((0, 0), h + 0.2, fill=False, linestyle="--", color="#bbbbbb", linewidth=0.6)
        ax.add_patch(circle)
    lim = float(np.max(r)) + 0.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "figure_case_study_panel_c.json"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "figures" / "panel_c_tei_case_study.png"),
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    _configure_matplotlib_cjk()

    data = _load_json(Path(args.json))
    pct = float(data.get("coverage_industry_pct", data.get("coverage_industry", 0) * 100))
    hop_dist = data.get("hop_distribution") or {}
    nodes = data.get("company_nodes") or []
    title = str(data.get("policy_title", "policy"))

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.4], wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    _ring_coverage_ax(ax0, pct, "Coverage\n(行业 C_ind)")
    _hop_bars_ax(ax1, hop_dist, "Hop depth 分布")
    _radial_scatter_ax(ax2, nodes, title)

    pid = data.get("policy_id", "")
    fig.suptitle(f"Panel C — TEI 可视化（policy_id={pid}）", fontsize=12, fontweight="bold", y=1.02)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已保存 {out_path}")


if __name__ == "__main__":
    main()
