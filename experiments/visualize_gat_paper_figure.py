#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAT 对比学习嵌入 + 异质子图示意图（论文用）。
输出到 images/（可通过 --out_dir 覆盖）。

依赖: numpy, matplotlib, dgl, torch, scikit-learn, networkx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import dgl
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    from sklearn.manifold import TSNE
except ImportError as e:
    raise SystemExit("需要 scikit-learn: pip install scikit-learn") from e

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# 默认图片目录：项目根目录下的 images/
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "images"


def _set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.4,
        }
    )


# Okabe–Ito 色盲友好配色
COL_POLICY = "#0072B2"  # blue
COL_COMPANY = "#D55E00"  # vermillion
COL_INDUSTRY = "#009E73"  # green
COL_EDGE = "#B0B0B0"


def load_embeddings(
    pol_path: Path, com_path: Path, ind_path: Path | None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pol = np.load(pol_path)
    com = np.load(com_path)
    ind = np.load(ind_path) if ind_path and ind_path.exists() else None
    return pol, com, ind


def plot_tsne_multitype(
    pol_emb: np.ndarray,
    com_emb: np.ndarray,
    ind_emb: np.ndarray | None,
    out_path: Path,
    seed: int,
    n_policy: int,
    n_company: int,
) -> None:
    rng = np.random.default_rng(seed)
    npol = pol_emb.shape[0]
    ncom = com_emb.shape[0]

    pi = rng.choice(npol, size=min(n_policy, npol), replace=False)
    ci = rng.choice(ncom, size=min(n_company, ncom), replace=False)

    parts: List[np.ndarray] = [pol_emb[pi], com_emb[ci]]
    types: List[np.ndarray] = [
        np.zeros(len(pi), dtype=np.int32),
        np.ones(len(ci), dtype=np.int32),
    ]
    if ind_emb is not None and len(ind_emb) > 0:
        parts.append(ind_emb)
        types.append(np.full(len(ind_emb), 2, dtype=np.int32))

    X = np.vstack(parts)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    y = np.concatenate(types)

    n_samples = X.shape[0]
    perplexity = min(30, max(5, n_samples // 4))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=1000,
    )
    Z = tsne.fit_transform(X)

    _set_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.2), layout="constrained")

    mask_p = y == 0
    mask_c = y == 1
    mask_i = y == 2

    ax.scatter(
        Z[mask_p, 0],
        Z[mask_p, 1],
        c=COL_POLICY,
        s=22,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.35,
        label="Policy",
        rasterized=True,
    )
    ax.scatter(
        Z[mask_c, 0],
        Z[mask_c, 1],
        c=COL_COMPANY,
        s=18,
        alpha=0.65,
        edgecolors="white",
        linewidths=0.3,
        label="Enterprise",
        rasterized=True,
    )
    if mask_i.any():
        ax.scatter(
            Z[mask_i, 0],
            Z[mask_i, 1],
            c=COL_INDUSTRY,
            s=120,
            marker="^",
            alpha=0.9,
            edgecolors="white",
            linewidths=0.6,
            label="Industry",
            zorder=5,
            rasterized=True,
        )

    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("Heterogeneous GAT contrastive embeddings (2D t-SNE)")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(frameon=True, fancybox=False, edgecolor="#888888", loc="best")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _node_key(ntype: str, nid: int) -> Tuple[str, int]:
    return (str(ntype), int(nid))


def hetero_khop_to_networkx(
    sub: dgl.DGLHeteroGraph,
) -> nx.Graph:
    """
    将 DGL 子图转为无向 NetworkX；节点键 (ntype, 原图节点 id)。
    relabel_nodes=False 时边端点为全局 id。
    """
    G = nx.Graph()
    for ntype in sub.ntypes:
        n = sub.number_of_nodes(ntype)
        if n == 0:
            continue
        ids = sub.nodes(ntype)
        ids_np = ids.numpy() if hasattr(ids, "numpy") else np.asarray(ids)
        for raw in ids_np.tolist():
            nid = int(raw)
            G.add_node(_node_key(ntype, nid), ntype=ntype)

    for cetype in sub.canonical_etypes:
        st, rel, dt = cetype
        if sub.num_edges(cetype) == 0:
            continue
        src, dst = sub.edges(etype=cetype)
        s_np = src.numpy()
        d_np = dst.numpy()
        for s, d in zip(s_np.tolist(), d_np.tolist()):
            u, v = _node_key(st, int(s)), _node_key(dt, int(d))
            if u in G and v in G:
                G.add_edge(u, v, rel=str(rel))

    return G


def count_belongs_edges(sub: dgl.DGLHeteroGraph) -> int:
    et = ("company", "belongsTo", "industry")
    if et not in sub.canonical_etypes:
        return 0
    return int(sub.num_edges(et))


def pick_largest_ego_subgraph(
    g: dgl.DGLHeteroGraph,
    k_list: List[int],
    max_policy_candidates: int,
) -> Tuple[int, int, dgl.DGLHeteroGraph]:
    """
    在若干政策种子与 k-hop 组合中选 ego 子图。
    优先 **含 belongsTo（企业–行业）** 的子图；在此前提下节点数最大，再并列看 belongsTo 边数。
    若用 (节点数, belongsTo) 排序，常会选中「巨大政策–企业簇但无任何行业」的子图，图中看不到 industry。
    """
    et = ("policy", "supports", "company")
    if et not in g.canonical_etypes:
        return 0, max(k_list) if k_list else 2, g
    src, _ = g.edges(etype=et)
    if src.numel() == 0:
        return 0, max(k_list) if k_list else 2, g
    s = src.numpy()
    uniq, cnt = np.unique(s, return_counts=True)
    order = np.argsort(-cnt)
    cand = uniq[order[: max_policy_candidates]].tolist()

    best_k = k_list[0]
    best_pid = int(cand[0])
    best_sub = None
    # (含 belongsTo?, 总节点数, belongsTo 边数)；先 True>False，再大图优先
    best_score = (False, -1, -1)

    for pid in cand:
        for k in k_list:
            try:
                # relabel_nodes=False 时 DGL 只返回子图，不能二元解包
                sub = dgl.khop_out_subgraph(
                    g, {"policy": [int(pid)]}, k=int(k), relabel_nodes=False
                )
            except Exception:
                continue
            n_nodes = sum(sub.number_of_nodes(t) for t in sub.ntypes)
            n_bel = count_belongs_edges(sub)
            score = (n_bel > 0, n_nodes, n_bel)
            if score > best_score:
                best_score = score
                best_pid = int(pid)
                best_k = int(k)
                best_sub = sub

    if best_sub is None:
        best_sub = dgl.khop_out_subgraph(
            g, {"policy": [best_pid]}, k=best_k, relabel_nodes=False
        )
    _, nv, nb = best_score
    print(
        f"  选定 ego: policy_node_id={best_pid}, k={best_k}, "
        f"|V|={nv}, |belongsTo|={nb} (优先含行业–企业边)"
    )
    return best_pid, best_k, best_sub


def _count_ntype(G: nx.Graph) -> Tuple[int, int, int]:
    p = c = i = 0
    for n in G.nodes():
        nt = G.nodes[n].get("ntype", "")
        if nt == "policy":
            p += 1
        elif nt == "company":
            c += 1
        elif nt == "industry":
            i += 1
    return p, c, i


def trim_subgraph_by_bfs(
    G: nx.Graph,
    center: Tuple[str, int],
    max_nodes: int,
    *,
    max_companions_per_industry: int = 24,
) -> nx.Graph:
    """
    按到 center 的最短距离保留至多 max_nodes 个节点。
    行业节点通常在 policy→company 之后（≥3 跳），若直接取最近的 max_nodes 个，
    往往会全是政策/企业而把 industry 全部挤掉；因此先取 BFS 基底，再强制纳入
    最近的若干 industry 及其部分 belongsTo 企业，并从最远端的非 industry 节点 eviction。
    """
    if G.number_of_nodes() <= max_nodes:
        return G
    dist = nx.single_source_shortest_path_length(G, center)
    nodes_sorted = sorted(G.nodes(), key=lambda n: dist.get(n, 10**9))
    keep: set = set(nodes_sorted[:max_nodes])

    industries = [n for n in G.nodes() if G.nodes[n].get("ntype") == "industry"]
    if not industries:
        return G.subgraph(keep).copy()

    def evict_farthest_non_industry() -> bool:
        removable = [
            x
            for x in keep
            if x != center and G.nodes[x].get("ntype") != "industry"
        ]
        if not removable:
            return False
        drop = max(removable, key=lambda z: dist.get(z, 0))
        keep.remove(drop)
        return True

    # 按距离把尚未进入 keep 的行业及其企业邻居拉进来，并维持 |keep|<=max_nodes
    for inn in sorted(industries, key=lambda x: dist.get(x, 10**9)):
        if inn in keep:
            continue
        keep.add(inn)
        added = 0
        for nb in G.neighbors(inn):
            if G.nodes[nb].get("ntype") != "company":
                continue
            keep.add(nb)
            added += 1
            if added >= max_companions_per_industry:
                break
        while len(keep) > max_nodes:
            if not evict_farthest_non_industry():
                break

    return G.subgraph(keep).copy()


def plot_subgraph_structure(
    sub: dgl.DGLHeteroGraph,
    out_path: Path,
    center_policy_id: int,
    ego_k: int,
    max_nodes: int,
    seed: int,
) -> None:
    G = hetero_khop_to_networkx(sub)
    if G.number_of_nodes() == 0:
        print("子图为空，跳过结构图")
        return

    center = _node_key("policy", int(center_policy_id))
    if center not in G:
        pol_nodes = [n for n in G.nodes() if n[0] == "policy"]
        center = pol_nodes[0] if pol_nodes else next(iter(G.nodes()))

    bp, bc, bi = _count_ntype(G)
    print(f"  子图结构图 截断前 NetworkX: policy={bp}, company={bc}, industry={bi}, |V|={G.number_of_nodes()}")
    G = trim_subgraph_by_bfs(G, center, max_nodes)
    ap, ac, ai = _count_ntype(G)
    print(f"  子图结构图 截断后 NetworkX: policy={ap}, company={ac}, industry={ai}, |V|={G.number_of_nodes()}")

    _set_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.0), layout="constrained")

    try:
        pos = nx.spring_layout(G, k=1.35, iterations=100, seed=seed)
    except TypeError:
        import random

        random.seed(seed)
        pos = nx.spring_layout(G, k=1.35, iterations=100)

    nodes_p = [n for n in G.nodes() if G.nodes[n].get("ntype") == "policy"]
    nodes_c = [n for n in G.nodes() if G.nodes[n].get("ntype") == "company"]
    nodes_i = [n for n in G.nodes() if G.nodes[n].get("ntype") == "industry"]

    # 按关系类型分层绘制边（企业–行业 belongsTo / includesCompany 加粗）
    e_pp = [(u, v) for u, v, d in G.edges(data=True) if d.get("rel") in ("transmitsTo", "transmitsFrom")]
    e_pc = [(u, v) for u, v, d in G.edges(data=True) if d.get("rel") in ("supports", "supportedByPolicy")]
    e_pi = [(u, v) for u, v, d in G.edges(data=True) if d.get("rel") in ("targetsIndustry", "targetedByPolicy")]
    e_ci = [(u, v) for u, v, d in G.edges(data=True) if d.get("rel") in ("belongsTo", "includesCompany")]

    def draw_elist(elist, **kw):
        if not elist:
            return
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=elist, **kw)

    draw_elist(e_pp, width=0.35, alpha=0.35, edge_color="#AAAAAA", style="dotted")
    draw_elist(e_pc, width=0.85, alpha=0.55, edge_color=COL_POLICY, style="solid")
    draw_elist(e_pi, width=0.75, alpha=0.55, edge_color="#7B68A6", style="dashdot")
    draw_elist(
        e_ci,
        width=1.35,
        alpha=0.75,
        edge_color=COL_INDUSTRY,
        style="solid",
    )

    def draw_nodes(nodelist, color: str, size: float, **kw):
        if not nodelist:
            return
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=nodelist,
            node_color=color,
            node_size=size,
            alpha=0.92,
            linewidths=0.5,
            edgecolors="white",
            **kw,
        )

    draw_nodes(nodes_p, COL_POLICY, 220)
    draw_nodes(nodes_c, COL_COMPANY, 95)
    # 三角形，避免与绿色 belongsTo 线段在视觉上混成一团
    draw_nodes(nodes_i, COL_INDUSTRY, 380, node_shape="^")

    ax.set_title(
        f"Largest k-hop ego-network (policy id={center_policy_id}, k={ego_k}, |V|={G.number_of_nodes()})"
    )
    ax.axis("off")

    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_POLICY, markersize=9, label="Policy"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_COMPANY, markersize=8, label="Enterprise"),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=COL_INDUSTRY,
            markersize=10,
            label="Industry",
        ),
        Line2D([0], [0], color=COL_INDUSTRY, lw=2.2, label="Enterprise–Industry (belongsTo)"),
        Line2D([0], [0], color=COL_POLICY, lw=1.5, label="Policy–Enterprise (supports)"),
        Line2D([0], [0], color="#7B68A6", lw=1.2, linestyle="dashdot", label="Policy–Industry (targets)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", frameon=True, edgecolor="#888888", fontsize=8.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--pol_emb", type=str, default="graph/gat_policy_emb_contrastive.npy")
    parser.add_argument("--com_emb", type=str, default="graph/gat_company_emb_contrastive.npy")
    parser.add_argument("--ind_emb", type=str, default="graph/gat_industry_emb_contrastive.npy")
    parser.add_argument("--graph_bin", type=str, default="graph/graph_data.bin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_policy_sample", type=int, default=450)
    parser.add_argument("--n_company_sample", type=int, default=650)
    parser.add_argument("--skip_subgraph", action="store_true")
    parser.add_argument(
        "--subgraph_k_list",
        type=str,
        default="2,3,4",
        help="逗号分隔，在这些 hop 中选节点数最大的 ego 子图",
    )
    parser.add_argument("--subgraph_max_nodes", type=int, default=200)
    parser.add_argument("--subgraph_policy_candidates", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pol_p = PROJECT_ROOT / args.pol_emb
    com_p = PROJECT_ROOT / args.com_emb
    ind_p = PROJECT_ROOT / args.ind_emb

    if not pol_p.exists() or not com_p.exists():
        print(f"缺少嵌入文件: {pol_p} 或 {com_p}", file=sys.stderr)
        sys.exit(1)

    pol_emb, com_emb, ind_emb = load_embeddings(pol_p, com_p, ind_p)
    if ind_emb is None:
        print("提示: 未找到 industry GAT 嵌入，t-SNE 仅含 Policy / Enterprise")

    plot_tsne_multitype(
        pol_emb,
        com_emb,
        ind_emb,
        out_dir / "gat_hetero_tsne_paper.png",
        seed=args.seed,
        n_policy=args.n_policy_sample,
        n_company=args.n_company_sample,
    )

    if not args.skip_subgraph:
        gb = PROJECT_ROOT / args.graph_bin
        if not gb.exists():
            print(f"跳过子图: 无 {gb}")
            return
        graphs, _ = dgl.load_graphs(str(gb))
        g = graphs[0]
        k_list = [int(x.strip()) for x in args.subgraph_k_list.split(",") if x.strip()]
        if not k_list:
            k_list = [2, 3, 4]
        pid, ek, sub = pick_largest_ego_subgraph(
            g,
            k_list=k_list,
            max_policy_candidates=args.subgraph_policy_candidates,
        )
        plot_subgraph_structure(
            sub,
            out_dir / "gat_hetero_subgraph_ego_paper.png",
            center_policy_id=pid,
            ego_k=ek,
            max_nodes=args.subgraph_max_nodes,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
