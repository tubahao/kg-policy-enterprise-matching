#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 风格的子图采样与重要性计算：
- 输入：graph/graph_data.bin，graph/policy_node_emb.npy
- 输出：
    graphrag/subgraphs.pkl
    graphrag/importance_scores.npy
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import dgl
import networkx as nx
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def load_graph(graph_path: Path) -> dgl.DGLHeteroGraph:
    graphs, _ = dgl.load_graphs(str(graph_path))
    return graphs[0]


def _fix_dangling_nodes(nx_g: nx.Graph) -> None:
    """为出度为0的悬空节点添加自环，防止PageRank概率泄漏。"""
    dangling = [n for n in nx_g.nodes() if nx_g.degree(n) == 0 or
                (nx_g.is_directed() and nx_g.out_degree(n) == 0)]
    for n in dangling:
        nx_g.add_edge(n, n)


def compute_importance(
    graph: dgl.DGLHeteroGraph,
    seed_nodes: Dict[str, List[int]] = None,
) -> Dict[str, Dict[int, float]]:
    """
    计算节点重要性。当提供 seed_nodes 时使用个性化 PageRank（PPR），
    否则使用全局 PageRank + 悬空节点补偿。
    """
    importance = {}
    
    if ("policy", "transmitsTo", "policy") in graph.canonical_etypes:
        policy_g = graph["policy", "transmitsTo", "policy"]
        nx_g = policy_g.to_networkx().to_undirected()
        _fix_dangling_nodes(nx_g)

        personalization = None
        if seed_nodes and "policy" in seed_nodes:
            all_nodes = list(nx_g.nodes())
            personalization = {n: 0.0 for n in all_nodes}
            for s in seed_nodes["policy"]:
                if s in personalization:
                    personalization[s] = 1.0
            total = sum(personalization.values())
            if total > 0:
                personalization = {k: v / total for k, v in personalization.items()}
            else:
                personalization = None

        pr = nx.pagerank(nx_g, personalization=personalization)
        importance["policy"] = pr
        pr_type = "PPR" if personalization else "全局PageRank"
        print(f"计算政策节点重要性({pr_type}): {len(pr)} 个节点")
    
    if ("company", "belongsTo", "industry") in graph.canonical_etypes:
        company_g = graph["company", "belongsTo", "industry"]
        nx_g = company_g.to_networkx().to_undirected()
        _fix_dangling_nodes(nx_g)

        personalization = None
        if seed_nodes and "company" in seed_nodes:
            all_nodes = list(nx_g.nodes())
            personalization = {n: 0.0 for n in all_nodes}
            for s in seed_nodes["company"]:
                if s in personalization:
                    personalization[s] = 1.0
            total = sum(personalization.values())
            if total > 0:
                personalization = {k: v / total for k, v in personalization.items()}
            else:
                personalization = None

        pr = nx.pagerank(nx_g, personalization=personalization)
        importance["company"] = pr
        pr_type = "PPR" if personalization else "全局PageRank"
        print(f"计算企业节点重要性({pr_type}): {len(pr)} 个节点")
    
    return importance


def sample_subgraphs(
    graph: dgl.DGLHeteroGraph, top_nodes: List[int], k_hop: int, node_type: str = "policy"
) -> List[Dict]:
    """为每个高重要性节点采样 k-hop 异构图子图。"""
    subgraphs = []
    
    for center in top_nodes:
        # 使用khop_out_subgraph采样异构图子图
        subgraph, inverse_indices = dgl.khop_out_subgraph(
            graph, {node_type: [center]}, k=k_hop
        )
        
        # 提取子图中的节点和边信息
        subgraph_info = {
            "center_node": center,
            "center_type": node_type,
            "nodes": {},
            "edges": []
        }
        
        # 提取各类型节点
        for ntype in subgraph.ntypes:
            nodes = subgraph.nodes(ntype).numpy().tolist()
            subgraph_info["nodes"][ntype] = nodes
        
        # 提取边
        for etype in subgraph.canonical_etypes:
            src, dst = subgraph.edges(etype=etype)
            edges = list(zip(src.numpy().tolist(), dst.numpy().tolist()))
            subgraph_info["edges"].append({
                "etype": etype,
                "edges": edges
            })
        
        subgraphs.append(subgraph_info)
    
    return subgraphs


def main():
    parser = argparse.ArgumentParser(description="GraphRAG 子图采样与重要性计算")
    parser.add_argument("--graph_path", type=str, default="graph/graph_data.bin")
    parser.add_argument("--node_emb", type=str, default="graph/policy_node_emb.npy")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--k_hop", type=int, default=2)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    graph_path = project_root / args.graph_path
    node_emb_path = project_root / args.node_emb

    graph = load_graph(graph_path)

    # 第一轮：全局重要性（含悬空节点修复）
    importance = compute_importance(graph)

    if "policy" in importance:
        sorted_nodes = sorted(importance["policy"].items(), key=lambda x: x[1], reverse=True)
        top_nodes = [nid for nid, _ in sorted_nodes[: args.top_k]]
        print(f"\n选取前 {args.top_k} 个重要政策节点进行子图采样")

        # 第二轮：以 top 节点为种子计算 PPR（局部相关性更强）
        ppr_importance = compute_importance(graph, seed_nodes={"policy": top_nodes})
        if "policy" in ppr_importance:
            importance["policy"] = ppr_importance["policy"]
    else:
        print("警告: 未找到政策节点重要性，使用前50个节点")
        top_nodes = list(range(min(50, graph.number_of_nodes("policy"))))

    subgraphs = sample_subgraphs(graph, top_nodes, args.k_hop, "policy")

    out_dir = project_root / "graphrag"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存重要性向量（与 node_id 对齐）
    if "policy" in importance:
        max_id = max(importance["policy"].keys()) if importance["policy"] else -1
        imp_arr = np.zeros(max_id + 1, dtype=float)
        for nid, score in importance["policy"].items():
            imp_arr[nid] = score
        np.save(out_dir / "importance_scores.npy", imp_arr)
        print(f"保存政策节点重要性分数: {len(imp_arr)} 个节点")

    # 保存子图
    with open(out_dir / "subgraphs.pkl", "wb") as f:
        pickle.dump(subgraphs, f)
    
    # 保存子图元信息（JSON格式，便于查看）
    import json
    subgraph_metadata = []
    for i, sg in enumerate(subgraphs):
        node_counts = {ntype: len(nodes) for ntype, nodes in sg["nodes"].items()}
        subgraph_metadata.append({
            "subgraph_id": i,
            "center_ntype": sg["center_type"],
            "center_node_id": sg["center_node"],
            "importance_score": importance.get("policy", {}).get(sg["center_node"], 0.0),
            "num_nodes": sum(node_counts.values()),
            "node_counts": node_counts,
            "num_edges": sum(len(e["edges"]) for e in sg["edges"])
        })
    
    with open(out_dir / "subgraph_metadata.json", "w", encoding="utf-8") as f:
        json.dump(subgraph_metadata, f, ensure_ascii=False, indent=2)

    print("子图采样完成")
    print(f"- 重要性分数: {out_dir / 'importance_scores.npy'}")
    print(f"- 子图列表: {out_dir / 'subgraphs.pkl'}")
    print(f"- 子图元信息: {out_dir / 'subgraph_metadata.json'}")
    print(f"- 共采样 {len(subgraphs)} 个子图")


if __name__ == "__main__":
    main()

