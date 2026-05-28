#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在政策子图上训练简化版GraphSAGE/GAT（默认GraphSAGE）：
- 输入：graph/graph_data.bin（由 build_graph.py 生成）
- 输出：
    graph/policy_node_emb.npy
    graph/edge_emb.npy
    graph/checkpoints/model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = in_dim if i == 0 else hidden_dim
            self.layers.append(dgl.nn.SAGEConv(in_d, hidden_dim, aggregator_type="mean"))

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = feat
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
        return h


def get_policy_graph(graph: dgl.DGLHeteroGraph) -> dgl.DGLGraph:
    if ("policy", "transmitsTo", "policy") not in graph.canonical_etypes:
        raise ValueError("图中不存在政策-政策关系 transmitsTo")
    policy_g = graph["policy", "transmitsTo", "policy"]
    return policy_g


def main():
    parser = argparse.ArgumentParser(description="训练政策子图的GNN")
    parser.add_argument("--graph_path", type=str, default="graph/graph_data.bin")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    graph_path = project_root / args.graph_path

    graphs, _ = dgl.load_graphs(str(graph_path))
    hetero_g = graphs[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_g = get_policy_graph(hetero_g)
    feat = hetero_g.nodes["policy"].data["feat"].to(device)

    # 将特征复制到子图
    policy_g = policy_g.to(device)
    policy_g.ndata["feat"] = feat

    model = GraphSAGE(in_dim=feat.shape[1], hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        out = model(policy_g, policy_g.ndata["feat"])
        loss = F.mse_loss(out, policy_g.ndata["feat"])
        loss.backward()
        opt.step()

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs} - loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        emb = model(policy_g, policy_g.ndata["feat"]).cpu().numpy()

    # 边向量：简单求源/目标平均
    src, dst = policy_g.edges()
    edge_emb = ((torch.tensor(emb[src.cpu()], dtype=torch.float32) + torch.tensor(emb[dst.cpu()], dtype=torch.float32)) / 2).numpy()

    out_dir = project_root / "graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
        },
        out_dir / "checkpoints" / "model.pt",
    )

    import numpy as np

    np.save(out_dir / "policy_node_emb.npy", emb)
    np.save(out_dir / "edge_emb.npy", edge_emb)

    print("✅ 训练完成并已保存向量/模型")
    print(f"- 节点向量: {out_dir / 'policy_node_emb.npy'}")
    print(f"- 边向量: {out_dir / 'edge_emb.npy'}")
    print(f"- 模型: {out_dir / 'checkpoints' / 'model.pt'}")


if __name__ == "__main__":
    main()

