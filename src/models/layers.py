#!/usr/bin/env python3
"""Session 4 — Step 4: 置信度感知异构图注意力层 (Confidence-Aware HeteroGAT)

核心改进:
    1. 从 DGL 迁移至 PyG (torch_geometric)，与 Session 3 HeteroData 兼容
    2. targetsSubIndustry 边使用 LLM confidence 分数作为边权重 (edge_attr)
    3. Jumping Knowledge 残差连接 + L2 归一化输出
    4. 严格仅使用合法的 Message Edges — 永不触及 supports 边

边类型:
    Message Edges (前向传播):
        transmitsTo          Policy→Policy        层级行政传导
        targetsSubIndustry   Policy→SubIndustry   LLM 行业靶向 (含 confidence)
        belongsTo           Enterprise→SubIndustry 确定性
        subClassOf          SubIndustry→MajorIndustry 确定性
    所有边通过 ToUndirected 自动添加反向边，实现双向消息传递。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.nn.conv.hetero_conv import group


# ---------------------------------------------------------------------------
# Confidence-Aware HeteroGAT
# ---------------------------------------------------------------------------

class ConfidenceAwareHeteroGAT(nn.Module):
    """置信度感知的异构图注意力网络。

    - 对 targetsSubIndustry 边使用 LLM confidence 分数作为边特征
    - Jumping Knowledge: 拼接输入投影与末层 GAT 输出
    - 输出 L2 归一化嵌入
    """

    def __init__(
        self,
        in_channels: Dict[str, int],
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.node_types = list(in_channels.keys())

        # —— 输入投影 (per node type) ——
        self.input_projs = nn.ModuleDict()
        for ntype, in_dim in in_channels.items():
            self.input_projs[ntype] = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # —— GAT 卷积层 ——
        if edge_types is None:
            edge_types = [
                ("Policy", "transmitsTo", "Policy"),
                ("Policy", "targetsSubIndustry", "SubIndustry"),
                ("Enterprise", "belongsTo", "SubIndustry"),
                ("SubIndustry", "subClassOf", "MajorIndustry"),
            ]

        self.edge_types = edge_types
        # 哪些边类型需要 confidence (edge_dim=1)
        self._conf_etypes: set = {et for et in edge_types if et[1] == "targetsSubIndustry"}

        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_ch = hidden_channels if layer_idx == 0 else hidden_channels * num_heads
            conv_dict = {}
            for etype in edge_types:
                edge_dim = 1 if etype in self._conf_etypes else None
                conv_dict[etype] = GATConv(
                    in_ch,
                    hidden_channels,
                    num_heads,
                    edge_dim=edge_dim,
                    concat=(layer_idx < num_layers - 1),
                    dropout=dropout,
                    add_self_loops=False,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

        # —— Jumping Knowledge 输出投影 ——
        jk_in_dim = hidden_channels + hidden_channels * num_heads
        self.output_projs = nn.ModuleDict()
        for ntype in self.node_types:
            self.output_projs[ntype] = nn.Sequential(
                nn.Linear(jk_in_dim, out_channels),
                nn.LayerNorm(out_channels),
            )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向传播。

        Args:
            x_dict: {node_type: [N_t, in_channels[t]]} 节点特征字典。
            edge_index_dict: {edge_type: [2, E]} 边索引字典。
            edge_attr_dict: {edge_type: [E, edge_dim]} 边特征字典。
                            targetsSubIndustry 边必须提供 confidence。

        Returns:
            {node_type: [N_t, out_channels]} L2 归一化嵌入字典。
        """
        # 1) 输入投影
        h = {}
        for ntype in self.node_types:
            if ntype in x_dict and ntype in self.input_projs:
                h[ntype] = self.input_projs[ntype](x_dict[ntype])
            else:
                # 无特征节点 — 零初始化
                num_nodes = 0
                for et, ei in edge_index_dict.items():
                    if et[0] == ntype:
                        num_nodes = max(num_nodes, int(ei[0].max()) + 1)
                    if et[2] == ntype:
                        num_nodes = max(num_nodes, int(ei[1].max()) + 1)
                if num_nodes == 0:
                    num_nodes = 1
                h[ntype] = torch.zeros(num_nodes, self.hidden_channels, device=self._device)

        h_initial = {k: v.clone() for k, v in h.items()}

        # 2) GAT 消息传递
        for conv in self.convs:
            h = conv(h, edge_index_dict, edge_attr_dict=edge_attr_dict)

        # 3) Jumping Knowledge: concat(initial, final_gat) → project → L2-norm
        out = {}
        for ntype in self.node_types:
            if ntype not in h or ntype not in self.output_projs:
                continue
            h_init = h_initial.get(ntype)
            h_gat = h[ntype]
            if h_init is not None and h_init.shape[0] == h_gat.shape[0]:
                combined = torch.cat([h_init, h_gat], dim=-1)
            else:
                pad = torch.zeros(h_gat.shape[0], self.hidden_channels, device=self._device)
                combined = torch.cat([pad, h_gat], dim=-1)
            out[ntype] = self.output_projs[ntype](combined)
            out[ntype] = F.normalize(out[ntype], p=2, dim=-1)

        return out

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# Feature loading helpers
# ---------------------------------------------------------------------------

def load_node_features(
    graph_path: str,
    text_emb_path: str,
    enterprises_path: str,
    temporal_emb_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """从磁盘加载并组装节点特征张量。

    Policy:  text_emb [768] + level [1]  →  769-dim
    Enterprise: static_feat [2] + temporal_emb [64] → 66-dim
    SubIndustry:  62-dim one-hot (从图读取)
    MajorIndustry: 6-dim one-hot (从图读取)

    Returns:
        {node_type: [num_nodes, feat_dim]} 字典。
    """
    import json

    g = torch.load(graph_path, weights_only=False)

    features: Dict[str, torch.Tensor] = {}

    # --- Policy: text embedding + level ---
    policy_emb = torch.load(text_emb_path, weights_only=False)  # [N_policy, 768]
    num_policy = g["Policy"].num_nodes

    # level 从 graph 的 Policy 节点属性读取
    level_arr = g["Policy"].level.numpy() if hasattr(g["Policy"], "level") else None
    if level_arr is not None:
        level_t = torch.tensor(level_arr, dtype=torch.float32).unsqueeze(-1)  # [N, 1]
    else:
        level_t = torch.zeros(num_policy, 1)

    if policy_emb.shape[0] >= num_policy:
        policy_emb = policy_emb[:num_policy]
    else:
        pad = torch.zeros(num_policy - policy_emb.shape[0], policy_emb.shape[1])
        policy_emb = torch.cat([policy_emb, pad], dim=0)

    policy_feat = torch.cat([policy_emb, level_t], dim=-1)
    features["Policy"] = policy_feat.to(device)

    # --- Enterprise: static + temporal ---
    num_ent = g["Enterprise"].num_nodes
    static = torch.tensor(
        g["Enterprise"].static_feat.numpy(), dtype=torch.float32
    )  # [N, 2]

    if temporal_emb_path is not None:
        temporal = torch.load(temporal_emb_path, weights_only=False)  # [N, 64]
        if temporal.shape[0] >= num_ent:
            temporal = temporal[:num_ent]
        else:
            pad = torch.zeros(num_ent - temporal.shape[0], temporal.shape[1])
            temporal = torch.cat([temporal, pad], dim=0)
    else:
        temporal = torch.zeros(num_ent, 64)

    ent_feat = torch.cat([static, temporal], dim=-1)
    features["Enterprise"] = ent_feat.to(device)

    # --- SubIndustry & MajorIndustry: one-hot ---
    for nt in ["SubIndustry", "MajorIndustry"]:
        num_n = g[nt].num_nodes
        x = torch.tensor(g[nt].x.numpy(), dtype=torch.float32)  # one-hot
        features[nt] = x.to(device)

    return features


def build_edge_attr_dict(
    g,
    edge_types: List[Tuple[str, str, str]],
    device: torch.device = torch.device("cpu"),
) -> Dict[Tuple, torch.Tensor]:
    """为 targetsSubIndustry 边构建 edge_attr 字典。

    仅 targetsSubIndustry 边携带 confidence 作为 edge_attr [E, 1]。
    其他边类型不提供 edge_attr (PyG GATConv 默认按 degree 计算)。
    """
    edge_attr_dict: Dict[Tuple, torch.Tensor] = {}
    for et in edge_types:
        if et[1] == "targetsSubIndustry" and et in g.edge_types:
            conf = g[et].confidence
            if conf is not None:
                edge_attr_dict[et] = conf.unsqueeze(-1).float().to(device)
    return edge_attr_dict
