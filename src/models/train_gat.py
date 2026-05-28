#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练异构图GAT模型，学习节点的结构表示
"""

import json
import sys
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, GATConv

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


class HeteroGAT(nn.Module):
    def __init__(self, in_dims, hidden_dim, out_dim, num_heads=8, num_layers=2):
        super(HeteroGAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        # 为每种节点类型创建输入投影层
        self.input_projs = nn.ModuleDict()
        for ntype, in_dim in in_dims.items():
            self.input_projs[ntype] = nn.Linear(in_dim, hidden_dim)
        
        # 构建GAT层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim * num_heads
            
            # 为每种边类型创建GAT层
            conv_dict = {}
            for etype in ['transmitsTo', 'belongsTo', 'targetsIndustry', 'supports']:
                conv_dict[etype] = GATConv(
                    in_dim, hidden_dim, num_heads, 
                    feat_drop=0.1, attn_drop=0.1, 
                    activation=F.elu if i < num_layers - 1 else None
                )
            
            self.layers.append(HeteroGraphConv(conv_dict, aggregate='mean'))
        
        # 输出投影层
        self.output_projs = nn.ModuleDict()
        for ntype in in_dims.keys():
            self.output_projs[ntype] = nn.Linear(hidden_dim * num_heads, out_dim)
    
    def forward(self, g, node_features):
        # 初始化节点特征
        h = {}
        for ntype in g.ntypes:
            if ntype in node_features:
                # 如果有初始特征，使用投影层
                h[ntype] = self.input_projs[ntype](node_features[ntype])
            else:
                # 如果没有初始特征，创建零向量
                num_nodes = g.number_of_nodes(ntype)
                h[ntype] = torch.zeros(num_nodes, self.hidden_dim, device=node_features[list(node_features.keys())[0]].device)
        
        # 通过GAT层
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            # 展平多头输出
            h = {k: v.flatten(1) if len(v.shape) > 2 else v for k, v in h.items()}
        
        # 输出投影
        out = {}
        for ntype in g.ntypes:
            if ntype in h:
                out[ntype] = self.output_projs[ntype](h[ntype])
            else:
                num_nodes = g.number_of_nodes(ntype)
                out[ntype] = torch.zeros(num_nodes, self.out_dim, device=list(h.values())[0].device)
        
        return out


def load_features(project_root):
    """加载节点特征"""
    features = {}
    
    # 政策特征
    policy_feat_path = project_root / "features" / "policy_feature_fused.npy"
    if policy_feat_path.exists():
        policy_feat = np.load(policy_feat_path)
        features["policy"] = torch.tensor(policy_feat, dtype=torch.float32)
        print(f"加载政策特征: {policy_feat.shape}")
    
    # 企业特征
    enterprise_feat_path = project_root / "features" / "enterprise_feature_fused.npy"
    if enterprise_feat_path.exists():
        enterprise_feat = np.load(enterprise_feat_path)
        features["company"] = torch.tensor(enterprise_feat, dtype=torch.float32)
        print(f"加载企业特征: {enterprise_feat.shape}")
    
    # 行业特征
    industry_feat_path = project_root / "embeddings" / "industry_text_emb.npy"
    if industry_feat_path.exists():
        industry_feat = np.load(industry_feat_path)
        features["industry"] = torch.tensor(industry_feat, dtype=torch.float32)
        print(f"加载行业特征: {industry_feat.shape}")
    
    return features


def main():
    project_root = Path(__file__).resolve().parents[1]
    graph_path = project_root / "graph" / "graph_data.bin"
    meta_path = project_root / "graph" / "meta.json"
    
    print("加载图...")
    graphs, _ = dgl.load_graphs(str(graph_path))
    graph = graphs[0]
    
    print(f"图节点类型: {graph.ntypes}")
    print(f"图边类型: {graph.canonical_etypes}")
    for ntype in graph.ntypes:
        print(f"  {ntype}: {graph.number_of_nodes(ntype)} 个节点")
    
    print("\n加载节点特征...")
    node_features = load_features(project_root)
    
    # 确保特征维度匹配
    for ntype in graph.ntypes:
        if ntype in node_features:
            num_nodes = graph.number_of_nodes(ntype)
            feat_nodes = node_features[ntype].shape[0]
            if feat_nodes < num_nodes:
                # 如果特征数量少于节点数量，用零向量填充
                print(f"警告: {ntype}节点特征数量({feat_nodes})少于节点数({num_nodes})，用零向量填充")
                zero_feat = torch.zeros(num_nodes - feat_nodes, node_features[ntype].shape[1], dtype=torch.float32)
                node_features[ntype] = torch.cat([node_features[ntype], zero_feat], dim=0)
            elif feat_nodes > num_nodes:
                # 如果特征数量多于节点数量，只取前num_nodes个
                print(f"警告: {ntype}节点特征数量({feat_nodes})多于节点数({num_nodes})，只取前{num_nodes}个")
                node_features[ntype] = node_features[ntype][:num_nodes]
    
    # 获取输入维度
    in_dims = {}
    for ntype in graph.ntypes:
        if ntype in node_features:
            in_dims[ntype] = node_features[ntype].shape[1]
        else:
            # 如果没有特征，使用隐藏维度作为输入维度
            in_dims[ntype] = 64
    
    print(f"\n输入维度: {in_dims}")
    
    # 创建模型
    hidden_dim = 64
    out_dim = 64
    num_heads = 8
    num_layers = 2
    
    model = HeteroGAT(in_dims, hidden_dim, out_dim, num_heads, num_layers)
    
    print(f"\n模型参数:")
    print(f"  隐藏维度: {hidden_dim}")
    print(f"  输出维度: {out_dim}")
    print(f"  注意力头数: {num_heads}")
    print(f"  层数: {num_layers}")
    
    # 训练（自监督学习，使用重构损失）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    
    print(f"\n开始训练（{num_epochs}轮）...")
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 前向传播
        output = model(graph, node_features)
        
        # 计算损失（使用输出特征的L2正则化 + 如果有初始特征，计算相似度损失）
        loss = 0
        for ntype in graph.ntypes:
            if ntype in output:
                # L2正则化
                loss += 0.01 * torch.mean(output[ntype] ** 2)
                
                # 如果有初始特征，计算相似度损失（使用余弦相似度）
                if ntype in node_features:
                    # 将输出特征投影到输入特征空间进行比较
                    output_feat = output[ntype]
                    input_feat = node_features[ntype]
                    
                    # 归一化
                    output_norm = F.normalize(output_feat, p=2, dim=1)
                    input_norm = F.normalize(input_feat, p=2, dim=1)
                    
                    # 计算相似度（使用点积）
                    # 为了匹配维度，我们需要投影
                    if output_feat.shape[1] != input_feat.shape[1]:
                        # 使用简单的线性投影
                        proj_weight = torch.randn(output_feat.shape[1], input_feat.shape[1], 
                                                  device=output_feat.device, requires_grad=False) * 0.01
                        output_proj = torch.matmul(output_norm, proj_weight)
                        output_proj = F.normalize(output_proj, p=2, dim=1)
                    else:
                        output_proj = output_norm
                    
                    # 计算相似度损失（最大化相似度）
                    similarity = torch.sum(output_proj * input_norm, dim=1)
                    loss += 0.1 * (1 - torch.mean(similarity))
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    print("\n训练完成！")
    
    # 保存模型和特征
    model.eval()
    with torch.no_grad():
        output = model(graph, node_features)
    
    # 保存结构特征
    output_dir = project_root / "graph"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for ntype in output.keys():
        feat_np = output[ntype].cpu().numpy()
        output_path = output_dir / f"gat_{ntype}_emb.npy"
        np.save(output_path, feat_np)
        print(f"保存{ntype}节点结构特征: {feat_np.shape} -> {output_path}")
    
    # 保存模型
    model_path = output_dir / "checkpoints" / "gat_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"保存模型: {model_path}")
    
    # 保存节点类型映射
    node_type_map = {}
    offset = 0
    for ntype in graph.ntypes:
        num_nodes = graph.number_of_nodes(ntype)
        node_type_map[ntype] = [offset, offset + num_nodes]
        offset += num_nodes
    
    map_path = output_dir / "gat_node_type_map.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(node_type_map, f, ensure_ascii=False, indent=2)
    print(f"保存节点类型映射: {map_path}")
    
    print("\n所有文件保存完成！")


if __name__ == "__main__":
    main()

