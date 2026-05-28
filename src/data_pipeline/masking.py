#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据中间文件构建异构图并保存：
- 输入：
    data_intermediate/policies_clean.parquet
    data_intermediate/triples_policy_policy.parquet
    data_intermediate/triples_policy_entity.parquet
    features/policy_feature_fused.npy
- 输出：
    graph/graph_data.bin
    graph/meta.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import dgl
import numpy as np
import pandas as pd
import torch


def build_node_maps(df_policies: pd.DataFrame, df_entities: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    node_maps: Dict[str, Dict[str, int]] = defaultdict(dict)

    # 政策节点：先从policies_clean.parquet中获取（使用policy_id）
    max_policy_id = -1
    for _, row in df_policies.iterrows():
        policy_id = int(row["policy_id"])
        node_maps["policy"][str(row["title"])] = policy_id
        max_policy_id = max(max_policy_id, policy_id)

    # 从policy-entity三元组中提取所有政策节点（合并），过滤掉entity节点
    next_policy_id = max_policy_id + 1
    for _, row in df_entities.iterrows():
        s_name, s_type = str(row["subject"]), row["subject_type"]
        o_name, o_type = str(row["object"]), row["object_type"]

        # 跳过entity节点
        if s_type == "entity" or o_type == "entity":
            continue

        # 处理subject
        if s_type == "policy" and s_name not in node_maps["policy"]:
            node_maps["policy"][s_name] = next_policy_id
            next_policy_id += 1
        elif s_type != "policy" and s_name not in node_maps[s_type]:
            if s_type not in node_maps:
                node_maps[s_type] = {}
            node_maps[s_type][s_name] = len(node_maps[s_type])
        
        # 处理object
        if o_type == "policy" and o_name not in node_maps["policy"]:
            node_maps["policy"][o_name] = next_policy_id
            next_policy_id += 1
        elif o_type != "policy" and o_name not in node_maps[o_type]:
            if o_type not in node_maps:
                node_maps[o_type] = {}
            node_maps[o_type][o_name] = len(node_maps[o_type])
    
    return node_maps


def build_edges(
    df_p2p: pd.DataFrame,
    df_p2e: pd.DataFrame,
    node_maps: Dict[str, Dict[str, int]],
) -> Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]]:
    edges = {}
    reverse_rel_map = {
        "transmitsTo": "transmitsFrom",
        "belongsTo": "includesCompany",
        "targetsIndustry": "targetedByPolicy",
        "supports": "supportedByPolicy",
    }

    # 政策-政策
    src = []
    dst = []
    for _, row in df_p2p.iterrows():
        src.append(int(row["head_id"]))
        dst.append(int(row["tail_id"]))
    edges[("policy", "transmitsTo", "policy")] = (
        torch.tensor(src, dtype=torch.int64),
        torch.tensor(dst, dtype=torch.int64),
    )
    # 反向边：弱化单向信息偏置
    edges[("policy", "transmitsFrom", "policy")] = (
        torch.tensor(dst, dtype=torch.int64),
        torch.tensor(src, dtype=torch.int64),
    )

    # 政策-企业/行业等（过滤掉entity节点）
    grouped = defaultdict(lambda: ([], []))
    for _, row in df_p2e.iterrows():
        s_name, o_name = str(row["subject"]), str(row["object"])
        s_type, o_type = row["subject_type"], row["object_type"]
        rel = row["predicate"]

        # 跳过包含entity的边
        if s_type == "entity" or o_type == "entity":
            continue

        # subject
        if s_type == "policy":
            s_id = node_maps["policy"].get(s_name)
        else:
            s_id = node_maps[s_type].get(s_name)
        # object
        if o_type == "policy":
            o_id = node_maps["policy"].get(o_name)
        else:
            o_id = node_maps[o_type].get(o_name)

        if s_id is None or o_id is None:
            continue

        key = (s_type, rel, o_type)
        grouped[key][0].append(s_id)
        grouped[key][1].append(o_id)

        # 同步构建反向边，保证消息传递对称
        rev_rel = reverse_rel_map.get(str(rel), f"rev_{rel}")
        rev_key = (o_type, rev_rel, s_type)
        grouped[rev_key][0].append(o_id)
        grouped[rev_key][1].append(s_id)

    for key, (s_list, o_list) in grouped.items():
        edges[key] = (
            torch.tensor(s_list, dtype=torch.int64),
            torch.tensor(o_list, dtype=torch.int64),
        )

    # 为主要异构关系补充反向边，提升政策/企业信息流对称性
    reverse_rel = {
        ("company", "belongsTo", "industry"): ("industry", "includesCompany", "company"),
        ("policy", "targetsIndustry", "industry"): ("industry", "targetedByPolicy", "policy"),
        ("policy", "supports", "company"): ("company", "supportedByPolicy", "policy"),
    }
    for fwd_key, rev_key in reverse_rel.items():
        if fwd_key in edges:
            s_tensor, o_tensor = edges[fwd_key]
            edges[rev_key] = (
                o_tensor.clone(),
                s_tensor.clone(),
            )

    return edges


def main():
    parser = argparse.ArgumentParser(description="构建异构图并保存")
    parser.add_argument("--policies", type=str, default="data_intermediate/policies_clean.parquet")
    parser.add_argument("--p2p", type=str, default="data_intermediate/triples_policy_policy.parquet")
    parser.add_argument("--p2e", type=str, default="data_intermediate/triples_policy_entity.parquet")
    parser.add_argument("--feat", type=str, default="features/policy_feature_fused.npy")
    parser.add_argument("--out", type=str, default="graph/graph_data.bin")
    parser.add_argument("--meta", type=str, default="graph/meta.json")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    df_policies = pd.read_parquet(project_root / args.policies)
    df_p2p = pd.read_parquet(project_root / args.p2p)
    df_p2e = pd.read_parquet(project_root / args.p2e)
    policy_feat = np.load(project_root / args.feat)

    node_maps = build_node_maps(df_policies, df_p2e)
    edges = build_edges(df_p2p, df_p2e, node_maps)

    num_nodes = {ntype: len(mapping) for ntype, mapping in node_maps.items()}
    graph = dgl.heterograph(edges, num_nodes_dict=num_nodes)

    # 为政策节点赋特征
    policy_node_count = num_nodes.get("policy", 0)
    feat_dim = policy_feat.shape[1] if len(policy_feat.shape) > 1 else policy_feat.shape[0]
    
    if policy_feat.shape[0] >= policy_node_count:
        # 如果特征数量大于等于节点数量，只取前policy_node_count个
        graph.nodes["policy"].data["feat"] = torch.tensor(policy_feat[:policy_node_count], dtype=torch.float32)
    else:
        # 如果特征数量少于节点数量，为剩余节点创建零向量
        print(f"警告: 特征数量({policy_feat.shape[0]})少于政策节点数({policy_node_count})，为剩余节点创建零向量")
        policy_feat_tensor = torch.tensor(policy_feat, dtype=torch.float32)
        if len(policy_feat.shape) == 1:
            # 1D特征，扩展为2D
            policy_feat_tensor = policy_feat_tensor.unsqueeze(1)
        zero_feat = torch.zeros((policy_node_count - policy_feat.shape[0], policy_feat_tensor.shape[1]), dtype=torch.float32)
        full_feat = torch.cat([policy_feat_tensor, zero_feat], dim=0)
        graph.nodes["policy"].data["feat"] = full_feat

    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(str(out_path), [graph])

    meta = {
        "policies_path": str(project_root / args.policies),
        "p2p_path": str(project_root / args.p2p),
        "p2e_path": str(project_root / args.p2e),
        "feature_path": str(project_root / args.feat),
        "graph_path": str(out_path),
        "node_maps": {k: v for k, v in node_maps.items()},
        "num_nodes": num_nodes,
        "edge_types": [str(k) for k in edges.keys()],
    }

    meta_path = project_root / args.meta
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("图构建完成")
    print(f"- 图文件: {out_path}")
    print(f"- 元信息: {meta_path}")
    print(f"- 节点数: {num_nodes}")


if __name__ == "__main__":
    main()

