import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 设置学术风格绘图
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42

# 1. 创建图
G = nx.DiGraph()

# 2. 定义节点 (添加层级、类型、重要性评分用于映射可视化)
G.add_node("Enterprise_E1", type="Enterprise", layer=0, pos=(0, 2))

# 桥梁节点
G.add_node("Ind_Software", type="Industry", layer=1, pos=(2, 3))
G.add_node("Ind_Mfg", type="Industry", layer=1, pos=(2, 1))

# Top-K 召回政策 (右侧，添加匹配重要性评分用于映射大小)
G.add_node("Policy_P1\n(Top-1)", type="Policy", layer=2, pos=(5, 3.5), score=0.95)
G.add_node("Policy_P2\n(Top-2)", type="Policy", layer=2, pos=(5, 2.5), score=0.85)
G.add_node("Policy_P3\n(Top-3)", type="Policy", layer=2, pos=(5, 1.5), score=0.75)
G.add_node("Policy_P4\n(Top-4)", type="Policy", layer=2, pos=(5, 0.5), score=0.60)

# 3. 添加边 (修复了这里的名字拼写错误)
edges = [
    ("Enterprise_E1", "Ind_Software", "belongsTo", 0.9),
    ("Enterprise_E1", "Ind_Mfg", "belongsTo", 0.4),
    ("Policy_P1\n(Top-1)", "Ind_Software", "targetsIndustry", 0.95),
    ("Policy_P2\n(Top-2)", "Ind_Software", "targetsIndustry", 0.80),
    ("Policy_P3\n(Top-3)", "Ind_Mfg", "targetsIndustry", 0.70), # <- 修复了这里的拼写
    # 直接支持
    ("Policy_P1\n(Top-1)", "Enterprise_E1", "supports", 0.90),
    # 政策间传导
    ("Policy_P1\n(Top-1)", "Policy_P4\n(Top-4)", "transmitsTo", 0.50)
]

for u, v, rel, att in edges:
    G.add_edge(u, v, relation=rel, attention=att)

# 4. 获取位置
pos = nx.get_node_attributes(G, 'pos')

# 5. 视觉映射配置
color_map = {"Enterprise": "#E45756", "Policy": "#4C78A8", "Industry": "#72B7B2"}
# 使用 .get() 增强健壮性，防止异常节点
node_colors = [color_map.get(G.nodes[n].get("type", "Policy"), "#CCCCCC") for n in G.nodes()]
node_sizes = [G.nodes[n].get("score", 0.5) * 4000 for n in G.nodes()] # 政策节点大小映射分数

edge_widths = [G[u][v]["attention"] * 5 for u, v in G.edges()]
edge_alphas = [max(0.3, G[u][v]["attention"]) for u, v in G.edges()]

# 6. 开始绘图
fig, ax = plt.subplots(figsize=(12, 7))

# 绘制边 (使用 FancyArrowPatch 增加高级感)
for i, (u, v) in enumerate(G.edges()):
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                           width=edge_widths[i], alpha=edge_alphas[i], 
                           edge_color="gray", connectionstyle="arc3,rad=0.1", 
                           arrowsize=20, ax=ax)

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                       edgecolors="black", linewidths=1.5, ax=ax)

# 绘制标签 (稍微偏移防止重叠，加上底色增加可读性)
for node, (x, y) in pos.items():
    ax.text(x, y - 0.25, node, fontsize=11, ha='center', va='top', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

# 绘制边标签
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color="#555555",
                             bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))

# 7. 添加学术图例
enterprise_patch = mpatches.Patch(color="#E45756", label='Enterprise (Query)')
policy_patch = mpatches.Patch(color="#4C78A8", label='Policy (Top-K Target)')
industry_patch = mpatches.Patch(color="#72B7B2", label='Industry (Bridge)')

# 说明映射含义
score_size_legend = ax.scatter([], [], s=0.9*4000, color='gray', edgecolors='black', label='Node Size ≈ Spatio-temporal Importance')
att_width_legend, = ax.plot([], [], '-', color='gray', linewidth=4, label='Edge Width ≈ GAT Attention Weight')

ax.legend(handles=[enterprise_patch, policy_patch, industry_patch, score_size_legend, att_width_legend], 
          loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, frameon=False)

ax.set_title("Interpretability of Top-4 Policy Retrieval via Unified Representation & Decay Modeling", fontsize=15, fontweight='bold', pad=25)
ax.axis("off")

# 保存为 PDF
plt.tight_layout()
plt.savefig("fig4_subgraph_case_study.pdf", format="pdf", dpi=300, bbox_inches="tight")
print("Top-K 子图可视化已生成：fig4_subgraph_case_study.pdf")