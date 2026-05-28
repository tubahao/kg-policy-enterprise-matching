import matplotlib.pyplot as plt
import numpy as np

# ----------------- 1. 数据准备 -----------------
# 评估指标
labels = ['P', 'R', 'F1', 'MAP', 'NDCG']
num_vars = len(labels)

# 挑选代表性的消融组数据 (E->P Task)
# 必须将第一个数据点复制到列表末尾，才能在雷达图中形成闭合的多边形
ours = [0.1732, 0.4383, 0.2466, 0.1658, 0.3943]
ours += ours[:1]

sep_encode = [0.1894, 0.4456, 0.2648, 0.1003, 0.3163]
sep_encode += sep_encode[:1]

wo_gnn = [0.1714, 0.4057, 0.2396, 0.1370, 0.3628]
wo_gnn += wo_gnn[:1]

title_only = [0.1044, 0.2423, 0.1451, 0.0644, 0.1845]
title_only += title_only[:1]

# 计算每个轴的角度
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# ----------------- 2. 初始化极坐标图表 -----------------
plt.rcParams['font.family'] = 'sans-serif'
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)

# 旋转图表，让第一个指标 (P) 位于正上方
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# ----------------- 3. 绘制坐标轴和网格 -----------------
# 设置轴标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=13, fontweight='bold', color='#333333')

# 设置径向刻度 (因为最大值 R 接近 0.48，将外圈设为 0.5 留白刚刚好)
ax.set_ylim(0, 0.5)
ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5'], color="grey", size=10)

# 网格线美化
ax.grid(color='#d3d3d3', linestyle='--', linewidth=1)
ax.spines['polar'].set_color('#cccccc')

# ----------------- 4. 绘制每个变体组 -----------------
# 1. Ours (Full Model) - 实线 + 较深填充色
ax.plot(angles, ours, color='#4A6FE3', linewidth=2.5, label='Ours (Full Model)')
ax.fill(angles, ours, color='#4A6FE3', alpha=0.25)

# 2. Variant C1 (w/o GNN) - 虚线
ax.plot(angles, wo_gnn, color='#2CA02C', linewidth=2, linestyle='--', label='w/o GNN')
ax.fill(angles, wo_gnn, color='#2CA02C', alpha=0.1)

# 3. Variant A1 (Sep. Encode) - 点划线
ax.plot(angles, sep_encode, color='#E377C2', linewidth=2, linestyle='-.', label='Sep. Encode')
ax.fill(angles, sep_encode, color='#E377C2', alpha=0.1)

# 4. Variant A2 (Title Only) - 点线，内部面积最小
ax.plot(angles, title_only, color='#FF7F0E', linewidth=2, linestyle=':', label='Title Only')
ax.fill(angles, title_only, color='#FF7F0E', alpha=0.15)

# ----------------- 5. 图例与标题设定 -----------------
# 调整图例到图表外部，防止遮挡数据
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=11, frameon=True, edgecolor='#d3d3d3')

plt.title('Ablation Study: Key Components (E→P Task)', size=15, fontweight='bold', y=1.1)

# 导出为高清 PDF 格式 (论文排版必备)
plt.savefig('ablation_radar.pdf', format='pdf', bbox_inches='tight')

# plt.show()