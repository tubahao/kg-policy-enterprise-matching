import matplotlib.pyplot as plt
import numpy as np

# 1. 整理好的论文数据 (包含 SubG-S(10%), SubG-M(20%), SubG-L(50%), Full(100%))
x_labels = ['SubG-S\n(10%)', 'SubG-M\n(20%)', 'SubG-L\n(50%)', 'Full Graph\n(100%)']
x = np.arange(len(x_labels))

# 数据结构: data[task][metric][model] = [val_10, val_20, val_50, val_100]
# 已经将 Ours 最后的 100% 数据更新为 ablation_results_11.md 中的最新 BASE 数据
data = {
    'E->P Task': {
        'F1-Score': {
            'Naive-TFIDF': [0.046, 0.023, 0.026, 0.0096],
            'Vector RAG':  [0.060, 0.057, 0.048, 0.0181],
            'OpenKE-TransE':[0.059, 0.101, 0.063, 0.0674],
            'KG-BERT':     [0.087, 0.029, 0.088, 0.0753],
            'ATISE':       [0.068, 0.077, 0.084, 0.1226],
            'HippoRAG':    [0.060, 0.046, 0.074, 0.0133],
            'LightRAG':    [0.055, 0.128, 0.041, 0.0246],
            'Ours':        [0.137, 0.209, 0.145, 0.2466]  # Updated 100%: 0.2466
        },
        'NDCG': {
            'Naive-TFIDF': [0.062, 0.033, 0.038, 0.0162],
            'Vector RAG':  [0.066, 0.072, 0.059, 0.0233],
            'OpenKE-TransE':[0.062, 0.125, 0.084, 0.0820],
            'KG-BERT':     [0.099, 0.041, 0.102, 0.0932],
            'ATISE':       [0.084, 0.091, 0.110, 0.1641],
            'HippoRAG':    [0.075, 0.059, 0.083, 0.0167],
            'LightRAG':    [0.095, 0.161, 0.103, 0.0939],
            'Ours':        [0.166, 0.234, 0.182, 0.3943]  # Updated 100%: 0.3943
        },
        'MAP': {
            'Naive-TFIDF': [0.037, 0.019, 0.021, 0.0035],
            'Vector RAG':  [0.031, 0.046, 0.026, 0.0054],
            'OpenKE-TransE':[0.048, 0.092, 0.049, 0.0483],
            'KG-BERT':     [0.073, 0.028, 0.050, 0.0295],
            'ATISE':       [0.059, 0.066, 0.067, 0.0957],
            'HippoRAG':    [0.049, 0.041, 0.043, 0.0037],
            'LightRAG':    [0.044, 0.118, 0.027, 0.0151],
            'Ours':        [0.098, 0.156, 0.091, 0.1658]  # Updated 100%: 0.1658
        }
    },
    'P->E Task': {
        'F1-Score': {
            'Naive-TFIDF': [0.092, 0.057, 0.062, 0.0428],
            'Vector RAG':  [0.122, 0.062, 0.077, 0.0510],
            'OpenKE-TransE':[0.107, 0.121, 0.048, 0.0255],
            'KG-BERT':     [0.114, 0.043, 0.021, 0.0692],
            'ATISE':       [0.096, 0.069, 0.055, 0.0495],
            'HippoRAG':    [0.142, 0.095, 0.124, 0.1220],
            'LightRAG':    [0.007, 0.006, 0.005, 0.0026],
            'Ours':        [0.174, 0.187, 0.156, 0.2594]  # Updated 100%: 0.2594
        },
        'NDCG': {
            'Naive-TFIDF': [0.177, 0.150, 0.092, 0.0594],
            'Vector RAG':  [0.267, 0.186, 0.120, 0.0727],
            'OpenKE-TransE':[0.176, 0.244, 0.085, 0.0399],
            'KG-BERT':     [0.234, 0.073, 0.038, 0.0866],
            'ATISE':       [0.200, 0.239, 0.127, 0.0921],
            'HippoRAG':    [0.315, 0.347, 0.239, 0.2044],
            'LightRAG':    [0.048, 0.042, 0.083, 0.0920],
            'Ours':        [0.284, 0.304, 0.284, 0.6479]  # Updated 100%: 0.6479
        },
        'MAP': {
            'Naive-TFIDF': [0.052, 0.031, 0.026, 0.0069],
            'Vector RAG':  [0.098, 0.050, 0.018, 0.0098],
            'OpenKE-TransE':[0.042, 0.052, 0.018, 0.0086],
            'KG-BERT':     [0.056, 0.019, 0.010, 0.0118],
            'ATISE':       [0.050, 0.050, 0.020, 0.0126],
            'HippoRAG':    [0.120, 0.100, 0.048, 0.0393],
            'LightRAG':    [0.004, 0.003, 0.003, 0.0013],
            'Ours':        [0.075, 0.085, 0.076, 0.1689]  # Updated 100%: 0.1689
        }
    }
}

# 2. 绘图样式配置 (颜色与标记)
models = ['Naive-TFIDF', 'Vector RAG', 'OpenKE-TransE', 'KG-BERT', 'ATISE', 'HippoRAG', 'LightRAG', 'Ours']

# 我们为除了Ours之外的模型分配统一色系，Ours使用红色突显
colors = {
    'Naive-TFIDF': '#1f77b4', 'Vector RAG': '#ff7f0e', 'OpenKE-TransE': '#2ca02c',
    'KG-BERT': '#9467bd', 'ATISE': '#8c564b', 'HippoRAG': '#e377c2',
    'LightRAG': '#7f7f7f', 'Ours': '#d62728'  # 红色
}

markers = {
    'Naive-TFIDF': 'o', 'Vector RAG': 's', 'OpenKE-TransE': '^',
    'KG-BERT': 'D', 'ATISE': 'v', 'HippoRAG': 'p',
    'LightRAG': 'X', 'Ours': '*'  # 星号
}

# 3. 创建 2x3 画布
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
plt.subplots_adjust(hspace=0.35, wspace=0.25, bottom=0.15) # 给底部图例留空间

tasks = ['E->P Task', 'P->E Task']
metrics = ['F1-Score', 'NDCG', 'MAP']

# 4. 循环绘制每个子图
lines_for_legend = []
labels_for_legend = []

for i, task in enumerate(tasks):
    for j, metric in enumerate(metrics):
        ax = axes[i, j]
        
        for model in models:
            y = data[task][metric][model]
            
            # 判断是否是Ours，加大线宽和点号，让其极度醒目
            if model == 'Ours':
                lw, ms, zorder = 3.5, 14, 10
            else:
                lw, ms, zorder = 1.5, 7, 5
                
            line, = ax.plot(x, y, label=model, color=colors[model], marker=markers[model],
                            linewidth=lw, markersize=ms, zorder=zorder)
            
            # 仅在画第一个图的时候收集图例句柄
            if i == 0 and j == 0:
                lines_for_legend.append(line)
                labels_for_legend.append(model)
        
        # 子图样式设置
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_title(f"{task}: {metric}", fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 给纵坐标加上名字
        ax.set_ylabel(metric, fontsize=12)

# 5. 添加全局共享图例 (放置在整个大图的底部)
fig.legend(lines_for_legend, labels_for_legend, 
           loc='lower center', ncol=8, fontsize=13, 
           bbox_to_anchor=(0.5, 0.02), frameon=True, shadow=True)

# 6. 保存为高清学术PDF
plt.savefig('scalability_results_updated.pdf', format='pdf', bbox_inches='tight', dpi=300)
print("学术折线大图已更新并生成：scalability_results_updated.pdf")
# 如果是在notebook环境，取消下面的注释预览：
# plt.show()