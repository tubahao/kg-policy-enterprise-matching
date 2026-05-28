# 双向匹配模块使用说明

## 概述

双向匹配模块实现了两个方向的匹配功能：
1. **企业/行业 → 政策查询**：根据企业或行业关键词查询相关政策
2. **政策 → 企业检索**：根据政策ID检索相关企业

## 快速开始

### 方式1: 交互式工具（最简单）

**直接运行**（推荐）：
```bash
venv_graph\Scripts\python.exe matching\interactive_matching.py
```

**或者双击批处理文件**：
```bash
matching\run_matching.bat
```

**功能**：
- 输入查询文本查询政策（如：开饭店、餐饮、制造业等）
- 输入政策ID检索企业
- 从文件批量查询

**使用示例**：
```
请选择操作:
  1. 企业/行业 → 政策查询（输入查询文本，如：开饭店）
  2. 政策 → 企业检索（输入政策ID）
  3. 批量查询（从文件读取查询）
  0. 退出

请输入选项 (0-3): 1

请输入查询文本（如：开饭店、餐饮、制造业等）: 开饭店
返回前几个结果（默认10）: 10
```

### 方式2: 命令行工具

```bash
# 企业/行业 → 政策查询
venv_graph\Scripts\python.exe matching\bidirectional_matching.py --query "开饭店" --top_k 10

# 政策 → 企业检索
venv_graph\Scripts\python.exe matching\bidirectional_matching.py --policy_id 0 --top_k 20
```

### 方式3: Python API

在Python代码中使用：

```python
from pathlib import Path
from matching.bidirectional_matching import BidirectionalMatcher

# 初始化匹配器
project_root = Path(".")  # project root
matcher = BidirectionalMatcher(project_root)

# 企业/行业 → 政策查询
results = matcher.query_policies_by_enterprise("开饭店", top_k=10)
for policy_id, score in results:
    print(f"政策ID: {policy_id}, 相似度: {score:.6f}")

# 政策 → 企业检索
results = matcher.retrieve_enterprises_by_policy(policy_id=0, top_k=50)
for company_id, score in results:
    print(f"企业ID: {company_id}, 优先级: {score:.4f}")
```

### 方式4: 运行示例代码

```bash
venv_graph\Scripts\python.exe matching\example_usage.py
```

## 详细说明

### 1. 企业/行业 → 政策查询

**输入**：
- 查询文本：任意中文文本（如"开饭店"、"餐饮业"、"制造业"等）
- top_k：返回前k个结果（默认10）

**输出**：
- List of (policy_id, score) tuples
- score：相似度分数（0-1之间，越高越相似）

**原理**：
- 使用BERT编码查询文本
- 使用预计算的政策向量
- 通过注意力机制计算相似度：`Score = softmax(QWq(VpolicyWk)T/√dk)`

**示例**：
```python
results = matcher.query_policies_by_enterprise("开饭店", top_k=10)
# 返回: [(608, 0.0007), (91, 0.0007), ...]
```

### 2. 政策 → 企业检索

**输入**：
- policy_id：政策ID（policies_clean.parquet中的policy_id）
- top_k：返回前k个结果（默认50）
- k_hop：子图采样跳数（默认2）

**输出**：
- List of (company_id, priority_score) tuples
- priority_score：优先级分数（越高优先级越高）

**原理**：
- Step 1: 从企业图谱采样k-hop子图（PageRank选Top-k相关企业）
- Step 2: GNN编码子图结构 → 企业匹配优先级排序

**示例**：
```python
results = matcher.retrieve_enterprises_by_policy(policy_id=0, top_k=50)
# 返回: [(3, 0.85), (5, 0.82), ...]
```

## 批量查询

### 从文件批量查询

创建查询文件 `queries.txt`（每行一个查询）：
```
开饭店
餐饮业
制造业
科技创新
中小企业
```

运行批量查询：
```bash
python matching/interactive_matching.py
# 选择选项3，输入文件路径
```

### Python批量查询

```python
queries = ["开饭店", "餐饮业", "制造业"]
for query in queries:
    results = matcher.query_policies_by_enterprise(query, top_k=10)
    print(f"查询: {query}, 找到 {len(results)} 个结果")
```

## 查看政策/企业信息

### 查看政策信息

```python
import pandas as pd

df = pd.read_parquet("data_intermediate/policies_clean.parquet")
# 查看所有政策
print(df[["policy_id", "title", "year", "level"]].head(10))

# 根据policy_id查找政策
policy_id = 0
policy = df[df["policy_id"] == policy_id]
print(policy[["title", "year", "level", "content"]].values)
```

### 查看企业信息

```python
import pandas as pd

df = pd.read_parquet("data_intermediate/enterprises_filtered.parquet")
# 查看所有企业
print(df[["enterprise_id", "name", "industry"]].head(10))

# 根据enterprise_id查找企业
enterprise_id = 3
company = df[df["enterprise_id"] == enterprise_id]
print(company[["name", "industry", "scope"]].values)
```

## 常见问题

### Q1: ModuleNotFoundError: No module named 'torch'

**原因**：使用了系统的Python而不是虚拟环境中的Python。

**解决方法**：
1. **最简单**：双击运行 `matching/run_matching.bat`（会自动使用虚拟环境）
2. **命令行**：使用 `venv_graph\Scripts\python.exe` 而不是 `python`
3. **激活虚拟环境**：
   ```bash
   venv_graph\Scripts\activate.bat
   python matching/interactive_matching.py
   ```

### Q2: 初始化失败，提示找不到文件

**A**: 确保已运行以下步骤：
1. 数据预处理：`python data_clean/preprocess_policies.py`
2. 生成嵌入向量：`python embeddings/embed_policies.py` 和 `python embeddings/embed_enterprises.py`
3. 构建图数据：`python graph/build_graph.py`

### Q2: BERT模型下载失败

**A**: 
- 已设置镜像源 `HF_ENDPOINT = 'https://hf-mirror.com'`
- 如果仍然失败，可以手动下载BERT模型到本地缓存目录
- 或者使用已下载的模型（如果之前运行过embed_policies.py）

### Q3: 查询结果相似度分数很低

**A**: 
- 这是正常现象，因为政策数量多（1,650个），softmax后分数分布较均匀
- 可以关注相对排名而不是绝对分数
- 后续会优化相似度计算，提高区分度

### Q4: 政策→企业检索返回空结果

**A**: 
- 检查policy_id是否存在于图中
- 检查该政策是否有关联的企业（通过policy-entity三元组）
- 可能需要完善节点映射逻辑

### Q5: 如何查看所有可用的政策ID？

**A**: 
```python
import pandas as pd
df = pd.read_parquet("data_intermediate/policies_clean.parquet")
print(df["policy_id"].tolist()[:20])  # 查看前20个
```

## 性能说明

- **初始化时间**：首次运行需要加载BERT模型，约10-30秒
- **查询时间**：单次查询约0.1-1秒（取决于top_k）
- **内存占用**：约2-4GB（包含BERT模型和政策/企业向量）

## 文件说明

- `bidirectional_matching.py`: 核心匹配模块
- `interactive_matching.py`: 交互式工具
- `example_usage.py`: 使用示例
- `README.md`: 本说明文档

## 下一步改进

1. 完善节点映射逻辑
2. 优化相似度计算，提高区分度
3. 增强GNN编码，使用GAT结构特征
4. 添加评估指标（准确率、召回率等）
5. 性能优化（批量查询、缓存等）

