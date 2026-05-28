# Project Status Log

> 贯穿项目的核心日志，由开发者与 Claude Code 共同维护。
> 每次重大变更后更新日期和内容。

**Last Updated:** 2026-05-29 (Session 1 完成)

---

## 1. 项目概述

**论文题目:** Multimodal Heterogeneous Graph Learning with Spatio-Temporal Importance Modeling for Policy-Enterprise Bidirectional Matching and Implementation Evaluation

**投稿目标:** ICDM (IEEE International Conference on Data Mining)

**核心重构方向:**
- 去除多模态/RAG泡沫，重新定位为 **"面向政策-企业双向匹配的属性异构图学习"** (Attributed Heterogeneous Graph Learning for Policy-Enterprise Bidirectional Matching)
- 彻底根治数据泄露问题（Train/Val/Test 严格按边掩码切分）
- 提升三元组提取严谨性（本体拓扑校验 + 黄金标准集验证）
- 简化评估指标（去除 Boost 项，只保留 NDCG/Recall/MAP）

---

## 2. 重构 Session 计划

| Session | 主题 | 状态 |
|---------|------|------|
| **Session 1** | 目录清洗与规范化 | **Done** |
| Session 2 | 三元组提取重构 (extraction/) | Pending |
| Session 3 | 数据管线重构 (data_pipeline/) | Pending |
| Session 4 | 模型层重构 (models/) | Pending |
| Session 5 | 评估体系重构 (evaluation/) + 实验脚本 | Pending |

---

## 3. 当前项目架构 (Session 1 完成态)

```
Kg/
├── PROJECT_STATUS.md
├── requirements.txt
├── config.toml
├── README.md
├── pyproject.toml
├── industry_mapping_complete.json
│
├── data/
│   ├── raw/                   # 原始政策PDF、企业Excel
│   │   ├── pdfs/              # 政策PDF文件 (gitignored)
│   │   ├── enterprises-source/# 原始企业数据 (gitignored)
│   │   └── kg-json/           # 原始KG JSON (test_data.json等)
│   ├── processed/             # LLM抽取后的结构化数据
│   │   └── enterprises/       # 企业处理结果 (gitignored)
│   ├── comparison-data/       # 对比基线数据集
│   │   ├── atise_policykg/    # ATISE格式
│   │   ├── kgbert_policykg/   # KG-BERT格式
│   │   └── openke_policykg/   # OpenKE格式
│   ├── intermediate/          # 管道中间产物 (gitignored)
│   └── statistics/            # 数据集统计结果 (待生成)
│
├── src/
│   ├── extraction/            # [Session 2] 三元组抽取
│   │   ├── generator.py       # KG流水线入口 (from main.py)
│   │   ├── verifier.py        # 实体标准化 (from entity_standardization.py)
│   │   ├── prompts.py         # LLM提示模板
│   │   ├── llm.py             # LLM API集成
│   │   ├── config.py          # 抽取配置
│   │   ├── text_utils.py      # 文本处理工具
│   │   ├── generate_graph.py  # 图生成脚本
│   │   ├── templates/         # HTML模板
│   │   └── ontology/          # 领域本体JSON (5个)
│   │
│   ├── data_pipeline/         # [Session 3] 数据管线
│   │   ├── loader.py          # 三元组清洗 (from clean_triples.py)
│   │   ├── masking.py         # DGL图构建 (from build_graph.py)
│   │   ├── filter_triples.py  # 企业三元组筛选
│   │   ├── preprocess_policies.py    # 政策预处理
│   │   ├── process_enterprise_data.py# 企业数据处理
│   │   └── *_index.json       # 嵌入索引元数据 (4个)
│   │
│   ├── models/                # [Session 4] 模型层
│   │   ├── encoders.py        # 政策BERT编码 (from embed_policies.py)
│   │   ├── layers.py          # 对比GAT训练 (from train_gat_contrastive.py)
│   │   ├── propagation.py     # PPR传播+衰减 (from graphrag_pipeline.py)
│   │   ├── attribute_embeddings.py    # 类别属性嵌入
│   │   ├── time_embeddings.py         # 时间序列嵌入
│   │   ├── fuse_features.py           # 通用特征融合
│   │   ├── fuse_enterprise_features.py# 企业特征融合
│   │   ├── embed_enterprises.py       # 企业文本嵌入
│   │   ├── train_gat.py               # 基础GAT训练
│   │   ├── train_gnn.py               # 基础GNN训练
│   │   └── *_meta.json                # 特征/图元数据 (8个)
│   │
│   └── evaluation/            # [Session 5] 评估
│       ├── metrics.py         # 匹配评估 (from evaluate_matching.py)
│       ├── tei_analyser.py    # 传导效率 (from transmission_efficiency.py)
│       ├── bidirectional_matching.py  # 核心双向匹配
│       ├── policy_importance.py       # 策略重要性评分
│       ├── policy_importance_with_decay.py  # 带衰减的重要性
│       ├── industry_coverage.py       # 行业覆盖率
│       ├── experiment_profiles.py     # 实验配置
│       ├── gat_importance_defaults.py # GAT重要性默认值
│       ├── policy_embedding_defaults.py# 策略嵌入默认值
│       ├── visualization.py           # 图可视化
│       └── ...                        # 其余辅助模块
│
├── experiments/               # 实验运行脚本
│   ├── run_main.py            # 主实验 (from run_a2_joint_full_pipeline.py)
│   ├── run_ablation.py        # 消融实验 (from run_ablation_11.py)
│   ├── run_real_comparison.py # 对比基线编排
│   ├── real_comparison_*.py   # 各基线实现 (14个)
│   ├── grid_search_*.py       # 超参数网格搜索 (3个)
│   ├── tune_*.py              # 参数调优 (2个)
│   ├── data_scale_*.py        # 数据扩展实验 (4个)
│   ├── run_*.py               # 其余实验变体 (5个)
│   ├── visualize_*.py         # 论文图表生成 (2个)
│   ├── compute_*.py           # 案例研究计算
│   ├── export_*.py            # 报告导出
│   ├── setup_env.ps1          # 环境配置
│   └── run_industry_*.ps1     # 行业子图管道 (2个)
│
├── third_party/               # 对比基线 (不纳入Session重构范围)
│   ├── HippoRAG/
│   ├── KG-BERT/
│   ├── LightRAG/
│   ├── atise/
│   └── openkg/
│
└── paper/                     # 论文LaTeX源 (ACM格式)
```

---

## 4. Session 1 变更清单

### 4.1 删除的中间产物 (~5.6GB)

| 目录/文件 | 说明 |
|-----------|------|
| `results/` (3.8GB) | 中间实验结果JSON/CSV/log |
| `outputs/` (1.7GB) | 管道输出三元组/映射 |
| `graphrag/` (18MB) | GraphRAG中间重要性数据 |
| `supplementary/` (24MB) | 补充材料 |
| `matching/` | 迁移至 src/evaluation/ |
| `evaluation/` | 迁移至 src/evaluation/ + src/models/ |
| `scripts/` | 迁移至 experiments/ |
| `data_clean/` | 迁移至 src/data_pipeline/ |
| `embeddings/` | 迁移至 src/models/ + src/data_pipeline/ |
| `features/` | 迁移至 src/models/ |
| `graph/` | 迁移至 src/models/ + src/data_pipeline/ |
| `ontology/` | 迁移至 src/extraction/ontology/ |
| `docs/` | 旧文档删除 |
| `train_session_*.txt` | 训练日志 |

### 4.2 目录修正

| 修改 | 旧名 | 新名 |
|------|------|------|
| 论文目录去括号 | `acmart-primary(1)` | `paper` |
| 统一英文命名 | `数据` | `data` |
| 修复拼写 | `LigntRAG` | `LightRAG` |
| 基线移入子目录 | 根目录 | `third_party/` |
| 展平嵌套 | `qwen-kge/ai-knowledge-graph-main/` | (消除) |

### 4.3 关键文件重命名

| 新路径 | 原名 | 原因 |
|--------|------|------|
| `src/extraction/generator.py` | `main.py` | 语义清晰 |
| `src/extraction/verifier.py` | `entity_standardization.py` | Session 2 目标 |
| `src/data_pipeline/loader.py` | `clean_triples.py` | Session 3 目标 |
| `src/data_pipeline/masking.py` | `build_graph.py` | Session 3 目标 |
| `src/models/encoders.py` | `embed_policies.py` | Session 4 目标 |
| `src/models/layers.py` | `train_gat_contrastive.py` | Session 4 目标 |
| `src/models/propagation.py` | `graphrag_pipeline.py` | Session 4 目标 |
| `src/evaluation/metrics.py` | `evaluate_matching.py` | Session 5 目标 |
| `src/evaluation/tei_analyser.py` | `transmission_efficiency.py` | Session 5 目标 |
| `experiments/run_main.py` | `run_a2_joint_full_pipeline.py` | 统一入口命名 |
| `experiments/run_ablation.py` | `run_ablation_11.py` | 统一入口命名 |

---

## 5. Session 2 待办事项

1. **重构 `src/extraction/generator.py`**: 从 generator.py + prompts.py + llm.py 整合，统一三元组抽取入口，明确输入输出契约
2. **创建 `src/extraction/verifier.py`**: 基于 `ontology/` 实现本体拓扑校验
3. **创建 `src/extraction/evaluator.py`**: 实现黄金标准集验证与错误传播敏感度分析
4. **自动生成 `data/statistics/dataset_statistics.json`**
5. **修复所有跨模块 import 路径**: 适配新目录结构
6. **数据泄露审计**: 检查 data/raw/ -> data/processed/ 全链路

---

## 6. Session 3 待办事项

1. **重构 `src/data_pipeline/loader.py`**: 统一 DataLoader 入口，控制全图加载与缓存
2. **重构 `src/data_pipeline/masking.py`**: 严格的边掩码与 Train/Val/Test 按时间切分
3. **修复数据泄露**: 确保企业-政策关联的切分不会导致验证/测试集信息泄露

---

## 7. Session 4 待办事项

1. **重构 `src/models/encoders.py`**: TEXT-Attributed 节点属性编码 (BERT+MLP)
2. **重构 `src/models/layers.py`**: 置信度感知的 HeteroGAT 层设计
3. **重构 `src/models/propagation.py`**: 层级-时间衰减的 PPR 能量传播引擎

---

## 8. Session 5 待办事项

1. **重构 `src/evaluation/metrics.py`**: 严密评测 (NDCG, Recall, MAP)，去除 Boost
2. **重构 `src/evaluation/tei_analyser.py`**: 宏观传播效率分析
3. **更新 `experiments/run_main.py`** + `experiments/run_ablation.py`**: 对接新模块
4. **强制同时输出 E->P 和 P->E 结果**

---

*本文件由开发者与 Claude Code 共同维护。每次 Session 完成后更新进度。*
