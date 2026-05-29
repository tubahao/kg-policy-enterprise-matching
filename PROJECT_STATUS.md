# Project Status Log

> 贯穿项目的核心日志，由开发者与 Claude Code 共同维护。
> 每次重大变更后更新日期和内容。

**Last Updated:** 2026-05-29 (Session 2 — 完全结束，进入 Session 3)

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
| **Session 1** | 目录清洗 + 政策数据清洗 + 数据准备大一统 | **✅ 完全结束** |
| Session 2 | 三元组提取重构 (extraction/) | **✅ 完全结束** |
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
│   │   ├── kg-json/           # 原始KG JSON (test_data.json等)
│   │   └── policy_data.xlsx   # 原始政策Excel (2,311条)
│   ├── processed/             # LLM抽取后的结构化数据
│   │   ├── enterprises/       # 企业处理结果 (gitignored)
│   │   ├── policies_cleaned.json  # [Session 2-pre] 清洗后政策数据集
│   │   ├── policies_final.json    # [Session 1-final] 时间截断后政策数据集 (1,892 active)
│   │   └── enterprises_final.json # [Session 1-final] 企业特征工程输出 (6,393 家)
│   ├── comparison-data/       # 对比基线数据集
│   │   ├── atise_policykg/    # ATISE格式
│   │   ├── kgbert_policykg/   # KG-BERT格式
│   │   └── openke_policykg/   # OpenKE格式
│   ├── intermediate/          # 管道中间产物 (gitignored)
│   └── statistics/            # 数据集统计结果
│       └── cleaning_report.json  # [Session 2-pre] 清洗统计报告
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
│   ├── data_pipeline/         # 数据管线 (Session 1 重构完毕)
│   │   ├── clean_policy_raw.py        # 原始政策清洗流水线
│   │   ├── filter_historic_policies.py # 政策时间截断
│   │   └── preprocess_enterprises_final.py # 企业特征工程+时序对齐
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

## 5. Session 2-pre 变更清单 — 政策原始数据清洗

### 5.1 新建文件

| 文件 | 说明 |
|------|------|
| `src/data_pipeline/clean_policy_raw.py` | 完整清洗流水线 (~880行)，含 CLI |
| `data/processed/policies_cleaned.json` | 清洗后数据集 (1,903 active + 19 pruned) |
| `data/statistics/cleaning_report.json` | 清洗统计报告 |

### 5.2 核心流水线 (4 阶段)

**阶段 1 — 行级分类**: 识别 1,876 原文政策 + 416 解读文章 + 19 空壳附件

**阶段 2 — 三级级联实体对齐 (原文 ↔ 解读)**:
- T1 强规则 (`《XXX》` 书名号精确提取): 291 matched (70.0%)
- T2 LCS 最长公共子串软匹配: 94 matched (22.6%)
- T3 发布单位+时间窗口约束: 4 matched (1.0%)
- 总对齐率: **93.5%** (389/416)，27 条未匹配解读保留为独立政策
- 多篇解读合并到同一原文: 14 条 (用 `---` 分隔)

**阶段 3 — 语义融合与剪枝**:
- 低信息熵剪枝: 15 条 (正文仅含附件占位符，字符数 < 50)
- 复活空壳节点: 3 条 (原文为附件列表但有高质量解读，用解读内容替换)
- 原文-解读融合: 373 条 (`【政策原文】...\n【官方解读】...`)

**阶段 4 — 字段补全与级别映射**:
- `pub_agency` 填补: 162 条 (从标题前缀 / 正文开头 / 解读文本提取)
  - 仍缺失 4 条 → `pruned_by_missing_publisher` (正文中无任何机构名字符串线索)
- `level` 映射: 1,137 Policy1 (国家) + 501 Policy2 (自治区) + 265 Policy3 (柳州)

### 5.3 最终数据卡片

| 指标 | 数值 |
|------|------|
| 输入 | 2,311 条 (policy_data.xlsx) |
| 输出 active | **1,903 条** (零 null 字段) |
| 输出 pruned | 15 (低信息熵) + 4 (发布单位缺失) |
| 所有字段 null | **0** |
| avg 融合文本长度 | 3,849 字符 |
| median 融合文本长度 | 3,072 字符 |

### 5.4 输出 JSON Schema

```
{
  policy_id:     "P_XXXX"
  region:        "国家" | "自治区" | "柳州"
  title:         string (原始标题，保留失效标记)
  pub_agency:    string (零 null，已填补)
  pub_date:      string (零 null)
  fusion_content: {
    main_text:             string (政策正文，复活空壳为 "")
    interpretation_text:   string (对齐后的解读，多篇 --- 分隔)
    is_main_empty:         bool
  }
  text_for_llm:  string (直接喂给 Session 2 LLM 生成器)
  level: {
    type:         "Policy1" | "Policy2" | "Policy3"
    level_index:  1 | 2 | 3
  }
  status:        "active" | "pruned_by_low_entropy" | "pruned_by_missing_publisher"
}
```

### 5.5 剪枝样本说明

| type | 样例 | 剪枝原因 |
|------|------|----------|
| 低信息熵 | "关于开展2022年未来工厂认定工作的通知" → "附件：XXX.wps XXX.zip" | 正文仅附件列表，无实质性政策内容 |
| 低信息熵 | "关于申报2021年模具产业发展奖励资金的通知" → "附件下载：附件1-2.xlsx" | 同上 |
| 缺失发布单位 | "柳州市发展壮大民营经济实施方案" | 方案全文嵌套在通知中，通知头发文机关未爬取 |
| 缺失发布单位 | "中小企业促进法系列讲解之人才篇" | 普法教育文章，非政府公文，无单一发布单位 |

---

## 6. Session 1-final 变更清单 — 数据准备大一统

### 6.1 新建文件

| 文件 | 说明 |
|------|------|
| `src/data_pipeline/filter_historic_policies.py` | 政策时间截断 (~70行) |
| `src/data_pipeline/preprocess_enterprises_final.py` | 企业特征工程+时序对齐 (~260行) |
| `data/processed/policies_final.json` | 截断后政策数据集 (1,892 active + 19 pruned = 1,911 条) |
| `data/processed/enterprises_final.json` | 企业特征数据集 (6,393 家) |

### 6.2 任务一: 政策时间截断

| 指标 | 数值 |
|------|------|
| 输入 active | 1,903 |
| pub_date < 2018-01-01 | 11 条 (分布于 2001-2017，年均 1-6 条) |
| 截断后 active | **1,892** |
| 去除占比 | 0.6%，对各级别分布无影响 |

### 6.3 任务二: 企业特征工程

**输入**: `data/raw/enterprises-source/` 下 6 个行业 Excel (6,453 行 × 41 列)

**行业分布 (开业企业)**:

| major_industry | 数量 |
|------|------|
| 制造业 | 1,776 |
| 科学研究和技术服务业 | 3,652 |
| 文化、体育和娱乐业 | 410 |
| 电力、热力、燃气及水生产和供应业 | 157 |
| 水利、环境和公共设施管理业 | 121 |
| 高新企业 | 277 |
| **合计** | **6,393** |

**特征工程详情**:

| 特征 | 方法 | 结果 |
|------|------|------|
| `sub_industry` | 从"所属行业"提取 | 60 种子行业 |
| `major_industry` | 从文件名映射 (6大类, 高新企业独立) | 6 类 |
| `capital_wan` | 正则提取 → 万元, 缺失/零用同子行业中位数填充 | 填充 296 条 (4.6%) |
| `capital_log` | `log(capital_wan + 1)` | — |
| `scale_category` | 基于 2024 年参保人数分箱 | 微型3716/小型1627/中型718/大型332 |
| `insurance_time_series.values` | 2017-2024 八位数组, mask=1 内线性插值 | 575 家企业有内部插值 |
| `insurance_time_series.log_values` | `log(values + 1)`, mask=0 处置 0.0 | — |
| `insurance_time_series.padding_mask` | 左端连续 NaN/0 → mask=0 (未成立), 首个非零起 → mask=1 | 2,420 家企业非全期存续 |

**mask 覆盖率 (年):**

| 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|------|------|------|------|------|------|------|------|
| 20.4% | 27.9% | 37.6% | 47.4% | 58.3% | 70.9% | 84.4% | 100.0% |

**注册资本分布 (万元):** min=0.3, p25=100, median=200, p75=500, max=1,142,756

### 6.4 输出 JSON Schema

**policy** (`policies_final.json`): 与 policies_cleaned.json 相同 schema，仅过滤 + policy_id 重编号。

**enterprise** (`enterprises_final.json`):
```json
{
  "name": "柳州五菱汽车有限责任公司",
  "major_industry": "高新企业",
  "sub_industry": "汽车制造业",
  "capital_wan": 122470.0,
  "capital_log": 11.71563,
  "scale_category": 3,
  "scope": "设计、开发、生产...",
  "insurance_time_series": {
    "years": [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "values": [4943.0, 5070.0, 4956.0, 5009.0, 4985.0, 5089.0, 5039.0, 4936.0],
    "log_values": [8.506, 8.531, 8.509, 8.519, 8.514, 8.535, 8.525, 8.505],
    "padding_mask": [1, 1, 1, 1, 1, 1, 1, 1]
  }
}
```

### 6.5 数据准备阶段总览

```
data/raw/policy_data.xlsx (2,311)
    │
    ▼  clean_policy_raw.py
data/processed/policies_cleaned.json (1,903 active)
    │
    ▼  filter_historic_policies.py
data/processed/policies_final.json (1,892 active)  ← Session 2 入口

data/raw/enterprises-source/*.xlsx (6,453)
    │
    ▼  preprocess_enterprises_final.py
data/processed/enterprises_final.json (6,393)       ← Session 3 入口
```

---

## 7. Session 2 变更清单 — 三元组提取重构

### 7.1 设计原则

**数据泄露防线**: 图谱底座**坚决不包含** Policy↔Enterprise 直连边 (`supports`/`implements`)。
这些边是 Session 4/5 的 Ground Truth 评估标签，建图阶段严格隔离。

**本体重构**: 从 v1 的 6 种实体类型 + 6 种关系 → v2 的 4 种实体类型 + 4 种关系，核心变化：
- 6 大行业从"各自独立 EntityType"→ 同一 MajorIndustry 类型的 5 个实例
- 新增 SubIndustry 实体类型（63 个实例），Enterprise→SubIndustry→MajorIndustry 两级层级
- 高新企业从"行业类别"→ Enterprise 节点的 `is_high_tech` 布尔属性

### 7.2 新建文件

| 文件 | 说明 |
|------|------|
| `src/extraction/deterministic_edges.py` | 确定性建边 (~280行)，零 LLM 参与 |
| `src/extraction/llm_classifier.py` | LLM 受控多标签分类 (~170行) |
| `src/extraction/verifier.py` | 字符串对齐校验 + 边合并 + 本体生成 (~250行) |
| `src/extraction/ontology/ontology_v2.json` | 统一新本体定义 |
| `data/processed/deterministic_graph_edges.json` | 确定性边输出 |
| `data/processed/llm_classification_results.json` | LLM 原始分类结果 |
| `data/processed/graph_edges_final.json` | **最终图谱边全集** → Session 3 入口 |
| `data/statistics/extraction_report.json` | 抽取质量报告 |

### 7.3 修改文件

| 文件 | 变更 |
|------|------|
| `config.toml` | DeepSeek v4 API 配置，弃用 chunking/standardization/inference 节 |
| `src/extraction/llm.py` | 重写为 OpenAI 兼容格式 |
| `src/extraction/prompts.py` | 替换为受控分类 prompt |
| `src/extraction/config.py` | 修复 tomli→tomllib 兼容 |
| `src/extraction/__init__.py` | 清理旧 import |

### 7.4 弃用文件

| 文件 | 说明 |
|------|------|
| `src/extraction/generator.py` | 旧的滑动窗口+自由 SPO 抽取 |
| `src/extraction/text_utils.py` | chunk_text 不再需要 |
| `src/extraction/generate_graph.py` | 旧入口 |
| `ontology/actual_*.json` (5个) | 移至 `ontology/deprecated/` |

### 7.5 图谱最终数据卡片

| 指标 | 数值 |
|------|------|
| 实体 — Policy (Policy1/2/3) | 1,892 |
| 实体 — SubIndustry | 63 |
| 实体 — MajorIndustry | 5 |
| 实体 — Enterprise | 6,393 (含 277 高新企业) |
| **总边数** | **8,274** |
| └ belongsTo (Enterprise→SubIndustry) | 6,393 (确定性, 100% 准确) |
| └ subClassOf (SubIndustry→MajorIndustry) | 63 (确定性, 100% 准确) |
| └ transmitsTo (Policy→Policy) | 157 (正则+标题检索) |
| └ targetsSubIndustry (Policy→SubIndustry) | 1,661 (LLM 分类, 接受率 100%) |
| LLM 平均置信度 | 0.864 |
| LLM 零幻觉率 | 100.0% (精确字符串匹配全部通过) |
| 有行业靶向的政策 | 738/1,892 (39.0%) |
| 无特定行业靶向的政策 | 1,154/1,892 (61.0%) |
| 被靶向的子行业 | 54/63 (85.7%) |

### 7.6 无行业靶向政策分析

61.0% 的政策无特定子行业靶向，主要类型：
- **财税金融通用政策**: 减税、研发加计扣除 — 跨行业普惠
- **人才/就业政策**: 人才引进、留工补助 — 不限行业
- **机构/平台管理办法**: 创新基地管理 — 制度性文件
- **申报流程/办事指南**: 操作性文件 — 纯流程说明
- **文本过短/内容缺失**: 清洗过程中丢失内容

按级别: Policy1 无靶向 58.6%, Policy2 67.3%, Policy3 62.3%

### 7.7 新本体 v2 摘要

```
实体类型: Policy, SubIndustry, MajorIndustry, Enterprise
关系:
  transmitsTo        Policy→Policy         行政纵向传导 (正则提取)
  targetsSubIndustry Policy→SubIndustry    行业靶向 (LLM分类, 带confidence)
  belongsTo          Enterprise→SubIndustry 企业归属 (确定性)
  subClassOf         SubIndustry→MajorIndustry 行业层级 (确定性)
Ground Truth (建图阶段不生成):
  supports           Policy2/3→Enterprise  真实资助记录 (评估标签)
```

### 7.8 未靶向的子行业 (9/63)

| 子行业 | 企业数 | 大类 |
|--------|--------|------|
| 木材加工和木/竹/藤/棕/草制品业 | 196 | 制造业 |
| 印刷和记录媒介复制业 | 51 | 制造业 |
| 水的生产和供应业 | 45 | 电力/热力/燃气/水 |
| 燃气生产和供应业 | 19 | 电力/热力/燃气/水 |
| 文教/工美/体育和娱乐用品制造业 | 18 | 制造业 |
| 金属制品/机械和设备修理业 | 14 | 制造业 |
| 烟草制品业 | 1 | 制造业 |
| 租赁业 | 1 | 科学研究和技术服务业 |
| 未分类 | 1 | 制造业 |

---


## 8. Session 3 待办事项

1. **重构 `src/data_pipeline/loader.py`**: 统一 DataLoader 入口，控制全图加载与缓存
2. **重构 `src/data_pipeline/masking.py`**: 严格的边掩码与 Train/Val/Test 按时间切分
3. **修复数据泄露**: 确保企业-政策关联的切分不会导致验证/测试集信息泄露

---

## 9. Session 4 待办事项

1. **重构 `src/models/encoders.py`**: TEXT-Attributed 节点属性编码 (BERT+MLP)
2. **重构 `src/models/layers.py`**: 置信度感知的 HeteroGAT 层设计
3. **重构 `src/models/propagation.py`**: 层级-时间衰减的 PPR 能量传播引擎

---

## 10. Session 5 待办事项

1. **重构 `src/evaluation/metrics.py`**: 严密评测 (NDCG, Recall, MAP)，去除 Boost
2. **重构 `src/evaluation/tei_analyser.py`**: 宏观传播效率分析
3. **更新 `experiments/run_main.py`** + `experiments/run_ablation.py`**: 对接新模块
4. **强制同时输出 E->P 和 P->E 结果**

---

*本文件由开发者与 Claude Code 共同维护。每次 Session 完成后更新进度。*
