# Project Status Log

> 贯穿项目的核心日志，由开发者与 Claude Code 共同维护。
> 每次重大变更后更新日期和内容。

**Last Updated:** 2026-05-30 (Session 4 — 完全结束，进入 Session 5)

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
| Session 3 | 数据管线重构 (data_pipeline/) | **✅ 完全结束** |
| Session 4 | 模型层重构 (models/) — 抗泄露 + HT-PPR | **✅ 完全结束** |
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
│   ├── splits/                # [Session 4] Train/Val/Test 切分
│   │   ├── message_graph.pt
│   │   ├── train_supports.pt
│   │   ├── val_supports.pt
│   │   ├── test_supports.pt
│   │   ├── full_supports.pt
│   │   └── supports_split_meta.json
│   ├── ht_ppr/                # [Session 4] HT-PPR 异构传播得分
│   │   ├── policy_raw.npy / policy_decayed.npy / policy_final.npy
│   │   ├── enterprise_raw.npy / enterprise_compensated.npy / enterprise_final.npy
│   │   ├── sub_industry_raw.npy / major_industry_raw.npy
│   │   └── ht_ppr_meta.json
│   ├── time_embeddings/       # [Session 4] 企业时间序列 GRU 编码
│   │   ├── enterprise_temporal_emb.pt
│   │   └── enterprise_temporal_meta.json
│   ├── gat_checkpoints/       # [Session 4] GAT 训练检查点 (gitignored)
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
│   ├── models/                # [Session 4 ✅] 模型层 — 抗泄露 + HT-PPR
│   │   ├── graph_splitter.py          # 抗泄露图拆分管线 (新建)
│   │   ├── layers.py                  # 置信度感知 HeteroGAT (DGL→PyG)
│   │   ├── train_gat.py               # 抗泄露对比学习训练
│   │   ├── propagation.py             # HT-PPR 异构传播引擎
│   │   ├── attribute_embeddings.py    # 层级/时间差 Delta 编码器
│   │   ├── time_embeddings.py         # 企业 GRU 时间序列编码
│   │   ├── embed_enterprises.py       # 企业 text2vec 离线嵌入
│   │   ├── fuse_features.py           # 强制 L2 归一化特征融合
│   │   ├── fuse_enterprise_features.py# 企业特征融合 (旧, 保留)
│   │   ├── encoders.py                # 政策 BERT 编码 (旧, 保留)
│   │   ├── train_gnn.py               # 基础 GNN 训练 (旧, 保留)
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


## 8. Session 3 变更清单 — 数据管线重构 & 图张量化

### 8.1 设计原则

**解耦嵌入与建图**: 文本嵌入 (text_embedder.py) 离线独立运行，图构建 (graph_builder.py) 加载预计算嵌入。GPU 推理与图组装完全分离，便于迭代调参。

**靶向继承**: 解决"能量死胡同"——政策接收上级 transmitsTo 传导但无 targetsSubIndustry 出边，导致 PPR 能量无法流入行业。通过沿行政链向上递归追溯父/祖父政策，继承其靶向行业边。

**核心隔离**: 5 大核心产业 (制造业 / 科学研究和技术服务业 / 文化体育和娱乐业 / 水利环境和公共设施管理业 / 电力热力燃气水) 的特征空间严格保护，跨域行业迁入独立的 MI_05。

### 8.2 新建文件

| 文件 | 说明 |
|------|------|
| `src/data_pipeline/ontology_corrector.py` | 本体修正 (~280行): 硬编码字典修复 subClassOf + SI 合并 + 企业 major 修复 |
| `src/data_pipeline/text_embedder.py` | 离线文本嵌入 (~200行): HuggingFace text2vec-base-chinese → 768-dim |
| `src/data_pipeline/graph_builder.py` | 图构建 (~700行): Phase A 靶向继承 + Phase B PyG HeteroData |

### 8.3 修改文件

| 文件 | 变更 |
|------|------|
| `config.toml` | 新增 `[session3]` 节 (嵌入模型/批量/设备配置) |
| `requirements.txt` | 新增 torch, torch-geometric, transformers, sentence-transformers, tqdm |

### 8.4 输出文件

| 文件 | 说明 |
|------|------|
| `data/processed/graph_edges_corrected.json` | 本体修正后边全集 (v2.1) |
| `data/processed/enterprises_corrected.json` | 277 家企业 major_industry 修正 ("高新企业"→正确产业) |
| `data/processed/text_embeddings/policy_text_emb.pt` | [1892, 768] 政策文本嵌入张量 |
| `data/processed/text_embeddings/policy_emb_index.json` | policy_id → row 索引映射 |
| `data/processed/graph/hetero_graph.pt` | **PyG HeteroData 最终图对象** → Session 4 入口 |
| `data/processed/graph/graph_meta.json` | 图元数据 (节点/边映射, 维度信息) |
| `data/statistics/ontology_correction_report.json` | 本体修正全程日志 |
| `data/statistics/inheritance_report.json` | 靶向继承详细报告 |

### 8.5 本体修正详情 (Task 1)

**subClassOf 修正 (8 条)**:

| SI_ID | 子行业 | 旧 MI | 新 MI |
|-------|--------|-------|-------|
| SI_00 | 专业技术服务业 | MI_02 (水利) | MI_04 (科研) |
| SI_09 | 农业 | MI_00 (制造业) | MI_05 (跨域) |
| SI_18 | 土木工程建筑业 | MI_00 | MI_05 |
| SI_23 | 建筑安装业 | MI_00 | MI_05 |
| SI_24 | 建筑装饰、装修和其他建筑业 | MI_00 | MI_05 |
| SI_26 | 房屋建筑业 | MI_00 | MI_05 |
| SI_27 | 批发业 | MI_00 | MI_05 |
| SI_59 | 零售业 | MI_00 | MI_05 |

**SI_11 "制造业" → SI_07 "其他制造业" 合并**:
- 2 条 belongsTo 边 + 24 条 targetsSubIndustry 边重新定向
- SI_11 的 subClassOf 边移除
- 2 家企业 sub_industry 字段更新

**企业 major_industry 修复**:
- 277 家企业 `major_industry = "高新企业"` 替换为正确的产业名
- 根据子行业归属映射到 MI_00~MI_04
- 修复后 0 家企业残留 "高新企业"

**新建跨域产业**:
- MI_05 = "其他跨域产业 (Cross-domain)"
- 收容 7 个不属于 5 大核心产业的子行业
- 最终 MajorIndustry: 5 核心 + 1 跨域 = **6 个**

### 8.6 文本嵌入详情 (Task 4)

| 指标 | 值 |
|------|-----|
| 模型 | shibing624/text2vec-base-chinese (SentenceTransformer) |
| 嵌入维度 | 768 |
| 活跃政策 | 1,892 条 |
| 平均 text_for_llm 长度 | 3,857 字符 |
| 最大序列长度 | 512 token (text2vec 架构上限) |
| 批量大小 | 32 |
| 设备 | NVIDIA GeForce RTX 3060 Laptop (CUDA) |
| 编码耗时 | ~42 秒 |
| L2 范数 | min=13.02, mean=14.22, max=16.64 |
| 零向量 | **0** |

**编码方式**: 将 Session 1 融合的 `text_for_llm` 字段（【政策原文】+【官方解读】拼接）整体送入 text2vec 做 mean-pooling，产生单一 768-dim 向量。标题位于文本最前端，位置编码保留标题信号。

### 8.7 靶向继承详情 (Task 2)

**问题**: 108 条政策接收了上级 transmitsTo 传导，但自身无 targetsSubIndustry 出边 → PPR 能量流入但无法扩散到行业。

**算法**:
1. 从 transmitsTo 边构建 child→[parents] 映射（边方向: 上级→下级，通过 object 逆向追溯 subject）
2. 对每个死胡同政策，沿父链递归向上 (max_depth=5, visited set 防环)
3. 找到第一个有靶向行业的祖先后，复制其 targetsSubIndustry 边，置信度 × 0.8^hops
4. 同一 (policy, SI) 对去重，保留最高置信度
5. `match_method = "inherited"`, 记录 `inherited_from` 和 `inheritance_hops`

**结果**:

| 指标 | 值 |
|------|-----|
| 死胡同总数 | 108 |
| **成功激活** | **23 条 (21.3%)** |
| 仍死胡同 | 85 条 (整条祖先链无靶向) |
| 继承边总数 | 58 |
| 1-hop 继承 | 54 条 |
| 2-hop 继承 | 4 条 |
| 平均每激活政策 | 2.5 条靶向边 |

**激活样例**:
- P_0199: 从父政策 P_0608 1-hop 继承 3 个行业 (医药制造业 0.9→0.72, 软件和信息技术服务业 0.85→0.68, 专用设备制造业 0.85→0.68)
- P_0255: 从祖父政策 P_0545 2-hop 继承 1 个行业 (商务服务业 0.92→0.5888)

### 8.8 企业去重 (belongsTo 修复)

Session 1 从 6 个行业 Excel 文件加载企业时，887 家企业在多个源文件中重复出现（行业分类完全一致，为同企业重复）。导致 belongsTo 边存在 898 条重边。

**修复**: 按 `(enterprise_name, SI_id)` 去重 belongsTo 边: 6,393 → **5,495**。

### 8.9 HeteroData 最终数据卡片

| 节点类型 | 数量 | 特征维度 |
|----------|------|----------|
| Policy | 1,892 | text_emb [768], level int [1] |
| Enterprise | 5,495 | static_feat [2], temporal_series [8], padding_mask [8] |
| SubIndustry | 62 | x one-hot [62] |
| MajorIndustry | 6 | x one-hot [6] |

| 边类型 | 数量 | 说明 |
|--------|------|------|
| transmitsTo (Policy→Policy) | 157 | 确定性正则，层级行政传导 |
| targetsSubIndustry (Policy→SubIndustry) | 1,719 | 1,661 原始 + 58 继承，含 confidence |
| belongsTo (Enterprise→SubIndustry) | 5,495 | 去重后，每企业唯一 |
| subClassOf (SubIndustry→MajorIndustry) | 62 | 含 MI_05 跨域产业 |
| **supports** | **0** | **严格隔离 — Session 4/5 评估标签** |

**环境**: PyTorch 2.5.1+cu121, torch-geometric 2.7.0, D:\miniconda\envs\kg (Python 3.12.13)

### 8.10 数据流通总览 (更新)

```text
data/raw/policy_data.xlsx (2,311)
    │
    ▼  clean_policy_raw.py       [Session 1]
data/processed/policies_cleaned.json (1,903)
    │
    ▼  filter_historic_policies.py [Session 1]
data/processed/policies_final.json (1,892)  ← Session 2 入口
    │
    ├──► deterministic_edges.py  [Session 2]
    ├──► llm_classifier.py       [Session 2]
    ├──► verifier.py             [Session 2]
    │
    ▼  graph_edges_final.json (8,274 边)
    │
    ├──► ontology_corrector.py   [Session 3] ──► graph_edges_corrected.json
    │                                               │
    ├──► text_embedder.py        [Session 3] ──► policy_text_emb.pt [1892×768]
    │                                               │
    └──► graph_builder.py        [Session 3] ──► hetero_graph.pt    ← Session 4 入口
                                                    (Phase A: Target Inheritance
                                                     Phase B: PyG HeteroData)
```

---


## 9. Session 4 变更清单 — 模型层重构 & 抗泄露防线

### 9.1 三大红线 (全局约束)

| 红线 | 措施 | 状态 |
|------|------|------|
| **绝对隔离评估标签** | `graph_splitter.py` 断言 HeteroData 零 supports → 构建 `message_graph.pt` (仅 4 种消息边) → supports 7:1:2 互斥切分 → Train/Val/Test 零交集 | ✅ |
| **术语全面净化** | `GraphRAG` → `HT-PPR Engine`, `Spatio-Temporal` → `Hierarchical-Temporal`, `spatial_` 变量全域清零 | ✅ |
| **消除 L2 范数失衡** | `fuse_features.py` 强制 L2 归一化文本嵌入 + 所有编码器 BatchNorm→LayerNorm | ✅ |

---

### 9.2 步骤 1: 企业文本离线嵌入与特征正则化

#### 新建 `src/models/embed_enterprises.py` (208 行)

| 维度 | 详情 |
|------|------|
| **模型** | `shibing624/text2vec-base-chinese` (SentenceTransformer) |
| **输入** | `enterprises_final.json` (6,393 家) |
| **文本拼接** | `"{name}，所属行业：{major_industry}-{sub_industry}。主营业务：{scope}"` |
| **输出** | `data/processed/text_embeddings/enterprise_text_emb.pt` [6393, 768] |
| **设备** | CUDA (RTX 3060 Laptop) |

#### 修改 `src/models/fuse_features.py` (+8 行)

| 维度 | 详情 |
|------|------|
| **修复** | 第 179-182 行: 无条件 L2 归一化 `aligned_text` (不依赖 `--normalize` flag) |
| **原因** | 768 维 text embedding L2 范数 ≈14.22 vs 32 维 level/time 范数 ≈1-5, 未归一化拼接导致梯度吞噬 |
| **meta** | 输出新增 `"text_l2_normalized": true` |

---

### 9.3 步骤 2: 层级与时间编码器重构

#### 重写 `src/models/time_embeddings.py` (289 行)

| 维度 | 旧 | 新 |
|------|-----|-----|
| **编码器** | `TimeSeriesMLP` (扁平 MLP, 忽略 padding_mask) | `TemporalGRUEncoder` (双向 2 层 GRU, hidden=64, output=64) |
| **padding_mask** | 忽略 | `pack_padded_sequence` 按有效长度打包 |
| **输出** | — | `enterprise_temporal_emb.pt` [N, 64] |
| **mask 覆盖率** | — | 20.4% (2017) → 100% (2024) |

#### 重写 `src/models/attribute_embeddings.py` (261 行)

| 新增类 | 用途 | 架构 |
|------|------|------|
| `DeltaEncoder` | 通用标量→64 维编码 | MLP (1→16→32→48→64, LayerNorm+Dropout+Tanh) |
| `HierarchicalEncoder` | delta_l_ref (层级差) | 继承 DeltaEncoder |
| `TemporalEncoder` | delta_t_ref (时间差) | 继承 DeltaEncoder |

保留 `LevelEmbedding` (离散层级 Embedding) 和 `TimeMLP` (年份标量 MLP)。所有 BatchNorm → LayerNorm。

---

### 9.4 步骤 3: 【核心防御】抗泄露图张量拆分管线

#### 新建 `src/models/graph_splitter.py` (383 行)

**设计原理**: 图谱底座坚决不包含 Policy↔Enterprise 的 `supports` 边。这些边是 Ground Truth 评估标签，必须按比例切分并隔离于模型前向传播之外。

| Phase | 操作 | 结果 |
|-------|------|------|
| **Phase 1** | 加载 `hetero_graph.pt` → 断言 4 种边类型均非 supports | ✅ 零泄露 |
| **Phase 2** | 加载 `policy_id_to_idx` (1,892) + `ent_name_to_idx` (5,495) | ✅ |
| **Phase 3** | `policies_final.json` 构建 title→P_XXXX 映射 | 1,892 active |
| **Phase 4** | 从 `triples_policy_entity.parquet` 提取 supports (133,403 条) → 标题匹配 | 成功映射 123,570 (92.6%), 失败 9,833 |
| **Phase 5** | `np.random.RandomState(42)` 排列 → 7:1:2 切分 → 互斥断言 | ✅ T∩V=0, T∩Te=0, V∩Te=0 |
| **Phase 6** | 持久化到 `data/processed/splits/` | ~10 MB |

| 切分 | 边数 | 占比 |
|------|------|------|
| **Train** | 86,499 | 70.0% |
| **Val** | 12,357 | 10.0% |
| **Test** | 24,714 | 20.0% |
| **Full (mask)** | 123,570 | 100% |

| Message Edges (允许) | 说明 |
|---------------------|------|
| `transmitsTo` | Policy→Policy 行政传导 |
| `targetsSubIndustry` | Policy→SubIndustry LLM 靶向 (含 confidence) |
| `belongsTo` | Enterprise→SubIndustry 确定性 |
| `subClassOf` | SubIndustry→MajorIndustry 确定性 |

| Forbidden (前向传播不可见) |
|---------------------------|
| `supports`, `supportedByPolicy`, `supportedBy` |

---

### 9.5 步骤 4: 置信度感知 HeteroGAT + 抗泄露对比学习

#### 重写 `src/models/layers.py` (261 行)

| 维度 | 旧 | 新 |
|------|-----|-----|
| **框架** | DGL (`dgl.nn.HeteroGraphConv` + `GATConv`) | **PyG 2.7.0** (`torch_geometric.nn.HeteroConv` + `GATConv`) |
| **核心类** | `HeteroGATContrastive` | `ConfidenceAwareHeteroGAT` |
| **置信度感知** | 无 | `targetsSubIndustry` 边 → `GATConv(edge_dim=1)`, 输入 LLM confidence [1719] (mean=0.858, [0.3, 1.0]) |
| **Jumping Knowledge** | concat(initial, final_gat) | 保留并增强: Linear→LayerNorm→L2-norm |
| **辅助函数** | — | `load_node_features()` (Policy 769d, Enterprise 66d) + `build_edge_attr_dict()` |

#### 重写 `src/models/train_gat.py` (559 行)

| 维度 | 详情 |
|------|------|
| **正样本 Source A** | **仅** `Train_supports` (86,499 对) — Val/Test 绝对不参与 |
| **正样本 Source B** | 元路径 `Policy → SubIndustry → Enterprise`, 每 policy ≤30 条, **排除 Val/Test** |
| **负样本** | 同 SubIndustry 困难负样本 → 全局采样 fallback → **全量 `full_supports` mask** 过滤假阴性 |
| **图前向传播** | `message_graph.pt` → `ToUndirected()` 加反向边 → 双层 `HeteroConv(GATConv)` |
| **损失** | InfoNCE 对比损失 + 可学习 temperature |
| **优化** | AdamW (lr=1e-3, wd=1e-4) + CosineAnnealingLR |

---

### 9.6 步骤 5: 异构 HT-PPR 传播引擎 (★ 核心贡献)

#### 重写 `src/models/propagation.py` (711 行) — 两次迭代修正

**迭代 1 — 异构 PPR**: 从仅 Policy→Policy 子图 (157 边) 扩展到全异构图 (14,866 非零元, 7,455 节点)。

**迭代 2 — 度数补偿 + Log-Z-Sigmoid**: 修复 Enterprise 分数偏离 (旧 MinMax mean=0.795) 到健康分布 (mean≈0.5)。

| Phase | 方法 | 详情 |
|-------|------|------|
| **Phase 0** | 全局转移矩阵 | `scipy.sparse` COO→CSC, 7,455×7,455, 14,866 nnz, 稀疏度 0.027% |
| **Phase 1** | 幂迭代 PPR | α=0.85, 76 轮收敛 (tol=1e-6), 悬空列 997/7,455 (leak 补偿) |
| **Phase 2** | Policy 双重衰减 | γ_h=0.8 层级衰减 (max 2 级) + λ_t=0.15 时间衰减 (max 6 年) |
| **Phase 3** | Enterprise 度数补偿 | raw_PPR × N_subindustry (补偿因子 1~2,019), 消除行业规模稀释 |
| **Phase 4** | Log-Z-Sigmoid | log(scores+ε) → Z-score → sigmoid, 按类型独立执行 |

**异构边权重设计**:

| 边类型 | 方向 | 权重 | 非零元 |
|--------|------|------|--------|
| transmitsTo | P→P fwd | 1.0 | 157 |
| rev_transmitsTo | P→P rev | w_rev=0.3 | 157 |
| targetsSubIndustry | P→SI fwd | LLM confidence | 1,719 |
| rev_targetsSubIndustry | SI→P rev | w_rev×conf | 1,719 |
| belongsTo | E→SI fwd | 1.0 | 5,495 |
| rev_belongsTo | SI→E rev | w_rev=0.3 ★ | 5,495 |
| subClassOf | SI→MI fwd | 1.0 | 62 |
| rev_subClassOf | MI→SI rev | w_rev=0.3 | 62 |

★ `rev_belongsTo (SI→E)` 是 PPR 能量从政策流入企业的**唯一通道**。

**最终分数分布 (Log-Z-Sigmoid)**:

| 指标 | Policy final | Enterprise final |
|------|-------------|-----------------|
| **mean** | **0.496** | **0.503** |
| **std** | **0.222** | **0.218** |
| median | 0.418 | 0.442 |
| min | 0.140 | 0.023 |
| max | 0.921 | 0.756 |

> 与旧版对比: Enterprise MinMax mean=0.795 (病态, 噪声放大) → Log-Z-Sig mean=0.503 (正态健康)。
> Policy 和 Enterprise 独立执行 Log-Z-Sigmoid，各自占据完整的 (0,1) 分布。

**输出文件** (`data/processed/ht_ppr/`):

| 文件 | 形状 | 说明 |
|------|------|------|
| `policy_raw.npy` | [1892] | 原始 PPR |
| `policy_decayed.npy` | [1892] | 层级+时间衰减后 |
| `policy_final.npy` | [1892] | Log-Z-Sigmoid [0,1] |
| `enterprise_raw.npy` | [5495] | 原始 PPR |
| `enterprise_compensated.npy` | [5495] | 度数补偿后 |
| `enterprise_final.npy` | [5495] | Log-Z-Sigmoid [0,1] |
| `sub_industry_raw.npy` | [62] | 中间传导分数 |
| `major_industry_raw.npy` | [6] | 中间传导分数 |

---

### 9.7 Session 4 文件变更总览

| 文件 | 状态 | 行数 | 核心内容 |
|------|------|------|----------|
| `src/models/embed_enterprises.py` | **新建** | 208 | 企业 text2vec 离线嵌入 |
| `src/models/graph_splitter.py` | **新建** | 383 | 抗泄露图拆分 (7:1:2) |
| `src/models/fuse_features.py` | **修改** | 259 | 强制 L2 归一化文本嵌入 |
| `src/models/attribute_embeddings.py` | **重写** | 261 | DeltaEncoder 系列 + LayerNorm |
| `src/models/time_embeddings.py` | **重写** | 289 | TemporalGRUEncoder + padding_mask |
| `src/models/layers.py` | **重写** | 261 | DGL→PyG, ConfidenceAwareHeteroGAT |
| `src/models/train_gat.py` | **重写** | 559 | 抗泄露 InfoNCE 对比训练 |
| `src/models/propagation.py` | **重写** | 711 | 异构 HT-PPR + 度数补偿 + Log-Z-Sig |

| 数据产物 | 位置 |
|----------|------|
| 企业文本嵌入 | `data/processed/text_embeddings/enterprise_text_emb.pt` |
| 企业时间序列编码 | `data/processed/time_embeddings/enterprise_temporal_emb.pt` |
| 抗泄露切分 (6 文件) | `data/processed/splits/` |
| HT-PPR 得分 (8 .npy + meta) | `data/processed/ht_ppr/` |
| GAT 检查点 | `data/processed/gat_checkpoints/` (gitignored) |

---

## 10. Session 5 待办事项

1. **重构 `src/evaluation/metrics.py`**: 严密评测 (NDCG, Recall, MAP)，去除 Boost
2. **重构 `src/evaluation/tei_analyser.py`**: 宏观传播效率分析
3. **更新 `experiments/run_main.py`** + `experiments/run_ablation.py`**: 对接新模块
4. **强制同时输出 E->P 和 P->E 结果**

---

*本文件由开发者与 Claude Code 共同维护。每次 Session 完成后更新进度。*
