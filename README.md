# Multimodal Heterogeneous Graph Learning with Spatio-Temporal Importance Modeling for Policy-Enterprise Bidirectional Matching and Implementation Evaluation

Research project on policy-enterprise knowledge graph construction, graph-based representation learning, and bidirectional matching with temporal decay mechanisms.

## Overview

This project implements an end-to-end pipeline for:

1. **Triple Extraction** - LLM-based SPO extraction from policy documents and enterprise data
2. **Multi-Modal Feature Fusion** - BERT text embeddings + MLP-mapped hierarchy/time features → heterogeneous node representations
3. **Graph Structure Learning** - GAT (Graph Attention Network) with contrastive training on heterogeneous policy-enterprise-industry graphs
4. **Temporal Importance Decay** - Exponential decay formula $V_{policy}^{(t)} = V_{policy} \cdot e^{-\beta \cdot \Delta t}$ for time-aware policy weighting
5. **Bidirectional Matching** - Enterprise-to-Policy and Policy-to-Enterprise retrieval with adaptive quantile-based candidate truncation
6. **Transmission Efficiency** - TEI (Transmission Efficiency Index) for evaluating policy impact propagation

## Directory Structure

```
├── src/                   # Core library (KG pipeline, LLM, prompts, visualization)
├── matching/              # Bidirectional matching system
├── graph/                 # Graph construction & GAT training
│   └── checkpoints/       # Model checkpoints (gitignored)
├── features/              # Feature engineering (attribute, time, fusion)
├── embeddings/            # Text embedding generation (BERT)
├── evaluation/            # Importance scoring, decay, TEI, coverage
├── scripts/               # Experiment scripts (pipelines, ablation, baselines)
├── data_clean/            # Data cleaning & preprocessing
├── ontology/              # Domain ontology definitions (JSON)
├── paper/                 # ACM LaTeX paper source
├── docs/                  # Experiment documentation & methodology reports
├── results/               # Experiment results (JSON, CSV, MD reports)
├── supplementary/         # Supplementary material (redacted code, artifacts)
├── data/                  # Data directory
│   ├── raw/               # Raw source data (gitignored)
│   ├── processed/         # Cleaned enterprise data (gitignored)
│   └── intermediate/      # Pipeline intermediates (gitignored)
├── baselines/             # Comparison baseline implementations
│   ├── HippoRAG/          # HippoRAG - PPR-based GraphRAG
│   ├── KG-BERT/           # KG-BERT - BERT for link prediction
│   ├── LightRAG/          # LightRAG - Dual-level retrieval
│   ├── atise/             # ATiSE - Temporal KGE
│   └── openkg/            # OpenKE - TransE baseline
└── config.toml            # Project configuration
```

## Key Documentation

- [A2 Main Pipeline](docs/A2-pipeline.md) - Primary experiment pipeline and ablation baseline
- [Experiment Summary](docs/experiment-summary.md) - Full experiment workflow
- [Methodology](docs/methodology.md) - System methodology report
- [Ablation Study](docs/ablation-design.md) - Ablation experiment design and results
- [Comparison Experiments](docs/comparison-design.md) - Baseline comparison design, steps, and results
- [Triple Extraction](docs/triple-extraction.md) - Ontology construction and triple extraction
- [Data Processing](docs/data-processing.md) - Enterprise data cleaning and preparation
- [Transmission Efficiency](docs/transmission-efficiency.md) - TEI evaluation

## Main Experiment Profiles

| Profile | Description | Script |
|---------|-------------|--------|
| `legacy` | Concat BERT + default GAT/decay | `scripts/run_a2_joint_full_pipeline.py` |
| `a2_base` | Joint BERT + a2_joint GAT/decay (BASE) | `scripts/run_a2_joint_full_pipeline.py` |
| `a3_title` | Title-only BERT + a3_title pipeline | `scripts/run_a3_title_full_pipeline.py` |

## Key Results (BASE - A2 Joint)

**Enterprise → Policy** (330 queries): NDCG 0.3417, MAP 0.1252, F1 0.2887

**Policy → Enterprise** (87 queries): NDCG 0.6667, MAP 0.1750, F1 0.2651

Full comparison results across all baselines: [Main Protocol Metrics](results/main_protocol_metrics_all_methods.md)

## Environment Setup

```bash
# Python 3.11+
pip install -r requirements.txt

# Configure API keys
cp local_dashscope.env.example local_dashscope.env  # if using DashScope
```

## Citation

Paper under review. ACM format source in `paper/` directory.
