# 行业子图三档上跑 induced_v2 评测协议（与 legacy 主协议+过滤区分，输出新文件名）。
# 协议：--subgraph_eval_protocol induced_v2 + OpenKE test supports 在子图内构造查询集（企业/行业 E→P 与 P→E 均来自子图实体池）。
# 前置：各子图目录 policies_clean.parquet、enterprises_filtered.parquet；GAT 另需 gat_*_emb_contrastive.npy。
# 工作目录：项目根目录。

$ErrorActionPreference = "Stop"
$Proj = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Proj

$ScaleBase = "data_intermediate/data_scale_by_industry"
$Tags = @("nodes_0_10", "nodes_0_20", "nodes_0_50")

foreach ($tag in $Tags) {
    $sd = "$ScaleBase/$tag"
    $pol = Join-Path $Proj "$sd/policies_clean.parquet"
    if (-not (Test-Path $pol)) {
        Write-Warning "Skip $tag : missing $pol"
        continue
    }
    Write-Host "========== induced_v2 scale_dir=$sd ==========" -ForegroundColor Cyan

    $ProtoArgs = @("--ground_truth_source", "test", "--scale_dir", $sd, "--subgraph_eval_protocol", "induced_v2")

    python scripts/real_comparison_text_rag_matching_eval.py `
        --mode naive @ProtoArgs `
        --output "reports/real_comparison_results/naive_tfidf_matching_eval_testsplit_subgraph_${tag}_induced_v2.json"

    # Vector：每次仍会调用 DashScope Embedding 编码语料（脚本无向量盘缓存）；需 DASHSCOPE_API_KEY
    if ($env:SKIP_VECTOR_RAG_INDUCED -ne "1") {
        python scripts/real_comparison_text_rag_matching_eval.py `
            --mode vector @ProtoArgs `
            --output "reports/real_comparison_results/vector_rag_matching_eval_testsplit_subgraph_${tag}_induced_v2.json"
    }

    python scripts/real_comparison_openke_matching_eval.py `
        @ProtoArgs `
        --output "reports/real_comparison_results/openke_matching_eval_subgraph_${tag}_induced_v2.json"

    python scripts/real_comparison_kgbert_matching_eval.py `
        @ProtoArgs `
        --output "reports/real_comparison_results/kgbert_matching_eval_subgraph_${tag}_induced_v2.json"

    python scripts/real_comparison_atise_matching_eval.py `
        @ProtoArgs `
        --output "reports/real_comparison_results/atise_matching_eval_subgraph_${tag}_induced_v2.json"

    python scripts/data_scale_gat_matching_eval.py `
        --scale_dir $sd --subgraph_eval_protocol induced_v2 `
        --eval_query_scope subgraph_entities `
        --output "reports/real_comparison_results/data_scale_gat_matching_${tag}_induced_v2.json"

    # LightRAG：--skip_ingest 复用已有 triples_api_subgraph_* 工作区，仅重跑 induced_v2 查询评测（仍走 hybrid LLM/rerank API）
    if ($env:SKIP_LIGHTRAG_INDUCED -ne "1") {
        python scripts/real_comparison_lightrag.py `
            --skip_ingest --scale_dir $sd --subgraph_eval_protocol induced_v2 `
            --output "reports/real_comparison_results/lightrag_results_subgraph_${tag}_induced_v2.json"
    }
    # HippoRAG：--reuse_cached_index 复用 dashscope_subgraph_* 下向量与图，避免重建索引时的批量 Embedding
    if ($env:SKIP_HIPPORAG_INDUCED -ne "1") {
        python scripts/real_comparison_hipporag.py `
            --reuse_cached_index --scale_dir $sd --subgraph_eval_protocol induced_v2 `
            --output "reports/real_comparison_results/hipporag_results_subgraph_${tag}_induced_v2.json"
    }
}

Write-Host "Done. See report/data_scale_subgraph_induced_eval_v2.md for aggregation." -ForegroundColor Green
