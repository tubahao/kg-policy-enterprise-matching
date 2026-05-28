# 在行业子图三档（nodes_0_10 / 20 / 50）上批量跑主协议匹配评测。
# 前置：各子图目录下已有 policies_clean.parquet、enterprises_filtered.parquet（及 triples_policy_entity.parquet，full GT 时需要）。
# 工作目录：项目根目录（本脚本位于 scripts/ 下）。

$ErrorActionPreference = "Stop"
$Proj = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Proj

$ScaleBase = "data_intermediate/data_scale_by_industry"
$Tags = @("nodes_0_10", "nodes_0_20", "nodes_0_50")

foreach ($tag in $Tags) {
    $sd = "$ScaleBase/$tag"
    $pol = Join-Path $Proj "$sd/policies_clean.parquet"
    if (-not (Test-Path $pol)) {
        Write-Warning "跳过 $tag ：缺少 $pol"
        continue
    }
    Write-Host "========== scale_dir=$sd ==========" -ForegroundColor Cyan

    # Naive / Vector：默认输出名已带 _subgraph_<tag>
    python scripts/real_comparison_text_rag_matching_eval.py `
        --mode naive --ground_truth_source test --scale_dir $sd
    if ($env:RUN_VECTOR_RAG -eq "1") {
        python scripts/real_comparison_text_rag_matching_eval.py `
            --mode vector --ground_truth_source test --scale_dir $sd
    }

    # OpenKE：子图默认按当前查询集 GT（mask 后）分组均值×倍数自适应 max_output_cap；固定 cap 请传 --no_adaptive_output_cap
    python scripts/real_comparison_openke_matching_eval.py `
        --ground_truth_source test --scale_dir $sd `
        --output "reports/real_comparison_results/openke_matching_eval_subgraph_${tag}_gt_adaptive.json"

    python scripts/real_comparison_kgbert_matching_eval.py `
        --ground_truth_source test --scale_dir $sd `
        --output "reports/real_comparison_results/kgbert_matching_eval_subgraph_${tag}.json"

    python scripts/real_comparison_atise_matching_eval.py `
        --ground_truth_source test --scale_dir $sd `
        --output "reports/real_comparison_results/atise_matching_eval_subgraph_${tag}.json"

    # LightRAG / HippoRAG：需预先设置 DASHSCOPE_API_KEY（或 QWEN_API_KEY）
    if ($env:RUN_LIGHTRAG_SUBGRAPH -eq "1") {
        python scripts/real_comparison_lightrag.py `
            --scale_dir $sd `
            --output "reports/real_comparison_results/lightrag_results_subgraph_${tag}.json"
    }
    if ($env:RUN_HIPPORAG_SUBGRAPH -eq "1") {
        python scripts/real_comparison_hipporag.py `
            --scale_dir $sd `
            --output "reports/real_comparison_results/hipporag_results_subgraph_${tag}.json"
    }
}

Write-Host "完成。子图结果副本在仓库根 report/（与各脚本约定文件名一致）。" -ForegroundColor Green
