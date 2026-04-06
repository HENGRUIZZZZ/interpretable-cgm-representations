#!/usr/bin/env bash
# 不训练，只把 P1 的图与报告全部刷新到最新（V4 figures + diagnostic）。
# 在项目根目录执行： bash scripts/rebuild_all_figures_and_reports.sh
# 依赖：已有 paper1_results_v2 / v3 / v4 的 run 目录与 CSV（若缺某 version 可改 P1_RESULTS_ROOT / results-roots）。

set -e
cd "$(dirname "$0")/.."

echo "=== 1. Refresh V4 figures (SI/MI, 6D, VAE if present) ==="
export P1_RESULTS_ROOT=paper1_results_v4
python run_auto_tune_and_report.py --report-only

echo ""
echo "=== 2. Refresh diagnostic (all scenarios linear vs nonlinear + report) ==="
python scripts/run_p1_full_diagnostic.py \
  --results-root paper1_results_v4 \
  --results-roots paper1_results_v2,paper1_results_v3,paper1_results_v4 \
  --out paper1_results_diagnostic

echo ""
echo "Done. Figures: paper1_results_v4/figures/ and paper1_results_diagnostic/figures/"
echo "Report: paper1_results_diagnostic/DIAGNOSTIC_REPORT.md"
