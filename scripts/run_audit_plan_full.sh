#!/bin/bash
# P1 审计计划 — 完整执行脚本（阶段一已跑完，本脚本跑阶段二+三）
set -e
cd "$(dirname "$0")/.."
BASE="P1_WIDE_PARAM_RANGE=1 P1_ONE_MEAL_PER_SUBJECT=1 P1_SEED=21 P1_ZSCORE_TARGETS=1 P1_DECOUPLE_SSPG=1"
OUT="paper1_results_audit_plan"

# ---- 2.1 损失权重扫描：单变量 ----
for lam in 0.01 0.05 0.1 0.2 0.5 1.0; do
  P1_RESULTS_DIR=$OUT/scan_sspg${lam} LAMBDA_SSPG=$lam LAMBDA_DI=0 LAMBDA_ORTHO=0 $BASE python run_p1_full_pipeline.py
done
for lam in 0.01 0.05 0.15 0.3; do
  P1_RESULTS_DIR=$OUT/scan_di${lam} LAMBDA_SSPG=0 LAMBDA_DI=$lam LAMBDA_ORTHO=0 $BASE python run_p1_full_pipeline.py
done
for lam in 0.05 0.1 0.2; do
  P1_RESULTS_DIR=$OUT/scan_ortho${lam} LAMBDA_SSPG=0 LAMBDA_DI=0 LAMBDA_ORTHO=$lam $BASE python run_p1_full_pipeline.py
done

# ---- 2.1 组合扫描 ----
P1_RESULTS_DIR=$OUT/scan_sspg0.1_di0.1 LAMBDA_SSPG=0.1 LAMBDA_DI=0.1 LAMBDA_ORTHO=0 $BASE python run_p1_full_pipeline.py
P1_RESULTS_DIR=$OUT/scan_sspg0.1_di0.15_ortho0.05 LAMBDA_SSPG=0.1 LAMBDA_DI=0.15 LAMBDA_ORTHO=0.05 $BASE python run_p1_full_pipeline.py
P1_RESULTS_DIR=$OUT/scan_sspg0.2_di0.15_ortho0.05 LAMBDA_SSPG=0.2 LAMBDA_DI=0.15 LAMBDA_ORTHO=0.05 $BASE python run_p1_full_pipeline.py

# ---- 2.2 ODE 简化（固定 sg/p2，只学 4 参数）----
P1_FIX_SG_P2=1 P1_RESULTS_DIR=$OUT/ode4param_sspg0.1_di0.15_ortho0.05 LAMBDA_SSPG=0.1 LAMBDA_DI=0.15 LAMBDA_ORTHO=0.05 $BASE python run_p1_full_pipeline.py

# ---- 2.3 先验 si×mi–DI（乘积约束 + 对数乘积）----
P1_DI_PRODUCT_CONSTRAINT=1 P1_RESULTS_DIR=$OUT/prior_prod_sspg0.1_di0.15_ortho0.05 LAMBDA_SSPG=0.1 LAMBDA_DI=0.15 LAMBDA_ORTHO=0.05 $BASE python run_p1_full_pipeline.py
P1_DI_LOG_PRODUCT=1 P1_RESULTS_DIR=$OUT/prior_logprod_sspg0.1_di0.15_ortho0.05 LAMBDA_SSPG=0.1 LAMBDA_DI=0.15 LAMBDA_ORTHO=0.05 $BASE python run_p1_full_pipeline.py

# ---- 阶段三：全量数据训练（最佳配置，不限制每受试者一餐窗）----
P1_ONE_MEAL_PER_SUBJECT=0 P1_RESULTS_DIR=$OUT/final_fullmeal_sspg0.1_di0.15_ortho0.05 LAMBDA_SSPG=0.1 LAMBDA_DI=0.15 LAMBDA_ORTHO=0.05 $BASE python run_p1_full_pipeline.py

echo "Done. Run: python scripts/evaluate_p1_metrics.py --csv $OUT/<dir>/latent_and_gold_all.csv --out $OUT/<dir>  for each dir."
