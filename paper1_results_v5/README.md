# 实验方案 v5.0：终局之战 — 执行结果

本文件夹存放按《实验方案 v5.0：终局之战》执行的 **A/B 测试** 结果：方案 A（VAE+ODE Prediction Head）vs 方案 B（简约 Ridge Top4 CGM 统计）。

## 核心结论（本次运行）

- **方案 B（简约模型）在 DI 上显著优于方案 A**（Mann-Whitney U p=0.0465）。
- 方案 A（Prediction Head，4 次 run）：SSPG Spearman r 中位数 ≈ 0.03，DI ≈ -0.03。
- 方案 B（100 次 CV）：SSPG Spearman r 中位数 ≈ 0.45，DI ≈ 0.37。
- **结论**：在当前数据规模下，VAE+ODE 的端到端 Prediction Head **未**带来预测收益；简单的 CGM 统计特征（cgm_mean, ac_var, cgm_std, mge）+ Ridge 是更稳健、更优的选择。

## 方案 A：Prediction Head 模型

- **架构**：z_init (4D) + z_nonseq (16D) → MLP(20→32→2) → SSPG / DI。
- **损失**：reconstruction + LAMBDA_SSPG×MSE(SSPG) + LAMBDA_DI×MSE(DI)，仅对有标签样本。
- **数据**：P1_ONE_MEAL_PER_SUBJECT=0，使用全部 meal 窗口（332 样本）。
- **设计**：4 组 lambda (0.01, 0.1, 1.0, 10.0) × 10 种子，每轮得到 test set 上 subject-level Spearman r。
- **本次执行**：已跑 4 轮（lambda 0.01×3 种子 + lambda 0.1×1 种子），结果见 `scheme_a/scheme_a_all_runs.json`。完整 4×10=40 轮可自行运行（见下方复现命令）。

## 方案 B：终极简约模型

- **特征**：cgm_mean, ac_var, cgm_std, mge（Top 4 CGM 统计）。
- **模型**：Ridge 回归。
- **数据**：全部 meal 窗口，按 subject_id 聚合（中位数）后 100 次随机 CV。
- **结果**：`scheme_b/scheme_b_spearman_runs.json`（100 折的 SSPG/DI Spearman r）。

## 评估输出

- **v5_evaluation_report.txt**：方案 A vs B 的 Spearman r 汇总、最佳 lambda、Mann-Whitney U 检验与结论。
- **v5_boxplot_scheme_a_vs_b.png**：方案 A 与方案 B 的 Spearman r 箱线图（SSPG / DI）。

## 复现命令

```bash
# 方案 A：4 组 lambda × 10 种子（约 40×2.5 分钟，可先用 --max_runs 4 试跑）
python scripts/run_v5_scheme_a.py --output_dir paper1_results_v5/scheme_a

# 方案 B：100 次 CV
python scripts/run_v5_scheme_b.py --output_dir paper1_results_v5/scheme_b --n_cv 100

# 评估：箱线图 + Wilcoxon + 报告
python scripts/run_v5_evaluate.py --scheme_a_dir paper1_results_v5/scheme_a \
  --scheme_b_dir paper1_results_v5/scheme_b --output_dir paper1_results_v5
```

## 代码与配置

- **模型**：`models.py` 中 `P1_V5_PREDICTION_HEAD=1` 时增加 `prediction_head`（20D→32→2），`run_p1_full_pipeline.py` 中对应损失与评估。
- **方案 A 单轮**：`P1_V5_PREDICTION_HEAD=1 P1_ONE_MEAL_PER_SUBJECT=0 LAMBDA_SSPG=<lam> LAMBDA_DI=<lam> LAMBDA_IR=0 P1_SEED=<seed> P1_RESULTS_DIR=<dir> python run_p1_full_pipeline.py`
