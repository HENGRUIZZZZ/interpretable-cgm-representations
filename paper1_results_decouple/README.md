# P1 解耦实验 (P1_Decoupling_Experiment_Manual v1.0)

本目录为 **si/mi 解耦实验** 结果：SSPG 监督头仅使用 **si (1维)** 作为输入，DI 头仍使用 6D latent，以实现潜在空间中 si 与 mi 的解耦。

## 代码修改

- **run_p1_full_pipeline.py**：通过环境变量 `P1_DECOUPLE_SSPG=1` 启用解耦模式。
  - `sspg_head` 改为 `Linear(1, 1)`，训练与评估时 SSPG 仅用 `latent_all[:, 3:4]`（si）。
  - DI 头保持 6D 输入不变。

## 实验与 5 折结果

| 实验 | λ_sspg | λ_di | SSPG r (5折) | DI r (5折) |
|------|--------|------|--------------|------------|
| m1_decouple | 0.1 | 0.1 | 0.416 ± 0.272 | 0.226 ± 0.252 |
| m1_decouple_sspg0.2 | 0.2 | 0.05 | 0.414 ± 0.262 | 0.235 ± 0.261 |
| m1_decouple_di0.15 | 0.05 | 0.15 | 0.434 ± 0.288 | 0.291 ± 0.280 |

每实验均含：`latent_and_gold_*.csv`、`e2e_head_metrics.json`、`evaluation_metrics_summary.txt`、`diagnostic/`（VAE 拟合、6D 图、DIAGNOSTIC_REPORT.md）。

## 复现命令

```bash
export CGM_PROJECT_OUTPUT=/path/to/cgm_project/output
export P1_ZSCORE_TARGETS=1 BETA_HAT=0.05 P1_DECOUPLE_SSPG=1

# 核心解耦
P1_RESULTS_DIR=paper1_results_decouple/m1_decouple LAMBDA_SSPG=0.1 LAMBDA_DI=0.1 python run_p1_full_pipeline.py

# 可选：增强 SSPG / 增强 DI
P1_RESULTS_DIR=paper1_results_decouple/m1_decouple_sspg0.2 LAMBDA_SSPG=0.2 LAMBDA_DI=0.05 python run_p1_full_pipeline.py
P1_RESULTS_DIR=paper1_results_decouple/m1_decouple_di0.15 LAMBDA_SSPG=0.05 LAMBDA_DI=0.15 python run_p1_full_pipeline.py
```

## 提交包（zip）

- `m1_decouple_results.zip`
- `m1_decouple_sspg0.2_results.zip`
- `m1_decouple_di0.15_results.zip`

将上述 zip 提交后即可进行解耦效果分析与论文图表制作。
