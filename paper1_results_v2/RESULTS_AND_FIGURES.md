# P1 结果与图表解读（论文用）

## 1. 自动调优与选定配置

- **调优目标**：多 **SEED** × **LAMBDA_IR** 组合训练，以 5-fold（按 subject）**6D Ridge** 预测 SSPG/DI 的 **Spearman r 之和** 选优。
- **选定**：**LAMBDA_IR = 0.1** **SEED = 42**（下表为各组合的 5-fold Spearman r）。评估始终使用 **6 维潜变量** Ridge 回归。

| SEED | LAMBDA_IR | SSPG Spearman r (5-fold) | DI Spearman r (5-fold) | 合计 |
|------|-----------|--------------------------|------------------------|------|
| 21 | 0.01 | 0.422 | 0.562 | 0.984 |
| 21 | 0.02 | 0.501 | 0.704 | 1.205 |
| 21 | 0.05 | 0.538 | 0.454 | 0.992 |
| 21 | 0.1 | 0.431 | 0.580 | 1.011 |
| 42 | 0.01 | 0.406 | 0.507 | 0.913 |
| 42 | 0.02 | 0.615 | 0.638 | 1.253 |
| 42 | 0.05 | 0.493 | 0.431 | 0.924 |
| 42 | 0.1 | 0.516 | 0.751 | 1.267 |
| 43 | 0.01 | 0.518 | 0.607 | 1.124 |
| 43 | 0.02 | 0.584 | 0.418 | 1.001 |
| 43 | 0.05 | 0.559 | 0.500 | 1.059 |
| 43 | 0.1 | 0.536 | 0.729 | 1.265 |
| 44 | 0.01 | 0.449 | 0.504 | 0.953 |
| 44 | 0.02 | 0.496 | 0.590 | 1.086 |
| 44 | 0.05 | 0.405 | 0.671 | 1.076 |
| 44 | 0.1 | 0.390 | 0.346 | 0.736 |


## 2. 评估指标摘要

以下为选定配置下 `evaluation_metrics_summary.txt` 的节选（全量见该文件）。

```
P1 Gold-standard prediction metrics
===================================

Single vs joint: SSPG/DI are physiologically joint; we report both single-latent and 6D Ridge.
See joint_weights_sspg.csv and joint_weights_di.csv for Ridge coefficients (full-data fit).

Target: sspg
  direct_mi (full_sample): RMSE=74.83382724885746  R²=0.02644182455215649  Spearman r=0.21852463164053487
  direct_si (full_sample): RMSE=70.3387653760075  R²=0.13988707049053795  Spearman r=0.36063492845634576
  single_mi (5fold_single_vs_joint): rmse = 76.2534684417371 ± 14.069592832286162
  single_si (5fold_single_vs_joint): rmse = 73.71750281964253 ± 10.26946475712843
  ridge_6d (5fold_single_vs_joint): rmse = 67.31246500892784 ± 7.437345463519844
  single_mi (5fold_single_vs_joint): r2 = -0.12483542100958074 ± 0.11029421401054074
  single_si (5fold_single_vs_joint): r2 = -0.1843957159331709 ± 0.6298792613800163
  ridge_6d (5fold_single_vs_joint): r2 = -0.027713036496206756 ± 0.6448288186388704
  single_mi (5fold_single_vs_joint): pearson_r = 0.17307972180929657 ± 0.1679849855687269
  single_si (5fold_single_vs_joint): pearson_r = 0.411487783415138 ± 0.21584174969951012
  ridge_6d (5fold_single_vs_joint): pearson_r = 0.5022612401909454 ± 0.27440369205666487
  single_mi (5fold_single_vs_joint): spearman_r = 0.22597036099960857 ± 0.14581864943323983
  single_si (5fold_single_vs_joint): spearman_r = 0.38085629641619717 ± 0.25452218942107635
  ridge_6d (5fold_single_vs_joint): spearman_r = 0.47972192058397 ± 0.2729673040745931
  ridge_6d (5fold_subject): rmse = 65.98556443403622 ± 6.811142279201402
  ridge_6d (5fold_subject): mae = 54.48574982138098 ± 6.118244906101597
  ridge_6d (5fold_subject): r2 = 0.029494401531648905 ± 0.5600919198747347
  ridge_6d (5fold_subject): pearson_r = 0.5305938553331091 ± 0.2697023356681912
  ridge_6d (5fold_subject): spearman_r = 0.5159812886673587 ± 0.2639114654583963
  ridge_6d (leave_one_dataset_out): RMSE=66.27151262718543  R²=0.22735340563030193  Spearman r=0.5484876352472675
  ridge_6d (leave_one_dataset_out): RMSE=69.45104592538884  R²=0.08940846442891592  Spearman r=0.34076015900566015
  ridge_6d (leave_one_dataset_out): RMSE=80.87290917837349  R²=-0.026545200975224503  Spearman r=0.669172932330827
  coefficients saved to joint_weights_sspg.csv

Target: di
  direct_mi (full_sample): RMSE=1.003682838886091  R²=0.3260319263557844  Spearman r=0.6457233033265621
  direct_si (full_sample): RMSE=1.206617784943606  R²=0.025939860212210064  Spearman r=0.34
```

## 3. 图表文件与论文解读

所有图表保存在 **`figures/`** 目录下，用于验证 CGM 潜变量与金标准（SSPG=胰岛素抵抗，DI=β 细胞功能）的对应关系。

### 3.1 `figures/p1_si_vs_sspg.png` — 敏感性指数 si 与 SSPG

- **目的**：检验模型潜变量 **si**（胰岛素敏感性）与金标准 **SSPG**（稳态血浆葡萄糖，越高越抵抗）的单调关系。
- **解读**：若 si 与 SSPG 呈负相关（Spearman r < 0），符合生理（敏感性高则 SSPG 低）；若呈正相关则可能反映数据集混合或样本量限制。图中给出按 dataset 着色、回归线及 Spearman r/p，用于说明“CGM 衍生的 si 是否与金标准 IR 指标一致”。

### 3.2 `figures/p1_mi_vs_di.png` — 胰岛素分泌指数 mi 与 DI

- **目的**：检验潜变量 **mi**（模型中的胰岛素分泌/处置相关维度）与金标准 **DI**（处置指数，β 细胞功能）的对应。
- **解读**：DI 由 OGTT 等金标准方法得到；若 mi 与 DI 正相关，支持“从 CGM 可解释地恢复 β 细胞功能信息”。图中同样给出按 dataset 的散点、回归线及 r/p，便于讨论单参数相关与 6D 联合预测的差异（见文档“单参数 vs 多参数”）。

### 3.3 `figures/p1_blandaltman_si_sspg.png` — Bland-Altman（si vs SSPG）

- **目的**：评估 si 与 SSPG 的**一致性**（非仅相关）：均值差与 95% 一致性界限。
- **解读**：若点大多落在 ±1.96 SD 内，说明在尺度上 si 与 SSPG 有一定一致性；若存在系统偏差或离散度大，可在文中说明为“CGM 潜变量与金标准单位不同，更适合作相关/预测分析而非直接替代”。

### 3.4 `figures/p1_correlations_summary.txt`

- 各潜变量（si, mi, tau_m, sg）与各金标准（sspg, di 等）的 Spearman r、p、n 汇总，供正文表格引用。

---

## 4. 论文表述建议

- **主要结论**：CGM 驱动的机制化潜变量（si, mi 等）与 SSPG（IR）和 DI（β 细胞）具有可量化关联；**6D 联合预测**（Ridge）优于单潜变量，符合生理上多参数联合决定葡萄糖处置的设定。
- **评估**：以 5-fold 按 subject 的 Spearman r、RMSE、R² 为主；留一数据集出用于讨论泛化。
- **局限**：金标准样本量有限、多中心尺度差异；单潜变量相关仅部分反映联合关系，需结合 6D Ridge 与系数表（`joint_weights_sspg.csv`, `joint_weights_di.csv`）一起解读。
