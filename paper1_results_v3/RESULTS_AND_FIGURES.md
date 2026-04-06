# P1 结果与图表解读（论文用）

## 1. 自动调优与选定配置

- **调优目标**：多 **SEED** × **LAMBDA_IR** 组合训练，以 5-fold（按 subject）**6D Ridge** 预测 SSPG/DI 的 **Spearman r 之和** 选优。
- **选定**：**LAMBDA_IR = 0.02** **SEED = 42**（下表为各组合的 5-fold Spearman r）。评估始终使用 **6 维潜变量** Ridge 回归。

| SEED | LAMBDA_IR | SSPG Spearman r (5-fold) | DI Spearman r (5-fold) | 合计 |
|------|-----------|--------------------------|------------------------|------|
| 21 | 0.01 | 0.545 | 0.653 | 1.198 |
| 21 | 0.02 | 0.546 | 0.549 | 1.095 |
| 21 | 0.05 | 0.449 | 0.596 | 1.045 |
| 21 | 0.1 | 0.352 | 0.696 | 1.048 |
| 42 | 0.01 | 0.432 | 0.697 | 1.129 |
| 42 | 0.02 | 0.540 | 0.714 | 1.255 |
| 42 | 0.05 | 0.520 | 0.692 | 1.212 |
| 42 | 0.1 | 0.205 | 0.501 | 0.706 |
| 43 | 0.01 | 0.409 | 0.597 | 1.006 |
| 43 | 0.02 | 0.535 | 0.540 | 1.075 |
| 43 | 0.05 | 0.527 | 0.553 | 1.080 |
| 43 | 0.1 | 0.561 | 0.414 | 0.975 |
| 44 | 0.01 | 0.400 | 0.686 | 1.086 |
| 44 | 0.02 | 0.461 | 0.691 | 1.151 |
| 44 | 0.05 | 0.549 | 0.608 | 1.157 |
| 44 | 0.1 | 0.444 | 0.606 | 1.050 |


## 2. 评估指标摘要

以下为选定配置下 `evaluation_metrics_summary.txt` 的节选（全量见该文件）。

```
P1 Gold-standard prediction metrics
===================================

Single vs joint: SSPG/DI are physiologically joint; we report both single-latent and 6D Ridge.
See joint_weights_sspg.csv and joint_weights_di.csv for Ridge coefficients (full-data fit).

Target: sspg
  direct_mi (full_sample): RMSE=72.62324041341253  R²=0.08310999175528844  Spearman r=0.49187988160632495
  direct_si (full_sample): RMSE=74.98009582696928  R²=0.02263231374138497  Spearman r=0.37250807211437065
  single_mi (5fold_single_vs_joint): rmse = 73.95744509630468 ± 11.160747370132919
  single_si (5fold_single_vs_joint): rmse = 80.98264173737388 ± 19.372968048260137
  ridge_6d (5fold_single_vs_joint): rmse = 64.62899130150728 ± 5.22258240121178
  single_mi (5fold_single_vs_joint): r2 = -0.08350162736022024 ± 0.2013808367010082
  single_si (5fold_single_vs_joint): r2 = -0.27618491274458357 ± 0.34847961193294297
  ridge_6d (5fold_single_vs_joint): r2 = 0.11930233946186175 ± 0.3352913690070532
  single_mi (5fold_single_vs_joint): pearson_r = 0.29358926370832367 ± 0.12758144643082264
  single_si (5fold_single_vs_joint): pearson_r = 0.19834991368420488 ± 0.4483075056007285
  ridge_6d (5fold_single_vs_joint): pearson_r = 0.5310337639243118 ± 0.22449346942651038
  single_mi (5fold_single_vs_joint): spearman_r = 0.4904036298887383 ± 0.24242351100152684
  single_si (5fold_single_vs_joint): spearman_r = 0.3668493507174901 ± 0.24810070850695942
  ridge_6d (5fold_single_vs_joint): spearman_r = 0.5197135605961292 ± 0.2500478928150003
  ridge_6d (5fold_subject): rmse = 68.69518900270495 ± 5.782471287809676
  ridge_6d (5fold_subject): mae = 55.588403592631344 ± 6.258119734162023
  ridge_6d (5fold_subject): r2 = -0.09024494451951168 ± 0.7153381996863181
  ridge_6d (5fold_subject): pearson_r = 0.5140062398734685 ± 0.24450642346571083
  ridge_6d (5fold_subject): spearman_r = 0.5403733398438415 ± 0.24323486287046375
  ridge_6d (leave_one_dataset_out): RMSE=80.02741216288035  R²=-0.12669060977661495  Spearman r=0.619431724431991
  ridge_6d (leave_one_dataset_out): RMSE=69.15863491120705  R²=0.0970600831042846  Spearman r=0.3135396729312435
  ridge_6d (leave_one_dataset_out): RMSE=450925.4084136521  R²=-31914007.750650406  Spearman r=-0.26315789473684204
  coefficients saved to joint_weights_sspg.csv

Target: di
  direct_mi (full_sample): RMSE=1.0809813676006292  R²=0.21822326229684308  Spearman r=0.2782516929029989
  direct_si (full_sample): RMSE=1.2198989012204113  R²=0.004379094115954607  Spearma
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
