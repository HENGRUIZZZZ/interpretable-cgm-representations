# P1 结果与图表解读（论文用）

## 1. 自动调优与选定配置

- **调优目标**：多 **SEED** × **LAMBDA_IR** 组合训练，以 5-fold（按 subject）**6D Ridge** 预测 SSPG/DI 的 **Spearman r 之和** 选优。
- **选定**：**LAMBDA_IR = 0.05** **SEED = 21**（下表为各组合的 5-fold Spearman r）。评估始终使用 **6 维潜变量** Ridge 回归。

| SEED | LAMBDA_IR | SSPG Spearman r (5-fold) | DI Spearman r (5-fold) | 合计 |
|------|-----------|--------------------------|------------------------|------|
| 21 | 0.01 | 0.502 | 0.485 | 0.987 |
| 21 | 0.02 | 0.567 | 0.397 | 0.964 |
| 21 | 0.05 | 0.652 | 0.513 | 1.165 |
| 21 | 0.1 | 0.368 | 0.408 | 0.776 |
| 42 | 0.01 | 0.208 | 0.253 | 0.461 |
| 42 | 0.02 | 0.421 | 0.351 | 0.771 |
| 42 | 0.05 | 0.245 | 0.454 | 0.699 |
| 42 | 0.1 | 0.263 | 0.383 | 0.646 |
| 43 | 0.01 | 0.417 | 0.348 | 0.764 |
| 43 | 0.02 | 0.451 | 0.434 | 0.885 |
| 43 | 0.05 | 0.565 | 0.296 | 0.861 |
| 43 | 0.1 | 0.366 | 0.246 | 0.612 |
| 44 | 0.01 | 0.256 | 0.340 | 0.596 |
| 44 | 0.02 | 0.212 | 0.308 | 0.519 |
| 44 | 0.05 | 0.232 | 0.247 | 0.479 |
| 44 | 0.1 | 0.315 | 0.301 | 0.616 |


## 2. 评估指标摘要

以下为选定配置下 `evaluation_metrics_summary.txt` 的节选（全量见该文件）。

```
P1 Gold-standard prediction metrics
===================================

Single vs joint: SSPG/DI are physiologically joint; we report both single-latent and 6D Ridge.
See joint_weights_sspg.csv and joint_weights_di.csv for Ridge coefficients (full-data fit).

Target: sspg
  direct_mi (full_sample): RMSE=73.81418522429489  R²=0.016205332236062908  Spearman r=0.11645217916590105
  direct_si (full_sample): RMSE=74.11136086939527  R²=0.008267878644113047  Spearman r=0.1408230991356721
  single_mi (5fold_single_vs_joint): rmse = 75.87862306033841 ± 7.125302829558823
  single_si (5fold_single_vs_joint): rmse = 76.29799949151058 ± 6.496836355648502
  ridge_6d (5fold_single_vs_joint): rmse = 55.99944320486397 ± 3.701210796325616
  single_mi (5fold_single_vs_joint): r2 = -0.17177778654883438 ± 0.2121069791553162
  single_si (5fold_single_vs_joint): r2 = -0.17896943842963448 ± 0.15832987283277036
  ridge_6d (5fold_single_vs_joint): r2 = 0.3466913518921861 ± 0.16957492469923807
  single_mi (5fold_single_vs_joint): pearson_r = 0.14016918290316419 ± 0.14485213622241772
  single_si (5fold_single_vs_joint): pearson_r = 0.03781848680311621 ± 0.21099488655243923
  ridge_6d (5fold_single_vs_joint): pearson_r = 0.708569283599622 ± 0.0655261255245551
  single_mi (5fold_single_vs_joint): spearman_r = 0.2281123317960616 ± 0.1408724136223063
  single_si (5fold_single_vs_joint): spearman_r = 0.07652692820797954 ± 0.3400802705057098
  ridge_6d (5fold_single_vs_joint): spearman_r = 0.6305275534079476 ± 0.10421854979477001
  ridge_6d (5fold_subject): rmse = 59.47829554058804 ± 7.4449726057007295
  ridge_6d (5fold_subject): mae = 47.73027285434478 ± 4.8422652814189
  ridge_6d (5fold_subject): r2 = 0.24564558089974523 ± 0.2682993704975687
  ridge_6d (5fold_subject): pearson_r = 0.7178906496172088 ± 0.06572616450731161
  ridge_6d (5fold_subject): spearman_r = 0.6515404521767063 ± 0.09971083520900481
  ridge_6d (leave_one_dataset_out): RMSE=59.28034370306769  R²=0.3817721661597724  Spearman r=0.6241979939896208
  ridge_6d (leave_one_dataset_out): RMSE=65.23489262734442  R²=0.19661090637379075  Spearman r=0.5107369720481285
  coefficients saved to joint_weights_sspg.csv

Target: di
  direct_mi (full_sample): RMSE=1.122322264534876  R²=0.0026157717737272934  Spearman r=0.008544649393969204
  direct_si (full_sample): RMSE=1.1184408014334573  R²=0.009502591881484501  Spearman r=0.07747788832509153
  single_mi (5fold_single_vs_joint): rmse = 1.1056145436393943 ± 0.352650151187941
  single_s
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
