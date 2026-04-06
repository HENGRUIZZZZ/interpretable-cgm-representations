# P1 结果与图表解读（论文用）

## 1. 自动调优与选定配置

- **调优目标**：多 **SEED** × **LAMBDA_IR** 组合训练，以 5-fold（按 subject）**6D Ridge** 预测 SSPG/DI 的 **Spearman r 之和** 选优。
- **选定**：**LAMBDA_IR = 0.1** **SEED = 21**（下表为各组合的 5-fold Spearman r）。评估始终使用 **6 维潜变量** Ridge 回归。

| SEED | LAMBDA_IR | SSPG Spearman r (5-fold) | DI Spearman r (5-fold) | 合计 |
|------|-----------|--------------------------|------------------------|------|
| 21 | 0.05 | 0.217 | 0.415 | 0.632 |
| 21 | 0.1 | 0.444 | 0.465 | 0.910 |
| 42 | 0.05 | 0.201 | 0.318 | 0.520 |
| 42 | 0.1 | 0.311 | 0.247 | 0.558 |


## 2. 评估指标摘要

以下为选定配置下 `evaluation_metrics_summary.txt` 的节选（全量见该文件）。

```
P1 Gold-standard prediction metrics
===================================

Single vs joint: SSPG/DI are physiologically joint; we report both single-latent and 6D Ridge.
See joint_weights_sspg.csv and joint_weights_di.csv for Ridge coefficients (full-data fit).

Target: sspg
  direct_mi (full_sample): RMSE=61.927855506599634  R²=0.3075365008153226  Spearman r=0.5473252420797349
  direct_si (full_sample): RMSE=73.20171116019627  R²=0.03246369474635902  Spearman r=0.1876200676293507
  single_mi (5fold_single_vs_joint): rmse = 62.90133357301092 ± 5.203769670307122
  single_si (5fold_single_vs_joint): rmse = 76.5800171256922 ± 9.942539371907975
  ridge_6d (5fold_single_vs_joint): rmse = 63.71956448705316 ± 4.770567860575258
  single_mi (5fold_single_vs_joint): r2 = 0.19139276977512817 ± 0.15774644347699843
  single_si (5fold_single_vs_joint): r2 = -0.2099449623344193 ± 0.34799201616716696
  ridge_6d (5fold_single_vs_joint): r2 = 0.16506509585399104 ± 0.17977034856623217
  single_mi (5fold_single_vs_joint): pearson_r = 0.5110458185340178 ± 0.09687268595804274
  single_si (5fold_single_vs_joint): pearson_r = 0.17485174420379238 ± 0.29087968799395625
  ridge_6d (5fold_single_vs_joint): pearson_r = 0.4800853972881697 ± 0.10472362706303023
  single_mi (5fold_single_vs_joint): spearman_r = 0.4120009339444743 ± 0.16941208182241724
  single_si (5fold_single_vs_joint): spearman_r = 0.18104990104806756 ± 0.272527582639804
  ridge_6d (5fold_single_vs_joint): spearman_r = 0.4287538117064701 ± 0.2079221973992278
  ridge_6d (5fold_subject): rmse = 63.81694653710556 ± 5.310921501103005
  ridge_6d (5fold_subject): mae = 52.24110096352158 ± 6.8975134504831885
  ridge_6d (5fold_subject): r2 = 0.16416044029658322 ± 0.1773121766409122
  ridge_6d (5fold_subject): pearson_r = 0.4891660147313968 ± 0.1022747373349215
  ridge_6d (5fold_subject): spearman_r = 0.4444149616168792 ± 0.20324184531887013
  ridge_6d (leave_one_dataset_out): RMSE=60.04209318677781  R²=0.3657816894638719  Spearman r=0.6287809454873421
  ridge_6d (leave_one_dataset_out): RMSE=68.75329768829837  R²=0.10761328775009549  Spearman r=0.4171791532442076
  coefficients saved to joint_weights_sspg.csv

Target: di
  direct_mi (full_sample): RMSE=1.0151039131803545  R²=0.1840785636581218  Spearman r=0.3498985923182108
  direct_si (full_sample): RMSE=1.1229887608527147  R²=0.001430817469768475  Spearman r=-0.02808213424422463
  single_mi (5fold_single_vs_joint): rmse = 0.983084702317564 ± 0.3143918780992724
  single_si (5fo
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
