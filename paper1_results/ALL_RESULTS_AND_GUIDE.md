# P1 全部结果汇总：r、R²、RMSE、MAE 与解读

本文档汇总 **SSPG、DI、HOMA-IR、HOMA-B** 的完整评估指标，并说明**如何理解、如何优化**。图见 `figures/` 目录。

---

## 一、指标含义（如何理解）


| 指标             | 含义                      | 怎么读                | 越好的表现                                     |
| -------------- | ----------------------- | ------------------ | ----------------------------------------- |
| **Spearman r** | 预测值与金标准的**秩相关**（单调关系强弱） | 与顺序是否一致；对异常值稳健     | |r| 接近 1，SSPG 期望负相关（si 高→SSPG 低），DI 期望正相关 |
| **Pearson r**  | **线性相关**                | 直线拟合程度             | |r| 接近 1                                  |
| **R²**         | 决定系数，预测解释了金标准多少方差       | R²=1−SS_res/SS_tot | 接近 1 好；**5-fold 小 test 时常见负值**，不单独用来否定结果  |
| **RMSE**       | 均方根误差，与金标准同单位           | 平均预测误差大小           | 越小越好（SSPG 约 mg/dL，DI 无量纲）                 |
| **MAE**        | 平均绝对误差                  | 平均偏差幅度             | 越小越好                                      |


- **full_sample**：全量拟合后在同一批数据上算指标，易过拟合，仅作描述。
- **5-fold (by subject)**：按人划分 5 折，每折用 4 份训 Ridge、1 份测，报告 5 次结果的 mean±std；**主看这个**。
- **leave_one_dataset_out**：用两个数据集训、在第三个上测，看跨数据集泛化。

---

## 二、SSPG（胰岛素抵抗）完整结果

### 2.1 当前 run（6D Ridge，5-fold 最优 SSPG run：run_s21_lam0.05）


| 验证方式                    | 方法                 | Spearman r       | Pearson r        | R²           | RMSE           | MAE            | n   |
| ----------------------- | ------------------ | ---------------- | ---------------- | ------------ | -------------- | -------------- | --- |
| full_sample             | direct_mi          | 0.31             | 0.30             | 0.09         | 72.2           | 61.0           | 83  |
| full_sample             | direct_si          | 0.37             | 0.34             | 0.11         | 71.5           | 57.8           | 83  |
| 5fold_subject           | ridge_6d           | **0.454** ± 0.34 | **0.426** ± 0.31 | -0.10 ± 0.33 | **73.3** ± 9.2 | **62.2** ± 7.8 | 83  |
| 5fold (single vs joint) | single_mi          | 0.30 ± 0.33      | 0.33 ± 0.32      | -0.17 ± 0.41 | 75.3 ± 10.2    | —              | 83  |
| 5fold (single vs joint) | single_si          | 0.37 ± 0.26      | 0.33 ± 0.34      | -0.20 ± 0.46 | 75.7 ± 7.5     | —              | 83  |
| 5fold (single vs joint) | ridge_6d           | 0.37 ± 0.28      | 0.35 ± 0.29      | -0.15 ± 0.45 | 74.0 ± 7.2     | —              | 83  |
| leave_one_dataset_out   | ridge_6d (test=D1) | 0.34             | 0.33             | 0.03         | 74.2           | 62.2           | 32  |
| leave_one_dataset_out   | ridge_6d (test=D2) | 0.34             | 0.27             | -0.10        | 76.3           | 63.2           | 31  |
| leave_one_dataset_out   | ridge_6d (test=D4) | **0.63**         | **0.67**         | **0.32**     | **65.7**       | 49.3           | 20  |


**解读**：5-fold 下 6D Ridge 的 SSPG Spearman r≈0.45，RMSE≈73 mg/dL；单潜变量 mi/si 略逊或相当。留一数据集出在 D4 上 r 最高（0.63），说明在部分数据上泛化尚可。

---

## 三、DI（β 细胞功能）完整结果

### 3.1 当前 run（6D Ridge，5-fold 最优 DI run：tune_0.01 或 run_s21_lam0.05）


| 验证方式                    | 方法                 | Spearman r       | Pearson r        | R²              | RMSE            | MAE             | n   |
| ----------------------- | ------------------ | ---------------- | ---------------- | --------------- | --------------- | --------------- | --- |
| full_sample             | direct_mi          | 0.10             | 0.02             | 0.00            | 1.22            | 0.97            | 85  |
| full_sample             | direct_si          | 0.21             | 0.14             | 0.02            | 1.21            | 0.95            | 85  |
| 5fold_subject           | ridge_6d           | **0.677** ± 0.19 | **0.569** ± 0.19 | **0.17** ± 0.23 | **1.04** ± 0.22 | **0.79** ± 0.09 | 85  |
| 5fold (single vs joint) | single_mi          | 0.03 ± 0.19      | -0.05 ± 0.11     | -0.09 ± 0.05    | 1.21 ± 0.27     | —               | 85  |
| 5fold (single vs joint) | single_si          | 0.29 ± 0.07      | 0.18 ± 0.08      | -0.07 ± 0.05    | 1.20 ± 0.28     | —               | 85  |
| 5fold (single vs joint) | ridge_6d           | **0.64** ± 0.18  | **0.56** ± 0.19  | 0.08 ± 0.36     | 1.07 ± 0.22     | —               | 85  |
| leave_one_dataset_out   | ridge_6d (test=D1) | 0.33             | 0.48             | -0.23           | 1.06            | 0.78            | 32  |
| leave_one_dataset_out   | ridge_6d (test=D2) | 0.34             | 0.22             | -1.06           | 1.80            | 1.37            | 31  |
| leave_one_dataset_out   | ridge_6d (test=D4) | -0.42            | -0.38            | —               | 1.92            | 1.90            | 22  |


**解读**：5-fold 下 6D Ridge 的 DI Spearman r≈0.68，R²≈0.17，RMSE≈1.04；**明显优于单潜变量**（single mi/si 的 r 接近 0 或弱）。留一数据集出在 D4 上不稳定（尺度/定义差异），D1/D2 上 r 约 0.33–0.34。

---

## 四、HOMA-IR / HOMA-B 简要结果（5-fold ridge_6d）


| 目标      | Spearman r   | Pearson r    | R²           | RMSE        | MAE         | n   |
| ------- | ------------ | ------------ | ------------ | ----------- | ----------- | --- |
| homa_ir | 0.547 ± 0.12 | 0.490 ± 0.11 | 0.13 ± 0.21  | 1.18 ± 0.21 | 0.83 ± 0.09 | 97  |
| homa_b  | 0.112 ± 0.28 | 0.057 ± 0.30 | -0.14 ± 0.17 | 53.8 ± 8.8  | 38.5 ± 5.6  | 97  |


---

## 五、图在哪里、分别表示什么


| 文件                              | 内容                               | 理解要点                             |
| ------------------------------- | -------------------------------- | -------------------------------- |
| **p1_si_vs_sspg.png**           | si（模型敏感性） vs SSPG（金标准）散点         | 负相关符合生理；来自 SSPG 最优 run           |
| **p1_mi_vs_di.png**             | mi（模型） vs DI（金标准）散点              | 正相关支持 CGM 反映 β 细胞功能；来自 DI 最优 run |
| **p1_blandaltman_si_sspg.png**  | si 与 SSPG 的一致性（均值差 ±1.96SD）      | 看系统偏差与离散度；单位不同时适合作相关分析           |
| **p1_metrics_summary.png**      | 5-fold 下 SSPG/DI 的 r、R²、RMSE 柱状图 | 一眼对比两个目标与指标                      |
| **p1_single_vs_joint.png**      | 单潜变量 vs 6D Ridge 的 Spearman r 对比 | 说明“联合预测”优于单参数                    |
| **p1_leave_one_dataset.png**    | 留一数据集出时各 test 集的 r / RMSE        | 看跨数据集泛化                          |
| **p1_correlations_summary.txt** | 各潜变量与各金标准的 r、p、n 表               | 制表引用                             |


---

## 六、如何优化

1. **数据**：仅用同时有 SSPG+DI 的受试者评估；按 dataset 分层划分；增加金标准样本量。
2. **训练**：多 seed 已做；可调 LAMBDA_IR（0.01/0.05/0.1）、BETA_HAT、epoch；**SSPG 与 DI 分开选优**（出图已按“各取最好”）。
3. **评估**：Ridge alpha 已网格搜索；可试 log(SSPG)、log(DI) 预测；对留一数据集出的极端 R² 单独说明（尺度差异）。
4. **报告**：主看 **5-fold Spearman r** 和 **RMSE**；R² 为正时写“有预测方差解释”，为负时写“更适合作相关/排序分析”。

---

## 七、原始数据与脚本

- 明细表：`paper1_results/evaluation_metrics.csv`
- 文本摘要：`paper1_results/evaluation_metrics_summary.txt`
- 出图（含各取最好）：`python scripts/plot_p1_results.py --best-per-target`
- 汇总图（指标柱状、单 vs 联合、留一数据集）：  
`python scripts/plot_p1_results.py --best-per-target --summary-figures --out paper1_results/figures`

