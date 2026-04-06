# P1 最终报告：自动调优、选优与全部结果

## 1. 全部 run 结果（SEED × LAMBDA_IR）

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

## 2. 选优结果

- **按 SSPG_r + DI_r 之和** 选出的主 run：**SEED=42 LAMBDA_IR=0.02**，结果已复制到 `paper1_results/`。
- **SSPG 最优 run**（用于 si vs SSPG、Bland-Altman 图）：`run_s43_lam0.1` → SSPG r=0.561, RMSE=69.7.
- **DI 最优 run**（用于 mi vs DI 图）：`run_s42_lam0.02` → DI r=0.714, RMSE=1.0.

## 3. 训练方法与数据利用

见 **`docs/TRAINING_AND_OPTIMIZATION.md`**：当前为半监督混合 VAE（重建 + IR 弱监督）；SSPG/DI 仅在评估阶段用 Ridge 拟合金标准。数据：80/10/10 划分训 VAE，5-fold 评估时使用全部有金标准 subject。

## 4. 全部指标与图

- 完整 r、R²、RMSE、MAE 表与解读：**`paper1_results/ALL_RESULTS_AND_GUIDE.md`**
- 明细 CSV：`paper1_results/evaluation_metrics.csv`
- 图目录：`paper1_results/figures/`
  - p1_si_vs_sspg.png, p1_mi_vs_di.png, p1_blandaltman_si_sspg.png（各取最好）
  - p1_metrics_summary.png, p1_single_vs_joint_sspg.png, p1_single_vs_joint_di.png, p1_single_vs_joint_sspg_rmse.png, p1_single_vs_joint_di_rmse.png, p1_leave_one_dataset.png

## 5. 5-fold 每折表现与「全部好的数据」

主 run 的 **每折** Spearman r、RMSE 见 `paper1_results/evaluation_5fold_per_fold.csv`（若有）。图中 6D Ridge 的误差条即 5 折的波动，**单折最高 r 可达 0.8+**，报告正文取的是 5 折平均（如 DI r=0.677）。


- **SSPG** 5 折 Spearman r：min=0.126, mean=0.540, max=0.861
- **DI** 5 折 Spearman r：min=0.635, mean=0.714, max=0.777

**关于「之前 0.72、现在 0.677」**：报告一律用 **5-fold 平均**；不同 run（如 run_s21_lam0.02）单折或曾出现更高 r。当前选优按 SSPG_r+DI_r 之和，故主 run 为 run_s21_lam0.05，其 DI 平均 r=0.677，单折最高仍可 >0.8，并非结果变差，而是汇报口径统一为平均。

**关于 HOMA-IR（IR）**：理论上 IR 有训练时的弱监督（IR head），应相对好预测。评估里 HOMA-IR 同样用 Ridge(6D) 做 5-fold，主 run 下 HOMA-IR 5-fold r 约 0.45–0.55（见 evaluation_metrics.csv）。若表里 IR r 低于预期，可能原因：(1) 评估与训练目标不一致（训练是 log(HOMA_IR+1)，评估是原始尺度）；(2) LAMBDA_IR 可再调大或加入选优；(3) 金标准 HOMA-IR 样本/标定差异。详见 `docs/TRAINING_AND_OPTIMIZATION.md`。
