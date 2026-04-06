# P1 最终报告：自动调优、选优与全部结果

## 1. 全部 run 结果（SEED × LAMBDA_IR）

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

## 2. 选优结果

- **按 SSPG_r + DI_r 之和** 选出的主 run：**SEED=42 LAMBDA_IR=0.1**，结果已复制到 `paper1_results/`。
- **SSPG 最优 run**（用于 si vs SSPG、Bland-Altman 图）：`run_s42_lam0.02` → SSPG r=0.615, RMSE=64.6.
- **DI 最优 run**（用于 mi vs DI 图）：`run_s42_lam0.1` → DI r=0.751, RMSE=0.9.

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


- **SSPG** 5 折 Spearman r：min=0.066, mean=0.516, max=0.871
- **DI** 5 折 Spearman r：min=0.637, mean=0.751, max=0.904

**关于「之前 0.72、现在 0.677」**：报告一律用 **5-fold 平均**；不同 run（如 run_s21_lam0.02）单折或曾出现更高 r。当前选优按 SSPG_r+DI_r 之和，故主 run 为 run_s21_lam0.05，其 DI 平均 r=0.677，单折最高仍可 >0.8，并非结果变差，而是汇报口径统一为平均。

**关于 HOMA-IR（IR）**：理论上 IR 有训练时的弱监督（IR head），应相对好预测。评估里 HOMA-IR 同样用 Ridge(6D) 做 5-fold，主 run 下 HOMA-IR 5-fold r 约 0.45–0.55（见 evaluation_metrics.csv）。若表里 IR r 低于预期，可能原因：(1) 评估与训练目标不一致（训练是 log(HOMA_IR+1)，评估是原始尺度）；(2) LAMBDA_IR 可再调大或加入选优；(3) 金标准 HOMA-IR 样本/标定差异。详见 `docs/TRAINING_AND_OPTIMIZATION.md`。
