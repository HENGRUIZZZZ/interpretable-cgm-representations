# P1 完整方案：一次性跑通 + 充分利用数据

## 方案概要

- **训练**：合并 **D1 + D2 + D4** 所有餐心 CGM，按 **subject 80/10/10** 划分，训练一个机理 VAE。
- **验证**：在 test 集及全量“有金标准”样本上计算 latent（si, mi, tau_m, sg 等）与 **SSPG / DI / HOMA_IR / HOMA_B** 的 Spearman 相关。
- **出图**：si vs SSPG、mi vs DI 散点（按数据集着色）、Bland-Altman、相关汇总表。

## 数据利用

| 数据集 | 样本数 | 受试者 | 金标准 (SSPG/DI 等) |
|--------|--------|--------|----------------------|
| D1     | 59     | 59     | 32 人有              |
| D2     | 98     | 38     | 43 SSPG, 41 DI       |
| D4     | 175    | 30     | 42 SSPG, 53 DI       |
| **合并** | **332** | **127** | test 约 13 人有金标准，全量更多 |

D4 使用 `output/D4_hall`（若存在），否则 `D3alt_hall`；loader 已兼容 D4 的连续 CGM + meals 切窗及 labels 列名（SSPG/DI 等）。

## 如何跑（本机）

```bash
# 1. 指定数据路径（解压 cgm_project_v2_final 或 cgm_all_datasets 后的 output 目录）
export CGM_PROJECT_OUTPUT=/path/to/cgm_project/output

# 2. 一键：训练 + 评估 + 保存
python run_p1_full_pipeline.py

# 3. 出图（读 paper1_results/*.csv 与 correlations.txt）
python scripts/plot_p1_results.py
```

## 产出文件

- `paper1_results/autoencoder_p1_full.pt`：模型与归一化参数
- `paper1_results/correlations.txt`：各 (gold, param) 的 Spearman r / p / n
- `paper1_results/latent_and_gold_test.csv`：test 集 subject 级 latent + 金标准
- `paper1_results/latent_and_gold_all.csv`：全量 subject 级 latent + 金标准（出图用）
- `paper1_results/figures/p1_si_vs_sspg.png`：si vs SSPG 散点
- `paper1_results/figures/p1_mi_vs_di.png`：mi vs DI 散点
- `paper1_results/figures/p1_blandaltman_si_sspg.png`：Bland-Altman 图
- `paper1_results/figures/p1_correlations_summary.txt`：相关汇总表

## 配置（可选）

- `paper1_experiment_config.py`：`P1_FULL_TRAIN_DATASETS`、`P1_TRAIN_FRAC` / `P1_VAL_FRAC` / `P1_TEST_FRAC`、`SPLIT_SEED`。
- `run_p1_full_pipeline.py`：`NUM_EPOCHS`、`BATCH_SIZE`、`LR`、`BETA_HAT`。

## 性能

- 本机（Mac MPS）约 **1–2 分钟** 完成 80 epoch 训练；整体流程几分钟内跑完。
