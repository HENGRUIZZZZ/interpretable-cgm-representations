# 实验方案 v4.0：尽善尽美 — 执行结果

本文件夹存放按《实验方案 v4.0：尽善尽美》全面执行后的**结果**，与方案中阶段一至阶段四对应。

## 执行配置

- **数据**：D1 + D2 + D4，`P1_ONE_MEAL_PER_SUBJECT=1`（每受试者 1 个餐窗，127 样本）
- **Pipeline**：`P1_SAVE_26D_LATENT=1`，`P1_SEED=42`
- **结果目录**：`paper1_results_v4/baseline_seed42/`

## 阶段一：特征工程

| 文件 | 说明 |
|------|------|
| `latent_and_gold_all_26d.csv` | 每样本 26D latent（6 ODE + 4 z_init + 16 z_nonseq）+ 金标准（sspg, di, homa_ir, homa_b） |
| `cgm_stats_per_sample.csv` | 每餐窗 CGM 统计：mean, std, cv, min, max, range, tir, tar, tbr, auc, ac_var, mge |
| `full_features_v4.csv` | 合并后的全量特征表（~45 列），供三路对决使用 |

## 阶段二：三路模型对决（5-fold CV，Spearman r）

| 路线 | 说明 | SSPG (r) | DI (r) |
|------|------|----------|--------|
| **A** | 6D ODE → PCA(2) → Ridge | 0.26 | 0.19 |
| **B** | 全量特征 → **XGBoost**（5-fold CV） | -0.005 | 0.25 |
| **C** | 仅 CGM 统计 → Ridge | 0.31 | 0.37 |

- 代理相关：IR_proxy (1/si) vs SSPG r=-0.12；BCF_proxy (mi) vs DI r=0.36
- 详细数值见 `bakeoff_results.json`

## 阶段三：分析与出图

- `bakeoff_bar_spearman.png` — 三路 SSPG/DI Spearman r 柱状图
- `scatter_IR_proxy_vs_SSPG.png` — IR_proxy vs SSPG 散点
- `scatter_BCF_proxy_vs_DI.png` — BCF_proxy vs DI 散点
- `pc1_pc2_colored_sspg.png` / `pc1_pc2_colored_di.png` — 6D ODE 的 PC1–PC2 着色 SSPG/DI
- `analysis_report_v4.txt` — 简短分析报告

## 阶段四：临床亚型（SSPG>180，DI 中位数分）

- `metabolic_subtype_quadrant.png` — 四象限图（Healthy / IR / β-cell defect / Mixed）
- `metabolic_subtype_counts.txt` — 各亚型人数（Healthy 28, IR 12, β-cell defect 20, Mixed 3）

## 复现命令

```bash
# 1. Pipeline（26D latent）
P1_SAVE_26D_LATENT=1 P1_SEED=42 P1_ONE_MEAL_PER_SUBJECT=1 P1_RESULTS_DIR=paper1_results_v4/baseline_seed42 python run_p1_full_pipeline.py

# 2. CGM 统计 + 合并
P1_ONE_MEAL_PER_SUBJECT=1 python scripts/compute_cgm_stats.py --latent_csv paper1_results_v4/baseline_seed42/latent_and_gold_all_26d.csv --output_dir paper1_results_v4/baseline_seed42

# 3. 三路对决（Route B 需先安装 xgboost: pip install xgboost）
python scripts/model_bakeoff.py --full_features paper1_results_v4/baseline_seed42/full_features_v4.csv --output_dir paper1_results_v4/baseline_seed42 --n_folds 5 --seed 42

# 4. 分析与出图
python scripts/analyze_results.py --full_features paper1_results_v4/baseline_seed42/full_features_v4.csv --bakeoff_json paper1_results_v4/baseline_seed42/bakeoff_results.json --output_dir paper1_results_v4/baseline_seed42
```
