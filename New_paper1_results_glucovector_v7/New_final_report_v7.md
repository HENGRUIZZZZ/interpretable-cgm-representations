# GlucoVector v7 实验结果（最新数据集）

数据源：`New_data/P1_final_with_D4_DI/P1_final`（来自 `P1_final_with_D4_DI.zip`）

## Exp1：SSPG 独立测试（D1+D2 训练，D4 测试）

- n = 20
- Pearson r = 0.8220 (p = 8.75e-06)
- Spearman r = 0.6962 (p = 6.50e-04)
- R² = 0.1350
- RMSE = 74.2376
- MAE = 47.9469

输出文件：
- `New_exp1_sspg/D4_sspg_metrics_v7.json`
- `New_exp1_sspg/D4_sspg_scatter_v7.csv`
- `New_exp1_sspg/shap_summary_sspg.png`
- `New_exp1_sspg/shap_feature_importance_sspg.csv`

## Exp2：DI 独立测试（D1+D2 训练，D4 测试）

- n = 16
- Pearson r = 0.6715 (p = 4.39e-03)
- Spearman r = 0.6029 (p = 1.34e-02)
- R² = 0.2746
- RMSE = 0.6175
- MAE = 0.5375

输出文件：
- `New_exp2_di/D4_di_metrics_v7.json`
- `New_exp2_di/D4_di_scatter_v7.csv`
- `New_exp2_di/shap_summary_di.png`
- `New_exp2_di/shap_feature_importance_di.csv`

## Exp3：D4 标准餐响应对比（Cornflakes / PB_sandwich / Protein_bar）

结果文件：`D4_meal_type_comparison.csv`

主要观察：
- SSPG：`Protein_bar` 的 Pearson r 最高（0.8436）
- DI：三种餐型下相关均为负（当前 head 在 DI 的餐型细分泛化较弱）

## Exp4：LODO 特征消融（D1 + D2 + D4）

结果文件：`lodo_ablation_results_v7.csv`

主要观察：
- SSPG：Tier 4（26D+Demographics）综合最好（Pearson 0.4846，R² 0.0831）
- DI：Tier 1（CGM统计）在 RMSE/MAE 上最稳（RMSE 1.0537，MAE 0.8099）；Tier 2-4 相关略升但误差放大

## SHAP Top Features（按 mean |SHAP|）

SSPG（前5）：
1. `z_init_0`
2. `Gb`
3. `z_init_3`
4. `tau_m`
5. `z_nonseq_12`

DI（前5）：
1. `z_init_3`
2. `z_init_0`
3. `z_nonseq_11`
4. `z_nonseq_7`
5. `tau_m`

