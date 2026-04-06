# New_GlucoVector v6 Final Report

Dataset used: `New_data/P1_final_with_D4_DI/P1_final` (from `P1_final_with_D4_DI.zip`)

## 1) Exp1 & Exp2: Independent D4 test

| Target | Train set | Test set | Test n | Pearson r | p-value | RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SSPG | D1 + D2 | D4 | 20 | 0.8220 | 8.75e-06 | 74.2376 |
| DI | D1 + D2 | D4 | 16 | 0.6715 | 4.39e-03 | 0.6175 |

Scatter CSVs (True vs Pred):
- `New_exp1_sspg/New_eval_on_D4/New_D4_sspg_true_vs_pred.csv`
- `New_exp2_di/New_eval_on_D4/New_D4_di_true_vs_pred.csv`

## 2) Exp3: D2 meal-type response

| Meal type | n | SSPG Pearson r | DI Pearson r |
| --- | ---: | ---: | ---: |
| Rice | 82 | -0.1475 | -0.1163 |
| Bread | 67 | 0.0334 | -0.1051 |
| Potatoes | 60 | 0.0583 | -0.1515 |
| Grapes | 65 | -0.3542 | -0.1163 |

Saved table:
- `New_exp3_meal_type/New_exp3_meal_type_table.csv`

## 3) Exp4: Feature ablation (LODO on D1 + D2)

| Feature combination | SSPG LODO Pearson r | DI LODO Pearson r |
| --- | ---: | ---: |
| Tier 1: CGM stats | 0.1496 | 0.1816 |
| Tier 2: 6D ODE params | -0.0524 | 0.0710 |
| Tier 3: 26D full latent | 0.1405 | 0.3813 |
| Tier 4: 26D + demographics | 0.3015 | 0.4330 |

Saved table:
- `New_exp4_ablation/New_exp4_lodo_ablation_table.csv`

## 4) Exp5: D4 cross-context stability

SSPG prediction consistency:
- OGTT vs Standard meal: r = 0.4456 (n=26, p=0.0225)
- OGTT vs Free-living: r = 0.1115 (n=51, p=0.4361)
- Standard meal vs Free-living: r = -0.3480 (n=30, p=0.0595)

DI prediction consistency:
- OGTT vs Standard meal: r = 0.6434 (n=26, p=3.92e-04)
- OGTT vs Free-living: r = -0.1398 (n=51, p=0.3279)
- Standard meal vs Free-living: r = -0.5269 (n=30, p=2.77e-03)

Saved outputs:
- `New_exp5_stability/New_exp5_context_stability.json`
- `New_exp5_stability/New_d4_standard_pred.csv`
- `New_exp5_stability/New_d4_ogtt_pred.csv`
- `New_exp5_stability/New_d4_freeliving_pred.csv`

