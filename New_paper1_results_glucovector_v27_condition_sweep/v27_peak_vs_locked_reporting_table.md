# Peak vs Locked Reporting Table

Use Peak for upper-bound claim and Locked for robustness claim.

| track              | metric        |    value | model                           | source                           |
|:-------------------|:--------------|---------:|:--------------------------------|:---------------------------------|
| PEAK               | DI_R2         | 0.413922 | GV_CorrLoss(Exp8)               | v19_overall_metrics.csv          |
| PEAK_ALL_GV_FAMILY | DI_R2         | 0.413922 | GV_CorrLoss(Exp8) (v19)         | v19/v21                          |
| V27_BEST_26D       | DI_R2         | 0.207352 | RidgeCV                         | v27_best_by_test_r2.csv          |
| LOCKED_PLUS        | DI_R2         | 0.207352 | Base26D                         | v26_metrics_summary.csv          |
| LOCKED             | DI_R2         | 0.207352 | Ridge26D                        | v22_primary_endpoints_locked.csv |
| LOCKED             | DI_Spearman   | 0.608824 | Ridge26D                        | v22_primary_endpoints_locked.csv |
| V27_BEST_26D       | DI_Spearman   | 0.608824 | RidgeCV                         | v27_best_by_test_r2.csv          |
| PEAK               | DI_Spearman   | 0.538235 | GV_CorrLoss(Exp8)               | v19_overall_metrics.csv          |
| PEAK_ALL_GV_FAMILY | SSPG_R2       | 0.266712 | GV_CorrLoss+VarMatch(v19) (v19) | v19/v21                          |
| PEAK               | SSPG_R2       | 0.2253   | GV_CorrLoss(Exp8)               | v19_overall_metrics.csv          |
| V27_BEST_26D       | SSPG_R2       | 0.225209 | RidgeCV_YJTarget                | v27_best_by_test_r2.csv          |
| LOCKED_PLUS        | SSPG_R2       | 0.160351 | Base26D                         | v26_metrics_summary.csv          |
| LOCKED             | SSPG_R2       | 0.160351 | Ridge26D                        | v22_primary_endpoints_locked.csv |
| V27_BEST_26D       | SSPG_Spearman | 0.790977 | RidgeCV_YJTarget                | v27_best_by_test_r2.csv          |
| LOCKED             | SSPG_Spearman | 0.778947 | Ridge26D                        | v22_primary_endpoints_locked.csv |
| PEAK               | SSPG_Spearman | 0.717293 | GV_CorrLoss(Exp8)               | v19_overall_metrics.csv          |