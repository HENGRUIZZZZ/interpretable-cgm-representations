# v28 Framework value — summary

Generated: 2026-04-08T02:04:04.328817

## D4 subject LOOCV: median 26D + meal context vs 26D only

| target    | model                                   |   n |   spearman |       r2 |      rmse |
|:----------|:----------------------------------------|----:|-----------:|---------:|----------:|
| sspg_true | LOOCV_Ridge_median26D                   |  20 |   0.709774 | 0.216516 | 70.6527   |
| sspg_true | LOOCV_Ridge_median26D_plus_meal_context |  20 |   0.687218 | 0.212596 | 70.8293   |
| di_true   | LOOCV_Ridge_median26D                   |  16 |   0.447059 | 0.232188 |  0.63528  |
| di_true   | LOOCV_Ridge_median26D_plus_meal_context |  16 |   0.426471 | 0.21551  |  0.642143 |

## D5 MSS (external): LOOCV meal response

| model                                           |    n |   spearman |       r2 |    rmse |
|:------------------------------------------------|-----:|-----------:|---------:|--------:|
| D5_LOOCV_pre_window_CGM_time_only               | 1369 |   0.511871 | 0.247537 | 1.10827 |
| D5_LOOCV_pre_window_CGM_time_plus_wearable_pre  | 1369 |   0.501172 | 0.238042 | 1.11524 |
| D5_LOOCV_pre_window_CGM_time_plus_wearable_post | 1369 |   0.515311 | 0.255233 | 1.10259 |

## Claims map

| claim_id   | claim                                                                                                                                                            | evidence                                                                                                                                                            |
|:-----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C1         | 26D locked readout matches clinical classification (IR/Decomp) strongly on D4                                                                                    | v22_secondary_clinical_endpoints.csv; Ridge26D decomp_auc                                                                                                           |
| C2         | Peak DI performance is higher with Exp8 neural heads than Ridge26D alone                                                                                         | v19_overall_metrics GV_CorrLoss(Exp8) vs v22 Ridge26D                                                                                                               |
| C3         | Subject-level median 26D latents provide competitive linear readout vs adding explicit meal-aggregated context (macros/uncertainty) on D4                        | v28_d4_subject_context_readout.csv                                                                                                                                  |
| C4         | On external MSS (Phillips et al. 2023), concurrent post-prandial Actiheart summaries improve LOOCV prediction of glycemic iAUC beyond pre-meal CGM-only features | v28_d5_mss_loocv_summary_iauc.csv                                                                                                                                   |
| C5         | D3 free-living supports cross-window probe and baseline comparisons (see scorecard)                                                                              | /Users/hertz1030/Documents/GitHub/interpretable-cgm-representations/New_paper1_results_glucovector_v21_comprehensive/v21_d3_free_living_comprehensive_scorecard.csv |
