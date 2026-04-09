# v20 D3 Free-living Benchmark (No SSPG/DI Gold)

Generated: 2026-04-08T00:04:18.087484

This benchmark compares model behavior on free-living D3 where SSPG/DI are unavailable. Metrics include within-subject prediction stability, meal-type ICC, same-day triplet (Breakfast/Lunch/Dinner) consistency, weak-label alignment (HOMA-IR/HbA1c), macro alignment, and subject retrieval from latent space.

## Macro Input Audit (D3 meals.csv)

| field         |   non_null |   missing_rate |
|:--------------|-----------:|---------------:|
| carb_g        |       1706 |    0           |
| fat_g         |       1706 |    0           |
| protein_g     |       1706 |    0           |
| fiber_g       |       1705 |    0.000586166 |
| calories_kcal |       1706 |    0           |

## Main Comparison

| model            |   subject_count |   mean_windows_per_subject |   sspg_within_subject_std_mean |   di_within_subject_std_mean |   n_major_meal_windows |   sspg_pred_meal_icc |   di_pred_meal_icc |   n_triplet_days |   sspg_pred_triplet_icc |   sspg_pred_triplet_std_mean |   di_pred_triplet_icc |   di_pred_triplet_std_mean |   corr_sspg_vs_homa_ir_spearman |   corr_sspg_vs_hba1c_spearman |   corr_di_vs_homa_ir_spearman |   corr_n_subjects |   corr_sspg_vs_carb_window_spearman |   corr_di_vs_carb_window_spearman |   subject_retrieval_top1_26d |   subject_retrieval_top1_10d |
|:-----------------|----------------:|---------------------------:|-------------------------------:|-----------------------------:|-----------------------:|---------------------:|-------------------:|-----------------:|------------------------:|-----------------------------:|----------------------:|---------------------------:|--------------------------------:|------------------------------:|------------------------------:|------------------:|------------------------------------:|----------------------------------:|-----------------------------:|-----------------------------:|
| Wang_Exp1        |              45 |                    37.9111 |                       0.305922 |                   0.00658496 |                    407 |           -0.0575285 |         -0.0575285 |               42 |               0.0972767 |                    0.0738965 |             0.0972767 |                 0.00159062 |                      nan        |                    nan        |                    nan        |                45 |                            0.097029 |                         -0.097029 |                    0.111111  |                    0.133333  |
| GV_26D_Exp8      |              45 |                    37.9111 |                     113.24     |                   1.78391    |                    407 |           -0.135288  |         -0.184155  |               42 |              -0.163097  |                   89.0064    |            -0.0358794 |                 1.27226    |                        0.181291 |                      0.409758 |                     -0.29552  |                45 |                            0.651203 |                         -0.571876 |                    0.0888889 |                    0.0888889 |
| GV_10D_head_v20  |              45 |                    37.9111 |                      40.4839   |                   0.839344   |                    407 |           -0.13114   |         -0.171681  |               42 |              -0.0724988 |                   26.9144    |            -0.119832  |                 0.58146    |                        0.33386  |                      0.632009 |                     -0.310277 |                45 |                            0.694829 |                         -0.763198 |                    0.0888889 |                    0.0888889 |
| Metwally14_Ridge |              45 |                    37.9111 |                      46.7284   |                   0.413695   |                    407 |            0.773511  |          0.738237  |               42 |               0.266378  |                   23.1886    |             0.264129  |                 0.190069   |                        0.366535 |                      0.669656 |                     -0.453491 |                45 |                            0.213995 |                         -0.23824  |                  nan         |                  nan         |
