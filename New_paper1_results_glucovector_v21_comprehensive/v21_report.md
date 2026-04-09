# v21 Comprehensive Story Audit (v6-v20 synthesis)

Generated: 2026-04-08T00:18:33.943129

## D4 Gold Scorecard

| model                | setting                   |   sspg_spearman |   sspg_rmse |   di_spearman |    di_rmse | invasive_feature   |   icc_sspg_pred |   icc_di_pred |
|:---------------------|:--------------------------|----------------:|------------:|--------------:|-----------:|:-------------------|----------------:|--------------:|
| GV_26D_Exp8          | D4_gold_overall           |        0.717293 |     70.2556 |      0.538235 |   0.555029 | no                 |       0.138868  |      0.128331 |
| Healey_with_FI_proxy | D4_gold_overall           |        0.452632 |     68.5419 |      0.205882 |   0.821289 | yes                |     nan         |    nan        |
| Metwally_Exp2        | D4_gold_overall           |        0.721805 |     69.7467 |      0.570588 |   0.765606 | no                 |     nan         |    nan        |
| SimpleStats_Exp4     | D4_gold_overall           |        0.736842 |     71.2042 |      0.697059 |   0.869227 | no                 |     nan         |    nan        |
| Wang_Exp1            | D4_gold_overall           |        0.103909 |     80.1289 |      0.181515 |   0.760958 | no                 |     nan         |    nan        |
| GV_10D_head_v20      | D4_gold_weighted_per_meal |        0.575439 |     78.9974 |      0.582353 |   0.763568 | no                 |      -0.201291  |     -0.258237 |
| GV_26D_Exp8          | D4_gold_weighted_per_meal |        0.597995 |     75.4262 |      0.570588 |   0.703842 | no                 |       0.138868  |      0.128331 |
| Healey_CGM_only      | D4_gold_weighted_per_meal |        0.326316 |     82.1066 |    nan        | nan        | no                 |       0.0405631 |    nan        |
| Healey_with_FI       | D4_gold_weighted_per_meal |        0.455138 |     69.5779 |    nan        | nan        | yes                |       0.719964  |    nan        |
| Metwally_Exp2        | D4_gold_weighted_per_meal |        0.520802 |     74.7938 |      0.509804 |   0.772798 | no                 |     nan         |    nan        |

## D3 Free-living Scorecard

| model            |   n_major_windows |   sspg_pred_meal_icc |   sspg_pred_cross_day_std_mean |   sspg_pred_cross_day_std_median |   sspg_pred_triplet_n |   sspg_pred_triplet_icc |   sspg_pred_triplet_std_mean |   di_pred_meal_icc |   di_pred_cross_day_std_mean |   di_pred_cross_day_std_median |   di_pred_triplet_n |   di_pred_triplet_icc |   di_pred_triplet_std_mean |   sspg_vs_hba1c_spearman_high |   sspg_vs_hba1c_spearman_low |   sspg_vs_homa_ir_spearman_high |   sspg_vs_homa_ir_spearman_low |   hba1c_probe_spearman_10D |   hba1c_probe_spearman_26D |   homa_ir_probe_spearman_10D |   homa_ir_probe_spearman_26D |
|:-----------------|------------------:|---------------------:|-------------------------------:|---------------------------------:|----------------------:|------------------------:|-----------------------------:|-------------------:|-----------------------------:|-------------------------------:|--------------------:|----------------------:|---------------------------:|------------------------------:|-----------------------------:|--------------------------------:|-------------------------------:|---------------------------:|---------------------------:|-----------------------------:|-----------------------------:|
| GV_10D_head_v20  |               407 |           -0.13114   |                     20.1269    |                       20.1892    |                    42 |              -0.0724988 |                   26.9144    |         -0.171681  |                   0.380172   |                     0.368464   |                  42 |            -0.119832  |                 0.58146    |                      0.698075 |                     0.397913 |                        0.448203 |                      0.085976  |                   0.792102 |                   0.747734 |                     0.323452 |                     0.322266 |
| GV_26D_Exp8      |               407 |           -0.135288  |                     63.8037    |                       61.8174    |                    42 |              -0.163097  |                   89.0064    |         -0.184155  |                   1.13987    |                     1.21259    |                  42 |            -0.0358794 |                 1.27226    |                      0.440722 |                     0.137034 |                        0.304581 |                      0.0837209 |                   0.792102 |                   0.747734 |                     0.323452 |                     0.322266 |
| Metwally14_Ridge |               407 |            0.773511  |                     27.3105    |                       28.4679    |                    42 |               0.266378  |                   23.1886    |          0.738237  |                   0.211726   |                     0.187875   |                  42 |             0.264129  |                 0.190069   |                      0.626631 |                     0.287185 |                        0.357435 |                      0.102748  |                 nan        |                 nan        |                   nan        |                   nan        |
| Wang_Exp1        |               407 |           -0.0575285 |                      0.0972636 |                        0.0675145 |                    42 |               0.0972767 |                    0.0738965 |         -0.0575285 |                   0.00209359 |                     0.00145325 |                  42 |             0.0972767 |                 0.00159062 |                      0.150232 |                     0.258399 |                        0.162125 |                      0.222171  |                   0.706201 |                   0.734417 |                     0.23386  |                     0.218972 |

## Quantified Value of 10D/26D and Fairness

| question                          | metric                                          |      value |
|:----------------------------------|:------------------------------------------------|-----------:|
| 10D_representation_value_vs_wang  | hba1c_probe_spearman_10D_delta                  |  0.0859017 |
| 10D_representation_value_vs_wang  | homa_ir_probe_spearman_10D_delta                |  0.0895916 |
| 26D_incremental_value_over_10D_D4 | sspg_spearman_delta                             |  0.0225564 |
| 26D_incremental_value_over_10D_D4 | di_spearman_delta                               | -0.0117647 |
| 10D_stability_value_over_26D_D3   | sspg_cross_day_std_reduction                    | 43.6768    |
| 16D_value_under_high_uncertainty  | high_uncertainty_sspg_hba1c_delta_26D_minus_10D | -0.257353  |
| healey_fairness_gap               | sspg_spearman_withFI_minus_cgmOnly              |  0.128822  |
