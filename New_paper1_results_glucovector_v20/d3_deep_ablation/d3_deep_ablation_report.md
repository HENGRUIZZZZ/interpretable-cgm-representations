# v20 D3 Deep Ablation (10D vs 26D vs Wang vs Metwally)

Generated: 2026-04-08T00:10:11.911618

## Cross-meal / Cross-day

| model            |   n_major_windows |   sspg_pred_meal_icc |   sspg_pred_cross_day_std_mean |   sspg_pred_cross_day_std_median |   sspg_pred_triplet_n |   sspg_pred_triplet_icc |   sspg_pred_triplet_std_mean |   di_pred_meal_icc |   di_pred_cross_day_std_mean |   di_pred_cross_day_std_median |   di_pred_triplet_n |   di_pred_triplet_icc |   di_pred_triplet_std_mean |
|:-----------------|------------------:|---------------------:|-------------------------------:|---------------------------------:|----------------------:|------------------------:|-----------------------------:|-------------------:|-----------------------------:|-------------------------------:|--------------------:|----------------------:|---------------------------:|
| Wang_Exp1        |               407 |           -0.0575285 |                      0.0972636 |                        0.0675145 |                    42 |               0.0972767 |                    0.0738965 |         -0.0575285 |                   0.00209359 |                     0.00145325 |                  42 |             0.0972767 |                 0.00159062 |
| GV_26D_Exp8      |               407 |           -0.135288  |                     63.8037    |                       61.8174    |                    42 |              -0.163097  |                   89.0064    |         -0.184155  |                   1.13987    |                     1.21259    |                  42 |            -0.0358794 |                 1.27226    |
| GV_10D_head_v20  |               407 |           -0.13114   |                     20.1269    |                       20.1892    |                    42 |              -0.0724988 |                   26.9144    |         -0.171681  |                   0.380172   |                     0.368464   |                  42 |            -0.119832  |                 0.58146    |
| Metwally14_Ridge |               407 |            0.773511  |                     27.3105    |                       28.4679    |                    42 |               0.266378  |                   23.1886    |          0.738237  |                   0.211726   |                     0.187875   |                  42 |             0.264129  |                 0.190069   |

## Uncertainty-stratified weak-label alignment

| model            | uncertainty_bin   |   n_subjects |   sspg_vs_homa_ir_spearman |   sspg_vs_hba1c_spearman |   di_vs_homa_ir_spearman |
|:-----------------|:------------------|-------------:|---------------------------:|-------------------------:|-------------------------:|
| Wang_Exp1        | low               |           44 |                  0.222171  |                 0.258399 |               -0.222171  |
| Wang_Exp1        | high              |           44 |                  0.162125  |                 0.150232 |               -0.162125  |
| GV_26D_Exp8      | low               |           44 |                  0.0837209 |                 0.137034 |               -0.0928823 |
| GV_26D_Exp8      | high              |           44 |                  0.304581  |                 0.440722 |               -0.39845   |
| GV_10D_head_v20  | low               |           44 |                  0.085976  |                 0.397913 |               -0.105285  |
| GV_10D_head_v20  | high              |           44 |                  0.448203  |                 0.698075 |               -0.388724  |
| Metwally14_Ridge | low               |           44 |                  0.102748  |                 0.287185 |               -0.314447  |
| Metwally14_Ridge | high              |           44 |                  0.357435  |                 0.626631 |               -0.400564  |

## Representation probe (same protocol)

| model            | feature_set   |   homa_ir_probe_spearman |   hba1c_probe_spearman |   probe_n |
|:-----------------|:--------------|-------------------------:|-----------------------:|----------:|
| Wang_Exp1        | 10D           |                 0.23386  |               0.706201 |        45 |
| Wang_Exp1        | 26D           |                 0.218972 |               0.734417 |        45 |
| GV_26D_Exp8      | 10D           |                 0.323452 |               0.792102 |        45 |
| GV_26D_Exp8      | 26D           |                 0.322266 |               0.747734 |        45 |
| GV_10D_head_v20  | 10D           |                 0.323452 |               0.792102 |        45 |
| GV_10D_head_v20  | 26D           |                 0.322266 |               0.747734 |        45 |
| Metwally14_Ridge | 10D           |               nan        |             nan        |         0 |
| Metwally14_Ridge | 26D           |               nan        |             nan        |         0 |

## Summary

{
  "delta_high_uncertainty_sspg_hba1c_spearman_gv26_minus_gv10": -0.2573525000416629,
  "generated_at": "2026-04-08T00:10:11.911618"
}
