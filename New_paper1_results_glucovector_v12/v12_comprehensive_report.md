# v12_comprehensive_report

Run stamp: 20260407_000407

## Main comparison (Exp2 fixed data vs Exp3 old data ablation)
| experiment                             |   sspg_pearson_r |   di_pearson_r |   sspg_rmse |   di_rmse |   ode_shap_share |
|:---------------------------------------|-----------------:|---------------:|------------:|----------:|-----------------:|
| Exp2_GlucoVectorV12_26D_Semi_FixedData |              nan |            nan |     85.1016 |  0.768874 |         0.27435  |
| Exp3_DataAblation_OldData_26D_Semi     |              nan |            nan |     85.1016 |  0.768874 |         0.155197 |

## Notes
- Exp1 is trained as unsupervised 10D baseline on fixed data (no SSPG/DI supervision).
- Exp2 uses separated 26D heads with weak gradient scale 0.01.
- Exp3 uses same model config as Exp2 but pre-fix old data.