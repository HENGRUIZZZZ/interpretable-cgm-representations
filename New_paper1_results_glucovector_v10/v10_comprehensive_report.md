# v10_comprehensive_report

Run stamp: 20260406_214802

## D4 calibrated summary
| experiment                  |   sspg_n |   sspg_pearson_r |   sspg_r2 |   sspg_rmse |   di_n |   di_pearson_r |      di_r2 |   di_rmse |
|:----------------------------|---------:|-----------------:|----------:|------------:|-------:|---------------:|-----------:|----------:|
| Exp1_GradientDetach_ConfigC |       20 |         0.56734  |  0.224028 |     70.3132 |     16 |       0.627313 | -0.205455  |  0.796001 |
| Exp2_UnbindAndIdent_ConfigC |       20 |         0.481394 |  0.115347 |     75.0759 |     16 |       0.644785 | -0.652077  |  0.931865 |
| Exp3_PureMechanism10D       |       20 |         0.543549 |  0.200981 |     71.3498 |     16 |       0.640091 | -0.0623171 |  0.747248 |

## Output structure
- One folder per experiment under `New_paper1_results_glucovector_v10/`
- Each folder includes metrics, scatter CSVs, latent 26D CSV, SHAP outputs, and per-subject per-meal predictions.