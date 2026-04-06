# v9_comprehensive_report

Data: `New_data/P1_final_with_D4_DI/P1_final`

## Exp1: Config C independent D4 performance (calibrated)
- SSPG metrics: {"n": 20, "pearson_r": 0.6886167296706259, "pearson_p": 0.0007872518581839921, "spearman_r": 0.613533834586466, "spearman_p": 0.004013734729903659, "r2": 0.19253830494655455, "rmse": 71.72573246537128, "mae": 48.72595975505878}
- DI metrics: {"n": 16, "pearson_r": 0.6215623708780045, "pearson_p": 0.010160049200293507, "spearman_r": 0.5470588235294118, "spearman_p": 0.028301087242108842, "r2": -0.23562056793755426, "rmse": 0.8058986511575434, "mae": 0.6947975423624853}

## Exp2: Meal-type comparison
| meal_type   |   sspg_pearson_r |   sspg_r2 |   sspg_rmse |   di_pearson_r |     di_r2 |   di_rmse |
|:------------|-----------------:|----------:|------------:|---------------:|----------:|----------:|
| Cornflakes  |         0.621183 | 0.258734  |     68.7228 |       0.581707 | -0.123721 |  0.768541 |
| PB_sandwich |         0.486823 | 0.0796506 |     76.5756 |       0.543319 | -0.313424 |  0.830884 |
| Protein_bar |         0.567749 | 0.150552  |     73.5669 |       0.391386 | -0.319162 |  0.832697 |

## Exp4: Tri-class
- tri_class_metrics: {"n": 95, "accuracy": 0.3684210526315789, "f1_macro": 0.2692307692307692}

Main outputs:
- D4_sspg_metrics.json / D4_di_metrics.json
- D4_sspg_true_vs_pred.csv / D4_di_true_vs_pred.csv
- per_subject_per_meal_preds.csv
- meal_type_comparison.csv
- shap_summary_sspg.png / shap_summary_di.png
- shap_feature_importance.csv
- tri_class_metrics.json / confusion_matrix.csv