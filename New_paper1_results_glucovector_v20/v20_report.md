# GlucoVector v20 Fairness + Cross-Meal Consistency

Generated: 2026-04-07T23:38:30.212736

## Key Consistency Findings

{
  "latent_icc_10d_mean": -0.0675839793908424,
  "latent_icc_16d_mean": -0.3115338939752682,
  "metwally_feature_icc_mean": 0.36913319791198423
}

## Prediction ICC Ablation

| model           |   icc_sspg_pred |   icc_di_pred |
|:----------------|----------------:|--------------:|
| GV_26D_Exp8     |       0.138868  |      0.128331 |
| GV_10D_head_v20 |      -0.201291  |     -0.258237 |
| Healey_with_FI  |       0.719964  |    nan        |
| Healey_CGM_only |       0.0405631 |    nan        |

## Per-Meal Metrics

| model           | meal_type   |   sspg_n |   sspg_pearson_r |   sspg_spearman_r |    sspg_r2 |   sspg_rmse |   sspg_mae |   di_n |   di_pearson_r |   di_spearman_r |      di_r2 |    di_rmse |     di_mae |
|:----------------|:------------|---------:|-----------------:|------------------:|-----------:|------------:|-----------:|-------:|---------------:|----------------:|-----------:|-----------:|-----------:|
| GV_26D_Exp8     | Cornflakes  |       20 |         0.508939 |          0.64812  |  0.251569  |     69.0542 |    46.0847 |     16 |       0.658294 |        0.585294 |   0.271766 |   0.61869  |   0.52643  |
| GV_10D_head_v20 | Cornflakes  |       20 |         0.393237 |          0.61203  | -0.0828765 |     83.0622 |    49.1679 |     16 |       0.572174 |        0.570588 |  -0.026493 |   0.734541 |   0.623631 |
| Healey_with_FI  | Cornflakes  |       20 |         0.418675 |          0.4      |  0.168775  |     72.7735 |    51.2047 |    nan |     nan        |      nan        | nan        | nan        | nan        |
| Healey_CGM_only | Cornflakes  |       20 |        -0.237844 |          0.227068 | -0.206061  |     87.6595 |    64.6646 |    nan |     nan        |      nan        | nan        | nan        | nan        |
| GV_26D_Exp8     | PB_sandwich |       20 |         0.322385 |          0.330827 | -0.223124  |     88.2773 |    75.7836 |     16 |       0.574652 |        0.588235 |  -0.378648 |   0.851264 |   0.61141  |
| GV_10D_head_v20 | PB_sandwich |       20 |         0.311559 |          0.365414 | -0.0453701 |     81.6111 |    64.6996 |     16 |       0.578439 |        0.582353 |  -0.652544 |   0.931997 |   0.706732 |
| Healey_with_FI  | PB_sandwich |       20 |         0.680122 |          0.505263 |  0.339857  |     64.8534 |    47.5786 |    nan |     nan        |      nan        | nan        | nan        | nan        |
| Healey_CGM_only | PB_sandwich |       20 |         0.153738 |          0.431579 | -0.0128859 |     80.3331 |    63.1162 |    nan |     nan        |      nan        | nan        | nan        | nan        |
| GV_26D_Exp8     | Protein_bar |       20 |         0.620843 |          0.815038 |  0.253889  |     68.947  |    51.9523 |     16 |       0.654152 |        0.538235 |   0.216907 |   0.641571 |   0.559123 |
| GV_10D_head_v20 | Protein_bar |       20 |         0.600529 |          0.748872 |  0.179128  |     72.3189 |    54.1739 |     16 |       0.59969  |        0.594118 |   0.258819 |   0.624166 |   0.472842 |
| Healey_with_FI  | Protein_bar |       20 |         0.465771 |          0.46015  |  0.206412  |     71.1069 |    49.0691 |    nan |     nan        |      nan        | nan        | nan        | nan        |
| Healey_CGM_only | Protein_bar |       20 |         0.230016 |          0.320301 |  0.0370646 |     78.3272 |    60.1746 |    nan |     nan        |      nan        | nan        | nan        | nan        |

## Latent ICC (26D)

| latent_dim   |        icc | group   |
|:-------------|-----------:|:--------|
| z00          | -0.463423  | 10D     |
| z01          |  0.0150057 | 10D     |
| z02          | -0.474167  | 10D     |
| z03          | -0.239755  | 10D     |
| z04          |  0.0167069 | 10D     |
| z05          | -0.0497722 | 10D     |
| z06          |  0.463673  | 10D     |
| z07          |  0.0274789 | 10D     |
| z08          | -0.0367525 | 10D     |
| z09          |  0.0651648 | 10D     |
| z10          |  0.381539  | 16D     |
| z11          | -0.237001  | 16D     |
| z12          | -0.371854  | 16D     |
| z13          | -0.326967  | 16D     |
| z14          | -0.490469  | 16D     |
| z15          | -0.0644778 | 16D     |
| z16          | -0.390896  | 16D     |
| z17          | -0.306745  | 16D     |
| z18          | -0.438788  | 16D     |
| z19          | -0.144768  | 16D     |
| z20          | -0.4696    | 16D     |
| z21          | -0.4786    | 16D     |
| z22          | -0.446455  | 16D     |
| z23          | -0.324607  | 16D     |
| z24          | -0.433501  | 16D     |
| z25          | -0.441352  | 16D     |

## Metwally Feature ICC

| feature         |       icc |
|:----------------|----------:|
| G_0             | 0.547061  |
| G_60            | 0.423483  |
| G_120           | 0.690028  |
| G_180           | 0.468486  |
| G_Peak          | 0.550792  |
| CurveSize       | 0.362129  |
| AUC             | 0.574907  |
| pAUC            | 0.350289  |
| nAUC            | 0.238014  |
| iAUC            | 0.32588   |
| CV              | 0.20886   |
| T_baseline2peak | 0.0911273 |
| S_baseline2peak | 0.105564  |
| S_peak2end      | 0.231245  |

## Interpretability Plus: Exp8 SHAP Multi-Seed Stability

`New_paper1_results_glucovector_v20/interpretability_plus/v20_shap_multiseed_stability_exp8.csv`

```json
{
  "sspg_10d_share_mean": 91.89083522022977,
  "sspg_10d_share_std": 1.3711438910549194,
  "di_10d_share_mean": 96.85092504492529,
  "di_10d_share_std": 0.20691802488885852
}
```
