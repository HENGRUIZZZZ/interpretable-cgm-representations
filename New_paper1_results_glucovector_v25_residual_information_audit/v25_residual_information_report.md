# v25 Residual + Information Quantity Audit

Generated: 2026-04-08T01:21:18.237143

## Residual calibration and structure

| model      | target   |   n |   calibration_slope_true_on_pred |   calibration_intercept |     corr |
|:-----------|:---------|----:|---------------------------------:|------------------------:|---------:|
| Ridge10D   | sspg     |  20 |                          4.13436 |              -434.727   | 0.525339 |
| Ridge10D   | di       |  16 |                          1.79551 |                -1.11014 | 0.609219 |
| Ridge26D   | sspg     |  20 |                          2.72486 |              -270.693   | 0.646988 |
| Ridge26D   | di       |  16 |                          2.04535 |                -1.17867 | 0.615632 |
| Metwally14 | sspg     |  20 |                          2.11442 |               -88.8426  | 0.567852 |
| Metwally14 | di       |  16 |                          2.00707 |                -2.37898 | 0.565507 |

### Residual vs covariates

| model      | target            | covariate         |   spearman_rho |      pvalue |   n |
|:-----------|:------------------|:------------------|---------------:|------------:|----:|
| Ridge10D   | sspg_abs_residual | uncertainty_score |     -0.0953936 | 0.148376    | 231 |
| Ridge10D   | sspg_abs_residual | carb_g            |     -0.114177  | 0.0833458   | 231 |
| Ridge10D   | sspg_abs_residual | fat_g             |      0.176889  | 0.00703494  | 231 |
| Ridge10D   | sspg_abs_residual | protein_g         |      0.0616897 | 0.350611    | 231 |
| Ridge10D   | sspg_abs_residual | fiber_g           |      0.176889  | 0.00703494  | 231 |
| Ridge10D   | sspg_abs_residual | G_120             |     -0.183662  | 0.00510812  | 231 |
| Ridge10D   | sspg_abs_residual | AUC               |     -0.17919   | 0.00631782  | 231 |
| Ridge10D   | sspg_abs_residual | CV                |     -0.0607376 | 0.35811     | 231 |
| Ridge26D   | sspg_abs_residual | uncertainty_score |     -0.174866  | 0.00772466  | 231 |
| Ridge26D   | sspg_abs_residual | carb_g            |     -0.262016  | 5.54575e-05 | 231 |
| Ridge26D   | sspg_abs_residual | fat_g             |      0.376421  | 3.44533e-09 | 231 |
| Ridge26D   | sspg_abs_residual | protein_g         |      0.111473  | 0.090967    | 231 |
| Ridge26D   | sspg_abs_residual | fiber_g           |      0.376421  | 3.44533e-09 | 231 |
| Ridge26D   | sspg_abs_residual | G_120             |     -0.150149  | 0.0224514   | 231 |
| Ridge26D   | sspg_abs_residual | AUC               |     -0.158323  | 0.0160202   | 231 |
| Ridge26D   | sspg_abs_residual | CV                |     -0.153431  | 0.0196417   | 231 |
| Metwally14 | sspg_abs_residual | uncertainty_score |      0.0837884 | 0.204513    | 231 |
| Metwally14 | sspg_abs_residual | carb_g            |     -0.0481893 | 0.466084    | 231 |
| Metwally14 | sspg_abs_residual | fat_g             |      0.107841  | 0.102065    | 231 |
| Metwally14 | sspg_abs_residual | protein_g         |      0.0598797 | 0.364952    | 231 |
| Metwally14 | sspg_abs_residual | fiber_g           |      0.107841  | 0.102065    | 231 |
| Metwally14 | sspg_abs_residual | G_120             |      0.223369  | 0.000626662 | 231 |
| Metwally14 | sspg_abs_residual | AUC               |      0.212791  | 0.00113811  | 231 |
| Metwally14 | sspg_abs_residual | CV                |     -0.0186258 | 0.778268    | 231 |

### Error by true-label quantiles

| target   | true_bin   |   n |   mae_10d |   mae_26d |    mae_met |
|:---------|:-----------|----:|----------:|----------:|-----------:|
| sspg     | Q1_low     |   5 | 77.235    | 74.0209   |  26.3313   |
| sspg     | Q2         |   5 | 60.3741   | 65.1094   |  23.5705   |
| sspg     | Q3         |   5 | 19.834    | 28.0622   |  27.551    |
| sspg     | Q4_high    |   5 | 84.2603   | 68.9975   | 105.161    |
| di       | Q1_low     |   4 |  0.511281 |  0.406016 |   1.06479  |
| di       | Q2         |   4 |  0.28058  |  0.164844 |   0.742849 |
| di       | Q3         |   4 |  0.150622 |  0.289514 |   0.352989 |
| di       | Q4_high    |   4 |  0.874711 |  1.03292  |   0.488782 |

### Hardest subjects (SSPG)

| subject_id   |   sspg_true |   p26_sspg |   abs_res_26_sspg |   pmet_sspg |   abs_res_met_sspg |
|:-------------|------------:|-----------:|------------------:|------------:|-------------------:|
| D3H_055      |         335 |    156.642 |          178.358  |     95.7215 |          239.279   |
| D3H_053      |         301 |    157.039 |          143.961  |    122.046  |          178.954   |
| D3H_049      |          45 |    134.063 |           89.0628 |     86.7044 |           41.7044  |
| D3H_040      |          57 |    137.436 |           80.4361 |     74.6197 |           17.6197  |
| D3H_054      |          64 |    138.406 |           74.406  |     89.597  |           25.597   |
| D3H_051      |          51 |    124.294 |           73.2935 |     86.5908 |           35.5908  |
| D3H_033      |          69 |    140.152 |           71.1517 |    100.708  |           31.7076  |
| D3H_050      |          59 |    126.137 |           67.1371 |     90.7208 |           31.7208  |
| D3H_057      |          58 |    122.969 |           64.9694 |     92.744  |           34.744   |
| D3H_044      |          49 |    111.343 |           62.3426 |     50.9976 |            1.99756 |

### Hardest subjects (DI)

| subject_id   |   di_true |   p26_di |   abs_res_26_di |   pmet_di |   abs_res_met_di |
|:-------------|----------:|---------:|----------------:|----------:|-----------------:|
| D3H_033      |  3.06304  | 1.30255  |        1.7605   |   1.93917 |        1.12387   |
| D3H_049      |  2.62     | 1.41529  |        1.20471  |   2.01112 |        0.608877  |
| D3H_046      |  2.10411  | 1.324    |        0.780105 |   1.95959 |        0.144518  |
| D3H_042      |  1.83716  | 1.31076  |        0.526403 |   1.92325 |        0.0860908 |
| D3H_055      |  0.78     | 1.27912  |        0.499119 |   1.9092  |        1.1292    |
| D3H_053      |  0.501827 | 0.996484 |        0.494657 |   1.73496 |        1.23313   |
| D3H_041      |  0.530913 | 0.988344 |        0.45743  |   1.74987 |        1.21895   |
| D3H_036      |  0.887658 | 1.27591  |        0.388252 |   1.80735 |        0.919695  |
| D3H_044      |  2.22092  | 1.83455  |        0.386369 |   2.29878 |        0.0778623 |
| D3H_028      |  1.7791   | 1.44425  |        0.334844 |   2.04948 |        0.270384  |

### R2 / rank tradeoff

| model      | target   |   n |   spearman |   pearson |         r2 |      rmse |       mae |
|:-----------|:---------|----:|-----------:|----------:|-----------:|----------:|----------:|
| Ridge10D   | sspg     |  20 |   0.609023 |  0.525339 |  0.0823606 | 76.4628   | 60.4258   |
| Ridge10D   | di       |  16 |   0.647059 |  0.609219 |  0.293406  |  0.609429 |  0.454298 |
| Ridge26D   | sspg     |  20 |   0.778947 |  0.646988 |  0.160351  | 73.1413   | 59.0475   |
| Ridge26D   | di       |  16 |   0.608824 |  0.615632 |  0.207352  |  0.645473 |  0.473324 |
| Metwally14 | sspg     |  20 |   0.827068 |  0.567852 |  0.165611  | 72.9119   | 45.6534   |
| Metwally14 | di       |  16 |   0.602941 |  0.565507 | -0.108725  |  0.763396 |  0.662352 |

## Multi-label information quantity (D3)

| target                 | feature_set   |   n |   spearman |          r2 |       rmse |
|:-----------------------|:--------------|----:|-----------:|------------:|-----------:|
| hba1c                  | 10D           |  45 |  0.762024  |  0.596502   |   0.570341 |
| hba1c                  | 16D           |  45 |  0.741652  |  0.540842   |   0.608408 |
| hba1c                  | 26D           |  45 |  0.757277  |  0.603218   |   0.565575 |
| HOMA_IR                | 10D           |  45 |  0.341502  | -0.351775   |   3.47228  |
| HOMA_IR                | 16D           |  45 | -0.129908  | -0.11921    |   3.1595   |
| HOMA_IR                | 26D           |  45 |  0.338472  |  0.0192916  |   2.95755  |
| HOMA_B                 | 10D           |  45 |  0.225823  |  0.0870459  |  56.5507   |
| HOMA_B                 | 16D           |  45 |  0.202108  | -0.00207594 |  59.2467   |
| HOMA_B                 | 26D           |  45 |  0.1917    | -0.00396493 |  59.3025   |
| fasting_glucose_mg_dl  | 10D           |  45 |  0.748657  |  0.546631   |  20.1283   |
| fasting_glucose_mg_dl  | 16D           |  45 |  0.712417  |  0.509799   |  20.9299   |
| fasting_glucose_mg_dl  | 26D           |  45 |  0.715316  |  0.52837    |  20.5296   |
| fasting_insulin_uiu_ml | 10D           |  45 |  0.0112008 | -0.662282   |  10.5352   |
| fasting_insulin_uiu_ml | 16D           |  45 | -0.919387  | -0.0611521  |   8.41742  |
| fasting_insulin_uiu_ml | 26D           |  45 | -0.506539  | -0.181148   |   8.8806   |
| triglycerides_mg_dl    | 10D           |  45 | -0.0210186 | -0.0565941  | 172.213    |
| triglycerides_mg_dl    | 16D           |  45 | -0.324504  | -0.0520233  | 171.84     |
| triglycerides_mg_dl    | 26D           |  45 | -0.0977136 | -0.0540526  | 172.006    |
| hdl_mg_dl              | 10D           |  45 |  0.370109  | -0.560604   |  19.0585   |
| hdl_mg_dl              | 16D           |  45 | -0.0100208 | -0.183371   |  16.596    |
| hdl_mg_dl              | 26D           |  45 |  0.316314  |  0.0425435  |  14.928    |
| ldl_mg_dl              | 10D           |  45 | -0.373455  | -0.0472986  | 111.308    |
| ldl_mg_dl              | 16D           |  45 | -0.404297  | -0.0482852  | 111.36     |
| ldl_mg_dl              | 26D           |  45 | -0.303733  | -0.0506274  | 111.485    |
| cho_hdl_ratio          | 10D           |  45 |  0.0618163 | -0.0468997  |  59.7607   |
| cho_hdl_ratio          | 16D           |  45 | -0.0769079 | -0.0473944  |  59.7748   |
| cho_hdl_ratio          | 26D           |  45 | -0.0245815 | -0.0491779  |  59.8257   |

## Key numbers summary

| topic            | item                                          |       value |
|:-----------------|:----------------------------------------------|------------:|
| calibration      | Ridge26D_sspg_slope                           |  2.72486    |
| residual_pattern | Ridge26D_absres_vs_uncertainty_rho            | -0.174866   |
| multilabel       | HOMA_B_spearman_26D_minus_10D                 | -0.0341238  |
| multilabel       | HOMA_B_spearman_16D_minus_10D                 | -0.0237154  |
| multilabel       | HOMA_IR_spearman_26D_minus_10D                | -0.0030303  |
| multilabel       | HOMA_IR_spearman_16D_minus_10D                | -0.47141    |
| multilabel       | cho_hdl_ratio_spearman_26D_minus_10D          | -0.0863978  |
| multilabel       | cho_hdl_ratio_spearman_16D_minus_10D          | -0.138724   |
| multilabel       | fasting_glucose_mg_dl_spearman_26D_minus_10D  | -0.033341   |
| multilabel       | fasting_glucose_mg_dl_spearman_16D_minus_10D  | -0.0362402  |
| multilabel       | fasting_insulin_uiu_ml_spearman_26D_minus_10D | -0.51774    |
| multilabel       | fasting_insulin_uiu_ml_spearman_16D_minus_10D | -0.930588   |
| multilabel       | hba1c_spearman_26D_minus_10D                  | -0.00474699 |
| multilabel       | hba1c_spearman_16D_minus_10D                  | -0.0203725  |
| multilabel       | hdl_mg_dl_spearman_26D_minus_10D              | -0.0537957  |
| multilabel       | hdl_mg_dl_spearman_16D_minus_10D              | -0.38013    |
| multilabel       | ldl_mg_dl_spearman_26D_minus_10D              |  0.0697222  |
| multilabel       | ldl_mg_dl_spearman_16D_minus_10D              | -0.0308412  |
| multilabel       | triglycerides_mg_dl_spearman_26D_minus_10D    | -0.076695   |
| multilabel       | triglycerides_mg_dl_spearman_16D_minus_10D    | -0.303486   |
