# v24 Semantic Ablation: Meaning of 26D / 16D / Complex Scenarios

Generated: 2026-04-08T01:15:16.191823

## Q1: 26D unique information over 10D

| target   | feature_set         |   n |    spearman |         r2 |     rmse |
|:---------|:--------------------|----:|------------:|-----------:|---------:|
| sspg     | 10D                 |  63 |  0.00523435 | -0.13473   | 79.2746  |
| sspg     | 16D                 |  63 |  0.23617    | -0.0665468 | 76.856   |
| sspg     | 26D                 |  63 |  0.0362803  | -0.142863  | 79.5582  |
| sspg     | 16D_on_10D_residual |  63 | -0.43558    | -0.237683  | 78.9807  |
| di       | 10D                 |  63 |  0.367108   |  0.0504195 |  1.0951  |
| di       | 16D                 |  63 |  0.272685   | -0.03102   |  1.14109 |
| di       | 26D                 |  63 |  0.342842   |  0.0210784 |  1.11189 |
| di       | 16D_on_10D_residual |  63 | -0.744816   | -0.0489342 |  1.00954 |

## Q2: What 16D adds (context encoding tests)

| task                     | feature_set   |    n |   accuracy |   macro_f1 |           r2 |   spearman |
|:-------------------------|:--------------|-----:|-----------:|-----------:|-------------:|-----------:|
| meal_type_classification | 10D           | 4969 |   0.607969 |   0.5541   | nan          | nan        |
| meal_type_classification | 16D           | 4969 |   0.650835 |   0.56979  | nan          | nan        |
| meal_type_classification | 26D           | 4969 |   0.748642 |   0.657199 | nan          | nan        |
| carb_g_regression        | 10D           | 4969 | nan        | nan        |   0.0441348  |   0.240983 |
| carb_g_regression        | 16D           | 4969 | nan        | nan        |   0.0368127  |   0.24606  |
| carb_g_regression        | 26D           | 4969 | nan        | nan        |  -0.00161713 |   0.210448 |
| fat_g_regression         | 10D           | 4969 | nan        | nan        |   0.0241283  |   0.225844 |
| fat_g_regression         | 16D           | 4969 | nan        | nan        |   0.0244714  |   0.238246 |
| fat_g_regression         | 26D           | 4969 | nan        | nan        |   0.0105384  |   0.213357 |
| protein_g_regression     | 10D           | 4969 | nan        | nan        |   0.0588164  |   0.273563 |
| protein_g_regression     | 16D           | 4969 | nan        | nan        |   0.0829498  |   0.28533  |
| protein_g_regression     | 26D           | 4969 | nan        | nan        |   0.0288891  |   0.27882  |

## Q3: Utility under complex scenarios (D4 uncertainty strata)

| unc_bin   |   n_meals |   mean_err10_sspg |   mean_err26_sspg |   delta_err26_minus10_sspg |   delta_ci_lo_sspg |   delta_ci_hi_sspg |   mean_err10_di |   mean_err26_di |   delta_err26_minus10_di |   delta_ci_lo_di |   delta_ci_hi_di |
|:----------|----------:|------------------:|------------------:|---------------------------:|-------------------:|-------------------:|----------------:|----------------:|-------------------------:|-----------------:|-----------------:|
| low       |       114 |           62.6293 |           70.6888 |                    8.05957 |            4.405   |           11.4042  |        0.944258 |        1.03174  |                0.0874768 |        0.0357804 |        0.13969   |
| mid       |       117 |           61.495  |           64.7334 |                    3.23841 |           -0.89191 |            7.2544  |        0.684077 |        0.718049 |                0.033972  |       -0.0235474 |        0.089412  |
| high      |       114 |           59.0905 |           56.4096 |                   -2.68094 |           -8.03328 |            2.56758 |        0.882813 |        0.834807 |               -0.0480058 |       -0.111476  |        0.0102081 |

## 26D component variance decomposition

| target   |   var_share_16d_component |   spearman_abs16_vs_uncertainty |
|:---------|--------------------------:|--------------------------------:|
| sspg     |                  0.857343 |                       0.0589809 |
| di       |                  0.683634 |                      -0.347842  |
