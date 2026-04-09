# v26 Calibration + 16D Residual Correction

Generated: 2026-04-08T01:24:04.279752

## Metrics Summary

| model                         |   sspg_n |   sspg_spearman |   sspg_pearson |   sspg_r2 |   sspg_rmse |   sspg_mae |   di_n |   di_spearman |   di_pearson |    di_r2 |   di_rmse |   di_mae |   ir_auc |   decomp_auc |
|:------------------------------|---------:|----------------:|---------------:|----------:|------------:|-----------:|-------:|--------------:|-------------:|---------:|----------:|---------:|---------:|-------------:|
| Base26D                       |       20 |        0.778947 |       0.646988 | 0.160351  |     73.1413 |    59.0475 |     16 |      0.608824 |     0.615632 | 0.207352 |  0.645473 | 0.473324 | 0.903409 |        0.96  |
| Calibrated26D                 |       20 |        0.778947 |       0.646988 | 0.0747776 |     76.7781 |    61.0615 |     16 |      0.608824 |     0.615632 | 0.207631 |  0.645359 | 0.503801 | 0.903409 |        0.96  |
| Calibrated26D_plus16DResidual |       20 |        0.780451 |       0.6468   | 0.067733  |     77.0698 |    61.6084 |     16 |      0.614706 |     0.604833 | 0.200679 |  0.648184 | 0.500373 | 0.903409 |        0.944 |

## Bootstrap CIs

| model                         | target   | metric   |      point |      ci_lo |      ci_hi |
|:------------------------------|:---------|:---------|-----------:|-----------:|-----------:|
| Base26D                       | sspg     | spearman |  0.778947  |  0.527862  |   0.909904 |
| Base26D                       | sspg     | r2       |  0.160351  | -0.847147  |   0.248755 |
| Base26D                       | sspg     | rmse     | 73.1413    | 50.4714    |  95.2146   |
| Base26D                       | di       | spearman |  0.608824  |  0.0431516 |   0.88087  |
| Base26D                       | di       | r2       |  0.207352  | -0.393476  |   0.563591 |
| Base26D                       | di       | rmse     |  0.645473  |  0.347504  |   0.923841 |
| Calibrated26D                 | sspg     | spearman |  0.778947  |  0.527862  |   0.909904 |
| Calibrated26D                 | sspg     | r2       |  0.0747776 | -0.781486  |   0.118103 |
| Calibrated26D                 | sspg     | rmse     | 76.7781    | 49.5766    | 102.007    |
| Calibrated26D                 | di       | spearman |  0.608824  |  0.0431516 |   0.88087  |
| Calibrated26D                 | di       | r2       |  0.207631  | -0.196896  |   0.402174 |
| Calibrated26D                 | di       | rmse     |  0.645359  |  0.405858  |   0.877322 |
| Calibrated26D_plus16DResidual | sspg     | spearman |  0.780451  |  0.529654  |   0.916573 |
| Calibrated26D_plus16DResidual | sspg     | r2       |  0.067733  | -0.937312  |   0.128786 |
| Calibrated26D_plus16DResidual | sspg     | rmse     | 77.0698    | 51.6544    | 100.938    |
| Calibrated26D_plus16DResidual | di       | spearman |  0.614706  |  0.0496952 |   0.898888 |
| Calibrated26D_plus16DResidual | di       | r2       |  0.200679  | -0.2557    |   0.43268  |
| Calibrated26D_plus16DResidual | di       | rmse     |  0.648184  |  0.397088  |   0.895003 |

## Framework Blocks

| block               | purpose                            |
|:--------------------|:-----------------------------------|
| 10D_mechanism_axis  | stable metabolic baseline          |
| 16D_context_axis    | meal/context-driven correction     |
| calibration         | decompress prediction range        |
| residual_correction | fix systematic tail/context errors |
