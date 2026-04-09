# v27.2 Mechanism-Anchored + Gated16D

Generated: 2026-04-08T01:45:58.422111

## D4 subject-level metrics

| target   | model                    |   n |   spearman |   pearson |        r2 |      rmse |       mae |
|:---------|:-------------------------|----:|-----------:|----------:|----------:|----------:|----------:|
| sspg     | Ridge10D                 |  20 |   0.609023 |  0.5265   | 0.0837999 | 76.4028   | 60.3723   |
| sspg     | Ridge26D                 |  20 |   0.768421 |  0.644439 | 0.178502  | 72.3465   | 57.9156   |
| sspg     | Anchored10D_plusGated16D |  20 |   0.634586 |  0.54443  | 0.083857  | 76.4004   | 60.4384   |
| di       | Ridge10D                 |  16 |   0.641176 |  0.615008 | 0.296908  |  0.607917 |  0.458739 |
| di       | Ridge26D                 |  16 |   0.608824 |  0.614667 | 0.206113  |  0.645977 |  0.472922 |
| di       | Anchored10D_plusGated16D |  16 |   0.641176 |  0.615008 | 0.296908  |  0.607917 |  0.458739 |

## Gate behavior across uncertainty bins (meal-level)

| target   | unc_bin   |   n |   gate_mean |   correction_mag_mean |
|:---------|:----------|----:|------------:|----------------------:|
| sspg     | low       | 116 |    0.421094 |               3.03004 |
| sspg     | mid       | 115 |    0.363332 |               2.58705 |
| sspg     | high      | 114 |    0.318764 |               1.98556 |
| di       | low       | 116 |    0.157195 |               0       |
| di       | mid       | 115 |    0.214564 |               0       |
| di       | high      | 114 |    0.260169 |               0       |

## Model config and OOF behavior

| target   |   alpha |   oof_hard_rate |   oof_gate_mean |   oof_corr_absres_gate |
|:---------|--------:|----------------:|----------------:|-----------------------:|
| sspg     |     1.2 |        0.253968 |        0.253967 |               0.250528 |
| di       |     0   |        0.253968 |        0.253965 |               0.221342 |
