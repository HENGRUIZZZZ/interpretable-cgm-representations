# GlucoVector v18 Results

Generated: 2026-04-07T20:28:57.695759

## D4 Performance

| Exp | SSPG r | SSPG ρ | SSPG R² | RMSE | pred_std | compress | auROC | DI r | DI R² |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| v18_Exp1_Wang_Baseline | 0.1598 | 0.1039 | -0.0077 | 80.1 | 0.1 | 0.001 | 0.5260 | 0.2871 | -0.1017 |
| v18_Exp2_Metwally_14Feature | 0.6128 | 0.7113 | 0.2364 | 69.8 | — | — | — | — | — |
| v18_Exp3_Healey_ODE | 0.4084 | 0.5858 | 0.1500 | 76.1 | — | — | — | — | — |
| v18_Exp4_Simple_Stats | 0.5664 | 0.6797 | 0.2158 | 70.7 | — | — | — | — | — |
| v18_Exp5_GV_Baseline | 0.6660 | 0.7594 | 0.1350 | 74.2 | 13.3 | 0.162 | 0.9479 | 0.6802 | 0.3084 |
| v18_Exp6_Separate_Heads | 0.6394 | 0.7459 | 0.1796 | 72.3 | 17.6 | 0.215 | 0.9479 | 0.6755 | 0.3699 |
| v18_Exp7_EarlyStop | 0.5772 | 0.6992 | 0.1792 | 72.3 | 18.7 | 0.228 | 0.9271 | 0.6607 | 0.3948 |
| v18_Exp8_CorrLoss | 0.5813 | 0.7173 | 0.2255 | 70.2 | 25.4 | 0.310 | 0.9479 | 0.6672 | 0.4139 |
| v18_Exp9_Full_Combo | 0.4677 | 0.7068 | -0.2843 | 90.5 | 19.2 | 0.234 | 0.9167 | 0.6758 | 0.1924 |
| v18_Exp10_Ridge_Probe | — | — | — | — | — | — | — | — | — |

## Prediction Compression

| Exp | pred_std | true_std | compression | pred_range |
|:---|:---:|:---:|:---:|:---:|
| v18_Exp1_Wang_Baseline | 0.1 | 81.9 | 0.001 | 0.3 |
| v18_Exp2_Metwally_14Feature | — | — | — | — |
| v18_Exp3_Healey_ODE | — | — | — | — |
| v18_Exp4_Simple_Stats | — | — | — | — |
| v18_Exp5_GV_Baseline | 13.3 | 81.9 | 0.162 | 43.9 |
| v18_Exp6_Separate_Heads | 17.6 | 81.9 | 0.215 | 67.6 |
| v18_Exp7_EarlyStop | 18.7 | 81.9 | 0.228 | 68.7 |
| v18_Exp8_CorrLoss | 25.4 | 81.9 | 0.310 | 97.9 |
| v18_Exp9_Full_Combo | 19.2 | 81.9 | 0.234 | 67.3 |
| v18_Exp10_Ridge_Probe | — | — | — | — |
