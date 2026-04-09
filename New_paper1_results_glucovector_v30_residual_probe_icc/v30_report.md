# v30: residual ablation, context probes, stratified ICC

## D4 readout ablation (subject-level mean pred vs gold; same Ridge26 trained on D1+D2)
| ablation             | target   |   n |   spearman |        r2 |       rmse |        mae |
|:---------------------|:---------|----:|-----------:|----------:|-----------:|-----------:|
| full_26d             | sspg     |  20 |   0.778947 |  0.160351 |  73.1413   |  59.0475   |
| full_26d             | di       |  16 |   0.608824 |  0.207352 |   0.645473 |   0.473324 |
| zero_residual_16d    | sspg     |  20 |   0.610526 | -1.60446  | 128.817    | 108.578    |
| zero_residual_16d    | di       |  16 |   0.605882 | -2.85834  |   1.42409  |   1.31086  |
| shuffle_residual_16d | sspg     |  20 |   0.675188 |  0.203248 |  71.2485   |  58.3143   |
| shuffle_residual_16d | di       |  16 |   0.555882 |  0.253109 |   0.626566 |   0.509242 |

## Context probes (GroupKFold by subject; higher R² or acc ⇒ block carries more target info)
| dataset   | task              | block   | metric      |       value |
|:----------|:------------------|:--------|:------------|------------:|
| D4        | carb_g            | mech10  | cv_r2_mean  |  0.850621   |
| D4        | carb_g            | res16   | cv_r2_mean  |  0.990672   |
| D4        | carb_g            | full26  | cv_r2_mean  |  0.967202   |
| D4        | fat_g             | mech10  | cv_r2_mean  |  0.965984   |
| D4        | fat_g             | res16   | cv_r2_mean  |  0.996238   |
| D4        | fat_g             | full26  | cv_r2_mean  |  0.993839   |
| D4        | protein_g         | mech10  | cv_r2_mean  |  0.83326    |
| D4        | protein_g         | res16   | cv_r2_mean  |  0.989566   |
| D4        | protein_g         | full26  | cv_r2_mean  |  0.97188    |
| D4        | uncertainty_score | mech10  | cv_r2_mean  |  0.212023   |
| D4        | uncertainty_score | res16   | cv_r2_mean  | -0.532546   |
| D4        | uncertainty_score | full26  | cv_r2_mean  |  0.243329   |
| D4        | meal_type         | mech10  | cv_acc_mean |  0.976812   |
| D4        | meal_type         | res16   | cv_acc_mean |  0.988406   |
| D4        | meal_type         | full26  | cv_acc_mean |  0.982609   |
| D3        | carb_g            | mech10  | cv_r2_mean  |  0.564651   |
| D3        | carb_g            | res16   | cv_r2_mean  |  0.886796   |
| D3        | carb_g            | full26  | cv_r2_mean  |  0.843238   |
| D3        | fat_g             | mech10  | cv_r2_mean  |  0.20709    |
| D3        | fat_g             | res16   | cv_r2_mean  |  0.775665   |
| D3        | fat_g             | full26  | cv_r2_mean  |  0.792834   |
| D3        | protein_g         | mech10  | cv_r2_mean  |  0.671764   |
| D3        | protein_g         | res16   | cv_r2_mean  |  0.886166   |
| D3        | protein_g         | full26  | cv_r2_mean  |  0.926613   |
| D3        | uncertainty_score | mech10  | cv_r2_mean  |  0.00936966 |
| D3        | uncertainty_score | res16   | cv_r2_mean  | -0.0346585  |
| D3        | uncertainty_score | full26  | cv_r2_mean  | -0.208479   |
| D3        | meal_type         | mech10  | cv_acc_mean |  0.403578   |
| D3        | meal_type         | res16   | cv_acc_mean |  0.375984   |
| D3        | meal_type         | full26  | cv_acc_mean |  0.432851   |

## Stratified ICC(1,1) (within-subject repeatability of mech dims vs ||z_res||)
| stratum               |   n_rows |   n_subjects | dim         |       icc1 |
|:----------------------|---------:|-------------:|:------------|-----------:|
| all                   |      345 |           30 | z00         | -0.0613689 |
| all                   |      345 |           30 | z01         |  0.227642  |
| all                   |      345 |           30 | z02         | -0.0733974 |
| all                   |      345 |           30 | z03         |  0.0684454 |
| all                   |      345 |           30 | z04         |  0.211099  |
| all                   |      345 |           30 | z05         |  0.192566  |
| all                   |      345 |           30 | z06         |  0.401878  |
| all                   |      345 |           30 | z07         |  0.255266  |
| all                   |      345 |           30 | z08         |  0.176332  |
| all                   |      345 |           30 | z09         |  0.245909  |
| all                   |      345 |           30 | ||z_res||16 | -0.0147044 |
| meal_type=Cornflakes  |      117 |           30 | z00         |  0.822964  |
| meal_type=Cornflakes  |      117 |           30 | z01         |  0.805103  |
| meal_type=Cornflakes  |      117 |           30 | z02         |  0.694723  |
| meal_type=Cornflakes  |      117 |           30 | z03         |  0.79124   |
| meal_type=Cornflakes  |      117 |           30 | z04         |  0.788398  |
| meal_type=Cornflakes  |      117 |           30 | z05         |  0.824356  |
| meal_type=Cornflakes  |      117 |           30 | z06         |  0.739141  |
| meal_type=Cornflakes  |      117 |           30 | z07         |  0.731125  |
| meal_type=Cornflakes  |      117 |           30 | z08         |  0.661807  |
| meal_type=Cornflakes  |      117 |           30 | z09         |  0.777972  |
| meal_type=Cornflakes  |      117 |           30 | ||z_res||16 |  0.732042  |
| meal_type=PB_sandwich |      111 |           30 | z00         |  0.54634   |
| meal_type=PB_sandwich |      111 |           30 | z01         |  0.655255  |
| meal_type=PB_sandwich |      111 |           30 | z02         |  0.437244  |
| meal_type=PB_sandwich |      111 |           30 | z03         |  0.509187  |
| meal_type=PB_sandwich |      111 |           30 | z04         |  0.313837  |
| meal_type=PB_sandwich |      111 |           30 | z05         |  0.626227  |
| meal_type=PB_sandwich |      111 |           30 | z06         |  0.38044   |
| meal_type=PB_sandwich |      111 |           30 | z07         |  0.515571  |
| meal_type=PB_sandwich |      111 |           30 | z08         |  0.464249  |
| meal_type=PB_sandwich |      111 |           30 | z09         |  0.674265  |
| meal_type=PB_sandwich |      111 |           30 | ||z_res||16 |  0.671394  |
| meal_type=Protein_bar |      117 |           30 | z00         |  0.440423  |
| meal_type=Protein_bar |      117 |           30 | z01         |  0.647126  |
| meal_type=Protein_bar |      117 |           30 | z02         |  0.368327  |
| meal_type=Protein_bar |      117 |           30 | z03         |  0.435543  |
| meal_type=Protein_bar |      117 |           30 | z04         |  0.635935  |
| meal_type=Protein_bar |      117 |           30 | z05         |  0.654173  |
| meal_type=Protein_bar |      117 |           30 | z06         |  0.558349  |

_(ICC table truncated in report; full CSV has 77 rows)_