# New_GlucoVector v6 experiments (using `P1_final_with_D4_DI.zip`)

This folder contains a **fresh rerun** of the GlucoVector v6 experiment plan using the **latest dataset** `P1_final_with_D4_DI.zip`.

## Data

- **Source zip**: `/Users/hertz1030/Downloads/P1_final_with_D4_DI.zip`
- **Extracted to**: `New_data/P1_final_with_D4_DI/P1_final/`
- All runs set: `CGM_PROJECT_OUTPUT=New_data/P1_final_with_D4_DI/P1_final`

## Experiments (per v6 guide)

- **Exp1**: SSPG prediction — train on D1+D2, independent test on D4
- **Exp2**: DI prediction — train on D1+D2, independent test on D4
- **Exp3**: D2 meal-type response analysis
- **Exp4**: Feature ablation (tiers) with LODO-CV on D1+D2
- **Exp5**: D4 cross-context stability (OGTT vs standard meal vs free-living)

## Outputs

- `New_exp1_sspg/` and `New_exp2_di/`: training artifacts + D4 evaluation CSVs and metrics
- `New_exp3_meal_type/`: per-meal-type tables and intermediate features
- `New_exp4_ablation/`: tier tables and LODO fold results
- `New_exp5_stability/`: context-wise predictions and correlation matrices

## Core code (New_ prefix)

- `scripts/New_run_glucovector_v6_all.py`: orchestrates all experiments
- `scripts/New_eval_trainD1D2_testD4.py`: shared training+independent-test evaluation utilities
- `scripts/New_exp3_meal_type_analysis.py`
- `scripts/New_exp4_feature_ablation_lodo.py`
- `scripts/New_exp5_d4_context_stability.py`

