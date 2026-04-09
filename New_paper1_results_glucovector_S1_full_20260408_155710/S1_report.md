# GlucoVector S1 Full Execution Report

Generated: 2026-04-08T15:59:24.432982

## Plan A (Risk stratification, from v19)

- Model: `GV_Baseline(Exp5)`
- SSPG Spearman: **0.759**
- DI Spearman: **0.662**
- IR AUROC: **0.903**
- Decomp AUROC: **0.928**
- Cross-meal ICC (SSPG pred): **-0.059**

## Plan B (E2E Joint, 10D-only)

- Checkpoint: `/Users/hertz1030/Documents/GitHub/interpretable-cgm-representations/New_paper1_results_glucovector_S1_full_20260408_155710/S1_planB_E2E_10D/autoencoder_p1_full.pt`
- SSPG Spearman: **0.794** (95% bootstrap CI [0.541, 0.944])
- DI Spearman: **0.544** (95% bootstrap CI [0.026, 0.875])
- IR AUROC: **0.903**
- Decomp AUROC: **0.904**
- Cross-meal ICC (SSPG pred): **-0.236**
- si(z03) vs SSPG Spearman: **-0.579**
- mi(z05) vs DI Spearman: **0.709**

## S1 Success Criteria Check

- C1 Param alignment (si<-0.5 and mi>0.5): **True**
- C2 Accuracy keep (SSPG>=0.70 and DI>=0.60): **False**
- C3 ICC uplift (SSPG ICC>0.5): **False**
- Overall pass: **False**

## Output Files

- `S1_summary.json`
- `S1_planA_overall_table_from_v19.csv`
- `S1_planA_clf_table_from_v19.csv`
- `S1_planA_icc_table_from_v19.csv`
- `S1_planB_subject_true_pred.csv`
- `S1_planB_overall_sspg.csv`
- `S1_planB_overall_di.csv`