# V9: Multi-task tri-class (IS / IR-Compensated / IR-Decompensated)

- **Model**: VAE-ODE with DI-expert regression (λ_sspg=0, λ_di=1.0) + tri-class head (6D latent → 3 classes).
- **Tri-class**: SSPG cut=120, DI cut=1.2 → 0=IS, 1=IR-Compensated, 2=IR-Decompensated.
- **λ_cls grid**: 0.1, 0.5, 1.0 (all three runs completed).

## Contents

- `lambda_cls_0.1/`, `lambda_cls_0.5/`, `lambda_cls_1.0/`: pipeline outputs (latent CSVs, training curves, e2e head metrics).
- `v9_tri_class_lodo_summary.json`: LODO tri-class accuracy per run.
- `v9_evaluation_report.txt`: short summary report.

## LODO tri-class evaluation

Leave-one-dataset-out (train on D2 test on D1, train on D1 test on D2). Classifier: LogisticRegression on 26D latent.  
See `scripts/run_v9_tri_class_lodo.py`.
