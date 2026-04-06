# P1 Diagnostic Report

## 1. Purpose

Check (1) VAE fit quality, (2) 6D latent structure, (3) regression method and data scope, to locate where prediction performance might be limited.

## 2. VAE fit (model must fit first)

- **Training curve**: see `figures/p1_vae_training_curve.png`. Train/val loss should decrease; val loss (reconstruction MSE) indicates fit.
- **Validation reconstruction MSE**: mean = 0.2778, std = 0.3664. See `figures/p1_vae_reconstruction_mse_hist.png`.
- **Example CGM**: `figures/p1_vae_reconstruction_examples.png` (actual vs reconstructed).

To generate VAE fit figures (training curve, reconstruction MSE, examples), re-run the pipeline once so the run dir gets `training_curves.json`, `reconstruction_val_mse.npy`, `reconstruction_examples.npz`, then re-run this diagnostic.

## 3. 6D latent space

We use 6 interpretable dimensions: tau_m, Gb, sg, si, p2, mi. Figures:
- `figures/p1_6d_pairwise_by_dataset.png`: pairwise scatter (lower triangle), colored by dataset.
- `figures/p1_6d_pca2d.png`: PCA 2D by dataset and by SSPG/DI.
- `figures/p1_6d_parallel_by_dataset.png`: parallel coordinates.
- `figures/p1_6d_boxplot_by_dataset.png`: per-dimension distribution by dataset.

## 4. Linear vs nonlinear (all scenarios)

Table: `all_linear_vs_nonlinear.csv`. Figures: `figures/p1_all_scenarios_sspg.png`, `figures/p1_all_scenarios_di.png`, `figures/p1_all_scenarios_heatmap.png`.

## 5. Where might the problem be?

- **If VAE reconstruction is poor**: model or data preprocessing issue; fix fit before interpreting latent.
- **If 6D structure is messy (e.g. no separation by dataset/gold)**: representation may not align with physiology.
- **If linear is consistently worse than poly2/GB in heatmap**: consider nonlinear heads or report both.
- **If D1+D2-only is much better than D1+D2+D4**: D4 scope or scaling may need separate handling.
