from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _build_e2e_head(input_dim: int = 26) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2),
    )


def _infer_input_dim(state_dict: dict) -> int:
    w = state_dict.get("0.weight", None)
    return int(w.shape[1]) if w is not None else 26


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--train_latent_csv", required=True, type=str)
    ap.add_argument("--target", required=True, choices=["sspg", "di"])
    ap.add_argument("--out_png", required=True, type=str)
    ap.add_argument("--out_csv", required=True, type=str)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    try:
        import shap
    except Exception as e:
        pd.DataFrame([{"feature": "SHAP_NOT_AVAILABLE", "importance": str(e)}]).to_csv(args.out_csv, index=False)
        return

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    input_dim = _infer_input_dim(ckpt["e2e_head_state"])
    head = _build_e2e_head(input_dim)
    head.load_state_dict(ckpt["e2e_head_state"], strict=True)
    head.eval()

    feat_cols_26 = ["tau_m", "Gb", "sg", "si", "p2", "mi"] + [f"z_init_{i}" for i in range(4)] + [f"z_nonseq_{i}" for i in range(16)]
    feat_cols_10 = ["tau_m", "Gb", "sg", "si", "p2", "mi"] + [f"z_init_{i}" for i in range(4)]
    feat_cols = feat_cols_10 if input_dim == 10 else feat_cols_26
    df = pd.read_csv(args.train_latent_csv)
    X = df[feat_cols].to_numpy(dtype=float)
    X_bg = X[: min(64, len(X))]
    X_eval = X[: min(48, len(X))]
    idx = 0 if args.target == "sspg" else 1

    def f(x: np.ndarray) -> np.ndarray:
        xt = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y = head(xt).numpy()[:, idx]
        return y

    explainer = shap.KernelExplainer(f, X_bg)
    shap_vals = np.asarray(explainer.shap_values(X_eval, nsamples=100))
    mean_abs = np.abs(shap_vals).mean(axis=0)
    imp = pd.DataFrame({"feature": feat_cols, "importance": mean_abs}).sort_values("importance", ascending=False)
    imp.to_csv(args.out_csv, index=False)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_eval, feature_names=feat_cols, show=False)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=220, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

