"""
VAE 拟合质量评估：训练曲线、验证集重建 MSE 分布、示例 CGM 实际 vs 预测。

依赖 run_p1_full_pipeline 产出的：
  - run_dir/training_curves.json（每 epoch train/val loss）
  - run_dir/reconstruction_val_mse.npy（每样本 val MSE）
  - run_dir/reconstruction_examples.npz（actual, pred 若干条）
  - run_dir/autoencoder_p1_full.pt（含 train_mean/train_std，用于反归一化出图）

用法（项目根目录）：
  python scripts/assess_vae_fit.py --run-dir paper1_results_v4/run_s21_lam0.05 --out paper1_results_v4/figures
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Assess VAE fit: training curve, reconstruction MSE, example curves.")
    parser.add_argument("--run-dir", required=True, help="Run directory (e.g. paper1_results_v4/run_s21_lam0.05)")
    parser.add_argument("--out", required=True, help="Output directory for figures")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"Run dir not found: {run_dir}")
        return
    os.makedirs(args.out, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required")
        return

    # ----- 1) Training curve -----
    curves_path = os.path.join(run_dir, "training_curves.json")
    if os.path.isfile(curves_path):
        import json
        with open(curves_path) as f:
            curves = json.load(f)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(curves["epoch"], curves["train_loss"], label="Train loss")
        ax.plot(curves["epoch"], curves["val_loss"], label="Val loss (recon MSE)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_title("VAE training curve")
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "p1_vae_training_curve.png"), dpi=150)
        plt.close()
        print(f"Saved {args.out}/p1_vae_training_curve.png")
    else:
        print(f"No {curves_path} (run pipeline to generate).")

    # ----- 2) Validation reconstruction MSE distribution -----
    mse_path = os.path.join(run_dir, "reconstruction_val_mse.npy")
    if os.path.isfile(mse_path):
        val_mse = np.load(mse_path)
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.hist(val_mse, bins=min(50, max(10, len(val_mse) // 5)), edgecolor="black", alpha=0.7)
        ax.axvline(val_mse.mean(), color="red", linestyle="--", label=f"Mean = {val_mse.mean():.4f}")
        ax.set_xlabel("Reconstruction MSE (per sample, val set)")
        ax.set_ylabel("Count")
        ax.set_title("VAE validation reconstruction MSE")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "p1_vae_reconstruction_mse_hist.png"), dpi=150)
        plt.close()
        print(f"Saved {args.out}/p1_vae_reconstruction_mse_hist.png (mean MSE = {val_mse.mean():.4f})")
    else:
        print(f"No {mse_path} (run pipeline to generate).")

    # ----- 3) Example CGM: actual vs predicted -----
    ex_path = os.path.join(run_dir, "reconstruction_examples.npz")
    ckpt_path = os.path.join(run_dir, "autoencoder_p1_full.pt")
    if os.path.isfile(ex_path) and os.path.isfile(ckpt_path):
        data = np.load(ex_path)
        actual = data["actual"]   # (n_ex, T, 1) normalized
        pred = data["pred"]       # (n_ex, T, 1) normalized
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        train_mean = ckpt.get("train_mean")
        train_std = ckpt.get("train_std")
        if train_mean is not None and train_std is not None:
            m0 = np.array(train_mean[0]).reshape(1, -1, 1)
            s0 = np.array(train_std[0]).reshape(1, -1, 1) + 1e-8
            actual = actual * s0 + m0
            pred = pred * s0 + m0
        n_ex = actual.shape[0]
        fig, axes = plt.subplots(n_ex, 1, figsize=(8, 2 * n_ex), sharex=True)
        if n_ex == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            t = np.arange(actual.shape[1]) * 5  # 5 min
            ax.plot(t, actual[i, :, 0], label="Actual CGM", color="C0")
            ax.plot(t, pred[i, :, 0], label="Reconstructed", color="C1", alpha=0.8)
            ax.set_ylabel("Glucose (mg/dL)")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"Example {i+1}")
        axes[-1].set_xlabel("Time (min)")
        plt.suptitle("VAE reconstruction: actual vs predicted CGM (validation samples)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "p1_vae_reconstruction_examples.png"), dpi=150)
        plt.close()
        print(f"Saved {args.out}/p1_vae_reconstruction_examples.png")
    else:
        if not os.path.isfile(ex_path):
            print(f"No {ex_path} (run pipeline to generate).")
        if os.path.isfile(ex_path) and not os.path.isfile(ckpt_path):
            print("No checkpoint for denormalization; plot in normalized space.")
            data = np.load(ex_path)
            actual, pred = data["actual"], data["pred"]
            n_ex = actual.shape[0]
            fig, axes = plt.subplots(n_ex, 1, figsize=(8, 2 * n_ex), sharex=True)
            if n_ex == 1:
                axes = [axes]
            for i, ax in enumerate(axes):
                t = np.arange(actual.shape[1]) * 5
                ax.plot(t, actual[i, :, 0], label="Actual (norm)")
                ax.plot(t, pred[i, :, 0], label="Pred (norm)", alpha=0.8)
                ax.set_ylabel("Normalized")
                ax.legend(loc="upper right", fontsize=8)
            axes[-1].set_xlabel("Time (min)")
            plt.suptitle("VAE reconstruction (normalized)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "p1_vae_reconstruction_examples.png"), dpi=150)
            plt.close()
            print(f"Saved {args.out}/p1_vae_reconstruction_examples.png")

    print("Done (VAE fit assessment).")


if __name__ == "__main__":
    main()
