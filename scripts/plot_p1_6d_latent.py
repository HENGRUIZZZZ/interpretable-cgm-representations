"""
6D 潜变量可视化：pairwise 散点矩阵、PCA 2D、平行坐标。

从 latent_and_gold_all.csv 读取 6 维 (tau_m, Gb, sg, si, p2, mi)，按 dataset 或金标准着色。

用法（项目根目录）：
  python scripts/plot_p1_6d_latent.py --csv paper1_results_v4/run_s21_lam0.05/latent_and_gold_all.csv --out paper1_results_v4/figures
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 6D 列名（与 pipeline 一致）
D6_COLS = ["tau_m", "Gb", "sg", "si", "p2", "mi"]
COLORS_DS = {"D1": "#1f77b4", "D2": "#ff7f0e", "D4": "#2ca02c"}
MARKERS_DS = {"D1": "o", "D2": "s", "D4": "^"}


def main():
    parser = argparse.ArgumentParser(description="Plot 6D latent: pairwise matrix, PCA, parallel coordinates.")
    parser.add_argument("--csv", required=True, help="Path to latent_and_gold_all.csv")
    parser.add_argument("--out", required=True, help="Output directory for figures")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"File not found: {args.csv}")
        return
    df = pd.read_csv(args.csv)
    for c in D6_COLS:
        if c not in df.columns:
            print(f"Missing column: {c}")
            return
    df = df.dropna(subset=D6_COLS).copy()
    if "dataset_id" not in df.columns:
        df["dataset_id"] = "all"
    df["dataset_id"] = df["dataset_id"].astype(str)
    os.makedirs(args.out, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required")
        return

    X = df[D6_COLS].values
    labels_ds = df["dataset_id"].values
    n_samples = len(df)

    # ----- 1) Pairwise scatter matrix (6x6, lower triangle), colored by dataset -----
    fig, axes = plt.subplots(6, 6, figsize=(12, 12))
    for i in range(6):
        for j in range(6):
            ax = axes[i, j]
            if i < j:
                ax.set_visible(False)
                continue
            if i == j:
                ax.hist(X[:, i], bins=min(25, max(5, n_samples // 5)), color="gray", alpha=0.7, edgecolor="black")
                ax.set_ylabel(D6_COLS[i], fontsize=8)
                ax.set_xlabel(D6_COLS[i], fontsize=8)
            else:
                for did in np.unique(labels_ds):
                    mask = labels_ds == did
                    if mask.sum() == 0:
                        continue
                    c = COLORS_DS.get(did, "gray")
                    m = MARKERS_DS.get(did, "o")
                    ax.scatter(X[mask, j], X[mask, i], alpha=0.6, s=15, c=c, marker=m, label=did)
                ax.set_ylabel(D6_COLS[i], fontsize=8)
                ax.set_xlabel(D6_COLS[j], fontsize=8)
            ax.tick_params(labelsize=6)
    # Legend on first scatter panel (1,0)
    axes[1, 0].legend(loc="upper right", fontsize=6)
    plt.suptitle("6D latent: pairwise (colored by dataset)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "p1_6d_pairwise_by_dataset.png"), dpi=150)
    plt.close()
    print(f"Saved {args.out}/p1_6d_pairwise_by_dataset.png")

    # ----- 2) PCA 2D: colored by dataset and by SSPG/DI (if available) -----
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("sklearn required for PCA; skip PCA figure.")
    else:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        Z = pca.fit_transform(Xs)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # Left: by dataset
        ax = axes[0]
        for did in np.unique(labels_ds):
            mask = labels_ds == did
            c = COLORS_DS.get(did, "gray")
            m = MARKERS_DS.get(did, "o")
            ax.scatter(Z[mask, 0], Z[mask, 1], alpha=0.7, s=30, c=c, marker=m, label=f"{did} (n={mask.sum()})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend()
        ax.set_title("6D latent → PCA 2D (by dataset)")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        # Right: by SSPG or DI if available
        ax = axes[1]
        if "sspg" in df.columns and df["sspg"].notna().sum() >= 5:
            idx = df["sspg"].notna().values
            Z_sub = Z[idx]
            sspg = df.loc[idx, "sspg"].values
            sc = ax.scatter(Z_sub[:, 0], Z_sub[:, 1], c=sspg, cmap="viridis", alpha=0.8, s=30)
            plt.colorbar(sc, ax=ax, label="SSPG (gold)")
            ax.set_title("6D latent → PCA 2D (colored by SSPG)")
        elif "di" in df.columns and df["di"].notna().sum() >= 5:
            idx = df["di"].notna().values
            Z_sub = Z[idx]
            di = df.loc[idx, "di"].values
            sc = ax.scatter(Z_sub[:, 0], Z_sub[:, 1], c=di, cmap="plasma", alpha=0.8, s=30)
            plt.colorbar(sc, ax=ax, label="DI (gold)")
            ax.set_title("6D latent → PCA 2D (colored by DI)")
        else:
            ax.text(0.5, 0.5, "No SSPG/DI for coloring", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "p1_6d_pca2d.png"), dpi=150)
        plt.close()
        print(f"Saved {args.out}/p1_6d_pca2d.png")

    # ----- 3) Parallel coordinates: 6D, colored by dataset -----
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    xx = np.arange(6)
    for did in np.unique(labels_ds):
        mask = labels_ds == did
        c = COLORS_DS.get(did, "gray")
        for k in range(min(100, mask.sum())):  # cap lines for readability
            idx = np.where(mask)[0][k % mask.sum()]
            ax.plot(xx, X[idx], color=c, alpha=0.3, linewidth=0.8)
    ax.set_xticks(xx)
    ax.set_xticklabels(D6_COLS)
    ax.set_ylabel("latent value")
    ax.set_title("6D latent: parallel coordinates (by dataset, max 100 lines per dataset)")
    ax.legend([plt.Line2D([0], [0], color=COLORS_DS.get(d, "gray"), lw=2) for d in np.unique(labels_ds)], list(np.unique(labels_ds)))
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "p1_6d_parallel_by_dataset.png"), dpi=150)
    plt.close()
    print(f"Saved {args.out}/p1_6d_parallel_by_dataset.png")

    # ----- 4) Per-dimension boxplot by dataset -----
    uniq_ds = list(np.unique(labels_ds))
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    positions = []
    data_list = []
    colors_list = []
    for di in range(6):
        for dsi, ds in enumerate(uniq_ds):
            mask = labels_ds == ds
            data_list.append(X[mask, di])
            positions.append(di * (len(uniq_ds) + 0.8) + dsi * 0.8)
            colors_list.append(COLORS_DS.get(ds, "gray"))
    bp = ax.boxplot(data_list, positions=positions, widths=0.35, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors_list):
        patch.set_facecolor(c)
    ax.set_xticks(np.arange(6) * (len(uniq_ds) + 0.8) + (len(uniq_ds) * 0.4))
    ax.set_xticklabels(D6_COLS)
    ax.set_ylabel("latent value")
    ax.set_title("6D latent: distribution by dimension and dataset")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COLORS_DS.get(d, "gray"), label=d) for d in uniq_ds])
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "p1_6d_boxplot_by_dataset.png"), dpi=150)
    plt.close()
    print(f"Saved {args.out}/p1_6d_boxplot_by_dataset.png")
    print("Done (6D figures).")


if __name__ == "__main__":
    main()
