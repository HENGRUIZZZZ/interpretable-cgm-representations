"""
实验方案 v4.0 阶段三、四：出图与临床亚型分析。

- 柱状图：三路 Spearman r (SSPG / DI)
- SHAP：XGBoost 特征重要性（若已跑 bakeoff）
- 散点：IR_proxy vs SSPG、BCF_proxy vs DI
- PC1–PC2 着色 SSPG/DI
- 代谢亚型：SSPG>180 + DI 中位数四分 → Healthy / IR / β-cell defect / Mixed；四象限图
- 简短报告

用法：
  python scripts/analyze_results.py --full_features paper1_results_v4/baseline_seed42/full_features_v4.csv \\
       --bakeoff_json paper1_results_v4/baseline_seed42/bakeoff_results.json \\
       --output_dir paper1_results_v4/baseline_seed42
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Analyze bake-off results and produce plots + metabolic subtype")
    parser.add_argument("--full_features", type=str, required=True, help="Path to full_features_v4.csv")
    parser.add_argument("--bakeoff_json", type=str, default=None, help="Path to bakeoff_results.json")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.full_features):
        print(f"File not found: {args.full_features}")
        sys.exit(1)
    out_dir = args.output_dir or os.path.dirname(args.full_features)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.full_features)
    has_sspg = "sspg" in df.columns and df["sspg"].notna().sum() > 5
    has_di = "di" in df.columns and df["di"].notna().sum() > 5

    # ----- Bar chart: 3 routes x SSPG/DI Spearman r -----
    if args.bakeoff_json and os.path.isfile(args.bakeoff_json):
        with open(args.bakeoff_json) as f:
            bake = json.load(f)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            bake = None
        if bake is not None:
            routes = ["RouteA_PCA2_Ridge", "RouteB_XGBoost", "RouteC_Ridge_stats"]
            targets = ["sspg", "di"]
            r_vals = {t: [bake.get(f"{r}_{t}_spearman_r", np.nan) for r in routes] for t in targets}
            x = np.arange(len(routes))
            w = 0.35
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(x - w/2, [r_vals["sspg"][i] for i in range(3)], w, label="SSPG")
            ax.bar(x + w/2, [r_vals["di"][i] for i in range(3)], w, label="DI")
            ax.set_xticks(x)
            ax.set_xticklabels(["A: PCA-Ridge", "B: XGBoost", "C: Stats-Ridge"])
            ax.set_ylabel("Spearman r")
            ax.legend()
            ax.set_title("Bake-off: Spearman r by route")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "bakeoff_bar_spearman.png"), dpi=150)
            plt.close()
            print(f"Saved {out_dir}/bakeoff_bar_spearman.png")
    else:
        bake = {}

    # ----- Scatter: IR_proxy (1/si) vs SSPG, BCF_proxy (mi) vs DI -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    if plt and "si" in df.columns and has_sspg:
        sub = df[["si", "sspg"]].dropna()
        if len(sub) > 5:
            ir_proxy = 1.0 / (sub["si"].values + 1e-12)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(ir_proxy, sub["sspg"].values, alpha=0.7)
            ax.set_xlabel("IR_proxy (1/si)")
            ax.set_ylabel("SSPG")
            r, _ = stats.spearmanr(ir_proxy, sub["sspg"].values)
            ax.set_title(f"IR_proxy vs SSPG (r={r:.3f})")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "scatter_IR_proxy_vs_SSPG.png"), dpi=150)
            plt.close()
            print(f"Saved {out_dir}/scatter_IR_proxy_vs_SSPG.png")
    if plt and "mi" in df.columns and has_di:
        sub = df[["mi", "di"]].dropna()
        if len(sub) > 5:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(sub["mi"].values, sub["di"].values, alpha=0.7)
            ax.set_xlabel("BCF_proxy (mi)")
            ax.set_ylabel("DI")
            r, _ = stats.spearmanr(sub["mi"].values, sub["di"].values)
            ax.set_title(f"BCF_proxy vs DI (r={r:.3f})")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "scatter_BCF_proxy_vs_DI.png"), dpi=150)
            plt.close()
            print(f"Saved {out_dir}/scatter_BCF_proxy_vs_DI.png")

    # ----- PC1 vs PC2 (6D ODE), colored by SSPG / DI -----
    ode_cols = ["tau_m", "Gb", "sg", "si", "p2", "mi"]
    if plt and all(c in df.columns for c in ode_cols):
        X_df = df[ode_cols].dropna()
        if len(X_df) > 10:
            X = StandardScaler().fit_transform(X_df)
            pc = PCA(n_components=2).fit_transform(X)
            df_pc = df.loc[X_df.index].copy()
            df_pc["PC1"] = pc[:, 0]
            df_pc["PC2"] = pc[:, 1]
            for target, label in [("sspg", "SSPG"), ("di", "DI")]:
                if target not in df_pc.columns or df_pc[target].notna().sum() < 5:
                    continue
                fig, ax = plt.subplots(figsize=(5, 4))
                sc = ax.scatter(df_pc["PC1"], df_pc["PC2"], c=df_pc[target], cmap="viridis", alpha=0.8)
                plt.colorbar(sc, ax=ax, label=label)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title(f"PCA(2) on 6D ODE, colored by {label}")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"pc1_pc2_colored_{target}.png"), dpi=150)
                plt.close()
                print(f"Saved {out_dir}/pc1_pc2_colored_{target}.png")

    # ----- Metabolic subtype: SSPG>180, DI median split → 4 quadrants -----
    if has_sspg and has_di:
        sub = df[["sspg", "di"]].dropna()
        if len(sub) > 10:
            sspg_high = sub["sspg"] > 180
            di_med = sub["di"].median()
            di_high = sub["di"] > di_med
            subtype = np.where(~sspg_high & di_high, "Healthy",
                               np.where(sspg_high & di_high, "Mixed",
                                        np.where(sspg_high & ~di_high, "IR", "β-cell defect")))
            sub = sub.copy()
            sub["subtype"] = subtype
            if plt:
                fig, ax = plt.subplots(figsize=(5, 5))
                for st in ["Healthy", "IR", "β-cell defect", "Mixed"]:
                    m = sub["subtype"] == st
                    if m.sum() > 0:
                        ax.scatter(sub.loc[m, "sspg"], sub.loc[m, "di"], label=st, alpha=0.7)
                ax.axvline(180, color="gray", linestyle="--")
                ax.axhline(di_med, color="gray", linestyle="--")
                ax.set_xlabel("SSPG")
                ax.set_ylabel("DI")
                ax.legend()
                ax.set_title("Metabolic subtype (SSPG>180, DI median)")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, "metabolic_subtype_quadrant.png"), dpi=150)
                plt.close()
                print(f"Saved {out_dir}/metabolic_subtype_quadrant.png")
            counts = sub["subtype"].value_counts()
            with open(os.path.join(out_dir, "metabolic_subtype_counts.txt"), "w") as f:
                f.write("Metabolic subtype counts:\n")
                for st in ["Healthy", "IR", "β-cell defect", "Mixed"]:
                    f.write(f"  {st}: {counts.get(st, 0)}\n")

    # ----- Short report -----
    lines = [
        "Experiment plan v4.0 — analysis report",
        "=====================================",
        f"Full features: {args.full_features}",
        f"Rows: {len(df)}, with SSPG: {df['sspg'].notna().sum() if 'sspg' in df.columns else 0}, with DI: {df['di'].notna().sum() if 'di' in df.columns else 0}",
        "",
    ]
    if bake:
        lines.append("Bake-off Spearman r:")
        for k in sorted(bake.keys()):
            if "spearman_r" in k:
                lines.append(f"  {k}: {bake[k]:.4f}")
    lines.append("")
    lines.append("Plots and subtype counts written to output_dir.")
    report_path = os.path.join(out_dir, "analysis_report_v4.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
