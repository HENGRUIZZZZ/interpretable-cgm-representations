"""
根据 run_p1_full_pipeline 产出的 latent_and_gold_*.csv 与 correlations.txt 出图。

生成：
  - si vs SSPG 散点（按 dataset 着色）+ 回归线 + Spearman r
  - DI_model (mi) vs DI 散点 + 回归线 + r
  - Bland-Altman：si vs SSPG
  - 汇总表（n, r, p）

用法（在项目根目录）：
  python scripts/plot_p1_results.py
  # 各取最好：SSPG 用 SSPG 最优 run 的 latent，DI 用 DI 最优 run 的 latent
  python scripts/plot_p1_results.py --csv-sspg paper1_results/run_s21_lam0.05/latent_and_gold_all.csv --csv-di paper1_results/tune_0.01/latent_and_gold_all.csv
"""
import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# 颜色与标记（全局，多处复用）
COLORS = {"D1": "#1f77b4", "D2": "#ff7f0e", "D4": "#2ca02c"}
MARKERS = {"D1": "o", "D2": "s", "D4": "^"}


def _parse_5fold_spearman(summary_path: str) -> tuple[float, float]:
    """Parse sspg_r, di_r from evaluation_metrics_summary.txt (first 5fold_subject spearman per target)."""
    sspg_r, di_r = float("nan"), float("nan")
    if not os.path.isfile(summary_path):
        return sspg_r, di_r
    with open(summary_path) as f:
        text = f.read()
    current = None
    for line in text.splitlines():
        if line.strip().startswith("Target:"):
            m = re.search(r"Target:\s*(\w+)", line)
            current = m.group(1).strip() if m else None
            continue
        if "5fold_subject" in line and "spearman_r" in line:
            m = re.search(r"spearman_r\s*=\s*([-\d.]+)", line)
            if m:
                v = float(m.group(1))
                if current == "sspg":
                    sspg_r = v
                elif current == "di":
                    di_r = v
            continue
    return sspg_r, di_r


def find_best_sspg_and_di_runs(results_root: str = "paper1_results") -> tuple[str | None, str | None]:
    """在 results_root 下扫描 tune_* 与 run_s*_lam*，返回 (best_sspg_dir, best_di_dir) 相对路径。"""
    cand = []
    for name in os.listdir(results_root):
        path = os.path.join(results_root, name)
        if not os.path.isdir(path):
            continue
        if not name.startswith("tune_") and not name.startswith("run_s"):
            continue
        summary = os.path.join(path, "evaluation_metrics_summary.txt")
        csv_path = os.path.join(path, "latent_and_gold_all.csv")
        if not os.path.isfile(summary) or not os.path.isfile(csv_path):
            continue
        sspg_r, di_r = _parse_5fold_spearman(summary)
        cand.append({"dir": path, "name": name, "sspg_r": sspg_r, "di_r": di_r})
    if not cand:
        return None, None
    best_sspg = max(cand, key=lambda x: x["sspg_r"] if np.isfinite(x["sspg_r"]) else -2)
    best_di = max(cand, key=lambda x: x["di_r"] if np.isfinite(x["di_r"]) else -2)
    return best_sspg["dir"], best_di["dir"]


def main():
    parser = argparse.ArgumentParser(description="Plot P1 latent vs gold standard results.")
    parser.add_argument("--csv", default="paper1_results/latent_and_gold_all.csv", help="Path to latent_and_gold CSV (used if --csv-sspg/--csv-di not set)")
    parser.add_argument("--csv-sspg", default=None, help="CSV for SSPG figures (si vs SSPG, Bland-Altman); overrides --csv for these")
    parser.add_argument("--csv-di", default=None, help="CSV for DI figure (mi vs DI); overrides --csv for this")
    parser.add_argument("--out", default="paper1_results/figures", help="Output directory for figures")
    parser.add_argument("--correlations", default="paper1_results/correlations.txt", help="Path to correlations.txt")
    parser.add_argument("--best-per-target", action="store_true", help="Auto-detect best SSPG run and best DI run, then use each for its figure")
    parser.add_argument("--summary-figures", action="store_true", help="Draw summary figures (metrics bar, single vs joint, leave-one-dataset) from evaluation_metrics.csv")
    parser.add_argument("--metrics-csv", default=None, help="Path to evaluation_metrics.csv for summary figures (default: <out_dir>/../evaluation_metrics.csv)")
    args = parser.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found; install with: pip install matplotlib")
        return

    csv_sspg = args.csv_sspg or args.csv
    csv_di = args.csv_di or args.csv
    if getattr(args, "best_per_target", False):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 用 --out 的父目录作为 results 根，这样 P1_RESULTS_ROOT=paper1_results_v4 时扫的是 v4 下的 run_*
        out_abs = os.path.abspath(args.out)
        results_root = os.path.dirname(out_abs)
        if not os.path.isdir(results_root):
            results_root = os.path.join(repo_root, "paper1_results")
        best_sspg_dir, best_di_dir = find_best_sspg_and_di_runs(results_root)
        if best_sspg_dir:
            csv_sspg = os.path.join(best_sspg_dir, "latent_and_gold_all.csv")
            print(f"Best SSPG run -> {csv_sspg}")
        if best_di_dir:
            csv_di = os.path.join(best_di_dir, "latent_and_gold_all.csv")
            print(f"Best DI run   -> {csv_di}")

    def load_df(path: str):
        if not os.path.isfile(path):
            return None
        df = pd.read_csv(path)
        if "dataset_id" not in df.columns:
            df["dataset_id"] = "all"
        return df

    # ----- Fig 1: si vs SSPG（用 SSPG 最优 run 的 latent） -----
    df_sspg = load_df(csv_sspg)
    if df_sspg is None:
        print(f"CSV not found for SSPG: {csv_sspg}. Skip SSPG figures.")
    elif "si" in df_sspg.columns and "sspg" in df_sspg.columns:
        df = df_sspg.dropna(subset=["sspg", "si"])
        if not df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            for did in df["dataset_id"].unique():
                sub = df[df["dataset_id"] == did]
                if sub.empty:
                    continue
                c = COLORS.get(str(did), "gray")
                m = MARKERS.get(str(did), "o")
                ax.scatter(sub["sspg"], sub["si"], label=f"{did} (n={len(sub)})", color=c, marker=m, alpha=0.8)
            x = df["sspg"].values
            y = df["si"].values
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 3:
                r, p = scipy_stats.spearmanr(x[valid], y[valid])
                z = np.polyfit(x[valid], y[valid], 1)
                xl = np.array([x[valid].min(), x[valid].max()])
                ax.plot(xl, np.poly1d(z)(xl), "k--", alpha=0.8, label=f"Spearman r={r:.3f}, p={p:.3f}")
            ax.set_xlabel("SSPG (gold standard)")
            ax.set_ylabel("si (model)")
            ax.legend(loc="best", fontsize=8)
            ax.set_title("Model si vs SSPG (best-SSPG run)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "p1_si_vs_sspg.png"), dpi=150)
            plt.close()
            print(f"Saved {out_dir}/p1_si_vs_sspg.png")

    # ----- Fig 2: mi vs DI（用 DI 最优 run 的 latent） -----
    df_di = load_df(csv_di)
    if df_di is None:
        print(f"CSV not found for DI: {csv_di}. Skip DI figure.")
    elif "mi" in df_di.columns and "di" in df_di.columns:
        df = df_di.dropna(subset=["di", "mi"])
        if not df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            for did in df["dataset_id"].unique():
                sub = df[df["dataset_id"] == did]
                if sub.empty:
                    continue
                c = COLORS.get(str(did), "gray")
                m = MARKERS.get(str(did), "o")
                ax.scatter(sub["di"], sub["mi"], label=f"{did} (n={len(sub)})", color=c, marker=m, alpha=0.8)
            x = df["di"].values
            y = df["mi"].values
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 3:
                r, p = scipy_stats.spearmanr(x[valid], y[valid])
                z = np.polyfit(x[valid], y[valid], 1)
                xl = np.array([x[valid].min(), x[valid].max()])
                ax.plot(xl, np.poly1d(z)(xl), "k--", alpha=0.8, label=f"Spearman r={r:.3f}, p={p:.3f}")
            ax.set_xlabel("DI (gold standard)")
            ax.set_ylabel("mi (model)")
            ax.legend(loc="best", fontsize=8)
            ax.set_title("Model mi vs DI (best-DI run)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "p1_mi_vs_di.png"), dpi=150)
            plt.close()
            print(f"Saved {out_dir}/p1_mi_vs_di.png")

    # ----- Bland-Altman (si vs SSPG)，用 SSPG 最优 run -----
    if df_sspg is not None and "si" in df_sspg.columns and "sspg" in df_sspg.columns:
        df = df_sspg.dropna(subset=["sspg", "si"])
        x, y = df["sspg"].values, df["si"].values
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 3:
            x, y = x[valid], y[valid]
            mean = (x + y) / 2
            diff = y - x
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.scatter(mean, diff, alpha=0.8)
            ax.axhline(diff.mean(), color="k", linestyle="-")
            ax.axhline(diff.mean() + 1.96 * np.std(diff), color="k", linestyle="--")
            ax.axhline(diff.mean() - 1.96 * np.std(diff), color="k", linestyle="--")
            ax.set_xlabel("(SSPG + si) / 2")
            ax.set_ylabel("si - SSPG")
            ax.set_title("Bland-Altman: si vs SSPG (best-SSPG run)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "p1_blandaltman_si_sspg.png"), dpi=150)
            plt.close()
            print(f"Saved {out_dir}/p1_blandaltman_si_sspg.png")

    # ----- 汇总表（从 correlations.txt 或现场算） -----
    if os.path.isfile(args.correlations):
        cor = pd.read_csv(args.correlations, sep="\t")
        summary_path = os.path.join(out_dir, "p1_correlations_summary.txt")
        with open(summary_path, "w") as f:
            f.write(cor.to_string(index=False))
        print(f"Saved {summary_path}")

    # ----- 汇总图（r、R²、RMSE 等，便于理解） -----
    if getattr(args, "summary_figures", False):
        metrics_path = args.metrics_csv or os.path.join(os.path.dirname(out_dir), "evaluation_metrics.csv")
        if os.path.isfile(metrics_path):
            _plot_summary_figures(metrics_path, out_dir)
        else:
            print(f"Skip summary figures: {metrics_path} not found.")
    print("Done.")


def _plot_summary_figures(metrics_path: str, out_dir: str):
    """从 evaluation_metrics.csv 画：5-fold 指标柱状图、单 vs 联合、留一数据集出."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(metrics_path)
    os.makedirs(out_dir, exist_ok=True)

    # 1) 5-fold ridge_6d: SSPG 与 DI 的 Spearman r, R², RMSE
    sub = df[(df["validation"] == "5fold_subject") & (df["method"] == "ridge_6d") & (df["target"].isin(["sspg", "di"]))]
    if len(sub) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        for ax, metric, ylabel in zip(
            axes,
            ["spearman_r", "r2", "rmse"],
            ["Spearman r", "R²", "RMSE"],
        ):
            rows = sub[sub["metric"] == metric]
            targets = rows["target"].tolist()
            means = rows["mean"].values
            stds = rows["std"].values if "std" in rows.columns else np.zeros_like(means)
            x = np.arange(len(targets))
            bars = ax.bar(x, means, yerr=stds if np.isfinite(stds).all() else None, capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels([t.upper() for t in targets])
            ax.set_ylabel(ylabel)
            ax.set_title(f"5-fold Ridge(6D) — {ylabel}")
            if metric == "r2":
                ax.axhline(0, color="gray", linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "p1_metrics_summary.png"), dpi=150)
        plt.close()
        print(f"Saved {out_dir}/p1_metrics_summary.png")

    # 2) Single vs Joint: 5fold_single_vs_joint 的 Spearman r 对比（SSPG / DI）
    sub = df[df["validation"] == "5fold_single_vs_joint"]
    sub = sub[sub["metric"] == "spearman_r"]
    for target in ["sspg", "di"]:
        rows = sub[sub["target"] == target]
        if len(rows) < 3:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        methods = rows["method"].map(lambda m: "mi" if m == "single_mi" else "si" if m == "single_si" else "6D Ridge")
        x = np.arange(len(methods))
        means = rows["mean"].values
        stds = rows["std"].values
        ax.bar(x, means, yerr=stds, capsize=4, color=["#ff7f0e", "#2ca02c", "#1f77b4"])
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("Spearman r (5-fold)")
        ax.set_title(f"Single latent vs 6D Ridge — {target.upper()}")
        ax.axhline(0, color="gray", linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"p1_single_vs_joint_{target}.png"), dpi=150)
        plt.close()
        print(f"Saved {out_dir}/p1_single_vs_joint_{target}.png")

    # 2b) Single vs Joint RMSE（SSPG / DI 分别作图）
    sub_rmse = df[df["validation"] == "5fold_single_vs_joint"].copy()
    sub_rmse = sub_rmse[sub_rmse["metric"] == "rmse"]
    for target in ["sspg", "di"]:
        rows = sub_rmse[sub_rmse["target"] == target]
        if len(rows) < 3:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        methods = rows["method"].map(lambda m: "mi" if m == "single_mi" else "si" if m == "single_si" else "6D Ridge")
        x = np.arange(len(methods))
        means = rows["mean"].values
        stds = rows["std"].values
        ax.bar(x, means, yerr=stds, capsize=4, color=["#ff7f0e", "#2ca02c", "#1f77b4"])
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("RMSE (5-fold)")
        ax.set_title(f"Single latent vs 6D Ridge — {target.upper()} (RMSE)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"p1_single_vs_joint_{target}_rmse.png"), dpi=150)
        plt.close()
        print(f"Saved {out_dir}/p1_single_vs_joint_{target}_rmse.png")

    # 3) Leave-one-dataset-out: 各 test 数据集的 Spearman r
    sub = df[(df["validation"] == "leave_one_dataset_out") & (df["target"].isin(["sspg", "di"]))]
    if not sub.empty:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x = np.arange(len(sub))
        r = sub["spearman_r"].values
        colors = ["#1f77b4" if t == "sspg" else "#ff7f0e" for t in sub["target"]]
        labels = [f"{row['target'].upper()} test={row['test_dataset']}" for _, row in sub.iterrows()]
        ax.bar(x, r, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Spearman r")
        ax.set_title("Leave-one-dataset-out")
        ax.axhline(0, color="gray", linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "p1_leave_one_dataset.png"), dpi=150)
        plt.close()
        print(f"Saved {out_dir}/p1_leave_one_dataset.png")


if __name__ == "__main__":
    main()
