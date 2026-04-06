"""
在所有「最佳」场景下跑线性 vs 非线性，汇总成一张表 + 一张图。

场景：各 version 的 best-SSPG run、best-DI run 的 latent CSV，以及同一 CSV 限制 D1+D2 的子集。
对每个场景用 5-fold 比较 Ridge_linear / Ridge_log_target / Ridge_poly2 / GradientBoosting，输出 Spearman r。

用法（项目根目录）：
  python scripts/run_all_linear_vs_nonlinear.py --results-roots paper1_results_v2,paper1_results_v3,paper1_results_v4 --out paper1_results_diagnostic
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

# Reuse find_best and run_5fold from plot_p1_results and compare_linear_vs_nonlinear
from scripts.plot_p1_results import find_best_sspg_and_di_runs
from scripts.compare_linear_vs_nonlinear import run_5fold_compare, LATENT_COLS


def main():
    parser = argparse.ArgumentParser(description="Run linear vs nonlinear on all best scenarios; output table + figure.")
    parser.add_argument("--results-roots", type=str, default="paper1_results_v2,paper1_results_v3,paper1_results_v4",
                        help="Comma-separated list of result roots (e.g. paper1_results_v2,paper1_results_v4)")
    parser.add_argument("--out", default="paper1_results_diagnostic", help="Output directory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    roots = [x.strip() for x in args.results_roots.split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)

    # Build list of (scenario_label, csv_path, datasets_filter)
    scenarios = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        abs_root = os.path.abspath(root)
        best_sspg_dir, best_di_dir = find_best_sspg_and_di_runs(abs_root)
        base_name = os.path.basename(root)
        if best_sspg_dir and os.path.isfile(os.path.join(best_sspg_dir, "latent_and_gold_all.csv")):
            csv_sspg = os.path.join(best_sspg_dir, "latent_and_gold_all.csv")
            scenarios.append((f"{base_name}_best_SSPG_run", csv_sspg, ""))
            scenarios.append((f"{base_name}_best_SSPG_run_D1D2", csv_sspg, "D1,D2"))
        if best_di_dir and os.path.isfile(os.path.join(best_di_dir, "latent_and_gold_all.csv")):
            csv_di = os.path.join(best_di_dir, "latent_and_gold_all.csv")
            if (f"{base_name}_best_DI_run", csv_di, "") not in [(s[0], s[1], s[2]) for s in scenarios]:
                scenarios.append((f"{base_name}_best_DI_run", csv_di, ""))
            scenarios.append((f"{base_name}_best_DI_run_D1D2", csv_di, "D1,D2"))

    # Deduplicate by (label, csv, filter)
    seen = set()
    uniq_scenarios = []
    for label, csv_path, ds_filter in scenarios:
        key = (label, csv_path, ds_filter)
        if key in seen:
            continue
        seen.add(key)
        uniq_scenarios.append((label, csv_path, ds_filter))

    rows = []
    for label, csv_path, ds_filter in uniq_scenarios:
        if not os.path.isfile(csv_path):
            print(f"Skip {label}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df = df.drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)
        if ds_filter:
            keep = [x.strip() for x in ds_filter.split(",") if x.strip()]
            if keep and "dataset_id" in df.columns:
                df = df[df["dataset_id"].astype(str).isin(keep)].reset_index(drop=True)
        for target in ["sspg", "di"]:
            res = run_5fold_compare(df, target, n_splits=5, seed=args.seed)
            for method, v in res.items():
                rows.append({
                    "scenario": label,
                    "target": target,
                    "method": method,
                    "spearman_r": v["spearman_r"],
                    "rmse": v["rmse"],
                    "n_folds": v["n_folds"],
                })
        print(f"Done scenario: {label} (datasets={ds_filter or 'all'})")

    if not rows:
        print("No results.")
        return
    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out, "all_linear_vs_nonlinear.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # Figure: for each (scenario, target), bar group of methods (Spearman r)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found; skip figure.")
        return

    # Pivot: scenario x (target_method) -> spearman_r. Simplify: one subplot per target (sspg, di), x = scenarios, grouped bars = methods
    for target in ["sspg", "di"]:
        sub = out_df[out_df["target"] == target]
        if sub.empty:
            continue
        scenarios_u = sub["scenario"].unique().tolist()
        methods_u = ["Ridge_linear", "Ridge_log_target", "Ridge_poly2", "GradientBoosting"]
        x = np.arange(len(scenarios_u))
        width = 0.2
        fig, ax = plt.subplots(1, 1, figsize=(max(8, len(scenarios_u) * 1.2), 5))
        for i, method in enumerate(methods_u):
            vals = []
            for sc in scenarios_u:
                row = sub[(sub["scenario"] == sc) & (sub["method"] == method)]
                vals.append(row["spearman_r"].iloc[0] if len(row) else np.nan)
            ax.bar(x + i * width, vals, width, label=method.replace("_", " "))
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(scenarios_u, rotation=30, ha="right")
        ax.set_ylabel("Spearman r (5-fold)")
        ax.set_title(f"Linear vs nonlinear — {target.upper()} (all scenarios)")
        ax.legend(loc="upper right", fontsize=8)
        ax.axhline(0, color="gray", linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"p1_all_scenarios_{target}.png"), dpi=150)
        plt.close()
        print(f"Saved {args.out}/p1_all_scenarios_{target}.png")

    # One heatmap: rows = scenario, cols = target_method, value = spearman_r
    sub = out_df.copy()
    sub["col"] = sub["target"].str.upper() + "_" + sub["method"].str.replace("_", " ")
    scenarios_u = sub["scenario"].unique().tolist()
    cols = sorted(sub["col"].unique().tolist())
    mat = np.nan * np.ones((len(scenarios_u), len(cols)))
    for i, sc in enumerate(scenarios_u):
        for j, c in enumerate(cols):
            r = sub[(sub["scenario"] == sc) & (sub["col"] == c)]
            if len(r):
                mat[i, j] = r["spearman_r"].iloc[0]
    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(cols) * 1.2), max(5, len(scenarios_u) * 0.4)))
    im = ax.imshow(mat, aspect="auto", vmin=-0.2, vmax=0.8, cmap="RdYlGn")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(scenarios_u)))
    ax.set_yticklabels(scenarios_u, fontsize=8)
    for i in range(len(scenarios_u)):
        for j in range(len(cols)):
            v = mat[i, j]
            t = f"{v:.2f}" if np.isfinite(v) else ""
            ax.text(j, i, t, ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="Spearman r")
    ax.set_title("All scenarios: linear vs nonlinear (5-fold Spearman r)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "p1_all_scenarios_heatmap.png"), dpi=150)
    plt.close()
    print(f"Saved {args.out}/p1_all_scenarios_heatmap.png")
    print("Done (all linear vs nonlinear).")


if __name__ == "__main__":
    main()
