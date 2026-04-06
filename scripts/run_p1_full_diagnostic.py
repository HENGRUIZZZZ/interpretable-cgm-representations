"""
P1 完整诊断流程：VAE 拟合、6D 潜变量图、线性 vs 非线性全场景、报告。

依次执行：
  1. VAE 拟合评估：训练曲线、验证重建 MSE、示例 CGM（需 run_dir 已有 pipeline 产出）
  2. 6D 潜变量图：pairwise、PCA、平行坐标、boxplot
  3. 全场景线性 vs 非线性对比，产出表与图
  4. 生成 DIAGNOSTIC_REPORT.md

用法（项目根目录）：
  python scripts/run_p1_full_diagnostic.py
  python scripts/run_p1_full_diagnostic.py --results-root paper1_results_v4 --out paper1_results_diagnostic
"""
from __future__ import annotations

import os
import sys
import argparse
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _find_best_run_dir(results_root: str) -> str | None:
    from scripts.plot_p1_results import find_best_sspg_and_di_runs
    best_sspg, best_di = find_best_sspg_and_di_runs(os.path.abspath(results_root))
    # Prefer the run that is best by sum of r (or use best_sspg as default)
    return best_sspg or best_di


def main():
    parser = argparse.ArgumentParser(description="Run full P1 diagnostic: VAE fit, 6D plots, linear vs nonlinear, report.")
    parser.add_argument("--results-root", default="paper1_results_v4", help="Result root (e.g. paper1_results_v4)")
    parser.add_argument("--results-roots", default="paper1_results_v2,paper1_results_v3,paper1_results_v4",
                        help="Comma-separated roots for linear vs nonlinear (all scenarios)")
    parser.add_argument("--out", default="paper1_results_diagnostic", help="Output directory for diagnostic figures and report")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    results_root = os.path.abspath(args.results_root)

    # 1) Best run dir for this root (for VAE assessment and 6D from same run)
    best_run = _find_best_run_dir(results_root)
    if not best_run or not os.path.isdir(best_run):
        # 回退：若 results_root 自身含 latent CSV（如解耦实验 m1_decouple），则直接用作 run 目录
        if os.path.isfile(os.path.join(results_root, "latent_and_gold_all.csv")):
            best_run = results_root
            print(f"Using results-root as run dir (flat layout): {best_run}")
        else:
            print(f"No best run found under {results_root}. Skip VAE and 6D (will still run linear vs nonlinear if other roots exist).")
            best_run = None
    else:
        print(f"Using run for VAE/6D: {best_run}")

    # 2) VAE fit assessment
    if best_run:
        csv_in_run = os.path.join(best_run, "latent_and_gold_all.csv")
        if os.path.isfile(csv_in_run):
            r = subprocess.run(
                [sys.executable, "scripts/assess_vae_fit.py", "--run-dir", best_run, "--out", fig_dir],
                cwd=REPO_ROOT,
                timeout=60,
            )
            if r.returncode != 0:
                print("assess_vae_fit.py had non-zero exit")
        else:
            print("No latent_and_gold_all.csv in run dir; VAE assessment may lack reconstruction files (run pipeline first).")

    # 3) 6D latent figures
    if best_run:
        csv_6d = os.path.join(best_run, "latent_and_gold_all.csv")
        if os.path.isfile(csv_6d):
            r = subprocess.run(
                [sys.executable, "scripts/plot_p1_6d_latent.py", "--csv", csv_6d, "--out", fig_dir],
                cwd=REPO_ROOT,
                timeout=120,
            )
            if r.returncode != 0:
                print("plot_p1_6d_latent.py had non-zero exit")

    # 4) All scenarios linear vs nonlinear
    r = subprocess.run(
        [sys.executable, "scripts/run_all_linear_vs_nonlinear.py",
         "--results-roots", args.results_roots, "--out", out_dir],
        cwd=REPO_ROOT,
        timeout=300,
    )
    if r.returncode != 0:
        print("run_all_linear_vs_nonlinear.py had non-zero exit")
    # Move or copy heatmap/bar figures to fig_dir for consistency
    for f in ["p1_all_scenarios_sspg.png", "p1_all_scenarios_di.png", "p1_all_scenarios_heatmap.png"]:
        src = os.path.join(out_dir, f)
        if os.path.isfile(src):
            dest = os.path.join(fig_dir, f)
            if src != dest:
                import shutil
                shutil.copy2(src, dest)

    # 5) Write report
    report_path = os.path.join(out_dir, "DIAGNOSTIC_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# P1 Diagnostic Report\n\n")
        f.write("## 1. Purpose\n\n")
        f.write("Check (1) VAE fit quality, (2) 6D latent structure, (3) regression method and data scope, ")
        f.write("to locate where prediction performance might be limited.\n\n")
        f.write("## 2. VAE fit (model must fit first)\n\n")
        if best_run:
            curves = os.path.join(best_run, "training_curves.json")
            mse_npy = os.path.join(best_run, "reconstruction_val_mse.npy")
            if os.path.isfile(curves):
                f.write("- **Training curve**: see `figures/p1_vae_training_curve.png`. ")
                f.write("Train/val loss should decrease; val loss (reconstruction MSE) indicates fit.\n")
            else:
                f.write("- Training curve not available (re-run pipeline to generate `training_curves.json`).\n")
            if os.path.isfile(mse_npy):
                import numpy as np
                arr = np.load(mse_npy)
                f.write(f"- **Validation reconstruction MSE**: mean = {arr.mean():.4f}, std = {arr.std():.4f}. ")
                f.write("See `figures/p1_vae_reconstruction_mse_hist.png`.\n")
            if os.path.isfile(os.path.join(best_run, "reconstruction_examples.npz")):
                f.write("- **Example CGM**: `figures/p1_vae_reconstruction_examples.png` (actual vs reconstructed).\n")
            f.write("\nTo generate VAE fit figures (training curve, reconstruction MSE, examples), re-run the pipeline once so the run dir gets `training_curves.json`, `reconstruction_val_mse.npy`, `reconstruction_examples.npz`, then re-run this diagnostic.\n\n")
        else:
            f.write("- No run dir used; skip VAE section.\n\n")
        f.write("## 3. 6D latent space\n\n")
        f.write("We use 6 interpretable dimensions: tau_m, Gb, sg, si, p2, mi. Figures:\n")
        f.write("- `figures/p1_6d_pairwise_by_dataset.png`: pairwise scatter (lower triangle), colored by dataset.\n")
        f.write("- `figures/p1_6d_pca2d.png`: PCA 2D by dataset and by SSPG/DI.\n")
        f.write("- `figures/p1_6d_parallel_by_dataset.png`: parallel coordinates.\n")
        f.write("- `figures/p1_6d_boxplot_by_dataset.png`: per-dimension distribution by dataset.\n\n")
        f.write("## 4. Linear vs nonlinear (all scenarios)\n\n")
        f.write("Table: `all_linear_vs_nonlinear.csv`. Figures: `figures/p1_all_scenarios_sspg.png`, ")
        f.write("`figures/p1_all_scenarios_di.png`, `figures/p1_all_scenarios_heatmap.png`.\n\n")
        f.write("## 5. Where might the problem be?\n\n")
        f.write("- **If VAE reconstruction is poor**: model or data preprocessing issue; fix fit before interpreting latent.\n")
        f.write("- **If 6D structure is messy (e.g. no separation by dataset/gold)**: representation may not align with physiology.\n")
        f.write("- **If linear is consistently worse than poly2/GB in heatmap**: consider nonlinear heads or report both.\n")
        f.write("- **If D1+D2-only is much better than D1+D2+D4**: D4 scope or scaling may need separate handling.\n")

    print(f"Wrote {report_path}")
    print("Done (full diagnostic).")


if __name__ == "__main__":
    main()
