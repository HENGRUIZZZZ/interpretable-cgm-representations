"""
实验方案 v5.0 终局之战 — 评估：箱线图 + Wilcoxon 检验 + 结论。

用法（项目根目录）：
  python scripts/run_v5_evaluate.py --scheme_a_dir paper1_results_v5/scheme_a \\
       --scheme_b_dir paper1_results_v5/scheme_b --output_dir paper1_results_v5
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme_a_dir", type=str, default="paper1_results_v5/scheme_a")
    parser.add_argument("--scheme_b_dir", type=str, default="paper1_results_v5/scheme_b")
    parser.add_argument("--output_dir", type=str, default="paper1_results_v5")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载方案 A：所有 run 的 Spearman r（取 DI 或 SSPG 的合并分布，或按 lambda 分组）
    a_path = os.path.join(args.scheme_a_dir, "scheme_a_all_runs.json")
    if not os.path.isfile(a_path):
        print(f"Scheme A results not found: {a_path}")
        a_sspg = np.array([])
        a_di = np.array([])
        a_by_lambda = {}
    else:
        with open(a_path) as f:
            a_runs = json.load(f)
        a_sspg = np.array([r["sspg_spearman_r"] for r in a_runs if r.get("sspg_spearman_r") is not None and np.isfinite(r["sspg_spearman_r"])])
        a_di = np.array([r["di_spearman_r"] for r in a_runs if r.get("di_spearman_r") is not None and np.isfinite(r["di_spearman_r"])])
        a_by_lambda = {}
        for r in a_runs:
            lam = r.get("lambda", r.get("lambda_sspg"))
            if lam not in a_by_lambda:
                a_by_lambda[lam] = {"sspg": [], "di": []}
            if r.get("sspg_spearman_r") is not None and np.isfinite(r["sspg_spearman_r"]):
                a_by_lambda[lam]["sspg"].append(r["sspg_spearman_r"])
            if r.get("di_spearman_r") is not None and np.isfinite(r["di_spearman_r"]):
                a_by_lambda[lam]["di"].append(r["di_spearman_r"])

    # 加载方案 B：100 CV 的 Spearman r
    b_path = os.path.join(args.scheme_b_dir, "scheme_b_spearman_runs.json")
    if not os.path.isfile(b_path):
        print(f"Scheme B results not found: {b_path}")
        b_sspg = np.array([])
        b_di = np.array([])
    else:
        with open(b_path) as f:
            b_runs = json.load(f)
        b_sspg = np.array([r["sspg_spearman_r"] for r in b_runs if np.isfinite(r.get("sspg_spearman_r", np.nan))])
        b_di = np.array([r["di_spearman_r"] for r in b_runs if np.isfinite(r.get("di_spearman_r", np.nan))])

    # Wilcoxon / Mann-Whitney：最佳方案 A（取某 lambda 的 10 次 run）vs 方案 B
    from scipy.stats import mannwhitneyu
    report = []
    report.append("=" * 60)
    report.append("实验方案 v5.0 终局之战 — 评估报告")
    report.append("=" * 60)
    report.append(f"方案 A runs: {len(a_sspg)} SSPG, {len(a_di)} DI")
    report.append(f"方案 B runs: {len(b_sspg)} SSPG, {len(b_di)} DI (100 CV)")
    if len(a_di) > 0:
        report.append(f"方案 A DI  Spearman r: median={np.median(a_di):.4f}  mean={np.mean(a_di):.4f}")
    if len(a_sspg) > 0:
        report.append(f"方案 A SSPG Spearman r: median={np.median(a_sspg):.4f}  mean={np.mean(a_sspg):.4f}")
    if len(b_di) > 0:
        report.append(f"方案 B DI  Spearman r: median={np.median(b_di):.4f}  mean={np.mean(b_di):.4f}")
    if len(b_sspg) > 0:
        report.append(f"方案 B SSPG Spearman r: median={np.median(b_sspg):.4f}  mean={np.mean(b_sspg):.4f}")

    # 最佳 lambda：取 DI 中位数最高的 lambda 的 10 次 run 与 B 比较
    best_lam = None
    best_med = -2
    for lam, v in a_by_lambda.items():
        if v["di"]:
            m = np.median(v["di"])
            if m > best_med:
                best_med = m
                best_lam = lam
    if best_lam is not None and len(b_di) > 0:
        best_a_di = np.array(a_by_lambda[best_lam]["di"])
        stat, p_val = mannwhitneyu(best_a_di, b_di, alternative="two-sided")
        report.append("")
        report.append(f"最佳方案 A: lambda={best_lam} (DI median={best_med:.4f})")
        report.append(f"Mann-Whitney U (方案 A best vs 方案 B, DI): stat={stat:.4f}  p={p_val:.4f}")
        if p_val < 0.05 and np.median(best_a_di) > np.median(b_di):
            report.append("结论: 方案 A（最佳 lambda）在 DI 上显著优于方案 B。")
        elif p_val < 0.05 and np.median(best_a_di) < np.median(b_di):
            report.append("结论: 方案 B 在 DI 上显著优于方案 A。")
        else:
            report.append("结论: 方案 A 与方案 B 在 DI 上无显著差异；在当前数据规模下简约模型与 VAE+ODE 端到端预测相当。")

    # 箱线图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        # SSPG
        data_s = [a_sspg, b_sspg]
        labels_s = ["Scheme A\n(Pred Head)", "Scheme B\n(Ridge Top4)"]
        to_plot_s = [d[d.size > 0] if d.size else np.array([np.nan]) for d in data_s]
        axes[0].boxplot([np.ravel(x) for x in to_plot_s], labels=labels_s)
        axes[0].set_ylabel("Spearman r")
        axes[0].set_title("SSPG")
        axes[0].set_ylim(-0.5, 0.8)
        # DI
        data_d = [a_di, b_di]
        to_plot_d = [d[d.size > 0] if d.size else np.array([np.nan]) for d in data_d]
        axes[1].boxplot([np.ravel(x) for x in to_plot_d], labels=labels_s)
        axes[1].set_ylabel("Spearman r")
        axes[1].set_title("DI")
        axes[1].set_ylim(-0.5, 0.8)
        fig.suptitle("v5 终局之战: Scheme A vs Scheme B")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "v5_boxplot_scheme_a_vs_b.png"), dpi=150)
        plt.close()
        report.append("")
        report.append(f"Saved {args.output_dir}/v5_boxplot_scheme_a_vs_b.png")
    except Exception as e:
        report.append(f"Box plot failed: {e}")

    text = "\n".join(report)
    print(text)
    with open(os.path.join(args.output_dir, "v5_evaluation_report.txt"), "w") as f:
        f.write(text)
    print(f"\nSaved {args.output_dir}/v5_evaluation_report.txt")


if __name__ == "__main__":
    main()
