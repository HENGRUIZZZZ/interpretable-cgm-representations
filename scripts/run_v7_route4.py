"""
V7 Route 4: 临床可解释性 - 四象限分层图（SSPG_hat vs DI_hat，以真实值中位数分割），
报告象限分类准确率；可选按 dataset 着色（种族/数据来源代理）。
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
OUT_V7 = os.path.join(REPO_ROOT, "paper1_results_v7")


def quadrant(sspg: float, di: float, sspg_med: float, di_med: float) -> int:
    """1=低IR高Beta, 2=高IR高Beta, 3=高IR低Beta, 4=低IR低Beta (临床常用：左上1, 右上2, 右下3, 左下4)."""
    if np.isnan(sspg) or np.isnan(di):
        return -1
    q = 0
    if sspg >= sspg_med:
        q += 2
    if di >= di_med:
        q += 1
    # q: 0=低SSPG低DI(4), 1=低SSPG高DI(1), 2=高SSPG低DI(3), 3=高SSPG高DI(2)
    map_q = {0: 4, 1: 1, 2: 3, 3: 2}
    return map_q.get(q, -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=None,
                    help="26D CSV with sspg_hat, di_hat, sspg, di (default: route3/combo_a or route1/1b_lambda_di_0.05)")
    args = p.parse_args()
    if args.csv and os.path.isfile(args.csv):
        csv_path = args.csv
    else:
        for cand in [os.path.join(OUT_V7, "route3", "combo_a", "latent_and_gold_all_26d.csv"),
                     os.path.join(OUT_V7, "route1", "1b_lambda_di_0.05", "latent_and_gold_all_26d.csv")]:
            if os.path.isfile(cand):
                csv_path = cand
                break
        else:
            print("No best model CSV found. Run Route 3 or Route 1B first.")
            return
    df = pd.read_csv(csv_path)
    need = ["sspg_hat", "di_hat", "sspg", "di"]
    if not all(c in df.columns for c in need):
        print(f"CSV missing columns: need {need}")
        return
    sub = df.dropna(subset=need)
    if len(sub) < 5:
        print("Too few samples with full gold and predictions")
        return
    sspg_med = sub["sspg"].median()
    di_med = sub["di"].median()
    sub = sub.copy()
    sub["quadrant_true"] = sub.apply(lambda r: quadrant(r["sspg"], r["di"], sspg_med, di_med), axis=1)
    sub["quadrant_pred"] = sub.apply(lambda r: quadrant(r["sspg_hat"], r["di_hat"], sspg_med, di_med), axis=1)
    ok = sub["quadrant_true"] >= 1
    acc = (sub.loc[ok, "quadrant_true"] == sub.loc[ok, "quadrant_pred"]).mean()
    n_ok = ok.sum()

    os.makedirs(os.path.join(OUT_V7, "route4"), exist_ok=True)
    report = [
        "V7 Route 4: IR-Beta 四象限分层",
        "=" * 50,
        f"数据: {csv_path}",
        f"分割线: SSPG_median = {sspg_med:.2f}, DI_median = {di_med:.4f}",
        f"象限分类准确率 (预测 vs 真实): {acc:.4f}  (n={n_ok})",
        "",
        "说明: 象限 1=低IR高Beta, 2=高IR高Beta, 3=高IR低Beta, 4=低IR低Beta。",
        "动态追踪: 若有纵向数据，可在此图上绘制个体轨迹，量化干预效果。",
    ]
    text = "\n".join(report)
    print(text)
    report_path = os.path.join(OUT_V7, "route4", "route4_quadrant_report.txt")
    with open(report_path, "w") as f:
        f.write(text)
    print(f"Saved {report_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axvline(sspg_med, color="grey", linestyle="--", label=f"SSPG median={sspg_med:.0f}")
        ax.axhline(di_med, color="grey", linestyle="--", label=f"DI median={di_med:.2f}")
        hue_col = "dataset_id" if "dataset_id" in sub.columns and sub["dataset_id"].notna().any() else None
        if hue_col:
            for ds in sub[hue_col].dropna().unique():
                m = sub[hue_col] == ds
                ax.scatter(sub.loc[m, "sspg_hat"], sub.loc[m, "di_hat"], label=str(ds), alpha=0.7)
        else:
            ax.scatter(sub["sspg_hat"], sub["di_hat"], alpha=0.7, label="predicted")
        ax.set_xlabel("Predicted Insulin Resistance (SSPG_hat)")
        ax.set_ylabel("Predicted Beta-Cell Function (DI_hat)")
        ax.set_title("V7 IR-Beta Quadrant Mapping from CGM Data")
        ax.legend()
        fig.savefig(os.path.join(OUT_V7, "route4", "v7_quadrant_plot.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved paper1_results_v7/route4/v7_quadrant_plot.png")
    except Exception as e:
        print(f"Plot skip: {e}")


if __name__ == "__main__":
    main()
