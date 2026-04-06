"""
线性 vs 非线性：6D latent → SSPG/DI 的预测方式对比。

我们并**不知道**潜变量与金标准的关系是线性的，只是默认用了 Ridge（线性）。
本脚本在相同 5-fold 划分下对比：
  - Ridge（线性）
  - Ridge 但对 log(SSPG+1) / log(DI+ε) 回归再反变换（常见非线性变换）
  - 二次项 + Ridge（PolynomialFeatures degree=2）
  - 梯度提升树（轻度非线性，max_depth=2）

用法（项目根目录）：
  python scripts/compare_linear_vs_nonlinear.py --csv paper1_results_v4/latent_and_gold_all.csv --out paper1_results_v4
  python scripts/compare_linear_vs_nonlinear.py --csv paper1_results_v4/latent_and_gold_all.csv --datasets D1,D2
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

LATENT_COLS = ["si", "mi", "tau_m", "Gb", "sg", "p2"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    if n < 2:
        return {"rmse": np.nan, "spearman_r": np.nan, "n": n}
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r, _ = stats.spearmanr(y_true, y_pred)
    return {"rmse": float(rmse), "spearman_r": float(r), "n": n}


def run_5fold_compare(df: pd.DataFrame, target: str, n_splits: int = 5, seed: int = 0) -> dict:
    """返回各方法的 5-fold 平均 spearman_r 和 rmse。"""
    sub = df.dropna(subset=[target]).copy()
    if len(sub) < 10:
        return {}
    X = sub[LATENT_COLS].values
    y = sub[target].values
    grp = sub["subject_id"].astype(str).values
    kf = GroupKFold(n_splits=n_splits)
    splits = list(kf.split(X, y, grp))

    methods = {}
    for tr_idx, te_idx in splits:
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        if len(y_te) < 2:
            continue
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # 1) Linear Ridge
        reg = Ridge(alpha=10.0 if target == "sspg" else 0.1)
        reg.fit(X_tr_s, y_tr)
        pred = reg.predict(X_te_s)
        m = compute_metrics(y_te, pred)
        methods.setdefault("Ridge_linear", []).append(m)

        # 2) Log-transform target (SSPG: log(1+x), DI: log(0.1+x))
        if target == "sspg":
            y_tr_t = np.log1p(y_tr)
            reg.fit(X_tr_s, y_tr_t)
            pred = np.expm1(reg.predict(X_te_s))
        else:
            eps = 0.1
            y_tr_t = np.log(y_tr + eps)
            reg.fit(X_tr_s, y_tr_t)
            pred = np.exp(reg.predict(X_te_s)) - eps
        m = compute_metrics(y_te, pred)
        methods.setdefault("Ridge_log_target", []).append(m)

        # 3) Polynomial (degree=2) + Ridge
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_tr_p = poly.fit_transform(X_tr_s)
        X_te_p = poly.transform(X_te_s)
        reg2 = Ridge(alpha=1.0)
        reg2.fit(X_tr_p, y_tr)
        pred = reg2.predict(X_te_p)
        m = compute_metrics(y_te, pred)
        methods.setdefault("Ridge_poly2", []).append(m)

        # 4) GradientBoosting (mild nonlinear)
        gb = GradientBoostingRegressor(max_depth=2, n_estimators=50, random_state=seed)
        gb.fit(X_tr_s, y_tr)
        pred = gb.predict(X_te_s)
        m = compute_metrics(y_te, pred)
        methods.setdefault("GradientBoosting", []).append(m)

    out = {}
    for name, list_m in methods.items():
        if not list_m:
            continue
        out[name] = {
            "spearman_r": float(np.nanmean([x["spearman_r"] for x in list_m])),
            "rmse": float(np.nanmean([x["rmse"] for x in list_m])),
            "n_folds": len(list_m),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare linear vs nonlinear prediction of SSPG/DI from 6D latent.")
    parser.add_argument("--csv", default="paper1_results_v4/latent_and_gold_all.csv", help="latent_and_gold CSV")
    parser.add_argument("--out", default=None, help="Write summary to this dir (optional)")
    parser.add_argument("--datasets", type=str, default="", help="e.g. D1,D2 to restrict subjects")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"File not found: {args.csv}")
        return
    df = pd.read_csv(args.csv)
    df = df.drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)
    if args.datasets:
        keep = [x.strip() for x in args.datasets.split(",") if x.strip()]
        if keep and "dataset_id" in df.columns:
            df = df[df["dataset_id"].astype(str).isin(keep)].reset_index(drop=True)
            print(f"Restricted to datasets {keep}: n={len(df)}")

    print("\n=== Linear vs nonlinear: 5-fold (by subject) ===\n")
    for target in ["sspg", "di"]:
        res = run_5fold_compare(df, target, n_splits=5, seed=args.seed)
        if not res:
            print(f"{target}: no enough data")
            continue
        print(f"--- {target.upper()} ---")
        for name in ["Ridge_linear", "Ridge_log_target", "Ridge_poly2", "GradientBoosting"]:
            if name not in res:
                continue
            r = res[name]["spearman_r"]
            rmse = res[name]["rmse"]
            print(f"  {name}:  Spearman r={r:.3f}  RMSE={rmse:.3f}")
        print()

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        rows = []
        for target in ["sspg", "di"]:
            res = run_5fold_compare(df, target, n_splits=5, seed=args.seed)
            for name, v in res.items():
                rows.append({"target": target, "method": name, "spearman_r": v["spearman_r"], "rmse": v["rmse"]})
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "linear_vs_nonlinear.csv"), index=False)
        print(f"Saved {args.out}/linear_vs_nonlinear.csv")


if __name__ == "__main__":
    main()
