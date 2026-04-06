"""
V6 路线 C：在 Route B 的 26D latent 上做 PCA(n=2,3,4,5)，每种 n 做 100×5 折 Ridge，记录 Spearman r 与方差解释率。
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
OUT_BASE = os.path.join(REPO_ROOT, "paper1_results_v6")


def latent_cols_26(df: pd.DataFrame):
    """26D: 6 ODE + 4 z_init + 16 z_nonseq."""
    ode = ["tau_m", "Gb", "sg", "si", "p2", "mi"]
    z_init = [c for c in df.columns if c.startswith("z_init_")]
    z_nonseq = [c for c in df.columns if c.startswith("z_nonseq_")]
    return [c for c in ode + z_init + z_nonseq if c in df.columns]


def evaluate_route(X: np.ndarray, y_sspg: np.ndarray, y_di: np.ndarray, n_cv: int = 100, n_splits: int = 5):
    """100×5-fold RidgeCV, return (rs_sspg, rs_di)."""
    rs_sspg, rs_di = [], []
    for seed in range(n_cv):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pred_s = np.full(len(y_sspg), np.nan)
        pred_d = np.full(len(y_di), np.nan)
        ok_s = np.isfinite(y_sspg)
        ok_d = np.isfinite(y_di)
        if ok_s.sum() < 10 or ok_d.sum() < 10:
            rs_sspg.append(np.nan)
            rs_di.append(np.nan)
            continue
        for tr, te in kf.split(X):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            if ok_s[tr].sum() >= 5:
                m_s = RidgeCV(alphas=np.logspace(-2, 2, 20)).fit(Xtr[ok_s[tr]], y_sspg[tr][ok_s[tr]])
                pred_s[te] = m_s.predict(Xte)
            if ok_d[tr].sum() >= 5:
                m_d = RidgeCV(alphas=np.logspace(-2, 2, 20)).fit(Xtr[ok_d[tr]], y_di[tr][ok_d[tr]])
                pred_d[te] = m_d.predict(Xte)
        r_s, _ = stats.spearmanr(pred_s[ok_s], y_sspg[ok_s], nan_policy="omit")
        r_d, _ = stats.spearmanr(pred_d[ok_d], y_di[ok_d], nan_policy="omit")
        rs_sspg.append(r_s)
        rs_di.append(r_d)
    return np.array(rs_sspg), np.array(rs_di)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent_csv", type=str, default=os.path.join(OUT_BASE, "routeB_seed42", "latent_and_gold_all_26d.csv"))
    p.add_argument("--output_dir", type=str, default=os.path.join(OUT_BASE, "routeC"))
    p.add_argument("--n_cv", type=int, default=100)
    p.add_argument("--n_components", type=str, default="2,3,4,5", help="Comma-separated PCA dimensions")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.latent_csv):
        print(f"Route B latent CSV not found: {args.latent_csv}")
        sys.exit(1)
    df = pd.read_csv(args.latent_csv)
    feat_cols = latent_cols_26(df)
    if len(feat_cols) < 5:
        print(f"Need at least 5 latent columns, got {len(feat_cols)}")
        sys.exit(1)

    agg = df.groupby("subject_id")[feat_cols].median().reset_index()
    if "sspg" in df.columns:
        agg = agg.merge(df.groupby("subject_id")["sspg"].first().reset_index(), on="subject_id")
    if "di" in df.columns:
        agg = agg.merge(df.groupby("subject_id")["di"].first().reset_index(), on="subject_id")
    X_full = agg[feat_cols].fillna(agg[feat_cols].median()).values
    y_sspg = agg["sspg"].values if "sspg" in agg.columns else np.full(len(agg), np.nan)
    y_di = agg["di"].values if "di" in agg.columns else np.full(len(agg), np.nan)

    n_components_list = [int(x) for x in args.n_components.split(",")]
    out = {"n_components": [], "var_explained": [], "sspg_spearman_r": [], "di_spearman_r": []}
    for n_comp in n_components_list:
        if n_comp > X_full.shape[1]:
            continue
        pca = PCA(n_components=n_comp, random_state=42)
        sc = StandardScaler()
        X_sc = sc.fit_transform(X_full)
        X_pca = pca.fit_transform(X_sc)
        var_exp = float(pca.explained_variance_ratio_.sum())
        rs_s, rs_d = evaluate_route(X_pca, y_sspg, y_di, n_cv=args.n_cv)
        out["n_components"].append(n_comp)
        out["var_explained"].append(var_exp)
        out["sspg_spearman_r"].append(rs_s.tolist())
        out["di_spearman_r"].append(rs_d.tolist())
        print(f"PCA(n={n_comp}) var_explained={var_exp:.4f}  SSPG r_median={np.nanmedian(rs_s):.4f}  DI r_median={np.nanmedian(rs_d):.4f}")

    with open(os.path.join(args.output_dir, "route_c_pca_spearman.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.output_dir}/route_c_pca_spearman.json")


if __name__ == "__main__":
    main()
