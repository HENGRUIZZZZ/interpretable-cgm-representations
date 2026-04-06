"""
V6 路线 F：26D latent (Route B) + 12D CGM 统计 → 38D，100×5 折 Ridge，记录 Spearman r。
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
OUT_BASE = os.path.join(REPO_ROOT, "paper1_results_v6")
OUTPUT_BASE = os.environ.get("CGM_PROJECT_OUTPUT", "/Users/hertz1030/Downloads/cgm_project/output")
CGM_STAT_COLS = [
    "cgm_mean", "cgm_std", "cgm_cv", "cgm_min", "cgm_max", "cgm_range",
    "tir", "tar", "tbr", "auc", "ac_var", "mge",
]


def latent_cols_26(df: pd.DataFrame):
    ode = ["tau_m", "Gb", "sg", "si", "p2", "mi"]
    z_init = [c for c in df.columns if c.startswith("z_init_")]
    z_nonseq = [c for c in df.columns if c.startswith("z_nonseq_")]
    return [c for c in ode + z_init + z_nonseq if c in df.columns]


def evaluate_route(X: np.ndarray, y_sspg: np.ndarray, y_di: np.ndarray, n_cv: int = 100, n_splits: int = 5):
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
    p.add_argument("--output_dir", type=str, default=os.path.join(OUT_BASE, "routeF"))
    p.add_argument("--n_cv", type=int, default=100)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.latent_csv):
        print(f"Route B latent CSV not found: {args.latent_csv}")
        sys.exit(1)
    latent_df = pd.read_csv(args.latent_csv)
    lat_cols = latent_cols_26(latent_df)
    agg_lat = latent_df.groupby("subject_id")[lat_cols].median().reset_index()
    if "sspg" in latent_df.columns:
        agg_lat = agg_lat.merge(latent_df.groupby("subject_id")["sspg"].first().reset_index(), on="subject_id")
    if "di" in latent_df.columns:
        agg_lat = agg_lat.merge(latent_df.groupby("subject_id")["di"].first().reset_index(), on="subject_id")

    from scripts.compute_cgm_stats import load_and_stack_same_as_pipeline, compute_one_window
    os.environ.setdefault("P1_ONE_MEAL_PER_SUBJECT", "1")
    batch, pids = load_and_stack_same_as_pipeline(OUTPUT_BASE)
    rows = []
    for i in range(batch.cgm.shape[0]):
        st = compute_one_window(batch.cgm[i])
        st["subject_id"] = pids[i]
        rows.append(st)
    cgm_df = pd.DataFrame(rows)
    cgm_avail = [c for c in CGM_STAT_COLS if c in cgm_df.columns]
    agg_cgm = cgm_df.groupby("subject_id")[cgm_avail].median().reset_index()

    merge = agg_lat.merge(agg_cgm, on="subject_id", how="inner")
    feat_cols = lat_cols + cgm_avail
    X = merge[feat_cols].fillna(merge[feat_cols].median()).values
    y_sspg = merge["sspg"].values if "sspg" in merge.columns else np.full(len(merge), np.nan)
    y_di = merge["di"].values if "di" in merge.columns else np.full(len(merge), np.nan)
    print(f"Route F: 26D + {len(cgm_avail)}D CGM = {X.shape[1]}D, n_subjects={len(merge)}")

    rs_s, rs_d = evaluate_route(X, y_sspg, y_di, n_cv=args.n_cv)
    out = {
        "n_features": int(X.shape[1]),
        "sspg_spearman_r": rs_s.tolist(),
        "di_spearman_r": rs_d.tolist(),
        "sspg_median": float(np.nanmedian(rs_s)),
        "di_median": float(np.nanmedian(rs_d)),
    }
    with open(os.path.join(args.output_dir, "route_f_spearman.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"SSPG r median={out['sspg_median']:.4f}  DI r median={out['di_median']:.4f}")
    print(f"Saved {args.output_dir}/route_f_spearman.json")


if __name__ == "__main__":
    main()
