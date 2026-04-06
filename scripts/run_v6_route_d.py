"""
V6 路线 D：CGM 统计特征（cgm_mean + ac_var）作为黄金标准基线。

ac_var = std(acf(cgm_window, nlags=10)[1:])（Sugimoto 2025）
100 次 5 折 CV，RidgeCV，记录 SSPG/DI 的 Spearman r。
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
OUTPUT_BASE = os.environ.get("CGM_PROJECT_OUTPUT", "/Users/hertz1030/Downloads/cgm_project/output")


def ac_var_sugimoto(y: np.ndarray, nlags: int = 10) -> float:
    """ac_var = std(acf(y, nlags=nlags)[1:])"""
    y = np.asarray(y).ravel()
    y = y[np.isfinite(y)]
    if len(y) < nlags + 5:
        return np.nan
    try:
        from statsmodels.tsa.stattools import acf
        ac = acf(y, nlags=nlags, fft=True)
        if ac is not None and len(ac) > 1:
            return float(np.std(ac[1:]))
    except Exception:
        pass
    # fallback: 相邻差分标准差
    d = np.diff(y)
    return float(np.std(d)) if len(d) > 0 else np.nan


def compute_cgm_mean_acvar(cgm: np.ndarray) -> tuple:
    """(cgm_mean, ac_var) per window."""
    y = np.asarray(cgm).ravel()
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return np.nan, np.nan
    return float(np.mean(y)), ac_var_sugimoto(y)


def evaluate_route(X: np.ndarray, y: np.ndarray, n_cv: int = 100, n_splits: int = 5):
    """100 次 5 折 CV，RidgeCV，返回每次的 Spearman r。"""
    rs = []
    valid = np.isfinite(y)
    Xv = X[valid]
    yv = y[valid]
    if len(yv) < 10:
        return np.array([np.nan] * n_cv)
    for seed in range(n_cv):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        preds = np.full_like(yv, np.nan)
        for tr, te in kf.split(Xv):
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xv[tr])
            Xte = sc.transform(Xv[te])
            m = RidgeCV(alphas=np.logspace(-2, 2, 20)).fit(Xtr, yv[tr])
            preds[te] = m.predict(Xte)
        r, _ = stats.spearmanr(preds, yv, nan_policy="omit")
        rs.append(r)
    return np.array(rs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default=os.path.join(REPO_ROOT, "paper1_results_v6", "routeD"))
    p.add_argument("--n_cv", type=int, default=100)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from scripts.compute_cgm_stats import load_and_stack_same_as_pipeline
    os.environ.setdefault("P1_ONE_MEAL_PER_SUBJECT", "1")
    batch, pids = load_and_stack_same_as_pipeline(OUTPUT_BASE)

    rows = []
    for i in range(batch.cgm.shape[0]):
        mu, acv = compute_cgm_mean_acvar(batch.cgm[i])
        rows.append({"subject_id": pids[i], "cgm_mean": mu, "ac_var": acv})
    df = pd.DataFrame(rows)
    agg = df.groupby("subject_id")[["cgm_mean", "ac_var"]].median().reset_index()

    from run_p1_full_pipeline import _stack_batches, GOLD_COLS
    from load_cgm_project_data import load_cgm_project_level1_level2, load_cgm_project_level3
    from paper1_experiment_config import get_data_dir, P1_FULL_TRAIN_DATASETS
    batch_list, info_list, labels_list, dataset_ids = [], [], [], []
    for did in P1_FULL_TRAIN_DATASETS:
        data_dir = get_data_dir(did, OUTPUT_BASE)
        if not os.path.isdir(data_dir):
            continue
        if did in ("D1", "D2"):
            b, info, lab = load_cgm_project_level1_level2(data_dir=data_dir, num_meals_threshold=1)
        else:
            try:
                b, info, lab = load_cgm_project_level3(dataset_id=did, output_base=OUTPUT_BASE)
            except Exception:
                continue
        batch_list.append(b)
        info_list.append(info)
        labels_list.append(lab)
        dataset_ids.append(did)
    _, _, labels_combined = _stack_batches(batch_list, info_list, labels_list, dataset_ids)
    gold = labels_combined.drop_duplicates(subset=["subject_id"], keep="first")
    gold = gold[["subject_id"] + [c for c in GOLD_COLS if c in labels_combined.columns]]
    sub = agg.merge(gold, on="subject_id", how="inner")
    if "sspg" not in sub.columns:
        sub["sspg"] = np.nan
    if "di" not in sub.columns:
        sub["di"] = np.nan
    sub = sub.dropna(subset=["cgm_mean", "ac_var"], how="all")
    X = sub[["cgm_mean", "ac_var"]].fillna(sub[["cgm_mean", "ac_var"]].median()).values
    y_sspg = sub["sspg"].values
    y_di = sub["di"].values

    rs_sspg = evaluate_route(X, y_sspg, n_cv=args.n_cv)
    rs_di = evaluate_route(X, y_di, n_cv=args.n_cv)
    out = {
        "sspg_spearman_r": rs_sspg.tolist(),
        "di_spearman_r": rs_di.tolist(),
        "n_subjects": len(sub),
        "n_cv": args.n_cv,
    }
    with open(os.path.join(args.output_dir, "route_d_spearman.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Route D: SSPG r median={np.nanmedian(rs_sspg):.4f}, DI r median={np.nanmedian(rs_di):.4f}, n_cv={args.n_cv}")
    print(f"Saved {args.output_dir}/route_d_spearman.json")


if __name__ == "__main__":
    main()
