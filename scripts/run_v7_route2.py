"""
V7 Route 2: 两阶段预测 - 强 IR 信号（SSPG_hat）辅助 DI 预测。
阶段一：使用 Route 1A 的 lambda_sspg_best 得到 26D + sspg_hat。
阶段二：X = [z_latents (26D), sspg_hat]，y = di_true，LODO Ridge，报告 Pearson r(DI)。
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
OUT_V7 = os.path.join(REPO_ROOT, "paper1_results_v7")

LATENT_26D = (
    ["tau_m", "Gb", "sg", "si", "p2", "mi"]
    + [f"z_init_{j}" for j in range(4)]
    + [f"z_nonseq_{j}" for j in range(16)]
)


def _ensure_dataset_id(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset_id" in df.columns and df["dataset_id"].notna().any():
        return df
    from run_p1_full_pipeline import _stack_batches
    from load_cgm_project_data import load_cgm_project_level1_level2, load_cgm_project_level3
    from paper1_experiment_config import get_data_dir, P1_FULL_TRAIN_DATASETS
    OUTPUT_BASE = os.environ.get("CGM_PROJECT_OUTPUT", "/Users/hertz1030/Downloads/cgm_project/output")
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
    if not labels_combined.empty and "dataset_id" in labels_combined.columns:
        did_map = labels_combined.set_index("subject_id")["dataset_id"].to_dict()
        df = df.copy()
        df["dataset_id"] = df["subject_id"].map(lambda x: did_map.get(str(x), ""))
    return df


def lodo_ridge_27d(df: pd.DataFrame, alphas=np.logspace(-3, 3, 50)) -> tuple[list[dict], float]:
    """X = [26D, sspg_hat], y = di. LODO Ridge, return folds and mean Pearson r."""
    feat = [c for c in LATENT_26D if c in df.columns]
    if "sspg_hat" not in df.columns or "di" not in df.columns or "dataset_id" not in df.columns:
        return [], np.nan
    feat = feat + ["sspg_hat"]
    sub = df.dropna(subset=["di"] + feat).copy()
    datasets = sub["dataset_id"].dropna().unique().tolist()
    if len(datasets) < 2:
        return [], np.nan
    out = []
    for test_ds in datasets:
        train_df = sub[sub["dataset_id"] != test_ds]
        test_df = sub[sub["dataset_id"] == test_ds]
        if len(train_df) < 3 or len(test_df) < 2:
            continue
        X_tr = train_df[feat].values
        y_tr = train_df["di"].values
        X_te = test_df[feat].values
        y_te = test_df["di"].values
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        reg = RidgeCV(alphas=alphas).fit(X_tr_s, y_tr)
        y_pred = reg.predict(X_te_s)
        r, p = stats.pearsonr(y_te, y_pred)
        out.append({"test_dataset": test_ds, "pearson_r": float(r), "n": int(len(y_te))})
    if not out:
        return [], np.nan
    mean_r = float(np.nanmean([x["pearson_r"] for x in out]))
    return out, mean_r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best_1a_dir", type=str, default=None,
                   help="Directory of Route 1A best run (default: route1/1a_lambda_sspg_0.1)")
    args = p.parse_args()
    best_dir = args.best_1a_dir or os.path.join(OUT_V7, "route1", "1a_lambda_sspg_0.1")
    csv_path = os.path.join(best_dir, "latent_and_gold_all_26d.csv")
    if not os.path.isfile(csv_path):
        print(f"CSV not found: {csv_path}. Run Route 1A best (lambda_sspg=0.1) first; pipeline must save sspg_hat in 26D CSV.")
        return
    df = pd.read_csv(csv_path)
    df = _ensure_dataset_id(df)
    if "sspg_hat" not in df.columns:
        print("sspg_hat not in CSV. Re-run pipeline for best 1A (P1_SAVE_26D_LATENT=1 with sspg_hat column support).")
        return
    folds, mean_r = lodo_ridge_27d(df)
    print(f"V7 Route 2 (X=[26D, sspg_hat] -> DI)  LODO-CV mean Pearson r = {mean_r:.4f}")
    for f in folds:
        print(f"  {f['test_dataset']}: r={f['pearson_r']:.4f}  n={f['n']}")
    os.makedirs(os.path.join(OUT_V7, "route2"), exist_ok=True)
    with open(os.path.join(OUT_V7, "route2", "route2_lodo_pearson.json"), "w") as f:
        json.dump({"mean_pearson_r_di": mean_r, "folds": folds}, f, indent=2)
    print(f"Saved paper1_results_v7/route2/route2_lodo_pearson.json")


if __name__ == "__main__":
    main()
