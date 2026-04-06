"""
V7 LODO 评估：对 26D latent CSV 做留一数据集出 (LODO)，报告 Pearson r（V7 统一使用 Pearson）。
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

LATENT_26D_COLS = (
    ["tau_m", "Gb", "sg", "si", "p2", "mi"]
    + [f"z_init_{j}" for j in range(4)]
    + [f"z_nonseq_{j}" for j in range(16)]
)


def lodo_pearson(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str] | None = None,
    alphas: np.ndarray | None = None,
) -> tuple[list[dict], float]:
    """LODO: 留一数据集出，Ridge 预测，返回每 fold 的 metrics 和 mean Pearson r。"""
    if feature_cols is None:
        feature_cols = [c for c in LATENT_26D_COLS if c in df.columns]
    if not feature_cols or target not in df.columns or "dataset_id" not in df.columns:
        return [], np.nan
    if alphas is None:
        alphas = np.logspace(-3, 3, 50)
    sub = df.dropna(subset=[target] + feature_cols).copy()
    datasets = sub["dataset_id"].dropna().unique().tolist()
    if len(datasets) < 2:
        return [], np.nan
    out = []
    for test_ds in datasets:
        train_df = sub[sub["dataset_id"] != test_ds]
        test_df = sub[sub["dataset_id"] == test_ds]
        if len(train_df) < 3 or len(test_df) < 2:
            continue
        X_tr = train_df[feature_cols].values
        y_tr = train_df[target].values
        X_te = test_df[feature_cols].values
        y_te = test_df[target].values
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        reg = RidgeCV(alphas=alphas).fit(X_tr_s, y_tr)
        y_pred = reg.predict(X_te_s)
        r, p = stats.pearsonr(y_te, y_pred)
        out.append({"test_dataset": test_ds, "pearson_r": float(r), "pearson_p": float(p), "n": int(len(y_te))})
    if not out:
        return [], np.nan
    mean_r = float(np.nanmean([x["pearson_r"] for x in out]))
    return out, mean_r


def main():
    p = argparse.ArgumentParser(description="V7 LODO Pearson r on 26D latent CSV")
    p.add_argument("--csv", type=str, required=True, help="Path to latent_and_gold_all_26d.csv")
    p.add_argument("--target", type=str, default="sspg", choices=["sspg", "di"])
    p.add_argument("--out", type=str, default=None, help="Write JSON here (optional)")
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    if "dataset_id" not in df.columns and "subject_id" in df.columns:
        from run_p1_full_pipeline import _stack_batches, GOLD_COLS
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
            df["dataset_id"] = df["subject_id"].map(lambda x: did_map.get(str(x), ""))
    folds, mean_r = lodo_pearson(df, args.target)
    print(f"Target={args.target}  LODO-CV mean Pearson r = {mean_r:.4f}  folds={[f['test_dataset'] for f in folds]}")
    for f in folds:
        print(f"  {f['test_dataset']}: r={f['pearson_r']:.4f}  n={f['n']}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fp:
            json.dump({"target": args.target, "mean_pearson_r": mean_r, "folds": folds}, fp, indent=2)
    return mean_r


if __name__ == "__main__":
    main()
