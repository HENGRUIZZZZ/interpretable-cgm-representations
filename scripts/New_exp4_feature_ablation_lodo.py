"""
New_ Exp4: feature ablation with LODO-CV on D1 + D2.

Tiers:
- Tier1: CGM statistics
- Tier2: 6D ODE params
- Tier3: 26D latent
- Tier4: 26D latent + demographics (age, bmi, sex)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _pearson(y: np.ndarray, yhat: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat)
    if ok.sum() < 3:
        return float("nan")
    return float(stats.pearsonr(y[ok], yhat[ok])[0])


def _lodo_predict(df: pd.DataFrame, feat_cols: List[str], target: str) -> float:
    all_pred = []
    all_true = []
    for test_ds in ["D1", "D2"]:
        tr = df[df["dataset_id"] != test_ds].dropna(subset=feat_cols + [target]).copy()
        te = df[df["dataset_id"] == test_ds].dropna(subset=feat_cols + [target]).copy()
        if len(tr) < 10 or len(te) < 5:
            continue
        Xtr = tr[feat_cols].to_numpy(dtype=float)
        ytr = tr[target].to_numpy(dtype=float)
        Xte = te[feat_cols].to_numpy(dtype=float)
        yte = te[target].to_numpy(dtype=float)
        reg = Pipeline(
            [("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))]
        )
        reg.fit(Xtr, ytr)
        yhat = reg.predict(Xte)
        all_pred.append(yhat)
        all_true.append(yte)
    if not all_true:
        return float("nan")
    return _pearson(np.concatenate(all_true), np.concatenate(all_pred))


def _load_tier1(data_root: str) -> pd.DataFrame:
    d1 = pd.read_csv(os.path.join(data_root, "D1_metwally", "cgm.csv"))
    d2 = pd.read_csv(os.path.join(data_root, "D2_stanford", "cgm.csv"))
    d1 = d1.rename(columns={"glucose_mg_dl": "glucose", "timepoint_mins": "t"})
    d2 = d2.rename(columns={"glucose_mg_dl": "glucose", "minutes_after_meal": "t"})
    rows = []
    for sid, g in d1.groupby("subject_id"):
        gg = g["glucose"].astype(float).values
        rows.append(
            {
                "subject_id": sid,
                "dataset_id": "D1",
                "cgm_mean": float(np.mean(gg)),
                "cgm_var": float(np.var(gg)),
                "tir_70_140": float(np.mean((gg >= 70) & (gg <= 140))),
            }
        )
    for sid, g in d2.groupby("subject_id"):
        vals = []
        for _, gg in g.groupby(["meal_type", "rep"]):
            x = gg["glucose"].astype(float).values
            vals.append([np.mean(x), np.var(x), np.mean((x >= 70) & (x <= 140))])
        if vals:
            vals = np.asarray(vals, dtype=float)
            rows.append(
                {
                    "subject_id": sid,
                    "dataset_id": "D2",
                    "cgm_mean": float(np.mean(vals[:, 0])),
                    "cgm_var": float(np.mean(vals[:, 1])),
                    "tir_70_140": float(np.mean(vals[:, 2])),
                }
            )
    feat = pd.DataFrame(rows)
    l1 = pd.read_csv(os.path.join(data_root, "D1_metwally", "labels.csv"))
    if "SSPG" in l1.columns and "sspg" not in l1.columns:
        l1["sspg"] = l1["SSPG"]
    if "DI" in l1.columns and "di" not in l1.columns:
        l1["di"] = l1["DI"]
    l1 = l1[["subject_id", "sspg", "di"]]
    l2 = (
        pd.read_csv(os.path.join(data_root, "D2_stanford", "labels.csv"))[["subject_id", "SSPG", "DI"]]
        .rename(columns={"SSPG": "sspg", "DI": "di"})
    )
    return feat.merge(pd.concat([l1, l2], ignore_index=True), on="subject_id", how="left")


def _load_tier2_tier3_tier4(new_results_root: str, data_root: str) -> pd.DataFrame:
    # Prefer Exp2 DI run's latent CSV (contains both sspg/di and 26D latent).
    exp2 = os.path.join(new_results_root, "New_exp2_di")
    train_dirs = [d for d in os.listdir(exp2) if d.startswith("train_D1D2_")]
    train_dirs = sorted(train_dirs)
    if not train_dirs:
        raise RuntimeError("No New_exp2_di train_D1D2_* directory found.")
    lat_csv = os.path.join(exp2, train_dirs[-1], "latent_and_gold_all_26d.csv")
    df = pd.read_csv(lat_csv)
    if "dataset_id" not in df.columns:
        df["dataset_id"] = df["subject_id"].astype(str).str.split("_").str[0]
    keep = [c for c in df.columns if c in ["subject_id", "dataset_id", "sspg", "di", "tau_m", "Gb", "sg", "si", "p2", "mi"] or c.startswith("z_init_") or c.startswith("z_nonseq_")]
    out = df[keep].copy()

    # Demographics (age, bmi, sex) from D1/D2 subjects tables
    s1 = pd.read_csv(os.path.join(data_root, "D1_metwally", "subjects.csv"))[["subject_id", "age", "bmi", "sex"]]
    s2 = pd.read_csv(os.path.join(data_root, "D2_stanford", "subjects.csv"))[["subject_id", "age", "bmi", "sex"]]
    sd = pd.concat([s1, s2], ignore_index=True).drop_duplicates("subject_id")
    sd["sex"] = sd["sex"].astype(str).str.upper().str.startswith("F").astype(float)
    return out.merge(sd, on="subject_id", how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--new_results_root", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t1 = _load_tier1(args.data_root)
    t234 = _load_tier2_tier3_tier4(args.new_results_root, args.data_root)

    tier1_feats = ["cgm_mean", "cgm_var", "tir_70_140"]
    tier2_feats = ["tau_m", "Gb", "sg", "si", "p2", "mi"]
    tier3_feats = tier2_feats + [f"z_init_{i}" for i in range(4)] + [f"z_nonseq_{i}" for i in range(16)]
    tier4_feats = tier3_feats + ["age", "bmi", "sex"]

    rows = []
    rows.append(
        {
            "feature_combo": "Tier 1: CGM stats",
            "sspg_lodo_pearson_r": _lodo_predict(t1, tier1_feats, "sspg"),
            "di_lodo_pearson_r": _lodo_predict(t1, tier1_feats, "di"),
        }
    )
    for name, feats in [
        ("Tier 2: 6D ODE params", tier2_feats),
        ("Tier 3: 26D full latent", tier3_feats),
        ("Tier 4: 26D + demographics", tier4_feats),
    ]:
        rows.append(
            {
                "feature_combo": name,
                "sspg_lodo_pearson_r": _lodo_predict(t234, feats, "sspg"),
                "di_lodo_pearson_r": _lodo_predict(t234, feats, "di"),
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "New_exp4_lodo_ablation_table.csv")
    out_df.to_csv(out_csv, index=False)

    with open(os.path.join(args.out_dir, "New_exp4_lodo_ablation_table.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

