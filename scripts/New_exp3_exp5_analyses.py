"""
New_ analyses for GlucoVector v6:
- Exp3: D2 meal-type response prediction (SSPG/DI)
- Exp5: D4 cross-context stability (OGTT vs standard meal vs free-living)

This script intentionally uses only the newest dataset tree.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def _curve_features(t: np.ndarray, g: np.ndarray, min_points: int = 5) -> Dict[str, float]:
    t = pd.to_numeric(pd.Series(t), errors="coerce").to_numpy(dtype=float)
    g = pd.to_numeric(pd.Series(g), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(g)
    t = t[ok]
    g = g[ok]
    if len(g) < min_points:
        return {}
    order = np.argsort(t)
    t = t[order]
    g = g[order]
    auc = float(np.trapz(g, t))
    return {
        "mean": float(np.mean(g)),
        "std": float(np.std(g)),
        "min": float(np.min(g)),
        "max": float(np.max(g)),
        "range": float(np.max(g) - np.min(g)),
        "delta_end_start": float(g[-1] - g[0]),
        "auc": auc,
    }


def _load_d2_meal_features(data_root: str) -> pd.DataFrame:
    cgm = pd.read_csv(os.path.join(data_root, "D2_stanford", "cgm.csv"))
    labels = pd.read_csv(os.path.join(data_root, "D2_stanford", "labels.csv"))
    cgm = cgm.rename(columns={"glucose_mg_dl": "glucose", "minutes_after_meal": "t"})
    rows: List[Dict[str, float]] = []
    for (sid, meal_type, rep), df in cgm.groupby(["subject_id", "meal_type", "rep"]):
        feat = _curve_features(df["t"].values, df["glucose"].values)
        if not feat:
            continue
        feat["subject_id"] = sid
        feat["meal_type"] = meal_type
        rows.append(feat)
    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        return feat_df
    lab = labels[["subject_id", "SSPG", "DI"]].rename(columns={"SSPG": "sspg", "DI": "di"})
    return feat_df.merge(lab, on="subject_id", how="left")


def _group_cv_pearson(df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fcols = [c for c in ["mean", "std", "min", "max", "range", "delta_end_start", "auc"] if c in df.columns]
    sub = df.dropna(subset=fcols + [target, "subject_id"]).copy()
    X = sub[fcols].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    g = sub["subject_id"].astype(str).to_numpy()

    y_true_all, y_hat_all, g_all = [], [], []
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, g):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        reg = RidgeCV(alphas=np.logspace(-3, 3, 13))
        reg.fit(Xtr, y[tr])
        yhat = reg.predict(Xte)
        y_true_all.append(y[te])
        y_hat_all.append(yhat)
        g_all.append(g[te])
    return np.concatenate(y_true_all), np.concatenate(y_hat_all), np.concatenate(g_all)


def run_exp3(data_root: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = _load_d2_meal_features(data_root)
    if df.empty:
        raise RuntimeError("Exp3: No D2 meal features extracted.")

    meal_types = ["Rice", "Bread", "Potatoes", "Grapes"]
    out_rows = []
    for mt in meal_types:
        sub = df[df["meal_type"] == mt].copy()
        if len(sub) < 20:
            out_rows.append({"meal_type": mt, "n": int(len(sub)), "sspg_pearson_r": np.nan, "di_pearson_r": np.nan})
            continue
        y_s, yhat_s, _ = _group_cv_pearson(sub, "sspg")
        y_d, yhat_d, _ = _group_cv_pearson(sub, "di")
        r_s = stats.pearsonr(y_s, yhat_s)[0] if len(y_s) > 2 else np.nan
        r_d = stats.pearsonr(y_d, yhat_d)[0] if len(y_d) > 2 else np.nan
        out_rows.append({"meal_type": mt, "n": int(len(sub)), "sspg_pearson_r": float(r_s), "di_pearson_r": float(r_d)})

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(os.path.join(out_dir, "New_exp3_meal_type_table.csv"), index=False)


def _fit_d1d2_subject_models(data_root: str) -> Tuple[object, object, List[str]]:
    d1 = pd.read_csv(os.path.join(data_root, "D1_metwally", "cgm.csv"))
    d1_lab = pd.read_csv(os.path.join(data_root, "D1_metwally", "labels.csv"))
    d2 = pd.read_csv(os.path.join(data_root, "D2_stanford", "cgm.csv"))
    d2_lab = pd.read_csv(os.path.join(data_root, "D2_stanford", "labels.csv"))

    d1 = d1.rename(columns={"timepoint_mins": "t", "glucose_mg_dl": "glucose"})
    d2 = d2.rename(columns={"minutes_after_meal": "t", "glucose_mg_dl": "glucose"})

    rows = []
    for sid, g in d1.groupby("subject_id"):
        feat = _curve_features(g["t"].values, g["glucose"].values)
        if feat:
            feat["subject_id"] = sid
            rows.append(feat)
    for sid, g in d2.groupby("subject_id"):
        # D2 has many meals; summarize by averaging rep-level summaries
        rr = []
        for (_, _mt, _rep), gg in g.groupby(["subject_id", "meal_type", "rep"]):
            f = _curve_features(gg["t"].values, gg["glucose"].values)
            if f:
                rr.append(f)
        if rr:
            m = pd.DataFrame(rr).mean().to_dict()
            m["subject_id"] = sid
            rows.append(m)
    feat = pd.DataFrame(rows)
    if "SSPG" in d1_lab.columns and "sspg" not in d1_lab.columns:
        d1_lab["sspg"] = d1_lab["SSPG"]
    if "DI" in d1_lab.columns and "di" not in d1_lab.columns:
        d1_lab["di"] = d1_lab["DI"]

    lab = pd.concat(
        [
            d1_lab[["subject_id", "sspg", "di"]],
            d2_lab[["subject_id", "SSPG", "DI"]].rename(columns={"SSPG": "sspg", "DI": "di"}),
        ],
        ignore_index=True,
    )
    df = feat.merge(lab, on="subject_id", how="inner").dropna(subset=["sspg", "di"])
    fcols = [c for c in ["mean", "std", "min", "max", "range", "delta_end_start", "auc"] if c in df.columns]
    X = StandardScaler().fit_transform(df[fcols].to_numpy(dtype=float))
    model_sspg = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X, df["sspg"].to_numpy(dtype=float))
    model_di = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X, df["di"].to_numpy(dtype=float))
    return model_sspg, model_di, fcols


def run_exp5(data_root: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model_s, model_d, fcols = _fit_d1d2_subject_models(data_root)
    s_map = pd.read_csv(os.path.join(data_root, "D4_hall", "subjects.csv"))
    d3h_to_d4 = {}
    if {"subject_id", "original_id"}.issubset(set(s_map.columns)):
        for _, r in s_map.dropna(subset=["subject_id", "original_id"]).iterrows():
            d3h_to_d4[str(r["subject_id"]).strip()] = f"D4_{str(r['original_id']).strip()}"

    def _canon_sid(s: str) -> str:
        s = str(s)
        if s in d3h_to_d4:
            return d3h_to_d4[s]
        return s

    # D4 standard meal context
    d4_std = pd.read_csv(os.path.join(data_root, "D4_hall", "cgm.csv")).rename(columns={"glucose_mg_dl": "glucose"})
    # each meal_context has repeated meal runs; summarize per subject by mean of contexts
    rows_std = []
    for (sid, ctx), g in d4_std.groupby(["subject_id", "meal_context"]):
        g = g.copy()
        g["timestamp"] = pd.to_datetime(g["timestamp"], errors="coerce")
        if g["timestamp"].isna().all():
            continue
        t = (g["timestamp"] - g["timestamp"].min()).dt.total_seconds() / 60.0
        feat = _curve_features(t.values, g["glucose"].values)
        if feat:
            feat["subject_id"] = _canon_sid(sid)
            feat["context"] = ctx
            rows_std.append(feat)
    std_df = pd.DataFrame(rows_std)
    std_subj = std_df.groupby("subject_id")[fcols].mean()

    # D4 OGTT context
    og = pd.read_csv(os.path.join(data_root, "D4_hall", "ogtt_timeseries.csv"))
    if "parameter" in og.columns:
        p = og["parameter"].astype(str).str.lower()
        og = og[p.str.contains("glucose", na=False) | p.eq("ogtt")].copy()
    if "value" in og.columns and "glucose" not in og.columns:
        og["glucose"] = pd.to_numeric(og["value"], errors="coerce")
    tcol = "timepoint_mins" if "timepoint_mins" in og.columns else "minutes_after_glucose"
    rows_og = []
    for sid, g in og.groupby("subject_id"):
        feat = _curve_features(g[tcol].values, g["glucose"].values, min_points=3)
        if feat:
            feat["subject_id"] = _canon_sid(sid)
            rows_og.append(feat)
    og_df = pd.DataFrame(rows_og)
    if og_df.empty:
        og_df = pd.DataFrame(columns=fcols)
        og_df.index.name = "subject_id"
    else:
        og_df = og_df.set_index("subject_id")

    # D4 free-living context
    fl = pd.read_csv(os.path.join(data_root, "D4_hall", "cgm_freeliving.csv"))
    fl = fl.rename(columns={"glucose_mg_dl": "glucose", "glucose_mgdl": "glucose"})
    fl["timestamp"] = pd.to_datetime(fl["timestamp"], errors="coerce")
    rows_fl = []
    for sid, g in fl.groupby("subject_id"):
        g = g.dropna(subset=["timestamp"])
        if g.empty:
            continue
        t = (g["timestamp"] - g["timestamp"].min()).dt.total_seconds() / 60.0
        feat = _curve_features(t.values, g["glucose"].values)
        if feat:
            feat["subject_id"] = _canon_sid(sid)
            rows_fl.append(feat)
    fl_df = pd.DataFrame(rows_fl)
    if fl_df.empty:
        fl_df = pd.DataFrame(columns=fcols)
        fl_df.index.name = "subject_id"
    else:
        fl_df = fl_df.set_index("subject_id")

    def _predict(df_ctx: pd.DataFrame) -> pd.DataFrame:
        if df_ctx.empty:
            return pd.DataFrame(columns=["sspg_pred", "di_pred"])
        X = StandardScaler().fit_transform(df_ctx[fcols].to_numpy(dtype=float))
        return pd.DataFrame(
            {
                "sspg_pred": model_s.predict(X),
                "di_pred": model_d.predict(X),
            },
            index=df_ctx.index,
        )

    p_std = _predict(std_subj)
    p_og = _predict(og_df)
    p_fl = _predict(fl_df)

    # Save predictions
    p_std.to_csv(os.path.join(out_dir, "New_d4_standard_pred.csv"))
    p_og.to_csv(os.path.join(out_dir, "New_d4_ogtt_pred.csv"))
    p_fl.to_csv(os.path.join(out_dir, "New_d4_freeliving_pred.csv"))

    # Pairwise subject-overlap correlations (SSPG prediction consistency)
    def _pair_corr(a: pd.Series, b: pd.Series) -> Dict[str, float]:
        common = sorted(set(a.index) & set(b.index))
        if len(common) < 3:
            return {"n": len(common), "r": np.nan, "p": np.nan}
        r, p = stats.pearsonr(a.loc[common].values, b.loc[common].values)
        return {"n": len(common), "r": float(r), "p": float(p)}

    corr = {
        "SSPG_pred": {
            "OGTT_vs_Standard": _pair_corr(p_og["sspg_pred"], p_std["sspg_pred"]),
            "OGTT_vs_FreeLiving": _pair_corr(p_og["sspg_pred"], p_fl["sspg_pred"]),
            "Standard_vs_FreeLiving": _pair_corr(p_std["sspg_pred"], p_fl["sspg_pred"]),
        },
        "DI_pred": {
            "OGTT_vs_Standard": _pair_corr(p_og["di_pred"], p_std["di_pred"]),
            "OGTT_vs_FreeLiving": _pair_corr(p_og["di_pred"], p_fl["di_pred"]),
            "Standard_vs_FreeLiving": _pair_corr(p_std["di_pred"], p_fl["di_pred"]),
        },
    }
    with open(os.path.join(out_dir, "New_exp5_context_stability.json"), "w") as f:
        json.dump(corr, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--out_root", required=True, type=str)
    args = ap.parse_args()

    run_exp3(args.data_root, os.path.join(args.out_root, "New_exp3_meal_type"))
    run_exp5(args.data_root, os.path.join(args.out_root, "New_exp5_stability"))
    print("Exp3 and Exp5 done.")


if __name__ == "__main__":
    main()

