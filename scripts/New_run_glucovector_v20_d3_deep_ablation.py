from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = "/Users/hertz1030/Documents/GitHub/interpretable-cgm-representations"
DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
IN_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v20", "d3_free_living")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v20", "d3_deep_ablation")


def _icc_oneway(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    row_means = np.nanmean(x, axis=1)
    grand_mean = np.nanmean(x)
    ss_between = k * np.nansum((row_means - grand_mean) ** 2)
    ss_within = np.nansum((x - row_means[:, None]) ** 2)
    ms_between = ss_between / max(n - 1, 1)
    ms_within = ss_within / max(n * (k - 1), 1)
    den = ms_between + (k - 1) * ms_within
    if den <= 0:
        return np.nan
    return float((ms_between - ms_within) / den)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    return float(stats.spearmanr(x[ok], y[ok])[0])


def _window_uncertainty() -> pd.DataFrame:
    meals = pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "meals.csv"))
    cgm = pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "cgm.csv"))
    if "glucose_mg_dl" in cgm.columns and "glucose_mgdl" not in cgm.columns:
        cgm = cgm.rename(columns={"glucose_mg_dl": "glucose_mgdl"})
    meals["timestamp"] = pd.to_datetime(meals["timestamp"], errors="coerce")
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")

    rows: List[Dict[str, object]] = []
    for _, m in meals.iterrows():
        sid = str(m["subject_id"])
        t0 = m["timestamp"]
        if pd.isna(t0):
            continue
        g = cgm[
            (cgm["subject_id"] == sid)
            & (cgm["timestamp"] >= t0 + pd.Timedelta(minutes=-30))
            & (cgm["timestamp"] <= t0 + pd.Timedelta(minutes=180))
        ].copy()
        if len(g) < 10:
            continue
        t = ((g["timestamp"] - t0).dt.total_seconds() / 60.0).to_numpy(float)
        y = pd.to_numeric(g["glucose_mgdl"], errors="coerce").to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(y)
        if ok.sum() < 10:
            continue
        t, y = t[ok], y[ok]
        order = np.argsort(t)
        t, y = t[order], y[order]
        grid = np.arange(-30, 181, 5, dtype=float)
        y_new = np.interp(grid, t, y)
        post = y_new[grid >= 0]
        g0 = float(y_new[grid == 0][0]) if np.any(grid == 0) else float(post[0])
        delta = post - g0
        cv_post = float(np.std(post) / max(np.mean(post), 1e-8))
        iauc_abs = float(np.trapz(np.abs(delta), grid[grid >= 0]))
        rows.append(
            {
                "subject_id": sid,
                "meal_type": str(m.get("meal_type", "Unknown")),
                "timestamp": t0,
                "date": pd.Timestamp(t0).date().isoformat(),
                "carb_g": float(pd.to_numeric(m.get("carb_g", np.nan), errors="coerce")),
                "fat_g": float(pd.to_numeric(m.get("fat_g", np.nan), errors="coerce")),
                "protein_g": float(pd.to_numeric(m.get("protein_g", np.nan), errors="coerce")),
                "cv_post": cv_post,
                "iauc_abs": iauc_abs,
            }
        )
    df = pd.DataFrame(rows)
    z1 = (df["cv_post"] - df["cv_post"].mean()) / (df["cv_post"].std() + 1e-8)
    z2 = (df["iauc_abs"] - df["iauc_abs"].mean()) / (df["iauc_abs"].std() + 1e-8)
    df["uncertainty_score"] = 0.5 * z1 + 0.5 * z2
    ql, qh = df["uncertainty_score"].quantile([0.3, 0.7]).tolist()
    df["uncertainty_bin"] = "mid"
    df.loc[df["uncertainty_score"] <= ql, "uncertainty_bin"] = "low"
    df.loc[df["uncertainty_score"] >= qh, "uncertainty_bin"] = "high"
    return df


def _load_labels() -> pd.DataFrame:
    labels = pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "labels.csv"))
    return labels[["subject_id", "hba1c", "HOMA_IR", "HOMA_B"]].drop_duplicates("subject_id")


def _probe_loocv(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) < 10:
        return {"n": int(len(y)), "spearman": np.nan, "mae": np.nan}
    loo = LeaveOneOut()
    y_hat = np.full_like(y, np.nan, dtype=float)
    for tr, te in loo.split(X):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X[tr], y[tr])
        y_hat[te] = model.predict(X[te])
    return {"n": int(len(y)), "spearman": _safe_spearman(y, y_hat), "mae": float(mean_absolute_error(y, y_hat))}


def _cross_meal_day_metrics(df: pd.DataFrame, model_name: str) -> Dict[str, float]:
    out: Dict[str, float] = {"model": model_name}
    major = df[df["meal_type"].isin(["Breakfast", "Lunch", "Dinner"])].copy()
    out["n_major_windows"] = int(len(major))

    for tgt in ["sspg_pred", "di_pred"]:
        w = major.pivot_table(index="subject_id", columns="meal_type", values=tgt, aggfunc="mean")
        cols = [c for c in ["Breakfast", "Lunch", "Dinner"] if c in w.columns]
        w = w[cols].dropna()
        out[f"{tgt}_meal_icc"] = _icc_oneway(w.to_numpy()) if len(w) >= 5 and len(cols) >= 2 else np.nan

        day = major.groupby(["subject_id", "date"], as_index=False)[tgt].mean()
        sd = day.groupby("subject_id")[tgt].std().dropna()
        out[f"{tgt}_cross_day_std_mean"] = float(sd.mean()) if len(sd) else np.nan
        out[f"{tgt}_cross_day_std_median"] = float(sd.median()) if len(sd) else np.nan

        day_trip = major.groupby(["subject_id", "date", "meal_type"], as_index=False)[tgt].mean()
        w_day = day_trip.pivot_table(index=["subject_id", "date"], columns="meal_type", values=tgt, aggfunc="mean")
        cols = [c for c in ["Breakfast", "Lunch", "Dinner"] if c in w_day.columns]
        w_day = w_day[cols].dropna()
        out[f"{tgt}_triplet_n"] = int(len(w_day))
        if len(w_day) >= 5 and len(cols) == 3:
            mat = w_day.to_numpy(float)
            out[f"{tgt}_triplet_icc"] = _icc_oneway(mat)
            out[f"{tgt}_triplet_std_mean"] = float(np.mean(np.std(mat, axis=1)))
        else:
            out[f"{tgt}_triplet_icc"] = np.nan
            out[f"{tgt}_triplet_std_mean"] = np.nan
    return out


def _uncertainty_stratified_corr(df: pd.DataFrame, model_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ub in ["low", "high"]:
        d = df[df["uncertainty_bin"] == ub].copy()
        sub = d.groupby("subject_id", as_index=False).median(numeric_only=True)
        rows.append(
            {
                "model": model_name,
                "uncertainty_bin": ub,
                "n_subjects": int(sub["subject_id"].nunique()),
                "sspg_vs_homa_ir_spearman": _safe_spearman(sub["sspg_pred"], sub["HOMA_IR"]),
                "sspg_vs_hba1c_spearman": _safe_spearman(sub["sspg_pred"], sub["hba1c"]),
                "di_vs_homa_ir_spearman": _safe_spearman(sub["di_pred"], sub["HOMA_IR"]),
            }
        )
    return rows


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    labels = _load_labels()
    win_aux = _window_uncertainty()
    win_aux.to_csv(os.path.join(OUT_ROOT, "d3_window_uncertainty.csv"), index=False)

    pred_files = {
        "Wang_Exp1": os.path.join(IN_ROOT, "d3_predictions_with_labels_Wang_Exp1.csv"),
        "GV_26D_Exp8": os.path.join(IN_ROOT, "d3_predictions_with_labels_GV_26D_Exp8.csv"),
        "GV_10D_head_v20": os.path.join(IN_ROOT, "d3_predictions_with_labels_GV_10D_head_v20.csv"),
        "Metwally14_Ridge": os.path.join(IN_ROOT, "d3_predictions_with_labels_Metwally14_Ridge.csv"),
    }

    cross_rows: List[Dict[str, object]] = []
    unc_rows: List[Dict[str, object]] = []
    probe_rows: List[Dict[str, object]] = []

    for name, path in pred_files.items():
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        win_aux2 = win_aux.copy()
        win_aux2["timestamp"] = pd.to_datetime(win_aux2["timestamp"], errors="coerce")
        m = df.merge(
            win_aux2[["subject_id", "meal_type", "timestamp", "uncertainty_score", "uncertainty_bin"]],
            on=["subject_id", "meal_type", "timestamp"],
            how="inner",
        )
        m = m.merge(labels, on="subject_id", how="left")
        for c in ["hba1c", "HOMA_IR", "HOMA_B"]:
            if c not in m.columns:
                if f"{c}_x" in m.columns:
                    m[c] = m[f"{c}_x"]
                elif f"{c}_y" in m.columns:
                    m[c] = m[f"{c}_y"]
        if "date" not in m.columns:
            m["date"] = pd.to_datetime(m["timestamp"], errors="coerce").dt.date.astype(str)
        m.to_csv(os.path.join(OUT_ROOT, f"d3_merged_{name}.csv"), index=False)

        cross_rows.append(_cross_meal_day_metrics(m, name))
        unc_rows.extend(_uncertainty_stratified_corr(m, name))

        # Representation contribution: same 10D probe protocol on weak labels.
        if all(f"z{i:02d}" in m.columns for i in range(10)):
            sub = m.groupby("subject_id", as_index=False).mean(numeric_only=True)
            x10 = sub[[f"z{i:02d}" for i in range(10)]].to_numpy(float)
            p_homa = _probe_loocv(x10, sub["HOMA_IR"].to_numpy(float))
            p_hba1c = _probe_loocv(x10, sub["hba1c"].to_numpy(float))
            probe_rows.append(
                {
                    "model": name,
                    "feature_set": "10D",
                    "homa_ir_probe_spearman": p_homa["spearman"],
                    "hba1c_probe_spearman": p_hba1c["spearman"],
                    "probe_n": p_homa["n"],
                }
            )
            if all(f"z{i:02d}" in m.columns for i in range(26)):
                x26 = sub[[f"z{i:02d}" for i in range(26)]].to_numpy(float)
                p_homa26 = _probe_loocv(x26, sub["HOMA_IR"].to_numpy(float))
                p_hba1c_26 = _probe_loocv(x26, sub["hba1c"].to_numpy(float))
                probe_rows.append(
                    {
                        "model": name,
                        "feature_set": "26D",
                        "homa_ir_probe_spearman": p_homa26["spearman"],
                        "hba1c_probe_spearman": p_hba1c_26["spearman"],
                        "probe_n": p_homa26["n"],
                    }
                )

    cross_df = pd.DataFrame(cross_rows)
    unc_df = pd.DataFrame(unc_rows)
    probe_df = pd.DataFrame(probe_rows)

    cross_df.to_csv(os.path.join(OUT_ROOT, "d3_cross_meal_cross_day_metrics.csv"), index=False)
    unc_df.to_csv(os.path.join(OUT_ROOT, "d3_uncertainty_stratified_correlations.csv"), index=False)
    probe_df.to_csv(os.path.join(OUT_ROOT, "d3_representation_probe_comparison.csv"), index=False)

    # 16D incremental value under high uncertainty (GV26 vs GV10)
    comp = unc_df.pivot_table(index="uncertainty_bin", columns="model", values="sspg_vs_hba1c_spearman")
    delta_high = np.nan
    if "high" in comp.index and "GV_26D_Exp8" in comp.columns and "GV_10D_head_v20" in comp.columns:
        delta_high = float(comp.loc["high", "GV_26D_Exp8"] - comp.loc["high", "GV_10D_head_v20"])
    summary = {
        "delta_high_uncertainty_sspg_hba1c_spearman_gv26_minus_gv10": delta_high,
        "generated_at": datetime.now().isoformat(),
    }
    with open(os.path.join(OUT_ROOT, "d3_deep_ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUT_ROOT, "d3_deep_ablation_report.md"), "w", encoding="utf-8") as f:
        f.write("# v20 D3 Deep Ablation (10D vs 26D vs Wang vs Metwally)\n\n")
        f.write(f"Generated: {summary['generated_at']}\n\n")
        f.write("## Cross-meal / Cross-day\n\n")
        f.write(cross_df.to_markdown(index=False))
        f.write("\n\n## Uncertainty-stratified weak-label alignment\n\n")
        f.write(unc_df.to_markdown(index=False))
        f.write("\n\n## Representation probe (same protocol)\n\n")
        f.write(probe_df.to_markdown(index=False))
        f.write("\n\n## Summary\n\n")
        f.write(json.dumps(summary, indent=2))
        f.write("\n")

    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    main()
