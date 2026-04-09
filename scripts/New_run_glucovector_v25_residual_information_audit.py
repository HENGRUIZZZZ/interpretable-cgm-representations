from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import MechanisticAutoencoder

DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
V22_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v22_locked_protocol")
V18_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v18")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v25_residual_information_audit")
SEED = 42


def _safe_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    if len(y) < 3:
        return {"n": int(len(y)), "spearman": np.nan, "pearson": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    return {
        "n": int(len(y)),
        "spearman": float(stats.spearmanr(y, p)[0]),
        "pearson": float(stats.pearsonr(y, p)[0]),
        "r2": float(r2_score(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
    }


def _residual_deep_dive() -> Dict[str, pd.DataFrame]:
    sub = pd.read_csv(os.path.join(V22_ROOT, "v22_d4_subject_level_predictions.csv"))
    meal = pd.read_csv(os.path.join(V22_ROOT, "v22_d4_meal_level_predictions.csv"))
    meal["abs_res_10_sspg"] = np.abs(meal["sspg_true"] - meal["p10_sspg"])
    meal["abs_res_26_sspg"] = np.abs(meal["sspg_true"] - meal["p26_sspg"])
    meal["abs_res_met_sspg"] = np.abs(meal["sspg_true"] - meal["pmet_sspg"])

    # subject-level residuals
    for m in ["10", "26", "met"]:
        sub[f"res_{m}_sspg"] = sub["sspg_true"] - sub[f"p{m}_sspg"]
        sub[f"abs_res_{m}_sspg"] = np.abs(sub[f"res_{m}_sspg"])
        sub[f"res_{m}_di"] = sub["di_true"] - sub[f"p{m}_di"]
        sub[f"abs_res_{m}_di"] = np.abs(sub[f"res_{m}_di"])

    # calibration/scale compression diagnostics
    cal_rows = []
    for model, ps_col, pd_col in [("Ridge10D", "p10_sspg", "p10_di"), ("Ridge26D", "p26_sspg", "p26_di"), ("Metwally14", "pmet_sspg", "pmet_di")]:
        for tgt, ycol, pcol in [("sspg", "sspg_true", ps_col), ("di", "di_true", pd_col)]:
            d = sub[[ycol, pcol]].dropna()
            if len(d) < 5:
                slope = intercept = r = np.nan
            else:
                reg = stats.linregress(d[pcol].to_numpy(float), d[ycol].to_numpy(float))
                slope, intercept, r = float(reg.slope), float(reg.intercept), float(reg.rvalue)
            cal_rows.append({"model": model, "target": tgt, "n": int(len(d)), "calibration_slope_true_on_pred": slope, "calibration_intercept": intercept, "corr": r})
    cal_df = pd.DataFrame(cal_rows)

    # residual-covariate correlations
    covs = ["uncertainty_score", "carb_g", "fat_g", "protein_g", "fiber_g", "G_120", "AUC", "CV"]
    rc_rows = []
    for model, col in [("Ridge10D", "abs_res_10_sspg"), ("Ridge26D", "abs_res_26_sspg"), ("Metwally14", "abs_res_met_sspg")]:
        for c in covs:
            d = meal[[col, c]].dropna()
            if len(d) < 10:
                rho = p = np.nan
            else:
                rr = stats.spearmanr(d[col].to_numpy(float), d[c].to_numpy(float))
                rho, p = float(rr.statistic), float(rr.pvalue)
            rc_rows.append({"model": model, "target": "sspg_abs_residual", "covariate": c, "spearman_rho": rho, "pvalue": p, "n": int(len(d))})
    rc_df = pd.DataFrame(rc_rows)

    # error by true quantiles
    q_rows = []
    for tgt, true_col in [("sspg", "sspg_true"), ("di", "di_true")]:
        d = sub[[true_col, "abs_res_10_" + tgt, "abs_res_26_" + tgt, "abs_res_met_" + tgt]].dropna()
        if len(d) < 8:
            continue
        d["bin"] = pd.qcut(d[true_col], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
        for b, g in d.groupby("bin"):
            q_rows.append(
                {
                    "target": tgt,
                    "true_bin": str(b),
                    "n": int(len(g)),
                    "mae_10d": float(g["abs_res_10_" + tgt].mean()),
                    "mae_26d": float(g["abs_res_26_" + tgt].mean()),
                    "mae_met": float(g["abs_res_met_" + tgt].mean()),
                }
            )
    q_df = pd.DataFrame(q_rows)

    # hardest subjects
    hard_sspg = sub.sort_values("abs_res_26_sspg", ascending=False)[["subject_id", "sspg_true", "p26_sspg", "abs_res_26_sspg", "pmet_sspg", "abs_res_met_sspg"]].head(10)
    hard_di = sub.sort_values("abs_res_26_di", ascending=False)[["subject_id", "di_true", "p26_di", "abs_res_26_di", "pmet_di", "abs_res_met_di"]].head(10)

    # R2 vs rank consistency illustration
    trade_rows = []
    for model, ps_col, pd_col in [("Ridge10D", "p10_sspg", "p10_di"), ("Ridge26D", "p26_sspg", "p26_di"), ("Metwally14", "pmet_sspg", "pmet_di")]:
        ms = _safe_metrics(sub["sspg_true"].to_numpy(float), sub[ps_col].to_numpy(float))
        md = _safe_metrics(sub["di_true"].to_numpy(float), sub[pd_col].to_numpy(float))
        trade_rows.append({"model": model, "target": "sspg", **ms})
        trade_rows.append({"model": model, "target": "di", **md})
    trade_df = pd.DataFrame(trade_rows)

    return {
        "subject_residuals": sub,
        "calibration": cal_df,
        "residual_covariates": rc_df,
        "error_by_true_quantile": q_df,
        "hard_subjects_sspg": hard_sspg,
        "hard_subjects_di": hard_di,
        "r2_rank_tradeoff": trade_df,
    }


class Encoder:
    def __init__(self):
        ck = torch.load(
            os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase2_finetune_head", "autoencoder_p1_full.pt"),
            map_location="cpu",
            weights_only=False,
        )
        self.ck = ck
        self.model = MechanisticAutoencoder(
            meal_size=6, demographics_size=3, embedding_size=8, hidden_size=32, num_layers=2,
            encoder_dropout_prob=0.0, decoder_dropout_prob=0.5
        )
        self.model.load_state_dict(ck["model_state"], strict=False)
        self.model.eval()

    def encode_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        tm, tsd = self.ck["train_mean"], self.ck["train_std"]
        out = []
        for _, r in rows.iterrows():
            c = r["curve"][None, :, None].astype(np.float32)
            ts = r["timestamps"][None, :, None].astype(np.float32)
            meal_series = np.zeros((c.shape[1], 6), dtype=np.float32)
            meal_series[:, 0] = float(r.get("carb_g", 0.0) + r.get("fat_g", 0.0) + r.get("protein_g", 0.0))
            meal_series[:, 1] = float(r.get("carb_g", 0.0))
            meal_series[:, 3] = float(r.get("fiber_g", 0.0))
            meal_series[:, 4] = float(r.get("fat_g", 0.0))
            meal_series[:, 5] = float(r.get("protein_g", 0.0))
            meals = meal_series[None, :, :]
            demo = np.array([[0.0, 45.0, 72.0]], dtype=np.float32)
            c = (c - tm[0]) / (tsd[0] + 1e-8)
            ts = (ts - tm[1]) / (tsd[1] + 1e-8)
            meals = (meals - tm[2]) / (tsd[2] + 1e-8)
            demo = (demo - tm[3]) / (tsd[3] + 1e-8)
            with torch.no_grad():
                p26, init26, z16 = self.model.get_all_latents(
                    torch.tensor(c), torch.tensor(ts), torch.tensor(meals), torch.tensor(demo)
                )
            z = np.concatenate([p26.numpy()[0], init26.numpy()[0], z16.numpy()[0]], axis=0)
            row = {"subject_id": r["subject_id"], "meal_type": r["meal_type"]}
            for i, v in enumerate(z):
                row[f"z{i:02d}"] = float(v)
            out.append(row)
        return pd.DataFrame(out)


def _build_d3_subject_latent_labels() -> pd.DataFrame:
    meals = pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "meals.csv"))
    cgm = pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "cgm.csv"))
    labels = pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "labels.csv"))
    if "glucose_mg_dl" in cgm.columns and "glucose_mgdl" not in cgm.columns:
        cgm = cgm.rename(columns={"glucose_mg_dl": "glucose_mgdl"})
    meals["timestamp"] = pd.to_datetime(meals["timestamp"], errors="coerce")
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")
    labels = labels.drop_duplicates("subject_id")

    grid = np.arange(-30, 181, 5, dtype=np.float64)
    rows = []
    for _, meal in meals.iterrows():
        sid = str(meal["subject_id"])
        t0 = meal["timestamp"]
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
        y_new = np.interp(grid, t, y).astype(np.float32)
        rows.append(
            {
                "subject_id": sid,
                "meal_type": str(meal.get("meal_type", "Unknown")),
                "curve": y_new,
                "timestamps": grid.astype(np.float32),
                "carb_g": float(pd.to_numeric(meal.get("carb_g", 0.0), errors="coerce") or 0.0),
                "fat_g": float(pd.to_numeric(meal.get("fat_g", 0.0), errors="coerce") or 0.0),
                "protein_g": float(pd.to_numeric(meal.get("protein_g", 0.0), errors="coerce") or 0.0),
                "fiber_g": float(pd.to_numeric(meal.get("fiber_g", 0.0), errors="coerce") or 0.0),
            }
        )
    meal_df = pd.DataFrame(rows)
    enc = Encoder()
    lat = enc.encode_rows(meal_df)
    sub_lat = lat.groupby("subject_id", as_index=False).mean(numeric_only=True)
    merged = sub_lat.merge(labels, on="subject_id", how="left")
    return merged


def _multi_label_information_audit() -> pd.DataFrame:
    df = _build_d3_subject_latent_labels()
    z10 = [f"z{i:02d}" for i in range(10)]
    z16 = [f"z{i:02d}" for i in range(10, 26)]
    z26 = [f"z{i:02d}" for i in range(26)]
    candidate_targets = [
        "hba1c",
        "HOMA_IR",
        "HOMA_B",
        "fasting_glucose_mg_dl",
        "fasting_insulin_uiu_ml",
        "triglycerides_mg_dl",
        "hdl_mg_dl",
        "ldl_mg_dl",
        "cho_hdl_ratio",
    ]
    targets = [t for t in candidate_targets if t in df.columns]
    rows = []
    for tgt in targets:
        y = pd.to_numeric(df[tgt], errors="coerce").to_numpy(float)
        for feat_name, feats in [("10D", z10), ("16D", z16), ("26D", z26)]:
            X = df[feats].to_numpy(float)
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
            Xok, yok = X[ok], y[ok]
            if len(yok) < 10:
                rows.append({"target": tgt, "feature_set": feat_name, "n": int(len(yok)), "spearman": np.nan, "r2": np.nan, "rmse": np.nan})
                continue
            loo = LeaveOneOut()
            pred = np.full_like(yok, np.nan, dtype=float)
            for tr, te in loo.split(Xok):
                mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50)))
                mdl.fit(Xok[tr], yok[tr])
                pred[te] = mdl.predict(Xok[te])
            rows.append(
                {
                    "target": tgt,
                    "feature_set": feat_name,
                    "n": int(len(yok)),
                    "spearman": float(stats.spearmanr(yok, pred)[0]),
                    "r2": float(r2_score(yok, pred)),
                    "rmse": float(np.sqrt(mean_squared_error(yok, pred))),
                }
            )
    out = pd.DataFrame(rows)
    return out


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Part 1: residual deep dive
    res = _residual_deep_dive()
    for k, df in res.items():
        df.to_csv(os.path.join(OUT_ROOT, f"v25_{k}.csv"), index=False)

    # Part 2: information quantity expansion
    info_df = _multi_label_information_audit()
    info_df.to_csv(os.path.join(OUT_ROOT, "v25_multilabel_information_audit_d3.csv"), index=False)

    # compact summary
    sum_rows = []
    # residual takeaways
    cal = res["calibration"]
    if len(cal):
        s = cal[(cal["model"] == "Ridge26D") & (cal["target"] == "sspg")]
        if len(s):
            sum_rows.append({"topic": "calibration", "item": "Ridge26D_sspg_slope", "value": float(s.iloc[0]["calibration_slope_true_on_pred"])})
    rc = res["residual_covariates"]
    if len(rc):
        r = rc[(rc["model"] == "Ridge26D") & (rc["covariate"] == "uncertainty_score")]
        if len(r):
            sum_rows.append({"topic": "residual_pattern", "item": "Ridge26D_absres_vs_uncertainty_rho", "value": float(r.iloc[0]["spearman_rho"])})
    # information quantity
    if len(info_df):
        piv = info_df.pivot_table(index="target", columns="feature_set", values="spearman", aggfunc="first")
        for tgt in piv.index:
            if {"10D", "16D", "26D"}.issubset(set(piv.columns)):
                sum_rows.append({"topic": "multilabel", "item": f"{tgt}_spearman_26D_minus_10D", "value": float(piv.loc[tgt, "26D"] - piv.loc[tgt, "10D"])})
                sum_rows.append({"topic": "multilabel", "item": f"{tgt}_spearman_16D_minus_10D", "value": float(piv.loc[tgt, "16D"] - piv.loc[tgt, "10D"])})
    summary_df = pd.DataFrame(sum_rows)
    summary_df.to_csv(os.path.join(OUT_ROOT, "v25_summary_key_numbers.csv"), index=False)

    with open(os.path.join(OUT_ROOT, "v25_residual_information_report.md"), "w", encoding="utf-8") as f:
        f.write("# v25 Residual + Information Quantity Audit\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Residual calibration and structure\n\n")
        f.write(res["calibration"].to_markdown(index=False))
        f.write("\n\n### Residual vs covariates\n\n")
        f.write(res["residual_covariates"].to_markdown(index=False))
        f.write("\n\n### Error by true-label quantiles\n\n")
        f.write(res["error_by_true_quantile"].to_markdown(index=False))
        f.write("\n\n### Hardest subjects (SSPG)\n\n")
        f.write(res["hard_subjects_sspg"].to_markdown(index=False))
        f.write("\n\n### Hardest subjects (DI)\n\n")
        f.write(res["hard_subjects_di"].to_markdown(index=False))
        f.write("\n\n### R2 / rank tradeoff\n\n")
        f.write(res["r2_rank_tradeoff"].to_markdown(index=False))
        f.write("\n\n## Multi-label information quantity (D3)\n\n")
        f.write(info_df.to_markdown(index=False))
        f.write("\n\n## Key numbers summary\n\n")
        if len(summary_df):
            f.write(summary_df.to_markdown(index=False))
        else:
            f.write("No summary rows.\n")
        f.write("\n")
    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    main()
