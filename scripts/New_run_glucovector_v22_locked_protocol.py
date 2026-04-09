from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from load_cgm_project_data import load_cgm_project_level1_level2
from models import MechanisticAutoencoder

DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
V18_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v18")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v22_locked_protocol")
SEED = 42
STD_MEALS = ["Cornflakes", "PB_sandwich", "Protein_bar"]


def _norm_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for old, new in [("SSPG", "sspg"), ("DI", "di"), ("HOMA_IR", "homa_ir"), ("HOMA_B", "homa_b")]:
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    return out


def _met14_features(cgm: np.ndarray, ts: np.ndarray) -> Dict[str, float]:
    cgm = np.asarray(cgm, dtype=float)
    ts = np.asarray(ts, dtype=float)
    mask = (ts >= 0.0) & (ts <= 180.0)
    t = ts[mask]
    g = cgm[mask]
    if len(t) < 5:
        t = ts
        g = cgm
    order = np.argsort(t)
    t = t[order]
    g = g[order]
    f = lambda x: float(np.interp(x, t, g))
    g0 = float(g[0])
    g60, g120, g180 = f(60), f(120), f(180)
    g_peak = float(np.max(g))
    delta = g - g0
    auc = float(np.trapz(g, t))
    p_auc = float(np.trapz(np.maximum(delta, 0), t))
    n_auc = float(np.trapz(np.minimum(delta, 0), t))
    i_auc = float(np.trapz(delta, t))
    curve_size = float(np.trapz(np.abs(delta), t))
    cv = float(np.std(g) / max(np.mean(g), 1e-8))
    i_peak = int(np.argmax(g))
    t_b2p = float(max(t[i_peak] - t[0], 1.0))
    s_b2p = float((g_peak - g0) / t_b2p)
    t_p2e = float(max(t[-1] - t[i_peak], 1.0))
    s_p2e = float((g[-1] - g_peak) / t_p2e)
    return {
        "G_0": g0, "G_60": g60, "G_120": g120, "G_180": g180, "G_Peak": g_peak,
        "CurveSize": curve_size, "AUC": auc, "pAUC": p_auc, "nAUC": n_auc, "iAUC": i_auc,
        "CV": cv, "T_baseline2peak": t_b2p, "S_baseline2peak": s_b2p, "S_peak2end": s_p2e,
    }


def _primary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
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


def _bootstrap_ci_metric(y: np.ndarray, p: np.ndarray, metric: str, n_boot: int = 4000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 5:
        return np.nan, np.nan, np.nan

    def calc(yy, pp):
        if metric == "spearman":
            return float(stats.spearmanr(yy, pp)[0])
        if metric == "r2":
            return float(r2_score(yy, pp))
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(yy, pp)))
        if metric == "mae":
            return float(mean_absolute_error(yy, pp))
        raise ValueError(metric)

    point = calc(y, p)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        vals.append(calc(y[b], p[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5]).tolist()
    return point, float(lo), float(hi)


def _paired_delta_bootstrap(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, metric: str, n_boot: int = 4000, seed: int = 123) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p_a) & np.isfinite(p_b)
    y, p_a, p_b = y[ok], p_a[ok], p_b[ok]
    if len(y) < 5:
        return {"delta": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    def calc(yy, pp):
        if metric == "spearman":
            return float(stats.spearmanr(yy, pp)[0])
        if metric == "r2":
            return float(r2_score(yy, pp))
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(yy, pp)))
        if metric == "mae":
            return float(mean_absolute_error(yy, pp))
        raise ValueError(metric)

    delta = calc(y, p_a) - calc(y, p_b)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        vals.append(calc(y[b], p_a[b]) - calc(y[b], p_b[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5]).tolist()
    return {"delta": float(delta), "ci_lo": float(lo), "ci_hi": float(hi)}


def _ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y).astype(int)
    ok = np.isfinite(probs) & np.isfinite(y)
    probs, y = probs[ok], y[ok]
    if len(probs) < 5:
        return np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = y[mask].mean()
        ece_val += (mask.sum() / len(probs)) * abs(acc - conf)
    return float(ece_val)


class Encoder26:
    def __init__(self, ckpt_path: str):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.ck = ck
        self.model = MechanisticAutoencoder(
            meal_size=6, demographics_size=3, embedding_size=8, hidden_size=32, num_layers=2,
            encoder_dropout_prob=0.0, decoder_dropout_prob=0.5
        )
        self.model.load_state_dict(ck["model_state"], strict=False)
        self.model.eval()

    def encode_rows(self, rows_df: pd.DataFrame) -> pd.DataFrame:
        tm, tsd = self.ck["train_mean"], self.ck["train_std"]
        out = []
        for _, r in rows_df.iterrows():
            c = r["curve"][None, :, None].astype(np.float32)
            ts = r["timestamps"][None, :, None].astype(np.float32)
            meals = r["meal_series"][None, :, :].astype(np.float32)
            demo = r["demographics"][None, :].astype(np.float32)
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


def _build_d4_windows() -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "subjects.csv"))
    meals = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "meals.csv"))
    cgm = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "cgm.csv"))
    labels = _norm_labels(pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "labels.csv")))
    label_df = labels[["subject_id", "sspg", "di", "fasting_insulin"]].drop_duplicates("subject_id")

    id_map: Dict[str, str] = {}
    for _, r in subjects.dropna(subset=["subject_id", "original_id"]).iterrows():
        id_map[str(r["original_id"]).strip()] = str(r["subject_id"]).strip()
        id_map[f"D4_{str(r['original_id']).strip()}"] = str(r["subject_id"]).strip()

    meals["timestamp"] = pd.to_datetime(meals["timestamp"], errors="coerce")
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")
    if "glucose_mg_dl" in cgm.columns and "glucose_mgdl" not in cgm.columns:
        cgm = cgm.rename(columns={"glucose_mg_dl": "glucose_mgdl"})

    grid = np.arange(-30, 181, 5, dtype=np.float64)
    rows = []
    for _, meal in meals[meals["meal_type"].isin(STD_MEALS)].iterrows():
        sid_raw = str(meal["subject_id"])
        sid = id_map.get(sid_raw, sid_raw)
        t0 = meal["timestamp"]
        if pd.isna(t0):
            continue
        g = cgm[
            (cgm["subject_id"] == sid_raw)
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

        meal_series = np.zeros((len(grid), 6), dtype=np.float32)
        carb = float(pd.to_numeric(meal.get("carb_g", 0.0), errors="coerce") or 0.0)
        fat = float(pd.to_numeric(meal.get("fat_g", 0.0), errors="coerce") or 0.0)
        protein = float(pd.to_numeric(meal.get("protein_g", 0.0), errors="coerce") or 0.0)
        fiber = float(pd.to_numeric(meal.get("fiber_g", 0.0), errors="coerce") or 0.0)
        meal_series[:, 0] = carb + fat + protein
        meal_series[:, 1] = carb
        meal_series[:, 3] = fiber
        meal_series[:, 4] = fat
        meal_series[:, 5] = protein

        srow = subjects[subjects["subject_id"].astype(str) == sid]
        if srow.empty:
            demo = np.array([0.0, 40.0, 72.0], dtype=np.float32)
        else:
            s = srow.iloc[0]
            gender = 1.0 if str(s.get("sex", "M")).upper().startswith("F") else 0.0
            age = float(pd.to_numeric(s.get("age", 40.0), errors="coerce") or 40.0)
            weight = float(pd.to_numeric(s.get("weight_kg", np.nan), errors="coerce"))
            if not np.isfinite(weight):
                bmi = float(pd.to_numeric(s.get("bmi", 25.0), errors="coerce") or 25.0)
                weight = bmi * (1.7 ** 2)
            demo = np.array([gender, age, weight], dtype=np.float32)

        feat = _met14_features(y_new, grid)
        cv_post = float(np.std(y_new[grid >= 0]) / max(np.mean(y_new[grid >= 0]), 1e-8))
        iauc_abs = float(np.trapz(np.abs(y_new[grid >= 0] - np.interp(0, grid, y_new)), grid[grid >= 0]))
        rows.append(
            {
                "subject_id": sid,
                "meal_type": str(meal["meal_type"]),
                "curve": y_new,
                "timestamps": grid.astype(np.float32),
                "meal_series": meal_series,
                "demographics": demo,
                "carb_g": carb,
                "fat_g": fat,
                "protein_g": protein,
                "fiber_g": fiber,
                "uncertainty_score": cv_post + iauc_abs / 1000.0,
                **feat,
            }
        )
    return pd.DataFrame(rows), label_df


def _build_train_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # latent train from Exp8
    lat = pd.read_csv(os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase2_finetune_head", "latent_and_gold_all_26d.csv"))
    lat = _norm_labels(lat)
    rename_map = {
        "tau_m": "z00", "Gb": "z01", "sg": "z02", "si": "z03", "p2": "z04", "mi": "z05",
        "z_init_0": "z06", "z_init_1": "z07", "z_init_2": "z08", "z_init_3": "z09",
    }
    for k, v in rename_map.items():
        if k in lat.columns and v not in lat.columns:
            lat[v] = lat[k]
    for i in range(16):
        src = f"z_nonseq_{i}"
        dst = f"z{10+i:02d}"
        if src in lat.columns and dst not in lat.columns:
            lat[dst] = lat[src]
    latent_train = lat.dropna(subset=[f"z{i:02d}" for i in range(26)] + ["sspg", "di"]).copy()

    # met train from D1+D2
    rows = []
    for ds in ["D1", "D2"]:
        batch, info, labels = load_cgm_project_level1_level2(dataset_id=ds, output_base=DATA_ROOT)
        labels = _norm_labels(labels)
        lab = labels.dropna(subset=["subject_id"]).drop_duplicates("subject_id").set_index("subject_id")
        for i, sid in enumerate(info.patient_ids):
            sid = str(sid)
            feat = _met14_features(batch.cgm[i, :, 0], batch.timestamps[i, :, 0])
            feat["subject_id"] = sid
            feat["sspg"] = float(lab["sspg"].get(sid, np.nan))
            feat["di"] = float(lab["di"].get(sid, np.nan))
            rows.append(feat)
    met_train = pd.DataFrame(rows).groupby("subject_id", as_index=False).median(numeric_only=True)
    return latent_train, met_train


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    latent_train, met_train = _build_train_tables()
    d4_rows, d4_labels = _build_d4_windows()
    d4_label_map = d4_labels.set_index("subject_id")

    # encode D4 latents with Exp8 encoder
    enc = Encoder26(os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase2_finetune_head", "autoencoder_p1_full.pt"))
    d4_lat = enc.encode_rows(d4_rows)
    met_cols = ["G_0", "G_60", "G_120", "G_180", "G_Peak", "CurveSize", "AUC", "pAUC", "nAUC", "iAUC", "CV", "T_baseline2peak", "S_baseline2peak", "S_peak2end"]
    keep_cols = ["subject_id", "meal_type", "uncertainty_score", "carb_g", "fat_g", "protein_g", "fiber_g"] + met_cols
    d4_full = d4_rows[keep_cols].merge(d4_lat, on=["subject_id", "meal_type"], how="inner")
    d4_full["sspg_true"] = d4_full["subject_id"].map(d4_label_map["sspg"])
    d4_full["di_true"] = d4_full["subject_id"].map(d4_label_map["di"])

    # train locked models
    z10 = [f"z{i:02d}" for i in range(10)]
    z26 = [f"z{i:02d}" for i in range(26)]
    tr_l = latent_train.groupby("subject_id", as_index=False).median(numeric_only=True)
    X10 = tr_l[z10].to_numpy(float)
    X26 = tr_l[z26].to_numpy(float)
    y_s = tr_l["sspg"].to_numpy(float)
    y_d = tr_l["di"].to_numpy(float)

    m10_s = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(X10, y_s)
    m10_d = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(X10, y_d)
    m26_s = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(X26, y_s)
    m26_d = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(X26, y_d)

    tr_m = met_train.dropna(subset=met_cols + ["sspg", "di"]).copy()
    mm_s = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(tr_m[met_cols].to_numpy(float), tr_m["sspg"].to_numpy(float))
    mm_d = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(tr_m[met_cols].to_numpy(float), tr_m["di"].to_numpy(float))

    # predict D4 meal-level
    d4_full["p10_sspg"] = m10_s.predict(d4_full[z10].to_numpy(float))
    d4_full["p10_di"] = m10_d.predict(d4_full[z10].to_numpy(float))
    d4_full["p26_sspg"] = m26_s.predict(d4_full[z26].to_numpy(float))
    d4_full["p26_di"] = m26_d.predict(d4_full[z26].to_numpy(float))
    d4_full["pmet_sspg"] = mm_s.predict(d4_full[met_cols].to_numpy(float))
    d4_full["pmet_di"] = mm_d.predict(d4_full[met_cols].to_numpy(float))
    d4_full.to_csv(os.path.join(OUT_ROOT, "v22_d4_meal_level_predictions.csv"), index=False)

    # aggregate subject-level for locked protocol
    sub = d4_full.groupby("subject_id", as_index=False).mean(numeric_only=True)
    sub["sspg_true"] = sub["subject_id"].map(d4_label_map["sspg"])
    sub["di_true"] = sub["subject_id"].map(d4_label_map["di"])
    sub.to_csv(os.path.join(OUT_ROOT, "v22_d4_subject_level_predictions.csv"), index=False)

    model_map = {
        "Ridge10D": ("p10_sspg", "p10_di"),
        "Ridge26D": ("p26_sspg", "p26_di"),
        "Metwally14_Ridge": ("pmet_sspg", "pmet_di"),
    }

    # primary endpoints with CI
    primary_rows = []
    for model, (cs, cd) in model_map.items():
        for tgt, y_col, p_col in [("sspg", "sspg_true", cs), ("di", "di_true", cd)]:
            y = sub[y_col].to_numpy(float)
            p = sub[p_col].to_numpy(float)
            m = _primary_metrics(y, p)
            sp = _bootstrap_ci_metric(y, p, "spearman", n_boot=3000, seed=SEED)
            r2 = _bootstrap_ci_metric(y, p, "r2", n_boot=3000, seed=SEED + 1)
            rm = _bootstrap_ci_metric(y, p, "rmse", n_boot=3000, seed=SEED + 2)
            primary_rows.append(
                {
                    "model": model,
                    "target": tgt,
                    "n": m["n"],
                    "spearman": m["spearman"],
                    "spearman_ci_lo": sp[1],
                    "spearman_ci_hi": sp[2],
                    "r2": m["r2"],
                    "r2_ci_lo": r2[1],
                    "r2_ci_hi": r2[2],
                    "rmse": m["rmse"],
                    "rmse_ci_lo": rm[1],
                    "rmse_ci_hi": rm[2],
                    "mae": m["mae"],
                }
            )
    primary_df = pd.DataFrame(primary_rows)
    primary_df.to_csv(os.path.join(OUT_ROOT, "v22_primary_endpoints_locked.csv"), index=False)

    # paired deltas
    delta_rows = []
    for metric in ["spearman", "r2", "rmse", "mae"]:
        for tgt, y_col in [("sspg", "sspg_true"), ("di", "di_true")]:
            y = sub[y_col].to_numpy(float)
            d_26_10 = _paired_delta_bootstrap(y, sub[f"p26_{tgt}"].to_numpy(float), sub[f"p10_{tgt}"].to_numpy(float), metric, n_boot=3000, seed=11)
            d_26_met = _paired_delta_bootstrap(y, sub[f"p26_{tgt}"].to_numpy(float), sub[f"pmet_{tgt}"].to_numpy(float), metric, n_boot=3000, seed=13)
            delta_rows.append({"target": tgt, "metric": metric, "comparison": "Ridge26D - Ridge10D", **d_26_10})
            delta_rows.append({"target": tgt, "metric": metric, "comparison": "Ridge26D - Metwally14", **d_26_met})
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(os.path.join(OUT_ROOT, "v22_paired_bootstrap_deltas.csv"), index=False)

    # secondary clinical endpoints (IR and decomp)
    sec_rows = []
    for model, (cs, cd) in model_map.items():
        # train-level calibration
        if model.startswith("Ridge10"):
            s_tr = m10_s.predict(X10)
            d_tr = m10_d.predict(X10)
        elif model.startswith("Ridge26"):
            s_tr = m26_s.predict(X26)
            d_tr = m26_d.predict(X26)
        else:
            s_tr = mm_s.predict(tr_m[met_cols].to_numpy(float))
            d_tr = mm_d.predict(tr_m[met_cols].to_numpy(float))

        y_ir_tr = (tr_l["sspg"].to_numpy(float) >= 120.0).astype(int) if model != "Metwally14_Ridge" else (tr_m["sspg"].to_numpy(float) >= 120.0).astype(int)
        y_de_tr = ((tr_l["sspg"].to_numpy(float) >= 120.0) & (tr_l["di"].to_numpy(float) < 1.0)).astype(int) if model != "Metwally14_Ridge" else ((tr_m["sspg"].to_numpy(float) >= 120.0) & (tr_m["di"].to_numpy(float) < 1.0)).astype(int)
        score_tr_de = stats.zscore(s_tr, nan_policy="omit") - stats.zscore(d_tr, nan_policy="omit")

        clf_ir = LogisticRegression(max_iter=2000).fit(s_tr.reshape(-1, 1), y_ir_tr)
        clf_de = LogisticRegression(max_iter=2000).fit(score_tr_de.reshape(-1, 1), y_de_tr)

        y_ir = (sub["sspg_true"].to_numpy(float) >= 120.0).astype(int)
        y_de = ((sub["sspg_true"].to_numpy(float) >= 120.0) & (sub["di_true"].to_numpy(float) < 1.0)).astype(int)
        s_te = sub[cs].to_numpy(float)
        score_te_de = stats.zscore(sub[cs].to_numpy(float), nan_policy="omit") - stats.zscore(sub[cd].to_numpy(float), nan_policy="omit")
        p_ir = clf_ir.predict_proba(s_te.reshape(-1, 1))[:, 1]
        p_de = clf_de.predict_proba(score_te_de.reshape(-1, 1))[:, 1]

        ir_auc = roc_auc_score(y_ir, s_te) if len(np.unique(y_ir)) > 1 else np.nan
        de_auc = roc_auc_score(y_de, score_te_de) if len(np.unique(y_de)) > 1 else np.nan
        ir_auprc = average_precision_score(y_ir, s_te) if len(np.unique(y_ir)) > 1 else np.nan
        de_auprc = average_precision_score(y_de, score_te_de) if len(np.unique(y_de)) > 1 else np.nan
        ir_brier = float(np.mean((p_ir - y_ir) ** 2))
        de_brier = float(np.mean((p_de - y_de) ** 2))
        ir_ece = _ece(p_ir, y_ir, n_bins=8)
        de_ece = _ece(p_de, y_de, n_bins=8)
        sec_rows.append(
            {
                "model": model,
                "ir_auc": ir_auc,
                "ir_auprc": ir_auprc,
                "ir_brier": ir_brier,
                "ir_ece": ir_ece,
                "decomp_auc": de_auc,
                "decomp_auprc": de_auprc,
                "decomp_brier": de_brier,
                "decomp_ece": de_ece,
            }
        )
    sec_df = pd.DataFrame(sec_rows)
    sec_df.to_csv(os.path.join(OUT_ROOT, "v22_secondary_clinical_endpoints.csv"), index=False)

    # 16D conditional utility analysis on meal windows
    d4_full["err10_sspg"] = np.abs(d4_full["p10_sspg"] - d4_full["sspg_true"])
    d4_full["err26_sspg"] = np.abs(d4_full["p26_sspg"] - d4_full["sspg_true"])
    d4_full["win26_sspg"] = (d4_full["err26_sspg"] < d4_full["err10_sspg"]).astype(int)
    q = d4_full["uncertainty_score"].quantile([0.33, 0.67]).to_list()
    d4_full["unc_bin"] = "mid"
    d4_full.loc[d4_full["uncertainty_score"] <= q[0], "unc_bin"] = "low"
    d4_full.loc[d4_full["uncertainty_score"] >= q[1], "unc_bin"] = "high"

    cond_rows = []
    for b in ["low", "mid", "high"]:
        dm = d4_full[d4_full["unc_bin"] == b]
        cond_rows.append(
            {
                "unc_bin": b,
                "n_meals": int(len(dm)),
                "win26_rate_sspg": float(dm["win26_sspg"].mean()) if len(dm) else np.nan,
                "mean_err10_sspg": float(dm["err10_sspg"].mean()) if len(dm) else np.nan,
                "mean_err26_sspg": float(dm["err26_sspg"].mean()) if len(dm) else np.nan,
            }
        )
    cond_df = pd.DataFrame(cond_rows)
    cond_df.to_csv(os.path.join(OUT_ROOT, "v22_16d_conditional_utility.csv"), index=False)

    # beyond-metwally evidence: can 16D explain Met residuals on D4?
    sub_res = sub.copy()
    sub_res["met_res_sspg"] = sub_res["sspg_true"] - sub_res["pmet_sspg"]
    z16_cols = [f"z{i:02d}" for i in range(10, 26)]
    sub_res = sub_res.dropna(subset=z16_cols + ["met_res_sspg"]).copy()
    X16 = sub_res[z16_cols].to_numpy(float)
    yres = sub_res["met_res_sspg"].to_numpy(float)
    loo = LeaveOneOut()
    pred_res = np.full_like(yres, np.nan, dtype=float)
    for tr_idx, te_idx in loo.split(X16):
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60)))
        mdl.fit(X16[tr_idx], yres[tr_idx])
        pred_res[te_idx] = mdl.predict(X16[te_idx])
    res_ev = {
        "n": int(len(yres)),
        "residual_spearman_pred_vs_true": float(stats.spearmanr(yres, pred_res)[0]),
        "residual_r2_cv": float(r2_score(yres, pred_res)),
        "residual_rmse_cv": float(np.sqrt(mean_squared_error(yres, pred_res))),
    }
    pd.DataFrame({"subject_id": sub_res["subject_id"], "met_res_sspg_true": yres, "met_res_sspg_pred_from16d": pred_res}).to_csv(
        os.path.join(OUT_ROOT, "v22_beyond_metwally_residual_test.csv"), index=False
    )
    with open(os.path.join(OUT_ROOT, "v22_beyond_metwally_summary.json"), "w") as f:
        json.dump(res_ev, f, indent=2)

    # report
    with open(os.path.join(OUT_ROOT, "v22_locked_protocol_report.md"), "w", encoding="utf-8") as f:
        f.write("# v22 Locked Protocol + 16D Conditional Utility\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Primary Endpoints (locked)\n\n")
        f.write(primary_df.to_markdown(index=False))
        f.write("\n\n## Paired Bootstrap Deltas\n\n")
        f.write(delta_df.to_markdown(index=False))
        f.write("\n\n## Secondary Clinical Endpoints\n\n")
        f.write(sec_df.to_markdown(index=False))
        f.write("\n\n## 16D Conditional Utility (meal-level)\n\n")
        f.write(cond_df.to_markdown(index=False))
        f.write("\n\n## Beyond-Metwally Residual Explainability (16D)\n\n")
        f.write(json.dumps(res_ev, indent=2))
        f.write("\n")

    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    main()
