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
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import MechanisticAutoencoder

DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
V18_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v18")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v24_semantic_ablation")
SEED = 42
STD_MEALS_D4 = ["Cornflakes", "PB_sandwich", "Protein_bar"]
MAJOR_D3_MEALS = ["Breakfast", "Lunch", "Dinner", "Snacks"]


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


def _build_d4_rows() -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "subjects.csv"))
    meals = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "meals.csv"))
    cgm = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "cgm.csv"))
    labels = _norm_labels(pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "labels.csv")))
    label_df = labels[["subject_id", "sspg", "di"]].drop_duplicates("subject_id")

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
    for _, meal in meals[meals["meal_type"].isin(STD_MEALS_D4)].iterrows():
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

        carb = float(pd.to_numeric(meal.get("carb_g", 0.0), errors="coerce") or 0.0)
        fat = float(pd.to_numeric(meal.get("fat_g", 0.0), errors="coerce") or 0.0)
        protein = float(pd.to_numeric(meal.get("protein_g", 0.0), errors="coerce") or 0.0)
        fiber = float(pd.to_numeric(meal.get("fiber_g", 0.0), errors="coerce") or 0.0)
        cv_post = float(np.std(y_new[grid >= 0]) / max(np.mean(y_new[grid >= 0]), 1e-8))
        iauc_abs = float(np.trapz(np.abs(y_new[grid >= 0] - np.interp(0, grid, y_new)), grid[grid >= 0]))
        rows.append(
            {
                "subject_id": sid,
                "meal_type": str(meal["meal_type"]),
                "curve": y_new,
                "timestamps": grid.astype(np.float32),
                "carb_g": carb,
                "fat_g": fat,
                "protein_g": protein,
                "fiber_g": fiber,
                "uncertainty_score": cv_post + iauc_abs / 1000.0,
                **_met14_features(y_new, grid),
            }
        )
    return pd.DataFrame(rows), label_df


def _build_d3_rows() -> pd.DataFrame:
    subjects = pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "subjects.csv"))
    meals = pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "meals.csv"))
    cgm = pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "cgm.csv"))
    labels = _norm_labels(pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "labels.csv")))
    lab_map = labels.set_index("subject_id")

    meals["timestamp"] = pd.to_datetime(meals["timestamp"], errors="coerce")
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")
    if "glucose_mg_dl" in cgm.columns and "glucose_mgdl" not in cgm.columns:
        cgm = cgm.rename(columns={"glucose_mg_dl": "glucose_mgdl"})

    grid = np.arange(-30, 181, 5, dtype=np.float64)
    rows = []
    for _, meal in meals[meals["meal_type"].isin(MAJOR_D3_MEALS)].iterrows():
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

        carb = float(pd.to_numeric(meal.get("carb_g", 0.0), errors="coerce") or 0.0)
        fat = float(pd.to_numeric(meal.get("fat_g", 0.0), errors="coerce") or 0.0)
        protein = float(pd.to_numeric(meal.get("protein_g", 0.0), errors="coerce") or 0.0)
        fiber = float(pd.to_numeric(meal.get("fiber_g", 0.0), errors="coerce") or 0.0)
        cv_post = float(np.std(y_new[grid >= 0]) / max(np.mean(y_new[grid >= 0]), 1e-8))
        iauc_abs = float(np.trapz(np.abs(y_new[grid >= 0] - np.interp(0, grid, y_new)), grid[grid >= 0]))
        rows.append(
            {
                "subject_id": sid,
                "meal_type": str(meal["meal_type"]),
                "curve": y_new,
                "timestamps": grid.astype(np.float32),
                "carb_g": carb,
                "fat_g": fat,
                "protein_g": protein,
                "fiber_g": fiber,
                "uncertainty_score": cv_post + iauc_abs / 1000.0,
                "hba1c": float(lab_map.loc[sid, "hba1c"]) if sid in lab_map.index and "hba1c" in lab_map.columns else np.nan,
                "homa_ir": float(lab_map.loc[sid, "homa_ir"]) if sid in lab_map.index and "homa_ir" in lab_map.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


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

    def encode(self, rows: pd.DataFrame) -> pd.DataFrame:
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


def _loo_regression(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) < 10:
        return {"n": int(len(y)), "spearman": np.nan, "r2": np.nan, "rmse": np.nan}
    loo = LeaveOneOut()
    pred = np.full_like(y, np.nan, dtype=float)
    for tr, te in loo.split(X):
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 40)))
        mdl.fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te])
    return {
        "n": int(len(y)),
        "spearman": float(stats.spearmanr(y, pred)[0]),
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
    }


def _paired_bootstrap_err_diff(e10: np.ndarray, e26: np.ndarray, n_boot: int = 4000, seed: int = 42) -> Dict[str, float]:
    e10 = np.asarray(e10, dtype=float)
    e26 = np.asarray(e26, dtype=float)
    ok = np.isfinite(e10) & np.isfinite(e26)
    e10, e26 = e10[ok], e26[ok]
    if len(e10) < 8:
        return {"delta_mean_err_26_minus_10": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(e10))
    vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        vals.append(float(np.mean(e26[b] - e10[b])))
    lo, hi = np.percentile(vals, [2.5, 97.5]).tolist()
    return {"delta_mean_err_26_minus_10": float(np.mean(e26 - e10)), "ci_lo": float(lo), "ci_hi": float(hi)}


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # data
    d4_rows, d4_labels = _build_d4_rows()
    d3_rows = _build_d3_rows()
    enc = Encoder()
    d4_lat = enc.encode(d4_rows)
    d3_lat = enc.encode(d3_rows)
    d4 = d4_rows[["subject_id", "meal_type", "uncertainty_score", "carb_g", "fat_g", "protein_g", "fiber_g"]].merge(d4_lat, on=["subject_id", "meal_type"], how="inner")
    d3 = d3_rows[["subject_id", "meal_type", "uncertainty_score", "carb_g", "fat_g", "protein_g", "fiber_g", "hba1c", "homa_ir"]].merge(d3_lat, on=["subject_id", "meal_type"], how="inner")
    d4 = d4.merge(d4_labels[["subject_id", "sspg", "di"]], on="subject_id", how="left")
    d4.rename(columns={"sspg": "sspg_true", "di": "di_true"}, inplace=True)

    # train latent from D1+D2 (exp8)
    tr = pd.read_csv(os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase2_finetune_head", "latent_and_gold_all_26d.csv"))
    tr = _norm_labels(tr)
    rename_map = {"tau_m": "z00", "Gb": "z01", "sg": "z02", "si": "z03", "p2": "z04", "mi": "z05", "z_init_0": "z06", "z_init_1": "z07", "z_init_2": "z08", "z_init_3": "z09"}
    for k, v in rename_map.items():
        if k in tr.columns and v not in tr.columns:
            tr[v] = tr[k]
    for i in range(16):
        s, d = f"z_nonseq_{i}", f"z{10+i:02d}"
        if s in tr.columns and d not in tr.columns:
            tr[d] = tr[s]
    tr = tr.dropna(subset=[f"z{i:02d}" for i in range(26)] + ["sspg", "di"]).copy()
    tr_sub = tr.groupby("subject_id", as_index=False).median(numeric_only=True)

    z10 = [f"z{i:02d}" for i in range(10)]
    z16 = [f"z{i:02d}" for i in range(10, 26)]
    z26 = [f"z{i:02d}" for i in range(26)]

    # Q1: 26D meaning via unique information (LOO)
    info_rows = []
    for tgt in ["sspg", "di"]:
        y = tr_sub[tgt].to_numpy(float)
        m10 = _loo_regression(tr_sub[z10].to_numpy(float), y)
        m16 = _loo_regression(tr_sub[z16].to_numpy(float), y)
        m26 = _loo_regression(tr_sub[z26].to_numpy(float), y)
        info_rows.append({"target": tgt, "feature_set": "10D", **m10})
        info_rows.append({"target": tgt, "feature_set": "16D", **m16})
        info_rows.append({"target": tgt, "feature_set": "26D", **m26})
        # residual from 10D explained by 16D
        mdl10 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 40))).fit(tr_sub[z10].to_numpy(float), y)
        res10 = y - mdl10.predict(tr_sub[z10].to_numpy(float))
        m_res = _loo_regression(tr_sub[z16].to_numpy(float), res10)
        info_rows.append({"target": tgt, "feature_set": "16D_on_10D_residual", **m_res})
    info_df = pd.DataFrame(info_rows)
    info_df.to_csv(os.path.join(OUT_ROOT, "v24_information_decomposition_loo.csv"), index=False)

    # fit 10D/26D models for D4 evaluation
    m10_s = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(tr_sub[z10].to_numpy(float), tr_sub["sspg"].to_numpy(float))
    m26_s = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(tr_sub[z26].to_numpy(float), tr_sub["sspg"].to_numpy(float))
    m10_d = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(tr_sub[z10].to_numpy(float), tr_sub["di"].to_numpy(float))
    m26_d = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(tr_sub[z26].to_numpy(float), tr_sub["di"].to_numpy(float))

    d4["p10_sspg"] = m10_s.predict(d4[z10].to_numpy(float))
    d4["p26_sspg"] = m26_s.predict(d4[z26].to_numpy(float))
    d4["p10_di"] = m10_d.predict(d4[z10].to_numpy(float))
    d4["p26_di"] = m26_d.predict(d4[z26].to_numpy(float))

    # Q2: what 16D adds = context encoding tests on D3
    ctx_rows = []
    gkf = GroupKFold(n_splits=5)
    X10 = d3[z10].to_numpy(float)
    X16 = d3[z16].to_numpy(float)
    X26 = d3[z26].to_numpy(float)
    groups = d3["subject_id"].astype(str).to_numpy()
    y_meal = d3["meal_type"].astype(str).to_numpy()

    for feat_name, X in [("10D", X10), ("16D", X16), ("26D", X26)]:
        pred = np.array([""] * len(y_meal), dtype=object)
        for tr_idx, te_idx in gkf.split(X, y_meal, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, multi_class="multinomial"))
            clf.fit(X[tr_idx], y_meal[tr_idx])
            pred[te_idx] = clf.predict(X[te_idx])
        ctx_rows.append(
            {
                "task": "meal_type_classification",
                "feature_set": feat_name,
                "n": int(len(y_meal)),
                "accuracy": float(accuracy_score(y_meal, pred)),
                "macro_f1": float(f1_score(y_meal, pred, average="macro")),
            }
        )

    for target in ["carb_g", "fat_g", "protein_g"]:
        y = d3[target].to_numpy(float)
        for feat_name, X in [("10D", X10), ("16D", X16), ("26D", X26)]:
            p = np.full_like(y, np.nan, dtype=float)
            for tr_idx, te_idx in gkf.split(X, y, groups):
                mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 40)))
                mdl.fit(X[tr_idx], y[tr_idx])
                p[te_idx] = mdl.predict(X[te_idx])
            ok = np.isfinite(y) & np.isfinite(p)
            ctx_rows.append(
                {
                    "task": f"{target}_regression",
                    "feature_set": feat_name,
                    "n": int(ok.sum()),
                    "r2": float(r2_score(y[ok], p[ok])),
                    "spearman": float(stats.spearmanr(y[ok], p[ok])[0]),
                }
            )
    ctx_df = pd.DataFrame(ctx_rows)
    ctx_df.to_csv(os.path.join(OUT_ROOT, "v24_16d_context_encoding_tests.csv"), index=False)

    # Q3: complex scenarios utility (D4 uncertainty strata)
    q1, q2 = d4["uncertainty_score"].quantile([0.33, 0.67]).tolist()
    d4["unc_bin"] = "mid"
    d4.loc[d4["uncertainty_score"] <= q1, "unc_bin"] = "low"
    d4.loc[d4["uncertainty_score"] >= q2, "unc_bin"] = "high"
    d4["e10_sspg"] = np.abs(d4["p10_sspg"] - d4["sspg_true"])
    d4["e26_sspg"] = np.abs(d4["p26_sspg"] - d4["sspg_true"])
    d4["e10_di"] = np.abs(d4["p10_di"] - d4["di_true"])
    d4["e26_di"] = np.abs(d4["p26_di"] - d4["di_true"])

    stress_rows = []
    for b in ["low", "mid", "high"]:
        dm = d4[d4["unc_bin"] == b]
        if len(dm) == 0:
            continue
        d_ss = _paired_bootstrap_err_diff(dm["e10_sspg"].to_numpy(float), dm["e26_sspg"].to_numpy(float), n_boot=3000, seed=11)
        d_di = _paired_bootstrap_err_diff(dm["e10_di"].to_numpy(float), dm["e26_di"].to_numpy(float), n_boot=3000, seed=13)
        stress_rows.append(
            {
                "unc_bin": b,
                "n_meals": int(len(dm)),
                "mean_err10_sspg": float(dm["e10_sspg"].mean()),
                "mean_err26_sspg": float(dm["e26_sspg"].mean()),
                "delta_err26_minus10_sspg": d_ss["delta_mean_err_26_minus_10"],
                "delta_ci_lo_sspg": d_ss["ci_lo"],
                "delta_ci_hi_sspg": d_ss["ci_hi"],
                "mean_err10_di": float(dm["e10_di"].mean()),
                "mean_err26_di": float(dm["e26_di"].mean()),
                "delta_err26_minus10_di": d_di["delta_mean_err_26_minus_10"],
                "delta_ci_lo_di": d_di["ci_lo"],
                "delta_ci_hi_di": d_di["ci_hi"],
            }
        )
    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(os.path.join(OUT_ROOT, "v24_complex_scenario_stress_test.csv"), index=False)

    # Linear variance decomposition of 26D head contributions
    # using fitted Ridge26D models (coef after StandardScaler not directly accessible in pipeline);
    # approximate by fitting plain RidgeCV on standardized features for interpretability.
    sc = StandardScaler().fit(tr_sub[z26].to_numpy(float))
    Xtr_z = sc.transform(tr_sub[z26].to_numpy(float))
    ridge_s = RidgeCV(alphas=np.logspace(-3, 3, 50)).fit(Xtr_z, tr_sub["sspg"].to_numpy(float))
    ridge_d = RidgeCV(alphas=np.logspace(-3, 3, 50)).fit(Xtr_z, tr_sub["di"].to_numpy(float))
    Xd4_z = sc.transform(d4[z26].to_numpy(float))

    comp_rows = []
    for name, coef in [("sspg", ridge_s.coef_), ("di", ridge_d.coef_)]:
        c10 = Xd4_z[:, :10] @ coef[:10]
        c16 = Xd4_z[:, 10:] @ coef[10:]
        total = c10 + c16
        share16 = float(np.var(c16) / max(np.var(total), 1e-8))
        corr_unc = float(stats.spearmanr(np.abs(c16), d4["uncertainty_score"].to_numpy(float))[0])
        comp_rows.append({"target": name, "var_share_16d_component": share16, "spearman_abs16_vs_uncertainty": corr_unc})
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(os.path.join(OUT_ROOT, "v24_26d_component_variance_decomposition.csv"), index=False)

    # report
    with open(os.path.join(OUT_ROOT, "v24_semantic_ablation_report.md"), "w", encoding="utf-8") as f:
        f.write("# v24 Semantic Ablation: Meaning of 26D / 16D / Complex Scenarios\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Q1: 26D unique information over 10D\n\n")
        f.write(info_df.to_markdown(index=False))
        f.write("\n\n## Q2: What 16D adds (context encoding tests)\n\n")
        f.write(ctx_df.to_markdown(index=False))
        f.write("\n\n## Q3: Utility under complex scenarios (D4 uncertainty strata)\n\n")
        f.write(stress_df.to_markdown(index=False))
        f.write("\n\n## 26D component variance decomposition\n\n")
        f.write(comp_df.to_markdown(index=False))
        f.write("\n")

    summary = {
        "q1_26d_over_10d_sspg_spearman_delta": float(
            info_df[(info_df["target"] == "sspg") & (info_df["feature_set"] == "26D")]["spearman"].iloc[0]
            - info_df[(info_df["target"] == "sspg") & (info_df["feature_set"] == "10D")]["spearman"].iloc[0]
        ),
        "q2_mealtype_macrof1_16d_minus_10d": float(
            ctx_df[(ctx_df["task"] == "meal_type_classification") & (ctx_df["feature_set"] == "16D")]["macro_f1"].iloc[0]
            - ctx_df[(ctx_df["task"] == "meal_type_classification") & (ctx_df["feature_set"] == "10D")]["macro_f1"].iloc[0]
        ),
        "q3_high_uncertainty_delta_err26_minus10_sspg": float(
            stress_df[stress_df["unc_bin"] == "high"]["delta_err26_minus10_sspg"].iloc[0]
            if "high" in set(stress_df["unc_bin"]) else np.nan
        ),
    }
    with open(os.path.join(OUT_ROOT, "v24_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    main()
