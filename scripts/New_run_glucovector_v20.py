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
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from load_cgm_project_data import load_cgm_project_level1_level2
from models import MechanisticAutoencoder
from scripts.New_eval_trainD1D2_testD4 import train_on_d1d2

DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
V18_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v18")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v20")
STD_MEALS = ["Cornflakes", "PB_sandwich", "Protein_bar"]
SEED = 42


def _norm_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for old, new in [("SSPG", "sspg"), ("DI", "di"), ("HOMA_IR", "homa_ir")]:
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    if len(y) < 3:
        return {"n": int(len(y)), "pearson_r": np.nan, "spearman_r": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    return {
        "n": int(len(y)),
        "pearson_r": float(stats.pearsonr(y, p)[0]),
        "spearman_r": float(stats.spearmanr(y, p)[0]),
        "r2": float(r2_score(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
    }


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


def _met_features(cgm: np.ndarray, ts: np.ndarray) -> Dict[str, float]:
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


def _fit_healey(g_points: List[float], fasting_insulin: float) -> Tuple[float, float]:
    alpha = 10000.0
    t = np.array([0.0, 60.0, 120.0, 180.0], dtype=float)
    g = np.asarray(g_points, dtype=float)
    g0 = float(g[0])
    iss = float(max(fasting_insulin, 1e-3))

    def ode(_t, y, si, imax, k_sto, k_gut, eg0):
        q_sto, q_gut, gg, ii = y
        k_i = imax * (g0 ** 2) / (iss * (alpha + g0 ** 2) + 1e-8)
        r0 = (eg0 + si * iss) * g0
        return [-k_sto * q_sto, k_sto * q_sto - k_gut * q_gut, r0 - (eg0 + si * ii) * gg + k_gut * q_gut, imax * (gg ** 2) / (alpha + gg ** 2) - k_i * ii]

    def obj(x):
        si, imax, k_sto, k_gut, eg0, q0 = np.exp(x)
        try:
            sol = solve_ivp(lambda tt, yy: ode(tt, yy, si, imax, k_sto, k_gut, eg0), [0.0, 180.0], [q0, 0.0, g0, iss], t_eval=t)
            if not sol.success:
                return 1e9
            return float(np.mean((sol.y[2] - g) ** 2))
        except Exception:
            return 1e9

    x0 = np.log([1e-4, 0.5, 0.05, 0.05, 0.01, 30.0])
    lb = np.log([1e-6, 1e-3, 1e-3, 1e-3, 1e-4, 1.0])
    ub = np.log([1e-1, 50.0, 1.0, 1.0, 1.0, 500.0])
    r = minimize(obj, x0=x0, bounds=list(zip(lb, ub)), method="L-BFGS-B", options={"maxiter": 80})
    if not r.success:
        return np.nan, np.nan
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))


def _build_d4_windows() -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "subjects.csv"))
    meals = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "meals.csv"))
    cgm = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "cgm.csv"))
    labels = _norm_labels(pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "labels.csv")))
    label_df = labels[["subject_id", "sspg", "di", "fasting_insulin"]].drop_duplicates("subject_id")
    fi_map = label_df.set_index("subject_id")["fasting_insulin"].to_dict()

    id_map: Dict[str, str] = {}
    for _, r in subjects.dropna(subset=["subject_id", "original_id"]).iterrows():
        orig = str(r["original_id"]).strip()
        sid = str(r["subject_id"]).strip()
        id_map[orig] = sid
        id_map[f"D4_{orig}"] = sid

    meals["timestamp"] = pd.to_datetime(meals["timestamp"], errors="coerce")
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")
    if "glucose_mg_dl" in cgm.columns and "glucose_mgdl" not in cgm.columns:
        cgm = cgm.rename(columns={"glucose_mg_dl": "glucose_mgdl"})

    grid = np.arange(-30, 181, 5, dtype=np.float64)
    rows: List[Dict[str, object]] = []
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
        t = ((g["timestamp"] - t0).dt.total_seconds() / 60.0).to_numpy(dtype=float)
        y = pd.to_numeric(g["glucose_mgdl"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(t) & np.isfinite(y)
        if ok.sum() < 10:
            continue
        t = t[ok]
        y = y[ok]
        order = np.argsort(t)
        t = t[order]
        y = y[order]
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

        rows.append(
            {
                "subject_id": sid,
                "meal_type": str(meal["meal_type"]),
                "curve": y_new,
                "timestamps": grid.astype(np.float32),
                "meal_series": meal_series,
                "demographics": demo,
                "fasting_insulin": float(fi_map.get(sid, np.nan)),
                **_met_features(y, t),
            }
        )
    return pd.DataFrame(rows), label_df


def _build_train_table() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    fi_map: Dict[str, Dict[str, float]] = {"D1": {}, "D2": {}}
    d2 = _norm_labels(pd.read_csv(os.path.join(DATA_ROOT, "D2_stanford", "labels.csv")))
    for _, r in d2.dropna(subset=["subject_id", "fasting_insulin"]).iterrows():
        fi_map["D2"][str(r["subject_id"])] = float(r["fasting_insulin"])
    d1_labels = _norm_labels(pd.read_csv(os.path.join(DATA_ROOT, "D1_metwally", "labels.csv")))
    d1_subj = pd.read_csv(os.path.join(DATA_ROOT, "D1_metwally", "subjects.csv"))
    fpg_map: Dict[str, float] = {}
    for _, r in d1_subj.dropna(subset=["subject_id", "fpg"]).iterrows():
        fpg_map[str(r["subject_id"])] = float(r["fpg"])
    for _, r in d1_labels.dropna(subset=["subject_id", "homa_ir"]).iterrows():
        sid = str(r["subject_id"])
        fpg = fpg_map.get(sid, np.nan)
        if np.isfinite(fpg) and fpg > 0:
            fi_map["D1"][sid] = float(r["homa_ir"]) * 405.0 / float(fpg)

    for ds in ["D1", "D2"]:
        b, info, lab = load_cgm_project_level1_level2(dataset_id=ds, output_base=DATA_ROOT)
        lab = _norm_labels(lab)
        lab_map = lab.dropna(subset=["subject_id"]).drop_duplicates("subject_id").set_index("subject_id")
        seen = set()
        for i, sid in enumerate(info.patient_ids):
            sid = str(sid)
            if sid in seen:
                continue
            seen.add(sid)
            rec = {
                "subject_id": sid,
                "sspg": float(lab_map["sspg"].get(sid, np.nan)) if "sspg" in lab_map.columns else np.nan,
                "di": float(lab_map["di"].get(sid, np.nan)) if "di" in lab_map.columns else np.nan,
                "fasting_insulin": fi_map[ds].get(sid, np.nan),
            }
            rec.update(_met_features(b.cgm[i, :, 0], b.timestamps[i, :, 0]))
            rows.append(rec)
    return pd.DataFrame(rows)


class NNPredictor:
    def __init__(self, ckpt_path: str):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.ck = ck
        self.model = MechanisticAutoencoder(
            meal_size=6, demographics_size=3, embedding_size=8, hidden_size=32, num_layers=2,
            encoder_dropout_prob=0.0, decoder_dropout_prob=0.5
        )
        ms = ck.get("model_state", {})
        if hasattr(self.model, "use_v8_recon_corr"):
            self.model.use_v8_recon_corr = bool(ck.get("P1_V8_RECON_CORR", any(k.startswith("correction_mlp.") for k in ms.keys())))
        if hasattr(self.model, "use_v8_ode_corr"):
            self.model.use_v8_ode_corr = bool(ck.get("P1_V8_ODE_CORR", any(k.startswith("ode_correction.") for k in ms.keys())))
        self.model.load_state_dict(ck["model_state"], strict=False)
        self.model.eval()
        self.separate = "sspg_head_state" in ck and "di_head_state" in ck and "e2e_head_state" not in ck
        if self.separate:
            self.in_dim = int(ck["sspg_head_state"]["0.weight"].shape[1])
            self.s_head = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, 16), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(16, 1)
            )
            self.d_head = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, 16), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(16, 1)
            )
            self.s_head.load_state_dict(ck["sspg_head_state"])
            self.d_head.load_state_dict(ck["di_head_state"])
            self.s_head.eval()
            self.d_head.eval()
            self.e2e = None
        else:
            self.in_dim = int(ck["e2e_head_state"]["0.weight"].shape[1])
            self.e2e = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
            )
            self.e2e.load_state_dict(ck["e2e_head_state"])
            self.e2e.eval()
            self.s_head = None
            self.d_head = None

    def predict(self, windows_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        tm, tsd = self.ck["train_mean"], self.ck["train_std"]
        for _, r in windows_df.iterrows():
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
                h = torch.cat([p26, init26], dim=-1) if self.in_dim == 10 else torch.cat([p26, init26, z16], dim=-1)
                if self.separate:
                    s_hat = float(self.s_head(h).squeeze().item())
                    d_hat = float(self.d_head(h).squeeze().item())
                else:
                    y = self.e2e(h).squeeze(0)
                    s_hat = float(y[0].item())
                    d_hat = float(y[1].item())
            if bool(self.ck.get("P1_ZSCORE_TARGETS", False)):
                s_hat = s_hat * float(self.ck["sspg_std"]) + float(self.ck["sspg_mean"])
                d_hat = d_hat * float(self.ck["di_std"]) + float(self.ck["di_mean"])
            rows.append({"subject_id": r["subject_id"], "meal_type": r["meal_type"], "sspg_pred": s_hat, "di_pred": d_hat})
        return pd.DataFrame(rows)

    def latent26(self, windows_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        tm, tsd = self.ck["train_mean"], self.ck["train_std"]
        for _, r in windows_df.iterrows():
            c = r["curve"][None, :, None].astype(np.float32)
            ts = r["timestamps"][None, :, None].astype(np.float32)
            meals = r["meal_series"][None, :, :].astype(np.float32)
            demo = r["demographics"][None, :].astype(np.float32)
            c = (c - tm[0]) / (tsd[0] + 1e-8)
            ts = (ts - tm[1]) / (tsd[1] + 1e-8)
            meals = (meals - tm[2]) / (tsd[2] + 1e-8)
            demo = (demo - tm[3]) / (tsd[3] + 1e-8)
            with torch.no_grad():
                p26, init26, z16 = self.model.get_all_latents_for_head(
                    torch.tensor(c), torch.tensor(ts), torch.tensor(meals), torch.tensor(demo)
                )
            vec = np.concatenate([p26.cpu().numpy()[0], init26.cpu().numpy()[0], z16.cpu().numpy()[0]], axis=0)
            row = {"subject_id": r["subject_id"], "meal_type": r["meal_type"]}
            for i, v in enumerate(vec):
                row[f"z{i:02d}"] = float(v)
            rows.append(row)
        return pd.DataFrame(rows)


def _train_10d_head() -> str:
    out_dir = os.path.join(OUT_ROOT, "v20_Exp3_GV_10D_Head", "phase2_finetune_head")
    ck = os.path.join(out_dir, "autoencoder_p1_full.pt")
    if os.path.isfile(ck):
        return ck
    os.makedirs(out_dir, exist_ok=True)
    pretrain = os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase1_unsupervised", "autoencoder_p1_full.pt")
    env = {
        "P1_TRAIN_DATASETS": "D1,D2",
        "P1_ONE_MEAL_PER_SUBJECT": "1",
        "P1_SAVE_26D_LATENT": "1",
        "P1_ZSCORE_TARGETS": "1",
        "P1_USE_LR_SCHEDULER": "1",
        "P1_HEAD_USE_26D": "1",
        "P1_V8_HEAD_10D": "1",
        "P1_V10_WIDE_BOUNDS": "1",
        "P1_FINETUNE_HEAD_ONLY": "1",
        "P1_PRETRAINED_MODEL": pretrain,
        "V18_EARLY_STOPPING_PATIENCE": "15",
    }
    train_on_d1d2(
        cgm_project_output=DATA_ROOT,
        results_dir=out_dir,
        seed=SEED,
        lambda_sspg=0.1,
        lambda_di=0.1,
        num_epochs=180,
        extra_env=env,
    )
    return ck


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    windows_df, labels_df = _build_d4_windows()
    label_map = labels_df.set_index("subject_id")
    train_tbl = _build_train_table()

    # Exp1 Fair Healey: with and without fasting insulin.
    healey_train = []
    for _, r in train_tbl.dropna(subset=["sspg", "fasting_insulin"]).iterrows():
        si, imax = _fit_healey([r["G_0"], r["G_60"], r["G_120"], r["G_180"]], float(r["fasting_insulin"]))
        if np.isfinite(si) and np.isfinite(imax):
            healey_train.append({"si": si, "imax": imax, "fasting_insulin": float(r["fasting_insulin"]), "sspg": float(r["sspg"])})
    ht = pd.DataFrame(healey_train)
    h_with = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(
        ht[["si", "imax", "fasting_insulin"]].to_numpy(float), ht["sspg"].to_numpy(float)
    )
    h_cgm = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(
        ht[["si", "imax"]].to_numpy(float), ht["sspg"].to_numpy(float)
    )
    d4_sub = windows_df.groupby(["subject_id", "meal_type"], as_index=False).mean(numeric_only=True)
    hx = []
    for _, r in d4_sub.iterrows():
        fi = float(r["fasting_insulin"]) if np.isfinite(r["fasting_insulin"]) else 10.0
        si, imax = _fit_healey([r["G_0"], r["G_60"], r["G_120"], r["G_180"]], fi)
        if not np.isfinite(si) or not np.isfinite(imax):
            si = 1.0 / (abs(float(r["iAUC"])) * max(fi, 1e-3) + 1e-6)
            imax = max(float(r["G_Peak"]) - float(r["G_0"]), 0.0) / max(fi, 1e-3)
        hx.append([si, imax, fi])
    hx = np.asarray(hx, dtype=float)
    d4_sub["pred_with_fi"] = h_with.predict(hx)
    d4_sub["pred_cgm_only"] = h_cgm.predict(hx[:, :2])
    d4_sub["sspg_true"] = d4_sub["subject_id"].map(label_map["sspg"])

    # Exp3: GV 10D-only head
    ck10 = _train_10d_head()
    gv10 = NNPredictor(ck10)
    gv26 = NNPredictor(os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase2_finetune_head", "autoencoder_p1_full.pt"))
    pred10 = gv10.predict(windows_df)
    pred26 = gv26.predict(windows_df)

    # Exp2: latent ICC for GV26
    lat26 = gv26.latent26(windows_df)
    latent_icc_rows = []
    for i in range(26):
        col = f"z{i:02d}"
        w = lat26.pivot_table(index="subject_id", columns="meal_type", values=col, aggfunc="mean")
        cols = [c for c in STD_MEALS if c in w.columns]
        w = w[cols].dropna()
        latent_icc_rows.append({"latent_dim": col, "icc": _icc_oneway(w.to_numpy()) if len(w) >= 3 and len(cols) >= 2 else np.nan})
    latent_icc_df = pd.DataFrame(latent_icc_rows)
    latent_icc_df["group"] = ["10D" if i < 10 else "16D" for i in range(26)]

    # Exp4: Metwally feature ICC
    met_cols = ["G_0", "G_60", "G_120", "G_180", "G_Peak", "CurveSize", "AUC", "pAUC", "nAUC", "iAUC", "CV", "T_baseline2peak", "S_baseline2peak", "S_peak2end"]
    met_icc_rows = []
    for c in met_cols:
        w = d4_sub.pivot_table(index="subject_id", columns="meal_type", values=c, aggfunc="mean")
        cols = [x for x in STD_MEALS if x in w.columns]
        w = w[cols].dropna()
        met_icc_rows.append({"feature": c, "icc": _icc_oneway(w.to_numpy()) if len(w) >= 3 and len(cols) >= 2 else np.nan})
    met_icc_df = pd.DataFrame(met_icc_rows)

    # Per-meal metrics for key models (GV26 vs GV10 + fair Healey variants)
    meal_rows = []
    for meal in STD_MEALS:
        # GV26
        p26_m = pred26[pred26["meal_type"] == meal].groupby("subject_id", as_index=False)[["sspg_pred", "di_pred"]].mean()
        p26_m["sspg_true"] = p26_m["subject_id"].map(label_map["sspg"])
        p26_m["di_true"] = p26_m["subject_id"].map(label_map["di"])
        ms, md = _metrics(p26_m["sspg_true"], p26_m["sspg_pred"]), _metrics(p26_m["di_true"], p26_m["di_pred"])
        meal_rows.append({"model": "GV_26D_Exp8", "meal_type": meal, **{f"sspg_{k}": v for k, v in ms.items()}, **{f"di_{k}": v for k, v in md.items()}})

        # GV10
        p10_m = pred10[pred10["meal_type"] == meal].groupby("subject_id", as_index=False)[["sspg_pred", "di_pred"]].mean()
        p10_m["sspg_true"] = p10_m["subject_id"].map(label_map["sspg"])
        p10_m["di_true"] = p10_m["subject_id"].map(label_map["di"])
        ms, md = _metrics(p10_m["sspg_true"], p10_m["sspg_pred"]), _metrics(p10_m["di_true"], p10_m["di_pred"])
        meal_rows.append({"model": "GV_10D_head_v20", "meal_type": meal, **{f"sspg_{k}": v for k, v in ms.items()}, **{f"di_{k}": v for k, v in md.items()}})

        # Healey fair
        h_m = d4_sub[d4_sub["meal_type"] == meal].copy()
        ms_with = _metrics(h_m["sspg_true"], h_m["pred_with_fi"])
        ms_cgm = _metrics(h_m["sspg_true"], h_m["pred_cgm_only"])
        meal_rows.append({"model": "Healey_with_FI", "meal_type": meal, **{f"sspg_{k}": v for k, v in ms_with.items()}})
        meal_rows.append({"model": "Healey_CGM_only", "meal_type": meal, **{f"sspg_{k}": v for k, v in ms_cgm.items()}})

    per_meal_df = pd.DataFrame(meal_rows)

    # ICC on predictions
    def pred_icc(df: pd.DataFrame, col: str) -> float:
        w = df.pivot_table(index="subject_id", columns="meal_type", values=col, aggfunc="mean")
        cols = [c for c in STD_MEALS if c in w.columns]
        w = w[cols].dropna()
        return _icc_oneway(w.to_numpy()) if len(w) >= 3 and len(cols) >= 2 else np.nan

    icc_pred_df = pd.DataFrame(
        [
            {"model": "GV_26D_Exp8", "icc_sspg_pred": pred_icc(pred26, "sspg_pred"), "icc_di_pred": pred_icc(pred26, "di_pred")},
            {"model": "GV_10D_head_v20", "icc_sspg_pred": pred_icc(pred10, "sspg_pred"), "icc_di_pred": pred_icc(pred10, "di_pred")},
            {"model": "Healey_with_FI", "icc_sspg_pred": pred_icc(d4_sub.rename(columns={"pred_with_fi": "sspg_pred"})[["subject_id", "meal_type", "sspg_pred"]], "sspg_pred"), "icc_di_pred": np.nan},
            {"model": "Healey_CGM_only", "icc_sspg_pred": pred_icc(d4_sub.rename(columns={"pred_cgm_only": "sspg_pred"})[["subject_id", "meal_type", "sspg_pred"]], "sspg_pred"), "icc_di_pred": np.nan},
        ]
    )

    # Save outputs
    latent_icc_df.to_csv(os.path.join(OUT_ROOT, "v20_latent_icc_26d.csv"), index=False)
    met_icc_df.to_csv(os.path.join(OUT_ROOT, "v20_metwally_feature_icc.csv"), index=False)
    per_meal_df.to_csv(os.path.join(OUT_ROOT, "v20_per_meal_metrics.csv"), index=False)
    icc_pred_df.to_csv(os.path.join(OUT_ROOT, "v20_prediction_icc_ablation.csv"), index=False)
    d4_sub.to_csv(os.path.join(OUT_ROOT, "v20_healey_fairness_predictions.csv"), index=False)

    summary = {
        "latent_icc_10d_mean": float(latent_icc_df[latent_icc_df["group"] == "10D"]["icc"].mean()),
        "latent_icc_16d_mean": float(latent_icc_df[latent_icc_df["group"] == "16D"]["icc"].mean()),
        "metwally_feature_icc_mean": float(met_icc_df["icc"].mean()),
    }
    with open(os.path.join(OUT_ROOT, "v20_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUT_ROOT, "v20_report.md"), "w", encoding="utf-8") as f:
        f.write("# GlucoVector v20 Fairness + Cross-Meal Consistency\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Key Consistency Findings\n\n")
        f.write(json.dumps(summary, indent=2))
        f.write("\n\n## Prediction ICC Ablation\n\n")
        f.write(icc_pred_df.to_markdown(index=False))
        f.write("\n\n## Per-Meal Metrics\n\n")
        f.write(per_meal_df.to_markdown(index=False))
        f.write("\n\n## Latent ICC (26D)\n\n")
        f.write(latent_icc_df.to_markdown(index=False))
        f.write("\n\n## Metwally Feature ICC\n\n")
        f.write(met_icc_df.to_markdown(index=False))
        f.write("\n")

    print("Saved v20 outputs to:", OUT_ROOT)


if __name__ == "__main__":
    main()
