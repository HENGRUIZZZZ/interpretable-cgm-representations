from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
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
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v19")
SEED = 42
STD_MEALS = ["Cornflakes", "PB_sandwich", "Protein_bar"]


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


def _auc_with_ci(y_true: np.ndarray, scores: np.ndarray, n_boot: int = 2000) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    if len(np.unique(y)) < 2 or len(y) < 8:
        return {"auroc": np.nan, "auroc_ci_lo": np.nan, "auroc_ci_hi": np.nan}
    auc = float(roc_auc_score(y, s))
    rng = np.random.RandomState(42)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], s[idx])))
    if not vals:
        return {"auroc": auc, "auroc_ci_lo": np.nan, "auroc_ci_hi": np.nan}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"auroc": auc, "auroc_ci_lo": float(lo), "auroc_ci_hi": float(hi)}


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
        return float("nan")
    return float((ms_between - ms_within) / den)


def _metwally_features(cgm: np.ndarray, ts: np.ndarray) -> Dict[str, float]:
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
        "mean": float(np.mean(g)), "std": float(np.std(g)), "min": float(np.min(g)),
        "max": float(np.max(g)), "range": float(np.max(g) - np.min(g)),
    }


def _build_d4_windows() -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "subjects.csv"))
    meals = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "meals.csv"))
    cgm = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "cgm.csv"))
    labels = _norm_labels(pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "labels.csv")))
    label_df = labels[["subject_id", "sspg", "di", "fasting_insulin"]].drop_duplicates("subject_id")
    fi_lookup = label_df.set_index("subject_id")["fasting_insulin"].to_dict()

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
                **_metwally_features(y, t),
                "fasting_insulin": float(fi_lookup.get(sid, np.nan)),
            }
        )

    return pd.DataFrame(rows), label_df


def _build_train_subject_table() -> pd.DataFrame:
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
                "dataset": ds,
                "sspg": float(lab_map["sspg"].get(sid, np.nan)) if "sspg" in lab_map.columns else np.nan,
                "di": float(lab_map["di"].get(sid, np.nan)) if "di" in lab_map.columns else np.nan,
                "fasting_insulin": fi_map[ds].get(sid, np.nan),
            }
            rec.update(_metwally_features(b.cgm[i, :, 0], b.timestamps[i, :, 0]))
            rows.append(rec)
    return pd.DataFrame(rows)


@dataclass
class NNPredictor:
    name: str
    ckpt_path: str
    model: MechanisticAutoencoder
    in_dim: int
    separate: bool
    e2e: torch.nn.Module | None
    s_head: torch.nn.Module | None
    d_head: torch.nn.Module | None
    train_mean: List[np.ndarray]
    train_std: List[np.ndarray]
    zscore_targets: bool
    sspg_mean: float
    sspg_std: float
    di_mean: float
    di_std: float

    @classmethod
    def from_ckpt(cls, name: str, ckpt_path: str) -> "NNPredictor":
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = MechanisticAutoencoder(
            meal_size=6, demographics_size=3, embedding_size=8, hidden_size=32, num_layers=2,
            encoder_dropout_prob=0.0, decoder_dropout_prob=0.5
        )
        ms = ckpt.get("model_state", {})
        if hasattr(model, "use_v8_recon_corr"):
            model.use_v8_recon_corr = bool(ckpt.get("P1_V8_RECON_CORR", any(k.startswith("correction_mlp.") for k in ms.keys())))
        if hasattr(model, "use_v8_ode_corr"):
            model.use_v8_ode_corr = bool(ckpt.get("P1_V8_ODE_CORR", any(k.startswith("ode_correction.") for k in ms.keys())))
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.eval()

        separate = "sspg_head_state" in ckpt and "di_head_state" in ckpt and "e2e_head_state" not in ckpt
        e2e = s_head = d_head = None
        if separate:
            in_dim = int(ckpt["sspg_head_state"]["0.weight"].shape[1])
            s_head = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 16), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(16, 1)
            )
            d_head = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 16), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(16, 1)
            )
            s_head.load_state_dict(ckpt["sspg_head_state"])
            d_head.load_state_dict(ckpt["di_head_state"])
            s_head.eval()
            d_head.eval()
        else:
            in_dim = int(ckpt["e2e_head_state"]["0.weight"].shape[1])
            e2e = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
            )
            e2e.load_state_dict(ckpt["e2e_head_state"])
            e2e.eval()

        return cls(
            name=name,
            ckpt_path=ckpt_path,
            model=model,
            in_dim=in_dim,
            separate=separate,
            e2e=e2e,
            s_head=s_head,
            d_head=d_head,
            train_mean=ckpt["train_mean"],
            train_std=ckpt["train_std"],
            zscore_targets=bool(ckpt.get("P1_ZSCORE_TARGETS", False)),
            sspg_mean=float(ckpt.get("sspg_mean", 0.0)),
            sspg_std=float(ckpt.get("sspg_std", 1.0)),
            di_mean=float(ckpt.get("di_mean", 0.0)),
            di_std=float(ckpt.get("di_std", 1.0)),
        )

    def predict_windows(self, windows_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for _, r in windows_df.iterrows():
            c = r["curve"][None, :, None].astype(np.float32)
            ts = r["timestamps"][None, :, None].astype(np.float32)
            meals = r["meal_series"][None, :, :].astype(np.float32)
            demo = r["demographics"][None, :].astype(np.float32)
            c = (c - self.train_mean[0]) / (self.train_std[0] + 1e-8)
            ts = (ts - self.train_mean[1]) / (self.train_std[1] + 1e-8)
            meals = (meals - self.train_mean[2]) / (self.train_std[2] + 1e-8)
            demo = (demo - self.train_mean[3]) / (self.train_std[3] + 1e-8)
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
            if self.zscore_targets:
                s_hat = s_hat * self.sspg_std + self.sspg_mean
                d_hat = d_hat * self.di_std + self.di_mean
            rows.append({"subject_id": r["subject_id"], "meal_type": r["meal_type"], "sspg_pred": s_hat, "di_pred": d_hat})
        return pd.DataFrame(rows)


def _fit_healey_params(g_points: List[float], fasting_insulin: float) -> Tuple[float, float]:
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
    si, imax = np.exp(r.x[0]), np.exp(r.x[1])
    return float(si), float(imax)


def _train_v19_extra_varmatch() -> str:
    exp_dir = os.path.join(OUT_ROOT, "v19_Exp_GV_CorrLoss_VarMatch")
    p1_ck = os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase1_unsupervised", "autoencoder_p1_full.pt")
    p2_dir = os.path.join(exp_dir, "phase2_finetune_head")
    ck = os.path.join(p2_dir, "autoencoder_p1_full.pt")
    if os.path.isfile(ck):
        return ck
    os.makedirs(p2_dir, exist_ok=True)
    env = {
        "P1_TRAIN_DATASETS": "D1,D2",
        "P1_SAVE_26D_LATENT": "1",
        "P1_ZSCORE_TARGETS": "1",
        "P1_USE_LR_SCHEDULER": "1",
        "P1_ONE_MEAL_PER_SUBJECT": "1",
        "P1_FINETUNE_HEAD_ONLY": "1",
        "P1_PRETRAINED_MODEL": p1_ck,
        "V18_SEPARATE_SMALL_HEAD": "1",
        "V18_EARLY_STOPPING_PATIENCE": "15",
        "V18_CORR_LOSS_ALPHA": "0.5",
        "LAMBDA_VAR_MATCH": "0.05",
    }
    train_on_d1d2(
        cgm_project_output=DATA_ROOT,
        results_dir=p2_dir,
        seed=SEED,
        lambda_sspg=0.1,
        lambda_di=0.1,
        num_epochs=200,
        extra_env=env,
    )
    return ck


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Optional necessary comparison: CorrLoss + VarMatch combined under separate-head path.
    ck_varmatch = _train_v19_extra_varmatch()

    windows_df, labels_df = _build_d4_windows()
    label_map = labels_df.set_index("subject_id")
    train_tbl = _build_train_subject_table()

    met_cols = ["G_0", "G_60", "G_120", "G_180", "G_Peak", "CurveSize", "AUC", "pAUC", "nAUC", "iAUC", "CV", "T_baseline2peak", "S_baseline2peak", "S_peak2end"]
    stat_cols = ["mean", "std", "CV", "min", "max", "range", "AUC"]

    # Train traditional baselines with both SSPG and DI.
    tr_met_s = train_tbl.dropna(subset=["sspg"] + met_cols)
    tr_met_d = train_tbl.dropna(subset=["di"] + met_cols)
    met_sspg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(tr_met_s[met_cols].to_numpy(float), tr_met_s["sspg"].to_numpy(float))
    met_di = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(tr_met_d[met_cols].to_numpy(float), tr_met_d["di"].to_numpy(float))

    tr_stat_s = train_tbl.dropna(subset=["sspg"] + stat_cols)
    tr_stat_d = train_tbl.dropna(subset=["di"] + stat_cols)
    stat_sspg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(tr_stat_s[stat_cols].to_numpy(float), tr_stat_s["sspg"].to_numpy(float))
    stat_di = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(tr_stat_d[stat_cols].to_numpy(float), tr_stat_d["di"].to_numpy(float))

    # Healey: train ridge on fitted params.
    h_rows: List[Dict[str, float]] = []
    for _, r in train_tbl.dropna(subset=["sspg", "di", "fasting_insulin"]).iterrows():
        si, imax = _fit_healey_params([r["G_0"], r["G_60"], r["G_120"], r["G_180"]], float(r["fasting_insulin"]))
        if np.isfinite(si) and np.isfinite(imax):
            h_rows.append({"si": si, "imax": imax, "fasting_insulin": float(r["fasting_insulin"]), "sspg": float(r["sspg"]), "di": float(r["di"])})
    healey_tbl = pd.DataFrame(h_rows)
    healey_sspg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(healey_tbl[["si", "imax", "fasting_insulin"]].to_numpy(float), healey_tbl["sspg"].to_numpy(float))
    healey_di = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(healey_tbl[["si", "imax", "fasting_insulin"]].to_numpy(float), healey_tbl["di"].to_numpy(float))

    # NN models
    nn_models = [
        NNPredictor.from_ckpt("Wang(Exp1)", os.path.join(V18_ROOT, "v18_Exp1_Wang_Baseline", "joint_training", "autoencoder_p1_full.pt")),
        NNPredictor.from_ckpt("GV_Baseline(Exp5)", os.path.join(V18_ROOT, "v18_Exp5_GV_Baseline", "phase2_finetune_head", "autoencoder_p1_full.pt")),
        NNPredictor.from_ckpt("GV_CorrLoss(Exp8)", os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase2_finetune_head", "autoencoder_p1_full.pt")),
        NNPredictor.from_ckpt("GV_CorrLoss+VarMatch(v19)", ck_varmatch),
    ]

    pred_frames: Dict[str, pd.DataFrame] = {}
    for m in nn_models:
        pred_frames[m.name] = m.predict_windows(windows_df)

    # Traditional model predictions on D4 windows
    d4_feats = windows_df.groupby(["subject_id", "meal_type"], as_index=False).mean(numeric_only=True)
    pred_frames["Metwally(Exp2)"] = d4_feats.assign(
        sspg_pred=met_sspg.predict(d4_feats[met_cols].to_numpy(float)),
        di_pred=met_di.predict(d4_feats[met_cols].to_numpy(float)),
    )[["subject_id", "meal_type", "sspg_pred", "di_pred"]]
    pred_frames["SimpleStats(Exp4)"] = d4_feats.assign(
        sspg_pred=stat_sspg.predict(d4_feats[stat_cols].to_numpy(float)),
        di_pred=stat_di.predict(d4_feats[stat_cols].to_numpy(float)),
    )[["subject_id", "meal_type", "sspg_pred", "di_pred"]]

    hx = []
    for _, r in d4_feats.iterrows():
        fi = float(r["fasting_insulin"]) if np.isfinite(r["fasting_insulin"]) else 10.0
        si, imax = _fit_healey_params([r["G_0"], r["G_60"], r["G_120"], r["G_180"]], fi)
        if not np.isfinite(si) or not np.isfinite(imax):
            si = 1.0 / (abs(float(r["iAUC"])) * max(fi, 1e-3) + 1e-6)
            imax = max(float(r["G_Peak"]) - float(r["G_0"]), 0.0) / max(fi, 1e-3)
        hx.append([si, imax, fi])
    hx = np.asarray(hx, dtype=float)
    pred_frames["Healey(Exp3)"] = d4_feats.assign(
        sspg_pred=healey_sspg.predict(hx),
        di_pred=healey_di.predict(hx),
    )[["subject_id", "meal_type", "sspg_pred", "di_pred"]]

    # Metrics tables
    per_meal_rows: List[Dict[str, object]] = []
    overall_rows: List[Dict[str, object]] = []
    icc_rows: List[Dict[str, object]] = []
    clf_rows: List[Dict[str, object]] = []

    for model_name, pred in pred_frames.items():
        # per-meal
        for meal in STD_MEALS:
            sub = pred[pred["meal_type"] == meal].groupby("subject_id", as_index=False)[["sspg_pred", "di_pred"]].mean()
            sub["sspg_true"] = sub["subject_id"].map(label_map["sspg"])
            sub["di_true"] = sub["subject_id"].map(label_map["di"])
            ms = _metrics(sub["sspg_true"].to_numpy(float), sub["sspg_pred"].to_numpy(float))
            md = _metrics(sub["di_true"].to_numpy(float), sub["di_pred"].to_numpy(float))
            per_meal_rows.append(
                {"model": model_name, "meal_type": meal, **{f"sspg_{k}": v for k, v in ms.items()}, **{f"di_{k}": v for k, v in md.items()}}
            )

        # overall subject-level
        ov = pred.groupby("subject_id", as_index=False)[["sspg_pred", "di_pred"]].mean()
        ov["sspg_true"] = ov["subject_id"].map(label_map["sspg"])
        ov["di_true"] = ov["subject_id"].map(label_map["di"])
        ms = _metrics(ov["sspg_true"].to_numpy(float), ov["sspg_pred"].to_numpy(float))
        md = _metrics(ov["di_true"].to_numpy(float), ov["di_pred"].to_numpy(float))
        overall_rows.append({"model": model_name, **{f"sspg_{k}": v for k, v in ms.items()}, **{f"di_{k}": v for k, v in md.items()}})

        # ICC across meals
        wide_s = pred.pivot_table(index="subject_id", columns="meal_type", values="sspg_pred", aggfunc="mean")
        wide_d = pred.pivot_table(index="subject_id", columns="meal_type", values="di_pred", aggfunc="mean")
        cols = [c for c in STD_MEALS if c in wide_s.columns]
        ws = wide_s[cols].dropna()
        wd = wide_d[cols].dropna()
        icc_rows.append(
            {
                "model": model_name,
                "n_subjects_sspg": int(len(ws)),
                "n_subjects_di": int(len(wd)),
                "icc_sspg_pred": _icc_oneway(ws.to_numpy()) if len(ws) >= 3 and len(cols) >= 2 else np.nan,
                "icc_di_pred": _icc_oneway(wd.to_numpy()) if len(wd) >= 3 and len(cols) >= 2 else np.nan,
            }
        )

        # classification: IR and Decompensation
        y_ir = (ov["sspg_true"].to_numpy(float) >= 120.0).astype(int)
        y_ir_pred = (ov["sspg_pred"].to_numpy(float) >= 120.0).astype(int)
        ir_auc = _auc_with_ci(y_ir, ov["sspg_pred"].to_numpy(float))

        y_dec = ((ov["sspg_true"].to_numpy(float) >= 120.0) & (ov["di_true"].to_numpy(float) < 1.0)).astype(int)
        dec_score = ov["sspg_pred"].to_numpy(float) - 40.0 * ov["di_pred"].to_numpy(float)
        dec_auc = _auc_with_ci(y_dec, dec_score)
        y_dec_pred = ((ov["sspg_pred"].to_numpy(float) >= 120.0) & (ov["di_pred"].to_numpy(float) < 1.0)).astype(int)

        clf_rows.append(
            {
                "model": model_name,
                "ir_auroc": ir_auc["auroc"],
                "ir_auroc_ci_lo": ir_auc["auroc_ci_lo"],
                "ir_auroc_ci_hi": ir_auc["auroc_ci_hi"],
                "ir_accuracy": float(accuracy_score(y_ir, y_ir_pred)),
                "ir_f1": float(f1_score(y_ir, y_ir_pred, zero_division=0)),
                "decomp_auroc": dec_auc["auroc"],
                "decomp_auroc_ci_lo": dec_auc["auroc_ci_lo"],
                "decomp_auroc_ci_hi": dec_auc["auroc_ci_hi"],
                "decomp_accuracy": float(accuracy_score(y_dec, y_dec_pred)),
                "decomp_f1": float(f1_score(y_dec, y_dec_pred, zero_division=0)),
            }
        )

    per_meal_df = pd.DataFrame(per_meal_rows)
    overall_df = pd.DataFrame(overall_rows)
    icc_df = pd.DataFrame(icc_rows)
    clf_df = pd.DataFrame(clf_rows)

    # Metwally DI classification supplementary comparison
    tr_met_clf = train_tbl.dropna(subset=["di"] + met_cols).copy()
    te_feat = d4_feats.groupby("subject_id", as_index=False).mean(numeric_only=True)
    te_feat["di_true"] = te_feat["subject_id"].map(label_map["di"])
    te_feat = te_feat.dropna(subset=["di_true"])
    y_tr_1 = (tr_met_clf["di"].to_numpy(float) < 1.0).astype(int)
    y_te_1 = (te_feat["di_true"].to_numpy(float) < 1.0).astype(int)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
    lr.fit(tr_met_clf[met_cols].to_numpy(float), y_tr_1)
    x_te = te_feat[met_cols].to_numpy(float)
    s_te = lr.predict_proba(x_te)[:, 1]
    met_di_clf = {"threshold": 1.0, **_auc_with_ci(y_te_1, s_te)}

    # Save
    per_meal_df.to_csv(os.path.join(OUT_ROOT, "v19_per_meal_metrics.csv"), index=False)
    overall_df.to_csv(os.path.join(OUT_ROOT, "v19_overall_metrics.csv"), index=False)
    icc_df.to_csv(os.path.join(OUT_ROOT, "v19_icc_across_meals.csv"), index=False)
    clf_df.to_csv(os.path.join(OUT_ROOT, "v19_joint_classification_metrics.csv"), index=False)
    with open(os.path.join(OUT_ROOT, "v19_metwally_di_classification.json"), "w") as f:
        json.dump(met_di_clf, f, indent=2)

    report_path = os.path.join(OUT_ROOT, "v19_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# GlucoVector v19 Extended Evaluation\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Overall Subject-Level Metrics\n\n")
        f.write(overall_df.to_markdown(index=False))
        f.write("\n\n## Per-Meal-Type Metrics\n\n")
        f.write(per_meal_df.to_markdown(index=False))
        f.write("\n\n## Cross-Meal ICC (Prediction Stability)\n\n")
        f.write(icc_df.to_markdown(index=False))
        f.write("\n\n## IR × β Joint Classification\n\n")
        f.write(clf_df.to_markdown(index=False))
        f.write("\n\n## Additional Necessary Comparison\n\n")
        f.write(f"- Metwally DI classification (DI<1.0) AUROC: {met_di_clf['auroc']:.4f} ")
        f.write(f"(95% CI {met_di_clf['auroc_ci_lo']:.4f} - {met_di_clf['auroc_ci_hi']:.4f})\n")

    print("Saved:")
    print(os.path.join(OUT_ROOT, "v19_per_meal_metrics.csv"))
    print(os.path.join(OUT_ROOT, "v19_overall_metrics.csv"))
    print(os.path.join(OUT_ROOT, "v19_icc_across_meals.csv"))
    print(os.path.join(OUT_ROOT, "v19_joint_classification_metrics.csv"))
    print(report_path)


if __name__ == "__main__":
    main()
