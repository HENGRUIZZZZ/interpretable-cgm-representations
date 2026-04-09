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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from load_cgm_project_data import load_cgm_project_level1_level2
from models import MechanisticAutoencoder

DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v20", "d3_free_living")
SEED = 42


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


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 5:
        return {"n": int(len(x)), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {"n": int(len(x)), "pearson_r": float(pr), "pearson_p": float(pp), "spearman_r": float(sr), "spearman_p": float(sp)}


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


def _build_d3_windows() -> pd.DataFrame:
    subjects = pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "subjects.csv"))
    meals = pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "meals.csv"))
    cgm = pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "cgm.csv"))
    labels = pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "labels.csv"))

    meals["timestamp"] = pd.to_datetime(meals["timestamp"], errors="coerce")
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")
    if "glucose_mg_dl" in cgm.columns and "glucose_mgdl" not in cgm.columns:
        cgm = cgm.rename(columns={"glucose_mg_dl": "glucose_mgdl"})

    label_cols = ["subject_id", "hba1c", "HOMA_IR", "HOMA_B", "fasting_glucose_mg_dl", "fasting_insulin_uiu_ml"]
    labels = labels[[c for c in label_cols if c in labels.columns]].drop_duplicates("subject_id")
    label_map = labels.set_index("subject_id")
    subj_map = subjects.set_index("subject_id")

    grid = np.arange(-30, 181, 5, dtype=np.float64)
    rows: List[Dict[str, object]] = []
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

        if sid in subj_map.index:
            s = subj_map.loc[sid]
            gender = 1.0 if str(s.get("sex", "M")).upper().startswith("F") else 0.0
            age = float(pd.to_numeric(s.get("age", 40.0), errors="coerce") or 40.0)
            weight = float(pd.to_numeric(s.get("weight_kg", np.nan), errors="coerce"))
            if not np.isfinite(weight):
                bmi = float(pd.to_numeric(s.get("bmi", 25.0), errors="coerce") or 25.0)
                weight = bmi * (1.7 ** 2)
            demo = np.array([gender, age, weight], dtype=np.float32)
        else:
            demo = np.array([0.0, 40.0, 72.0], dtype=np.float32)

        row = {
            "subject_id": sid,
            "meal_type": str(meal.get("meal_type", "Unknown")),
            "timestamp": t0,
            "date": pd.Timestamp(t0).date().isoformat(),
            "curve": y_new,
            "timestamps": grid.astype(np.float32),
            "meal_series": meal_series,
            "demographics": demo,
            "carb_g": carb,
            "fat_g": fat,
            "protein_g": protein,
            "fiber_g": fiber,
            "calories_kcal": float(pd.to_numeric(meal.get("calories_kcal", np.nan), errors="coerce")),
        }
        row.update(_met14_features(y_new, grid))
        if sid in label_map.index:
            for c in label_map.columns:
                row[c] = float(label_map.loc[sid, c]) if pd.notna(label_map.loc[sid, c]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


class NNPredictor:
    def __init__(self, name: str, ckpt_path: str):
        self.name = name
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

    def infer(self, windows_df: pd.DataFrame) -> pd.DataFrame:
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
                z26 = torch.cat([p26, init26, z16], dim=-1).cpu().numpy()[0]
                h = z26[:10] if self.in_dim == 10 else z26
                h = torch.tensor(h[None, :], dtype=torch.float32)
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
            row = {"subject_id": r["subject_id"], "meal_type": r["meal_type"], "timestamp": r["timestamp"], "sspg_pred": s_hat, "di_pred": d_hat}
            for i, v in enumerate(z26):
                row[f"z{i:02d}"] = float(v)
            rows.append(row)
        return pd.DataFrame(rows)


def _subject_retrieval(df: pd.DataFrame, feat_cols: List[str]) -> float:
    feat_cols = [c for c in feat_cols if c in df.columns]
    if len(feat_cols) == 0:
        return np.nan
    hit, total = 0, 0
    for sid, g in df.sort_values("timestamp").groupby("subject_id"):
        if len(g) < 4:
            continue
        n = len(g)
        a = g.iloc[: n // 2][feat_cols].to_numpy(float)
        b = g.iloc[n // 2 :][feat_cols].to_numpy(float)
        if not np.isfinite(a).any() or not np.isfinite(b).any():
            continue
        a = np.nanmean(a, axis=0)
        norm_a = np.linalg.norm(a) + 1e-8
        best_sid = None
        best_score = -1e9
        for sid2, g2 in df.sort_values("timestamp").groupby("subject_id"):
            if len(g2) < 4:
                continue
            bb = g2.iloc[len(g2) // 2 :][feat_cols].to_numpy(float)
            if not np.isfinite(bb).any():
                continue
            bb = np.nanmean(bb, axis=0)
            score = float(np.dot(a, bb) / (norm_a * (np.linalg.norm(bb) + 1e-8)))
            if score > best_score:
                best_score = score
                best_sid = sid2
        total += 1
        hit += int(best_sid == sid)
    return float(hit / total) if total > 0 else np.nan


def _build_metwally_train_table() -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for ds in ["D1", "D2"]:
        batch, info, labels = load_cgm_project_level1_level2(dataset_id=ds, output_base=DATA_ROOT)
        labels = _norm_labels(labels)
        lab = labels.dropna(subset=["subject_id"]).drop_duplicates("subject_id").set_index("subject_id")
        for i, sid in enumerate(info.patient_ids):
            sid = str(sid)
            feat = _met14_features(batch.cgm[i, :, 0], batch.timestamps[i, :, 0])
            feat["subject_id"] = sid
            feat["sspg"] = float(lab["sspg"].get(sid, np.nan)) if "sspg" in lab.columns else np.nan
            feat["di"] = float(lab["di"].get(sid, np.nan)) if "di" in lab.columns else np.nan
            rows.append(feat)
    tbl = pd.DataFrame(rows)
    # aggregate to subject-level since gold labels are subject-level
    agg = tbl.groupby("subject_id", as_index=False).median(numeric_only=True)
    return agg


def _fit_metwally_models(train_tbl: pd.DataFrame) -> Dict[str, object]:
    x_cols = ["G_0", "G_60", "G_120", "G_180", "G_Peak", "CurveSize", "AUC", "pAUC", "nAUC", "iAUC", "CV", "T_baseline2peak", "S_baseline2peak", "S_peak2end"]
    out: Dict[str, object] = {"x_cols": x_cols}
    tr_s = train_tbl.dropna(subset=x_cols + ["sspg"]).copy()
    tr_d = train_tbl.dropna(subset=x_cols + ["di"]).copy()
    out["sspg_model"] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(
        tr_s[x_cols].to_numpy(float), tr_s["sspg"].to_numpy(float)
    )
    out["di_model"] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(
        tr_d[x_cols].to_numpy(float), tr_d["di"].to_numpy(float)
    )
    return out


def _infer_metwally(models: Dict[str, object], d3: pd.DataFrame) -> pd.DataFrame:
    x_cols = models["x_cols"]
    x = d3[x_cols].to_numpy(float)
    s = models["sspg_model"].predict(x)
    d = models["di_model"].predict(x)
    pred = d3[["subject_id", "meal_type", "timestamp"]].copy()
    pred["sspg_pred"] = s
    pred["di_pred"] = d
    # keep schema aligned for downstream logic
    for i in range(26):
        pred[f"z{i:02d}"] = np.nan
    return pred


def _analyze_model(name: str, pred: pd.DataFrame, d3: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {"model": name}
    merged = pred.merge(
        d3[
            [
                "subject_id",
                "meal_type",
                "timestamp",
                "date",
                "hba1c",
                "HOMA_IR",
                "HOMA_B",
                "fasting_glucose_mg_dl",
                "carb_g",
                "fat_g",
                "protein_g",
                "fiber_g",
                "calories_kcal",
            ]
        ],
        on=["subject_id", "meal_type", "timestamp"],
        how="left",
    )

    # window-level consistency
    by_sub = merged.groupby("subject_id", as_index=False).agg(
        n_windows=("sspg_pred", "size"),
        sspg_std=("sspg_pred", "std"),
        di_std=("di_pred", "std"),
        sspg_mean=("sspg_pred", "mean"),
        di_mean=("di_pred", "mean"),
    )
    out["subject_count"] = int(len(by_sub))
    out["mean_windows_per_subject"] = float(by_sub["n_windows"].mean())
    out["sspg_within_subject_std_mean"] = float(by_sub["sspg_std"].mean())
    out["di_within_subject_std_mean"] = float(by_sub["di_std"].mean())

    # meal-type ICC on predictions (major free-living meals only)
    major = merged[merged["meal_type"].isin(["Breakfast", "Lunch", "Dinner"])].copy()
    out["n_major_meal_windows"] = int(len(major))
    for tgt in ["sspg_pred", "di_pred"]:
        w = major.pivot_table(index="subject_id", columns="meal_type", values=tgt, aggfunc="mean")
        keep_cols = [c for c in ["Breakfast", "Lunch", "Dinner"] if c in w.columns]
        w = w[keep_cols].dropna()
        out[f"{tgt}_meal_icc"] = _icc_oneway(w.to_numpy()) if len(w) >= 5 and len(keep_cols) >= 2 else np.nan

    # day-level triplet (same subject-day with breakfast/lunch/dinner all present)
    day_rows = major.groupby(["subject_id", "date", "meal_type"], as_index=False).mean(numeric_only=True)
    out["n_triplet_days"] = 0
    for tgt in ["sspg_pred", "di_pred"]:
        w_day = day_rows.pivot_table(index=["subject_id", "date"], columns="meal_type", values=tgt, aggfunc="mean")
        keep_cols = [c for c in ["Breakfast", "Lunch", "Dinner"] if c in w_day.columns]
        w_day = w_day[keep_cols].dropna()
        out["n_triplet_days"] = max(out["n_triplet_days"], int(len(w_day)))
        if len(w_day) >= 5 and len(keep_cols) == 3:
            mat = w_day.to_numpy(float)
            out[f"{tgt}_triplet_icc"] = _icc_oneway(mat)
            out[f"{tgt}_triplet_std_mean"] = float(np.mean(np.std(mat, axis=1)))
        else:
            out[f"{tgt}_triplet_icc"] = np.nan
            out[f"{tgt}_triplet_std_mean"] = np.nan

    # weak-label alignment (subject median pred vs lab)
    sub_med = merged.groupby("subject_id", as_index=False).median(numeric_only=True)
    c_homa_sspg = _safe_corr(sub_med["sspg_pred"], sub_med["HOMA_IR"])
    c_hba1c_sspg = _safe_corr(sub_med["sspg_pred"], sub_med["hba1c"])
    c_homa_di = _safe_corr(sub_med["di_pred"], sub_med["HOMA_IR"])
    out.update(
        {
            "corr_sspg_vs_homa_ir_spearman": c_homa_sspg["spearman_r"],
            "corr_sspg_vs_hba1c_spearman": c_hba1c_sspg["spearman_r"],
            "corr_di_vs_homa_ir_spearman": c_homa_di["spearman_r"],
            "corr_n_subjects": int(c_homa_sspg["n"]),
        }
    )

    # macro alignment (window-level) to show meal sensitivity
    out["corr_sspg_vs_carb_window_spearman"] = _safe_corr(merged["sspg_pred"], merged["carb_g"])["spearman_r"]
    out["corr_di_vs_carb_window_spearman"] = _safe_corr(merged["di_pred"], merged["carb_g"])["spearman_r"]

    # subject retrieval with latent space
    zcols = [f"z{i:02d}" for i in range(26)]
    if np.isfinite(merged[zcols].to_numpy(dtype=float)).any():
        out["subject_retrieval_top1_26d"] = _subject_retrieval(merged, zcols)
        out["subject_retrieval_top1_10d"] = _subject_retrieval(merged, zcols[:10])
    else:
        out["subject_retrieval_top1_26d"] = np.nan
        out["subject_retrieval_top1_10d"] = np.nan
    return out


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    d3 = _build_d3_windows()
    d3 = d3.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)
    d3.to_csv(os.path.join(OUT_ROOT, "d3_windows_index.csv"), index=False)
    macro_audit = pd.DataFrame(
        {
            "field": ["carb_g", "fat_g", "protein_g", "fiber_g", "calories_kcal"],
            "non_null": [int(d3[c].notna().sum()) for c in ["carb_g", "fat_g", "protein_g", "fiber_g", "calories_kcal"]],
            "missing_rate": [float(1.0 - d3[c].notna().mean()) for c in ["carb_g", "fat_g", "protein_g", "fiber_g", "calories_kcal"]],
        }
    )
    macro_audit.to_csv(os.path.join(OUT_ROOT, "d3_macro_input_audit.csv"), index=False)

    nn_models = [
        NNPredictor("Wang_Exp1", os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v18", "v18_Exp1_Wang_Baseline", "joint_training", "autoencoder_p1_full.pt")),
        NNPredictor("GV_26D_Exp8", os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v18", "v18_Exp8_CorrLoss", "phase2_finetune_head", "autoencoder_p1_full.pt")),
        NNPredictor("GV_10D_head_v20", os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v20", "v20_Exp3_GV_10D_Head", "phase2_finetune_head", "autoencoder_p1_full.pt")),
    ]
    met_train = _build_metwally_train_table()
    met_models = _fit_metwally_models(met_train)

    summary_rows = []
    for m in nn_models:
        pred = m.infer(d3)
        pred.to_csv(os.path.join(OUT_ROOT, f"d3_predictions_{m.name}.csv"), index=False)
        merged = pred.merge(d3[["subject_id", "meal_type", "timestamp", "date", "hba1c", "HOMA_IR", "HOMA_B", "fasting_glucose_mg_dl", "carb_g", "fat_g", "protein_g", "fiber_g", "calories_kcal"]], on=["subject_id", "meal_type", "timestamp"], how="left")
        merged.to_csv(os.path.join(OUT_ROOT, f"d3_predictions_with_labels_{m.name}.csv"), index=False)
        summary_rows.append(_analyze_model(m.name, pred, d3))

    met_pred = _infer_metwally(met_models, d3)
    met_pred.to_csv(os.path.join(OUT_ROOT, "d3_predictions_Metwally14_Ridge.csv"), index=False)
    met_merged = met_pred.merge(d3[["subject_id", "meal_type", "timestamp", "date", "hba1c", "HOMA_IR", "HOMA_B", "fasting_glucose_mg_dl", "carb_g", "fat_g", "protein_g", "fiber_g", "calories_kcal"]], on=["subject_id", "meal_type", "timestamp"], how="left")
    met_merged.to_csv(os.path.join(OUT_ROOT, "d3_predictions_with_labels_Metwally14_Ridge.csv"), index=False)
    summary_rows.append(_analyze_model("Metwally14_Ridge", met_pred, d3))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_ROOT, "d3_free_living_benchmark_summary.csv"), index=False)
    # dedicated day-triplet comparison table
    triplet_cols = [
        "model",
        "n_triplet_days",
        "sspg_pred_triplet_icc",
        "di_pred_triplet_icc",
        "sspg_pred_triplet_std_mean",
        "di_pred_triplet_std_mean",
    ]
    summary[[c for c in triplet_cols if c in summary.columns]].to_csv(
        os.path.join(OUT_ROOT, "d3_same_day_triplet_comparison.csv"), index=False
    )

    with open(os.path.join(OUT_ROOT, "d3_free_living_benchmark_report.md"), "w", encoding="utf-8") as f:
        f.write("# v20 D3 Free-living Benchmark (No SSPG/DI Gold)\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("This benchmark compares model behavior on free-living D3 where SSPG/DI are unavailable. ")
        f.write("Metrics include within-subject prediction stability, meal-type ICC, same-day triplet (Breakfast/Lunch/Dinner) consistency, weak-label alignment (HOMA-IR/HbA1c), macro alignment, and subject retrieval from latent space.\n\n")
        f.write("## Macro Input Audit (D3 meals.csv)\n\n")
        f.write(macro_audit.to_markdown(index=False))
        f.write("\n\n## Main Comparison\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n")

    with open(os.path.join(OUT_ROOT, "d3_free_living_benchmark_summary.json"), "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    main()
