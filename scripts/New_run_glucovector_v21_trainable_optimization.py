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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import MechanisticAutoencoder

DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
V18_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v18")
V19_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v19")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v21_optimization")
STD_MEALS = ["Cornflakes", "PB_sandwich", "Protein_bar"]
SEED = 42


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


def _bootstrap_auc(y: np.ndarray, s: np.ndarray, n_boot: int = 3000, seed: int = 42) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return {"auc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": int(len(y))}
    auc = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        yy, ss = y[b], s[b]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(float(roc_auc_score(yy, ss)))
    if len(vals) < 20:
        return {"auc": auc, "ci_low": np.nan, "ci_high": np.nan, "n": int(len(y))}
    return {"auc": auc, "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5)), "n": int(len(y))}


def _norm_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for old, new in [("SSPG", "sspg"), ("DI", "di"), ("HOMA_IR", "homa_ir"), ("HOMA_B", "homa_b")]:
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    return out


def _build_d4_windows() -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "subjects.csv"))
    meals = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "meals.csv"))
    cgm = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "cgm.csv"))
    labels = _norm_labels(pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "labels.csv")))
    label_df = labels[["subject_id", "sspg", "di", "fasting_insulin"]].drop_duplicates("subject_id")

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
        t = ((g["timestamp"] - t0).dt.total_seconds() / 60.0).to_numpy(float)
        y = pd.to_numeric(g["glucose_mgdl"], errors="coerce").to_numpy(float)
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
            }
        )
    return pd.DataFrame(rows), label_df


class Encoder26:
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

    def encode(self, windows_df: pd.DataFrame) -> pd.DataFrame:
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
            z = np.concatenate([p26.numpy()[0], init26.numpy()[0], z16.numpy()[0]], axis=0)
            row = {"subject_id": r["subject_id"], "meal_type": r["meal_type"]}
            for i, v in enumerate(z):
                row[f"z{i:02d}"] = float(v)
            rows.append(row)
        return pd.DataFrame(rows)


def _train_stack_models(train_lat: pd.DataFrame) -> Dict[str, object]:
    z10 = [f"z{i:02d}" for i in range(10)]
    z26 = [f"z{i:02d}" for i in range(26)]
    tr = train_lat.dropna(subset=z26 + ["sspg", "di"]).copy()
    X10 = tr[z10].to_numpy(float)
    X26 = tr[z26].to_numpy(float)
    y_s = tr["sspg"].to_numpy(float)
    y_d = tr["di"].to_numpy(float)

    s10 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(X10, y_s)
    d10 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(X10, y_d)
    s26 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(X26, y_s)
    d26 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(X26, y_d)

    # OOF stack features to avoid optimistic training for meta heads
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_s = np.zeros((len(tr), 3), dtype=float)
    oof_d = np.zeros((len(tr), 3), dtype=float)
    for tr_idx, va_idx in kf.split(X26):
        s10_i = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))).fit(X10[tr_idx], y_s[tr_idx])
        s26_i = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))).fit(X26[tr_idx], y_s[tr_idx])
        d10_i = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))).fit(X10[tr_idx], y_d[tr_idx])
        d26_i = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))).fit(X26[tr_idx], y_d[tr_idx])
        p10s = s10_i.predict(X10[va_idx])
        p26s = s26_i.predict(X26[va_idx])
        p10d = d10_i.predict(X10[va_idx])
        p26d = d26_i.predict(X26[va_idx])
        oof_s[va_idx] = np.c_[p10s, p26s, np.abs(p10s - p26s)]
        oof_d[va_idx] = np.c_[p10d, p26d, np.abs(p10d - p26d)]

    ms = GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=2, random_state=SEED).fit(oof_s, y_s)
    md = GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=2, random_state=SEED).fit(oof_d, y_d)
    return {"s10": s10, "d10": d10, "s26": s26, "d26": d26, "ms": ms, "md": md}


def _predict_models(models: Dict[str, object], xdf: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    z10 = [f"z{i:02d}" for i in range(10)]
    z26 = [f"z{i:02d}" for i in range(26)]
    X10 = xdf[z10].to_numpy(float)
    X26 = xdf[z26].to_numpy(float)
    p10s = models["s10"].predict(X10)
    p10d = models["d10"].predict(X10)
    p26s = models["s26"].predict(X26)
    p26d = models["d26"].predict(X26)
    meta_s = models["ms"].predict(np.c_[p10s, p26s, np.abs(p10s - p26s)])
    meta_d = models["md"].predict(np.c_[p10d, p26d, np.abs(p10d - p26d)])
    out = {}
    for name, ps, pd_ in [
        ("Ridge10D", p10s, p10d),
        ("Ridge26D", p26s, p26d),
        ("StackGated10D26D", meta_s, meta_d),
    ]:
        out[name] = xdf[["subject_id", "meal_type"]].assign(sspg_pred=ps, di_pred=pd_)
    return out


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    # train latents from Exp8 pooled D1+D2
    train_lat = pd.read_csv(os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase2_finetune_head", "latent_and_gold_all_26d.csv"))
    rename_map = {
        "tau_m": "z00", "Gb": "z01", "sg": "z02", "si": "z03", "p2": "z04", "mi": "z05",
        "z_init_0": "z06", "z_init_1": "z07", "z_init_2": "z08", "z_init_3": "z09",
    }
    for k, v in rename_map.items():
        if k in train_lat.columns and v not in train_lat.columns:
            train_lat[v] = train_lat[k]
    for i in range(16):
        src = f"z_nonseq_{i}"
        dst = f"z{10+i:02d}"
        if src in train_lat.columns and dst not in train_lat.columns:
            train_lat[dst] = train_lat[src]
    train_lat = _norm_labels(train_lat)

    models = _train_stack_models(train_lat)

    # D4 encode and evaluate
    d4_windows, d4_labels = _build_d4_windows()
    enc = Encoder26(os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase2_finetune_head", "autoencoder_p1_full.pt"))
    d4_lat = enc.encode(d4_windows)
    preds = _predict_models(models, d4_lat)
    label_map = d4_labels.set_index("subject_id")

    rows = []
    auc_rows = []
    pred_exports = []
    for name, p in preds.items():
        sub = p.groupby("subject_id", as_index=False)[["sspg_pred", "di_pred"]].mean()
        sub["sspg_true"] = sub["subject_id"].map(label_map["sspg"])
        sub["di_true"] = sub["subject_id"].map(label_map["di"])
        ms = _metrics(sub["sspg_true"], sub["sspg_pred"])
        md = _metrics(sub["di_true"], sub["di_pred"])
        rows.append({"model": name, **{f"sspg_{k}": v for k, v in ms.items()}, **{f"di_{k}": v for k, v in md.items()}})

        y_ir = (sub["sspg_true"].to_numpy(float) >= 120.0).astype(int)
        s_ir = sub["sspg_pred"].to_numpy(float)
        ir_auc = _bootstrap_auc(y_ir, s_ir, n_boot=2500, seed=SEED)
        y_de = ((sub["sspg_true"].to_numpy(float) >= 120.0) & (sub["di_true"].to_numpy(float) < 1.0)).astype(int)
        s_de = stats.zscore(sub["sspg_pred"].to_numpy(float), nan_policy="omit") - stats.zscore(sub["di_pred"].to_numpy(float), nan_policy="omit")
        de_auc = _bootstrap_auc(y_de, s_de, n_boot=2500, seed=SEED + 1)
        auc_rows.append(
            {
                "model": name,
                "ir_auc": ir_auc["auc"],
                "ir_ci_low": ir_auc["ci_low"],
                "ir_ci_high": ir_auc["ci_high"],
                "decomp_auc": de_auc["auc"],
                "decomp_ci_low": de_auc["ci_low"],
                "decomp_ci_high": de_auc["ci_high"],
            }
        )
        pred_exports.append(sub.assign(model=name))

    reg_df = pd.DataFrame(rows)
    auc_df = pd.DataFrame(auc_rows)

    # bring baseline references for same report
    v19 = pd.read_csv(os.path.join(V19_ROOT, "v19_overall_metrics.csv"))
    base = v19[v19["model"].isin(["GV_CorrLoss(Exp8)", "Metwally(Exp2)", "Healey(Exp3)", "Wang(Exp1)"])].copy()
    base = base.rename(
        columns={
            "sspg_spearman_r": "sspg_spearman_ref",
            "sspg_rmse": "sspg_rmse_ref",
            "di_spearman_r": "di_spearman_ref",
            "di_rmse": "di_rmse_ref",
        }
    )

    reg_df.to_csv(os.path.join(OUT_ROOT, "v21_opt_regression_metrics.csv"), index=False)
    auc_df.to_csv(os.path.join(OUT_ROOT, "v21_opt_auroc_bootstrap.csv"), index=False)
    pd.concat(pred_exports, ignore_index=True).to_csv(os.path.join(OUT_ROOT, "v21_opt_predictions_subject_level.csv"), index=False)
    base.to_csv(os.path.join(OUT_ROOT, "v21_reference_baselines_v19.csv"), index=False)

    with open(os.path.join(OUT_ROOT, "v21_opt_report.md"), "w", encoding="utf-8") as f:
        f.write("# v21 Trainable Optimization (new trained heads)\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## New Models Trained on D1+D2 Latents\n\n")
        f.write(reg_df.to_markdown(index=False))
        f.write("\n\n## AUROC with Bootstrap CI\n\n")
        f.write(auc_df.to_markdown(index=False))
        f.write("\n\n## Baseline Reference (v19)\n\n")
        f.write(base.to_markdown(index=False))
        f.write("\n")

    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    main()
