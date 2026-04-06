from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import MechanisticAutoencoder


def _build_e2e_head(input_dim: int = 26) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2),
    )


def _infer_input_dim(state_dict: dict) -> int:
    w = state_dict.get("0.weight", None)
    return int(w.shape[1]) if w is not None else 26


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) < 3:
        return {"n": int(len(y_true)), "pearson_r": np.nan, "spearman_r": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    return {
        "n": int(len(y_true)),
        "pearson_r": float(stats.pearsonr(y_true, y_pred)[0]),
        "spearman_r": float(stats.spearmanr(y_true, y_pred)[0]),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _d4_maps(data_root: str):
    s = pd.read_csv(os.path.join(data_root, "D4_hall", "subjects.csv"))
    cgm2d3h = {}
    for _, r in s.dropna(subset=["subject_id", "original_id"]).iterrows():
        cgm2d3h[f"D4_{str(r['original_id']).strip()}"] = str(r["subject_id"]).strip()
    return cgm2d3h


def _extract_windows(data_root: str) -> pd.DataFrame:
    cgm = pd.read_csv(os.path.join(data_root, "D4_hall", "cgm.csv"))
    meals = pd.read_csv(os.path.join(data_root, "D4_hall", "meals.csv"))
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")
    meals["timestamp"] = pd.to_datetime(meals["timestamp"], errors="coerce")
    cgm = cgm.rename(columns={"glucose_mg_dl": "glucose"})
    meals = meals[meals["meal_type"].isin(["Cornflakes", "PB_sandwich", "Protein_bar"])].copy()

    grid = np.arange(-30, 181, 5, dtype=float)
    rows: List[Dict[str, object]] = []
    for _, m in meals.iterrows():
        sid = str(m["subject_id"])
        mt = str(m["meal_type"])
        t0 = m["timestamp"]
        g = cgm[(cgm["subject_id"] == sid) & (cgm["timestamp"] >= t0 + pd.Timedelta(minutes=-30)) & (cgm["timestamp"] <= t0 + pd.Timedelta(minutes=180))].copy()
        if len(g) < 10:
            continue
        t = ((g["timestamp"] - t0).dt.total_seconds() / 60.0).to_numpy(dtype=float)
        y = pd.to_numeric(g["glucose"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(t) & np.isfinite(y)
        if ok.sum() < 10:
            continue
        t = t[ok]
        y = y[ok]
        order = np.argsort(t)
        t = t[order]
        y = y[order]
        y_new = np.interp(grid, t, y)
        rows.append({"subject_id": sid, "meal_type": mt, "cgm_curve": y_new})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out_csv", required=True, type=str)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    input_dim = _infer_input_dim(ckpt["e2e_head_state"])
    model = MechanisticAutoencoder(
        meal_size=6, demographics_size=3, embedding_size=8, hidden_size=32, num_layers=2, encoder_dropout_prob=0.0, decoder_dropout_prob=0.5
    )
    ms = ckpt.get("model_state", {})
    if hasattr(model, "use_v8_recon_corr"):
        model.use_v8_recon_corr = bool(ckpt.get("P1_V8_RECON_CORR", any(k.startswith("correction_mlp.") for k in ms.keys())))
    if hasattr(model, "use_v8_ode_corr"):
        model.use_v8_ode_corr = bool(ckpt.get("P1_V8_ODE_CORR", any(k.startswith("ode_correction.") for k in ms.keys())))
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    head = _build_e2e_head(input_dim)
    head.load_state_dict(ckpt["e2e_head_state"], strict=True)
    head.eval()

    w = _extract_windows(args.data_root)
    if w.empty:
        pd.DataFrame().to_csv(args.out_csv, index=False)
        return

    # minimal meal/demographics placeholders (aligned with run_p1 loader's dimensions)
    T = 43
    n = len(w)
    cgm = np.stack(w["cgm_curve"].to_list()).astype(np.float32)[:, :, None]
    ts = np.tile(np.arange(-30, 181, 5, dtype=np.float32)[None, :, None], (n, 1, 1))
    meals = np.zeros((n, T, 6), dtype=np.float32)
    demo = np.zeros((n, 3), dtype=np.float32)

    train_mean = ckpt["train_mean"]
    train_std = ckpt["train_std"]
    cgm = (cgm - train_mean[0]) / (train_std[0] + 1e-8)
    ts = (ts - train_mean[1]) / (train_std[1] + 1e-8)
    meals = (meals - train_mean[2]) / (train_std[2] + 1e-8)
    demo = (demo - train_mean[3]) / (train_std[3] + 1e-8)

    with torch.no_grad():
        p26, init26, z16 = model.get_all_latents(
            torch.tensor(cgm), torch.tensor(ts), torch.tensor(meals), torch.tensor(demo)
        )
        h_in = torch.cat([p26, init26], dim=-1) if input_dim == 10 else torch.cat([p26, init26, z16], dim=-1)
        y2 = head(h_in).numpy()
    sspg_hat = y2[:, 0]
    di_hat = y2[:, 1]
    if bool(ckpt.get("P1_ZSCORE_TARGETS", False)):
        sspg_hat = sspg_hat * float(ckpt["sspg_std"]) + float(ckpt["sspg_mean"])
        di_hat = di_hat * float(ckpt["di_std"]) + float(ckpt["di_mean"])

    pred = w[["subject_id", "meal_type"]].copy()
    pred["sspg_hat"] = sspg_hat
    pred["di_hat"] = di_hat

    cgm2d3h = _d4_maps(args.data_root)
    pred["subject_id"] = pred["subject_id"].map(lambda x: cgm2d3h.get(x, x))
    lab = pd.read_csv(os.path.join(args.data_root, "D4_hall", "labels.csv"))
    if "SSPG" in lab.columns and "sspg" not in lab.columns:
        lab["sspg"] = lab["SSPG"]
    if "DI" in lab.columns and "di" not in lab.columns:
        lab["di"] = lab["DI"]
    lab = lab[["subject_id", "sspg", "di"]].drop_duplicates("subject_id")

    rows = []
    for mt in ["Cornflakes", "PB_sandwich", "Protein_bar"]:
        sub = pred[pred["meal_type"] == mt].groupby("subject_id")[["sspg_hat", "di_hat"]].mean().reset_index()
        m = sub.merge(lab, on="subject_id", how="left")
        ms = _metrics(m["sspg"].to_numpy(), m["sspg_hat"].to_numpy())
        md = _metrics(m["di"].to_numpy(), m["di_hat"].to_numpy())
        rows.append(
            {
                "meal_type": mt,
                "n_sspg": ms["n"],
                "sspg_pearson_r": ms["pearson_r"],
                "sspg_spearman_r": ms["spearman_r"],
                "sspg_r2": ms["r2"],
                "sspg_rmse": ms["rmse"],
                "sspg_mae": ms["mae"],
                "n_di": md["n"],
                "di_pearson_r": md["pearson_r"],
                "di_spearman_r": md["spearman_r"],
                "di_r2": md["r2"],
                "di_rmse": md["rmse"],
                "di_mae": md["mae"],
            }
        )
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()

