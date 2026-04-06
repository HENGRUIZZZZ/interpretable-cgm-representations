from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from load_cgm_project_data import load_cgm_project_level1_level2, load_cgm_project_level3
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


def _metrics(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y = y[ok]
    yhat = yhat[ok]
    if len(y) < 3:
        return {"n": int(len(y)), "pearson_r": np.nan, "spearman_r": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    return {
        "n": int(len(y)),
        "pearson_r": float(stats.pearsonr(y, yhat)[0]),
        "spearman_r": float(stats.spearmanr(y, yhat)[0]),
        "r2": float(r2_score(y, yhat)),
        "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
        "mae": float(mean_absolute_error(y, yhat)),
    }


def _lodo(df: pd.DataFrame, feats: List[str], target: str):
    all_y, all_yhat = [], []
    for test in ["D1", "D2", "D4"]:
        tr = df[df["dataset_id"] != test].dropna(subset=feats + [target]).copy()
        te = df[df["dataset_id"] == test].dropna(subset=feats + [target]).copy()
        if len(tr) < 10 or len(te) < 5:
            continue
        reg = Pipeline([("sc", StandardScaler()), ("rg", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
        reg.fit(tr[feats].to_numpy(dtype=float), tr[target].to_numpy(dtype=float))
        yhat = reg.predict(te[feats].to_numpy(dtype=float))
        all_y.append(te[target].to_numpy(dtype=float))
        all_yhat.append(yhat)
    if not all_y:
        return {"n": 0, "pearson_r": np.nan, "spearman_r": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    return _metrics(np.concatenate(all_y), np.concatenate(all_yhat))


def _tier1(data_root: str) -> pd.DataFrame:
    rows = []
    # D1
    d1 = pd.read_csv(os.path.join(data_root, "D1_metwally", "cgm.csv")).rename(columns={"glucose_mg_dl": "g"})
    for sid, g in d1.groupby("subject_id"):
        x = pd.to_numeric(g["g"], errors="coerce").to_numpy(dtype=float)
        rows.append({"subject_id": sid, "dataset_id": "D1", "cgm_mean": np.nanmean(x), "cgm_var": np.nanvar(x), "tir_70_140": np.nanmean((x >= 70) & (x <= 140))})
    # D2
    d2 = pd.read_csv(os.path.join(data_root, "D2_stanford", "cgm.csv")).rename(columns={"glucose_mg_dl": "g"})
    for sid, g in d2.groupby("subject_id"):
        x = pd.to_numeric(g["g"], errors="coerce").to_numpy(dtype=float)
        rows.append({"subject_id": sid, "dataset_id": "D2", "cgm_mean": np.nanmean(x), "cgm_var": np.nanvar(x), "tir_70_140": np.nanmean((x >= 70) & (x <= 140))})
    # D4 standard cgm
    d4 = pd.read_csv(os.path.join(data_root, "D4_hall", "cgm.csv")).rename(columns={"glucose_mg_dl": "g"})
    for sid, g in d4.groupby("subject_id"):
        x = pd.to_numeric(g["g"], errors="coerce").to_numpy(dtype=float)
        rows.append({"subject_id": sid, "dataset_id": "D4", "cgm_mean": np.nanmean(x), "cgm_var": np.nanvar(x), "tir_70_140": np.nanmean((x >= 70) & (x <= 140))})
    feat = pd.DataFrame(rows)

    l1 = pd.read_csv(os.path.join(data_root, "D1_metwally", "labels.csv"))
    if "SSPG" in l1.columns and "sspg" not in l1.columns:
        l1["sspg"] = l1["SSPG"]
    if "DI" in l1.columns and "di" not in l1.columns:
        l1["di"] = l1["DI"]
    l1 = l1[["subject_id", "sspg", "di"]]
    l2 = pd.read_csv(os.path.join(data_root, "D2_stanford", "labels.csv"))[["subject_id", "SSPG", "DI"]].rename(columns={"SSPG": "sspg", "DI": "di"})
    l4 = pd.read_csv(os.path.join(data_root, "D4_hall", "labels.csv"))[["subject_id", "sspg", "DI"]].rename(columns={"DI": "di"})
    labels = pd.concat([l1, l2, l4], ignore_index=True).drop_duplicates("subject_id")

    # map D4 cgm IDs -> D3H label IDs
    s = pd.read_csv(os.path.join(data_root, "D4_hall", "subjects.csv"))
    idmap = {f"D4_{str(r['original_id']).strip()}": str(r["subject_id"]).strip() for _, r in s.dropna(subset=["subject_id", "original_id"]).iterrows()}
    feat["subject_id_lab"] = feat.apply(lambda r: idmap.get(r["subject_id"], r["subject_id"]) if r["dataset_id"] == "D4" else r["subject_id"], axis=1)
    out = feat.merge(labels, left_on="subject_id_lab", right_on="subject_id", how="left", suffixes=("", "_lab"))
    return out


def _tier234(data_root: str, ckpt_path: str) -> pd.DataFrame:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MechanisticAutoencoder(meal_size=6, demographics_size=3, embedding_size=8, hidden_size=32, num_layers=2, encoder_dropout_prob=0.0, decoder_dropout_prob=0.5)
    ms = ckpt.get("model_state", {})
    if hasattr(model, "use_v8_recon_corr"):
        model.use_v8_recon_corr = bool(ckpt.get("P1_V8_RECON_CORR", any(k.startswith("correction_mlp.") for k in ms.keys())))
    if hasattr(model, "use_v8_ode_corr"):
        model.use_v8_ode_corr = bool(ckpt.get("P1_V8_ODE_CORR", any(k.startswith("ode_correction.") for k in ms.keys())))
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    input_dim = _infer_input_dim(ckpt["e2e_head_state"])
    head = _build_e2e_head(input_dim)
    head.load_state_dict(ckpt["e2e_head_state"], strict=True)
    head.eval()
    train_mean, train_std = ckpt["train_mean"], ckpt["train_std"]

    frames = []
    for did in ["D1", "D2", "D4"]:
        if did in ("D1", "D2"):
            b, info, lab = load_cgm_project_level1_level2(dataset_id=did, output_base=data_root, num_meals_threshold=1)
        else:
            b, info, lab = load_cgm_project_level3(dataset_id="D4", output_base=data_root)
        cgm = (b.cgm - train_mean[0]) / (train_std[0] + 1e-8)
        ts = (b.timestamps - train_mean[1]) / (train_std[1] + 1e-8)
        meals = (b.meals - train_mean[2]) / (train_std[2] + 1e-8)
        demo = (b.demographics - train_mean[3]) / (train_std[3] + 1e-8)
        with torch.no_grad():
            p26, init26, z16 = model.get_all_latents(torch.tensor(cgm, dtype=torch.float32), torch.tensor(ts, dtype=torch.float32), torch.tensor(meals, dtype=torch.float32), torch.tensor(demo, dtype=torch.float32))
        x = np.concatenate([p26.numpy(), init26.numpy(), z16.numpy()], axis=1)
        cols = ["tau_m", "Gb", "sg", "si", "p2", "mi"] + [f"z_init_{i}" for i in range(4)] + [f"z_nonseq_{i}" for i in range(16)]
        d = pd.DataFrame(x, columns=cols)
        d["subject_id"] = np.asarray(info.patient_ids).astype(str)
        d = d.groupby("subject_id").mean(numeric_only=True).reset_index()
        if "sspg" not in lab.columns and "SSPG" in lab.columns:
            lab["sspg"] = lab["SSPG"]
        if "di" not in lab.columns and "DI" in lab.columns:
            lab["di"] = lab["DI"]
        g = lab[["subject_id", "sspg", "di"]].dropna(how="all").drop_duplicates("subject_id")
        if did == "D4":
            s = pd.read_csv(os.path.join(data_root, "D4_hall", "subjects.csv"))
            idmap = {f"D4_{str(r['original_id']).strip()}": str(r["subject_id"]).strip() for _, r in s.dropna(subset=["subject_id", "original_id"]).iterrows()}
            d["subject_id_lab"] = d["subject_id"].map(lambda z: idmap.get(z, z))
            g2 = g.rename(columns={"subject_id": "subject_id_lab"})
            d = d.merge(g2, on="subject_id_lab", how="left")
        else:
            d = d.merge(g, on="subject_id", how="left")
        d["dataset_id"] = did
        # demographics
        sfile = "D1_metwally" if did == "D1" else ("D2_stanford" if did == "D2" else "D4_hall")
        s = pd.read_csv(os.path.join(data_root, sfile, "subjects.csv"))
        keep = [c for c in ["subject_id", "age", "bmi", "sex"] if c in s.columns]
        s = s[keep].drop_duplicates("subject_id")
        if did == "D4":
            s = s.rename(columns={"subject_id": "subject_id_lab"})
            s["subject_id_lab"] = s["subject_id_lab"].astype(str)
            d = d.merge(s, on="subject_id_lab", how="left", suffixes=("", "_demo"))
        else:
            d = d.merge(s, on="subject_id", how="left", suffixes=("", "_demo"))
        if "sex" in d.columns:
            d["sex"] = d["sex"].astype(str).str.upper().str.startswith("F").astype(float)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out_csv", required=True, type=str)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    t1 = _tier1(args.data_root)
    t234 = _tier234(args.data_root, args.ckpt)

    tier_defs = [
        ("Tier 1: CGM 统计特征", t1, ["cgm_mean", "cgm_var", "tir_70_140"]),
        ("Tier 2: 6D ODE 参数", t234, ["tau_m", "Gb", "sg", "si", "p2", "mi"]),
        ("Tier 3: 26D 全潜变量", t234, ["tau_m", "Gb", "sg", "si", "p2", "mi"] + [f"z_init_{i}" for i in range(4)] + [f"z_nonseq_{i}" for i in range(16)]),
        ("Tier 4: 26D + Demographics", t234, ["tau_m", "Gb", "sg", "si", "p2", "mi"] + [f"z_init_{i}" for i in range(4)] + [f"z_nonseq_{i}" for i in range(16)] + ["age", "bmi", "sex"]),
    ]
    rows: List[Dict[str, float]] = []
    for name, df, feats in tier_defs:
        ms = _lodo(df, feats, "sspg")
        md = _lodo(df, feats, "di")
        rows.append(
            {
                "tier": name,
                "sspg_n": ms["n"],
                "sspg_pearson_r": ms["pearson_r"],
                "sspg_spearman_r": ms["spearman_r"],
                "sspg_r2": ms["r2"],
                "sspg_rmse": ms["rmse"],
                "sspg_mae": ms["mae"],
                "di_n": md["n"],
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

