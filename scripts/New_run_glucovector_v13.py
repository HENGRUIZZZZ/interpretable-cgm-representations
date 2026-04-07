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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from load_cgm_project_data import DEMOGRAPHICS_COVARIATES, MEAL_COVARIATES, load_cgm_project_level3
from models import MechanisticAutoencoder
from scripts.New_eval_trainD1D2_testD4 import train_on_d1d2


DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v13")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) < 3:
        return {"n": int(len(y_true)), "pearson_r": float("nan"), "spearman_r": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    pr, pp = stats.pearsonr(y_true, y_pred)
    sr, sp = stats.spearmanr(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _subject_map_d4() -> Dict[str, str]:
    s = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "subjects.csv"))
    return {f"D4_{str(r['original_id']).strip()}": str(r["subject_id"]).strip() for _, r in s.dropna(subset=["subject_id", "original_id"]).iterrows()}


def _load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = MechanisticAutoencoder(
        meal_size=len(MEAL_COVARIATES),
        demographics_size=len(DEMOGRAPHICS_COVARIATES),
        embedding_size=8,
        hidden_size=32,
        num_layers=2,
        encoder_dropout_prob=0.0,
        decoder_dropout_prob=0.5,
    )
    if hasattr(m, "use_v8_recon_corr"):
        m.use_v8_recon_corr = bool(ck.get("P1_V8_RECON_CORR", False))
    if hasattr(m, "use_v8_ode_corr"):
        m.use_v8_ode_corr = bool(ck.get("P1_V8_ODE_CORR", False))
    m.load_state_dict(ck["model_state"], strict=False)
    m.eval()
    return ck, m


def _extract_d4_latent_26d(ckpt_path: str) -> pd.DataFrame:
    ck, model = _load_model(ckpt_path)
    b, info, lab = load_cgm_project_level3(dataset_id="D4", output_base=DATA_ROOT)
    x0 = (b.cgm - ck["train_mean"][0]) / (ck["train_std"][0] + 1e-8)
    x1 = (b.timestamps - ck["train_mean"][1]) / (ck["train_std"][1] + 1e-8)
    x2 = (b.meals - ck["train_mean"][2]) / (ck["train_std"][2] + 1e-8)
    x3 = (b.demographics - ck["train_mean"][3]) / (ck["train_std"][3] + 1e-8)
    with torch.no_grad():
        p26, z0, z16 = model.get_all_latents(
            torch.tensor(x0, dtype=torch.float32),
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(x3, dtype=torch.float32),
        )
    p = p26.numpy()
    z_i = z0.numpy()
    z_n = z16.numpy()
    sid_map = _subject_map_d4()
    rows = []
    for i, sid in enumerate(info.patient_ids):
        r = {"subject_id": sid_map.get(str(sid), str(sid))}
        for j, n in enumerate(["tau_m", "Gb", "sg", "si", "p2", "mi"]):
            r[n] = float(p[i, j])
        for j in range(4):
            r[f"z_init_{j}"] = float(z_i[i, j])
        for j in range(16):
            r[f"z_nonseq_{j}"] = float(z_n[i, j])
        rows.append(r)
    d = pd.DataFrame(rows)
    if "SSPG" in lab.columns and "sspg" not in lab.columns:
        lab["sspg"] = lab["SSPG"]
    if "DI" in lab.columns and "di" not in lab.columns:
        lab["di"] = lab["DI"]
    gold = lab[["subject_id", "sspg", "di"]].drop_duplicates("subject_id")
    return d.merge(gold, on="subject_id", how="left")


def _build_train_latent_and_gold(train_dir: str, out_csv: str) -> pd.DataFrame:
    l26 = pd.read_csv(os.path.join(train_dir, "latent_and_gold_all_26d.csv"))
    lall = pd.read_csv(os.path.join(train_dir, "latent_and_gold_all.csv"))
    keep = ["subject_id", "dataset_id", "sspg", "di", "si", "mi", "tau_m", "Gb", "sg", "p2"] + [f"z_nonseq_{i}" for i in range(16)]
    base = l26.copy()
    if "dataset_id" not in base.columns and "dataset_id" in lall.columns:
        did = lall[["subject_id", "dataset_id"]].drop_duplicates("subject_id")
        base = base.merge(did, on="subject_id", how="left")
    # subject-level one-row table
    num_cols = [c for c in base.columns if c != "subject_id" and pd.api.types.is_numeric_dtype(base[c])]
    agg_num = base.groupby("subject_id", as_index=False)[num_cols].mean()
    if "dataset_id" in base.columns:
        agg_did = base.groupby("subject_id", as_index=False)["dataset_id"].first()
        agg = agg_num.merge(agg_did, on="subject_id", how="left")
    else:
        agg = agg_num
    for c in keep:
        if c not in agg.columns:
            agg[c] = np.nan
    out = agg[keep].copy()
    out.to_csv(out_csv, index=False)
    return out


def _fit_ridge_and_predict(train_df: pd.DataFrame, d4_df: pd.DataFrame, out_csv: str) -> Dict[str, Dict[str, float]]:
    feat_cols = ["si", "mi", "tau_m", "Gb", "sg", "p2"] + [f"z_nonseq_{i}" for i in range(16)]
    tr = train_df.dropna(subset=feat_cols).copy()
    d4 = d4_df.dropna(subset=feat_cols).copy()
    out = d4[["subject_id", "sspg", "di"]].copy().rename(columns={"sspg": "sspg_true", "di": "di_true"})

    # SSPG
    tr_s = tr.dropna(subset=["sspg"])
    mdl_s = RidgeCV(alphas=np.logspace(-4, 3, 20), cv=min(5, len(tr_s))).fit(tr_s[feat_cols].to_numpy(), tr_s["sspg"].to_numpy())
    out["sspg_pred"] = mdl_s.predict(d4[feat_cols].to_numpy())
    # DI
    tr_d = tr.dropna(subset=["di"])
    mdl_d = RidgeCV(alphas=np.logspace(-4, 3, 20), cv=min(5, len(tr_d))).fit(tr_d[feat_cols].to_numpy(), tr_d["di"].to_numpy())
    out["di_pred"] = mdl_d.predict(d4[feat_cols].to_numpy())

    # keep canonical n=16 (both labels present in external set)
    out = out.dropna(subset=["sspg_true", "di_true"]).drop_duplicates("subject_id")
    out.to_csv(out_csv, index=False)
    m_sspg = _metrics(out["sspg_true"].to_numpy(), out["sspg_pred"].to_numpy())
    m_di = _metrics(out["di_true"].to_numpy(), out["di_pred"].to_numpy())
    return {"sspg": m_sspg, "di": m_di}


def _run_exp(name: str, lambda_sspg: float, lambda_di: float, env: Dict[str, str]) -> Dict[str, object]:
    exp_dir = os.path.join(OUT_ROOT, name)
    train_dir = os.path.join(exp_dir, "train")
    os.makedirs(exp_dir, exist_ok=True)
    ckpt = os.path.join(train_dir, "autoencoder_p1_full.pt")
    if not os.path.isfile(ckpt):
        train_on_d1d2(
            cgm_project_output=DATA_ROOT,
            results_dir=train_dir,
            seed=21,
            lambda_sspg=lambda_sspg,
            lambda_di=lambda_di,
            num_epochs=100,
            extra_env=env,
        )
    train_csv = os.path.join(exp_dir, "train_latent_and_gold.csv")
    train_df = _build_train_latent_and_gold(train_dir, train_csv)
    d4_lat = _extract_d4_latent_26d(ckpt)
    d4_pred_csv = os.path.join(exp_dir, "D4_predictions.csv")
    mets = _fit_ridge_and_predict(train_df, d4_lat, d4_pred_csv)
    with open(os.path.join(exp_dir, "D4_metrics.json"), "w") as f:
        json.dump(mets, f, indent=2)
    # copy training metrics requested by v13
    src_tm = os.path.join(train_dir, "training_metrics.json")
    if os.path.isfile(src_tm):
        import shutil
        shutil.copy2(src_tm, os.path.join(exp_dir, "training_metrics.json"))
    return {"name": name, "metrics": mets, "exp_dir": exp_dir}


def run_v13() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res = []

    res.append(
        _run_exp(
            name="Exp1_ODE10_Unsupervised_Baseline",
            lambda_sspg=0.0,
            lambda_di=0.0,
            env={
                "P1_V8_ODE_CORR": "0",
                "P1_V8_HEAD_10D": "1",
                "P1_V5_PREDICTION_HEAD": "0",
                "P1_HEAD_USE_26D": "0",
                "P1_SEPARATE_HEAD_26D": "0",
                "P1_ZSCORE_TARGETS": "1",
                "LAMBDA_IR": "0.0",
                "LAMBDA_CLS": "0.0",
                "LAMBDA_ORTHO": "0.0",
                "P1_IDENT_LOSS_LAMBDA": "0.0",
                "P1_WIDE_PARAM_RANGE": "1",
                "P1_V10_WIDE_BOUNDS": "1",
            },
        )
    )
    res.append(
        _run_exp(
            name="Exp2_ODE10_WeakSupervision",
            lambda_sspg=0.1,
            lambda_di=0.1,
            env={
                "P1_V8_ODE_CORR": "0",
                "P1_V8_HEAD_10D": "1",
                "P1_V5_PREDICTION_HEAD": "0",
                "P1_HEAD_USE_26D": "0",
                "P1_SEPARATE_HEAD_26D": "0",
                "P1_ZSCORE_TARGETS": "1",
                "LAMBDA_IR": "0.0",
                "LAMBDA_CLS": "0.0",
                "LAMBDA_ORTHO": "0.0",
                "P1_IDENT_LOSS_LAMBDA": "0.1",
                "P1_WIDE_PARAM_RANGE": "1",
                "P1_V10_WIDE_BOUNDS": "1",
            },
        )
    )
    res.append(
        _run_exp(
            name="Exp3_Hybrid26_WeakSupervision_Ortho",
            lambda_sspg=0.1,
            lambda_di=0.1,
            env={
                "P1_V8_ODE_CORR": "1",
                "P1_V8_HEAD_10D": "0",
                "P1_V5_PREDICTION_HEAD": "0",
                "P1_HEAD_USE_26D": "0",
                "P1_SEPARATE_HEAD_26D": "1",
                "P1_ZSCORE_TARGETS": "1",
                "LAMBDA_IR": "0.0",
                "LAMBDA_CLS": "0.0",
                "LAMBDA_ORTHO": "0.1",
                "P1_IDENT_LOSS_LAMBDA": "0.1",
                "P1_DETACH_HEAD_INPUT": "1",
                "P1_HEAD_GRAD_SCALE": "0.01",
                "P1_WIDE_PARAM_RANGE": "1",
                "P1_V10_WIDE_BOUNDS": "1",
            },
        )
    )

    rows: List[Dict[str, float]] = []
    for r in res:
        rows.append(
            {
                "experiment": r["name"],
                "sspg_pearson_r": r["metrics"]["sspg"]["pearson_r"],
                "sspg_rmse": r["metrics"]["sspg"]["rmse"],
                "di_pearson_r": r["metrics"]["di"]["pearson_r"],
                "di_rmse": r["metrics"]["di"]["rmse"],
            }
        )
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT_ROOT, "v13_summary.csv"), index=False)
    rep = ["# v13_report", "", f"run_stamp: {stamp}", "", s.to_markdown(index=False)]
    with open(os.path.join(OUT_ROOT, "v13_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))

    zip_name = f"{os.path.basename(OUT_ROOT)}.zip"
    subprocess_cmd = f'cd "{REPO_ROOT}" && zip -r "{zip_name}" "{os.path.basename(OUT_ROOT)}" >/dev/null'
    os.system(subprocess_cmd)


if __name__ == "__main__":
    run_v13()

