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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from load_cgm_project_data import DEMOGRAPHICS_COVARIATES, MEAL_COVARIATES, load_cgm_project_level3
from models import MechanisticAutoencoder
from scripts.New_eval_trainD1D2_testD4 import eval_ckpt_on_d4, train_on_d1d2


DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v15")


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


def _build_train_latent(train_dir: str, out_csv: str) -> pd.DataFrame:
    l26 = pd.read_csv(os.path.join(train_dir, "latent_and_gold_all_26d.csv"))
    lall = pd.read_csv(os.path.join(train_dir, "latent_and_gold_all.csv"))
    base = l26.copy()
    if "dataset_id" not in base.columns and "dataset_id" in lall.columns:
        did = lall[["subject_id", "dataset_id"]].drop_duplicates("subject_id")
        base = base.merge(did, on="subject_id", how="left")
    keep = ["subject_id", "dataset_id", "sspg", "di", "si", "mi", "tau_m", "Gb", "sg", "p2"] + [f"z_nonseq_{i}" for i in range(16)]
    num_cols = [c for c in base.columns if c != "subject_id" and pd.api.types.is_numeric_dtype(base[c])]
    agg_num = base.groupby("subject_id", as_index=False)[num_cols].mean()
    if "dataset_id" in base.columns:
        agg = agg_num.merge(base.groupby("subject_id", as_index=False)["dataset_id"].first(), on="subject_id", how="left")
    else:
        agg = agg_num
    for c in keep:
        if c not in agg.columns:
            agg[c] = np.nan
    out = agg[keep].copy()
    out.to_csv(out_csv, index=False)
    return out


def _ridge_eval(train_df: pd.DataFrame, d4_df: pd.DataFrame, out_csv: str) -> Dict[str, Dict[str, float]]:
    feat_cols = ["si", "mi", "tau_m", "Gb", "sg", "p2"] + [f"z_nonseq_{i}" for i in range(16)]
    tr = train_df.dropna(subset=feat_cols).copy()
    d4 = d4_df.dropna(subset=feat_cols).copy()
    out = d4[["subject_id", "sspg", "di"]].copy().rename(columns={"sspg": "sspg_true", "di": "di_true"})
    tr_s = tr.dropna(subset=["sspg"])
    tr_d = tr.dropna(subset=["di"])
    mdl_s = RidgeCV(alphas=np.logspace(-4, 3, 20), cv=min(5, len(tr_s))).fit(tr_s[feat_cols].to_numpy(), tr_s["sspg"].to_numpy())
    mdl_d = RidgeCV(alphas=np.logspace(-4, 3, 20), cv=min(5, len(tr_d))).fit(tr_d[feat_cols].to_numpy(), tr_d["di"].to_numpy())
    out["sspg_pred"] = mdl_s.predict(d4[feat_cols].to_numpy())
    out["di_pred"] = mdl_d.predict(d4[feat_cols].to_numpy())
    out = out.dropna(subset=["sspg_true", "di_true"]).drop_duplicates("subject_id")
    out.to_csv(out_csv, index=False)
    return {"sspg": _metrics(out["sspg_true"], out["sspg_pred"]), "di": _metrics(out["di_true"], out["di_pred"])}


def _param_diag(latent_csv: str) -> Dict[str, float]:
    d = pd.read_csv(latent_csv)
    out: Dict[str, float] = {}
    for k in ("tau_m", "Gb", "si", "mi"):
        v = pd.to_numeric(d[k], errors="coerce").to_numpy(dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(v))
        out[f"{k}_std"] = float(np.nanstd(v))
        out[f"{k}_cv"] = float(np.nanstd(v) / (abs(np.nanmean(v)) + 1e-8))
    return out


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


def _multi_meal_icc(latent_csv: str, out_json: str) -> None:
    d = pd.read_csv(latent_csv)
    out: Dict[str, float] = {}
    for p in ("tau_m", "Gb", "si", "mi"):
        grp = d[["subject_id", p]].dropna().groupby("subject_id")[p].apply(list)
        k_min = min((len(v) for v in grp.values), default=0)
        k_use = min(3, k_min)
        if k_use < 2:
            out[f"icc_{p}"] = float("nan")
            continue
        mat = []
        for v in grp.values:
            if len(v) >= k_use:
                mat.append(v[:k_use])
        out[f"icc_{p}"] = _icc_oneway(np.asarray(mat, dtype=float)) if len(mat) >= 3 else float("nan")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)


def _base_env() -> Dict[str, str]:
    return {
        "P1_LR": "1e-2",
        "P1_SEED": "42",
        "P1_USE_LR_SCHEDULER": "1",
        "P1_SAVE_26D_LATENT": "1",
        "P1_ZSCORE_TARGETS": "1",
        "P1_V8_ODE_CORR": "0",
        "P1_V8_RECON_CORR": "0",
        "LAMBDA_CLS": "0.0",
    }


def _train_if_needed(train_dir: str, lambda_sspg: float, lambda_di: float, num_epochs: int, env: Dict[str, str]) -> str:
    ckpt = os.path.join(train_dir, "autoencoder_p1_full.pt")
    if not os.path.isfile(ckpt):
        train_on_d1d2(
            cgm_project_output=DATA_ROOT,
            results_dir=train_dir,
            seed=42,
            lambda_sspg=lambda_sspg,
            lambda_di=lambda_di,
            num_epochs=num_epochs,
            extra_env=env,
        )
    return ckpt


def _evaluate_exp(exp_dir: str, train_dir: str, ckpt: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, float]]:
    os.makedirs(exp_dir, exist_ok=True)
    # e2e eval (if head exists)
    e2e_metrics: Dict[str, Dict[str, float]] = {}
    if torch.load(ckpt, map_location="cpu", weights_only=False).get("e2e_head_state") is not None:
        d4_dir = os.path.join(exp_dir, "d4_eval_e2e")
        os.makedirs(d4_dir, exist_ok=True)
        m_s, _ = eval_ckpt_on_d4(cgm_project_output=DATA_ROOT, ckpt_path=ckpt, out_dir=d4_dir, target="sspg")
        m_d, _ = eval_ckpt_on_d4(cgm_project_output=DATA_ROOT, ckpt_path=ckpt, out_dir=d4_dir, target="di")
        e2e_metrics = {"sspg": m_s.__dict__, "di": m_d.__dict__}
        with open(os.path.join(exp_dir, "D4_metrics_e2e.json"), "w") as f:
            json.dump(e2e_metrics, f, indent=2)
    # ridge eval
    train_lat = _build_train_latent(train_dir, os.path.join(exp_dir, "train_latent_and_gold.csv"))
    d4_lat = _extract_d4_latent_26d(ckpt)
    ridge_metrics = _ridge_eval(train_lat, d4_lat, os.path.join(exp_dir, "D4_predictions_ridge.csv"))
    with open(os.path.join(exp_dir, "D4_metrics_ridge.json"), "w") as f:
        json.dump(ridge_metrics, f, indent=2)
    diag = _param_diag(os.path.join(train_dir, "latent_and_gold_all_26d.csv"))
    with open(os.path.join(exp_dir, "param_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)
    return e2e_metrics, ridge_metrics, diag


def run_v15() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    rows: List[Dict[str, float]] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _base_env()

    # Exp1: baseline 10D unsupervised + multi-meal
    name = "v15_Exp1_Baseline_MultiMeal"
    exp_dir = os.path.join(OUT_ROOT, name)
    train_dir = os.path.join(exp_dir, "train")
    env1 = dict(base)
    env1.update({"P1_HEAD_USE_26D": "1", "P1_V8_HEAD_10D": "1", "P1_SEPARATE_HEAD_26D": "0", "P1_V10_WIDE_BOUNDS": "0", "P1_ONE_MEAL_PER_SUBJECT": "0", "LAMBDA_IR": "0.0"})
    ck1 = _train_if_needed(train_dir, 0.0, 0.0, 100, env1)
    with open(os.path.join(exp_dir, "config_snapshot.json"), "w") as f:
        json.dump({"name": name, "env": env1, "lambda_sspg": 0.0, "lambda_di": 0.0}, f, indent=2)
    e2e1, ridge1, d1 = _evaluate_exp(exp_dir, train_dir, ck1)
    _multi_meal_icc(os.path.join(train_dir, "latent_and_gold_all_26d.csv"), os.path.join(exp_dir, "multi_meal_icc.json"))
    rows.append({"experiment": name, "mode": "ridge", "sspg_r": ridge1["sspg"]["pearson_r"], "sspg_rmse": ridge1["sspg"]["rmse"], "di_r": ridge1["di"]["pearson_r"], "di_rmse": ridge1["di"]["rmse"], "si_cv": d1["si_cv"], "mi_cv": d1["mi_cv"], "Gb_cv": d1["Gb_cv"]})
    if e2e1:
        rows.append({"experiment": name, "mode": "e2e", "sspg_r": e2e1["sspg"]["pearson_r"], "sspg_rmse": e2e1["sspg"]["rmse"], "di_r": e2e1["di"]["pearson_r"], "di_rmse": e2e1["di"]["rmse"], "si_cv": d1["si_cv"], "mi_cv": d1["mi_cv"], "Gb_cv": d1["Gb_cv"]})

    # Exp2: ConfigD + narrow + e2e + multi-meal
    name = "v15_Exp2_ConfigD_Narrow_MultiMeal"
    exp_dir = os.path.join(OUT_ROOT, name)
    train_dir = os.path.join(exp_dir, "train")
    env2 = dict(base)
    env2.update({"P1_HEAD_USE_26D": "1", "P1_V8_HEAD_10D": "0", "P1_SEPARATE_HEAD_26D": "0", "P1_V10_WIDE_BOUNDS": "0", "P1_ONE_MEAL_PER_SUBJECT": "0", "LAMBDA_IR": "0.05"})
    ck2 = _train_if_needed(train_dir, 0.1, 0.1, 100, env2)
    with open(os.path.join(exp_dir, "config_snapshot.json"), "w") as f:
        json.dump({"name": name, "env": env2, "lambda_sspg": 0.1, "lambda_di": 0.1}, f, indent=2)
    e2e2, ridge2, d2 = _evaluate_exp(exp_dir, train_dir, ck2)
    _multi_meal_icc(os.path.join(train_dir, "latent_and_gold_all_26d.csv"), os.path.join(exp_dir, "multi_meal_icc.json"))
    rows.append({"experiment": name, "mode": "ridge", "sspg_r": ridge2["sspg"]["pearson_r"], "sspg_rmse": ridge2["sspg"]["rmse"], "di_r": ridge2["di"]["pearson_r"], "di_rmse": ridge2["di"]["rmse"], "si_cv": d2["si_cv"], "mi_cv": d2["mi_cv"], "Gb_cv": d2["Gb_cv"]})
    if e2e2:
        rows.append({"experiment": name, "mode": "e2e", "sspg_r": e2e2["sspg"]["pearson_r"], "sspg_rmse": e2e2["sspg"]["rmse"], "di_r": e2e2["di"]["pearson_r"], "di_rmse": e2e2["di"]["rmse"], "si_cv": d2["si_cv"], "mi_cv": d2["mi_cv"], "Gb_cv": d2["Gb_cv"]})

    # Exp3: ConfigD + wide + two-phase + one-meal
    name = "v15_Exp3_ConfigD_Wide_TwoPhase_OneMeal"
    exp_dir = os.path.join(OUT_ROOT, name)
    p1_dir = os.path.join(exp_dir, "phase1_unsupervised")
    p2_dir = os.path.join(exp_dir, "phase2_finetune_head")
    env3 = dict(base)
    env3.update({"P1_HEAD_USE_26D": "1", "P1_V8_HEAD_10D": "0", "P1_SEPARATE_HEAD_26D": "0", "P1_V10_WIDE_BOUNDS": "1", "P1_ONE_MEAL_PER_SUBJECT": "1", "LAMBDA_IR": "0.0", "P1_FINETUNE_HEAD_ONLY": "0", "P1_PRETRAINED_MODEL": "", "P1_FINETUNE_16D_ONLY": "0"})
    ck3p1 = _train_if_needed(p1_dir, 0.0, 0.0, 100, env3)
    env3p2 = dict(env3)
    env3p2.update({"LAMBDA_IR": "0.05", "P1_FINETUNE_HEAD_ONLY": "1", "P1_PRETRAINED_MODEL": ck3p1, "P1_FINETUNE_16D_ONLY": "0"})
    ck3 = _train_if_needed(p2_dir, 0.1, 0.1, 100, env3p2)
    with open(os.path.join(exp_dir, "config_snapshot.json"), "w") as f:
        json.dump({"name": name, "phase1_env": env3, "phase2_env": env3p2}, f, indent=2)
    e2e3, ridge3, d3 = _evaluate_exp(exp_dir, p2_dir, ck3)
    rows.append({"experiment": name, "mode": "ridge", "sspg_r": ridge3["sspg"]["pearson_r"], "sspg_rmse": ridge3["sspg"]["rmse"], "di_r": ridge3["di"]["pearson_r"], "di_rmse": ridge3["di"]["rmse"], "si_cv": d3["si_cv"], "mi_cv": d3["mi_cv"], "Gb_cv": d3["Gb_cv"]})
    if e2e3:
        rows.append({"experiment": name, "mode": "e2e", "sspg_r": e2e3["sspg"]["pearson_r"], "sspg_rmse": e2e3["sspg"]["rmse"], "di_r": e2e3["di"]["pearson_r"], "di_rmse": e2e3["di"]["rmse"], "si_cv": d3["si_cv"], "mi_cv": d3["mi_cv"], "Gb_cv": d3["Gb_cv"]})

    # Exp4: ConfigD + wide + two-phase + multi-meal
    name = "v15_Exp4_ConfigD_Wide_TwoPhase_MultiMeal"
    exp_dir = os.path.join(OUT_ROOT, name)
    p1_dir = os.path.join(exp_dir, "phase1_unsupervised")
    p2_dir = os.path.join(exp_dir, "phase2_finetune_head")
    env4 = dict(base)
    env4.update({"P1_HEAD_USE_26D": "1", "P1_V8_HEAD_10D": "0", "P1_SEPARATE_HEAD_26D": "0", "P1_V10_WIDE_BOUNDS": "1", "P1_ONE_MEAL_PER_SUBJECT": "0", "LAMBDA_IR": "0.0", "P1_FINETUNE_HEAD_ONLY": "0", "P1_PRETRAINED_MODEL": "", "P1_FINETUNE_16D_ONLY": "0"})
    ck4p1 = _train_if_needed(p1_dir, 0.0, 0.0, 100, env4)
    env4p2 = dict(env4)
    env4p2.update({"LAMBDA_IR": "0.05", "P1_FINETUNE_HEAD_ONLY": "1", "P1_PRETRAINED_MODEL": ck4p1, "P1_FINETUNE_16D_ONLY": "0"})
    ck4 = _train_if_needed(p2_dir, 0.1, 0.1, 100, env4p2)
    with open(os.path.join(exp_dir, "config_snapshot.json"), "w") as f:
        json.dump({"name": name, "phase1_env": env4, "phase2_env": env4p2}, f, indent=2)
    e2e4, ridge4, d4 = _evaluate_exp(exp_dir, p2_dir, ck4)
    _multi_meal_icc(os.path.join(p2_dir, "latent_and_gold_all_26d.csv"), os.path.join(exp_dir, "multi_meal_icc.json"))
    rows.append({"experiment": name, "mode": "ridge", "sspg_r": ridge4["sspg"]["pearson_r"], "sspg_rmse": ridge4["sspg"]["rmse"], "di_r": ridge4["di"]["pearson_r"], "di_rmse": ridge4["di"]["rmse"], "si_cv": d4["si_cv"], "mi_cv": d4["mi_cv"], "Gb_cv": d4["Gb_cv"]})
    if e2e4:
        rows.append({"experiment": name, "mode": "e2e", "sspg_r": e2e4["sspg"]["pearson_r"], "sspg_rmse": e2e4["sspg"]["rmse"], "di_r": e2e4["di"]["pearson_r"], "di_rmse": e2e4["di"]["rmse"], "si_cv": d4["si_cv"], "mi_cv": d4["mi_cv"], "Gb_cv": d4["Gb_cv"]})

    # Exp5: ConfigD + wide + two-phase + multi-meal + 16D finetune
    name = "v15_Exp5_ConfigD_Wide_TwoPhase_16DFinetune"
    exp_dir = os.path.join(OUT_ROOT, name)
    p1_dir = os.path.join(exp_dir, "phase1_unsupervised")
    p2_dir = os.path.join(exp_dir, "phase2_finetune_16d")
    env5 = dict(base)
    env5.update({"P1_HEAD_USE_26D": "1", "P1_V8_HEAD_10D": "0", "P1_SEPARATE_HEAD_26D": "0", "P1_V10_WIDE_BOUNDS": "1", "P1_ONE_MEAL_PER_SUBJECT": "0", "LAMBDA_IR": "0.0", "P1_FINETUNE_HEAD_ONLY": "0", "P1_PRETRAINED_MODEL": "", "P1_FINETUNE_16D_ONLY": "0"})
    ck5p1 = _train_if_needed(p1_dir, 0.0, 0.0, 100, env5)
    env5p2 = dict(env5)
    env5p2.update({"LAMBDA_IR": "0.05", "P1_FINETUNE_HEAD_ONLY": "1", "P1_PRETRAINED_MODEL": ck5p1, "P1_FINETUNE_16D_ONLY": "1"})
    ck5 = _train_if_needed(p2_dir, 0.1, 0.1, 100, env5p2)
    with open(os.path.join(exp_dir, "config_snapshot.json"), "w") as f:
        json.dump({"name": name, "phase1_env": env5, "phase2_env": env5p2}, f, indent=2)
    e2e5, ridge5, d5 = _evaluate_exp(exp_dir, p2_dir, ck5)
    _multi_meal_icc(os.path.join(p2_dir, "latent_and_gold_all_26d.csv"), os.path.join(exp_dir, "multi_meal_icc.json"))
    rows.append({"experiment": name, "mode": "ridge", "sspg_r": ridge5["sspg"]["pearson_r"], "sspg_rmse": ridge5["sspg"]["rmse"], "di_r": ridge5["di"]["pearson_r"], "di_rmse": ridge5["di"]["rmse"], "si_cv": d5["si_cv"], "mi_cv": d5["mi_cv"], "Gb_cv": d5["Gb_cv"]})
    if e2e5:
        rows.append({"experiment": name, "mode": "e2e", "sspg_r": e2e5["sspg"]["pearson_r"], "sspg_rmse": e2e5["sspg"]["rmse"], "di_r": e2e5["di"]["pearson_r"], "di_rmse": e2e5["di"]["rmse"], "si_cv": d5["si_cv"], "mi_cv": d5["mi_cv"], "Gb_cv": d5["Gb_cv"]})

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_ROOT, "v15_summary.csv"), index=False)
    with open(os.path.join(OUT_ROOT, "v15_report.md"), "w", encoding="utf-8") as f:
        f.write("# v15_report\n\n")
        f.write(f"run_stamp: {stamp}\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n")

    # evidence checklist for user verification
    checklist = {
        "guide_requirements_mapped": {
            "exp_count": 5,
            "multi_meal_exps": ["Exp1", "Exp2", "Exp4", "Exp5"],
            "one_meal_exp": ["Exp3"],
            "two_phase_exps": ["Exp3", "Exp4", "Exp5"],
            "finetune_16d_only_exp": ["Exp5"],
        },
        "artifacts": {
            "summary_csv": os.path.join(OUT_ROOT, "v15_summary.csv"),
            "report_md": os.path.join(OUT_ROOT, "v15_report.md"),
        },
    }
    with open(os.path.join(OUT_ROOT, "implementation_audit.json"), "w") as f:
        json.dump(checklist, f, indent=2)

    zip_name = f"{os.path.basename(OUT_ROOT)}.zip"
    os.system(f'cd "{REPO_ROOT}" && zip -r "{zip_name}" "{os.path.basename(OUT_ROOT)}" >/dev/null')


if __name__ == "__main__":
    run_v15()

