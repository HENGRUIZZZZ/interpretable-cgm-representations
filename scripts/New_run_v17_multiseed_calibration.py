from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.New_eval_trainD1D2_testD4 import eval_ckpt_on_d4, train_on_d1d2

DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v17_multiseed")
SEEDS = [7, 13, 21, 42, 84]


@dataclass
class Metric:
    n: int
    pearson_r: float
    spearman_r: float
    r2: float
    rmse: float
    mae: float


def _metrics(y: np.ndarray, yhat: np.ndarray) -> Metric:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    m = np.isfinite(y) & np.isfinite(yhat)
    y = y[m]
    yhat = yhat[m]
    if y.size < 2:
        return Metric(n=int(y.size), pearson_r=float("nan"), spearman_r=float("nan"), r2=float("nan"), rmse=float("nan"), mae=float("nan"))
    pr = float(stats.pearsonr(y, yhat)[0]) if np.std(y) > 0 and np.std(yhat) > 0 else float("nan")
    sr = float(stats.spearmanr(y, yhat)[0]) if np.std(y) > 0 and np.std(yhat) > 0 else float("nan")
    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum((yhat - y) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    mae = float(np.mean(np.abs(yhat - y)))
    return Metric(n=int(y.size), pearson_r=pr, spearman_r=sr, r2=r2, rmse=rmse, mae=mae)


def _base_env() -> Dict[str, str]:
    return {
        "P1_TRAIN_DATASETS": "D1,D2",
        "P1_LR": "1e-2",
        "P1_USE_LR_SCHEDULER": "1",
        "P1_SAVE_26D_LATENT": "1",
        "P1_ZSCORE_TARGETS": "1",
        "P1_V8_ODE_CORR": "0",
        "P1_V8_RECON_CORR": "0",
        "P1_ONE_MEAL_PER_SUBJECT": "1",
        "LAMBDA_CLS": "0.0",
        "LAMBDA_ORTHO_16D": "0.0",
        "LAMBDA_VAR_MATCH": "0.0",
        "P1_FINETUNE_16D_ONLY": "0",
        "P1_HEAD_USE_26D": "1",
        "P1_V8_HEAD_10D": "0",
        "P1_V10_WIDE_BOUNDS": "1",
    }


def _train_two_phase(exp_dir: str, seed: int, env_p2_extra: Dict[str, str]) -> str:
    env_p1 = _base_env()
    p1_dir = os.path.join(exp_dir, "phase1_unsupervised")
    p2_dir = os.path.join(exp_dir, "phase2_finetune_head")
    ck1 = os.path.join(p1_dir, "autoencoder_p1_full.pt")
    ck2 = os.path.join(p2_dir, "autoencoder_p1_full.pt")
    if not os.path.isfile(ck1):
        train_on_d1d2(
            cgm_project_output=DATA_ROOT,
            results_dir=p1_dir,
            seed=seed,
            lambda_sspg=0.0,
            lambda_di=0.0,
            num_epochs=100,
            extra_env=env_p1,
        )
    if not os.path.isfile(ck2):
        env_p2 = dict(env_p1)
        env_p2.update(env_p2_extra)
        env_p2["P1_FINETUNE_HEAD_ONLY"] = "1"
        env_p2["P1_PRETRAINED_MODEL"] = ck1
        train_on_d1d2(
            cgm_project_output=DATA_ROOT,
            results_dir=p2_dir,
            seed=seed,
            lambda_sspg=0.1,
            lambda_di=0.1,
            num_epochs=100,
            extra_env=env_p2,
        )
    return ck2


def _denorm_train_predictions(train_csv: str, ckpt_path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if target == "sspg":
        mu = float(ck.get("sspg_mean", np.nan))
        sd = float(ck.get("sspg_std", np.nan))
        pred_col = "sspg_hat_head"
        true_col = "sspg"
    else:
        mu = float(ck.get("di_mean", np.nan))
        sd = float(ck.get("di_std", np.nan))
        pred_col = "di_hat_head"
        true_col = "di"
    out = df[["subject_id", true_col, pred_col]].dropna().copy()
    out["pred_denorm"] = out[pred_col].astype(float) * sd + mu
    out = out.groupby("subject_id", as_index=False)[[true_col, "pred_denorm"]].mean()
    out = out.rename(columns={true_col: "true", "pred_denorm": "pred"})
    return out


def _fit_linear_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.asarray(y_true[m], dtype=float)
    y_pred = np.asarray(y_pred[m], dtype=float)
    if y_true.size < 2:
        return 1.0, 0.0
    a, b = np.polyfit(y_pred, y_true, deg=1)
    return float(a), float(b)


def _apply_calibration(y_pred: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.asarray(y_pred, dtype=float) + b


def _evaluate_one(target: str, ckpt: str, eval_dir: str, train_csv: str) -> Dict[str, object]:
    os.makedirs(eval_dir, exist_ok=True)
    eval_ckpt_on_d4(cgm_project_output=DATA_ROOT, ckpt_path=ckpt, out_dir=eval_dir, target=target)
    d4_csv = os.path.join(eval_dir, f"New_D4_{target}_true_vs_pred.csv")
    d4 = pd.read_csv(d4_csv).dropna()
    train = _denorm_train_predictions(train_csv=train_csv, ckpt_path=ckpt, target=target)
    a, b = _fit_linear_calibration(train["true"].to_numpy(float), train["pred"].to_numpy(float))
    d4_cal = d4.copy()
    d4_cal["pred_cal"] = _apply_calibration(d4_cal["pred"].to_numpy(float), a, b)
    raw = _metrics(d4_cal["true"].to_numpy(float), d4_cal["pred"].to_numpy(float))
    cal = _metrics(d4_cal["true"].to_numpy(float), d4_cal["pred_cal"].to_numpy(float))
    train_m = _metrics(train["true"].to_numpy(float), train["pred"].to_numpy(float))
    return {
        "target": target,
        "train_metric": asdict(train_m),
        "d4_raw_metric": asdict(raw),
        "d4_cal_metric": asdict(cal),
        "calibration": {"a": a, "b": b},
    }


def _run_config(config_name: str, seed: int, env_p2_extra: Dict[str, str]) -> Dict[str, object]:
    run_dir = os.path.join(OUT_ROOT, config_name, f"seed_{seed}")
    ckpt = _train_two_phase(exp_dir=run_dir, seed=seed, env_p2_extra=env_p2_extra)
    p2_dir = os.path.join(run_dir, "phase2_finetune_head")
    train_csv = os.path.join(p2_dir, "latent_and_gold_all.csv")
    eval_dir = os.path.join(run_dir, "eval_D4")
    sspg_out = _evaluate_one("sspg", ckpt, eval_dir, train_csv)
    di_out = _evaluate_one("di", ckpt, eval_dir, train_csv)
    out = {
        "config": config_name,
        "seed": seed,
        "ckpt": ckpt,
        "sspg": sspg_out,
        "di": di_out,
    }
    with open(os.path.join(run_dir, "multiseed_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def _flatten(results: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for r in results:
        for t in ("sspg", "di"):
            d = r[t]
            rows.append(
                {
                    "config": r["config"],
                    "seed": r["seed"],
                    "target": t,
                    "train_r": d["train_metric"]["pearson_r"],
                    "train_rho": d["train_metric"]["spearman_r"],
                    "train_r2": d["train_metric"]["r2"],
                    "d4_raw_r": d["d4_raw_metric"]["pearson_r"],
                    "d4_raw_rho": d["d4_raw_metric"]["spearman_r"],
                    "d4_raw_r2": d["d4_raw_metric"]["r2"],
                    "d4_raw_rmse": d["d4_raw_metric"]["rmse"],
                    "d4_cal_r": d["d4_cal_metric"]["pearson_r"],
                    "d4_cal_rho": d["d4_cal_metric"]["spearman_r"],
                    "d4_cal_r2": d["d4_cal_metric"]["r2"],
                    "d4_cal_rmse": d["d4_cal_metric"]["rmse"],
                }
            )
    return pd.DataFrame(rows)


def run() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    results: List[Dict[str, object]] = []
    for seed in SEEDS:
        results.append(_run_config("Exp2_GV_Baseline", seed, {}))
        results.append(_run_config("Exp5_Finetune_16D", seed, {"P1_FINETUNE_16D_ONLY": "1"}))

    df = _flatten(results)
    raw_csv = os.path.join(OUT_ROOT, "multiseed_raw_results.csv")
    df.to_csv(raw_csv, index=False)
    agg = (
        df.groupby(["config", "target"], as_index=False)
        .agg(
            train_r_mean=("train_r", "mean"),
            train_r_std=("train_r", "std"),
            d4_raw_r_mean=("d4_raw_r", "mean"),
            d4_raw_r_std=("d4_raw_r", "std"),
            d4_raw_r2_mean=("d4_raw_r2", "mean"),
            d4_raw_r2_std=("d4_raw_r2", "std"),
            d4_cal_r_mean=("d4_cal_r", "mean"),
            d4_cal_r_std=("d4_cal_r", "std"),
            d4_cal_r2_mean=("d4_cal_r2", "mean"),
            d4_cal_r2_std=("d4_cal_r2", "std"),
            d4_raw_rmse_mean=("d4_raw_rmse", "mean"),
            d4_cal_rmse_mean=("d4_cal_rmse", "mean"),
        )
        .sort_values(["target", "config"])
    )
    agg_csv = os.path.join(OUT_ROOT, "multiseed_summary.csv")
    agg.to_csv(agg_csv, index=False)
    with open(os.path.join(OUT_ROOT, "multiseed_summary.md"), "w", encoding="utf-8") as f:
        f.write("# v17 Multi-seed Stability + Calibration\n\n")
        f.write(agg.to_markdown(index=False))
        f.write("\n")
    print("Saved:", raw_csv)
    print("Saved:", agg_csv)


if __name__ == "__main__":
    run()

