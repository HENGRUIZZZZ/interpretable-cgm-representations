from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from load_cgm_project_data import DEMOGRAPHICS_COVARIATES, MEAL_COVARIATES, load_cgm_project_level3
from models import MechanisticAutoencoder
from scripts.New_eval_trainD1D2_testD4 import eval_ckpt_on_d4, train_on_d1d2


DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v11")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) < 3:
        return {"n": int(len(y_true)), "pearson_r": float("nan"), "spearman_r": float("nan"), "r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    pr, _ = stats.pearsonr(y_true, y_pred)
    sr, _ = stats.spearmanr(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "pearson_r": float(pr),
        "spearman_r": float(sr),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _fit_linear_calibration(train_lat_csv: str, target: str) -> Tuple[float, float, float]:
    df = pd.read_csv(train_lat_csv)
    sub = df[[target, f"{target}_hat"]].dropna().copy()
    x = sub[f"{target}_hat"].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    sign = 1.0
    if target == "di" and len(x) >= 3:
        r = np.corrcoef(x, y)[0, 1]
        if np.isfinite(r) and r < 0:
            sign = -1.0
            x = -x
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b), float(sign)


def _apply_calibration(scatter_csv: str, a: float, b: float, out_csv: str, out_metrics_json: str, pred_sign: float = 1.0) -> Dict[str, float]:
    df = pd.read_csv(scatter_csv)
    if "true" not in df.columns:
        t = [c for c in df.columns if c.endswith("_true")][0]
        p = [c for c in df.columns if c.endswith("_pred")][0]
        df = df.rename(columns={t: "true", p: "pred"})
    y = df["true"].to_numpy(dtype=float)
    yhat = a * (pred_sign * df["pred"].to_numpy(dtype=float)) + b
    met = _metrics(y, yhat)
    pd.DataFrame({"subject_id": df["subject_id"], "true": y, "pred": yhat}).to_csv(out_csv, index=False)
    with open(out_metrics_json, "w") as f:
        json.dump(met, f, indent=2)
    return met


def _infer_e2e_input_dim(state: Dict[str, torch.Tensor]) -> int:
    w = state.get("0.weight", None)
    return int(w.shape[1]) if w is not None else 26


def _build_e2e_head(input_dim: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2),
    )


def _d4_maps() -> Dict[str, str]:
    s = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "subjects.csv"))
    return {f"D4_{str(r['original_id']).strip()}": str(r["subject_id"]).strip() for _, r in s.dropna(subset=["subject_id", "original_id"]).iterrows()}


def _extract_d4_standard_meal_windows() -> pd.DataFrame:
    cgm = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "cgm.csv"))
    cgm = cgm.rename(columns={"glucose_mg_dl": "glucose"})
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")
    meals = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "meals.csv"))
    meals["timestamp"] = pd.to_datetime(meals["timestamp"], errors="coerce")
    meals = meals[meals["meal_type"].isin(["Cornflakes", "PB_sandwich", "Protein_bar"])].copy()
    grid = np.arange(-30, 181, 5, dtype=float)

    rows: List[Dict[str, object]] = []
    for _, m in meals.iterrows():
        sid = str(m["subject_id"])
        meal_type = str(m["meal_type"])
        rep = int(m["repeat"]) if pd.notna(m.get("repeat", np.nan)) else -1
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
        y_new = np.interp(grid, t[order], y[order])
        rows.append({"subject_id": sid, "meal_type": meal_type, "repeat": rep, "cgm_curve": y_new})
    return pd.DataFrame(rows)


def _run_model_on_d4_meals(ckpt_path: str, cal_sspg: Tuple[float, float], cal_di: Tuple[float, float], di_pred_sign: float = 1.0) -> pd.DataFrame:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    e2e_dim = _infer_e2e_input_dim(ck["e2e_head_state"])
    model = MechanisticAutoencoder(
        meal_size=len(MEAL_COVARIATES),
        demographics_size=len(DEMOGRAPHICS_COVARIATES),
        embedding_size=8,
        hidden_size=32,
        num_layers=2,
        encoder_dropout_prob=0.0,
        decoder_dropout_prob=0.5,
    )
    ms = ck.get("model_state", {})
    if hasattr(model, "use_v8_recon_corr"):
        model.use_v8_recon_corr = bool(ck.get("P1_V8_RECON_CORR", False))
    if hasattr(model, "use_v8_ode_corr"):
        model.use_v8_ode_corr = bool(ck.get("P1_V8_ODE_CORR", False))
    model.load_state_dict(ms, strict=False)
    model.eval()

    e2e = _build_e2e_head(e2e_dim)
    e2e.load_state_dict(ck["e2e_head_state"], strict=True)
    e2e.eval()

    d4w = _extract_d4_standard_meal_windows()
    n = len(d4w)
    if n == 0:
        raise RuntimeError("No D4 standard-meal windows found.")
    T = 43
    cgm = np.stack(d4w["cgm_curve"].to_list()).astype(np.float32)[:, :, None]
    ts = np.tile(np.arange(-30, 181, 5, dtype=np.float32)[None, :, None], (n, 1, 1))
    meals = np.zeros((n, T, len(MEAL_COVARIATES)), dtype=np.float32)
    demo = np.zeros((n, len(DEMOGRAPHICS_COVARIATES)), dtype=np.float32)
    tm, tsd = ck["train_mean"], ck["train_std"]
    cgm = (cgm - tm[0]) / (tsd[0] + 1e-8)
    ts = (ts - tm[1]) / (tsd[1] + 1e-8)
    meals = (meals - tm[2]) / (tsd[2] + 1e-8)
    demo = (demo - tm[3]) / (tsd[3] + 1e-8)

    with torch.no_grad():
        p26, init26, z16 = model.get_all_latents(
            torch.tensor(cgm, dtype=torch.float32),
            torch.tensor(ts, dtype=torch.float32),
            torch.tensor(meals, dtype=torch.float32),
            torch.tensor(demo, dtype=torch.float32),
        )
        head_in = torch.cat([p26, init26], dim=-1) if e2e_dim == 10 else torch.cat([p26, init26, z16], dim=-1)
        pred2 = e2e(head_in).numpy()

    sspg_hat = pred2[:, 0]
    di_hat = pred2[:, 1]
    if bool(ck.get("P1_ZSCORE_TARGETS", False)):
        sspg_hat = sspg_hat * float(ck["sspg_std"]) + float(ck["sspg_mean"])
        di_hat = di_hat * float(ck["di_std"]) + float(ck["di_mean"])

    a_s, b_s = cal_sspg
    a_d, b_d = cal_di
    d4w = d4w[["subject_id", "meal_type", "repeat"]].copy()
    d4w["subject_id"] = d4w["subject_id"].map(lambda x: _d4_maps().get(x, x))
    d4w["sspg_pred_raw"] = sspg_hat
    d4w["di_pred_raw"] = di_pred_sign * di_hat
    d4w["sspg_pred"] = a_s * sspg_hat + b_s
    d4w["di_pred"] = a_d * (di_pred_sign * di_hat) + b_d
    return d4w


def _icc_oneway(df: pd.DataFrame, subject_col: str, value_col: str) -> Dict[str, float]:
    d = df[[subject_col, value_col]].dropna().copy()
    counts = d.groupby(subject_col)[value_col].count()
    keep_sub = counts[counts >= 2].index
    d = d[d[subject_col].isin(keep_sub)]
    n = d[subject_col].nunique()
    N = len(d)
    if n < 2 or N <= n:
        return {"n_subjects": int(n), "n_rows": int(N), "icc1": float("nan")}
    gm = float(d[value_col].mean())
    g = d.groupby(subject_col)[value_col]
    ni = g.size()
    mi = g.mean()
    ss_between = float((ni * (mi - gm) ** 2).sum())
    ss_within = float(sum(((x - x.mean()) ** 2).sum() for _, x in g))
    df_between = n - 1
    df_within = N - n
    ms_between = ss_between / max(df_between, 1)
    ms_within = ss_within / max(df_within, 1)
    k_eff = (N - (ni.pow(2).sum() / N)) / max(n - 1, 1)
    icc = (ms_between - ms_within) / (ms_between + (k_eff - 1.0) * ms_within + 1e-12)
    return {"n_subjects": int(n), "n_rows": int(N), "k_eff": float(k_eff), "icc1": float(icc)}


def _aggregate_subject_metrics(pred_meal_df: pd.DataFrame, out_dir: str) -> Dict[str, Dict[str, float]]:
    lab = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "labels.csv"))
    if "SSPG" in lab.columns and "sspg" not in lab.columns:
        lab["sspg"] = lab["SSPG"]
    if "DI" in lab.columns and "di" not in lab.columns:
        lab["di"] = lab["DI"]
    gold = lab[["subject_id", "sspg", "di"]].dropna().drop_duplicates("subject_id")

    agg = pred_meal_df.groupby("subject_id")[["sspg_pred", "di_pred"]].mean().reset_index()
    merged = agg.merge(gold, on="subject_id", how="inner")
    merged.to_csv(os.path.join(out_dir, "aggregated_true_vs_pred.csv"), index=False)

    sspg_met = _metrics(merged["sspg"].to_numpy(), merged["sspg_pred"].to_numpy())
    di_met = _metrics(merged["di"].to_numpy(), merged["di_pred"].to_numpy())
    icc_sspg = _icc_oneway(pred_meal_df, "subject_id", "sspg_pred")
    icc_di = _icc_oneway(pred_meal_df, "subject_id", "di_pred")
    out = {"sspg": sspg_met, "di": di_met, "icc_sspg": icc_sspg, "icc_di": icc_di}
    with open(os.path.join(out_dir, "aggregated_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def _di_bug_hunt(exp_dir: str) -> Dict[str, float]:
    di_raw = pd.read_csv(os.path.join(exp_dir, "D4_di_true_vs_pred_raw.csv")).rename(columns=lambda c: c.replace("di_", ""))
    sspg_raw = pd.read_csv(os.path.join(exp_dir, "D4_sspg_true_vs_pred_raw.csv")).rename(columns=lambda c: c.replace("sspg_", ""))
    d = di_raw[["subject_id", "true", "pred"]].dropna().copy()
    s = sspg_raw[["subject_id", "pred"]].rename(columns={"pred": "sspg_head_pred"}).dropna()
    m = d.merge(s, on="subject_id", how="left")
    r_di = stats.pearsonr(m["true"], m["pred"])[0] if len(m) >= 3 else np.nan
    r_di_flip = stats.pearsonr(m["true"], -m["pred"])[0] if len(m) >= 3 else np.nan
    r_wrong_head = stats.pearsonr(m["true"], m["sspg_head_pred"])[0] if m["sspg_head_pred"].notna().sum() >= 3 else np.nan
    out = {"n": int(len(m)), "pearson_di_correct_head": float(r_di), "pearson_di_if_sign_flipped": float(r_di_flip), "pearson_di_if_wrong_head0_used": float(r_wrong_head)}
    with open(os.path.join(exp_dir, "di_bug_hunt_report.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def _extract_d3_meta_like_loader(data_root: str, min_cgm_points: int = 10) -> pd.DataFrame:
    d3 = os.path.join(data_root, "D3_cgmacros")
    meals = pd.read_csv(os.path.join(d3, "meals.csv"))
    meals["timestamp"] = pd.to_datetime(meals["timestamp"], errors="coerce")
    cgm = pd.read_csv(os.path.join(d3, "cgm.csv"))
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")
    cgm = cgm.rename(columns={"glucose_mg_dl": "glucose_mgdl"})
    subj = pd.read_csv(os.path.join(d3, "subjects.csv"))
    valid_sub = set(subj["subject_id"].astype(str).tolist())
    rows = []
    for _, r in meals.iterrows():
        sid = str(r["subject_id"])
        if sid not in valid_sub or pd.isna(r["timestamp"]):
            continue
        t0 = r["timestamp"]
        g = cgm[(cgm["subject_id"].astype(str) == sid) & (cgm["timestamp"] >= t0 + pd.Timedelta(minutes=-30)) & (cgm["timestamp"] <= t0 + pd.Timedelta(minutes=180))]
        if len(g) < min_cgm_points:
            continue
        rows.append({"subject_id": sid, "meal_timestamp": t0})
    return pd.DataFrame(rows)


def _physiology_validation(best_ckpt: str, out_dir: str) -> None:
    ck = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model = MechanisticAutoencoder(
        meal_size=len(MEAL_COVARIATES),
        demographics_size=len(DEMOGRAPHICS_COVARIATES),
        embedding_size=8,
        hidden_size=32,
        num_layers=2,
        encoder_dropout_prob=0.0,
        decoder_dropout_prob=0.5,
    )
    if hasattr(model, "use_v8_recon_corr"):
        model.use_v8_recon_corr = bool(ck.get("P1_V8_RECON_CORR", False))
    if hasattr(model, "use_v8_ode_corr"):
        model.use_v8_ode_corr = bool(ck.get("P1_V8_ODE_CORR", False))
    model.load_state_dict(ck["model_state"], strict=False)
    model.eval()

    b, info, _ = load_cgm_project_level3(dataset_id="D3", output_base=DATA_ROOT)
    x0 = torch.tensor((b.cgm - ck["train_mean"][0]) / (ck["train_std"][0] + 1e-8), dtype=torch.float32)
    x1 = torch.tensor((b.timestamps - ck["train_mean"][1]) / (ck["train_std"][1] + 1e-8), dtype=torch.float32)
    x2 = torch.tensor((b.meals - ck["train_mean"][2]) / (ck["train_std"][2] + 1e-8), dtype=torch.float32)
    x3 = torch.tensor((b.demographics - ck["train_mean"][3]) / (ck["train_std"][3] + 1e-8), dtype=torch.float32)
    with torch.no_grad():
        p26, _, _ = model.get_all_latents(x0, x1, x2, x3)
    p = p26.numpy()

    cov_ix = {c: i for i, c in enumerate(MEAL_COVARIATES)}
    meal_cov = b.meals[:, 0, :]
    phys = pd.DataFrame({
        "subject_id": info.patient_ids.astype(str),
        "tau_m": p[:, 0],
        "si": p[:, 3],
        "total_carb": meal_cov[:, cov_ix.get("total_carb", 1)],
        "total_fat": meal_cov[:, cov_ix.get("total_fat", 4)],
        "total_fiber": meal_cov[:, cov_ix.get("total_fiber", 3)],
    })
    phys.to_csv(os.path.join(out_dir, "d3_latent_with_macros.csv"), index=False)
    macro_rows = []
    for c in ["total_carb", "total_fat", "total_fiber"]:
        ok = np.isfinite(phys["tau_m"]) & np.isfinite(phys[c])
        if ok.sum() >= 3:
            r, pval = stats.spearmanr(phys.loc[ok, "tau_m"], phys.loc[ok, c])
            macro_rows.append({"macro": c, "spearman_r_tau_m": float(r), "p": float(pval), "n": int(ok.sum())})
    pd.DataFrame(macro_rows).to_csv(os.path.join(out_dir, "tau_m_macro_correlations.csv"), index=False)

    # Circadian validation: attach timestamps with loader-like filtered meals
    meta = _extract_d3_meta_like_loader(DATA_ROOT)
    n = min(len(meta), len(phys))
    circ = phys.iloc[:n].copy()
    circ["meal_timestamp"] = pd.to_datetime(meta.iloc[:n]["meal_timestamp"].values)
    circ["hour"] = circ["meal_timestamp"].dt.hour
    circ["time_bin"] = pd.cut(circ["hour"], bins=[-1, 4, 11, 17, 23], labels=["night", "morning", "afternoon", "evening"])
    circ.to_csv(os.path.join(out_dir, "si_circadian_samples.csv"), index=False)
    grp = circ.groupby("time_bin", observed=True)["si"].agg(["count", "mean", "median"]).reset_index()
    grp.to_csv(os.path.join(out_dir, "si_circadian_summary.csv"), index=False)
    mw_rows = []
    bins = ["morning", "afternoon", "evening", "night"]
    for i in range(len(bins)):
        for j in range(i + 1, len(bins)):
            a = circ.loc[circ["time_bin"] == bins[i], "si"].dropna()
            b = circ.loc[circ["time_bin"] == bins[j], "si"].dropna()
            if len(a) >= 10 and len(b) >= 10:
                stat, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
                mw_rows.append({"group_a": bins[i], "group_b": bins[j], "n_a": int(len(a)), "n_b": int(len(b)), "u_stat": float(stat), "p": float(pval)})
    pd.DataFrame(mw_rows).to_csv(os.path.join(out_dir, "si_circadian_pairwise_mwu.csv"), index=False)


def _run_one_experiment(name: str, env_cfg: Dict[str, str]) -> Dict[str, object]:
    exp_dir = os.path.join(OUT_ROOT, name)
    train_dir = os.path.join(exp_dir, "train")
    os.makedirs(exp_dir, exist_ok=True)
    ckpt = os.path.join(train_dir, "autoencoder_p1_full.pt")
    if not os.path.isfile(ckpt):
        train_on_d1d2(
            cgm_project_output=DATA_ROOT,
            results_dir=train_dir,
            seed=21,
            lambda_sspg=0.1,
            lambda_di=0.05,
            num_epochs=100,
            extra_env=env_cfg,
        )
    if not os.path.isfile(ckpt):
        raise RuntimeError(f"Checkpoint not found: {ckpt}")

    shutil.copy2(os.path.join(train_dir, "latent_and_gold_all_26d.csv"), os.path.join(exp_dir, "latent_and_gold_all_26d.csv"))
    eval_ckpt_on_d4(cgm_project_output=DATA_ROOT, ckpt_path=ckpt, out_dir=exp_dir, target="sspg", metrics_filename="D4_sspg_metrics_raw.json", scatter_filename="D4_sspg_true_vs_pred_raw.csv")
    eval_ckpt_on_d4(cgm_project_output=DATA_ROOT, ckpt_path=ckpt, out_dir=exp_dir, target="di", metrics_filename="D4_di_metrics_raw.json", scatter_filename="D4_di_true_vs_pred_raw.csv")

    a_s, b_s, sign_s = _fit_linear_calibration(os.path.join(train_dir, "latent_and_gold_all_26d.csv"), "sspg")
    a_d, b_d, sign_d = _fit_linear_calibration(os.path.join(train_dir, "latent_and_gold_all_26d.csv"), "di")
    m_sspg = _apply_calibration(
        os.path.join(exp_dir, "D4_sspg_true_vs_pred_raw.csv"),
        a_s,
        b_s,
        os.path.join(exp_dir, "D4_sspg_true_vs_pred.csv"),
        os.path.join(exp_dir, "D4_sspg_metrics.json"),
        pred_sign=sign_s,
    )
    m_di = _apply_calibration(
        os.path.join(exp_dir, "D4_di_true_vs_pred_raw.csv"),
        a_d,
        b_d,
        os.path.join(exp_dir, "D4_di_true_vs_pred.csv"),
        os.path.join(exp_dir, "D4_di_metrics.json"),
        pred_sign=sign_d,
    )

    pred_meal = _run_model_on_d4_meals(ckpt, (a_s, b_s), (a_d, b_d), di_pred_sign=sign_d)
    pred_meal.to_csv(os.path.join(exp_dir, "per_subject_per_meal_preds.csv"), index=False)
    agg = _aggregate_subject_metrics(pred_meal, exp_dir)
    bug = _di_bug_hunt(exp_dir)
    bug["train_di_pred_sign_applied"] = float(sign_d)
    with open(os.path.join(exp_dir, "di_bug_hunt_report.json"), "w") as f:
        json.dump(bug, f, indent=2)

    subprocess.run(
        [
            "python",
            os.path.join(REPO_ROOT, "scripts", "New_run_shap_analysis_v7.py"),
            "--ckpt",
            ckpt,
            "--train_latent_csv",
            os.path.join(train_dir, "latent_and_gold_all_26d.csv"),
            "--target",
            "sspg",
            "--out_png",
            os.path.join(exp_dir, "shap_summary_sspg.png"),
            "--out_csv",
            os.path.join(exp_dir, "shap_feature_importance.csv"),
        ],
        check=False,
    )
    return {"name": name, "sspg": m_sspg, "di": m_di, "agg": agg, "ckpt": ckpt, "di_sign": sign_d}


def run_v11() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiments = [
        (
            "Exp1_26D_Detach_WideIdent",
            {
                "P1_V8_ODE_CORR": "1",
                "P1_V8_HEAD_10D": "0",
                "P1_DETACH_HEAD_INPUT": "1",
                "P1_HEAD_GRAD_SCALE": "0.0",
                "P1_V10_WIDE_BOUNDS": "1",
                "P1_IDENT_LOSS_LAMBDA": "0.1",
                "P1_USE_TRI_CLASS": "1",
                "LAMBDA_CLS": "0.1",
            },
        ),
        (
            "Exp2_26D_SemiSupervised_WideIdent",
            {
                "P1_V8_ODE_CORR": "1",
                "P1_V8_HEAD_10D": "0",
                "P1_DETACH_HEAD_INPUT": "1",
                "P1_HEAD_GRAD_SCALE": "0.01",
                "P1_V10_WIDE_BOUNDS": "1",
                "P1_IDENT_LOSS_LAMBDA": "0.1",
                "P1_USE_TRI_CLASS": "1",
                "LAMBDA_CLS": "0.1",
            },
        ),
    ]
    rows = []
    results = []
    for name, cfg in experiments:
        r = _run_one_experiment(name, cfg)
        results.append(r)
        rows.append(
            {
                "experiment": name,
                "sspg_pearson_r": r["sspg"]["pearson_r"],
                "di_pearson_r": r["di"]["pearson_r"],
                "sspg_agg_pearson_r": r["agg"]["sspg"]["pearson_r"],
                "di_agg_pearson_r": r["agg"]["di"]["pearson_r"],
                "icc_sspg": r["agg"]["icc_sspg"]["icc1"],
                "icc_di": r["agg"]["icc_di"]["icc1"],
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_ROOT, "v11_experiment_summary.csv"), index=False)
    best_ix = summary["sspg_agg_pearson_r"].astype(float).idxmax()
    best_name = str(summary.iloc[int(best_ix)]["experiment"])
    best_ckpt = next(x["ckpt"] for x in results if x["name"] == best_name)
    _physiology_validation(best_ckpt, os.path.join(OUT_ROOT, best_name))

    rep = [
        "# v11_comprehensive_report",
        "",
        f"Run stamp: {stamp}",
        "",
        "## D4 Calibrated + Aggregated summary",
        summary.to_markdown(index=False),
        "",
        f"Best model by SSPG aggregated Pearson: `{best_name}`",
        f"- Check physiology outputs in `{best_name}/tau_m_macro_correlations.csv` and `{best_name}/si_circadian_summary.csv`.",
    ]
    with open(os.path.join(OUT_ROOT, "v11_comprehensive_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))

    zip_path = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v11.zip")
    subprocess.run(["/bin/zsh", "-lc", f'cd "{REPO_ROOT}" && zip -r "{zip_path}" "New_paper1_results_glucovector_v11" >/dev/null'], check=False)


if __name__ == "__main__":
    run_v11()

