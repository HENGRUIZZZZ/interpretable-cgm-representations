from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple

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
from scripts.New_eval_trainD1D2_testD4 import train_on_d1d2


ZIP_PATH = "/Users/hertz1030/Downloads/v12_final_package.zip"
DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v12")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) < 3:
        return {"n": int(len(y_true)), "pearson_r": float("nan"), "spearman_r": float("nan"), "r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    pr, pp = stats.pearsonr(y_true, y_pred)
    sr, sp = stats.spearmanr(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _head_26d() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(26, 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(32, 1),
    )


def _extract_and_apply_data_fixes(data_root: str, zip_path: str, backup_dir: str) -> None:
    os.makedirs(backup_dir, exist_ok=True)
    targets = [
        ("D4_hall/meals.csv", "D4_meals_fixed.csv"),
        ("D2_stanford/meals.csv", "D2_meals_fixed.csv"),
        ("D3_cgmacros/subjects.csv", "D3_subjects_fixed.csv"),
        ("D3_cgmacros/meals.csv", "D3_meals_fixed.csv"),
    ]
    for rel, _ in targets:
        src = os.path.join(data_root, rel)
        if os.path.isfile(src):
            dst = os.path.join(backup_dir, rel.replace("/", "__"))
            shutil.copy2(src, dst)

    tmp_dir = os.path.join(backup_dir, "_unzipped")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)
    fix_root = os.path.join(tmp_dir, "v12_final", "data_fixes")
    for rel, fix_name in targets:
        src = os.path.join(fix_root, fix_name)
        dst = os.path.join(data_root, rel)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Missing fixed file in package: {src}")
        shutil.copy2(src, dst)


def _d4_subject_map(data_root: str) -> Dict[str, str]:
    s = pd.read_csv(os.path.join(data_root, "D4_hall", "subjects.csv"))
    return {f"D4_{str(r['original_id']).strip()}": str(r["subject_id"]).strip() for _, r in s.dropna(subset=["subject_id", "original_id"]).iterrows()}


def _fit_calibration_from_train(train_lat_csv: str, target: str) -> Tuple[float, float]:
    df = pd.read_csv(train_lat_csv)
    pred_candidates = [f"{target}_hat_head", f"{target}_hat", f"{target}_pred"]
    pred_col = None
    for c in pred_candidates:
        if c in df.columns:
            pred_col = c
            break
    if pred_col is None:
        raise KeyError(f"No prediction column found for {target} in {train_lat_csv}; tried {pred_candidates}")
    sub = df[[target, pred_col]].dropna().copy()
    x = sub[pred_col].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def _eval_d4_with_separate_heads(data_root: str, ckpt_path: str, out_dir: str, calibrate_from: str) -> Dict[str, Dict[str, float]]:
    os.makedirs(out_dir, exist_ok=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MechanisticAutoencoder(
        meal_size=len(MEAL_COVARIATES),
        demographics_size=len(DEMOGRAPHICS_COVARIATES),
        embedding_size=8,
        hidden_size=32,
        num_layers=2,
        encoder_dropout_prob=0.0,
        decoder_dropout_prob=0.5,
    )
    model.load_state_dict(ck["model_state"], strict=False)
    model.eval()
    sspg_head = _head_26d()
    di_head = _head_26d()
    sspg_head.load_state_dict(ck["sspg_head_state"], strict=True)
    di_head.load_state_dict(ck["di_head_state"], strict=True)
    sspg_head.eval()
    di_head.eval()

    b, info, lab = load_cgm_project_level3(dataset_id="D4", output_base=data_root)
    x0 = (b.cgm - ck["train_mean"][0]) / (ck["train_std"][0] + 1e-8)
    x1 = (b.timestamps - ck["train_mean"][1]) / (ck["train_std"][1] + 1e-8)
    x2 = (b.meals - ck["train_mean"][2]) / (ck["train_std"][2] + 1e-8)
    x3 = (b.demographics - ck["train_mean"][3]) / (ck["train_std"][3] + 1e-8)
    t = [torch.tensor(x, dtype=torch.float32) for x in (x0, x1, x2, x3)]
    with torch.no_grad():
        p, z0, z16 = model.get_all_latents(*t)
        lat26 = torch.cat([p, z0, z16], dim=-1)
        sspg_pred = sspg_head(lat26).squeeze(-1).numpy()
        di_pred = di_head(lat26).squeeze(-1).numpy()
    if bool(ck.get("P1_ZSCORE_TARGETS", False)):
        sspg_pred = sspg_pred * float(ck["sspg_std"]) + float(ck["sspg_mean"])
        di_pred = di_pred * float(ck["di_std"]) + float(ck["di_mean"])

    # sample-level predictions for meal aggregation/ICC
    sid_map = _d4_subject_map(data_root)
    per = pd.DataFrame(
        {
            "subject_id": [sid_map.get(str(x), str(x)) for x in info.patient_ids],
            "sspg_pred_raw": sspg_pred,
            "di_pred_raw": di_pred,
        }
    )
    per.to_csv(os.path.join(out_dir, "per_subject_per_meal_preds_raw.csv"), index=False)

    # calibration from train predictions
    a_s, b_s = _fit_calibration_from_train(calibrate_from, "sspg")
    a_d, b_d = _fit_calibration_from_train(calibrate_from, "di")
    per["sspg_pred"] = a_s * per["sspg_pred_raw"] + b_s
    per["di_pred"] = a_d * per["di_pred_raw"] + b_d
    per.to_csv(os.path.join(out_dir, "per_subject_per_meal_preds.csv"), index=False)

    g = lab.copy()
    if "SSPG" in g.columns and "sspg" not in g.columns:
        g["sspg"] = g["SSPG"]
    if "DI" in g.columns and "di" not in g.columns:
        g["di"] = g["DI"]
    gold = g[["subject_id", "sspg", "di"]].dropna().drop_duplicates("subject_id")
    sub = per.groupby("subject_id")[["sspg_pred", "di_pred"]].mean().reset_index().merge(gold, on="subject_id", how="inner")

    d_sspg = sub.rename(columns={"sspg": "true", "sspg_pred": "pred"})[["subject_id", "true", "pred"]]
    d_di = sub.rename(columns={"di": "true", "di_pred": "pred"})[["subject_id", "true", "pred"]]
    d_sspg.to_csv(os.path.join(out_dir, "D4_sspg_true_vs_pred.csv"), index=False)
    d_di.to_csv(os.path.join(out_dir, "D4_di_true_vs_pred.csv"), index=False)
    m_sspg = _metrics(d_sspg["true"].to_numpy(), d_sspg["pred"].to_numpy())
    m_di = _metrics(d_di["true"].to_numpy(), d_di["pred"].to_numpy())
    with open(os.path.join(out_dir, "D4_sspg_metrics.json"), "w") as f:
        json.dump(m_sspg, f, indent=2)
    with open(os.path.join(out_dir, "D4_di_metrics.json"), "w") as f:
        json.dump(m_di, f, indent=2)
    return {"sspg": m_sspg, "di": m_di}


def _run_shap_for_sspg(train_latent_csv: str, ckpt_path: str, out_png: str, out_csv: str) -> float:
    import shap
    import matplotlib.pyplot as plt

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = _head_26d()
    head.load_state_dict(ck["sspg_head_state"], strict=True)
    head.eval()
    feat_cols = ["tau_m", "Gb", "sg", "si", "p2", "mi"] + [f"z_init_{i}" for i in range(4)] + [f"z_nonseq_{i}" for i in range(16)]
    df = pd.read_csv(train_latent_csv)
    X = df[feat_cols].dropna().to_numpy(dtype=float)
    if len(X) > 120:
        X = X[:120]
    bg = X[: min(40, len(X))]

    def f_np(x_np):
        with torch.no_grad():
            y = head(torch.tensor(x_np, dtype=torch.float32)).squeeze(-1).numpy()
        return y

    explainer = shap.KernelExplainer(f_np, bg)
    n_eval = min(60, len(X))
    sv = explainer.shap_values(X[:n_eval], nsamples=100)
    shap.summary_plot(sv, X[:n_eval], feature_names=feat_cols, show=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    imp = np.mean(np.abs(np.asarray(sv)), axis=0)
    out_df = pd.DataFrame({"feature": feat_cols, "importance": imp}).sort_values("importance", ascending=False)
    out_df.to_csv(out_csv, index=False)
    ode_set = {"tau_m", "Gb", "sg", "si", "p2", "mi"}
    ode_share = float(out_df[out_df["feature"].isin(ode_set)]["importance"].sum() / max(out_df["importance"].sum(), 1e-12))
    return ode_share


def _physiology_validation(best_exp_dir: str) -> None:
    ckpt = os.path.join(best_exp_dir, "train", "autoencoder_p1_full.pt")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = MechanisticAutoencoder(
        meal_size=len(MEAL_COVARIATES),
        demographics_size=len(DEMOGRAPHICS_COVARIATES),
        embedding_size=8,
        hidden_size=32,
        num_layers=2,
        encoder_dropout_prob=0.0,
        decoder_dropout_prob=0.5,
    )
    model.load_state_dict(ck["model_state"], strict=False)
    model.eval()

    b, info, _ = load_cgm_project_level3(dataset_id="D3", output_base=DATA_ROOT)
    x0 = (b.cgm - ck["train_mean"][0]) / (ck["train_std"][0] + 1e-8)
    x1 = (b.timestamps - ck["train_mean"][1]) / (ck["train_std"][1] + 1e-8)
    x2 = (b.meals - ck["train_mean"][2]) / (ck["train_std"][2] + 1e-8)
    x3 = (b.demographics - ck["train_mean"][3]) / (ck["train_std"][3] + 1e-8)
    with torch.no_grad():
        p26, _, _ = model.get_all_latents(
            torch.tensor(x0, dtype=torch.float32),
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(x3, dtype=torch.float32),
        )
    p = p26.numpy()
    cov_ix = {c: i for i, c in enumerate(MEAL_COVARIATES)}
    meal_cov = b.meals[:, 0, :]
    phys = pd.DataFrame(
        {
            "subject_id": info.patient_ids.astype(str),
            "tau_m": p[:, 0],
            "si": p[:, 3],
            "fat_g": meal_cov[:, cov_ix.get("total_fat", 4)],
            "protein_g": meal_cov[:, cov_ix.get("total_protein", 5)],
        }
    )
    phys.to_csv(os.path.join(best_exp_dir, "d3_latent_with_macros.csv"), index=False)
    rows = []
    for c in ["fat_g", "protein_g"]:
        ok = np.isfinite(phys["tau_m"]) & np.isfinite(phys[c])
        if ok.sum() >= 3:
            r, p = stats.spearmanr(phys.loc[ok, "tau_m"], phys.loc[ok, c])
            rows.append({"feature": c, "spearman_r_tau_m": float(r), "p": float(p), "n": int(ok.sum())})
    pd.DataFrame(rows).to_csv(os.path.join(best_exp_dir, "tau_m_macro_correlations.csv"), index=False)

    meals = pd.read_csv(os.path.join(DATA_ROOT, "D3_cgmacros", "meals.csv"))
    meals["timestamp"] = pd.to_datetime(meals["timestamp"], errors="coerce")
    # Approx join by order after loader filters; sufficient for aggregate circadian signal
    n = min(len(meals), len(phys))
    circ = phys.iloc[:n].copy()
    circ["hour"] = meals.iloc[:n]["timestamp"].dt.hour.values
    circ["time_bin"] = pd.cut(circ["hour"], bins=[-1, 4, 11, 17, 23], labels=["night", "morning", "afternoon", "evening"])
    summ = circ.groupby("time_bin", observed=True)["si"].agg(["count", "mean", "median"]).reset_index()
    summ.to_csv(os.path.join(best_exp_dir, "si_circadian_summary.csv"), index=False)


def _run_experiment(name: str, data_root: str, lambda_sspg: float, lambda_di: float, extra_env: Dict[str, str], do_eval: bool = True) -> Dict[str, object]:
    exp_dir = os.path.join(OUT_ROOT, name)
    train_dir = os.path.join(exp_dir, "train")
    os.makedirs(exp_dir, exist_ok=True)
    ckpt = os.path.join(train_dir, "autoencoder_p1_full.pt")
    if not os.path.isfile(ckpt):
        train_on_d1d2(
            cgm_project_output=data_root,
            results_dir=train_dir,
            seed=21,
            lambda_sspg=lambda_sspg,
            lambda_di=lambda_di,
            num_epochs=100,
            extra_env=extra_env,
        )
    if not os.path.isfile(ckpt):
        raise RuntimeError(f"Checkpoint not found after training: {ckpt}")
    shutil.copy2(os.path.join(train_dir, "latent_and_gold_all_26d.csv"), os.path.join(exp_dir, "latent_and_gold_all_26d.csv"))
    out = {"name": name, "exp_dir": exp_dir}
    if do_eval:
        met = _eval_d4_with_separate_heads(data_root, ckpt, exp_dir, os.path.join(train_dir, "latent_and_gold_all.csv"))
        ode_share = _run_shap_for_sspg(
            train_latent_csv=os.path.join(train_dir, "latent_and_gold_all_26d.csv"),
            ckpt_path=ckpt,
            out_png=os.path.join(exp_dir, "shap_summary_sspg.png"),
            out_csv=os.path.join(exp_dir, "shap_feature_importance.csv"),
        )
        out.update({"metrics": met, "ode_shap_share": ode_share})
    return out


def run_v12() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: List[Dict[str, object]] = []

    # Exp3 first (old data, before applying fixes)
    exp3 = _run_experiment(
        name="Exp3_DataAblation_OldData_26D_Semi",
        data_root=DATA_ROOT,
        lambda_sspg=0.01,
        lambda_di=0.01,
        extra_env={
            "P1_V5_PREDICTION_HEAD": "0",
            "P1_HEAD_USE_26D": "0",
            "P1_SEPARATE_HEAD_26D": "1",
            "P1_DECOUPLE_SSPG": "0",
            "P1_DI_PRODUCT_CONSTRAINT": "0",
            "P1_DI_LOG_PRODUCT": "0",
            "P1_DI_MLP_HEAD": "0",
            "P1_ZSCORE_TARGETS": "1",
            "P1_WIDE_PARAM_RANGE": "1",
            "P1_V10_WIDE_BOUNDS": "1",
            "LAMBDA_IR": "0.0",
            "LAMBDA_CLS": "0.0",
            "LAMBDA_ORTHO": "0.0",
            "P1_DETACH_HEAD_INPUT": "1",
            "P1_HEAD_GRAD_SCALE": "0.01",
            "P1_IDENT_LOSS_LAMBDA": "0.1",
            "P1_V8_ODE_CORR": "1",
        },
        do_eval=True,
    )

    # apply fixed meal/subject files
    _extract_and_apply_data_fixes(DATA_ROOT, ZIP_PATH, os.path.join(OUT_ROOT, "_data_backup_before_fix"))

    # Exp1 fixed baseline (unsupervised style; no supervised losses)
    exp1 = _run_experiment(
        name="Exp1_WangBaseline_10D_Unsupervised_FixedData",
        data_root=DATA_ROOT,
        lambda_sspg=0.0,
        lambda_di=0.0,
        extra_env={
            "P1_V5_PREDICTION_HEAD": "0",
            "P1_HEAD_USE_26D": "0",
            "P1_SEPARATE_HEAD_26D": "0",
            "P1_V8_HEAD_10D": "1",
            "P1_ZSCORE_TARGETS": "1",
            "P1_WIDE_PARAM_RANGE": "1",
            "P1_V10_WIDE_BOUNDS": "1",
            "LAMBDA_IR": "0.0",
            "LAMBDA_CLS": "0.0",
            "LAMBDA_ORTHO": "0.0",
            "P1_IDENT_LOSS_LAMBDA": "0.1",
        },
        do_eval=False,
    )

    # Exp2 fixed main model
    exp2 = _run_experiment(
        name="Exp2_GlucoVectorV12_26D_Semi_FixedData",
        data_root=DATA_ROOT,
        lambda_sspg=0.01,
        lambda_di=0.01,
        extra_env={
            "P1_V5_PREDICTION_HEAD": "0",
            "P1_HEAD_USE_26D": "0",
            "P1_SEPARATE_HEAD_26D": "1",
            "P1_DECOUPLE_SSPG": "0",
            "P1_DI_PRODUCT_CONSTRAINT": "0",
            "P1_DI_LOG_PRODUCT": "0",
            "P1_DI_MLP_HEAD": "0",
            "P1_ZSCORE_TARGETS": "1",
            "P1_WIDE_PARAM_RANGE": "1",
            "P1_V10_WIDE_BOUNDS": "1",
            "LAMBDA_IR": "0.0",
            "LAMBDA_CLS": "0.0",
            "LAMBDA_ORTHO": "0.0",
            "P1_DETACH_HEAD_INPUT": "1",
            "P1_HEAD_GRAD_SCALE": "0.01",
            "P1_IDENT_LOSS_LAMBDA": "0.1",
            "P1_V8_ODE_CORR": "1",
        },
        do_eval=True,
    )
    _physiology_validation(exp2["exp_dir"])

    for x in [exp2, exp3]:
        m = x["metrics"]
        rows.append(
            {
                "experiment": x["name"],
                "sspg_pearson_r": m["sspg"]["pearson_r"],
                "di_pearson_r": m["di"]["pearson_r"],
                "sspg_rmse": m["sspg"]["rmse"],
                "di_rmse": m["di"]["rmse"],
                "ode_shap_share": x["ode_shap_share"],
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(OUT_ROOT, "v12_experiment_summary.csv"), index=False)

    rep = [
        "# v12_comprehensive_report",
        "",
        f"Run stamp: {stamp}",
        "",
        "## Main comparison (Exp2 fixed data vs Exp3 old data ablation)",
        pd.DataFrame(rows).to_markdown(index=False),
        "",
        "## Notes",
        "- Exp1 is trained as unsupervised 10D baseline on fixed data (no SSPG/DI supervision).",
        "- Exp2 uses separated 26D heads with weak gradient scale 0.01.",
        "- Exp3 uses same model config as Exp2 but pre-fix old data.",
    ]
    with open(os.path.join(OUT_ROOT, "v12_comprehensive_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))

    zip_path = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v12.zip")
    subprocess.run(["/bin/zsh", "-lc", f'cd "{REPO_ROOT}" && zip -r "{zip_path}" "New_paper1_results_glucovector_v12" >/dev/null'], check=False)


if __name__ == "__main__":
    run_v12()

