from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from load_cgm_project_data import DEMOGRAPHICS_COVARIATES, MEAL_COVARIATES, load_cgm_project_level1_level2
from models import MechanisticAutoencoder
from scripts.New_eval_trainD1D2_testD4 import eval_ckpt_on_d4, train_on_d1d2

DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v16")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < 3:
        return {"n": int(len(y_true)), "pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_r": float("nan"), "spearman_p": float("nan"), "r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}
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
        "true_mean": float(np.mean(y_true)),
        "true_std": float(np.std(y_true, ddof=0)),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred, ddof=0)),
    }


def _base_env(train_datasets: str) -> Dict[str, str]:
    return {
        "P1_TRAIN_DATASETS": train_datasets,
        "P1_LR": "1e-2",
        "P1_SEED": "42",
        "P1_USE_LR_SCHEDULER": "1",
        "P1_SAVE_26D_LATENT": "1",
        "P1_ZSCORE_TARGETS": "1",
        "P1_V8_ODE_CORR": "0",
        "P1_V8_RECON_CORR": "0",
        "P1_HEAD_USE_26D": "1",
        "P1_V8_HEAD_10D": "0",
        "P1_SEPARATE_HEAD_26D": "0",
        "P1_V10_WIDE_BOUNDS": "1",
        "P1_ONE_MEAL_PER_SUBJECT": "1",
        "LAMBDA_CLS": "0.0",
        "LAMBDA_IR": "0.05",
        "LAMBDA_ORTHO_16D": "0.0",
        "LAMBDA_VAR_MATCH": "0.0",
        "P1_FINETUNE_16D_ONLY": "0",
    }


def _train_two_phase(exp_dir: str, env_p1: Dict[str, str], env_p2_extra: Dict[str, str]) -> str:
    p1_dir = os.path.join(exp_dir, "phase1_unsupervised")
    p2_dir = os.path.join(exp_dir, "phase2_finetune_head")
    ck1 = os.path.join(p1_dir, "autoencoder_p1_full.pt")
    if not os.path.isfile(ck1):
        train_on_d1d2(
            cgm_project_output=DATA_ROOT,
            results_dir=p1_dir,
            seed=42,
            lambda_sspg=0.0,
            lambda_di=0.0,
            num_epochs=100,
            extra_env=env_p1,
        )
    env2 = dict(env_p1)
    env2.update(env_p2_extra)
    env2["P1_FINETUNE_HEAD_ONLY"] = "1"
    env2["P1_PRETRAINED_MODEL"] = ck1
    ck2 = os.path.join(p2_dir, "autoencoder_p1_full.pt")
    if not os.path.isfile(ck2):
        train_on_d1d2(
            cgm_project_output=DATA_ROOT,
            results_dir=p2_dir,
            seed=42,
            lambda_sspg=0.1,
            lambda_di=0.1,
            num_epochs=100,
            extra_env=env2,
        )
    return ck2


def _eval_d4(ckpt: str, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    ms, _ = eval_ckpt_on_d4(cgm_project_output=DATA_ROOT, ckpt_path=ckpt, out_dir=out_dir, target="sspg")
    md, _ = eval_ckpt_on_d4(cgm_project_output=DATA_ROOT, ckpt_path=ckpt, out_dir=out_dir, target="di")
    return {"sspg": ms.__dict__, "di": md.__dict__}


def _eval_heldout(ckpt: str, test_dataset: str, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
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
    ms = ck.get("model_state", {})
    if hasattr(model, "use_v8_recon_corr"):
        model.use_v8_recon_corr = bool(ck.get("P1_V8_RECON_CORR", any(k.startswith("correction_mlp.") for k in ms.keys())))
    if hasattr(model, "use_v8_ode_corr"):
        model.use_v8_ode_corr = bool(ck.get("P1_V8_ODE_CORR", any(k.startswith("ode_correction.") for k in ms.keys())))
    model.load_state_dict(ms, strict=False)
    model.eval()
    if "e2e_head_state" not in ck:
        return {}
    w = ck["e2e_head_state"].get("0.weight", None)
    in_dim = int(w.shape[1]) if w is not None else 26
    head = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 64), torch.nn.ReLU(), torch.nn.Dropout(0.0),
        torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
    )
    head.load_state_dict(ck["e2e_head_state"], strict=True)
    head.eval()

    b, info, lab = load_cgm_project_level1_level2(dataset_id=test_dataset, output_base=DATA_ROOT)
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
        h = torch.cat([p26, z0], dim=-1) if in_dim == 10 else torch.cat([p26, z0, z16], dim=-1)
        y2 = head(h).numpy()
    sspg_pred, di_pred = y2[:, 0], y2[:, 1]
    if bool(ck.get("P1_ZSCORE_TARGETS", False)):
        sspg_pred = sspg_pred * float(ck["sspg_std"]) + float(ck["sspg_mean"])
        di_pred = di_pred * float(ck["di_std"]) + float(ck["di_mean"])

    if "SSPG" in lab.columns and "sspg" not in lab.columns:
        lab["sspg"] = lab["SSPG"]
    if "DI" in lab.columns and "di" not in lab.columns:
        lab["di"] = lab["DI"]
    pred = pd.DataFrame({"subject_id": info.patient_ids, "sspg_pred": sspg_pred, "di_pred": di_pred})
    pred = pred.groupby("subject_id", as_index=False)[["sspg_pred", "di_pred"]].mean()
    gold = lab[["subject_id", "sspg", "di"]].drop_duplicates("subject_id")
    m = pred.merge(gold, on="subject_id", how="left")
    m.to_csv(os.path.join(out_dir, f"predictions_{test_dataset}.csv"), index=False)
    out = {"sspg": _metrics(m["sspg"], m["sspg_pred"]), "di": _metrics(m["di"], m["di_pred"])}
    with open(os.path.join(out_dir, f"metrics_{test_dataset}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def _param_diag(lat26_csv: str) -> Dict[str, float]:
    d = pd.read_csv(lat26_csv)
    out = {}
    for k in ("tau_m", "Gb", "si", "mi"):
        v = pd.to_numeric(d[k], errors="coerce").dropna()
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_std"] = float(v.std(ddof=0))
        out[f"{k}_cv"] = float(v.std(ddof=0) / (abs(v.mean()) + 1e-8))
        out[f"{k}_min"] = float(v.min())
        out[f"{k}_max"] = float(v.max())
    return out


def _corr_16d_10d(lat26_csv: str, out_csv: str) -> Dict[str, float]:
    d = pd.read_csv(lat26_csv)
    mech = ["tau_m", "Gb", "sg", "si", "p2", "mi"] + [f"z_init_{i}" for i in range(4)]
    ns = [f"z_nonseq_{i}" for i in range(16)]
    rows = []
    for a in ns:
        for b in mech:
            x = pd.to_numeric(d[a], errors="coerce")
            y = pd.to_numeric(d[b], errors="coerce")
            ok = x.notna() & y.notna()
            r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 5 else float("nan")
            rows.append({"nonseq_dim": a, "mech_param": b, "pearson_r": r})
    cdf = pd.DataFrame(rows)
    cdf.to_csv(out_csv, index=False)
    g = cdf.groupby("nonseq_dim")["pearson_r"].apply(lambda s: s.abs().max())
    return {"mean_max_abs_corr_16d_vs_10d": float(np.nanmean(g.to_numpy(dtype=float)))}


def _run_mealtype(ckpt: str, out_csv: str) -> pd.DataFrame:
    script = os.path.join(REPO_ROOT, "scripts", "New_run_exp3_d4_mealtype_v7.py")
    if not os.path.isfile(script):
        return pd.DataFrame()
    os.system(f'python "{script}" --data_root "{DATA_ROOT}" --ckpt "{ckpt}" --out_csv "{out_csv}" >/dev/null 2>&1')
    return pd.read_csv(out_csv) if os.path.isfile(out_csv) else pd.DataFrame()


def _run_shap(ckpt: str, lat26_csv: str, out_dir: str) -> Dict[str, Any]:
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}
    os.makedirs(out_dir, exist_ok=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ck.get("e2e_head_state")
    if state is None:
        return {}
    w = state.get("0.weight", None)
    in_dim = int(w.shape[1]) if w is not None else 26
    head = torch.nn.Sequential(torch.nn.Linear(in_dim, 64), torch.nn.ReLU(), torch.nn.Dropout(0.0), torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    head.load_state_dict(state, strict=True)
    head.eval()
    feat = ["tau_m", "Gb", "sg", "si", "p2", "mi"] + [f"z_init_{i}" for i in range(4)] + [f"z_nonseq_{i}" for i in range(16)]
    feat = feat[:10] if in_dim == 10 else feat
    d = pd.read_csv(lat26_csv)
    feat = [c for c in feat if c in d.columns]
    X = d[feat].dropna().to_numpy(dtype=float)
    if len(X) < 20:
        return {}
    bg = X[:min(40, len(X))]
    x_eval = X[:min(60, len(X))]

    def f0(x):
        with torch.no_grad():
            return head(torch.tensor(x, dtype=torch.float32))[:, 0].numpy()

    ex = shap.KernelExplainer(f0, bg)
    sv = ex.shap_values(x_eval, nsamples=100)
    shap.summary_plot(sv, x_eval, feature_names=feat, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary_sspg.png"), dpi=140, bbox_inches="tight")
    plt.close()
    imp = np.mean(np.abs(np.asarray(sv)), axis=0)
    imp_df = pd.DataFrame({"feature": feat, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False)
    imp_df["pct"] = imp_df["mean_abs_shap"] / max(imp_df["mean_abs_shap"].sum(), 1e-12) * 100.0
    imp_df.to_csv(os.path.join(out_dir, "shap_feature_importance.csv"), index=False)
    ode = {"tau_m", "Gb", "sg", "si", "p2", "mi"}
    init = {f"z_init_{i}" for i in range(4)}
    nonseq = {f"z_nonseq_{i}" for i in range(16)}
    total = max(float(imp_df["mean_abs_shap"].sum()), 1e-12)
    out = {
        "mechanism_10d_share_sspg": float((imp_df[imp_df["feature"].isin(ode | init)]["mean_abs_shap"].sum()) / total * 100.0),
        "nonseq_16d_share_sspg": float((imp_df[imp_df["feature"].isin(nonseq)]["mean_abs_shap"].sum()) / total * 100.0),
    }
    with open(os.path.join(out_dir, "shap_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def run_v16() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    results: List[Dict[str, Any]] = []
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiments = [
        ("v16_Exp3_LODO_D4", "D1,D2", "D4", {}, {}),
        ("v16_Exp1_LODO_D1", "D2", "D1", {}, {}),
        ("v16_Exp2_LODO_D2", "D1", "D2", {}, {}),
        ("v16_Exp4_Orthogonality", "D1,D2", "D4", {}, {"LAMBDA_ORTHO_16D": "0.1", "P1_FINETUNE_16D_ONLY": "1"}),
        ("v16_Exp5_DistMatch", "D1,D2", "D4", {}, {"LAMBDA_VAR_MATCH": "0.05"}),
    ]

    for name, train_ds, test_ds, env_p1_add, env_p2_add in experiments:
        exp_dir = os.path.join(OUT_ROOT, name)
        os.makedirs(exp_dir, exist_ok=True)
        env_p1 = _base_env(train_ds)
        env_p1.update(env_p1_add)
        ckpt = _train_two_phase(exp_dir, env_p1, env_p2_add)
        with open(os.path.join(exp_dir, "config_snapshot.json"), "w") as f:
            json.dump({"name": name, "train_datasets": train_ds, "test_dataset": test_ds, "phase2_extra_env": env_p2_add}, f, indent=2)

        eval_dir = os.path.join(exp_dir, f"eval_{test_ds}")
        if test_ds == "D4":
            mets = _eval_d4(ckpt, eval_dir)
            rec = {
                "experiment": name, "test_dataset": test_ds,
                "sspg_pearson_r": mets["sspg"]["pearson_r"], "sspg_spearman_r": mets["sspg"]["spearman_r"], "sspg_r2": mets["sspg"]["r2"], "sspg_rmse": mets["sspg"]["rmse"], "sspg_mae": mets["sspg"]["mae"], "sspg_pvalue": mets["sspg"]["pearson_p"],
                "di_pearson_r": mets["di"]["pearson_r"], "di_spearman_r": mets["di"]["spearman_r"], "di_r2": mets["di"]["r2"], "di_rmse": mets["di"]["rmse"], "di_mae": mets["di"]["mae"], "di_pvalue": mets["di"]["pearson_p"],
            }
            mt_df = _run_mealtype(ckpt, os.path.join(exp_dir, "d4_per_meal_type.csv"))
            if not mt_df.empty:
                rec["d4_per_meal_rows"] = int(len(mt_df))
            else:
                rec["d4_per_meal_rows"] = 0
        else:
            mets = _eval_heldout(ckpt, test_ds, eval_dir)
            rec = {
                "experiment": name, "test_dataset": test_ds,
                "sspg_pearson_r": mets.get("sspg", {}).get("pearson_r"), "sspg_spearman_r": mets.get("sspg", {}).get("spearman_r"), "sspg_r2": mets.get("sspg", {}).get("r2"), "sspg_rmse": mets.get("sspg", {}).get("rmse"), "sspg_mae": mets.get("sspg", {}).get("mae"), "sspg_pvalue": mets.get("sspg", {}).get("pearson_p"),
                "di_pearson_r": mets.get("di", {}).get("pearson_r"), "di_spearman_r": mets.get("di", {}).get("spearman_r"), "di_r2": mets.get("di", {}).get("r2"), "di_rmse": mets.get("di", {}).get("rmse"), "di_mae": mets.get("di", {}).get("mae"), "di_pvalue": mets.get("di", {}).get("pearson_p"),
            }

        lat26 = os.path.join(exp_dir, "phase2_finetune_head", "latent_and_gold_all_26d.csv")
        if os.path.isfile(lat26):
            pdg = _param_diag(lat26)
            rec.update(pdg)
            corr = _corr_16d_10d(lat26, os.path.join(exp_dir, "16d_vs_10d_correlation.csv"))
            rec.update(corr)
            with open(os.path.join(exp_dir, "param_diagnostics.json"), "w") as f:
                json.dump(pdg, f, indent=2)
            if "Exp3" in name or "Exp4" in name:
                shap_out = _run_shap(ckpt, lat26, os.path.join(exp_dir, "shap_analysis"))
                rec.update(shap_out)
        results.append(rec)

    summ = pd.DataFrame(results)
    summ.to_csv(os.path.join(OUT_ROOT, "v16_comprehensive_summary.csv"), index=False)
    with open(os.path.join(OUT_ROOT, "v16_comprehensive_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT_ROOT, "v16_report.md"), "w", encoding="utf-8") as f:
        f.write("# GlucoVector v16 Report\n\n")
        f.write(f"run_stamp: {run_stamp}\n\n")
        f.write(summ.to_markdown(index=False))
        f.write("\n")

    zip_name = f"{os.path.basename(OUT_ROOT)}.zip"
    os.system(f'cd "{REPO_ROOT}" && zip -r "{zip_name}" "{os.path.basename(OUT_ROOT)}" >/dev/null')


if __name__ == "__main__":
    run_v16()

