from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.New_eval_trainD1D2_testD4 import eval_ckpt_on_d4, train_on_d1d2

DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v17")


def _base_env() -> Dict[str, str]:
    return {
        "P1_TRAIN_DATASETS": "D1,D2",
        "P1_LR": "1e-2",
        "P1_SEED": "42",
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
    }


def _train_single(exp_dir: str, env: Dict[str, str], lambda_sspg: float, lambda_di: float) -> str:
    train_dir = os.path.join(exp_dir, "joint_training")
    ckpt = os.path.join(train_dir, "autoencoder_p1_full.pt")
    if not os.path.isfile(ckpt):
        train_on_d1d2(
            cgm_project_output=DATA_ROOT,
            results_dir=train_dir,
            seed=42,
            lambda_sspg=lambda_sspg,
            lambda_di=lambda_di,
            num_epochs=100,
            extra_env=env,
        )
    return ckpt


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
    ck2 = os.path.join(p2_dir, "autoencoder_p1_full.pt")
    if not os.path.isfile(ck2):
        env2 = dict(env_p1)
        env2.update(env_p2_extra)
        env2["P1_FINETUNE_HEAD_ONLY"] = "1"
        env2["P1_PRETRAINED_MODEL"] = ck1
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
    out = {"sspg": ms.__dict__, "di": md.__dict__}
    # compression stats from saved scatter csv
    for t in ("sspg", "di"):
        p = os.path.join(out_dir, f"New_D4_{t}_true_vs_pred.csv")
        if os.path.isfile(p):
            d = pd.read_csv(p)
            true_std = float(np.nanstd(d["true"].to_numpy(dtype=float), ddof=0))
            pred_std = float(np.nanstd(d["pred"].to_numpy(dtype=float), ddof=0))
            out[t]["true_std"] = true_std
            out[t]["pred_std"] = pred_std
            out[t]["compression_ratio"] = float(pred_std / max(true_std, 1e-8))
    with open(os.path.join(out_dir, "metrics_d4_full.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def _run_mealtype_eval(ckpt: str, out_csv: str) -> pd.DataFrame:
    script = os.path.join(REPO_ROOT, "scripts", "New_run_exp3_d4_mealtype_v7.py")
    if not os.path.isfile(script):
        return pd.DataFrame()
    os.system(f'python "{script}" --data_root "{DATA_ROOT}" --ckpt "{ckpt}" --out_csv "{out_csv}" >/dev/null 2>&1')
    if os.path.isfile(out_csv):
        return pd.read_csv(out_csv)
    return pd.DataFrame()


def _run_shap_simple(ckpt: str, train_latent_csv: str, out_dir: str) -> Dict[str, Any]:
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
    head = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 64), torch.nn.ReLU(), torch.nn.Dropout(0.0),
        torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
    )
    head.load_state_dict(state, strict=True)
    head.eval()
    feat = ["tau_m", "Gb", "sg", "si", "p2", "mi"] + [f"z_init_{i}" for i in range(4)] + [f"z_nonseq_{i}" for i in range(16)]
    feat = feat[:10] if in_dim == 10 else feat
    d = pd.read_csv(train_latent_csv)
    feat = [c for c in feat if c in d.columns]
    X = d[feat].dropna().to_numpy(dtype=float)
    if len(X) < 20:
        return {}
    bg = X[:min(40, len(X))]
    x_eval = X[:min(60, len(X))]

    def f_sspg(x):
        with torch.no_grad():
            return head(torch.tensor(x, dtype=torch.float32))[:, 0].numpy()

    ex = shap.KernelExplainer(f_sspg, bg)
    sv = ex.shap_values(x_eval, nsamples=100)
    shap.summary_plot(sv, x_eval, feature_names=feat, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary_sspg.png"), dpi=140, bbox_inches="tight")
    plt.close()
    imp = np.mean(np.abs(np.asarray(sv)), axis=0)
    imp_df = pd.DataFrame({"feature": feat, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False)
    imp_df["pct"] = imp_df["mean_abs_shap"] / max(imp_df["mean_abs_shap"].sum(), 1e-12) * 100.0
    imp_df.to_csv(os.path.join(out_dir, "shap_feature_importance.csv"), index=False)
    ode = {"tau_m", "Gb", "sg", "si", "p2", "mi", "z_init_0", "z_init_1", "z_init_2", "z_init_3"}
    total = max(float(imp_df["mean_abs_shap"].sum()), 1e-12)
    out = {
        "shap_10d_share": float(imp_df[imp_df["feature"].isin(ode)]["mean_abs_shap"].sum() / total * 100.0),
        "shap_16d_share": float(imp_df[~imp_df["feature"].isin(ode)]["mean_abs_shap"].sum() / total * 100.0),
    }
    with open(os.path.join(out_dir, "shap_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def _record_result(name: str, mets: Dict[str, Any], meal_df: pd.DataFrame) -> Dict[str, Any]:
    r: Dict[str, Any] = {"experiment": name}
    for t in ("sspg", "di"):
        m = mets[t]
        r[f"d4_{t}_pearson_r"] = m.get("pearson_r")
        r[f"d4_{t}_spearman_r"] = m.get("spearman_r")
        r[f"d4_{t}_r2"] = m.get("r2")
        r[f"d4_{t}_rmse"] = m.get("rmse")
        r[f"d4_{t}_mae"] = m.get("mae")
        r[f"{t}_pred_std"] = m.get("pred_std")
        r[f"{t}_true_std"] = m.get("true_std")
        r[f"{t}_compression_ratio"] = m.get("compression_ratio")
    if not meal_df.empty:
        r["protein_sspg_spearman"] = float(meal_df.loc[meal_df["meal_type"] == "Protein_bar", "sspg_spearman_r"].iloc[0])
        r["protein_sspg_pearson"] = float(meal_df.loc[meal_df["meal_type"] == "Protein_bar", "sspg_pearson_r"].iloc[0])
        r["protein_di_pearson"] = float(meal_df.loc[meal_df["meal_type"] == "Protein_bar", "di_pearson_r"].iloc[0])
    return r


def run_v17() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows: List[Dict[str, Any]] = []

    # Exp1 Wang baseline
    exp1 = os.path.join(OUT_ROOT, "v17_Exp1_Wang_Baseline")
    env1 = _base_env()
    env1.update({"P1_HEAD_USE_26D": "1", "P1_V8_HEAD_10D": "1", "P1_V10_WIDE_BOUNDS": "0", "P1_FINETUNE_HEAD_ONLY": "0", "P1_PRETRAINED_MODEL": ""})
    ck1 = _train_single(exp1, env1, lambda_sspg=0.1, lambda_di=0.1)
    m1 = _eval_d4(ck1, os.path.join(exp1, "eval_D4"))
    mt1 = _run_mealtype_eval(ck1, os.path.join(exp1, "d4_per_meal_type.csv"))
    all_rows.append(_record_result("v17_Exp1_Wang_Baseline", m1, mt1))

    # Exp2 GV baseline
    exp2 = os.path.join(OUT_ROOT, "v17_Exp2_GV_Baseline")
    env2 = _base_env()
    env2.update({"P1_HEAD_USE_26D": "1", "P1_V8_HEAD_10D": "0", "P1_V10_WIDE_BOUNDS": "1"})
    ck2 = _train_two_phase(exp2, env2, {})
    m2 = _eval_d4(ck2, os.path.join(exp2, "eval_D4"))
    mt2 = _run_mealtype_eval(ck2, os.path.join(exp2, "d4_per_meal_type.csv"))
    row2 = _record_result("v17_Exp2_GV_Baseline", m2, mt2)
    shap2 = _run_shap_simple(ck2, os.path.join(exp2, "phase2_finetune_head", "latent_and_gold_all_26d.csv"), os.path.join(exp2, "shap_analysis"))
    row2.update(shap2)
    all_rows.append(row2)

    # Exp3 VarMatch
    exp3 = os.path.join(OUT_ROOT, "v17_Exp3_VarMatch")
    env3 = _base_env()
    env3.update({"P1_HEAD_USE_26D": "1", "P1_V8_HEAD_10D": "0", "P1_V10_WIDE_BOUNDS": "1"})
    ck3 = _train_two_phase(exp3, env3, {"LAMBDA_VAR_MATCH": "0.05"})
    m3 = _eval_d4(ck3, os.path.join(exp3, "eval_D4"))
    mt3 = _run_mealtype_eval(ck3, os.path.join(exp3, "d4_per_meal_type.csv"))
    all_rows.append(_record_result("v17_Exp3_VarMatch", m3, mt3))

    # Exp4 Ortho16D
    exp4 = os.path.join(OUT_ROOT, "v17_Exp4_Ortho_16D")
    env4 = _base_env()
    env4.update({"P1_HEAD_USE_26D": "1", "P1_V8_HEAD_10D": "0", "P1_V10_WIDE_BOUNDS": "1"})
    ck4 = _train_two_phase(exp4, env4, {"LAMBDA_ORTHO_16D": "0.1"})
    m4 = _eval_d4(ck4, os.path.join(exp4, "eval_D4"))
    mt4 = _run_mealtype_eval(ck4, os.path.join(exp4, "d4_per_meal_type.csv"))
    all_rows.append(_record_result("v17_Exp4_Ortho_16D", m4, mt4))

    # Exp5 Finetune16D
    exp5 = os.path.join(OUT_ROOT, "v17_Exp5_Finetune_16D")
    env5 = _base_env()
    env5.update({"P1_HEAD_USE_26D": "1", "P1_V8_HEAD_10D": "0", "P1_V10_WIDE_BOUNDS": "1"})
    ck5 = _train_two_phase(exp5, env5, {"P1_FINETUNE_16D_ONLY": "1"})
    m5 = _eval_d4(ck5, os.path.join(exp5, "eval_D4"))
    mt5 = _run_mealtype_eval(ck5, os.path.join(exp5, "d4_per_meal_type.csv"))
    all_rows.append(_record_result("v17_Exp5_Finetune_16D", m5, mt5))

    # Exp6 Combined
    exp6 = os.path.join(OUT_ROOT, "v17_Exp6_Combined")
    env6 = _base_env()
    env6.update({"P1_HEAD_USE_26D": "1", "P1_V8_HEAD_10D": "0", "P1_V10_WIDE_BOUNDS": "1"})
    ck6 = _train_two_phase(exp6, env6, {"LAMBDA_VAR_MATCH": "0.05", "LAMBDA_ORTHO_16D": "0.1", "P1_FINETUNE_16D_ONLY": "1"})
    m6 = _eval_d4(ck6, os.path.join(exp6, "eval_D4"))
    mt6 = _run_mealtype_eval(ck6, os.path.join(exp6, "d4_per_meal_type.csv"))
    row6 = _record_result("v17_Exp6_Combined", m6, mt6)
    shap6 = _run_shap_simple(ck6, os.path.join(exp6, "phase2_finetune_head", "latent_and_gold_all_26d.csv"), os.path.join(exp6, "shap_analysis"))
    row6.update(shap6)
    all_rows.append(row6)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_ROOT, "v17_comprehensive_summary.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(OUT_ROOT, "v17_comprehensive_summary.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    with open(os.path.join(OUT_ROOT, "v17_report.md"), "w", encoding="utf-8") as f:
        f.write("# GlucoVector v17 Experiment Results\n\n")
        f.write(f"Generated: {run_stamp}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    zip_name = f"{os.path.basename(OUT_ROOT)}.zip"
    os.system(f'cd "{REPO_ROOT}" && zip -r "{zip_name}" "{os.path.basename(OUT_ROOT)}" >/dev/null')


if __name__ == "__main__":
    run_v17()

