from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.New_eval_trainD1D2_testD4 import eval_ckpt_on_d4, train_on_d1d2


DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v14")


def _base_env() -> Dict[str, str]:
    return {
        "P1_LR": "1e-2",
        "P1_SEED": "42",
        "P1_USE_LR_SCHEDULER": "1",
        "P1_SAVE_26D_LATENT": "1",
        # Config D: 16D does not participate in ODE/recon correction
        "P1_HEAD_USE_26D": "1",
        "P1_V8_HEAD_10D": "0",
        "P1_V8_ODE_CORR": "0",
        "P1_V8_RECON_CORR": "0",
        # In current training code, P1_HEAD_USE_26D branch uses e2e_head.
        # Keep this for fair comparability across Exp1/2/3 as requested.
        "P1_SEPARATE_HEAD_26D": "0",
        "P1_ZSCORE_TARGETS": "1",
        "LAMBDA_CLS": "0.0",
    }


def _diagnose_param_distribution(latent_csv: str) -> Dict[str, float]:
    d = pd.read_csv(latent_csv)
    out: Dict[str, float] = {}
    for k in ("tau_m", "Gb", "si", "mi"):
        v = pd.to_numeric(d[k], errors="coerce").to_numpy(dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(v))
        out[f"{k}_std"] = float(np.nanstd(v))
        out[f"{k}_cv"] = float(np.nanstd(v) / (abs(np.nanmean(v)) + 1e-8))
    # boundary saturation checks (narrow and wide canonical limits)
    lims = {
        "narrow": {"tau_m": 120.0, "Gb": 250.0, "si": 1e-3, "mi": 3.0},
        "wide": {"tau_m": 200.0, "Gb": 300.0, "si": 1e-2, "mi": 5.0},
    }
    for mode, mode_lims in lims.items():
        for p, hi in mode_lims.items():
            v = pd.to_numeric(d[p], errors="coerce").to_numpy(dtype=float)
            out[f"{p}_hit_hi_{mode}_pct"] = float(np.nanmean(np.isclose(v, hi, atol=max(1e-8, hi * 1e-6))) * 100.0)
    return out


def _phase_stability(phase1_csv: str, phase2_csv: str) -> Dict[str, float]:
    a = pd.read_csv(phase1_csv).groupby("subject_id", as_index=False)[["tau_m", "Gb", "si", "mi"]].mean()
    b = pd.read_csv(phase2_csv).groupby("subject_id", as_index=False)[["tau_m", "Gb", "si", "mi"]].mean()
    m = a.merge(b, on="subject_id", suffixes=("_p1", "_p2"))
    out: Dict[str, float] = {"n_subjects_overlap": float(len(m))}
    for p in ("tau_m", "Gb", "si", "mi"):
        x = m[f"{p}_p1"].to_numpy(dtype=float)
        y = m[f"{p}_p2"].to_numpy(dtype=float)
        out[f"{p}_mae_phase1_vs_phase2"] = float(np.nanmean(np.abs(x - y)))
    return out


def _eval_and_save(exp_dir: str, ckpt_path: str) -> Dict[str, Dict[str, float]]:
    d4_dir = os.path.join(exp_dir, "d4_eval")
    os.makedirs(d4_dir, exist_ok=True)
    m_sspg, _ = eval_ckpt_on_d4(
        cgm_project_output=DATA_ROOT,
        ckpt_path=ckpt_path,
        out_dir=d4_dir,
        target="sspg",
    )
    m_di, _ = eval_ckpt_on_d4(
        cgm_project_output=DATA_ROOT,
        ckpt_path=ckpt_path,
        out_dir=d4_dir,
        target="di",
    )
    mets = {"sspg": m_sspg.__dict__, "di": m_di.__dict__}
    with open(os.path.join(exp_dir, "D4_metrics.json"), "w") as f:
        json.dump(mets, f, indent=2)
    return mets


def run_v14() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _base_env()
    rows: List[Dict[str, float]] = []

    # Exp1: ConfigD + narrow + e2e weak supervision
    exp1_dir = os.path.join(OUT_ROOT, "v14_Exp1_ConfigD_Narrow_E2E")
    exp1_train = os.path.join(exp1_dir, "train")
    os.makedirs(exp1_dir, exist_ok=True)
    if not os.path.isfile(os.path.join(exp1_train, "autoencoder_p1_full.pt")):
        env1 = dict(base)
        env1.update(
            {
                "P1_V10_WIDE_BOUNDS": "0",
                "LAMBDA_IR": "0.05",
                "P1_FINETUNE_HEAD_ONLY": "0",
                "P1_PRETRAINED_MODEL": "",
            }
        )
        train_on_d1d2(
            cgm_project_output=DATA_ROOT,
            results_dir=exp1_train,
            seed=42,
            lambda_sspg=0.1,
            lambda_di=0.1,
            num_epochs=100,
            extra_env=env1,
        )
    exp1_ckpt = os.path.join(exp1_train, "autoencoder_p1_full.pt")
    m1 = _eval_and_save(exp1_dir, exp1_ckpt)
    d1 = _diagnose_param_distribution(os.path.join(exp1_train, "latent_and_gold_all_26d.csv"))
    with open(os.path.join(exp1_dir, "param_diagnostics.json"), "w") as f:
        json.dump(d1, f, indent=2)
    rows.append(
        {
            "experiment": "v14_Exp1_ConfigD_Narrow_E2E",
            "sspg_pearson_r": m1["sspg"]["pearson_r"],
            "sspg_rmse": m1["sspg"]["rmse"],
            "di_pearson_r": m1["di"]["pearson_r"],
            "di_rmse": m1["di"]["rmse"],
            "si_cv": d1["si_cv"],
            "mi_cv": d1["mi_cv"],
            "Gb_mean": d1["Gb_mean"],
        }
    )

    # Exp2: ConfigD + wide + e2e weak supervision
    exp2_dir = os.path.join(OUT_ROOT, "v14_Exp2_ConfigD_Wide_E2E")
    exp2_train = os.path.join(exp2_dir, "train")
    os.makedirs(exp2_dir, exist_ok=True)
    if not os.path.isfile(os.path.join(exp2_train, "autoencoder_p1_full.pt")):
        env2 = dict(base)
        env2.update(
            {
                "P1_V10_WIDE_BOUNDS": "1",
                "LAMBDA_IR": "0.05",
                "P1_FINETUNE_HEAD_ONLY": "0",
                "P1_PRETRAINED_MODEL": "",
            }
        )
        train_on_d1d2(
            cgm_project_output=DATA_ROOT,
            results_dir=exp2_train,
            seed=42,
            lambda_sspg=0.1,
            lambda_di=0.1,
            num_epochs=100,
            extra_env=env2,
        )
    exp2_ckpt = os.path.join(exp2_train, "autoencoder_p1_full.pt")
    m2 = _eval_and_save(exp2_dir, exp2_ckpt)
    d2 = _diagnose_param_distribution(os.path.join(exp2_train, "latent_and_gold_all_26d.csv"))
    with open(os.path.join(exp2_dir, "param_diagnostics.json"), "w") as f:
        json.dump(d2, f, indent=2)
    rows.append(
        {
            "experiment": "v14_Exp2_ConfigD_Wide_E2E",
            "sspg_pearson_r": m2["sspg"]["pearson_r"],
            "sspg_rmse": m2["sspg"]["rmse"],
            "di_pearson_r": m2["di"]["pearson_r"],
            "di_rmse": m2["di"]["rmse"],
            "si_cv": d2["si_cv"],
            "mi_cv": d2["mi_cv"],
            "Gb_mean": d2["Gb_mean"],
        }
    )

    # Exp3: ConfigD + wide + two phase decoupling
    exp3_dir = os.path.join(OUT_ROOT, "v14_Exp3_ConfigD_Wide_TwoPhase")
    exp3_p1 = os.path.join(exp3_dir, "phase1_unsupervised")
    exp3_p2 = os.path.join(exp3_dir, "phase2_finetune_head")
    os.makedirs(exp3_dir, exist_ok=True)

    ckpt_p1 = os.path.join(exp3_p1, "autoencoder_p1_full.pt")
    if not os.path.isfile(ckpt_p1):
        env3_p1 = dict(base)
        env3_p1.update(
            {
                "P1_V10_WIDE_BOUNDS": "1",
                "LAMBDA_IR": "0.0",
                "P1_FINETUNE_HEAD_ONLY": "0",
                "P1_PRETRAINED_MODEL": "",
            }
        )
        train_on_d1d2(
            cgm_project_output=DATA_ROOT,
            results_dir=exp3_p1,
            seed=42,
            lambda_sspg=0.0,
            lambda_di=0.0,
            num_epochs=100,
            extra_env=env3_p1,
        )

    ckpt_p2 = os.path.join(exp3_p2, "autoencoder_p1_full.pt")
    if not os.path.isfile(ckpt_p2):
        env3_p2 = dict(base)
        env3_p2.update(
            {
                "P1_V10_WIDE_BOUNDS": "1",
                "LAMBDA_IR": "0.05",
                "P1_FINETUNE_HEAD_ONLY": "1",
                "P1_PRETRAINED_MODEL": ckpt_p1,
            }
        )
        train_on_d1d2(
            cgm_project_output=DATA_ROOT,
            results_dir=exp3_p2,
            seed=42,
            lambda_sspg=0.1,
            lambda_di=0.1,
            num_epochs=100,
            extra_env=env3_p2,
        )

    m3 = _eval_and_save(exp3_dir, ckpt_p2)
    d3_p1 = _diagnose_param_distribution(os.path.join(exp3_p1, "latent_and_gold_all_26d.csv"))
    d3_p2 = _diagnose_param_distribution(os.path.join(exp3_p2, "latent_and_gold_all_26d.csv"))
    st3 = _phase_stability(
        os.path.join(exp3_p1, "latent_and_gold_all_26d.csv"),
        os.path.join(exp3_p2, "latent_and_gold_all_26d.csv"),
    )
    with open(os.path.join(exp3_dir, "param_diagnostics_phase1.json"), "w") as f:
        json.dump(d3_p1, f, indent=2)
    with open(os.path.join(exp3_dir, "param_diagnostics_phase2.json"), "w") as f:
        json.dump(d3_p2, f, indent=2)
    with open(os.path.join(exp3_dir, "phase_stability.json"), "w") as f:
        json.dump(st3, f, indent=2)
    rows.append(
        {
            "experiment": "v14_Exp3_ConfigD_Wide_TwoPhase",
            "sspg_pearson_r": m3["sspg"]["pearson_r"],
            "sspg_rmse": m3["sspg"]["rmse"],
            "di_pearson_r": m3["di"]["pearson_r"],
            "di_rmse": m3["di"]["rmse"],
            "si_cv": d3_p2["si_cv"],
            "mi_cv": d3_p2["mi_cv"],
            "Gb_mean": d3_p2["Gb_mean"],
            "si_mae_phase1_vs_phase2": st3["si_mae_phase1_vs_phase2"],
            "mi_mae_phase1_vs_phase2": st3["mi_mae_phase1_vs_phase2"],
        }
    )

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_ROOT, "v14_summary.csv"), index=False)
    rep_lines = ["# v14_report", "", f"run_stamp: {stamp}", "", summary.to_markdown(index=False)]
    with open(os.path.join(OUT_ROOT, "v14_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep_lines))

    zip_name = f"{os.path.basename(OUT_ROOT)}.zip"
    os.system(f'cd "{REPO_ROOT}" && zip -r "{zip_name}" "{os.path.basename(OUT_ROOT)}" >/dev/null')


if __name__ == "__main__":
    run_v14()

