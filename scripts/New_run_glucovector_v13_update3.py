from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Dict, List

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import scripts.New_run_glucovector_v13 as v13
from scripts.New_eval_trainD1D2_testD4 import train_on_d1d2


OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v13_update3")


def _run_exp_two_phase(name: str, env_base: Dict[str, str], lambda_sspg: float, lambda_di: float) -> Dict[str, object]:
    exp_dir = os.path.join(OUT_ROOT, name)
    p1_dir = os.path.join(exp_dir, "phase1_unsup")
    p2_dir = os.path.join(exp_dir, "train")
    os.makedirs(exp_dir, exist_ok=True)

    ck1 = os.path.join(p1_dir, "autoencoder_p1_full.pt")
    if not os.path.isfile(ck1):
        env1 = dict(env_base)
        env1["LAMBDA_DIV"] = "0.0"
        train_on_d1d2(
            cgm_project_output=v13.DATA_ROOT,
            results_dir=p1_dir,
            seed=21,
            lambda_sspg=0.0,
            lambda_di=0.0,
            num_epochs=50,
            extra_env=env1,
        )

    ck2 = os.path.join(p2_dir, "autoencoder_p1_full.pt")
    if not os.path.isfile(ck2):
        env2 = dict(env_base)
        env2["P1_RESUME_CKPT"] = ck1
        env2["LAMBDA_DIV"] = "0.1"
        train_on_d1d2(
            cgm_project_output=v13.DATA_ROOT,
            results_dir=p2_dir,
            seed=21,
            lambda_sspg=lambda_sspg,
            lambda_di=lambda_di,
            num_epochs=50,
            extra_env=env2,
        )

    # Reuse v13 evaluation/export helpers
    train_csv = os.path.join(exp_dir, "train_latent_and_gold.csv")
    train_df = v13._build_train_latent_and_gold(p2_dir, train_csv)
    d4_lat = v13._extract_d4_latent_26d(ck2)
    d4_pred_csv = os.path.join(exp_dir, "D4_predictions.csv")
    mets = v13._fit_ridge_and_predict(train_df, d4_lat, d4_pred_csv)
    import json, shutil
    with open(os.path.join(exp_dir, "D4_metrics.json"), "w") as f:
        json.dump(mets, f, indent=2)
    tm = os.path.join(p2_dir, "training_metrics.json")
    if os.path.isfile(tm):
        shutil.copy2(tm, os.path.join(exp_dir, "training_metrics.json"))
    return {"name": name, "metrics": mets, "exp_dir": exp_dir}


def run_v13_update3() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res: List[Dict[str, object]] = []

    # Exp1 single phase baseline
    res.append(
        v13._run_exp(
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

    env_exp2 = {
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
    }
    res.append(_run_exp_two_phase("Exp2_ODE10_WeakSupervision", env_exp2, 0.1, 0.1))

    env_exp3 = {
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
    }
    res.append(_run_exp_two_phase("Exp3_Hybrid26_WeakSupervision_Ortho", env_exp3, 0.1, 0.1))

    rows = []
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
    with open(os.path.join(OUT_ROOT, "v13_report.md"), "w", encoding="utf-8") as f:
        f.write("# v13_update3_report\n\n")
        f.write(f"run_stamp: {stamp}\n\n")
        f.write(s.to_markdown(index=False))

    os.system(f'cd "{REPO_ROOT}" && zip -r "New_paper1_results_glucovector_v13_update3.zip" "New_paper1_results_glucovector_v13_update3" >/dev/null')


if __name__ == "__main__":
    # ensure imported module writes to update3 folder when _run_exp is used
    v13.OUT_ROOT = OUT_ROOT
    run_v13_update3()

