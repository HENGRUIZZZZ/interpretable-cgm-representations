"""
New_ (GlucoVector v6) utilities:

- Train on D1+D2 via existing run_p1_full_pipeline.py (with CGM_PROJECT_OUTPUT pointing to the new dataset root)
- Independently evaluate on D4 (Pearson r, RMSE) and save (true, pred) CSV for scatter plots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@dataclass(frozen=True)
class EvalResult:
    n: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    r2: float
    rmse: float
    mae: float


def _comprehensive_metrics(y: np.ndarray, yhat: np.ndarray) -> EvalResult:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y = y[ok]
    yhat = yhat[ok]
    if len(y) < 3:
        return EvalResult(
            n=int(len(y)),
            pearson_r=float("nan"),
            pearson_p=float("nan"),
            spearman_r=float("nan"),
            spearman_p=float("nan"),
            r2=float("nan"),
            rmse=float("nan"),
            mae=float("nan"),
        )
    pr, pp = stats.pearsonr(y, yhat)
    sr, sp = stats.spearmanr(y, yhat)
    return EvalResult(
        n=int(len(y)),
        pearson_r=float(pr),
        pearson_p=float(pp),
        spearman_r=float(sr),
        spearman_p=float(sp),
        r2=float(r2_score(y, yhat)),
        rmse=float(np.sqrt(mean_squared_error(y, yhat))),
        mae=float(mean_absolute_error(y, yhat)),
    )


def _device() -> torch.device:
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def _build_e2e_head(input_dim: int = 26) -> torch.nn.Module:
    # Must match run_p1_full_pipeline.py (V6 e2e_head)
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2),
    )


def _build_small_head(input_dim: int = 26) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 16),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(16, 1),
    )


def _infer_e2e_input_dim(e2e_state: Dict[str, torch.Tensor]) -> int:
    # first linear layer weight shape: [64, input_dim]
    w = e2e_state.get("0.weight", None)
    if w is None:
        return 26
    return int(w.shape[1])


def train_on_d1d2(
    *,
    cgm_project_output: str,
    results_dir: str,
    seed: int,
    lambda_sspg: float,
    lambda_di: float,
    num_epochs: int = 100,
    extra_env: Optional[Dict[str, str]] = None,
) -> None:
    env = os.environ.copy()
    env["CGM_PROJECT_OUTPUT"] = cgm_project_output
    env["P1_TRAIN_DATASETS"] = "D1,D2"
    env["P1_RESULTS_DIR"] = results_dir
    env["P1_SEED"] = str(seed)
    env["P1_NUM_EPOCHS"] = str(num_epochs)
    env["P1_ONE_MEAL_PER_SUBJECT"] = "1"
    env["P1_SAVE_26D_LATENT"] = "1"
    env["LAMBDA_SSPG"] = str(lambda_sspg)
    env["LAMBDA_DI"] = str(lambda_di)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    os.makedirs(results_dir, exist_ok=True)

    cmd = [
        "python",
        os.path.join(REPO_ROOT, "run_p1_full_pipeline.py"),
        "--lambda_sspg",
        str(lambda_sspg),
        "--lambda_di",
        str(lambda_di),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _load_ckpt(ckpt_path: str) -> dict:
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _load_d4_batch(cgm_project_output: str):
    from load_cgm_project_data import load_cgm_project_level3

    b, info, lab = load_cgm_project_level3(dataset_id="D4", output_base=cgm_project_output)
    return b, info, lab


def eval_ckpt_on_d4(
    *,
    cgm_project_output: str,
    ckpt_path: str,
    out_dir: str,
    target: str,
    metrics_filename: Optional[str] = None,
    scatter_filename: Optional[str] = None,
) -> Tuple[EvalResult, str]:
    """
    Evaluate a trained checkpoint on D4.
    Uses the V6 26D e2e head in the checkpoint (if present).

    Returns: (metrics, scatter_csv_path)
    """
    assert target in ("sspg", "di")
    os.makedirs(out_dir, exist_ok=True)

    ckpt = _load_ckpt(ckpt_path)
    train_mean = ckpt["train_mean"]
    train_std = ckpt["train_std"]

    device = _device()

    from models import MechanisticAutoencoder
    from load_cgm_project_data import MEAL_COVARIATES, DEMOGRAPHICS_COVARIATES

    model = MechanisticAutoencoder(
        meal_size=len(MEAL_COVARIATES),
        demographics_size=len(DEMOGRAPHICS_COVARIATES),
        embedding_size=8,
        hidden_size=32,
        num_layers=2,
        encoder_dropout_prob=0.0,
        decoder_dropout_prob=0.5,
    ).to(device)
    # Infer optional v8 modules from checkpoint keys and enable corresponding forward paths.
    ms = ckpt.get("model_state", {})
    if hasattr(model, "use_v8_recon_corr"):
        model.use_v8_recon_corr = bool(ckpt.get("P1_V8_RECON_CORR", any(k.startswith("correction_mlp.") for k in ms.keys())))
    if hasattr(model, "use_v8_ode_corr"):
        model.use_v8_ode_corr = bool(ckpt.get("P1_V8_ODE_CORR", any(k.startswith("ode_correction.") for k in ms.keys())))
    # strict=False to allow optional architecture-specific blocks.
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    separate_heads = False
    if "e2e_head_state" in ckpt:
        e2e_in_dim = _infer_e2e_input_dim(ckpt["e2e_head_state"])
        e2e = _build_e2e_head(e2e_in_dim).to(device)
        e2e.load_state_dict(ckpt["e2e_head_state"], strict=True)
        e2e.eval()
        sspg_h = None
        di_h = None
    elif "sspg_head_state" in ckpt and "di_head_state" in ckpt:
        # v18 separate heads fallback
        separate_heads = True
        w = ckpt["sspg_head_state"].get("0.weight", None)
        in_dim = int(w.shape[1]) if w is not None else 26
        e2e_in_dim = in_dim
        e2e = None
        sspg_h = _build_small_head(in_dim).to(device)
        di_h = _build_small_head(in_dim).to(device)
        sspg_h.load_state_dict(ckpt["sspg_head_state"], strict=True)
        di_h.load_state_dict(ckpt["di_head_state"], strict=True)
        sspg_h.eval()
        di_h.eval()
    else:
        raise RuntimeError("Checkpoint missing compatible prediction head state.")

    b, info, lab = _load_d4_batch(cgm_project_output)

    # Normalize using train statistics stored in ckpt (same order as normalize_train_test in data_utils.py)
    cgm = (b.cgm - train_mean[0]) / (train_std[0] + 1e-8)
    ts = (b.timestamps - train_mean[1]) / (train_std[1] + 1e-8)
    meals = (b.meals - train_mean[2]) / (train_std[2] + 1e-8)
    demo = (b.demographics - train_mean[3]) / (train_std[3] + 1e-8)

    tens = (
        torch.tensor(cgm, dtype=torch.float32, device=device),
        torch.tensor(ts, dtype=torch.float32, device=device),
        torch.tensor(meals, dtype=torch.float32, device=device),
        torch.tensor(demo, dtype=torch.float32, device=device),
    )

    with torch.no_grad():
        p26, init26, z16 = model.get_all_latents(*tens)
        head_in = torch.cat([p26, init26], dim=-1) if e2e_in_dim == 10 else torch.cat([p26, init26, z16], dim=-1)
        if separate_heads:
            s = sspg_h(head_in).squeeze(-1)
            d = di_h(head_in).squeeze(-1)
            pred_2 = torch.stack([s, d], dim=-1).detach().cpu().numpy()
        else:
            pred_2 = e2e(head_in).detach().cpu().numpy()

    # Convert to per-sample pred values
    sspg_hat = pred_2[:, 0].astype(np.float64)
    di_hat = pred_2[:, 1].astype(np.float64)

    # Un-zscore if training used z-scoring
    if bool(ckpt.get("P1_ZSCORE_TARGETS", False)):
        sspg_hat = sspg_hat * float(ckpt["sspg_std"]) + float(ckpt["sspg_mean"])
        di_hat = di_hat * float(ckpt["di_std"]) + float(ckpt["di_mean"])

    pred = sspg_hat if target == "sspg" else di_hat

    # Build D4 ID mapping (cgm ID like D4_2133-001 / D4_1636-69-001 -> labels subject_id like D3H_001)
    subj_path = os.path.join(cgm_project_output, "D4_hall", "subjects.csv")
    id_map = {}
    if os.path.isfile(subj_path):
        s_df = pd.read_csv(subj_path)
        if {"subject_id", "original_id"}.issubset(set(s_df.columns)):
            for _, r in s_df.dropna(subset=["subject_id", "original_id"]).iterrows():
                orig = str(r["original_id"]).strip()
                sid = str(r["subject_id"]).strip()
                id_map[orig] = sid
                id_map[f"D4_{orig}"] = sid

    # Get gold label per subject
    if lab is None or lab.empty or target not in lab.columns:
        raise RuntimeError(f"D4 labels missing {target}.")
    gold_by_subject = (
        lab[["subject_id", target]]
        .dropna()
        .drop_duplicates(subset=["subject_id"])
        .set_index("subject_id")[target]
    )

    # Aggregate predictions per subject (mean across windows)
    subj = np.asarray(info.patient_ids).astype(str)
    subj = np.array([id_map.get(s, s) for s in subj], dtype=str)
    df_pred = pd.DataFrame({"subject_id": subj, "pred": pred})
    pred_by_subject = df_pred.groupby("subject_id")["pred"].mean()

    # Align
    common = sorted(set(pred_by_subject.index.astype(str)) & set(gold_by_subject.index.astype(str)))
    y = gold_by_subject.loc[common].astype(float).to_numpy()
    yhat = pred_by_subject.loc[common].astype(float).to_numpy()

    if len(y) < 3:
        raise RuntimeError(f"Too few aligned subjects for D4 {target}: n={len(y)}")

    res = _comprehensive_metrics(y, yhat)

    scatter_path = os.path.join(out_dir, scatter_filename or f"New_D4_{target}_true_vs_pred.csv")
    pd.DataFrame({"subject_id": common, "true": y, "pred": yhat}).to_csv(
        scatter_path, index=False
    )

    metrics_path = os.path.join(out_dir, metrics_filename or f"New_D4_{target}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(res.__dict__, f, indent=2)

    return res, scatter_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cgm_project_output", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--target", required=True, choices=["sspg", "di"])
    ap.add_argument("--metrics_filename", type=str, default=None)
    ap.add_argument("--scatter_filename", type=str, default=None)
    args = ap.parse_args()

    res, scatter = eval_ckpt_on_d4(
        cgm_project_output=args.cgm_project_output,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        target=args.target,
        metrics_filename=args.metrics_filename,
        scatter_filename=args.scatter_filename,
    )
    print(
        f"D4 {args.target}: n={res.n} Pearson={res.pearson_r:.4f} "
        f"Spearman={res.spearman_r:.4f} R2={res.r2:.4f} RMSE={res.rmse:.4f} MAE={res.mae:.4f}"
    )
    print(f"Saved scatter CSV: {scatter}")


if __name__ == "__main__":
    main()

