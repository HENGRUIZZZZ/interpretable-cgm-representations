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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import MechanisticAutoencoder
from scripts.New_eval_trainD1D2_testD4 import train_on_d1d2
from scripts.New_run_glucovector_v19 import (
    NNPredictor,
    _auc_with_ci,
    _build_d4_windows,
    _icc_oneway,
    _metrics,
)

DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
V19_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v19")
OUT_ROOT = os.environ.get(
    "S1_OUT_ROOT",
    os.path.join(REPO_ROOT, f"New_paper1_results_glucovector_S1_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
)
SEED = 42


def _bootstrap_ci_spearman(y: np.ndarray, p: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    point = float(stats.spearmanr(y, p)[0]) if len(y) >= 3 else np.nan
    if len(y) < 6:
        return point, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        vals.append(float(stats.spearmanr(y[idx], p[idx])[0]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def _subject_level_true_pred(pred_df: pd.DataFrame, label_map: pd.DataFrame) -> pd.DataFrame:
    sub = pred_df.groupby("subject_id", as_index=False)[["sspg_pred", "di_pred"]].mean()
    sub["sspg_true"] = sub["subject_id"].map(label_map["sspg"])
    sub["di_true"] = sub["subject_id"].map(label_map["di"])
    return sub


def _collect_plan_a() -> Dict[str, object]:
    overall = pd.read_csv(os.path.join(V19_ROOT, "v19_overall_metrics.csv"))
    clf = pd.read_csv(os.path.join(V19_ROOT, "v19_joint_classification_metrics.csv"))
    icc = pd.read_csv(os.path.join(V19_ROOT, "v19_icc_across_meals.csv"))

    model = "GV_Baseline(Exp5)"
    oa = overall[overall["model"] == model].iloc[0].to_dict()
    ca = clf[clf["model"] == model].iloc[0].to_dict()
    ia = icc[icc["model"] == model].iloc[0].to_dict()

    return {
        "model": model,
        "overall": oa,
        "classification": ca,
        "icc": ia,
        "full_overall_table": overall,
        "full_clf_table": clf,
        "full_icc_table": icc,
    }


def _run_plan_b_training() -> str:
    out_dir = os.path.join(OUT_ROOT, "S1_planB_E2E_10D")
    ckpt_path = os.path.join(out_dir, "autoencoder_p1_full.pt")
    if os.path.isfile(ckpt_path):
        return ckpt_path
    env = {
        "P1_V8_HEAD_10D": "1",
        "P1_V10_WIDE_BOUNDS": "1",
        "P1_ZSCORE_TARGETS": "1",
        "P1_ONE_MEAL_PER_SUBJECT": "1",
        "P1_SAVE_26D_LATENT": "1",
        "P1_USE_LR_SCHEDULER": "1",
        "P1_FINETUNE_HEAD_ONLY": "0",
    }
    train_on_d1d2(
        cgm_project_output=DATA_ROOT,
        results_dir=out_dir,
        seed=SEED,
        lambda_sspg=0.1,
        lambda_di=0.1,
        num_epochs=160,
        extra_env=env,
    )
    return ckpt_path


def _evaluate_plan_b(ckpt_path: str) -> Dict[str, object]:
    windows_df, labels_df = _build_d4_windows()
    label_map = labels_df.set_index("subject_id")
    pred = _predict_windows_generic(ckpt_path, windows_df)
    sub = _subject_level_true_pred(pred, label_map)

    ms = _metrics(sub["sspg_true"].to_numpy(float), sub["sspg_pred"].to_numpy(float))
    md = _metrics(sub["di_true"].to_numpy(float), sub["di_pred"].to_numpy(float))

    # classification
    y_ir = (sub["sspg_true"].to_numpy(float) >= 120.0).astype(int)
    s_ir = sub["sspg_pred"].to_numpy(float)
    ir_auc = _auc_with_ci(y_ir, s_ir)
    y_dec = ((sub["sspg_true"].to_numpy(float) >= 120.0) & (sub["di_true"].to_numpy(float) < 1.0)).astype(int)
    dec_score = sub["sspg_pred"].to_numpy(float) - 40.0 * sub["di_pred"].to_numpy(float)
    dec_auc = _auc_with_ci(y_dec, dec_score)

    # cross-meal ICC
    wide_s = pred.pivot_table(index="subject_id", columns="meal_type", values="sspg_pred", aggfunc="mean")
    wide_d = pred.pivot_table(index="subject_id", columns="meal_type", values="di_pred", aggfunc="mean")
    cols = [c for c in ["Cornflakes", "PB_sandwich", "Protein_bar"] if c in wide_s.columns]
    ws = wide_s[cols].dropna()
    wd = wide_d[cols].dropna()
    icc_sspg = _icc_oneway(ws.to_numpy()) if len(ws) >= 3 and len(cols) >= 2 else np.nan
    icc_di = _icc_oneway(wd.to_numpy()) if len(wd) >= 3 and len(cols) >= 2 else np.nan

    # mechanism alignment proxies on subject medians
    med = pred.groupby("subject_id", as_index=False).median(numeric_only=True)
    for col in ["z03", "z05"]:
        if col not in med.columns:
            med[col] = np.nan
    med["sspg_true"] = med["subject_id"].map(label_map["sspg"])
    med["di_true"] = med["subject_id"].map(label_map["di"])
    d_si = med[["z03", "sspg_true"]].dropna()
    d_mi = med[["z05", "di_true"]].dropna()
    rho_si = float(stats.spearmanr(d_si["z03"], d_si["sspg_true"])[0]) if len(d_si) >= 3 else np.nan
    rho_mi = float(stats.spearmanr(d_mi["z05"], d_mi["di_true"])[0]) if len(d_mi) >= 3 else np.nan

    # store aligned predictions for CI
    sub.to_csv(os.path.join(OUT_ROOT, "S1_planB_subject_true_pred.csv"), index=False)

    return {
        "overall_sspg": ms,
        "overall_di": md,
        "ir_auroc": ir_auc,
        "decomp_auroc": dec_auc,
        "icc_sspg_pred": icc_sspg,
        "icc_di_pred": icc_di,
        "si_vs_sspg_spearman": rho_si,
        "mi_vs_di_spearman": rho_mi,
        "subject_table": sub,
    }


def _predict_windows_generic(ckpt_path: str, windows_df: pd.DataFrame) -> pd.DataFrame:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MechanisticAutoencoder(
        meal_size=6, demographics_size=3, embedding_size=8, hidden_size=32, num_layers=2, encoder_dropout_prob=0.0, decoder_dropout_prob=0.5
    )
    ms = ckpt.get("model_state", {})
    if hasattr(model, "use_v8_recon_corr"):
        model.use_v8_recon_corr = bool(ckpt.get("P1_V8_RECON_CORR", any(k.startswith("correction_mlp.") for k in ms.keys())))
    if hasattr(model, "use_v8_ode_corr"):
        model.use_v8_ode_corr = bool(ckpt.get("P1_V8_ODE_CORR", any(k.startswith("ode_correction.") for k in ms.keys())))
    model.load_state_dict(ms, strict=False)
    model.eval()

    e2e = s_head = d_head = None
    in_dim = 10
    if "e2e_head_state" in ckpt:
        e2e_state = ckpt["e2e_head_state"]
        in_dim = int(e2e_state["0.weight"].shape[1])
        e2e = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
        )
        e2e.load_state_dict(e2e_state, strict=True)
        e2e.eval()
    elif "sspg_head_state" in ckpt and "di_head_state" in ckpt:
        s_state = ckpt["sspg_head_state"]
        d_state = ckpt["di_head_state"]
        if "0.weight" in s_state:
            in_dim = int(s_state["0.weight"].shape[1])
            s_head = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 16), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(16, 1)
            )
            d_head = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 16), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(16, 1)
            )
        else:
            in_dim = int(s_state["weight"].shape[1])
            s_head = torch.nn.Linear(in_dim, 1)
            d_head = torch.nn.Linear(in_dim, 1)
        s_head.load_state_dict(s_state, strict=True)
        d_head.load_state_dict(d_state, strict=True)
        s_head.eval()
        d_head.eval()
    else:
        raise RuntimeError("Unsupported checkpoint head structure for Plan B.")

    tm = ckpt["train_mean"]
    ts = ckpt["train_std"]
    zscore = bool(ckpt.get("P1_ZSCORE_TARGETS", False))
    s_mean, s_std = float(ckpt.get("sspg_mean", 0.0)), float(ckpt.get("sspg_std", 1.0))
    d_mean, d_std = float(ckpt.get("di_mean", 0.0)), float(ckpt.get("di_std", 1.0))

    rows = []
    for _, r in windows_df.iterrows():
        c = r["curve"][None, :, None].astype(np.float32)
        t = r["timestamps"][None, :, None].astype(np.float32)
        m = r["meal_series"][None, :, :].astype(np.float32)
        d = r["demographics"][None, :].astype(np.float32)
        c = (c - tm[0]) / (ts[0] + 1e-8)
        t = (t - tm[1]) / (ts[1] + 1e-8)
        m = (m - tm[2]) / (ts[2] + 1e-8)
        d = (d - tm[3]) / (ts[3] + 1e-8)
        with torch.no_grad():
            p26, init26, z16 = model.get_all_latents(torch.tensor(c), torch.tensor(t), torch.tensor(m), torch.tensor(d))
            if in_dim == 6:
                h = p26
            elif in_dim == 10:
                h = torch.cat([p26, init26], dim=-1)
            else:
                h = torch.cat([p26, init26, z16], dim=-1)
            if e2e is not None:
                y = e2e(h).squeeze(0)
                s_hat = float(y[0].item())
                d_hat = float(y[1].item())
            else:
                s_hat = float(s_head(h).squeeze().item())
                d_hat = float(d_head(h).squeeze().item())
            if zscore:
                s_hat = s_hat * s_std + s_mean
                d_hat = d_hat * d_std + d_mean
        row = {"subject_id": r["subject_id"], "meal_type": r["meal_type"], "sspg_pred": s_hat, "di_pred": d_hat}
        for i in range(6):
            row[f"z{i:02d}"] = float(p26.squeeze(0)[i].item())
        for i in range(4):
            row[f"z{6+i:02d}"] = float(init26.squeeze(0)[i].item())
        rows.append(row)
    return pd.DataFrame(rows)


def _success_criteria(plan_b: Dict[str, object]) -> Dict[str, object]:
    sspg_rho = float(plan_b["overall_sspg"]["spearman_r"])
    di_rho = float(plan_b["overall_di"]["spearman_r"])
    icc_sspg = float(plan_b["icc_sspg_pred"])
    si_rho = float(plan_b["si_vs_sspg_spearman"])
    mi_rho = float(plan_b["mi_vs_di_spearman"])

    c1 = (si_rho < -0.5) and (mi_rho > 0.5)
    c2 = (sspg_rho >= 0.70) and (di_rho >= 0.60)
    c3 = icc_sspg > 0.5
    return {"C1_param_alignment": c1, "C2_accuracy": c2, "C3_icc": c3, "all_pass": c1 and c2 and c3}


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)

    plan_a = _collect_plan_a()
    ckpt_b = _run_plan_b_training()
    plan_b = _evaluate_plan_b(ckpt_b)
    crit = _success_criteria(plan_b)

    # bootstrap CIs for S1 focal model A and B
    a_sub = pd.read_csv(os.path.join(V19_ROOT, "v19_per_meal_metrics.csv"))  # only for archive reference
    _ = a_sub  # intentionally kept for traceability
    b_sub = plan_b["subject_table"]
    s_point, s_lo, s_hi = _bootstrap_ci_spearman(b_sub["sspg_true"].to_numpy(float), b_sub["sspg_pred"].to_numpy(float))
    d_point, d_lo, d_hi = _bootstrap_ci_spearman(b_sub["di_true"].to_numpy(float), b_sub["di_pred"].to_numpy(float))

    summary = {
        "generated_at": datetime.now().isoformat(),
        "output_root": OUT_ROOT,
        "plan_a_model": plan_a["model"],
        "plan_a_sspg_spearman": float(plan_a["overall"]["sspg_spearman_r"]),
        "plan_a_di_spearman": float(plan_a["overall"]["di_spearman_r"]),
        "plan_a_ir_auroc": float(plan_a["classification"]["ir_auroc"]),
        "plan_a_decomp_auroc": float(plan_a["classification"]["decomp_auroc"]),
        "plan_a_icc_sspg": float(plan_a["icc"]["icc_sspg_pred"]),
        "plan_b_ckpt": ckpt_b,
        "plan_b_sspg_spearman": float(plan_b["overall_sspg"]["spearman_r"]),
        "plan_b_di_spearman": float(plan_b["overall_di"]["spearman_r"]),
        "plan_b_sspg_spearman_ci": [s_lo, s_hi],
        "plan_b_di_spearman_ci": [d_lo, d_hi],
        "plan_b_ir_auroc": float(plan_b["ir_auroc"]["auroc"]),
        "plan_b_decomp_auroc": float(plan_b["decomp_auroc"]["auroc"]),
        "plan_b_icc_sspg": float(plan_b["icc_sspg_pred"]),
        "plan_b_si_vs_sspg_spearman": float(plan_b["si_vs_sspg_spearman"]),
        "plan_b_mi_vs_di_spearman": float(plan_b["mi_vs_di_spearman"]),
        "criteria": crit,
    }
    with open(os.path.join(OUT_ROOT, "S1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plan_a["full_overall_table"].to_csv(os.path.join(OUT_ROOT, "S1_planA_overall_table_from_v19.csv"), index=False)
    plan_a["full_clf_table"].to_csv(os.path.join(OUT_ROOT, "S1_planA_clf_table_from_v19.csv"), index=False)
    plan_a["full_icc_table"].to_csv(os.path.join(OUT_ROOT, "S1_planA_icc_table_from_v19.csv"), index=False)
    pd.DataFrame([plan_b["overall_sspg"]]).to_csv(os.path.join(OUT_ROOT, "S1_planB_overall_sspg.csv"), index=False)
    pd.DataFrame([plan_b["overall_di"]]).to_csv(os.path.join(OUT_ROOT, "S1_planB_overall_di.csv"), index=False)

    report = []
    report.append("# GlucoVector S1 Full Execution Report")
    report.append("")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("")
    report.append("## Plan A (Risk stratification, from v19)")
    report.append("")
    report.append(f"- Model: `{plan_a['model']}`")
    report.append(f"- SSPG Spearman: **{plan_a['overall']['sspg_spearman_r']:.3f}**")
    report.append(f"- DI Spearman: **{plan_a['overall']['di_spearman_r']:.3f}**")
    report.append(f"- IR AUROC: **{plan_a['classification']['ir_auroc']:.3f}**")
    report.append(f"- Decomp AUROC: **{plan_a['classification']['decomp_auroc']:.3f}**")
    report.append(f"- Cross-meal ICC (SSPG pred): **{plan_a['icc']['icc_sspg_pred']:.3f}**")
    report.append("")
    report.append("## Plan B (E2E Joint, 10D-only)")
    report.append("")
    report.append(f"- Checkpoint: `{ckpt_b}`")
    report.append(f"- SSPG Spearman: **{plan_b['overall_sspg']['spearman_r']:.3f}** (95% bootstrap CI [{s_lo:.3f}, {s_hi:.3f}])")
    report.append(f"- DI Spearman: **{plan_b['overall_di']['spearman_r']:.3f}** (95% bootstrap CI [{d_lo:.3f}, {d_hi:.3f}])")
    report.append(f"- IR AUROC: **{plan_b['ir_auroc']['auroc']:.3f}**")
    report.append(f"- Decomp AUROC: **{plan_b['decomp_auroc']['auroc']:.3f}**")
    report.append(f"- Cross-meal ICC (SSPG pred): **{plan_b['icc_sspg_pred']:.3f}**")
    report.append(f"- si(z03) vs SSPG Spearman: **{plan_b['si_vs_sspg_spearman']:.3f}**")
    report.append(f"- mi(z05) vs DI Spearman: **{plan_b['mi_vs_di_spearman']:.3f}**")
    report.append("")
    report.append("## S1 Success Criteria Check")
    report.append("")
    report.append(f"- C1 Param alignment (si<-0.5 and mi>0.5): **{crit['C1_param_alignment']}**")
    report.append(f"- C2 Accuracy keep (SSPG>=0.70 and DI>=0.60): **{crit['C2_accuracy']}**")
    report.append(f"- C3 ICC uplift (SSPG ICC>0.5): **{crit['C3_icc']}**")
    report.append(f"- Overall pass: **{crit['all_pass']}**")
    report.append("")
    report.append("## Output Files")
    report.append("")
    report.append("- `S1_summary.json`")
    report.append("- `S1_planA_overall_table_from_v19.csv`")
    report.append("- `S1_planA_clf_table_from_v19.csv`")
    report.append("- `S1_planA_icc_table_from_v19.csv`")
    report.append("- `S1_planB_subject_true_pred.csv`")
    report.append("- `S1_planB_overall_sspg.csv`")
    report.append("- `S1_planB_overall_di.csv`")
    with open(os.path.join(OUT_ROOT, "S1_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"S1 done. Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
