from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = "/Users/hertz1030/Documents/GitHub/interpretable-cgm-representations"
V18_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v18")
V22_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v22_locked_protocol")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v26_residual_correction")
SEED = 42


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    if len(y) < 3:
        return {"n": int(len(y)), "spearman": np.nan, "pearson": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    return {
        "n": int(len(y)),
        "spearman": float(stats.spearmanr(y, p)[0]),
        "pearson": float(stats.pearsonr(y, p)[0]),
        "r2": float(r2_score(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
    }


def _bootstrap_ci(y: np.ndarray, p: np.ndarray, metric: str, n_boot: int = 3000, seed: int = 42) -> Tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)

    def calc(yy, pp):
        if metric == "r2":
            return float(r2_score(yy, pp))
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(yy, pp)))
        if metric == "spearman":
            return float(stats.spearmanr(yy, pp)[0])
        raise ValueError(metric)

    point = calc(y, p)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        vals.append(calc(y[b], p[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5]).tolist()
    return point, float(lo), float(hi)


def _prepare_train() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase2_finetune_head", "latent_and_gold_all_26d.csv"))
    if "SSPG" in df.columns and "sspg" not in df.columns:
        df["sspg"] = df["SSPG"]
    if "DI" in df.columns and "di" not in df.columns:
        df["di"] = df["DI"]
    rename_map = {
        "tau_m": "z00", "Gb": "z01", "sg": "z02", "si": "z03", "p2": "z04", "mi": "z05",
        "z_init_0": "z06", "z_init_1": "z07", "z_init_2": "z08", "z_init_3": "z09",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    for i in range(16):
        s, d = f"z_nonseq_{i}", f"z{10+i:02d}"
        if s in df.columns and d not in df.columns:
            df[d] = df[s]
    df = df.dropna(subset=[f"z{i:02d}" for i in range(26)] + ["sspg", "di"]).copy()
    return df.groupby("subject_id", as_index=False).median(numeric_only=True)


def _prepare_d4() -> pd.DataFrame:
    meal = pd.read_csv(os.path.join(V22_ROOT, "v22_d4_meal_level_predictions.csv"))
    zcols = [f"z{i:02d}" for i in range(26)]
    keep = ["subject_id", "sspg_true", "di_true", "uncertainty_score", "carb_g", "fat_g", "protein_g", "fiber_g"] + zcols
    sub = meal[keep].groupby("subject_id", as_index=False).mean(numeric_only=True)
    return sub


def _fit_base_and_correction(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    z10 = [f"z{i:02d}" for i in range(10)]
    z16 = [f"z{i:02d}" for i in range(10, 26)]
    z26 = [f"z{i:02d}" for i in range(26)]

    Xtr26 = train[z26].to_numpy(float)
    Xte26 = test[z26].to_numpy(float)
    Xtr16 = train[z16].to_numpy(float)
    Xte16 = test[z16].to_numpy(float)
    ytr_s = train["sspg"].to_numpy(float)
    ytr_d = train["di"].to_numpy(float)

    base_s = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(Xtr26, ytr_s)
    base_d = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(Xtr26, ytr_d)

    # OOF base predictions for calibration/residual models
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_s = np.full(len(train), np.nan, dtype=float)
    oof_d = np.full(len(train), np.nan, dtype=float)
    for tr_idx, va_idx in kf.split(Xtr26):
        m_s = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))).fit(Xtr26[tr_idx], ytr_s[tr_idx])
        m_d = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))).fit(Xtr26[tr_idx], ytr_d[tr_idx])
        oof_s[va_idx] = m_s.predict(Xtr26[va_idx])
        oof_d[va_idx] = m_d.predict(Xtr26[va_idx])

    # global linear calibration true = a + b*pred
    reg_s = stats.linregress(oof_s, ytr_s)
    reg_d = stats.linregress(oof_d, ytr_d)

    # residual model on 16D (learn correction not captured by base)
    res_s = ytr_s - (reg_s.intercept + reg_s.slope * oof_s)
    res_d = ytr_d - (reg_d.intercept + reg_d.slope * oof_d)
    cor_s = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(Xtr16, res_s)
    cor_d = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(Xtr16, res_d)

    p_base_s = base_s.predict(Xte26)
    p_base_d = base_d.predict(Xte26)

    p_cal_s = reg_s.intercept + reg_s.slope * p_base_s
    p_cal_d = reg_d.intercept + reg_d.slope * p_base_d

    p_corr_s = p_cal_s + cor_s.predict(Xte16)
    p_corr_d = p_cal_d + cor_d.predict(Xte16)

    out = {}
    out["Base26D"] = test[["subject_id", "sspg_true", "di_true"]].assign(sspg_pred=p_base_s, di_pred=p_base_d)
    out["Calibrated26D"] = test[["subject_id", "sspg_true", "di_true"]].assign(sspg_pred=p_cal_s, di_pred=p_cal_d)
    out["Calibrated26D_plus16DResidual"] = test[["subject_id", "sspg_true", "di_true"]].assign(sspg_pred=p_corr_s, di_pred=p_corr_d)
    return out


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    tr = _prepare_train()
    te = _prepare_d4()
    preds = _fit_base_and_correction(tr, te)

    # evaluate
    rows = []
    ci_rows = []
    pred_all = []
    for model, df in preds.items():
        ms = _metrics(df["sspg_true"], df["sspg_pred"])
        md = _metrics(df["di_true"], df["di_pred"])
        rows.append({"model": model, **{f"sspg_{k}": v for k, v in ms.items()}, **{f"di_{k}": v for k, v in md.items()}})
        for tgt, y, p in [("sspg", df["sspg_true"].to_numpy(float), df["sspg_pred"].to_numpy(float)), ("di", df["di_true"].to_numpy(float), df["di_pred"].to_numpy(float))]:
            for m in ["spearman", "r2", "rmse"]:
                pt, lo, hi = _bootstrap_ci(y, p, m, n_boot=2500, seed=SEED + (1 if tgt == "di" else 0))
                ci_rows.append({"model": model, "target": tgt, "metric": m, "point": pt, "ci_lo": lo, "ci_hi": hi})

        # clinical AUROC
        y_ir = (df["sspg_true"].to_numpy(float) >= 120).astype(int)
        y_de = ((df["sspg_true"].to_numpy(float) >= 120) & (df["di_true"].to_numpy(float) < 1.0)).astype(int)
        s_ir = df["sspg_pred"].to_numpy(float)
        s_de = stats.zscore(df["sspg_pred"].to_numpy(float), nan_policy="omit") - stats.zscore(df["di_pred"].to_numpy(float), nan_policy="omit")
        rows[-1]["ir_auc"] = float(roc_auc_score(y_ir, s_ir)) if len(np.unique(y_ir)) > 1 else np.nan
        rows[-1]["decomp_auc"] = float(roc_auc_score(y_de, s_de)) if len(np.unique(y_de)) > 1 else np.nan
        pred_all.append(df.assign(model=model))

    metrics_df = pd.DataFrame(rows)
    ci_df = pd.DataFrame(ci_rows)
    pred_df = pd.concat(pred_all, ignore_index=True)
    metrics_df.to_csv(os.path.join(OUT_ROOT, "v26_metrics_summary.csv"), index=False)
    ci_df.to_csv(os.path.join(OUT_ROOT, "v26_bootstrap_ci.csv"), index=False)
    pred_df.to_csv(os.path.join(OUT_ROOT, "v26_subject_predictions.csv"), index=False)

    # framework articulation block: what each block contributes
    contribution = pd.DataFrame(
        [
            {"block": "10D_mechanism_axis", "purpose": "stable metabolic baseline"},
            {"block": "16D_context_axis", "purpose": "meal/context-driven correction"},
            {"block": "calibration", "purpose": "decompress prediction range"},
            {"block": "residual_correction", "purpose": "fix systematic tail/context errors"},
        ]
    )
    contribution.to_csv(os.path.join(OUT_ROOT, "v26_framework_blocks.csv"), index=False)

    with open(os.path.join(OUT_ROOT, "v26_report.md"), "w", encoding="utf-8") as f:
        f.write("# v26 Calibration + 16D Residual Correction\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Metrics Summary\n\n")
        f.write(metrics_df.to_markdown(index=False))
        f.write("\n\n## Bootstrap CIs\n\n")
        f.write(ci_df.to_markdown(index=False))
        f.write("\n\n## Framework Blocks\n\n")
        f.write(contribution.to_markdown(index=False))
        f.write("\n")

    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    main()
