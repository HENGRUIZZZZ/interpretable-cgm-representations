from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = "/Users/hertz1030/Documents/GitHub/interpretable-cgm-representations"
TRAIN_LAT = os.path.join(
    REPO_ROOT, "New_paper1_results_glucovector_v18", "v18_Exp8_CorrLoss", "phase2_finetune_head", "latent_and_gold_all_26d.csv"
)
D4_SUB = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v22_locked_protocol", "v22_d4_subject_level_predictions.csv")
D4_MEAL = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v22_locked_protocol", "v22_d4_meal_level_predictions.csv")
OUT_DIR = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v27_2_mechanism_anchor")
SEED = 42


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < 3:
        return {"n": int(len(y_true)), "spearman": np.nan, "pearson": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    return {
        "n": int(len(y_true)),
        "spearman": float(stats.spearmanr(y_true, y_pred)[0]),
        "pearson": float(stats.pearsonr(y_true, y_pred)[0]),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _normalize_latent_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "tau_m": "z00",
        "Gb": "z01",
        "sg": "z02",
        "si": "z03",
        "p2": "z04",
        "mi": "z05",
        "z_init_0": "z06",
        "z_init_1": "z07",
        "z_init_2": "z08",
        "z_init_3": "z09",
    }
    for k, v in rename_map.items():
        if k in out.columns and v not in out.columns:
            out[v] = out[k]
    for i in range(16):
        s = f"z_nonseq_{i}"
        d = f"z{10+i:02d}"
        if s in out.columns and d not in out.columns:
            out[d] = out[s]
    return out


def _prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(TRAIN_LAT)
    tr = _normalize_latent_columns(tr)
    if "SSPG" in tr.columns and "sspg" not in tr.columns:
        tr["sspg"] = tr["SSPG"]
    if "DI" in tr.columns and "di" not in tr.columns:
        tr["di"] = tr["DI"]
    z_cols = [f"z{i:02d}" for i in range(26)]
    tr = tr.dropna(subset=z_cols + ["subject_id", "sspg", "di"]).copy()
    tr_sub = tr.groupby("subject_id", as_index=False).median(numeric_only=True)

    d4_sub = pd.read_csv(D4_SUB)
    d4_sub = _normalize_latent_columns(d4_sub).dropna(subset=z_cols + ["subject_id"]).copy()

    d4_meal = pd.read_csv(D4_MEAL)
    d4_meal = _normalize_latent_columns(d4_meal).dropna(subset=z_cols + ["subject_id"]).copy()
    return tr_sub, d4_sub, d4_meal


@dataclass
class AnchoredModel:
    base10: object
    residual16: object
    gate_clf: object
    alpha: float


def _fit_anchored(
    X10: np.ndarray, X16: np.ndarray, y: np.ndarray, n_splits: int = 5
) -> Tuple[AnchoredModel, np.ndarray, np.ndarray, np.ndarray]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    n = len(y)
    oof_base = np.full(n, np.nan, dtype=float)
    oof_res = np.full(n, np.nan, dtype=float)
    oof_hard = np.full(n, np.nan, dtype=float)

    for tr, va in kf.split(X10):
        m10 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 80)))
        m10.fit(X10[tr], y[tr])
        p10 = m10.predict(X10[va])
        res = y[tr] - m10.predict(X10[tr])
        mr = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 80)))
        mr.fit(X16[tr], res)
        oof_base[va] = p10
        oof_res[va] = mr.predict(X16[va])

    abs_res = np.abs(y - oof_base)
    thr = np.nanquantile(abs_res, 0.75)
    hard = (abs_res >= thr).astype(int)
    g = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    g.fit(X16, hard)
    oof_hard = g.predict_proba(X16)[:, 1]

    # tune alpha on OOF
    best_alpha = 0.0
    best_mse = np.inf
    for a in np.linspace(0.0, 1.2, 49):
        p = oof_base + a * oof_hard * oof_res
        mse = np.mean((y - p) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(a)

    m10 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 80)))
    m10.fit(X10, y)
    res_full = y - m10.predict(X10)
    mr = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 80)))
    mr.fit(X16, res_full)

    model = AnchoredModel(base10=m10, residual16=mr, gate_clf=g, alpha=best_alpha)
    return model, oof_base, oof_res, oof_hard


def _predict_anchored(model: AnchoredModel, X10: np.ndarray, X16: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p10 = model.base10.predict(X10)
    r16 = model.residual16.predict(X16)
    gate = model.gate_clf.predict_proba(X16)[:, 1]
    p = p10 + model.alpha * gate * r16
    return p, p10, r16, gate


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tr_sub, d4_sub, d4_meal = _prepare_data()

    z10 = [f"z{i:02d}" for i in range(10)]
    z16 = [f"z{i:02d}" for i in range(10, 26)]
    z26 = [f"z{i:02d}" for i in range(26)]

    summary_rows = []
    pred_sub = d4_sub[["subject_id", "sspg_true", "di_true", "uncertainty_score"]].copy()
    meal_gate_rows = []
    config_rows = []

    for target in ["sspg", "di"]:
        ytr = tr_sub[target].to_numpy(float)
        x10_tr = tr_sub[z10].to_numpy(float)
        x16_tr = tr_sub[z16].to_numpy(float)
        x26_tr = tr_sub[z26].to_numpy(float)

        yte = d4_sub[f"{target}_true"].to_numpy(float)
        x10_te = d4_sub[z10].to_numpy(float)
        x16_te = d4_sub[z16].to_numpy(float)
        x26_te = d4_sub[z26].to_numpy(float)

        # Baselines
        m10 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 80)))
        m26 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 80)))
        m10.fit(x10_tr, ytr)
        m26.fit(x26_tr, ytr)
        p10 = m10.predict(x10_te)
        p26 = m26.predict(x26_te)

        # Anchored 10D + gated 16D residual
        am, oof_base, oof_res, oof_gate = _fit_anchored(x10_tr, x16_tr, ytr, n_splits=5)
        pa, p10a, r16a, gatea = _predict_anchored(am, x10_te, x16_te)

        for name, pred in [("Ridge10D", p10), ("Ridge26D", p26), ("Anchored10D_plusGated16D", pa)]:
            m = _metrics(yte, pred)
            summary_rows.append({"target": target, "model": name, **m})

        pred_sub[f"pred10_{target}"] = p10
        pred_sub[f"pred26_{target}"] = p26
        pred_sub[f"predA_{target}"] = pa
        pred_sub[f"gateA_{target}"] = gatea
        pred_sub[f"r16A_{target}"] = r16a

        config_rows.append(
            {
                "target": target,
                "alpha": am.alpha,
                "oof_hard_rate": float(np.mean((np.abs(ytr - oof_base) >= np.quantile(np.abs(ytr - oof_base), 0.75)).astype(float))),
                "oof_gate_mean": float(np.mean(oof_gate)),
                "oof_corr_absres_gate": float(stats.spearmanr(np.abs(ytr - oof_base), oof_gate)[0]),
            }
        )

    # Meal-level gate behavior: does gate activate more in hard/high-uncertainty windows?
    for target in ["sspg", "di"]:
        # fit on full train for projecting to meal-level
        ytr = tr_sub[target].to_numpy(float)
        x10_tr = tr_sub[z10].to_numpy(float)
        x16_tr = tr_sub[z16].to_numpy(float)
        am, *_ = _fit_anchored(x10_tr, x16_tr, ytr, n_splits=5)
        _, _, r16_meal, g16_meal = _predict_anchored(
            am, d4_meal[z10].to_numpy(float), d4_meal[z16].to_numpy(float)
        )
        tmp = d4_meal[["subject_id", "meal_type", "uncertainty_score"]].copy()
        tmp["target"] = target
        tmp["gate"] = g16_meal
        tmp["corr_mag"] = np.abs(am.alpha * g16_meal * r16_meal)
        tmp["unc_bin"] = pd.qcut(tmp["uncertainty_score"], q=3, labels=["low", "mid", "high"], duplicates="drop")
        agg = (
            tmp.groupby(["target", "unc_bin"], as_index=False)
            .agg(n=("gate", "size"), gate_mean=("gate", "mean"), correction_mag_mean=("corr_mag", "mean"))
        )
        meal_gate_rows.append(agg)

    summary = pd.DataFrame(summary_rows)
    meal_gate = pd.concat(meal_gate_rows, ignore_index=True)
    cfg = pd.DataFrame(config_rows)

    summary.to_csv(os.path.join(OUT_DIR, "v27_2_d4_subject_metrics.csv"), index=False)
    pred_sub.to_csv(os.path.join(OUT_DIR, "v27_2_d4_subject_predictions.csv"), index=False)
    meal_gate.to_csv(os.path.join(OUT_DIR, "v27_2_meal_gate_behavior.csv"), index=False)
    cfg.to_csv(os.path.join(OUT_DIR, "v27_2_model_config_and_oof_behavior.csv"), index=False)

    # concise "how to do" document
    howto = {
        "principle": "Mechanism-anchored prediction with conditional 16D correction",
        "equation": "y_hat = f10(z10) + alpha * gate(z16) * r16(z16)",
        "why": [
            "f10 keeps ODE axis as primary driver",
            "r16 models residual only (not full replacement)",
            "gate activates correction mainly in hard/high-uncertainty cases",
        ],
        "next_upgrade": [
            "replace gate(z16) with gate(z16, circadian, activity)",
            "add penalty forcing low correction in low-uncertainty bins",
            "jointly optimize SSPG and DI under shared gate",
        ],
    }
    with open(os.path.join(OUT_DIR, "v27_2_how_to_operationalize.json"), "w") as f:
        json.dump(howto, f, indent=2)

    with open(os.path.join(OUT_DIR, "v27_2_report.md"), "w", encoding="utf-8") as f:
        f.write("# v27.2 Mechanism-Anchored + Gated16D\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## D4 subject-level metrics\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n## Gate behavior across uncertainty bins (meal-level)\n\n")
        f.write(meal_gate.to_markdown(index=False))
        f.write("\n\n## Model config and OOF behavior\n\n")
        f.write(cfg.to_markdown(index=False))
        f.write("\n")

    print("Saved:", OUT_DIR)


if __name__ == "__main__":
    main()
