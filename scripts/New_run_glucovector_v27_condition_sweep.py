from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

REPO_ROOT = "/Users/hertz1030/Documents/GitHub/interpretable-cgm-representations"
V18_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v18")
V22_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v22_locked_protocol")
V20_D3_MERGED = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v20", "d3_deep_ablation", "d3_merged_GV_26D_Exp8.csv")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v27_condition_sweep")
SEED = 42


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
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


def _prepare_train_test() -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase2_finetune_head", "latent_and_gold_all_26d.csv"))
    if "SSPG" in tr.columns and "sspg" not in tr.columns:
        tr["sspg"] = tr["SSPG"]
    if "DI" in tr.columns and "di" not in tr.columns:
        tr["di"] = tr["DI"]
    rename_map = {
        "tau_m": "z00", "Gb": "z01", "sg": "z02", "si": "z03", "p2": "z04", "mi": "z05",
        "z_init_0": "z06", "z_init_1": "z07", "z_init_2": "z08", "z_init_3": "z09",
    }
    for k, v in rename_map.items():
        if k in tr.columns and v not in tr.columns:
            tr[v] = tr[k]
    for i in range(16):
        s, d = f"z_nonseq_{i}", f"z{10+i:02d}"
        if s in tr.columns and d not in tr.columns:
            tr[d] = tr[s]
    tr = tr.dropna(subset=[f"z{i:02d}" for i in range(26)] + ["sspg", "di"]).copy()
    tr_sub = tr.groupby("subject_id", as_index=False).median(numeric_only=True)

    te = pd.read_csv(os.path.join(V22_ROOT, "v22_d4_subject_level_predictions.csv"))
    return tr_sub, te


def _reg_candidates():
    return {
        "RidgeCV": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))),
        "RidgeCV_YJTarget": lambda: TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))),
            transformer=PowerTransformer(method="yeo-johnson", standardize=True),
        ),
        "GBR_small": lambda: GradientBoostingRegressor(n_estimators=160, learning_rate=0.05, max_depth=2, random_state=SEED),
        "GBR_medium": lambda: GradientBoostingRegressor(n_estimators=260, learning_rate=0.03, max_depth=3, random_state=SEED),
        "RF_mid": lambda: RandomForestRegressor(n_estimators=220, max_depth=6, random_state=SEED, n_jobs=-1),
    }


def _loo_cv_score(X: np.ndarray, y: np.ndarray, builder) -> Dict[str, float]:
    loo = LeaveOneOut()
    pred = np.full(len(y), np.nan, dtype=float)
    for tr, te in loo.split(X):
        mdl = builder()
        mdl.fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te])
    return _safe_metrics(y, pred)


def _condition_sweep(tr_sub: pd.DataFrame, te_sub: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    z10 = [f"z{i:02d}" for i in range(10)]
    z16 = [f"z{i:02d}" for i in range(10, 26)]
    z26 = [f"z{i:02d}" for i in range(26)]
    sets = {"10D": z10, "16D": z16, "26D": z26}
    cands = _reg_candidates()

    rows = []
    pred_rows = []
    for feat_name, feats in sets.items():
        Xtr = tr_sub[feats].to_numpy(float)
        Xte = te_sub[feats].to_numpy(float)
        for target, ycol, true_col in [("sspg", "sspg", "sspg_true"), ("di", "di", "di_true")]:
            ytr = tr_sub[ycol].to_numpy(float)
            for model_name, builder in cands.items():
                cvm = _loo_cv_score(Xtr, ytr, builder)
                mdl = builder()
                mdl.fit(Xtr, ytr)
                pte = mdl.predict(Xte)
                tm = _safe_metrics(te_sub[true_col].to_numpy(float), pte)
                rows.append(
                    {
                        "feature_set": feat_name,
                        "target": target,
                        "model": model_name,
                        "cv_spearman": cvm["spearman"],
                        "cv_r2": cvm["r2"],
                        "cv_rmse": cvm["rmse"],
                        "test_spearman": tm["spearman"],
                        "test_r2": tm["r2"],
                        "test_rmse": tm["rmse"],
                        "test_mae": tm["mae"],
                    }
                )
                pred_rows.append(
                    te_sub[["subject_id", true_col]].assign(
                        feature_set=feat_name, target=target, model=model_name, pred=pte
                    )
                )
    res = pd.DataFrame(rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    return res, preds


def _select_best_models(sweep: pd.DataFrame) -> pd.DataFrame:
    # locked selection by cv_spearman, then cv_r2
    best = (
        sweep.sort_values(["target", "feature_set", "cv_spearman", "cv_r2"], ascending=[True, True, False, False])
        .groupby(["target", "feature_set"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best


def _classification_from_preds(preds: pd.DataFrame, best: pd.DataFrame, te_sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in ["10D", "16D", "26D"]:
        bs = best[(best["target"] == "sspg") & (best["feature_set"] == feat)].iloc[0]
        bd = best[(best["target"] == "di") & (best["feature_set"] == feat)].iloc[0]
        ps = preds[(preds["feature_set"] == feat) & (preds["target"] == "sspg") & (preds["model"] == bs["model"])].rename(columns={"pred": "sspg_pred", "sspg_true": "sspg_true"})
        pd_ = preds[(preds["feature_set"] == feat) & (preds["target"] == "di") & (preds["model"] == bd["model"])].rename(columns={"pred": "di_pred", "di_true": "di_true"})
        m = te_sub[["subject_id", "sspg_true", "di_true"]].merge(ps[["subject_id", "sspg_pred"]], on="subject_id", how="left").merge(pd_[["subject_id", "di_pred"]], on="subject_id", how="left")
        y_ir = (m["sspg_true"].to_numpy(float) >= 120).astype(int)
        y_de = ((m["sspg_true"].to_numpy(float) >= 120) & (m["di_true"].to_numpy(float) < 1.0)).astype(int)
        s_ir = m["sspg_pred"].to_numpy(float)
        s_de = stats.zscore(m["sspg_pred"].to_numpy(float), nan_policy="omit") - stats.zscore(m["di_pred"].to_numpy(float), nan_policy="omit")
        ir_auc = float(roc_auc_score(y_ir, s_ir)) if len(np.unique(y_ir)) > 1 else np.nan
        de_auc = float(roc_auc_score(y_de, s_de)) if len(np.unique(y_de)) > 1 else np.nan
        ir_hat = (s_ir >= np.nanmedian(s_ir)).astype(int)
        de_hat = (s_de >= np.nanpercentile(s_de, 65)).astype(int)
        rows.append(
            {
                "feature_set": feat,
                "best_model_sspg": bs["model"],
                "best_model_di": bd["model"],
                "ir_auc": ir_auc,
                "ir_acc": float(accuracy_score(y_ir, ir_hat)),
                "ir_f1": float(f1_score(y_ir, ir_hat, zero_division=0)),
                "decomp_auc": de_auc,
                "decomp_acc": float(accuracy_score(y_de, de_hat)),
                "decomp_f1": float(f1_score(y_de, de_hat, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def _ibt_like_error_profile(preds: pd.DataFrame, best: pd.DataFrame, te_sub: pd.DataFrame) -> pd.DataFrame:
    # interval-bias tendency: signed residual by true quantiles
    rows = []
    for feat in ["10D", "16D", "26D"]:
        for target, true_col in [("sspg", "sspg_true"), ("di", "di_true")]:
            bm = best[(best["target"] == target) & (best["feature_set"] == feat)].iloc[0]
            p = preds[(preds["feature_set"] == feat) & (preds["target"] == target) & (preds["model"] == bm["model"])]
            d = te_sub[["subject_id", true_col]].merge(p[["subject_id", "pred"]], on="subject_id", how="left")
            d["res"] = d[true_col] - d["pred"]
            d["bin"] = pd.qcut(d[true_col], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
            for b, g in d.groupby("bin"):
                rows.append(
                    {
                        "feature_set": feat,
                        "target": target,
                        "bin": str(b),
                        "n": int(len(g)),
                        "mean_signed_residual": float(g["res"].mean()),
                        "mean_abs_error": float(np.abs(g["res"]).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _circadian_extension() -> pd.DataFrame:
    d3 = pd.read_csv(V20_D3_MERGED)
    if "timestamp" in d3.columns:
        ts = pd.to_datetime(d3["timestamp"], errors="coerce")
        hour = ts.dt.hour.fillna(12).to_numpy(float)
    else:
        hour = np.full(len(d3), 12.0)
    d3["c_sin"] = np.sin(2 * np.pi * hour / 24.0)
    d3["c_cos"] = np.cos(2 * np.pi * hour / 24.0)
    d3["is_night"] = ((hour <= 6) | (hour >= 22)).astype(int)
    d3 = d3.dropna(subset=["subject_id", "meal_type"]).copy()
    z16 = [f"z{i:02d}" for i in range(10, 26)]
    if not all(c in d3.columns for c in z16):
        z16 = [f"z{i:02d}" for i in range(10, 26) if f"z{i:02d}" in d3.columns]

    X16 = d3[z16].to_numpy(float)
    X16c = np.c_[X16, d3[["c_sin", "c_cos", "is_night"]].to_numpy(float)]
    y_meal = d3["meal_type"].astype(str).to_numpy()
    groups = d3["subject_id"].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=5)
    rows = []
    for name, X in [("16D", X16), ("16D_plus_circadian", X16c)]:
        pred = np.array([""] * len(y_meal), dtype=object)
        for tr, te in gkf.split(X, y_meal, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, multi_class="multinomial"))
            clf.fit(X[tr], y_meal[tr])
            pred[te] = clf.predict(X[te])
        rows.append(
            {
                "task": "meal_type_classification",
                "feature_set": name,
                "accuracy": float(accuracy_score(y_meal, pred)),
                "macro_f1": float(f1_score(y_meal, pred, average="macro")),
                "n": int(len(y_meal)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    os.makedirs(OUT_ROOT, exist_ok=True)
    tr_sub, te_sub = _prepare_train_test()
    sweep, preds = _condition_sweep(tr_sub, te_sub)
    best = _select_best_models(sweep)
    clf = _classification_from_preds(preds, best, te_sub)
    ibt = _ibt_like_error_profile(preds, best, te_sub)
    circ = _circadian_extension()

    sweep.to_csv(os.path.join(OUT_ROOT, "v27_condition_sweep_results.csv"), index=False)
    preds.to_csv(os.path.join(OUT_ROOT, "v27_condition_sweep_predictions.csv"), index=False)
    best.to_csv(os.path.join(OUT_ROOT, "v27_best_models_by_feature_set.csv"), index=False)
    clf.to_csv(os.path.join(OUT_ROOT, "v27_classification_comparison_10d_16d_26d.csv"), index=False)
    ibt.to_csv(os.path.join(OUT_ROOT, "v27_ibt_like_error_profile.csv"), index=False)
    circ.to_csv(os.path.join(OUT_ROOT, "v27_circadian_extension_results.csv"), index=False)

    summary = {
        "best_sspg_models": best[best["target"] == "sspg"][["feature_set", "model", "test_r2", "test_spearman"]].to_dict(orient="records"),
        "best_di_models": best[best["target"] == "di"][["feature_set", "model", "test_r2", "test_spearman"]].to_dict(orient="records"),
    }
    with open(os.path.join(OUT_ROOT, "v27_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUT_ROOT, "v27_report.md"), "w", encoding="utf-8") as f:
        f.write("# v27 Condition Sweep + Framework Extension Tests\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Best models by feature set and target\n\n")
        f.write(best.to_markdown(index=False))
        f.write("\n\n## Classification comparison (IR/Decomp)\n\n")
        f.write(clf.to_markdown(index=False))
        f.write("\n\n## IBT-like error profile\n\n")
        f.write(ibt.to_markdown(index=False))
        f.write("\n\n## Circadian extension feasibility\n\n")
        f.write(circ.to_markdown(index=False))
        f.write("\n")

    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    main()
