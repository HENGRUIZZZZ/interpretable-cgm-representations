from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = "/Users/hertz1030/Documents/GitHub/interpretable-cgm-representations"
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v21_optimization", "diagnostics")
V18_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v18")
V21_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v21_optimization")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    if len(y) < 3:
        return {"n": int(len(y)), "spearman": np.nan, "pearson": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    return {
        "n": int(len(y)),
        "spearman": float(stats.spearmanr(y, p)[0]),
        "pearson": float(stats.pearsonr(y, p)[0]),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
    }


def _prepare_train_latent() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase2_finetune_head", "latent_and_gold_all_26d.csv"))
    rename_map = {
        "tau_m": "z00", "Gb": "z01", "sg": "z02", "si": "z03", "p2": "z04", "mi": "z05",
        "z_init_0": "z06", "z_init_1": "z07", "z_init_2": "z08", "z_init_3": "z09",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    for i in range(16):
        s = f"z_nonseq_{i}"
        d = f"z{10+i:02d}"
        if s in df.columns and d not in df.columns:
            df[d] = df[s]
    if "SSPG" in df.columns and "sspg" not in df.columns:
        df["sspg"] = df["SSPG"]
    if "DI" in df.columns and "di" not in df.columns:
        df["di"] = df["DI"]
    return df


def _main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    tr = _prepare_train_latent().dropna(subset=[f"z{i:02d}" for i in range(26)] + ["sspg", "di"]).copy()
    z10 = [f"z{i:02d}" for i in range(10)]
    z26 = [f"z{i:02d}" for i in range(26)]
    X10 = tr[z10].to_numpy(float)
    X26 = tr[z26].to_numpy(float)
    y_s = tr["sspg"].to_numpy(float)
    y_d = tr["di"].to_numpy(float)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = pd.DataFrame({"y_sspg": y_s, "y_di": y_d})
    oof_feats_s = np.zeros((len(tr), 3), dtype=float)
    oof_feats_d = np.zeros((len(tr), 3), dtype=float)
    oof["p10_sspg"] = np.nan
    oof["p26_sspg"] = np.nan
    oof["p10_di"] = np.nan
    oof["p26_di"] = np.nan

    for tr_idx, va_idx in kf.split(X26):
        s10 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))).fit(X10[tr_idx], y_s[tr_idx])
        s26 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))).fit(X26[tr_idx], y_s[tr_idx])
        d10 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))).fit(X10[tr_idx], y_d[tr_idx])
        d26 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))).fit(X26[tr_idx], y_d[tr_idx])
        p10s = s10.predict(X10[va_idx]); p26s = s26.predict(X26[va_idx])
        p10d = d10.predict(X10[va_idx]); p26d = d26.predict(X26[va_idx])
        oof.loc[va_idx, "p10_sspg"] = p10s
        oof.loc[va_idx, "p26_sspg"] = p26s
        oof.loc[va_idx, "p10_di"] = p10d
        oof.loc[va_idx, "p26_di"] = p26d
        oof_feats_s[va_idx] = np.c_[p10s, p26s, np.abs(p10s - p26s)]
        oof_feats_d[va_idx] = np.c_[p10d, p26d, np.abs(p10d - p26d)]

    ms = GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=2, random_state=42).fit(oof_feats_s, y_s)
    md = GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=2, random_state=42).fit(oof_feats_d, y_d)
    oof["pstack_sspg"] = ms.predict(oof_feats_s)
    oof["pstack_di"] = md.predict(oof_feats_d)
    oof.to_csv(os.path.join(OUT_ROOT, "train_oof_predictions.csv"), index=False)

    # D4 predictions from previously generated file
    te = pd.read_csv(os.path.join(V21_ROOT, "v21_opt_predictions_subject_level.csv"))
    # model-wise metrics
    rows: List[Dict[str, object]] = []
    rows.extend(
        [
            {"split": "train_oof", "target": "sspg", "model": "Ridge10D", **_metrics(oof["y_sspg"], oof["p10_sspg"])},
            {"split": "train_oof", "target": "sspg", "model": "Ridge26D", **_metrics(oof["y_sspg"], oof["p26_sspg"])},
            {"split": "train_oof", "target": "sspg", "model": "StackGated10D26D", **_metrics(oof["y_sspg"], oof["pstack_sspg"])},
            {"split": "train_oof", "target": "di", "model": "Ridge10D", **_metrics(oof["y_di"], oof["p10_di"])},
            {"split": "train_oof", "target": "di", "model": "Ridge26D", **_metrics(oof["y_di"], oof["p26_di"])},
            {"split": "train_oof", "target": "di", "model": "StackGated10D26D", **_metrics(oof["y_di"], oof["pstack_di"])},
        ]
    )

    for m in te["model"].unique():
        dm = te[te["model"] == m]
        rows.append({"split": "test_D4", "target": "sspg", "model": m, **_metrics(dm["sspg_true"], dm["sspg_pred"])})
        rows.append({"split": "test_D4", "target": "di", "model": m, **_metrics(dm["di_true"], dm["di_pred"])})
    perf = pd.DataFrame(rows)
    perf.to_csv(os.path.join(OUT_ROOT, "stack_train_vs_test_metrics.csv"), index=False)

    # shift on meta features between train OOF and test
    test_p10 = te[te["model"] == "Ridge10D"][["subject_id", "sspg_pred", "di_pred"]].rename(columns={"sspg_pred": "p10_sspg_t", "di_pred": "p10_di_t"})
    test_p26 = te[te["model"] == "Ridge26D"][["subject_id", "sspg_pred", "di_pred"]].rename(columns={"sspg_pred": "p26_sspg_t", "di_pred": "p26_di_t"})
    test_st = te[te["model"] == "StackGated10D26D"][["subject_id", "sspg_pred", "di_pred"]].rename(columns={"sspg_pred": "pst_sspg_t", "di_pred": "pst_di_t"})
    t = test_p10.merge(test_p26, on="subject_id", how="inner").merge(test_st, on="subject_id", how="inner")
    t["absdiff_sspg_t"] = np.abs(t["p10_sspg_t"] - t["p26_sspg_t"])
    t["absdiff_di_t"] = np.abs(t["p10_di_t"] - t["p26_di_t"])

    shift_rows = []
    for tr_col, te_col in [
        ("p10_sspg", "p10_sspg_t"),
        ("p26_sspg", "p26_sspg_t"),
        ("pstack_sspg", "pst_sspg_t"),
        ("p10_di", "p10_di_t"),
        ("p26_di", "p26_di_t"),
        ("pstack_di", "pst_di_t"),
    ]:
        a = oof[tr_col].to_numpy(float)
        b = t[te_col].to_numpy(float)
        ks = stats.ks_2samp(a[np.isfinite(a)], b[np.isfinite(b)])
        shift_rows.append(
            {
                "feature": tr_col,
                "train_mean": float(np.nanmean(a)),
                "test_mean": float(np.nanmean(b)),
                "train_std": float(np.nanstd(a)),
                "test_std": float(np.nanstd(b)),
                "ks_stat": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv(os.path.join(OUT_ROOT, "distribution_shift_train_oof_vs_d4.csv"), index=False)

    feat_imp = pd.DataFrame(
        {
            "meta_feature": ["p10", "p26", "absdiff"],
            "sspg_importance": ms.feature_importances_,
            "di_importance": md.feature_importances_,
        }
    )
    feat_imp.to_csv(os.path.join(OUT_ROOT, "meta_feature_importance_gbr.csv"), index=False)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "note": "If stack beats train_oof but fails on test_D4 while base ridge models remain stable, this supports overfitting/meta-shift hypothesis.",
    }
    with open(os.path.join(OUT_ROOT, "stack_diagnostic_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUT_ROOT, "stack_diagnostic_report.md"), "w", encoding="utf-8") as f:
        f.write("# v21 Stack Diagnostic Report\n\n")
        f.write(f"Generated: {summary['generated_at']}\n\n")
        f.write("## Train OOF vs Test D4 Performance\n\n")
        f.write(perf.to_markdown(index=False))
        f.write("\n\n## Meta Feature Importance (GBR)\n\n")
        f.write(feat_imp.to_markdown(index=False))
        f.write("\n\n## Distribution Shift (Train OOF vs Test D4)\n\n")
        f.write(shift_df.to_markdown(index=False))
        f.write("\n")
    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    _main()
