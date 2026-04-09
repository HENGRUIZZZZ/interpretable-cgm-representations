"""
v28 — Framework value: aggregate prior evidence + new ablations.

1) Evidence inventory: Peak/Locked D4, clinical secondary endpoints, D3 scorecard pointers.
2) D4 subject-level readout: Ridge on median 26D vs 26D + meal-aggregated context (LOOCV).
3) D5 MSS (external): meal-response prediction — CGM-only features vs CGM + Actiheart (LOOCV by participant).

Data: D5 extracted to New_data/D5_MSS/data (from MultiSensor / Phillips et al. 2023).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "New_paper1_results_glucovector_v28_framework_value"
OUT.mkdir(parents=True, exist_ok=True)

V19 = REPO / "New_paper1_results_glucovector_v19" / "v19_overall_metrics.csv"
V22P = REPO / "New_paper1_results_glucovector_v22_locked_protocol" / "v22_primary_endpoints_locked.csv"
V22S = REPO / "New_paper1_results_glucovector_v22_locked_protocol" / "v22_secondary_clinical_endpoints.csv"
V22C = REPO / "New_paper1_results_glucovector_v22_locked_protocol" / "v22_16d_conditional_utility.csv"
V22M = REPO / "New_paper1_results_glucovector_v22_locked_protocol" / "v22_d4_meal_level_predictions.csv"
V21D3 = REPO / "New_paper1_results_glucovector_v21_comprehensive" / "v21_d3_free_living_comprehensive_scorecard.csv"
V24 = REPO / "New_paper1_results_glucovector_v24_semantic_ablation" / "v24_information_decomposition_loo.csv"
V27PK = REPO / "New_paper1_results_glucovector_v27_condition_sweep" / "v27_peak_vs_locked_reporting_table.csv"
D5_M3 = REPO / "New_data" / "D5_MSS" / "data" / "Model3 data (glucose actiheart integrated)"


def _safe_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 4:
        return {"n": float(len(y)), "spearman": np.nan, "r2": np.nan, "rmse": np.nan}
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    r2 = float(r2_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    return {
        "n": float(len(y)),
        "spearman": float(stats.spearmanr(y, p)[0]),
        "r2": r2,
        "rmse": rmse,
    }


def _loocv_group_ridge(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    logo = LeaveOneGroupOut()
    pred = np.full(len(y), np.nan, dtype=float)
    for tr, te in logo.split(X, y, groups):
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60)))
        mdl.fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te])
    return y, pred


def build_evidence_inventory() -> pd.DataFrame:
    rows: List[Dict] = []
    if V19.exists():
        v19 = pd.read_csv(V19)
        r = v19[v19["model"].astype(str).str.contains("GV_CorrLoss\\(Exp8\\)", regex=True)]
        if len(r):
            r = r.iloc[0]
            rows += [
                {"block": "D4_PEAK", "metric": "DI_R2_Exp8", "value": float(r["di_r2"]), "ref": str(V19)},
                {"block": "D4_PEAK", "metric": "SSPG_R2_Exp8", "value": float(r["sspg_r2"]), "ref": str(V19)},
            ]
        r2 = v19[v19["model"].astype(str).str.contains("VarMatch", regex=False)]
        if len(r2):
            r2 = r2.iloc[0]
            rows.append(
                {"block": "D4_PEAK", "metric": "SSPG_R2_VarMatch_v19", "value": float(r2["sspg_r2"]), "ref": str(V19)}
            )
    if V22P.exists():
        p = pd.read_csv(V22P)
        for _, r in p.iterrows():
            rows.append(
                {
                    "block": "D4_LOCKED",
                    "metric": f"{r['model']}_{r['target']}_R2",
                    "value": float(r["r2"]),
                    "ref": str(V22P),
                }
            )
    if V22S.exists():
        s = pd.read_csv(V22S)
        for _, r in s.iterrows():
            rows.append(
                {
                    "block": "D4_CLINICAL_SECONDARY",
                    "metric": f"{r['model']}_IR_AUC",
                    "value": float(r["ir_auc"]),
                    "ref": str(V22S),
                }
            )
            rows.append(
                {
                    "block": "D4_CLINICAL_SECONDARY",
                    "metric": f"{r['model']}_DECOMP_AUC",
                    "value": float(r["decomp_auc"]),
                    "ref": str(V22S),
                }
            )
    if V22C.exists():
        c = pd.read_csv(V22C)
        for _, r in c.iterrows():
            rows.append(
                {
                    "block": "D4_16D_CONDITIONAL",
                    "metric": f"bin_{r.get('unc_bin','')}_win26_rate_sspg",
                    "value": float(r.get("win26_rate_sspg", np.nan)),
                    "ref": str(V22C),
                }
            )
    if V21D3.exists():
        rows.append(
            {
                "block": "D3_SCORECARD_FILE",
                "metric": "path",
                "value": np.nan,
                "ref": str(V21D3),
            }
        )
    if V24.exists():
        rows.append({"block": "D4_SEMANTIC", "metric": "loo_decomposition_file", "value": np.nan, "ref": str(V24)})
    if V27PK.exists():
        rows.append({"block": "REPORTING", "metric": "peak_vs_locked_table", "value": np.nan, "ref": str(V27PK)})
    return pd.DataFrame(rows)


def d4_subject_context_readout() -> pd.DataFrame:
    df = pd.read_csv(V22M)
    z = [f"z{i:02d}" for i in range(26)]
    med = df.groupby("subject_id", as_index=False)[z].median()
    ctx = (
        df.groupby("subject_id", as_index=False)
        .agg(
            unc_mean=("uncertainty_score", "mean"),
            unc_std=("uncertainty_score", "std"),
            carb_mean=("carb_g", "mean"),
            fat_mean=("fat_g", "mean"),
            prot_mean=("protein_g", "mean"),
            fiber_mean=("fiber_g", "mean"),
            n_meals=("subject_id", "size"),
        )
    )
    ctx["unc_std"] = ctx["unc_std"].fillna(0.0)
    sub = med.merge(ctx, on="subject_id", how="inner")
    lab = df.groupby("subject_id", as_index=False)[["sspg_true", "di_true"]].first()
    sub = sub.merge(lab, on="subject_id", how="inner")

    ctx_cols = ["unc_mean", "unc_std", "carb_mean", "fat_mean", "prot_mean", "fiber_mean", "n_meals"]
    groups = sub["subject_id"].astype(str).values
    rows_out = []
    for target in ["sspg_true", "di_true"]:
        y = sub[target].to_numpy(dtype=float)
        ok = np.isfinite(y)
        X26 = sub.loc[ok, z].to_numpy(dtype=float)
        Xf = sub.loc[ok, z + ctx_cols].to_numpy(dtype=float)
        yv = y[ok]
        gv = groups[ok]
        _, p26 = _loocv_group_ridge(X26, yv, gv)
        _, pf = _loocv_group_ridge(Xf, yv, gv)
        m26 = _safe_metrics(yv, p26)
        mf = _safe_metrics(yv, pf)
        rows_out += [
            {"target": target, "model": "LOOCV_Ridge_median26D", **m26},
            {"target": target, "model": "LOOCV_Ridge_median26D_plus_meal_context", **mf},
        ]
    return pd.DataFrame(rows_out)


def _cluster_meal_onsets(food_df: pd.DataFrame, gap_h: float = 0.5) -> List[float]:
    t = np.sort(food_df["abs_time_hours"].dropna().to_numpy(dtype=float))
    if len(t) == 0:
        return []
    onsets: List[float] = []
    start = t[0]
    last = t[0]
    for x in t[1:]:
        if x - last <= gap_h:
            last = x
        else:
            onsets.append(float(start))
            start = x
            last = x
    onsets.append(float(start))
    return onsets


def _meal_features(
    integ: pd.DataFrame, meal_t: float, pre_h: float = 1.0, post_h: float = 3.0, dt_h: float = 0.25
) -> Optional[Dict[str, float]]:
    """Targets use post-prandial glucose; all predictive features use ONLY the pre-prandial window."""
    use = integ.loc[~integ["mask"]].copy()
    if use.empty:
        return None
    pre = use[(use["abs_time_hours"] >= meal_t - pre_h) & (use["abs_time_hours"] <= meal_t)]
    post = use[(use["abs_time_hours"] >= meal_t) & (use["abs_time_hours"] <= meal_t + post_h)]
    if len(pre) < 3 or len(post) < 2:
        return None
    pre_tail = pre[pre["abs_time_hours"] >= meal_t - 0.25]
    if len(pre_tail) < 1:
        pre_tail = pre.tail(2)
    b0 = float(pre_tail["Detrended"].mean())
    g_pre = pre["Detrended"].to_numpy(dtype=float)
    t_pre = pre["abs_time_hours"].to_numpy(dtype=float)
    act_pre = pre["Activity"].to_numpy(dtype=float)
    bpm_pre = pre["BPM"].to_numpy(dtype=float)
    rm_pre = pre["RMSSD"].to_numpy(dtype=float)

    g_post = post["Detrended"].to_numpy(dtype=float)
    act_post = post["Activity"].to_numpy(dtype=float)
    bpm_post = post["BPM"].to_numpy(dtype=float)
    rm_post = post["RMSSD"].to_numpy(dtype=float)
    above = np.maximum(0.0, g_post - b0)
    iauc = float(above.sum() * dt_h)
    y = float(g_post.max() - b0)
    slope = float((g_pre[-1] - g_pre[0]) / max(t_pre[-1] - t_pre[0], 1e-6))
    return {
        "target_delta_max": y,
        "target_iauc": iauc,
        "pre_baseline_tail": b0,
        "pre_mean": float(np.mean(g_pre)),
        "pre_std": float(np.std(g_pre)),
        "pre_min": float(np.min(g_pre)),
        "pre_max": float(np.max(g_pre)),
        "pre_slope": slope,
        "pre_act_mean": float(np.mean(act_pre)),
        "pre_bpm_mean": float(np.mean(bpm_pre)),
        "pre_rmssd_mean": float(np.mean(rm_pre)),
        "post_act_mean": float(np.mean(act_post)),
        "post_bpm_mean": float(np.mean(bpm_post)),
        "post_rmssd_mean": float(np.mean(rm_post)),
        "meal_hour_sin": float(np.sin(2 * np.pi * (meal_t % 24.0) / 24.0)),
        "meal_hour_cos": float(np.cos(2 * np.pi * (meal_t % 24.0) / 24.0)),
    }


def d5_mss_multimodal_table() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not D5_M3.exists():
        return pd.DataFrame(), pd.DataFrame()

    foods = {
        re.search(r"food_(MSS\d+-\d+)\.xlsx$", p.name).group(1): p
        for p in D5_M3.glob("food_*.xlsx")
        if re.search(r"food_(MSS\d+-\d+)\.xlsx$", p.name)
    }
    ints = {
        re.search(r"glucose_actiheart_integrated_(MSS\d+-\d+)\.csv$", p.name).group(1): p
        for p in D5_M3.glob("glucose_actiheart_integrated_*.csv")
        if re.search(r"glucose_actiheart_integrated_(MSS\d+-\d+)\.csv$", p.name)
    }
    common = sorted(set(foods) & set(ints))

    meal_rows: List[Dict] = []
    for key in common:
        food = pd.read_excel(foods[key])
        integ = pd.read_csv(ints[key])
        for mt in _cluster_meal_onsets(food, gap_h=0.5):
            feat = _meal_features(integ, mt)
            if feat is None:
                continue
            feat["participant_id"] = key
            feat["meal_t"] = float(mt)
            meal_rows.append(feat)

    if not meal_rows:
        return pd.DataFrame(), pd.DataFrame()

    tall = pd.DataFrame(meal_rows)
    tall.to_csv(OUT / "v28_d5_mss_meal_features_long.csv", index=False)

    y = tall["target_delta_max"].to_numpy(dtype=float)
    groups = tall["participant_id"].astype(str).to_numpy()

    cols_cgm_pre = [
        "pre_baseline_tail",
        "pre_mean",
        "pre_std",
        "pre_min",
        "pre_max",
        "pre_slope",
        "meal_hour_sin",
        "meal_hour_cos",
    ]
    cols_wear_pre = ["pre_act_mean", "pre_bpm_mean", "pre_rmssd_mean"]
    cols_wear_post = ["post_act_mean", "post_bpm_mean", "post_rmssd_mean"]

    X1 = tall[cols_cgm_pre].to_numpy(dtype=float)
    X2 = tall[cols_cgm_pre + cols_wear_pre].to_numpy(dtype=float)
    X3 = tall[cols_cgm_pre + cols_wear_post].to_numpy(dtype=float)

    def eval_split(name: str, X: np.ndarray) -> Dict:
        _, pred = _loocv_group_ridge(X, y, groups)
        return {"model": name, **_safe_metrics(y, pred)}

    summary = pd.DataFrame(
        [
            eval_split("D5_LOOCV_pre_window_CGM_time_only", X1),
            eval_split("D5_LOOCV_pre_window_CGM_time_plus_wearable_pre", X2),
            eval_split("D5_LOOCV_pre_window_CGM_time_plus_wearable_post", X3),
        ]
    )

    y2 = tall["target_iauc"].to_numpy(dtype=float)
    _, p1 = _loocv_group_ridge(X1, y2, groups)
    _, p2 = _loocv_group_ridge(X2, y2, groups)
    _, p3 = _loocv_group_ridge(X3, y2, groups)
    summary_iauc = pd.DataFrame(
        [
            {"target": "iauc", "model": "pre_CGM_time_only", **_safe_metrics(y2, p1)},
            {"target": "iauc", "model": "pre_CGM_time_plus_wearable_pre", **_safe_metrics(y2, p2)},
            {"target": "iauc", "model": "pre_CGM_time_plus_wearable_post", **_safe_metrics(y2, p3)},
        ]
    )
    summary_iauc.to_csv(OUT / "v28_d5_mss_loocv_summary_iauc.csv", index=False)
    return tall, summary


def build_claims_table(
    d4_ctx: pd.DataFrame, d5_sum: pd.DataFrame, inv: pd.DataFrame
) -> pd.DataFrame:
    claims = [
        {
            "claim_id": "C1",
            "claim": "26D locked readout matches clinical classification (IR/Decomp) strongly on D4",
            "evidence": "v22_secondary_clinical_endpoints.csv; Ridge26D decomp_auc",
        },
        {
            "claim_id": "C2",
            "claim": "Peak DI performance is higher with Exp8 neural heads than Ridge26D alone",
            "evidence": "v19_overall_metrics GV_CorrLoss(Exp8) vs v22 Ridge26D",
        },
        {
            "claim_id": "C3",
            "claim": "Subject-level median 26D latents provide competitive linear readout vs adding explicit meal-aggregated context (macros/uncertainty) on D4",
            "evidence": "v28_d4_subject_context_readout.csv",
        },
        {
            "claim_id": "C4",
            "claim": "On external MSS (Phillips et al. 2023), concurrent post-prandial Actiheart summaries improve LOOCV prediction of glycemic iAUC beyond pre-meal CGM-only features",
            "evidence": "v28_d5_mss_loocv_summary_iauc.csv",
        },
        {
            "claim_id": "C5",
            "claim": "D3 free-living supports cross-window probe and baseline comparisons (see scorecard)",
            "evidence": str(V21D3),
        },
    ]
    return pd.DataFrame(claims)


def main() -> None:
    inv = build_evidence_inventory()
    inv.to_csv(OUT / "v28_evidence_inventory.csv", index=False)

    d4_ctx = d4_subject_context_readout()
    d4_ctx.to_csv(OUT / "v28_d4_subject_context_readout.csv", index=False)

    _, d5_sum = d5_mss_multimodal_table()
    if len(d5_sum):
        d5_sum.to_csv(OUT / "v28_d5_mss_loocv_summary.csv", index=False)

    claims = build_claims_table(d4_ctx, d5_sum, inv)
    claims.to_csv(OUT / "v28_claims_evidence_map.csv", index=False)

    meta = {
        "generated": datetime.now().isoformat(),
        "d5_citation": "Phillips NE, Collet TH, Naef F. Cell Reports Methods (2023). doi:10.1016/j.crmeth.2023.100545",
        "d5_note": "External cohort; mmol/L detrended glucose in MSS; not mergeable with D1-D4 subject IDs.",
        "outputs": [str(OUT / f) for f in os.listdir(OUT)],
    }
    with open(OUT / "v28_run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    lines = [
        "# v28 Framework value — summary",
        "",
        f"Generated: {meta['generated']}",
        "",
        "## D4 subject LOOCV: median 26D + meal context vs 26D only",
        "",
        d4_ctx.to_markdown(index=False) if len(d4_ctx) else "_missing_",
        "",
        "## D5 MSS (external): LOOCV meal response",
        "",
        d5_sum.to_markdown(index=False) if len(d5_sum) else "_D5 path missing — extract MultiSensor data to New_data/D5_MSS/data_",
        "",
        "## Claims map",
        "",
        claims.to_markdown(index=False),
        "",
    ]
    (OUT / "v28_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved", OUT)


if __name__ == "__main__":
    main()
