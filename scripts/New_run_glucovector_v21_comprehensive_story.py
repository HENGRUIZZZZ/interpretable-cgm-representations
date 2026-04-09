from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = "/Users/hertz1030/Documents/GitHub/interpretable-cgm-representations"
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v21_comprehensive")


def _weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    d = df[[value_col, weight_col]].dropna().copy()
    if d.empty:
        return np.nan
    w = d[weight_col].to_numpy(float)
    v = d[value_col].to_numpy(float)
    if np.sum(w) <= 0:
        return float(np.mean(v))
    return float(np.sum(v * w) / np.sum(w))


def _build_d4_scorecard() -> pd.DataFrame:
    v19 = pd.read_csv(os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v19", "v19_overall_metrics.csv"))
    v20_meal = pd.read_csv(os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v20", "v20_per_meal_metrics.csv"))
    v20_icc = pd.read_csv(os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v20", "v20_prediction_icc_ablation.csv"))
    met_meal = pd.read_csv(os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v18", "v18_metwally_per_meal_real.csv"))

    rows: List[Dict[str, object]] = []
    # pull main overall metrics from v19 (D4 gold)
    map_v19 = {
        "Wang_Exp1": "Wang(Exp1)",
        "GV_26D_Exp8": "GV_CorrLoss(Exp8)",
        "Metwally_Exp2": "Metwally(Exp2)",
        "SimpleStats_Exp4": "SimpleStats(Exp4)",
        "Healey_with_FI_proxy": "Healey(Exp3)",
    }
    for model_out, model_v19 in map_v19.items():
        d = v19[v19["model"] == model_v19]
        if d.empty:
            continue
        r = d.iloc[0].to_dict()
        rows.append(
            {
                "model": model_out,
                "setting": "D4_gold_overall",
                "sspg_spearman": r.get("sspg_spearman_r", np.nan),
                "sspg_rmse": r.get("sspg_rmse", np.nan),
                "di_spearman": r.get("di_spearman_r", np.nan),
                "di_rmse": r.get("di_rmse", np.nan),
                "invasive_feature": "yes" if "Healey" in model_out else "no",
            }
        )

    # Add GV10 and Healey_CGM_only from v20 per-meal weighted aggregation
    for m in ["GV_10D_head_v20", "Healey_CGM_only", "Healey_with_FI", "GV_26D_Exp8"]:
        d = v20_meal[v20_meal["model"] == m].copy()
        if d.empty:
            continue
        rows.append(
            {
                "model": m,
                "setting": "D4_gold_weighted_per_meal",
                "sspg_spearman": _weighted_mean(d, "sspg_spearman_r", "sspg_n"),
                "sspg_rmse": _weighted_mean(d, "sspg_rmse", "sspg_n"),
                "di_spearman": _weighted_mean(d, "di_spearman_r", "di_n"),
                "di_rmse": _weighted_mean(d, "di_rmse", "di_n"),
                "invasive_feature": "yes" if m == "Healey_with_FI" else "no",
            }
        )

    # Metwally per-meal weighted (to align with v20 per-meal view)
    rows.append(
        {
            "model": "Metwally_Exp2",
            "setting": "D4_gold_weighted_per_meal",
            "sspg_spearman": _weighted_mean(met_meal, "sspg_spearman_r", "sspg_n"),
            "sspg_rmse": _weighted_mean(met_meal, "sspg_rmse", "sspg_n"),
            "di_spearman": _weighted_mean(met_meal, "di_spearman_r", "di_n"),
            "di_rmse": _weighted_mean(met_meal, "di_rmse", "di_n"),
            "invasive_feature": "no",
        }
    )

    score = pd.DataFrame(rows)
    score = score.merge(v20_icc, on="model", how="left")
    return score.sort_values(["setting", "model"]).reset_index(drop=True)


def _build_d3_scorecard() -> pd.DataFrame:
    cross = pd.read_csv(os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v20", "d3_deep_ablation", "d3_cross_meal_cross_day_metrics.csv"))
    unc = pd.read_csv(os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v20", "d3_deep_ablation", "d3_uncertainty_stratified_correlations.csv"))
    probe = pd.read_csv(os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v20", "d3_deep_ablation", "d3_representation_probe_comparison.csv"))

    # keep 10D probe for Wang/GV and 26D probe where available
    probe_p = probe.pivot_table(index="model", columns="feature_set", values=["homa_ir_probe_spearman", "hba1c_probe_spearman"], aggfunc="first")
    probe_p.columns = [f"{a}_{b}" for a, b in probe_p.columns]
    probe_p = probe_p.reset_index()

    unc_p = unc.pivot_table(index="model", columns="uncertainty_bin", values=["sspg_vs_hba1c_spearman", "sspg_vs_homa_ir_spearman"], aggfunc="first")
    unc_p.columns = [f"{a}_{b}" for a, b in unc_p.columns]
    unc_p = unc_p.reset_index()

    d3 = cross.merge(unc_p, on="model", how="left").merge(probe_p, on="model", how="left")
    return d3.sort_values("model").reset_index(drop=True)


def _build_value_quantification(d4: pd.DataFrame, d3: pd.DataFrame) -> pd.DataFrame:
    vals: List[Dict[str, object]] = []
    # 10D representation value vs Wang
    d3i = d3.set_index("model")
    if "Wang_Exp1" in d3i.index and "GV_10D_head_v20" in d3i.index:
        vals.append(
            {
                "question": "10D_representation_value_vs_wang",
                "metric": "hba1c_probe_spearman_10D_delta",
                "value": float(d3i.loc["GV_10D_head_v20", "hba1c_probe_spearman_10D"] - d3i.loc["Wang_Exp1", "hba1c_probe_spearman_10D"]),
            }
        )
        vals.append(
            {
                "question": "10D_representation_value_vs_wang",
                "metric": "homa_ir_probe_spearman_10D_delta",
                "value": float(d3i.loc["GV_10D_head_v20", "homa_ir_probe_spearman_10D"] - d3i.loc["Wang_Exp1", "homa_ir_probe_spearman_10D"]),
            }
        )

    # 26D incremental value over 10D on D4 (accuracy side)
    d4w = d4[d4["setting"] == "D4_gold_weighted_per_meal"].set_index("model")
    if "GV_26D_Exp8" in d4w.index and "GV_10D_head_v20" in d4w.index:
        vals.append(
            {
                "question": "26D_incremental_value_over_10D_D4",
                "metric": "sspg_spearman_delta",
                "value": float(d4w.loc["GV_26D_Exp8", "sspg_spearman"] - d4w.loc["GV_10D_head_v20", "sspg_spearman"]),
            }
        )
        vals.append(
            {
                "question": "26D_incremental_value_over_10D_D4",
                "metric": "di_spearman_delta",
                "value": float(d4w.loc["GV_26D_Exp8", "di_spearman"] - d4w.loc["GV_10D_head_v20", "di_spearman"]),
            }
        )

    # 10D stability value over 26D on D3 (cross-day variance smaller is better)
    if "GV_26D_Exp8" in d3i.index and "GV_10D_head_v20" in d3i.index:
        vals.append(
            {
                "question": "10D_stability_value_over_26D_D3",
                "metric": "sspg_cross_day_std_reduction",
                "value": float(d3i.loc["GV_26D_Exp8", "sspg_pred_cross_day_std_mean"] - d3i.loc["GV_10D_head_v20", "sspg_pred_cross_day_std_mean"]),
            }
        )
        vals.append(
            {
                "question": "16D_value_under_high_uncertainty",
                "metric": "high_uncertainty_sspg_hba1c_delta_26D_minus_10D",
                "value": float(d3i.loc["GV_26D_Exp8", "sspg_vs_hba1c_spearman_high"] - d3i.loc["GV_10D_head_v20", "sspg_vs_hba1c_spearman_high"]),
            }
        )

    # Fairness: Healey with vs without invasive feature
    if "Healey_with_FI" in d4w.index and "Healey_CGM_only" in d4w.index:
        vals.append(
            {
                "question": "healey_fairness_gap",
                "metric": "sspg_spearman_withFI_minus_cgmOnly",
                "value": float(d4w.loc["Healey_with_FI", "sspg_spearman"] - d4w.loc["Healey_CGM_only", "sspg_spearman"]),
            }
        )
    return pd.DataFrame(vals)


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    d4 = _build_d4_scorecard()
    d3 = _build_d3_scorecard()
    value_df = _build_value_quantification(d4, d3)

    d4.to_csv(os.path.join(OUT_ROOT, "v21_d4_gold_comprehensive_scorecard.csv"), index=False)
    d3.to_csv(os.path.join(OUT_ROOT, "v21_d3_free_living_comprehensive_scorecard.csv"), index=False)
    value_df.to_csv(os.path.join(OUT_ROOT, "v21_value_quantification.csv"), index=False)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "n_models_d4": int(d4["model"].nunique()),
        "n_models_d3": int(d3["model"].nunique()),
    }
    with open(os.path.join(OUT_ROOT, "v21_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUT_ROOT, "v21_report.md"), "w", encoding="utf-8") as f:
        f.write("# v21 Comprehensive Story Audit (v6-v20 synthesis)\n\n")
        f.write(f"Generated: {summary['generated_at']}\n\n")
        f.write("## D4 Gold Scorecard\n\n")
        f.write(d4.to_markdown(index=False))
        f.write("\n\n## D3 Free-living Scorecard\n\n")
        f.write(d3.to_markdown(index=False))
        f.write("\n\n## Quantified Value of 10D/26D and Fairness\n\n")
        f.write(value_df.to_markdown(index=False))
        f.write("\n")

    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    main()
