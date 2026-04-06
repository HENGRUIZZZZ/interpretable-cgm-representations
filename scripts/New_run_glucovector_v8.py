from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from New_eval_trainD1D2_testD4 import eval_ckpt_on_d4, train_on_d1d2


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v8")


def _metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y = y[ok]
    yhat = yhat[ok]
    if len(y) < 3:
        return {
            "n": int(len(y)),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "r2": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
        }
    pr, pp = stats.pearsonr(y, yhat)
    sr, sp = stats.spearmanr(y, yhat)
    return {
        "n": int(len(y)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "r2": float(r2_score(y, yhat)),
        "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
        "mae": float(mean_absolute_error(y, yhat)),
    }


def _fit_linear_calibration(train_lat_csv: str, target: str) -> Tuple[float, float]:
    df = pd.read_csv(train_lat_csv)
    pred_col = f"{target}_hat"
    if target not in df.columns or pred_col not in df.columns:
        raise RuntimeError(f"Missing columns in {train_lat_csv}: {target}, {pred_col}")
    sub = df[[target, pred_col]].dropna().copy()
    x = sub[pred_col].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def _apply_calibration(scatter_csv: str, a: float, b: float, out_csv: str, out_metrics_json: str) -> Dict[str, float]:
    df = pd.read_csv(scatter_csv)
    if "pred" not in df.columns or "true" not in df.columns:
        # backward-compat
        cols = list(df.columns)
        true_col = "true" if "true" in cols else [c for c in cols if c.endswith("_true")][0]
        pred_col = "pred" if "pred" in cols else [c for c in cols if c.endswith("_pred")][0]
        df = df.rename(columns={true_col: "true", pred_col: "pred"})
    df["pred_calibrated"] = a * df["pred"].astype(float) + b
    met = _metrics(df["true"].to_numpy(), df["pred_calibrated"].to_numpy())
    out = df[["subject_id", "true", "pred_calibrated"]].rename(columns={"pred_calibrated": "pred"})
    out.to_csv(out_csv, index=False)
    with open(out_metrics_json, "w") as f:
        json.dump(met, f, indent=2)
    return met


def _run_shap(ckpt: str, train_lat_csv: str, target: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"shap_summary_{target}.png")
    out_csv = os.path.join(out_dir, f"shap_feature_importance_{target}.csv")
    cmd = (
        f'python "{os.path.join(REPO_ROOT, "scripts", "New_run_shap_analysis_v7.py")}" '
        f'--ckpt "{ckpt}" --train_latent_csv "{train_lat_csv}" --target "{target}" '
        f'--out_png "{out_png}" --out_csv "{out_csv}"'
    )
    os.system(cmd)


def _run_mealtype_compare(ckpt: str, out_csv: str) -> None:
    cmd = (
        f'python "{os.path.join(REPO_ROOT, "scripts", "New_run_exp3_d4_mealtype_v7.py")}" '
        f'--data_root "{DATA_ROOT}" --ckpt "{ckpt}" --out_csv "{out_csv}"'
    )
    os.system(cmd)


def _run_lodo_ablation(ckpt: str, out_csv: str) -> None:
    cmd = (
        f'python "{os.path.join(REPO_ROOT, "scripts", "New_run_exp4_lodo_v7.py")}" '
        f'--data_root "{DATA_ROOT}" --ckpt "{ckpt}" --out_csv "{out_csv}"'
    )
    os.system(cmd)


def _latest_train_dir(config_dir: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config_dir, "train_D1D2_*")))
    if not dirs:
        raise RuntimeError(f"No train_D1D2_* in {config_dir}")
    return dirs[-1]


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    configs = {
        "Config_A_10D": {"P1_V8_HEAD_10D": "1"},
        "Config_B_16D_Recon": {"P1_V8_RECON_CORR": "1"},
        "Config_C_16D_ODEHybrid": {"P1_V8_ODE_CORR": "1"},
        "Config_D_26D_Original": {},
    }

    summary_rows: List[Dict[str, object]] = []

    for cfg_name, extra in configs.items():
        cfg_dir = os.path.join(OUT_ROOT, cfg_name)
        os.makedirs(cfg_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # SSPG model
        tr_sspg = os.path.join(cfg_dir, f"train_D1D2_{stamp}_sspg")
        train_on_d1d2(
            cgm_project_output=DATA_ROOT,
            results_dir=tr_sspg,
            seed=21,
            lambda_sspg=0.1,
            lambda_di=0.0,
            num_epochs=100,
            extra_env=extra,
        )
        ck_sspg = os.path.join(tr_sspg, "autoencoder_p1_full.pt")
        raw_scatter_sspg = os.path.join(cfg_dir, "D4_sspg_scatter_raw.csv")
        eval_ckpt_on_d4(
            cgm_project_output=DATA_ROOT,
            ckpt_path=ck_sspg,
            out_dir=cfg_dir,
            target="sspg",
            metrics_filename="D4_sspg_metrics_raw.json",
            scatter_filename=os.path.basename(raw_scatter_sspg),
        )
        a_s, b_s = _fit_linear_calibration(os.path.join(tr_sspg, "latent_and_gold_all_26d.csv"), "sspg")
        m_sspg = _apply_calibration(
            raw_scatter_sspg,
            a_s,
            b_s,
            os.path.join(cfg_dir, "scatter_data_sspg.csv"),
            os.path.join(cfg_dir, "metrics_sspg.json"),
        )

        # DI model
        tr_di = os.path.join(cfg_dir, f"train_D1D2_{stamp}_di")
        train_on_d1d2(
            cgm_project_output=DATA_ROOT,
            results_dir=tr_di,
            seed=21,
            lambda_sspg=0.0,
            lambda_di=0.05,
            num_epochs=100,
            extra_env=extra,
        )
        ck_di = os.path.join(tr_di, "autoencoder_p1_full.pt")
        raw_scatter_di = os.path.join(cfg_dir, "D4_di_scatter_raw.csv")
        eval_ckpt_on_d4(
            cgm_project_output=DATA_ROOT,
            ckpt_path=ck_di,
            out_dir=cfg_dir,
            target="di",
            metrics_filename="D4_di_metrics_raw.json",
            scatter_filename=os.path.basename(raw_scatter_di),
        )
        a_d, b_d = _fit_linear_calibration(os.path.join(tr_di, "latent_and_gold_all_26d.csv"), "di")
        m_di = _apply_calibration(
            raw_scatter_di,
            a_d,
            b_d,
            os.path.join(cfg_dir, "scatter_data_di.csv"),
            os.path.join(cfg_dir, "metrics_di.json"),
        )

        # SHAP
        _run_shap(ck_sspg, os.path.join(tr_sspg, "latent_and_gold_all_26d.csv"), "sspg", cfg_dir)
        _run_shap(ck_di, os.path.join(tr_di, "latent_and_gold_all_26d.csv"), "di", cfg_dir)

        # Meal-type and LODO per config (use SSPG-model encoder/head as representative architecture)
        _run_mealtype_compare(ck_sspg, os.path.join(cfg_dir, "D4_meal_type_comparison.csv"))
        _run_lodo_ablation(ck_sspg, os.path.join(cfg_dir, "lodo_ablation_results_v8.csv"))

        summary_rows.append(
            {
                "config": cfg_name,
                "sspg_n": m_sspg["n"],
                "sspg_pearson_r": m_sspg["pearson_r"],
                "sspg_r2": m_sspg["r2"],
                "sspg_rmse": m_sspg["rmse"],
                "sspg_mae": m_sspg["mae"],
                "di_n": m_di["n"],
                "di_pearson_r": m_di["pearson_r"],
                "di_r2": m_di["r2"],
                "di_rmse": m_di["rmse"],
                "di_mae": m_di["mae"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_ROOT, "v8_config_comparison_summary.csv"), index=False)

    # quick markdown
    lines = [
        "# GlucoVector v8 summary",
        "",
        "Data: `New_data/P1_final_with_D4_DI/P1_final`",
        "",
        "## Config comparison (calibrated on D1+D2, tested on D4)",
        "",
        summary.to_markdown(index=False),
        "",
        "Per-config details are saved under:",
        "- `Config_A_10D/`",
        "- `Config_B_16D_Recon/`",
        "- `Config_C_16D_ODEHybrid/`",
        "- `Config_D_26D_Original/`",
    ]
    with open(os.path.join(OUT_ROOT, "New_v8_comprehensive_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

