from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import MechanisticAutoencoder
from scripts.New_eval_trainD1D2_testD4 import eval_ckpt_on_d4, train_on_d1d2


DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v9")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) < 3:
        return {
            "n": int(len(y_true)),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "r2": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
        }
    pr, pp = stats.pearsonr(y_true, y_pred)
    sr, sp = stats.spearmanr(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _fit_linear_calibration(train_lat_csv: str, target: str) -> Tuple[float, float]:
    df = pd.read_csv(train_lat_csv)
    sub = df[[target, f"{target}_hat"]].dropna().copy()
    a, b = np.polyfit(sub[f"{target}_hat"].to_numpy(dtype=float), sub[target].to_numpy(dtype=float), 1)
    return float(a), float(b)


def _apply_calibration(scatter_csv: str, a: float, b: float, out_csv: str, out_metrics_json: str) -> Dict[str, float]:
    df = pd.read_csv(scatter_csv)
    if "true" not in df.columns:
        t = [c for c in df.columns if c.endswith("_true")][0]
        p = [c for c in df.columns if c.endswith("_pred")][0]
        df = df.rename(columns={t: "true", p: "pred"})
    y = df["true"].to_numpy(dtype=float)
    yhat = a * df["pred"].to_numpy(dtype=float) + b
    met = _metrics(y, yhat)
    pd.DataFrame({"subject_id": df["subject_id"], "true": y, "pred": yhat}).to_csv(out_csv, index=False)
    with open(out_metrics_json, "w") as f:
        json.dump(met, f, indent=2)
    return met


def _infer_e2e_input_dim(state: Dict[str, torch.Tensor]) -> int:
    w = state.get("0.weight", None)
    return int(w.shape[1]) if w is not None else 26


def _build_e2e_head(input_dim: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2),
    )


def _build_cls_head() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(6, 16),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(16, 3),
    )


def _d4_maps() -> Dict[str, str]:
    s = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "subjects.csv"))
    return {f"D4_{str(r['original_id']).strip()}": str(r["subject_id"]).strip() for _, r in s.dropna(subset=["subject_id", "original_id"]).iterrows()}


def _extract_d4_standard_meal_windows() -> pd.DataFrame:
    cgm = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "cgm.csv"))
    cgm = cgm.rename(columns={"glucose_mg_dl": "glucose"})
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")
    meals = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "meals.csv"))
    meals["timestamp"] = pd.to_datetime(meals["timestamp"], errors="coerce")
    meals = meals[meals["meal_type"].isin(["Cornflakes", "PB_sandwich", "Protein_bar"])].copy()

    grid = np.arange(-30, 181, 5, dtype=float)
    rows: List[Dict[str, object]] = []
    for _, m in meals.iterrows():
        sid = str(m["subject_id"])
        meal_type = str(m["meal_type"])
        rep = int(m["repeat"]) if pd.notna(m.get("repeat", np.nan)) else -1
        t0 = m["timestamp"]
        g = cgm[(cgm["subject_id"] == sid) & (cgm["timestamp"] >= t0 + pd.Timedelta(minutes=-30)) & (cgm["timestamp"] <= t0 + pd.Timedelta(minutes=180))].copy()
        if len(g) < 10:
            continue
        t = ((g["timestamp"] - t0).dt.total_seconds() / 60.0).to_numpy(dtype=float)
        y = pd.to_numeric(g["glucose"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(t) & np.isfinite(y)
        if ok.sum() < 10:
            continue
        t = t[ok]
        y = y[ok]
        order = np.argsort(t)
        t = t[order]
        y = y[order]
        y_new = np.interp(grid, t, y)
        rows.append({"subject_id": sid, "meal_type": meal_type, "repeat": rep, "cgm_curve": y_new})
    return pd.DataFrame(rows)


def _tri_true_from_labels(label_df: pd.DataFrame) -> pd.DataFrame:
    d = label_df.copy()
    if "SSPG" in d.columns and "sspg" not in d.columns:
        d["sspg"] = d["SSPG"]
    if "DI" in d.columns and "di" not in d.columns:
        d["di"] = d["DI"]
    d = d[["subject_id", "sspg", "di"]].dropna().drop_duplicates("subject_id")
    tri = np.full(len(d), -1, dtype=int)
    s = d["sspg"].to_numpy(dtype=float)
    di = d["di"].to_numpy(dtype=float)
    tri[s < 120.0] = 0
    tri[(s >= 120.0) & (di >= 1.2)] = 1
    tri[(s >= 120.0) & (di < 1.2)] = 2
    d["tri_true"] = tri
    return d


def run_v9() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir = os.path.join(OUT_ROOT, f"train_ConfigC_{stamp}")
    train_on_d1d2(
        cgm_project_output=DATA_ROOT,
        results_dir=train_dir,
        seed=21,
        lambda_sspg=0.1,
        lambda_di=0.05,
        num_epochs=100,
        extra_env={
            "P1_V8_ODE_CORR": "1",
            "LAMBDA_CLS": "0.1",
            "P1_USE_TRI_CLASS": "1",
        },
    )
    ckpt = os.path.join(train_dir, "autoencoder_p1_full.pt")

    # Exp1: D4 SSPG/DI + calibration
    eval_ckpt_on_d4(
        cgm_project_output=DATA_ROOT,
        ckpt_path=ckpt,
        out_dir=OUT_ROOT,
        target="sspg",
        metrics_filename="D4_sspg_metrics_raw.json",
        scatter_filename="D4_sspg_true_vs_pred_raw.csv",
    )
    eval_ckpt_on_d4(
        cgm_project_output=DATA_ROOT,
        ckpt_path=ckpt,
        out_dir=OUT_ROOT,
        target="di",
        metrics_filename="D4_di_metrics_raw.json",
        scatter_filename="D4_di_true_vs_pred_raw.csv",
    )
    a_s, b_s = _fit_linear_calibration(os.path.join(train_dir, "latent_and_gold_all_26d.csv"), "sspg")
    a_d, b_d = _fit_linear_calibration(os.path.join(train_dir, "latent_and_gold_all_26d.csv"), "di")
    m_sspg = _apply_calibration(
        os.path.join(OUT_ROOT, "D4_sspg_true_vs_pred_raw.csv"),
        a_s,
        b_s,
        os.path.join(OUT_ROOT, "D4_sspg_true_vs_pred.csv"),
        os.path.join(OUT_ROOT, "D4_sspg_metrics.json"),
    )
    m_di = _apply_calibration(
        os.path.join(OUT_ROOT, "D4_di_true_vs_pred_raw.csv"),
        a_d,
        b_d,
        os.path.join(OUT_ROOT, "D4_di_true_vs_pred.csv"),
        os.path.join(OUT_ROOT, "D4_di_metrics.json"),
    )

    # Exp2 + Exp4: per-subject per-meal, meal-type comparison, tri-class
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    e2e_dim = _infer_e2e_input_dim(ck["e2e_head_state"])
    model = MechanisticAutoencoder(
        meal_size=6, demographics_size=3, embedding_size=8, hidden_size=32, num_layers=2, encoder_dropout_prob=0.0, decoder_dropout_prob=0.5
    )
    ms = ck.get("model_state", {})
    if hasattr(model, "use_v8_recon_corr"):
        model.use_v8_recon_corr = bool(ck.get("P1_V8_RECON_CORR", False))
    if hasattr(model, "use_v8_ode_corr"):
        model.use_v8_ode_corr = bool(ck.get("P1_V8_ODE_CORR", False))
    model.load_state_dict(ms, strict=False)
    model.eval()
    e2e = _build_e2e_head(e2e_dim)
    e2e.load_state_dict(ck["e2e_head_state"], strict=True)
    e2e.eval()
    cls_head = None
    if "cls_head_state" in ck:
        cls_head = _build_cls_head()
        cls_head.load_state_dict(ck["cls_head_state"], strict=True)
        cls_head.eval()

    d4w = _extract_d4_standard_meal_windows()
    if d4w.empty:
        raise RuntimeError("No valid D4 standard-meal windows extracted.")
    n = len(d4w)
    T = 43
    cgm = np.stack(d4w["cgm_curve"].to_list()).astype(np.float32)[:, :, None]
    ts = np.tile(np.arange(-30, 181, 5, dtype=np.float32)[None, :, None], (n, 1, 1))
    meals = np.zeros((n, T, 6), dtype=np.float32)
    demo = np.zeros((n, 3), dtype=np.float32)
    tm = ck["train_mean"]
    tsd = ck["train_std"]
    cgm = (cgm - tm[0]) / (tsd[0] + 1e-8)
    ts = (ts - tm[1]) / (tsd[1] + 1e-8)
    meals = (meals - tm[2]) / (tsd[2] + 1e-8)
    demo = (demo - tm[3]) / (tsd[3] + 1e-8)

    with torch.no_grad():
        p26, init26, z16 = model.get_all_latents(
            torch.tensor(cgm, dtype=torch.float32),
            torch.tensor(ts, dtype=torch.float32),
            torch.tensor(meals, dtype=torch.float32),
            torch.tensor(demo, dtype=torch.float32),
        )
        head_in = torch.cat([p26, init26], dim=-1) if e2e_dim == 10 else torch.cat([p26, init26, z16], dim=-1)
        pred2 = e2e(head_in).numpy()
        tri_pred = None
        if cls_head is not None:
            tri_logits = cls_head(p26).numpy()
            tri_pred = np.argmax(tri_logits, axis=1)

    sspg_hat = pred2[:, 0]
    di_hat = pred2[:, 1]
    if bool(ck.get("P1_ZSCORE_TARGETS", False)):
        sspg_hat = sspg_hat * float(ck["sspg_std"]) + float(ck["sspg_mean"])
        di_hat = di_hat * float(ck["di_std"]) + float(ck["di_mean"])
    # apply calibration
    sspg_hat_cal = a_s * sspg_hat + b_s
    di_hat_cal = a_d * di_hat + b_d

    cgm2d3h = _d4_maps()
    pred_df = d4w[["subject_id", "meal_type", "repeat"]].copy()
    pred_df["subject_id"] = pred_df["subject_id"].map(lambda x: cgm2d3h.get(x, x))
    pred_df["sspg_pred"] = sspg_hat_cal
    pred_df["di_pred"] = di_hat_cal
    if tri_pred is not None:
        pred_df["tri_pred"] = tri_pred
    pred_df.to_csv(os.path.join(OUT_ROOT, "per_subject_per_meal_preds.csv"), index=False)

    # meal type comparison (subject-mean per meal type)
    lab = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "labels.csv"))
    if "SSPG" in lab.columns and "sspg" not in lab.columns:
        lab["sspg"] = lab["SSPG"]
    if "DI" in lab.columns and "di" not in lab.columns:
        lab["di"] = lab["DI"]
    gold = lab[["subject_id", "sspg", "di"]].drop_duplicates("subject_id")
    rows = []
    for mt in ["Cornflakes", "PB_sandwich", "Protein_bar"]:
        sub = pred_df[pred_df["meal_type"] == mt].groupby("subject_id")[["sspg_pred", "di_pred"]].mean().reset_index()
        m = sub.merge(gold, on="subject_id", how="left")
        ms = _metrics(m["sspg"].to_numpy(), m["sspg_pred"].to_numpy())
        md = _metrics(m["di"].to_numpy(), m["di_pred"].to_numpy())
        rows.append(
            {
                "meal_type": mt,
                "sspg_pearson_r": ms["pearson_r"],
                "sspg_r2": ms["r2"],
                "sspg_rmse": ms["rmse"],
                "di_pearson_r": md["pearson_r"],
                "di_r2": md["r2"],
                "di_rmse": md["rmse"],
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(OUT_ROOT, "meal_type_comparison.csv"), index=False)

    # tri-class evaluation
    tri_true = _tri_true_from_labels(lab)
    tri_eval = pred_df.merge(tri_true[["subject_id", "tri_true"]], on="subject_id", how="left")
    tri_eval = tri_eval.dropna(subset=["tri_true"]).copy()
    tri_eval["tri_true"] = tri_eval["tri_true"].astype(int)
    tri_metrics = {"n": int(len(tri_eval))}
    if "tri_pred" in tri_eval.columns and len(tri_eval) > 0:
        y = tri_eval["tri_true"].to_numpy(dtype=int)
        yhat = tri_eval["tri_pred"].to_numpy(dtype=int)
        tri_metrics["accuracy"] = float(accuracy_score(y, yhat))
        tri_metrics["f1_macro"] = float(f1_score(y, yhat, average="macro"))
        cm = confusion_matrix(y, yhat, labels=[0, 1, 2])
    else:
        tri_metrics["accuracy"] = float("nan")
        tri_metrics["f1_macro"] = float("nan")
        cm = np.zeros((3, 3), dtype=int)
    with open(os.path.join(OUT_ROOT, "tri_class_metrics.json"), "w") as f:
        json.dump(tri_metrics, f, indent=2)
    pd.DataFrame(cm, index=["true_0", "true_1", "true_2"], columns=["pred_0", "pred_1", "pred_2"]).to_csv(
        os.path.join(OUT_ROOT, "confusion_matrix.csv")
    )

    # Exp3: SHAP
    subprocess.run(
        [
            "python",
            os.path.join(REPO_ROOT, "scripts", "New_run_shap_analysis_v7.py"),
            "--ckpt",
            ckpt,
            "--train_latent_csv",
            os.path.join(train_dir, "latent_and_gold_all_26d.csv"),
            "--target",
            "sspg",
            "--out_png",
            os.path.join(OUT_ROOT, "shap_summary_sspg.png"),
            "--out_csv",
            os.path.join(OUT_ROOT, "shap_feature_importance_sspg.csv"),
        ],
        check=False,
    )
    subprocess.run(
        [
            "python",
            os.path.join(REPO_ROOT, "scripts", "New_run_shap_analysis_v7.py"),
            "--ckpt",
            ckpt,
            "--train_latent_csv",
            os.path.join(train_dir, "latent_and_gold_all_26d.csv"),
            "--target",
            "di",
            "--out_png",
            os.path.join(OUT_ROOT, "shap_summary_di.png"),
            "--out_csv",
            os.path.join(OUT_ROOT, "shap_feature_importance_di.csv"),
        ],
        check=False,
    )
    # combined shap importance
    rows_shap = []
    for tgt in ["sspg", "di"]:
        p = os.path.join(OUT_ROOT, f"shap_feature_importance_{tgt}.csv")
        if os.path.isfile(p):
            d = pd.read_csv(p)
            if {"feature", "importance"}.issubset(set(d.columns)):
                d["target"] = tgt
                rows_shap.append(d[["target", "feature", "importance"]])
    if rows_shap:
        pd.concat(rows_shap, ignore_index=True).to_csv(os.path.join(OUT_ROOT, "shap_feature_importance.csv"), index=False)

    # final report
    rep = [
        "# v9_comprehensive_report",
        "",
        "Data: `New_data/P1_final_with_D4_DI/P1_final`",
        "",
        "## Exp1: Config C independent D4 performance (calibrated)",
        f"- SSPG metrics: {json.dumps(m_sspg, ensure_ascii=False)}",
        f"- DI metrics: {json.dumps(m_di, ensure_ascii=False)}",
        "",
        "## Exp2: Meal-type comparison",
        pd.read_csv(os.path.join(OUT_ROOT, "meal_type_comparison.csv")).to_markdown(index=False),
        "",
        "## Exp4: Tri-class",
        f"- tri_class_metrics: {json.dumps(tri_metrics, ensure_ascii=False)}",
        "",
        "Main outputs:",
        "- D4_sspg_metrics.json / D4_di_metrics.json",
        "- D4_sspg_true_vs_pred.csv / D4_di_true_vs_pred.csv",
        "- per_subject_per_meal_preds.csv",
        "- meal_type_comparison.csv",
        "- shap_summary_sspg.png / shap_summary_di.png",
        "- shap_feature_importance.csv",
        "- tri_class_metrics.json / confusion_matrix.csv",
    ]
    with open(os.path.join(OUT_ROOT, "v9_comprehensive_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))

    # zip package
    zip_path = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v9.zip")
    subprocess.run(["/bin/zsh", "-lc", f'cd "{REPO_ROOT}" && zip -r "{zip_path}" "New_paper1_results_glucovector_v9" >/dev/null'], check=False)


if __name__ == "__main__":
    run_v9()

