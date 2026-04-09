"""
v29 — (1) D5 MSS: Exp8 encoder on meal windows + LOOCV vs raw wearable ablations.
        (2) D4: subject-level context-augmented gate (10D base + gated 16D residual).

Requires: New_data/D5_MSS/data (MultiSensor Model3), Exp8 checkpoint.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models import MechanisticAutoencoder

V18_EXP8 = REPO / "New_paper1_results_glucovector_v18" / "v18_Exp8_CorrLoss" / "phase2_finetune_head" / "autoencoder_p1_full.pt"
V22_MEAL = REPO / "New_paper1_results_glucovector_v22_locked_protocol" / "v22_d4_meal_level_predictions.csv"
D5_M3 = REPO / "New_data" / "D5_MSS" / "data" / "Model3 data (glucose actiheart integrated)"
OUT = REPO / "New_paper1_results_glucovector_v29_mss_and_gate"
SEED = 42
MMOL_TO_MGDL = 18.0182
GRID_MIN = np.arange(-30, 181, 5, dtype=np.float64)


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 4:
        return {"n": float(len(y)), "spearman": np.nan, "r2": np.nan, "rmse": np.nan}
    return {
        "n": float(len(y)),
        "spearman": float(stats.spearmanr(y, p)[0]),
        "r2": float(r2_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
    }


class Encoder26:
    def __init__(self, ckpt_path: Path):
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.ck = ck
        self.model = MechanisticAutoencoder(
            meal_size=6,
            demographics_size=3,
            embedding_size=8,
            hidden_size=32,
            num_layers=2,
            encoder_dropout_prob=0.0,
            decoder_dropout_prob=0.5,
        )
        ms = ck.get("model_state", {})
        if hasattr(self.model, "use_v8_recon_corr"):
            self.model.use_v8_recon_corr = bool(
                ck.get("P1_V8_RECON_CORR", any(k.startswith("correction_mlp.") for k in ms.keys()))
            )
        if hasattr(self.model, "use_v8_ode_corr"):
            self.model.use_v8_ode_corr = bool(
                ck.get("P1_V8_ODE_CORR", any(k.startswith("ode_correction.") for k in ms.keys()))
            )
        self.model.load_state_dict(ck["model_state"], strict=False)
        self.model.eval()

    def encode_df(self, windows_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict] = []
        tm, tsd = self.ck["train_mean"], self.ck["train_std"]
        for _, r in windows_df.iterrows():
            c = r["curve"][None, :, None].astype(np.float32)
            ts = r["timestamps"][None, :, None].astype(np.float32)
            meals = r["meal_series"][None, :, :].astype(np.float32)
            demo = r["demographics"][None, :].astype(np.float32)
            c = (c - tm[0]) / (tsd[0] + 1e-8)
            ts = (ts - tm[1]) / (tsd[1] + 1e-8)
            meals = (meals - tm[2]) / (tsd[2] + 1e-8)
            demo = (demo - tm[3]) / (tsd[3] + 1e-8)
            with torch.no_grad():
                p26, init26, z16 = self.model.get_all_latents(
                    torch.tensor(c), torch.tensor(ts), torch.tensor(meals), torch.tensor(demo)
                )
            z = np.concatenate([p26.numpy()[0], init26.numpy()[0], z16.numpy()[0]], axis=0)
            row = {
                "participant_id": r["participant_id"],
                "meal_t": float(r["meal_t"]),
                "meal_type": r["meal_type"],
            }
            for i, v in enumerate(z):
                row[f"z{i:02d}"] = float(v)
            rows.append(row)
        return pd.DataFrame(rows)


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


def _meal_targets_and_wear(integ: pd.DataFrame, meal_t: float) -> Optional[Dict[str, float]]:
    use = integ.loc[~integ["mask"]].copy()
    if use.empty:
        return None
    pre = use[(use["abs_time_hours"] >= meal_t - 1.0) & (use["abs_time_hours"] <= meal_t)]
    post = use[(use["abs_time_hours"] >= meal_t) & (use["abs_time_hours"] <= meal_t + 3.0)]
    if len(pre) < 3 or len(post) < 2:
        return None
    pre_tail = pre[pre["abs_time_hours"] >= meal_t - 0.25]
    if len(pre_tail) < 1:
        pre_tail = pre.tail(2)
    b0 = float(pre_tail["Detrended"].mean())
    g_post = post["Detrended"].to_numpy(dtype=float)
    dt_h = 0.25
    above = np.maximum(0.0, g_post - b0)
    iauc = float(above.sum() * dt_h)
    y_delta = float(g_post.max() - b0)
    t_rel = (use["abs_time_hours"].to_numpy(dtype=float) - meal_t) * 60.0
    g_mg = use["Detrended"].to_numpy(dtype=float) * MMOL_TO_MGDL
    m = (t_rel >= -30) & (t_rel <= 180)
    t_rel, g_mg = t_rel[m], g_mg[m]
    if len(np.unique(t_rel)) < 5:
        return None
    order = np.argsort(t_rel)
    t_rel, g_mg = t_rel[order], g_mg[order]
    try:
        f = interp1d(t_rel, g_mg, kind="linear", fill_value="extrapolate")
        curve_full = f(GRID_MIN).astype(np.float32)
        g0 = float(f(0.0))
        # Encoder input must NOT contain post-prandial excursions (else targets leak).
        curve_enc = np.where(GRID_MIN <= 0.0, curve_full, np.float32(g0)).astype(np.float32)
    except Exception:
        return None
    if not np.all(np.isfinite(curve_enc)):
        return None
    act_post = post["Activity"].to_numpy(dtype=float)
    bpm_post = post["BPM"].to_numpy(dtype=float)
    rm_post = post["RMSSD"].to_numpy(dtype=float)
    act_pre = pre["Activity"].to_numpy(dtype=float)
    bpm_pre = pre["BPM"].to_numpy(dtype=float)
    rm_pre = pre["RMSSD"].to_numpy(dtype=float)
    return {
        "target_delta_max": y_delta,
        "target_iauc": iauc,
        "pre_act_mean": float(np.mean(act_pre)),
        "pre_bpm_mean": float(np.mean(bpm_pre)),
        "pre_rmssd_mean": float(np.mean(rm_pre)),
        "post_act_mean": float(np.mean(act_post)),
        "post_bpm_mean": float(np.mean(bpm_post)),
        "post_rmssd_mean": float(np.mean(rm_post)),
        "curve": curve_enc,
    }


def build_mss_windows() -> pd.DataFrame:
    if not D5_M3.exists():
        return pd.DataFrame()

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
    rows: List[Dict] = []
    demo = np.array([0.0, 32.0, 72.0], dtype=np.float32)
    meal_series = np.zeros((len(GRID_MIN), 6), dtype=np.float32)

    for key in sorted(set(foods) & set(ints)):
        food = pd.read_excel(foods[key])
        integ = pd.read_csv(ints[key])
        for mt in _cluster_meal_onsets(food, gap_h=0.5):
            pack = _meal_targets_and_wear(integ, mt)
            if pack is None:
                continue
            rows.append(
                {
                    "participant_id": key,
                    "meal_t": mt,
                    "meal_type": "MSS_freeliving",
                    "curve": pack["curve"],
                    "timestamps": GRID_MIN.astype(np.float32),
                    "meal_series": meal_series,
                    "demographics": demo,
                    "target_delta_max": pack["target_delta_max"],
                    "target_iauc": pack["target_iauc"],
                    "pre_act_mean": pack["pre_act_mean"],
                    "pre_bpm_mean": pack["pre_bpm_mean"],
                    "pre_rmssd_mean": pack["pre_rmssd_mean"],
                    "post_act_mean": pack["post_act_mean"],
                    "post_bpm_mean": pack["post_bpm_mean"],
                    "post_rmssd_mean": pack["post_rmssd_mean"],
                }
            )
    return pd.DataFrame(rows)


def _loocv_ridge(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.full(len(y), np.nan, dtype=float)
    groups = np.asarray(groups).astype(str).ravel()
    for g in np.unique(groups):
        te = (groups == g).ravel()
        tr = ~te
        if tr.sum() < 3:
            continue
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50)))
        mdl.fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te])
    return y, pred


def run_mss_encode_and_loocv(enc: Encoder26, win: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if win.empty:
        return pd.DataFrame(), pd.DataFrame()
    enc_df = enc.encode_df(win)
    meta = win[
        [
            "target_delta_max",
            "target_iauc",
            "pre_act_mean",
            "pre_bpm_mean",
            "pre_rmssd_mean",
            "post_act_mean",
            "post_bpm_mean",
            "post_rmssd_mean",
        ]
    ].reset_index(drop=True)
    full = pd.concat([enc_df.reset_index(drop=True), meta], axis=1)

    z10 = [f"z{i:02d}" for i in range(10)]
    z26 = [f"z{i:02d}" for i in range(26)]
    wear_pre = ["pre_act_mean", "pre_bpm_mean", "pre_rmssd_mean"]
    wear_post = ["post_act_mean", "post_bpm_mean", "post_rmssd_mean"]
    groups = full["participant_id"].astype(str).to_numpy().ravel()

    summary_rows = []
    for tgt in ["target_delta_max", "target_iauc"]:
        y = full[tgt].to_numpy(dtype=float)
        for name, cols in [
            ("Exp8_z10_only", z10),
            ("Exp8_z26_only", z26),
            ("Exp8_z26_plus_wearable_pre_post", z26 + wear_pre + wear_post),
        ]:
            X = full[cols].to_numpy(dtype=float)
            _, pred = _loocv_ridge(X, y, groups)
            summary_rows.append({"target": tgt, "model": name, **_metrics(y, pred)})
    return full, pd.DataFrame(summary_rows)


def _fit_context_gate_loocv(
    sub: pd.DataFrame,
    target: str,
    z10: List[str],
    z16: List[str],
    gate_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return y_true, pred_baseline10, pred_ridge26, pred_gated (subject LOOCV)."""
    groups = sub["subject_id"].astype(str).to_numpy().ravel()
    y_all = sub[target].to_numpy(dtype=float)
    z26 = z10 + z16
    pred10 = np.full(len(y_all), np.nan)
    pred26 = np.full(len(y_all), np.nan)
    pred_gate = np.full(len(y_all), np.nan)
    alphas = np.linspace(0.0, 2.0, 41)

    for g_hold in np.unique(groups):
        tr = groups != g_hold
        te = groups == g_hold
        if tr.sum() < 4:
            continue
        tr_idx = np.where(tr)[0]
        te_idx = np.where(te)[0]
        X10_tr = sub.iloc[tr_idx][z10].to_numpy(dtype=float)
        X16_tr = sub.iloc[tr_idx][z16].to_numpy(dtype=float)
        G_tr_raw = sub.iloc[tr_idx][gate_cols].to_numpy(dtype=float)
        y_tr = y_all[tr_idx]
        scaler_g = StandardScaler().fit(G_tr_raw)
        G_tr = scaler_g.transform(G_tr_raw)
        Xgate_tr = np.hstack([X16_tr, G_tr])

        base = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 40))).fit(X10_tr, y_tr)
        p10_tr = base.predict(X10_tr)
        res_tr = y_tr - p10_tr
        r16 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 40))).fit(X16_tr, res_tr)
        r16_tr = r16.predict(X16_tr)
        abs_e = np.abs(res_tr)
        thr = np.median(abs_e)
        hard = (abs_e >= thr).astype(int)
        gate = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=4000, random_state=SEED)
        ).fit(Xgate_tr, hard)

        best_a = 0.0
        best_mse = np.inf
        gprob_tr = gate.predict_proba(Xgate_tr)[:, 1]
        for a in alphas:
            pr = p10_tr + a * gprob_tr * r16_tr
            mse = float(np.mean((y_tr - pr) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_a = float(a)

        X10_te = sub.iloc[te_idx][z10].to_numpy(dtype=float)
        X16_te = sub.iloc[te_idx][z16].to_numpy(dtype=float)
        G_te = scaler_g.transform(sub.iloc[te_idx][gate_cols].to_numpy(dtype=float))
        Xgate_te = np.hstack([X16_te, G_te])
        p10 = base.predict(X10_te)
        rg = r16.predict(X16_te)
        gp = gate.predict_proba(Xgate_te)[:, 1]
        pg = p10 + best_a * gp * rg

        m26 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 40))).fit(
            np.hstack([X10_tr, X16_tr]), y_tr
        )
        pred26[te_idx] = m26.predict(np.hstack([X10_te, X16_te]))
        pred10[te_idx] = p10
        pred_gate[te_idx] = pg

    return y_all, pred10, pred26, pred_gate


def run_d4_subject_gate() -> pd.DataFrame:
    df = pd.read_csv(V22_MEAL)
    z10 = [f"z{i:02d}" for i in range(10)]
    z16 = [f"z{i:02d}" for i in range(10, 26)]
    z26 = z10 + z16

    med = df.groupby("subject_id", as_index=False)[z26].median()
    ctx = df.groupby("subject_id", as_index=False).agg(
        unc_mean=("uncertainty_score", "mean"),
        carb_mean=("carb_g", "mean"),
        fat_mean=("fat_g", "mean"),
        prot_mean=("protein_g", "mean"),
        fiber_mean=("fiber_g", "mean"),
        n_meals=("subject_id", "size"),
    )
    sub = med.merge(ctx, on="subject_id", how="inner")
    lab = df.groupby("subject_id", as_index=False)[["sspg_true", "di_true"]].first()
    sub = sub.merge(lab, on="subject_id", how="inner")

    mt = pd.crosstab(df["subject_id"], df["meal_type"])
    mt = mt.reset_index()
    sub = sub.merge(mt, on="subject_id", how="left").fillna(0.0)

    gate_cols = [
        "unc_mean",
        "carb_mean",
        "fat_mean",
        "prot_mean",
        "fiber_mean",
        "n_meals",
    ]
    for c in mt.columns:
        if c != "subject_id":
            gate_cols.append(c)

    rows = []
    for tgt in ["sspg_true", "di_true"]:
        ok = np.isfinite(sub[tgt].to_numpy(dtype=float))
        sub_t = sub.loc[ok].reset_index(drop=True)
        y, p10, p26, pg = _fit_context_gate_loocv(sub_t, tgt, z10, z16, gate_cols)
        rows.append({"target": tgt, "model": "LOOCV_Ridge10D_baseline", **_metrics(y, p10)})
        rows.append({"target": tgt, "model": "LOOCV_Ridge26D", **_metrics(y, p26)})
        rows.append({"target": tgt, "model": "LOOCV_Gated10D_16Dresidual_ctx", **_metrics(y, pg)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    enc = Encoder26(V18_EXP8)

    win = build_mss_windows()
    if len(win):
        win.to_pickle(OUT / "v29_d5_mss_windows_meta.pkl")
        full, mss_sum = run_mss_encode_and_loocv(enc, win)
        full.to_csv(OUT / "v29_d5_mss_exp8_latents_and_targets.csv", index=False)
        mss_sum.to_csv(OUT / "v29_d5_mss_loocv_exp8_vs_wearable.csv", index=False)
    else:
        mss_sum = pd.DataFrame()
        full = pd.DataFrame()

    d4g = run_d4_subject_gate()
    d4g.to_csv(OUT / "v29_d4_subject_context_gate_loocv.csv", index=False)

    meta = {
        "generated": datetime.now().isoformat(),
        "d5_note": "Encoder sees pre-meal CGM only: post 0..180 min held flat at meal-start glucose (no post-excursion leakage). Targets still from true post curve. MSS mmol/L->mg/dL. Meal macros zero.",
        "d4_note": "Subject-level LOOCV; gate sees z16 + meal context aggregates + meal-type counts; alpha tuned on train MSE (in-fold).",
    }
    (OUT / "v29_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "# v29 Results",
        "",
        "## D5 MSS — Exp8 latents + LOOCV (by participant)",
        "",
        mss_sum.to_markdown(index=False) if len(mss_sum) else "_no D5 data_",
        "",
        "## D4 — Context gate (subject LOOCV)",
        "",
        d4g.to_markdown(index=False),
        "",
        "## 结果解读（中文要点）",
        "",
        "1. **D5 行**：在 **留一参与者（LOOCV）** 下，用餐后血糖动力学目标（`target_delta_max` / `target_iauc`）做回归。",
        "   - `Exp8_z10_only` / `Exp8_z26_only`：把 **在你们数据上训练的 Exp8 编码器** 直接用到 MSS 的餐窗上（血糖已从 mmol/L 转成 mg/dL 并对齐 -30~180 min 网格；**宏量营养在 MSS 里缺失，此处为全零**）。",
        "   - 若 `z26_plus_wearable` 优于纯 `z26`，说明 **Actiheart 的活动/心率摘要** 在域外数据上仍能补充表征；若接近或更差，说明编码器域偏移或任务更难，应在文中如实写。",
        "",
        "2. **D4 行**：每个被试只保留一行（跨餐 **median 26D** + 餐次上下文汇总）。",
        "   - `Ridge10D`：只用机制 10 维读金标准。",
        "   - `Ridge26D`：全 26 维线性读数（与 locked 口径一致）。",
        "   - `Gated10D_16Dresidual_ctx`：10D 做主预测，16D 只预测残差；**门控**用 16D+宏量/不确定度/餐型计数学习「何时放大残差校正」，alpha 在训练折内按 MSE 网格选。",
        "   - 若门控版在 Spearman/R² 上略优于 10D 但不及 26D，叙事是：**分解式机制+条件残差** 部分追回全 26D 的收益；若接近 26D，则说明门控成功把 16D 用在「该用的时候」。",
        "",
        "3. **与 v28 的关系**：v28 的 MSS 表格是 **纯手工特征**（餐前 CGM 形状 + 可穿戴），**不经过 GlucoVector**；v29 补上 **同一批餐的 Exp8 潜变量**，把「框架」和「多传感器」接在同一张 D5 结果里。",
        "",
    ]
    (OUT / "v29_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved", OUT)


if __name__ == "__main__":
    main()
