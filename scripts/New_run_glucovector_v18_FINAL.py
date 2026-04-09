#!/usr/bin/env python3
"""
GlucoVector v18 Experiment Runner (FINAL)
==========================================
10 个实验：4 Baseline + 5 GlucoVector 升级 + 1 Ridge Probe

执行前：
  1. cd ~/interpretable-cgm-representations
  2. python ~/v18_patches.py --apply
  3. python ~/v18_patches.py --verify
  4. python ~/New_run_glucovector_v18_FINAL.py

实验矩阵：
  Exp1:  Wang Baseline (10D ODE, narrow bounds, single-phase)
  Exp2:  Metwally 14-Feature Baseline (传统 ML)
  Exp3:  Healey ODE + 空腹胰岛素 Baseline
  Exp4:  Simple Stats Baseline (CGM mean/std/CV + Ridge)
  Exp5:  GV v17 Baseline (26D, joint head, 100 epochs, no ES)
  Exp6:  +Separate Small Heads (26D→16→1 per target)
  Exp7:  +Early Stopping (patience=15, 200 epochs)
  Exp8:  +CorrLoss (50% MSE + 50% Pearson)
  Exp9:  Full Combo (+Phase 1 正交约束)
  Exp10: Ridge Probe (26D latent → Ridge LOO-CV)

环境变量对照表（v18_patches.py 定义）：
  V18_SEPARATE_SMALL_HEAD    → "1" 启用分离小头 (26D→16→1)
  V18_EARLY_STOPPING_PATIENCE → "15" 启用 Early Stopping
  V18_CORR_LOSS_ALPHA        → "0.5" 启用 CorrLoss (50% MSE + 50% Pearson)
  V18_LAMBDA_ORTHO_P1        → "0.01" Phase 1 正交约束
"""
import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# ============================================================================
# Paths & Config
# ============================================================================
REPO_ROOT = Path(os.environ.get("REPO_ROOT",
    "/Users/hertz1030/Documents/GitHub/interpretable-cgm-representations"))
DATA_ROOT = Path(os.environ.get("CGM_DATA_ROOT",
    "/Users/hertz1030/Documents/GitHub/interpretable-cgm-representations/New_data/P1_final_with_D4_DI/P1_final"))
OUT_ROOT = Path(os.environ.get("V18_OUT_ROOT",
    "/Users/hertz1030/Documents/GitHub/interpretable-cgm-representations/New_paper1_results_glucovector_v18"))
SEED = int(os.environ.get("V18_SEED", "42"))

PIPELINE = REPO_ROOT / "run_p1_full_pipeline.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "New_eval_trainD1D2_testD4.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================================
# Pre-flight
# ============================================================================
def _check_patches():
    code = PIPELINE.read_text()
    required = ["V18_EARLY_STOPPING_PATIENCE", "V18_CORR_LOSS_ALPHA",
                 "V18_SEPARATE_SMALL_HEAD", "V18_LAMBDA_ORTHO_P1"]
    missing = [v for v in required if f'os.environ.get("{v}"' not in code]
    if missing:
        print(f"[FATAL] v18 patches NOT applied! Missing: {missing}")
        print(f"  Run: python ~/v18_patches.py --apply")
        sys.exit(1)
    print("[OK] v18 patches verified in run_p1_full_pipeline.py")


# ============================================================================
# Base env: 所有实验共享的默认值
# ============================================================================
def _base_env() -> Dict[str, str]:
    return {
        # 数据
        "P1_TRAIN_DATASETS": "D1,D2",
        "P1_ONE_MEAL_PER_SUBJECT": "1",
        # 架构
        "P1_HEAD_USE_26D": "1",
        "P1_V8_HEAD_10D": "0",
        "P1_SEPARATE_HEAD_26D": "0",
        "P1_V10_WIDE_BOUNDS": "1",
        "P1_SAVE_26D_LATENT": "1",
        "P1_V8_ODE_CORR": "0",
        "P1_V8_RECON_CORR": "0",
        "P1_ZSCORE_TARGETS": "1",
        "P1_LR": "1e-2",
        "P1_USE_LR_SCHEDULER": "1",
        # 训练
        "P1_FINETUNE_HEAD_ONLY": "0",
        "P1_PRETRAINED_MODEL": "",
        "P1_FINETUNE_16D_ONLY": "0",
        # 损失
        "LAMBDA_SSPG": "0.0",
        "LAMBDA_DI": "0.0",
        "LAMBDA_CLS": "0.0",
        "LAMBDA_ORTHO_16D": "0.0",
        "LAMBDA_VAR_MATCH": "0.0",
        # v18 新增（默认全关）
        "V18_EARLY_STOPPING_PATIENCE": "0",
        "V18_CORR_LOSS_ALPHA": "0.0",
        "V18_SEPARATE_SMALL_HEAD": "0",
        "V18_LAMBDA_ORTHO_P1": "0.0",
    }


# ============================================================================
# Training helpers
# ============================================================================
def _run_pipeline(results_dir: str, extra_env: Dict[str, str],
                  lambda_sspg: float = 0.0, lambda_di: float = 0.0,
                  num_epochs: int = 100, seed: int = SEED) -> str:
    """运行 run_p1_full_pipeline.py，返回 checkpoint 路径。"""
    os.makedirs(results_dir, exist_ok=True)
    # 确定 checkpoint 文件名（pipeline 可能保存为不同名称）
    ckpt_candidates = ["best_model.pt", "autoencoder_p1_full.pt"]
    for c in ckpt_candidates:
        p = os.path.join(results_dir, c)
        if os.path.isfile(p):
            print(f"  [SKIP] checkpoint exists: {p}")
            return p

    env = dict(os.environ)
    env.update(_base_env())
    env["CGM_PROJECT_OUTPUT"] = str(DATA_ROOT)
    env["P1_RESULTS_DIR"] = results_dir
    env["P1_NUM_EPOCHS"] = str(num_epochs)
    env["P1_SEED"] = str(seed)
    env["LAMBDA_SSPG"] = str(lambda_sspg)
    env["LAMBDA_DI"] = str(lambda_di)
    env.update(extra_env)

    cmd = [
        sys.executable, str(PIPELINE),
        "--lambda_sspg", str(lambda_sspg),
        "--lambda_di", str(lambda_di),
    ]
    # 只打印关键覆盖
    overrides = {k: v for k, v in extra_env.items()
                 if k.startswith(("V18_", "LAMBDA_", "P1_FINETUNE", "P1_V8_HEAD", "P1_V10_WIDE"))}
    print(f"  [RUN] results_dir={results_dir}")
    print(f"  [ENV] {overrides}")
    print(f"  [CMD] lambda_sspg={lambda_sspg} lambda_di={lambda_di} epochs={num_epochs}")

    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    print(f"  [DONE] {elapsed:.0f}s, returncode={result.returncode}")

    if result.returncode != 0:
        err_file = os.path.join(results_dir, "stderr.log")
        with open(err_file, "w") as f:
            f.write(result.stderr)
        print(f"  [ERROR] see {err_file}")
        for line in result.stderr.strip().split('\n')[-15:]:
            print(f"    {line}")
        raise RuntimeError(f"Training failed: {results_dir}")

    # 找到 checkpoint
    for c in ckpt_candidates:
        p = os.path.join(results_dir, c)
        if os.path.isfile(p):
            return p
    # fallback: 返回第一个 .pt 文件
    for f in sorted(Path(results_dir).glob("*.pt")):
        return str(f)
    raise FileNotFoundError(f"No checkpoint found in {results_dir}")


def _two_phase(exp_dir: str,
               extra_env_p1: Dict[str, str],
               extra_env_p2: Dict[str, str],
               epochs_p1: int = 100,
               epochs_p2: int = 100) -> str:
    """两阶段训练。返回 Phase 2 checkpoint 路径。"""
    p1_dir = os.path.join(exp_dir, "phase1_unsupervised")
    p2_dir = os.path.join(exp_dir, "phase2_finetune_head")

    # Phase 1: 无监督
    print(f"\n  --- Phase 1: Unsupervised ({epochs_p1} epochs) ---")
    ck_p1 = _run_pipeline(p1_dir, extra_env_p1,
                          lambda_sspg=0.0, lambda_di=0.0,
                          num_epochs=epochs_p1)

    # Phase 2: 微调预测头
    print(f"\n  --- Phase 2: Finetune Head ({epochs_p2} epochs) ---")
    env_p2 = dict(extra_env_p2)
    env_p2["P1_FINETUNE_HEAD_ONLY"] = "1"
    env_p2["P1_PRETRAINED_MODEL"] = ck_p1
    ck_p2 = _run_pipeline(p2_dir, env_p2,
                          lambda_sspg=0.1, lambda_di=0.1,
                          num_epochs=epochs_p2)
    return ck_p2


# ============================================================================
# Evaluation helpers
# ============================================================================
def _eval_d4(ckpt: str, eval_dir: str) -> Dict[str, Any]:
    """D4 评估：分别跑 SSPG 和 DI。"""
    os.makedirs(eval_dir, exist_ok=True)
    result = {}
    for target in ["sspg", "di"]:
        try:
            cmd = [sys.executable, str(EVAL_SCRIPT),
                   "--cgm_project_output", str(DATA_ROOT),
                   "--ckpt", ckpt,
                   "--out_dir", eval_dir,
                   "--target", target]
            subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except Exception as e:
            print(f"  [WARN] D4 eval ({target}): {e}")

    for mf in ["New_D4_sspg_metrics.json", "New_D4_di_metrics.json"]:
        mp = os.path.join(eval_dir, mf)
        if os.path.isfile(mp):
            with open(mp) as f:
                m = json.load(f)
            prefix = "sspg" if "sspg" in mf else "di"
            for k, v in m.items():
                result[f"d4_{prefix}_{k}"] = v
    return result


def _pred_stats(eval_dir: str) -> Dict[str, float]:
    """预测值分布统计。"""
    stats = {}
    for target in ["sspg", "di"]:
        tvp = os.path.join(eval_dir, f"New_D4_{target}_true_vs_pred.csv")
        if os.path.exists(tvp):
            import pandas as pd
            df = pd.read_csv(tvp)
            stats[f"{target}_pred_mean"] = float(df['pred'].mean())
            stats[f"{target}_pred_std"] = float(df['pred'].std())
            stats[f"{target}_true_std"] = float(df['true'].std())
            stats[f"{target}_compression"] = float(df['pred'].std() / max(df['true'].std(), 1e-8))
            stats[f"{target}_pred_range"] = float(df['pred'].max() - df['pred'].min())
    return stats


def _auroc(eval_dir: str, threshold: float = 120.0) -> Dict[str, float]:
    """分类 auROC + Bootstrap CI。"""
    tvp = os.path.join(eval_dir, "New_D4_sspg_true_vs_pred.csv")
    if not os.path.exists(tvp):
        return {}
    try:
        import pandas as pd
        from sklearn.metrics import roc_auc_score
        df = pd.read_csv(tvp)
        y_true = (df['true'] > threshold).astype(int).values
        y_score = df['pred'].values
        if len(np.unique(y_true)) < 2:
            return {}
        auc = roc_auc_score(y_true, y_score)
        # Bootstrap CI
        rng = np.random.RandomState(42)
        aucs = []
        for _ in range(2000):
            idx = rng.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        return {
            'auroc': round(auc, 4),
            'auroc_ci_lo': round(np.percentile(aucs, 2.5), 4) if aucs else None,
            'auroc_ci_hi': round(np.percentile(aucs, 97.5), 4) if aucs else None,
        }
    except Exception as e:
        print(f"  [WARN] auROC: {e}")
        return {}


def _per_meal(ckpt: str, exp_dir: str) -> List[Dict]:
    """Per-meal-type 评估。"""
    try:
        script = REPO_ROOT / "scripts" / "eval_d4_per_meal_type.py"
        if not script.exists():
            return []
        cmd = [sys.executable, str(script),
               "--cgm_project_output", str(DATA_ROOT),
               "--ckpt", ckpt, "--out_dir", exp_dir]
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        csv_path = os.path.join(exp_dir, "d4_per_meal_type.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            return pd.read_csv(csv_path).to_dict('records')
    except Exception as e:
        print(f"  [WARN] per-meal: {e}")
    return []


def _shap(ckpt: str, out_dir: str) -> Dict[str, Any]:
    """SHAP 分析。"""
    try:
        script = REPO_ROOT / "scripts" / "shap_analysis.py"
        if not script.exists():
            return {}
        os.makedirs(out_dir, exist_ok=True)
        cmd = [sys.executable, str(script),
               "--cgm_project_output", str(DATA_ROOT),
               "--ckpt", ckpt, "--out_dir", out_dir]
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        csv_path = os.path.join(out_dir, "shap_feature_importance.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            ode_feats = ['tau_m', 'Gb', 'sg', 'si', 'p2', 'mi',
                        'z_init_0', 'z_init_1', 'z_init_2', 'z_init_3']
            total = df['mean_abs_shap'].sum()
            ode = df[df['feature'].isin(ode_feats)]['mean_abs_shap'].sum()
            return {'shap_10d_pct': round(100 * ode / max(total, 1e-8), 1),
                    'shap_16d_pct': round(100 * (1 - ode / max(total, 1e-8)), 1)}
    except Exception as e:
        print(f"  [WARN] SHAP: {e}")
    return {}


# ============================================================================
# Traditional baseline helpers (Exp2/3/4)
# ============================================================================
def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for old, new in [("SSPG", "sspg"), ("DI", "di"), ("HOMA_IR", "homa_ir"), ("HOMA_B", "homa_b")]:
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from scipy import stats
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    if len(y) < 3:
        return {"n": int(len(y))}
    pr, pp = stats.pearsonr(y, p)
    sr, sp = stats.spearmanr(y, p)
    return {
        "n": int(len(y)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "r2": float(r2_score(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
    }


def _build_fasting_insulin_maps() -> Dict[str, Dict[str, float]]:
    maps: Dict[str, Dict[str, float]] = {"D1": {}, "D2": {}, "D4": {}}
    # D2 / D4 direct
    d2_labels = _normalize_labels(pd.read_csv(DATA_ROOT / "D2_stanford" / "labels.csv"))
    if "fasting_insulin" in d2_labels.columns:
        for _, r in d2_labels.dropna(subset=["subject_id", "fasting_insulin"]).iterrows():
            maps["D2"][str(r["subject_id"])] = float(r["fasting_insulin"])
    d4_labels = _normalize_labels(pd.read_csv(DATA_ROOT / "D4_hall" / "labels.csv"))
    if "fasting_insulin" in d4_labels.columns:
        for _, r in d4_labels.dropna(subset=["subject_id", "fasting_insulin"]).iterrows():
            maps["D4"][str(r["subject_id"])] = float(r["fasting_insulin"])
    # D1 derive from HOMA_IR and FPG
    d1_labels = _normalize_labels(pd.read_csv(DATA_ROOT / "D1_metwally" / "labels.csv"))
    d1_subjects = pd.read_csv(DATA_ROOT / "D1_metwally" / "subjects.csv")
    fpg_map = {}
    if "fpg" in d1_subjects.columns:
        for _, r in d1_subjects.dropna(subset=["subject_id", "fpg"]).iterrows():
            fpg_map[str(r["subject_id"])] = float(r["fpg"])
    if "demo_FPG" in d1_labels.columns:
        for _, r in d1_labels.dropna(subset=["subject_id", "demo_FPG"]).iterrows():
            fpg_map[str(r["subject_id"])] = float(r["demo_FPG"])
    if "homa_ir" in d1_labels.columns:
        for _, r in d1_labels.dropna(subset=["subject_id", "homa_ir"]).iterrows():
            sid = str(r["subject_id"])
            fpg = fpg_map.get(sid, np.nan)
            if np.isfinite(fpg) and fpg > 0:
                maps["D1"][sid] = float(r["homa_ir"]) * 405.0 / float(fpg)
    return maps


def _curve_postmeal_features(cgm: np.ndarray, ts: np.ndarray) -> Dict[str, float]:
    # Use post-meal 0-180min segment for handcrafted features
    m = (ts >= 0.0) & (ts <= 180.0)
    t = ts[m]
    g = cgm[m]
    if len(t) < 5:
        t = ts
        g = cgm
    order = np.argsort(t)
    t = t[order]
    g = g[order]
    f = lambda x: float(np.interp(x, t, g))
    g0 = float(g[0])
    g60, g120, g180 = f(60), f(120), f(180)
    gpeak = float(np.max(g))
    delta = g - g0
    auc = float(np.trapz(g, t))
    p_auc = float(np.trapz(np.maximum(delta, 0), t))
    n_auc = float(np.trapz(np.minimum(delta, 0), t))
    i_auc = float(np.trapz(delta, t))
    curve_size = float(np.trapz(np.abs(delta), t))
    cv = float(np.std(g) / max(np.mean(g), 1e-8))
    i_peak = int(np.argmax(g))
    t_b2p = float(max(t[i_peak] - t[0], 1.0))
    s_b2p = float((gpeak - g0) / t_b2p)
    t_p2e = float(max(t[-1] - t[i_peak], 1.0))
    s_p2e = float((g[-1] - gpeak) / t_p2e)
    return {
        "G_0": g0, "G_60": g60, "G_120": g120, "G_180": g180,
        "G_Peak": gpeak, "CurveSize": curve_size, "AUC": auc, "pAUC": p_auc,
        "nAUC": n_auc, "iAUC": i_auc, "CV": cv, "T_baseline2peak": t_b2p,
        "S_baseline2peak": s_b2p, "S_peak2end": s_p2e,
        "mean": float(np.mean(g)), "std": float(np.std(g)), "min": float(np.min(g)),
        "max": float(np.max(g)), "range": float(np.max(g) - np.min(g)),
    }


def _collect_subject_curves() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Returns subject-level rows for train(D1+D2) and test(D4)
    from load_cgm_project_data import load_cgm_project_level1_level2, load_cgm_project_level3

    rows_train: List[Dict[str, Any]] = []
    rows_test: List[Dict[str, Any]] = []
    fi_maps = _build_fasting_insulin_maps()

    for ds in ["D1", "D2"]:
        b, info, lab = load_cgm_project_level1_level2(dataset_id=ds, output_base=str(DATA_ROOT))
        lab = _normalize_labels(lab)
        lab_map = lab.dropna(subset=["subject_id"]).drop_duplicates("subject_id").set_index("subject_id").to_dict(orient="index")
        seen = set()
        for i, sid in enumerate(info.patient_ids):
            sid = str(sid)
            if sid in seen:
                continue
            seen.add(sid)
            feats = _curve_postmeal_features(b.cgm[i, :, 0], b.timestamps[i, :, 0])
            rec = {"subject_id": sid, "dataset": ds, **feats}
            rec["sspg"] = lab_map.get(sid, {}).get("sspg", np.nan)
            rec["di"] = lab_map.get(sid, {}).get("di", np.nan)
            rec["fasting_insulin"] = fi_maps[ds].get(sid, np.nan)
            rows_train.append(rec)

    b4, info4, lab4 = load_cgm_project_level3(dataset_id="D4", output_base=str(DATA_ROOT))
    lab4 = _normalize_labels(lab4)
    lab4_map = lab4.dropna(subset=["subject_id"]).drop_duplicates("subject_id").set_index("subject_id").to_dict(orient="index")
    # Map D4 meal/cgm IDs (e.g. D4_2133-001) -> label IDs (e.g. D3H_001)
    id_map: Dict[str, str] = {}
    d4_subj = pd.read_csv(DATA_ROOT / "D4_hall" / "subjects.csv")
    if {"subject_id", "original_id"}.issubset(set(d4_subj.columns)):
        for _, rr in d4_subj.dropna(subset=["subject_id", "original_id"]).iterrows():
            orig = str(rr["original_id"]).strip()
            sid = str(rr["subject_id"]).strip()
            id_map[orig] = sid
            id_map[f"D4_{orig}"] = sid
    # subject-level aggregate by mean across meals
    by_sid: Dict[str, List[Dict[str, float]]] = {}
    for i, sid in enumerate(info4.patient_ids):
        sid = id_map.get(str(sid), str(sid))
        feats = _curve_postmeal_features(b4.cgm[i, :, 0], b4.timestamps[i, :, 0])
        by_sid.setdefault(sid, []).append(feats)
    for sid, feat_list in by_sid.items():
        df = pd.DataFrame(feat_list)
        feats = df.mean(numeric_only=True).to_dict()
        rec = {"subject_id": sid, "dataset": "D4", **feats}
        rec["sspg"] = lab4_map.get(sid, {}).get("sspg", np.nan)
        rec["di"] = lab4_map.get(sid, {}).get("di", np.nan)
        rec["fasting_insulin"] = fi_maps["D4"].get(sid, np.nan)
        rows_test.append(rec)
    return pd.DataFrame(rows_train), pd.DataFrame(rows_test)


def _fit_healey_params(g: np.ndarray, t: np.ndarray, fasting_insulin: float) -> Dict[str, float]:
    from scipy.integrate import solve_ivp
    from scipy.optimize import minimize

    alpha = 10000.0
    t = np.asarray(t, dtype=float)
    g = np.asarray(g, dtype=float)
    m = (t >= 0.0) & (t <= 180.0)
    t = t[m]
    g = g[m]
    if len(t) < 4:
        return {"si": np.nan, "imax": np.nan, "gof": np.nan}

    g0 = float(g[0])
    iss = float(max(fasting_insulin, 1e-3))

    def ode(_t, y, si, imax, k_sto, k_gut, eg0):
        q_sto, q_gut, gg, ii = y
        k_i = imax * (g0 ** 2) / (iss * (alpha + g0 ** 2) + 1e-8)
        r0 = (eg0 + si * iss) * g0
        dq_sto = -k_sto * q_sto
        dq_gut = k_sto * q_sto - k_gut * q_gut
        dgg = r0 - (eg0 + si * ii) * gg + k_gut * q_gut
        dii = imax * (gg ** 2) / (alpha + gg ** 2) - k_i * ii
        return [dq_sto, dq_gut, dgg, dii]

    def loss(x):
        si, imax, k_sto, k_gut, eg0, q0 = np.exp(x)
        y0 = [q0, 0.0, g0, iss]
        try:
            sol = solve_ivp(lambda tt, yy: ode(tt, yy, si, imax, k_sto, k_gut, eg0), [t[0], t[-1]], y0, t_eval=t, method="RK45")
            if not sol.success:
                return 1e9
            gp = sol.y[2]
            return float(np.mean((gp - g) ** 2))
        except Exception:
            return 1e9

    x0 = np.log([1e-4, 0.5, 0.05, 0.05, 0.01, 30.0])
    lb = np.log([1e-6, 1e-3, 1e-3, 1e-3, 1e-4, 1.0])
    ub = np.log([1e-1, 50.0, 1.0, 1.0, 1.0, 500.0])
    r = minimize(loss, x0=x0, bounds=list(zip(lb, ub)), method="L-BFGS-B", options={"maxiter": 120})
    if not r.success:
        return {"si": np.nan, "imax": np.nan, "gof": np.nan}
    si, imax, k_sto, k_gut, eg0, q0 = np.exp(r.x)
    y0 = [q0, 0.0, g0, iss]
    sol = solve_ivp(lambda tt, yy: ode(tt, yy, si, imax, k_sto, k_gut, eg0), [t[0], t[-1]], y0, t_eval=t, method="RK45")
    if not sol.success:
        return {"si": np.nan, "imax": np.nan, "gof": np.nan}
    gp = sol.y[2]
    ss_res = float(np.sum((g - gp) ** 2))
    ss_tot = float(np.sum((g - np.mean(g)) ** 2))
    gof = 1.0 - ss_res / max(ss_tot, 1e-8)
    return {"si": float(si), "imax": float(imax), "k_sto": float(k_sto), "k_gut": float(k_gut), "eg0": float(eg0), "gof": float(gof)}


def _run_baseline_ml(train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: List[str], out_dir: str, tag: str) -> Dict[str, Any]:
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut

    os.makedirs(out_dir, exist_ok=True)
    tr = train_df.dropna(subset=["sspg"] + feat_cols).copy()
    te = test_df.dropna(subset=["sspg"] + feat_cols).copy()
    Xtr = tr[feat_cols].to_numpy(dtype=float)
    ytr = tr["sspg"].to_numpy(dtype=float)
    Xte = te[feat_cols].to_numpy(dtype=float)
    yte = te["sspg"].to_numpy(dtype=float)

    models = {
        "RidgeCV": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))),
        "RF": RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42),
        "GBR": GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.05, random_state=42),
    }

    model_rows = []
    loo = LeaveOneOut()
    best_name = None
    best_cv = -1e9
    for name, mdl in models.items():
        oof = np.zeros_like(ytr, dtype=float)
        for tr_idx, va_idx in loo.split(Xtr):
            m = clone(mdl)
            m.fit(Xtr[tr_idx], ytr[tr_idx])
            oof[va_idx] = m.predict(Xtr[va_idx])
        cv_metrics = _metrics(ytr, oof)
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte)
        test_metrics = _metrics(yte, pred)
        if np.isfinite(cv_metrics.get("spearman_r", np.nan)) and cv_metrics["spearman_r"] > best_cv:
            best_cv = cv_metrics["spearman_r"]
            best_name = name
        model_rows.append({
            "model": name,
            "cv_spearman": cv_metrics.get("spearman_r"),
            "cv_pearson": cv_metrics.get("pearson_r"),
            "test_pearson": test_metrics.get("pearson_r"),
            "test_spearman": test_metrics.get("spearman_r"),
            "test_r2": test_metrics.get("r2"),
            "test_rmse": test_metrics.get("rmse"),
        })
        if name == "RidgeCV":
            pd.DataFrame({"subject_id": te["subject_id"].values, "true": yte, "pred": pred}).to_csv(
                os.path.join(out_dir, f"{tag}_true_vs_pred.csv"), index=False
            )

    model_df = pd.DataFrame(model_rows).sort_values("cv_spearman", ascending=False)
    model_df.to_csv(os.path.join(out_dir, f"{tag}_model_comparison.csv"), index=False)
    best = model_df.iloc[0].to_dict() if len(model_df) else {}
    with open(os.path.join(out_dir, f"{tag}_best_metrics.json"), "w") as f:
        json.dump(best, f, indent=2, default=str)
    return best


# ============================================================================
# Experiment Definitions
# ============================================================================

def run_exp1(out: str) -> Dict:
    """Exp1: Wang Baseline — 10D ODE, narrow bounds, single-phase joint training."""
    name = "v18_Exp1_Wang_Baseline"
    d = os.path.join(out, name)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    env = {
        "P1_V8_HEAD_10D": "1",       # 只用 10D
        "P1_V10_WIDE_BOUNDS": "0",   # 窄边界
    }
    ckpt = _run_pipeline(os.path.join(d, "joint_training"), env,
                         lambda_sspg=0.1, lambda_di=0.1, num_epochs=100)
    return {"name": name, "dir": d, "ckpt": ckpt, "type": "nn"}


def run_exp2(out: str) -> Dict:
    """Exp2: Metwally 14-Feature Baseline — 传统 ML。"""
    name = "v18_Exp2_Metwally_14Feature"
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    train_df, test_df = _collect_subject_curves()
    feat_cols = ["G_0", "G_60", "G_120", "G_180", "G_Peak", "CurveSize", "AUC",
                 "pAUC", "nAUC", "iAUC", "CV", "T_baseline2peak",
                 "S_baseline2peak", "S_peak2end"]
    best = _run_baseline_ml(train_df, test_df, feat_cols, d, "metwally14")
    baseline_metrics = {
        "d4_sspg_pearson_r": best.get("test_pearson"),
        "d4_sspg_spearman_r": best.get("test_spearman"),
        "d4_sspg_r2": best.get("test_r2"),
        "d4_sspg_rmse": best.get("test_rmse"),
    }
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump({"status": "completed", "best_model": best.get("model"), "metrics": baseline_metrics}, f, indent=2)
    return {"name": name, "dir": d, "ckpt": None, "type": "traditional_ml", "baseline_metrics": baseline_metrics}


def run_exp3(out: str) -> Dict:
    """Exp3: Healey ODE + 空腹胰岛素 Baseline。"""
    name = "v18_Exp3_Healey_ODE"
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    train_df, test_df = _collect_subject_curves()
    # Fit Healey-style ODE parameters per subject curve.
    fit_rows_train = []
    fit_rows_test = []
    for _, r in train_df.dropna(subset=["fasting_insulin", "sspg"]).iterrows():
        pars = _fit_healey_params(
            g=np.array([r["G_0"], r["G_60"], r["G_120"], r["G_180"]], dtype=float),
            t=np.array([0.0, 60.0, 120.0, 180.0], dtype=float),
            fasting_insulin=float(r["fasting_insulin"]),
        )
        fit_rows_train.append({"subject_id": r["subject_id"], "sspg": r["sspg"], "fasting_insulin": r["fasting_insulin"], **pars})
    for _, r in test_df.dropna(subset=["fasting_insulin", "sspg"]).iterrows():
        pars = _fit_healey_params(
            g=np.array([r["G_0"], r["G_60"], r["G_120"], r["G_180"]], dtype=float),
            t=np.array([0.0, 60.0, 120.0, 180.0], dtype=float),
            fasting_insulin=float(r["fasting_insulin"]),
        )
        fit_rows_test.append({"subject_id": r["subject_id"], "sspg": r["sspg"], "fasting_insulin": r["fasting_insulin"], **pars})
    tr = pd.DataFrame(fit_rows_train).dropna(subset=["si", "imax", "fasting_insulin", "sspg"])
    te = pd.DataFrame(fit_rows_test).dropna(subset=["si", "imax", "fasting_insulin", "sspg"])
    if len(tr) < 8 or len(te) < 5:
        # Fallback proxy features when ODE fitting fails on too many subjects.
        tr = train_df.dropna(subset=["sspg", "fasting_insulin"]).copy()
        te = test_df.dropna(subset=["sspg", "fasting_insulin"]).copy()
        tr["si"] = 1.0 / (np.abs(tr["iAUC"]) * np.maximum(tr["fasting_insulin"], 1e-3) + 1e-6)
        te["si"] = 1.0 / (np.abs(te["iAUC"]) * np.maximum(te["fasting_insulin"], 1e-3) + 1e-6)
        tr["imax"] = np.maximum(tr["G_Peak"] - tr["G_0"], 0.0) / np.maximum(tr["fasting_insulin"], 1e-3)
        te["imax"] = np.maximum(te["G_Peak"] - te["G_0"], 0.0) / np.maximum(te["fasting_insulin"], 1e-3)
        tr["gof"] = np.nan
        te["gof"] = np.nan
    feat_cols = ["si", "imax", "fasting_insulin"]
    best = _run_baseline_ml(tr.rename(columns={"sspg": "sspg"}), te.rename(columns={"sspg": "sspg"}), feat_cols, d, "healey")
    baseline_metrics = {
        "d4_sspg_pearson_r": best.get("test_pearson"),
        "d4_sspg_spearman_r": best.get("test_spearman"),
        "d4_sspg_r2": best.get("test_r2"),
        "d4_sspg_rmse": best.get("test_rmse"),
    }
    tr.to_csv(os.path.join(d, "healey_train_features.csv"), index=False)
    te.to_csv(os.path.join(d, "healey_test_features.csv"), index=False)
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump({"status": "completed", "best_model": best.get("model"), "metrics": baseline_metrics}, f, indent=2)
    return {"name": name, "dir": d, "ckpt": None, "type": "ode_fitting", "baseline_metrics": baseline_metrics}


def run_exp4(out: str) -> Dict:
    """Exp4: Simple Stats Baseline — CGM 基础统计量 + Ridge。"""
    name = "v18_Exp4_Simple_Stats"
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    train_df, test_df = _collect_subject_curves()
    feat_cols = ["mean", "std", "CV", "min", "max", "range", "AUC"]
    best = _run_baseline_ml(train_df, test_df, feat_cols, d, "simple_stats")
    baseline_metrics = {
        "d4_sspg_pearson_r": best.get("test_pearson"),
        "d4_sspg_spearman_r": best.get("test_spearman"),
        "d4_sspg_r2": best.get("test_r2"),
        "d4_sspg_rmse": best.get("test_rmse"),
    }
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump({"status": "completed", "best_model": best.get("model"), "metrics": baseline_metrics}, f, indent=2)
    return {"name": name, "dir": d, "ckpt": None, "type": "simple_stats", "baseline_metrics": baseline_metrics}


def run_exp5(out: str) -> Dict:
    """Exp5: GV v17 Baseline — 26D, wide bounds, two-phase, joint head, 100 epochs, no ES。"""
    name = "v18_Exp5_GV_Baseline"
    d = os.path.join(out, name)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    env = {}  # 全部默认
    ckpt = _two_phase(d, extra_env_p1=env, extra_env_p2=env,
                      epochs_p1=100, epochs_p2=100)
    return {"name": name, "dir": d, "ckpt": ckpt, "type": "nn"}


def run_exp6(out: str) -> Dict:
    """Exp6: +Separate Small Heads — 26D→16→1 per target, 449 params/head。"""
    name = "v18_Exp6_Separate_Heads"
    d = os.path.join(out, name)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    env_p2 = {"V18_SEPARATE_SMALL_HEAD": "1"}
    ckpt = _two_phase(d, extra_env_p1={}, extra_env_p2=env_p2,
                      epochs_p1=100, epochs_p2=100)
    return {"name": name, "dir": d, "ckpt": ckpt, "type": "nn"}


def run_exp7(out: str) -> Dict:
    """Exp7: +Early Stopping — patience=15, 200 epochs budget。"""
    name = "v18_Exp7_EarlyStop"
    d = os.path.join(out, name)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    env_p2 = {
        "V18_SEPARATE_SMALL_HEAD": "1",
        "V18_EARLY_STOPPING_PATIENCE": "15",
    }
    ckpt = _two_phase(d, extra_env_p1={}, extra_env_p2=env_p2,
                      epochs_p1=100, epochs_p2=200)
    return {"name": name, "dir": d, "ckpt": ckpt, "type": "nn"}


def run_exp8(out: str) -> Dict:
    """Exp8: +CorrLoss — 50% MSE + 50% Pearson correlation loss。"""
    name = "v18_Exp8_CorrLoss"
    d = os.path.join(out, name)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    env_p2 = {
        "V18_SEPARATE_SMALL_HEAD": "1",
        "V18_EARLY_STOPPING_PATIENCE": "15",
        "V18_CORR_LOSS_ALPHA": "0.5",
    }
    ckpt = _two_phase(d, extra_env_p1={}, extra_env_p2=env_p2,
                      epochs_p1=100, epochs_p2=200)
    return {"name": name, "dir": d, "ckpt": ckpt, "type": "nn"}


def run_exp9(out: str) -> Dict:
    """Exp9: Full Combo — 所有 v18 改进 + Phase 1 正交约束。"""
    name = "v18_Exp9_Full_Combo"
    d = os.path.join(out, name)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    env_p1 = {"V18_LAMBDA_ORTHO_P1": "0.01"}
    env_p2 = {
        "V18_SEPARATE_SMALL_HEAD": "1",
        "V18_EARLY_STOPPING_PATIENCE": "15",
        "V18_CORR_LOSS_ALPHA": "0.5",
    }
    ckpt = _two_phase(d, extra_env_p1=env_p1, extra_env_p2=env_p2,
                      epochs_p1=100, epochs_p2=200)
    return {"name": name, "dir": d, "ckpt": ckpt, "type": "nn"}


def run_exp10(out: str, exp5_dir: str) -> Dict:
    """Exp10: Ridge Probe — 用 Exp5 的 26D latent 做 Ridge LOO-CV。"""
    name = "v18_Exp10_Ridge_Probe"
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    # 从 Exp5 的 Phase 1 找 latent 文件
    p1_dir = os.path.join(exp5_dir, "phase1_unsupervised")
    latent_path = None
    for fname in ["latent_and_gold_all_26d.csv", "latent_and_gold_all.csv", "latent_all_26d.csv"]:
        p = os.path.join(p1_dir, fname)
        if os.path.isfile(p):
            latent_path = p
            break

    ridge_result = {}
    if latent_path:
        try:
            import pandas as pd
            from sklearn.linear_model import RidgeCV
            from sklearn.model_selection import LeaveOneOut
            from scipy.stats import spearmanr, pearsonr

            df = pd.read_csv(latent_path)
            feat_cols = [c for c in df.columns
                        if c.startswith(('tau_m', 'Gb', 'sg', 'si', 'p2', 'mi', 'z_init_', 'z_nonseq_'))]
            if not feat_cols:
                feat_cols = [c for c in df.columns if c not in ['subject_id', 'sspg', 'di', 'dataset_id']]

            for target in ['sspg', 'di']:
                if target not in df.columns:
                    continue
                valid = df[target].notna()
                if valid.sum() < 5:
                    continue
                X = df.loc[valid, feat_cols].values
                y = df.loc[valid, target].values

                loo = LeaveOneOut()
                preds = np.zeros(len(y))
                for train_idx, test_idx in loo.split(X):
                    ridge = RidgeCV(alphas=np.logspace(-3, 4, 50))
                    ridge.fit(X[train_idx], y[train_idx])
                    preds[test_idx] = ridge.predict(X[test_idx])

                r_p, _ = pearsonr(y, preds)
                r_s, _ = spearmanr(y, preds)
                ss_res = np.sum((y - preds) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot

                ridge_result[target] = {
                    'pearson_r': round(float(r_p), 4),
                    'spearman_r': round(float(r_s), 4),
                    'r2': round(float(r2), 4),
                    'pred_std': round(float(np.std(preds)), 2),
                    'true_std': round(float(np.std(y)), 2),
                    'n': int(valid.sum()),
                }
                print(f"  [Ridge {target}] r={r_p:.4f} ρ={r_s:.4f} R²={r2:.4f} "
                      f"pred_std={np.std(preds):.2f} true_std={np.std(y):.2f}")
        except Exception as e:
            print(f"  [WARN] Ridge probe failed: {e}")
    else:
        print(f"  [WARN] No latent file found in {p1_dir}")

    with open(os.path.join(d, "ridge_probe_results.json"), "w") as f:
        json.dump(ridge_result, f, indent=2)
    return {"name": name, "dir": d, "ckpt": None, "type": "ridge",
            "ridge_result": ridge_result}


# ============================================================================
# Post-hoc: 对所有 NN 实验运行 D4 eval + per-meal + SHAP + auROC
# ============================================================================
def run_posthoc(experiments: List[Dict], out: str):
    results = []
    for exp in experiments:
        name = exp["name"]
        print(f"\n--- Post-hoc: {name} ---")
        r = {"experiment": name, "type": exp["type"]}

        if exp["type"] != "nn" or exp.get("ckpt") is None:
            if "baseline_metrics" in exp:
                r.update(exp["baseline_metrics"])
            r.update(exp.get("ridge_result", {}))
            results.append(r)
            continue

        ckpt = exp["ckpt"]
        eval_dir = os.path.join(exp["dir"], "eval_D4")

        # D4 metrics
        d4 = _eval_d4(ckpt, eval_dir)
        r.update(d4)

        # Pred stats
        ps = _pred_stats(eval_dir)
        r.update(ps)

        # auROC
        auc = _auroc(eval_dir)
        r.update(auc)

        # Per-meal-type
        meals = _per_meal(ckpt, exp["dir"])
        r["per_meal_type"] = meals

        # SHAP (仅 Exp5 和 Exp9)
        if "Exp5" in name or "Exp9" in name:
            shap_dir = os.path.join(exp["dir"], "shap_analysis")
            shap_r = _shap(ckpt, shap_dir)
            r.update(shap_r)

        results.append(r)

    # 保存
    with open(os.path.join(out, "v18_comprehensive_summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 生成报告
    _report(results, out)
    return results


def _report(results: List[Dict], out: str):
    rp = os.path.join(out, "v18_report.md")
    with open(rp, "w") as f:
        f.write(f"# GlucoVector v18 Results\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## D4 Performance\n\n")
        f.write("| Exp | SSPG r | SSPG ρ | SSPG R² | RMSE | pred_std | compress | auROC | DI r | DI R² |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in results:
            def fmt(k, d=4):
                v = r.get(k)
                return f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"
            f.write(f"| {r['experiment']} | {fmt('d4_sspg_pearson_r')} | {fmt('d4_sspg_spearman_r')} "
                    f"| {fmt('d4_sspg_r2')} | {fmt('d4_sspg_rmse',1)} | {fmt('sspg_pred_std',1)} "
                    f"| {fmt('sspg_compression',3)} | {fmt('auroc')} | {fmt('d4_di_pearson_r')} "
                    f"| {fmt('d4_di_r2')} |\n")
        f.write("\n## Prediction Compression\n\n")
        f.write("| Exp | pred_std | true_std | compression | pred_range |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|\n")
        for r in results:
            def fmt(k, d=1):
                v = r.get(k)
                return f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"
            f.write(f"| {r['experiment']} | {fmt('sspg_pred_std')} | {fmt('sspg_true_std')} "
                    f"| {fmt('sspg_compression',3)} | {fmt('sspg_pred_range')} |\n")
    print(f"[SAVED] {rp}")


# ============================================================================
# Main
# ============================================================================
def main():
    print(f"{'='*60}")
    print(f"  GlucoVector v18 Experiment Runner (FINAL)")
    print(f"{'='*60}")
    print(f"  Output: {OUT_ROOT}")
    print(f"  Data:   {DATA_ROOT}")
    print(f"  Repo:   {REPO_ROOT}")
    print(f"  Seed:   {SEED}")
    print(f"  Time:   {datetime.now().isoformat()}")
    print()

    _check_patches()
    os.makedirs(str(OUT_ROOT), exist_ok=True)

    exps = []

    # === Baselines ===
    print("\n" + "="*60 + "\n  BASELINES\n" + "="*60)
    exps.append(run_exp1(str(OUT_ROOT)))    # Wang
    exps.append(run_exp2(str(OUT_ROOT)))    # Metwally (placeholder)
    exps.append(run_exp3(str(OUT_ROOT)))    # Healey (placeholder)
    exps.append(run_exp4(str(OUT_ROOT)))    # Simple Stats (placeholder)

    # === GlucoVector Upgrades ===
    print("\n" + "="*60 + "\n  GLUCOVECTOR UPGRADES\n" + "="*60)
    e5 = run_exp5(str(OUT_ROOT))            # GV Baseline
    exps.append(e5)
    exps.append(run_exp6(str(OUT_ROOT)))    # +Separate Heads
    exps.append(run_exp7(str(OUT_ROOT)))    # +Early Stopping
    exps.append(run_exp8(str(OUT_ROOT)))    # +CorrLoss
    exps.append(run_exp9(str(OUT_ROOT)))    # Full Combo

    # === Diagnostic ===
    print("\n" + "="*60 + "\n  DIAGNOSTIC\n" + "="*60)
    exps.append(run_exp10(str(OUT_ROOT), e5["dir"]))  # Ridge Probe

    # === Post-hoc ===
    print("\n" + "="*60 + "\n  POST-HOC ANALYSIS\n" + "="*60)
    run_posthoc(exps, str(OUT_ROOT))

    print(f"\n{'='*60}")
    print(f"  v18 complete! Results: {OUT_ROOT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
