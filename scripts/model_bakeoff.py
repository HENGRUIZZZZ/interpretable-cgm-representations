"""
实验方案 v4.0 阶段二：三路模型对决，5-fold CV，主指标 Spearman r (SSPG / DI)。

路线 A：6D ODE → PCA(2) → Ridge 预测 SSPG/DI；并报 IR_proxy(si) vs SSPG、BCF_proxy(mi) vs DI。
路线 B：全量 ~36D 特征 → XGBoost (GridSearchCV) 预测 SSPG/DI。
路线 C：仅 CGM 统计 → Ridge 预测 SSPG/DI。

用法：
  python scripts/model_bakeoff.py --full_features paper1_results_v4/baseline_seed42/full_features_v4.csv \\
       --output_dir paper1_results_v4/baseline_seed42 --n_folds 5
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ODE_COLS = ["tau_m", "Gb", "sg", "si", "p2", "mi"]
CGM_STAT_COLS = [
    "cgm_mean", "cgm_std", "cgm_cv", "cgm_min", "cgm_max", "cgm_range",
    "tir", "tar", "tbr", "auc", "ac_var", "mge",
]
TARGETS = ["sspg", "di"]
SKIP_COLS = {"subject_id", "sample_ix", "sspg", "di", "homa_ir", "homa_b", "dataset_id"}


def route_a_pca_ridge(df: pd.DataFrame, target: str, n_folds: int, seed: int):
    """Route A: 6D ODE → PCA(2) → Ridge. 返回 (spearman_r, p_value, y_true, y_pred)."""
    X = df[ODE_COLS].copy()
    for c in ODE_COLS:
        if c not in df.columns:
            return np.nan, np.nan, None, None
    y = df[target].values
    valid = np.isfinite(y)
    X, y = X.values[valid], y[valid]
    if X.shape[0] < 10:
        return np.nan, np.nan, None, None

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    valid_ix = np.where(valid)[0]
    preds = np.full(len(df), np.nan)

    for train_idx, test_idx in kf.split(X):
        Xt, Xv = X[train_idx], X[test_idx]
        yt = y[train_idx]
        Xt_s = scaler.fit_transform(Xt)
        Xv_s = scaler.transform(Xv)
        pca.fit(Xt_s)
        Xv_pc = pca.transform(Xv_s)
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=3).fit(pca.transform(Xt_s), yt)
        preds[valid_ix[test_idx]] = model.predict(Xv_pc)

    y_true = y
    y_pred = preds[valid]
    r, p = stats.spearmanr(y_true, y_pred, nan_policy="omit")
    return float(r) if not np.isnan(r) else np.nan, float(p) if not np.isnan(p) else np.nan, y_true, y_pred


def route_a_proxy(df: pd.DataFrame, n_folds: int):
    """IR_proxy = 1/si vs SSPG; BCF_proxy = mi vs DI (Spearman on full data)."""
    out = {}
    if "si" in df.columns and "sspg" in df.columns:
        sub = df[["si", "sspg"]].dropna()
        if len(sub) > 5:
            ir_proxy = 1.0 / (sub["si"].values + 1e-12)
            r, p = stats.spearmanr(ir_proxy, sub["sspg"].values)
            out["IR_proxy_vs_SSPG_spearman_r"] = float(r)
            out["IR_proxy_vs_SSPG_p"] = float(p)
    if "mi" in df.columns and "di" in df.columns:
        sub = df[["mi", "di"]].dropna()
        if len(sub) > 5:
            r, p = stats.spearmanr(sub["mi"].values, sub["di"].values)
            out["BCF_proxy_vs_DI_spearman_r"] = float(r)
            out["BCF_proxy_vs_DI_p"] = float(p)
    return out


def route_b_xgboost(df: pd.DataFrame, target: str, n_folds: int, seed: int):
    """Route B: 全量特征 XGBoost，5-fold CV 预测。小样本时用保守参数 + 轻量 GridSearch。"""
    try:
        import xgboost as xgb
    except ImportError:
        return np.nan, np.nan, None, None

    feat_cols = [c for c in df.columns if c not in SKIP_COLS and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]
    if not feat_cols:
        return np.nan, np.nan, None, None
    X_df = df[feat_cols].copy().fillna(df[feat_cols].median())
    # 去掉常数列，避免 XGB 退化
    var = X_df.var()
    nonconst = [c for c in X_df.columns if var.get(c, 0) > 1e-10]
    if not nonconst:
        return np.nan, np.nan, None, None
    X_df = X_df[nonconst]
    X = X_df.values
    y = df[target].values
    valid = np.isfinite(y)
    X, y = X[valid], y[valid]
    if X.shape[0] < 10:
        return np.nan, np.nan, None, None

    n = X.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = np.full(X.shape[0], np.nan)
    scaler = StandardScaler()
    # 小样本时用单模型 XGBoost 避免 GridSearch 选出常数预测；大样本时用 GridSearch
    use_grid = n >= 150
    for train_idx, test_idx in kf.split(X):
        Xt, Xv = X[train_idx], X[test_idx]
        yt = y[train_idx]
        Xt_s = scaler.fit_transform(Xt)
        Xv_s = scaler.transform(Xv)
        if use_grid:
            cv_inner = min(3, max(2, len(yt) // 5))
            from sklearn.model_selection import GridSearchCV
            reg = GridSearchCV(
                xgb.XGBRegressor(
                    objective="reg:squarederror", random_state=seed, n_estimators=100,
                    min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
                ),
                param_grid={"max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1], "reg_alpha": [0.3, 1.0], "reg_lambda": [1.0, 2.0]},
                cv=cv_inner, scoring="neg_mean_squared_error", n_jobs=-1,
            )
            reg.fit(Xt_s, yt)
        else:
            reg = xgb.XGBRegressor(
                objective="reg:squarederror", random_state=seed, n_estimators=100,
                max_depth=3, learning_rate=0.08, reg_alpha=0.5, reg_lambda=1.0,
                min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
            )
            reg.fit(Xt_s, yt)
        preds[test_idx] = reg.predict(Xv_s)
    r, p = stats.spearmanr(y, preds, nan_policy="omit")
    return float(r) if not np.isnan(r) else np.nan, float(p) if not np.isnan(p) else np.nan, y, preds


def route_c_ridge_stats(df: pd.DataFrame, target: str, n_folds: int, seed: int):
    """Route C: 仅 CGM 统计 → Ridge。"""
    cols = [c for c in CGM_STAT_COLS if c in df.columns]
    if not cols:
        return np.nan, np.nan, None, None
    X = df[cols].copy().fillna(df[cols].median())
    y = df[target].values
    valid = np.isfinite(y)
    X, y = X.values[valid], y[valid]
    if X.shape[0] < 10:
        return np.nan, np.nan, None, None

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = np.full(X.shape[0], np.nan)
    for train_idx, test_idx in kf.split(X):
        Xt, Xv = X[train_idx], X[test_idx]
        yt = y[train_idx]
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=3).fit(Xt, yt)
        preds[test_idx] = model.predict(Xv)
    r, p = stats.spearmanr(y, preds, nan_policy="omit")
    return float(r) if not np.isnan(r) else np.nan, float(p) if not np.isnan(p) else np.nan, y, preds


def main():
    parser = argparse.ArgumentParser(description="Model bake-off: Route A/B/C, 5-fold CV, Spearman r")
    parser.add_argument("--full_features", type=str, required=True, help="Path to full_features_v4.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: dir of full_features)")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.full_features):
        print(f"File not found: {args.full_features}")
        sys.exit(1)
    out_dir = args.output_dir or os.path.dirname(args.full_features)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.full_features)
    # 至少需要金标准
    for t in TARGETS:
        if t not in df.columns:
            df[t] = np.nan

    results = {}
    # Route A
    for t in TARGETS:
        r, p, _, _ = route_a_pca_ridge(df, t, args.n_folds, args.seed)
        results[f"RouteA_PCA2_Ridge_{t}_spearman_r"] = r
        results[f"RouteA_PCA2_Ridge_{t}_p"] = p
    results.update(route_a_proxy(df, args.n_folds))

    # Route B
    for t in TARGETS:
        r, p, _, _ = route_b_xgboost(df, t, args.n_folds, args.seed)
        results[f"RouteB_XGBoost_{t}_spearman_r"] = r
        results[f"RouteB_XGBoost_{t}_p"] = p

    # Route C
    for t in TARGETS:
        r, p, _, _ = route_c_ridge_stats(df, t, args.n_folds, args.seed)
        results[f"RouteC_Ridge_stats_{t}_spearman_r"] = r
        results[f"RouteC_Ridge_stats_{t}_p"] = p

    with open(os.path.join(out_dir, "bakeoff_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Bake-off results (Spearman r):")
    for k, v in results.items():
        if "spearman_r" in k:
            print(f"  {k}: {v:.4f}" if np.isfinite(np.asarray(v)) else f"  {k}: {v}")
    if not np.isfinite(results.get("RouteB_XGBoost_sspg_spearman_r", 0)):
        print("  (Route B requires xgboost: pip install xgboost)")
    print(f"Saved {out_dir}/bakeoff_results.json")


if __name__ == "__main__":
    main()
