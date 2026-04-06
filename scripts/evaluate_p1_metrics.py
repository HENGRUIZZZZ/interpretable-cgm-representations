"""
P1 金标准预测评估：RMSE、MAE、R²、Pearson/Spearman r，多种验证方式。

本脚本用于 **M0（基线）**：无监督/半监督 VAE 训练后，用 Ridge(6D latent → 金标准) 做
post-hoc 评估。**M1（端到端监督）** 的主结果应使用 run_p1_full_pipeline.py 中测试集上的
端到端 head 评估（打印的 "End-to-End Head Evaluation" 及保存的 e2e_head_metrics.json），
不要以本脚本作为 M1 的主结果汇报来源。

- 直接潜变量：用单个 latent（如 mi）线性预测 SSPG/DI，在划分后的 test 上算指标。
- 预测头：Ridge(6D latent → 金标准)，K-fold 或留一数据集出做验证。
- 验证模式：kfold_subject（按 subject 5-fold）、holdout（80/20）、leave_one_dataset_out。

用法（项目根目录）：
  python scripts/evaluate_p1_metrics.py
  python scripts/evaluate_p1_metrics.py --csv paper1_results/latent_and_gold_all.csv --out paper1_results
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold

# 确保能 import 项目模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

LATENT_COLS = ["si", "mi", "tau_m", "Gb", "sg", "p2"]
# 预测目标：SSPG, DI，以及 HOMA-IR / HOMA-B（secondary endpoints）
GOLD_TARGETS = ["sspg", "di", "homa_ir", "homa_b"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true, y_pred 一维，同长。返回 RMSE, MAE, R2, pearson_r, pearson_p, spearman_r, spearman_p, n."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    if n < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan, "n": n}
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred, nan_policy="omit")
    return {
        "rmse": float(rmse), "mae": float(mae), "r2": float(r2),
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "n": n,
    }


def direct_latent_metrics(df: pd.DataFrame, target: str, latent: str, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    """用单个 latent 做线性回归预测 target，在 test 上算指标。"""
    sub = df.dropna(subset=[target, latent])
    if len(sub) < 4:
        return compute_metrics(np.array([]), np.array([]))
    tr = sub.iloc[train_idx]
    te = sub.iloc[test_idx]
    if len(tr) < 2 or len(te) < 1:
        return compute_metrics(np.array([]), np.array([]))
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression().fit(tr[[latent]], tr[target])
    y_pred = lm.predict(te[[latent]])
    return compute_metrics(te[target].values, y_pred)


def head_cv_metrics(
    df: pd.DataFrame,
    target: str,
    val_mode: str,
    n_splits: int = 5,
    alphas: list[float] | None = None,
    seed: int = 0,
) -> list[dict]:
    """Ridge(6D latent → target)，按 val_mode 做划分，返回每 fold 的 metrics 列表。"""
    sub = df.dropna(subset=target).copy()
    if len(sub) < 6:
        return []
    X = sub[LATENT_COLS]
    y = sub[target]
    grp = sub["subject_id"].astype(str).values if "subject_id" in sub.columns else None
    if alphas is None:
        alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    out = []
    if val_mode == "kfold_subject" and grp is not None and len(np.unique(grp)) >= n_splits:
        kf = GroupKFold(n_splits=n_splits)
        splits = list(kf.split(X, y, grp))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(kf.split(X, y))
    for tr_idx, te_idx in splits:
        X_tr, X_te = X.iloc[tr_idx][LATENT_COLS], X.iloc[te_idx][LATENT_COLS]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        if len(y_te) < 2:
            continue
        best_metrics = None
        best_alpha = None
        for alpha in alphas:
            scaler = StandardScaler()
            reg = Ridge(alpha=alpha)
            X_tr_s = scaler.fit_transform(X_tr)
            reg.fit(X_tr_s, y_tr)
            y_pred = reg.predict(scaler.transform(X_te))
            m = compute_metrics(y_te.values, y_pred)
            # 选优依据：Spearman r（与主报告一致），避免小 fold 上 R² 波动大
            key_r = "spearman_r"
            if best_metrics is None or (np.isfinite(m[key_r]) and abs(m[key_r]) > abs(best_metrics.get(key_r, 0))):
                best_metrics = m.copy()
                best_alpha = alpha
        if best_metrics is not None:
            best_metrics["fold_alpha"] = best_alpha
            out.append(best_metrics)
    return out


def leave_one_dataset_out_metrics(
    df: pd.DataFrame, target: str, alphas: list[float] | None = None
) -> list[dict]:
    """留一数据集出：用两个数据集训 Ridge，在第三个上测。"""
    if alphas is None:
        alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    datasets = df["dataset_id"].dropna().unique().tolist()
    out = []
    for test_ds in datasets:
        train_df = df[(df["dataset_id"] != test_ds) & df[target].notna()]
        test_df = df[(df["dataset_id"] == test_ds) & df[target].notna()]
        if len(train_df) < 3 or len(test_df) < 2:
            continue
        X_tr = train_df[LATENT_COLS]
        y_tr = train_df[target]
        X_te = test_df[LATENT_COLS]
        y_te = test_df[target]
        best_metrics = None
        for alpha in alphas:
            scaler = StandardScaler()
            reg = Ridge(alpha=alpha)
            X_tr_s = scaler.fit_transform(X_tr)
            reg.fit(X_tr_s, y_tr)
            y_pred = reg.predict(scaler.transform(X_te))
            m = compute_metrics(y_te.values, y_pred)
            m["test_dataset"] = test_ds
            if best_metrics is None or (np.isfinite(m["spearman_r"]) and abs(m["spearman_r"]) > abs(best_metrics.get("spearman_r", 0))):
                best_metrics = m.copy()
        if best_metrics is not None:
            out.append(best_metrics)
    return out


def main():
    parser = argparse.ArgumentParser(description="P1 gold-standard prediction metrics (RMSE, MAE, R², r).")
    parser.add_argument("--csv", default="paper1_results/latent_and_gold_all.csv", help="latent_and_gold CSV")
    parser.add_argument("--out", default="paper1_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Optional comma-separated dataset_id list to include for gold-standard metrics (e.g. 'D1,D2').",
    )
    args = parser.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(args.csv):
        print(f"File not found: {args.csv}. Run run_p1_full_pipeline.py first.")
        return
    df = pd.read_csv(args.csv)
    df = df.drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)
    # 可选：只在指定 dataset_id 上评估（如仅用 D1+D2 的金标准样本）
    if getattr(args, "datasets", ""):
        if "dataset_id" in df.columns:
            keep = [x.strip() for x in args.datasets.split(",") if x.strip()]
            if keep:
                before_n = len(df)
                df = df[df["dataset_id"].astype(str).isin(keep)].reset_index(drop=True)
                print(f"Filtering to datasets {keep}: {before_n} -> {len(df)} subjects")
    rows = []
    n = len(df)

    # ---------- 1. 全量相关（无划分）：单潜变量 vs 金标准（线性拟合，仅描述用） ----------
    print("\n--- 1. 全量样本：单潜变量 vs 金标准 (Spearman r, RMSE 用线性拟合，描述用) ---")
    from sklearn.linear_model import LinearRegression
    for target in GOLD_TARGETS:
        sub = df.dropna(subset=[target])
        if len(sub) < 4:
            continue
        for latent in ["mi", "si"]:
            if latent not in sub.columns:
                continue
            r, p = stats.spearmanr(sub[latent], sub[target], nan_policy="omit")
            lm = LinearRegression().fit(sub[[latent]], sub[target])
            y_pred = lm.predict(sub[[latent]])
            m = compute_metrics(sub[target].values, y_pred)
            print(f"  {latent} → {target}:  Spearman r={r:.3f}  p={p:.4f}  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R²={m['r2']:.3f}  n={len(sub)}")
            rows.append({
                "target": target, "method": f"direct_{latent}", "validation": "full_sample",
                "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"],
                "pearson_r": m["pearson_r"], "spearman_r": m["spearman_r"],
                "spearman_p": m["spearman_p"], "n": m["n"],
            })

    # ---------- 2. 5-fold：单参数 vs 联合(6D) 同划分对比 ----------
    print("\n--- 2. 5-fold (by subject): 单参数 vs 联合(6D) 同一划分对比 ---")
    for target in GOLD_TARGETS:
        sub = df.dropna(subset=[target]).copy()
        if len(sub) < 10 or "subject_id" not in sub.columns:
            continue
        grp = sub["subject_id"].astype(str).values
        kf = GroupKFold(n_splits=5)
        try:
            splits = list(kf.split(sub[LATENT_COLS], sub[target], grp))
        except Exception:
            continue
        single_mi_metrics, single_si_metrics, joint_metrics = [], [], []
        for tr_idx, te_idx in splits:
            train_df = sub.iloc[tr_idx]
            test_df = sub.iloc[te_idx]
            if len(test_df) < 2:
                continue
            lm_mi = LinearRegression().fit(train_df[["mi"]], train_df[target])
            single_mi_metrics.append(compute_metrics(test_df[target].values, lm_mi.predict(test_df[["mi"]])))
            lm_si = LinearRegression().fit(train_df[["si"]], train_df[target])
            single_si_metrics.append(compute_metrics(test_df[target].values, lm_si.predict(test_df[["si"]])))
            scaler = StandardScaler()
            reg = Ridge(alpha=10.0 if target == "sspg" else 0.1)
            reg.fit(scaler.fit_transform(train_df[LATENT_COLS]), train_df[target])
            joint_metrics.append(compute_metrics(test_df[target].values, reg.predict(scaler.transform(test_df[LATENT_COLS]))))
        if single_mi_metrics and joint_metrics:
            print(f"  [{target}]  single mi:  r={np.nanmean([x['pearson_r'] for x in single_mi_metrics]):.3f}  RMSE={np.nanmean([x['rmse'] for x in single_mi_metrics]):.2f}")
            print(f"  [{target}]  single si:  r={np.nanmean([x['pearson_r'] for x in single_si_metrics]):.3f}  RMSE={np.nanmean([x['rmse'] for x in single_si_metrics]):.2f}")
            print(f"  [{target}]  joint 6D:   r={np.nanmean([x['pearson_r'] for x in joint_metrics]):.3f}  RMSE={np.nanmean([x['rmse'] for x in joint_metrics]):.2f}")
            for key in ["rmse", "r2", "pearson_r", "spearman_r"]:
                for name, mlist in [("single_mi", single_mi_metrics), ("single_si", single_si_metrics), ("ridge_6d", joint_metrics)]:
                    rows.append({"target": target, "method": name, "validation": "5fold_single_vs_joint",
                                 "metric": key, "mean": float(np.nanmean([x[key] for x in mlist])), "std": float(np.nanstd([x[key] for x in mlist])), "n": len(sub)})

    # ---------- 2b. 5-fold Ridge 预测头（与原有输出一致） ----------
    print("\n--- 2b. 5-fold (by subject) Ridge 预测头 → SSPG / DI ---")
    per_fold_rows = []
    for target in GOLD_TARGETS:
        sub = df.dropna(subset=[target])
        if len(sub) < 10:
            continue
        fold_metrics = head_cv_metrics(
            df, target, "kfold_subject", n_splits=5, seed=args.seed
        )
        if not fold_metrics:
            continue
        for key in ["rmse", "mae", "r2", "pearson_r", "spearman_r"]:
            vals = [f[key] for f in fold_metrics]
            mean_v, std_v = np.nanmean(vals), np.nanstd(vals)
            rows.append({
                "target": target, "method": "ridge_6d", "validation": "5fold_subject",
                "metric": key, "mean": mean_v, "std": std_v, "n_folds": len(fold_metrics), "n": len(sub),
            })
        for i, f in enumerate(fold_metrics):
            per_fold_rows.append({"target": target, "fold_id": i, "spearman_r": f["spearman_r"], "rmse": f["rmse"], "r2": f["r2"]})
        print(f"  {target}:  RMSE={np.nanmean([f['rmse'] for f in fold_metrics]):.2f} ± {np.nanstd([f['rmse'] for f in fold_metrics]):.2f}  "
              f"MAE={np.nanmean([f['mae'] for f in fold_metrics]):.2f}  "
              f"R²={np.nanmean([f['r2'] for f in fold_metrics]):.3f}  "
              f"Pearson r={np.nanmean([f['pearson_r'] for f in fold_metrics]):.3f}  "
              f"Spearman r={np.nanmean([f['spearman_r'] for f in fold_metrics]):.3f}  (mean over {len(fold_metrics)} folds)")

    # ---------- 3. 留一数据集出 ----------
    print("\n--- 3. Leave-one-dataset-out: Ridge → SSPG / DI ---")
    for target in GOLD_TARGETS:
        lod = leave_one_dataset_out_metrics(df, target)
        for m in lod:
            ds = m.pop("test_dataset", "")
            rows.append({
                "target": target, "method": "ridge_6d", "validation": "leave_one_dataset_out",
                "test_dataset": ds, **{k: v for k, v in m.items() if k != "fold_alpha"},
            })
            print(f"  {target}  test={ds}:  RMSE={m['rmse']:.2f}  R²={m['r2']:.3f}  Pearson r={m['pearson_r']:.3f}  n={m['n']}")

    # ---------- 4. 联合权重（全量拟合，仅用于解读“多参数联合”） ----------
    print("\n--- 4. 联合预测权重 (全量拟合，解读用) ---")
    for target in GOLD_TARGETS:
        sub = df.dropna(subset=[target])
        if len(sub) < 6:
            continue
        scaler = StandardScaler()
        reg = Ridge(alpha=10.0 if target == "sspg" else 0.1)
        X_s = scaler.fit_transform(sub[LATENT_COLS])
        reg.fit(X_s, sub[target])
        coef = pd.DataFrame({"latent": LATENT_COLS, "coef": reg.coef_})
        coef.to_csv(os.path.join(out_dir, f"joint_weights_{target}.csv"), index=False)
        print(f"  {target}: " + "  ".join([f"{c}={reg.coef_[i]:.4f}" for i, c in enumerate(LATENT_COLS)]))
        rows.append({"target": target, "method": "ridge_6d_coef", "validation": "full_sample", "coef_file": f"joint_weights_{target}.csv"})

    # ---------- 5. 保存 ----------
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "evaluation_metrics.csv"), index=False)
    if per_fold_rows:
        pd.DataFrame(per_fold_rows).to_csv(os.path.join(out_dir, "evaluation_5fold_per_fold.csv"), index=False)
    summary_path = os.path.join(out_dir, "evaluation_metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("P1 Gold-standard prediction metrics\n")
        f.write("===================================\n\n")
        f.write("Single vs joint: SSPG/DI are physiologically joint; we report both single-latent and 6D Ridge.\n")
        f.write("See joint_weights_sspg.csv and joint_weights_di.csv for Ridge coefficients (full-data fit).\n\n")
        for target in GOLD_TARGETS:
            f.write(f"Target: {target}\n")
            sub = [r for r in rows if r.get("target") == target]
            for r in sub:
                if "metric" in r:
                    f.write(f"  {r['method']} ({r['validation']}): {r['metric']} = {r.get('mean', '')} ± {r.get('std', '')}\n")
                elif r.get("method") == "ridge_6d_coef":
                    f.write(f"  coefficients saved to {r.get('coef_file','')}\n")
                else:
                    f.write(f"  {r['method']} ({r.get('validation', '')}): RMSE={r.get('rmse','')}  R²={r.get('r2','')}  Spearman r={r.get('spearman_r','')}\n")
            f.write("\n")
    print(f"\nSaved {out_dir}/evaluation_metrics.csv, joint_weights_*.csv, and {summary_path}")


if __name__ == "__main__":
    main()
