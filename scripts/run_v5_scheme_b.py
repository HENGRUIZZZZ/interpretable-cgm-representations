"""
实验方案 v5.0 终局之战 — 方案 B：终极简约模型。

特征：cgm_mean, ac_var, cgm_std, mge (Top 4 CGM stats)
模型：Ridge 回归
数据：全部 meal 窗口，按 subject_id 聚合（中位数）后做 100 次随机 CV，得到 Spearman r 分布。

用法（项目根目录）：
  python scripts/run_v5_scheme_b.py --output_dir paper1_results_v5/scheme_b
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# 方案 B 使用全部 meal 窗口，在 import 前设置以免 compute_cgm_stats 读错
os.environ.setdefault("P1_ONE_MEAL_PER_SUBJECT", "0")

OUTPUT_BASE = os.environ.get("CGM_PROJECT_OUTPUT", "/Users/hertz1030/Downloads/cgm_project/output")
TOP4_COLS = ["cgm_mean", "ac_var", "cgm_std", "mge"]


def compute_one_window(y: np.ndarray) -> dict:
    y = np.asarray(y).ravel()
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return {"cgm_mean": np.nan, "cgm_std": np.nan, "ac_var": np.nan, "mge": np.nan}
    n = len(y)
    mean = float(np.mean(y))
    std = float(np.std(y))
    diff = np.diff(y)
    ac_var = float(np.std(diff)) if len(diff) > 0 else np.nan
    baseline = float(np.min(y[: max(1, n // 4)])) if n else np.nan
    mge = float(np.mean(np.maximum(y - baseline, 0)))
    return {"cgm_mean": mean, "cgm_std": std, "ac_var": ac_var, "mge": mge}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="paper1_results_v5/scheme_b")
    parser.add_argument("--n_cv", type=int, default=100)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed_base", type=int, default=2025)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from scripts.compute_cgm_stats import load_and_stack_same_as_pipeline
    batch, pids = load_and_stack_same_as_pipeline(OUTPUT_BASE)

    rows = []
    for i in range(batch.cgm.shape[0]):
        st = compute_one_window(batch.cgm[i])
        st["subject_id"] = pids[i]
        rows.append(st)
    meal_df = pd.DataFrame(rows)

    # 金标准：从 labels 取（需与 pipeline 一致）
    from run_p1_full_pipeline import _stack_batches, GOLD_COLS
    from load_cgm_project_data import load_cgm_project_level1_level2, load_cgm_project_level3
    from paper1_experiment_config import get_data_dir, P1_FULL_TRAIN_DATASETS
    batch_list, info_list, labels_list, dataset_ids = [], [], [], []
    for did in P1_FULL_TRAIN_DATASETS:
        data_dir = get_data_dir(did, OUTPUT_BASE)
        if not os.path.isdir(data_dir):
            continue
        if did in ("D1", "D2"):
            b, info, lab = load_cgm_project_level1_level2(data_dir=data_dir, num_meals_threshold=1)
        else:
            try:
                b, info, lab = load_cgm_project_level3(dataset_id=did, output_base=OUTPUT_BASE)
            except Exception:
                continue
        batch_list.append(b)
        info_list.append(info)
        labels_list.append(lab)
        dataset_ids.append(did)
    _, _, labels_combined = _stack_batches(batch_list, info_list, labels_list, dataset_ids)

    # 按 subject 聚合：特征取中位数
    agg = meal_df.groupby("subject_id")[TOP4_COLS].median().reset_index()
    if labels_combined.empty or "subject_id" not in labels_combined.columns:
        print("No labels; need D1/D2/D4 labels for Scheme B.")
        sys.exit(1)
    # 每个 subject 一条金标准（取第一个 dataset 的 label）
    gold = labels_combined.drop_duplicates(subset=["subject_id"], keep="first")[["subject_id"] + [c for c in GOLD_COLS if c in labels_combined.columns]]
    sub_df = agg.merge(gold, on="subject_id", how="inner")
    sub_df = sub_df.dropna(subset=TOP4_COLS, how="all")
    if "sspg" not in sub_df.columns:
        sub_df["sspg"] = np.nan
    if "di" not in sub_df.columns:
        sub_df["di"] = np.nan
    sub_df = sub_df.dropna(subset=["subject_id"])

    X = sub_df[TOP4_COLS].fillna(sub_df[TOP4_COLS].median()).values
    y_sspg = sub_df["sspg"].values
    y_di = sub_df["di"].values
    n = len(sub_df)
    if n < 10:
        print(f"Too few subjects with gold and features: n={n}")
        sys.exit(1)

    results = []
    for i in range(args.n_cv):
        seed = args.seed_base + i
        train_idx, test_idx = train_test_split(np.arange(n), test_size=args.test_frac, random_state=seed)
        ok_sspg = np.isfinite(y_sspg[train_idx])
        ok_di = np.isfinite(y_di[train_idx])
        if ok_sspg.sum() < 5 or ok_di.sum() < 5:
            results.append({"fold": i, "sspg_spearman_r": np.nan, "di_spearman_r": np.nan})
            continue
        X_tr, X_te = X[train_idx], X[test_idx]
        # SSPG
        tr_sspg = y_sspg[train_idx]
        te_sspg = y_sspg[test_idx]
        ok_tr_s = np.isfinite(tr_sspg)
        if ok_tr_s.sum() < 5:
            r_sspg = np.nan
        else:
            model_s = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=3).fit(X_tr[ok_tr_s], tr_sspg[ok_tr_s])
            pred_s = model_s.predict(X_te)
            ok_te_s = np.isfinite(te_sspg)
            if ok_te_s.sum() > 2:
                r_sspg, _ = stats.spearmanr(pred_s[ok_te_s], te_sspg[ok_te_s])
            else:
                r_sspg = np.nan
        # DI
        tr_di = y_di[train_idx]
        te_di = y_di[test_idx]
        ok_tr_d = np.isfinite(tr_di)
        if ok_tr_d.sum() < 5:
            r_di = np.nan
        else:
            model_d = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=3).fit(X_tr[ok_tr_d], tr_di[ok_tr_d])
            pred_d = model_d.predict(X_te)
            ok_te_d = np.isfinite(te_di)
            if ok_te_d.sum() > 2:
                r_di, _ = stats.spearmanr(pred_d[ok_te_d], te_di[ok_te_d])
            else:
                r_di = np.nan
        results.append({"fold": i, "sspg_spearman_r": float(r_sspg) if np.isfinite(r_sspg) else np.nan, "di_spearman_r": float(r_di) if np.isfinite(r_di) else np.nan})

    with open(os.path.join(args.output_dir, "scheme_b_spearman_runs.json"), "w") as f:
        json.dump(results, f, indent=2)
    r_sspg_vals = [x["sspg_spearman_r"] for x in results if np.isfinite(x["sspg_spearman_r"])]
    r_di_vals = [x["di_spearman_r"] for x in results if np.isfinite(x["di_spearman_r"])]
    print(f"Scheme B: {args.n_cv} CV folds")
    print(f"  SSPG Spearman r: median={np.median(r_sspg_vals):.4f}  mean={np.mean(r_sspg_vals):.4f}  n={len(r_sspg_vals)}")
    print(f"  DI   Spearman r: median={np.median(r_di_vals):.4f}  mean={np.mean(r_di_vals):.4f}  n={len(r_di_vals)}")
    print(f"Saved {args.output_dir}/scheme_b_spearman_runs.json")


if __name__ == "__main__":
    main()
