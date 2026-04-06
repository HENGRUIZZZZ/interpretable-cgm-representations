"""
实验方案 v4.0 阶段一：对每个 meal 窗口计算 CGM 统计特征，与 26D latent 合并供三路对决使用。

用法（项目根目录）：
  python scripts/compute_cgm_stats.py --latent_csv paper1_results_v4/baseline_seed42/latent_and_gold_all_26d.csv \\
       --output_dir paper1_results_v4/baseline_seed42

需与 pipeline 使用相同数据顺序（D1/D2/D4、P1_ONE_MEAL_PER_SUBJECT）以保证 sample_ix 对齐。
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUTPUT_BASE = os.environ.get("CGM_PROJECT_OUTPUT", "/Users/hertz1030/Downloads/cgm_project/output")
P1_ONE_MEAL_PER_SUBJECT = os.environ.get("P1_ONE_MEAL_PER_SUBJECT", "").strip().lower() in ("1", "true", "yes")


def compute_one_window(y: np.ndarray) -> dict:
    """y: (T,) 或 (T,1) 一个 meal 窗口的 CGM 序列。返回统计量。"""
    y = np.asarray(y).ravel()
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return {
            "cgm_mean": np.nan, "cgm_std": np.nan, "cgm_cv": np.nan,
            "cgm_min": np.nan, "cgm_max": np.nan, "cgm_range": np.nan,
            "tir": np.nan, "tar": np.nan, "tbr": np.nan,
            "auc": np.nan, "ac_var": np.nan, "mge": np.nan,
        }
    n = len(y)
    mean = float(np.mean(y))
    std = float(np.std(y))
    cv = std / (mean + 1e-8)
    tir = float(np.sum((y >= 70) & (y <= 180)) / n * 100)
    tar = float(np.sum(y > 180) / n * 100)
    tbr = float(np.sum(y < 70) / n * 100)
    x = np.arange(n) * 5.0
    auc = float(np.trapz(y, x))
    diff = np.diff(y)
    ac_var = float(np.std(diff)) if len(diff) > 0 else np.nan
    baseline = float(np.min(y[: max(1, n // 4)])) if n else np.nan
    mge = float(np.mean(np.maximum(y - baseline, 0)))

    return {
        "cgm_mean": mean, "cgm_std": std, "cgm_cv": cv,
        "cgm_min": float(np.min(y)), "cgm_max": float(np.max(y)), "cgm_range": float(np.max(y) - np.min(y)),
        "tir": tir, "tar": tar, "tbr": tbr,
        "auc": auc, "ac_var": ac_var, "mge": mge,
    }


def load_and_stack_same_as_pipeline(output_base: str):
    """与 run_p1_full_pipeline 相同顺序加载 D1/D2/D4 并堆叠；若 P1_ONE_MEAL_PER_SUBJECT 则每受试者保留首样本。"""
    from load_cgm_project_data import load_cgm_project_level1_level2, load_cgm_project_level3
    from paper1_experiment_config import get_data_dir, P1_FULL_TRAIN_DATASETS
    from data_utils import Batch

    batch_list, info_list, labels_list, dataset_ids = [], [], [], []
    for did in P1_FULL_TRAIN_DATASETS:
        data_dir = get_data_dir(did, output_base)
        if not os.path.isdir(data_dir):
            continue
        if did in ("D1", "D2"):
            b, info, lab = load_cgm_project_level1_level2(data_dir=data_dir, num_meals_threshold=1)
        else:
            try:
                b, info, lab = load_cgm_project_level3(dataset_id=did, output_base=output_base)
            except Exception:
                continue
        batch_list.append(b)
        info_list.append(info)
        labels_list.append(lab)
        dataset_ids.append(did)

    cgm_list = [b.cgm for b in batch_list]
    pid_list = [np.asarray(info.patient_ids) for info in info_list]
    batch = Batch(
        cgm=np.concatenate(cgm_list, axis=0),
        timestamps=np.concatenate([b.timestamps for b in batch_list], axis=0),
        meals=np.concatenate([b.meals for b in batch_list], axis=0),
        demographics=np.concatenate([b.demographics for b in batch_list], axis=0),
        diagnosis=np.concatenate([b.diagnosis for b in batch_list], axis=0),
    )
    pids = np.concatenate(pid_list, axis=0)

    if P1_ONE_MEAL_PER_SUBJECT:
        _, first_ix = np.unique(pids, return_index=True)
        first_ix = np.sort(first_ix)
        batch = Batch(
            cgm=batch.cgm[first_ix],
            timestamps=batch.timestamps[first_ix],
            meals=batch.meals[first_ix],
            demographics=batch.demographics[first_ix],
            diagnosis=batch.diagnosis[first_ix],
        )
        pids = pids[first_ix]

    return batch, pids


def main():
    parser = argparse.ArgumentParser(description="Compute CGM stats per sample and merge with 26D latent")
    parser.add_argument("--latent_csv", type=str, help="Path to latent_and_gold_all_26d.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: dir of latent_csv)")
    parser.add_argument("--output_base", type=str, default=OUTPUT_BASE, help="cgm_project output base")
    args = parser.parse_args()

    if not args.latent_csv or not os.path.isfile(args.latent_csv):
        print("Need --latent_csv pointing to latent_and_gold_all_26d.csv")
        sys.exit(1)
    out_dir = args.output_dir or os.path.dirname(args.latent_csv)
    os.makedirs(out_dir, exist_ok=True)

    batch, pids = load_and_stack_same_as_pipeline(args.output_base)
    n = batch.cgm.shape[0]
    rows = []
    for i in range(n):
        stats = compute_one_window(batch.cgm[i])
        stats["sample_ix"] = i
        stats["subject_id"] = pids[i]
        rows.append(stats)
    cgm_df = pd.DataFrame(rows)

    latent_df = pd.read_csv(args.latent_csv)
    if "sample_ix" not in latent_df.columns:
        print("latent CSV has no sample_ix; merge by row order (lengths must match)")
        if len(latent_df) != len(cgm_df):
            print(f"Length mismatch: latent {len(latent_df)} vs cgm_stats {len(cgm_df)}")
            sys.exit(1)
        for c in cgm_df.columns:
            if c not in latent_df.columns:
                latent_df[c] = cgm_df[c].values
        full = latent_df
    else:
        full = latent_df.merge(cgm_df, on="sample_ix", how="left")

    out_path = os.path.join(out_dir, "full_features_v4.csv")
    full.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {full.shape[1]} columns, {len(full)} rows")
    cgm_df.to_csv(os.path.join(out_dir, "cgm_stats_per_sample.csv"), index=False)
    print(f"Saved {out_dir}/cgm_stats_per_sample.csv")


if __name__ == "__main__":
    main()
