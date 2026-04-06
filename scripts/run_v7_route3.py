"""
V7 Route 3: 多目标学习 - 组合 A（lambda_sspg_best + lambda_di_best）训练，评估 DI 的 LODO Pearson r。
组合 B（+ HOMA）需 pipeline 支持 LAMBDA_HOMA_IR / LAMBDA_HOMA_B，此处仅跑组合 A。
"""
from __future__ import annotations

import os
import sys
import subprocess
import json
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
OUT_V7 = os.path.join(REPO_ROOT, "paper1_results_v7")
# From Route 1 table
LAMBDA_SSPG_BEST = 0.1
LAMBDA_DI_BEST = 0.05


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()
    os.makedirs(os.path.join(OUT_V7, "route3"), exist_ok=True)
    out_dir = os.path.join(OUT_V7, "route3", "combo_a")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "latent_and_gold_all_26d.csv")

    if not args.skip_train:
        env = os.environ.copy()
        env["P1_HEAD_USE_26D"] = "1"
        env["P1_SAVE_26D_LATENT"] = "1"
        env["P1_ONE_MEAL_PER_SUBJECT"] = "1"
        env["LAMBDA_SSPG"] = str(LAMBDA_SSPG_BEST)
        env["LAMBDA_DI"] = str(LAMBDA_DI_BEST)
        env["P1_RESULTS_DIR"] = out_dir
        env["P1_SEED"] = "42"
        ret = subprocess.run(
            [sys.executable, "run_p1_full_pipeline.py", "--lambda_sspg", str(LAMBDA_SSPG_BEST), "--lambda_di", str(LAMBDA_DI_BEST)],
            cwd=REPO_ROOT, env=env, timeout=args.timeout,
        )
        if ret.returncode != 0 or not os.path.isfile(csv_path):
            print("Route 3 combo A pipeline failed or no 26D CSV")
            return

    _scripts = os.path.join(REPO_ROOT, "scripts")
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from run_v7_lodo_pearson import lodo_pearson
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "dataset_id" not in df.columns and "subject_id" in df.columns:
        from run_p1_full_pipeline import _stack_batches
        from load_cgm_project_data import load_cgm_project_level1_level2, load_cgm_project_level3
        from paper1_experiment_config import get_data_dir, P1_FULL_TRAIN_DATASETS
        OUTPUT_BASE = os.environ.get("CGM_PROJECT_OUTPUT", "/Users/hertz1030/Downloads/cgm_project/output")
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
        if not labels_combined.empty and "dataset_id" in labels_combined.columns:
            did_map = labels_combined.set_index("subject_id")["dataset_id"].to_dict()
            df["dataset_id"] = df["subject_id"].map(lambda x: did_map.get(str(x), ""))
    _, mean_r = lodo_pearson(df, "di")
    print(f"V7 Route 3 Combo A (lambda_sspg={LAMBDA_SSPG_BEST}, lambda_di={LAMBDA_DI_BEST})  LODO-CV Pearson r(DI) = {mean_r:.4f}")
    with open(os.path.join(OUT_V7, "route3", "route3_combo_a_pearson.json"), "w") as f:
        json.dump({"mean_pearson_r_di": mean_r, "lambda_sspg": LAMBDA_SSPG_BEST, "lambda_di": LAMBDA_DI_BEST}, f, indent=2)
    print("Saved paper1_results_v7/route3/route3_combo_a_pearson.json")


if __name__ == "__main__":
    main()
