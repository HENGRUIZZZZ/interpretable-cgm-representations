"""
V7 Route 1: 基线优化 - λ 参数扫描。
实验 1A: 扫描 lambda_sspg (lambda_di=0)，LODO-CV Pearson r 选最优。
实验 1B: 扫描 lambda_di (lambda_sspg=0)，LODO-CV Pearson r 选最优。
所有运行使用 V6 Route B 框架（26D 全潜在空间，P1_HEAD_USE_26D=1）。
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
LAMBDAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]


def run_pipeline(lambda_sspg: float, lambda_di: float, out_dir: str, timeout: int = 600) -> bool:
    env = os.environ.copy()
    env["P1_HEAD_USE_26D"] = "1"
    env["P1_SAVE_26D_LATENT"] = "1"
    env["P1_ONE_MEAL_PER_SUBJECT"] = "1"
    env["LAMBDA_SSPG"] = str(lambda_sspg)
    env["LAMBDA_DI"] = str(lambda_di)
    env["P1_RESULTS_DIR"] = out_dir
    env["P1_SEED"] = "42"
    cmd = [
        sys.executable, "run_p1_full_pipeline.py",
        "--lambda_sspg", str(lambda_sspg),
        "--lambda_di", str(lambda_di),
    ]
    ret = subprocess.run(cmd, cwd=REPO_ROOT, env=env, timeout=timeout)
    return ret.returncode == 0 and os.path.isfile(os.path.join(out_dir, "latent_and_gold_all_26d.csv"))


def lodo_pearson_for_csv(csv_path: str, target: str) -> float:
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
    _, mean_r = lodo_pearson(df, target)
    return mean_r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=str, choices=["1A", "1B", "all"], default="all")
    p.add_argument("--skip_train", action="store_true", help="Only run LODO eval on existing dirs")
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()
    os.makedirs(OUT_V7, exist_ok=True)
    os.makedirs(os.path.join(OUT_V7, "route1"), exist_ok=True)

    results_1a = []
    results_1b = []

    if args.exp in ("1A", "all"):
        print("--- V7 Route 1A: lambda_sspg scan (lambda_di=0) ---")
        for lam in LAMBDAS:
            out_dir = os.path.join(OUT_V7, "route1", f"1a_lambda_sspg_{lam}")
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "latent_and_gold_all_26d.csv")
            if not args.skip_train:
                ok = run_pipeline(lambda_sspg=lam, lambda_di=0.0, out_dir=out_dir, timeout=args.timeout)
                if not ok:
                    print(f"  lambda_sspg={lam} pipeline failed or no 26D CSV")
                    results_1a.append({"lambda_sspg": lam, "lambda_di": 0, "mean_pearson_r": None})
                    continue
            if os.path.isfile(csv_path):
                r = lodo_pearson_for_csv(csv_path, "sspg")
                results_1a.append({"lambda_sspg": lam, "lambda_di": 0, "mean_pearson_r": r})
                print(f"  lambda_sspg={lam}  LODO-CV Pearson r(SSPG) = {r:.4f}")
            else:
                results_1a.append({"lambda_sspg": lam, "lambda_di": 0, "mean_pearson_r": None})

    if args.exp in ("1B", "all"):
        print("--- V7 Route 1B: lambda_di scan (lambda_sspg=0) ---")
        for lam in LAMBDAS:
            out_dir = os.path.join(OUT_V7, "route1", f"1b_lambda_di_{lam}")
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "latent_and_gold_all_26d.csv")
            if not args.skip_train:
                ok = run_pipeline(lambda_sspg=0.0, lambda_di=lam, out_dir=out_dir, timeout=args.timeout)
                if not ok:
                    print(f"  lambda_di={lam} pipeline failed or no 26D CSV")
                    results_1b.append({"lambda_sspg": 0, "lambda_di": lam, "mean_pearson_r": None})
                    continue
            if os.path.isfile(csv_path):
                r = lodo_pearson_for_csv(csv_path, "di")
                results_1b.append({"lambda_sspg": 0, "lambda_di": lam, "mean_pearson_r": r})
                print(f"  lambda_di={lam}  LODO-CV Pearson r(DI) = {r:.4f}")
            else:
                results_1b.append({"lambda_sspg": 0, "lambda_di": lam, "mean_pearson_r": None})

    # Summary table
    report = ["V7 Route 1: Lambda scan (LODO-CV Pearson r)", "=" * 50]
    if results_1a:
        report.append("\n1A: lambda_sspg (lambda_di=0) -> SSPG")
        valid = [x for x in results_1a if x["mean_pearson_r"] is not None]
        for x in results_1a:
            r = x["mean_pearson_r"]
            report.append(f"  lambda_sspg={x['lambda_sspg']:.3f}  Pearson r = {r:.4f}" if r is not None else f"  lambda_sspg={x['lambda_sspg']:.3f}  (no result)")
        if valid:
            best = max(valid, key=lambda t: abs(t["mean_pearson_r"]))
            report.append(f"  -> lambda_sspg_best = {best['lambda_sspg']}  (r = {best['mean_pearson_r']:.4f})")
    if results_1b:
        report.append("\n1B: lambda_di (lambda_sspg=0) -> DI")
        for x in results_1b:
            r = x["mean_pearson_r"]
            report.append(f"  lambda_di={x['lambda_di']:.3f}  Pearson r = {r:.4f}" if r is not None else f"  lambda_di={x['lambda_di']:.3f}  (no result)")
        valid = [x for x in results_1b if x["mean_pearson_r"] is not None]
        if valid:
            best = max(valid, key=lambda t: abs(t["mean_pearson_r"]))
            report.append(f"  -> lambda_di_best = {best['lambda_di']}  (r = {best['mean_pearson_r']:.4f})")
    text = "\n".join(report)
    print("\n" + text)
    table_path = os.path.join(OUT_V7, "route1", "route1_lambda_scan_table.txt")
    with open(table_path, "w") as f:
        f.write(text)
    with open(os.path.join(OUT_V7, "route1", "route1_lambda_scan.json"), "w") as f:
        json.dump({"1A": results_1a, "1B": results_1b}, f, indent=2)
    print(f"\nSaved {table_path}")


if __name__ == "__main__":
    main()
