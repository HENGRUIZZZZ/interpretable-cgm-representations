"""
实验方案 v5.0 终局之战 — 方案 A：Prediction Head 模型。

运行 4 组 lambda (0.01, 0.1, 1.0, 10.0) × 10 个随机种子，收集每轮 test set 上 subject-level Spearman r。

用法（项目根目录）：
  python scripts/run_v5_scheme_a.py --output_dir paper1_results_v5/scheme_a
"""
import os
import sys
import json
import subprocess
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

LAMBDAS = [0.01, 0.1, 1.0, 10.0]
SEEDS = list(range(42, 52))  # 10 seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="paper1_results_v5/scheme_a")
    parser.add_argument("--max_runs", type=int, default=None, help="Limit runs (e.g. 2 for quick test)")
    args = parser.parse_args()
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    results = []
    run_ix = 0
    for lam in LAMBDAS:
        for seed in SEEDS:
            run_ix += 1
            if args.max_runs and run_ix > args.max_runs:
                break
            run_name = f"lambda{lam}_seed{seed}"
            run_dir = os.path.join(out_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
            env = os.environ.copy()
            env["P1_V5_PREDICTION_HEAD"] = "1"
            env["P1_ONE_MEAL_PER_SUBJECT"] = "0"
            env["LAMBDA_SSPG"] = str(lam)
            env["LAMBDA_DI"] = str(lam)
            env["LAMBDA_IR"] = "0"
            env["P1_SEED"] = str(seed)
            env["P1_RESULTS_DIR"] = run_dir
            print(f"--- Run {run_ix}: lambda={lam} seed={seed} ---")
            ret = subprocess.run(
                [sys.executable, "run_p1_full_pipeline.py"],
                env=env,
                cwd=REPO_ROOT,
                timeout=600,
            )
            if ret.returncode != 0:
                print(f"  WARNING: pipeline exited {ret.returncode}")
            v5_path = os.path.join(run_dir, "v5_spearman.json")
            if os.path.isfile(v5_path):
                with open(v5_path) as f:
                    r = json.load(f)
                r["run_name"] = run_name
                r["lambda"] = lam
                r["seed"] = seed
                results.append(r)
                print(f"  SSPG r={r.get('sspg_spearman_r', float('nan')):.4f}  DI r={r.get('di_spearman_r', float('nan')):.4f}")
            else:
                results.append({"run_name": run_name, "lambda": lam, "seed": seed, "sspg_spearman_r": None, "di_spearman_r": None})
                print("  (v5_spearman.json not found)")

    with open(os.path.join(out_dir, "scheme_a_all_runs.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_dir}/scheme_a_all_runs.json ({len(results)} runs)")


if __name__ == "__main__":
    main()
