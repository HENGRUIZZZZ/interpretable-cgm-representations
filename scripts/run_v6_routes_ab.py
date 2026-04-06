"""
V6 路线 A 与 B：批量运行 10 种子，保存到 paper1_results_v6/routeA_seed* 与 routeB_seed*。

路线 A：6D ODE 弱监督，LAMBDA_SSPG=0.05, LAMBDA_DI=0.05, P1_ONE_MEAL_PER_SUBJECT=1
路线 B：26D 全 latent 弱监督，P1_HEAD_USE_26D=1，其余同 A。
"""
import os
import sys
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
OUT_BASE = "paper1_results_v6"


def run_route(route: str, max_runs: int = None):
    assert route in ("A", "B")
    for i, seed in enumerate(SEEDS):
        if max_runs is not None and i >= max_runs:
            break
        out_dir = os.path.join(REPO_ROOT, OUT_BASE, f"route{route}_seed{seed}")
        os.makedirs(out_dir, exist_ok=True)
        env = os.environ.copy()
        env["P1_SEED"] = str(seed)
        env["LAMBDA_SSPG"] = "0.05"
        env["LAMBDA_DI"] = "0.05"
        env["LAMBDA_IR"] = "0"
        env["P1_ONE_MEAL_PER_SUBJECT"] = "1"
        env["P1_SAVE_26D_LATENT"] = "1"
        env["P1_RESULTS_DIR"] = out_dir
        if route == "B":
            env["P1_HEAD_USE_26D"] = "1"
        print(f"--- Route {route} seed {seed} -> {out_dir} ---")
        subprocess.run([sys.executable, "run_p1_full_pipeline.py"], env=env, cwd=REPO_ROOT, timeout=600, check=False)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--route", type=str, choices=["A", "B"], required=True)
    p.add_argument("--max_runs", type=int, default=None)
    args = p.parse_args()
    run_route(args.route, args.max_runs)
