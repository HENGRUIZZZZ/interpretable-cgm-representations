"""
V6 路线 E：冻结 encoder，仅微调 26D→2 e2e_head。需先跑完 Route A 得到预训练模型。
"""
import os
import sys
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
OUT_BASE = os.path.join(REPO_ROOT, "paper1_results_v6")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max_runs", type=int, default=None)
    args = p.parse_args()
    for i, seed in enumerate(SEEDS):
        if args.max_runs is not None and i >= args.max_runs:
            break
        pretrained = os.path.join(OUT_BASE, f"routeA_seed{seed}", "autoencoder_p1_full.pt")
        if not os.path.isfile(pretrained):
            print(f"Skip seed {seed}: {pretrained} not found")
            continue
        out_dir = os.path.join(OUT_BASE, f"routeE_seed{seed}")
        os.makedirs(out_dir, exist_ok=True)
        env = os.environ.copy()
        env["P1_FINETUNE_HEAD_ONLY"] = "1"
        env["P1_PRETRAINED_MODEL"] = pretrained
        env["P1_SEED"] = str(seed)
        env["LAMBDA_SSPG"] = "0.05"
        env["LAMBDA_DI"] = "0.05"
        env["LAMBDA_IR"] = "0"
        env["P1_ONE_MEAL_PER_SUBJECT"] = "1"
        env["P1_SAVE_26D_LATENT"] = "1"
        env["P1_RESULTS_DIR"] = out_dir
        print(f"--- Route E seed {seed} (finetune head only) ---")
        subprocess.run([sys.executable, "run_p1_full_pipeline.py"], env=env, cwd=REPO_ROOT, timeout=600, check=False)


if __name__ == "__main__":
    main()
