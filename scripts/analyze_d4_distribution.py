"""
分析 latent_and_gold_all.csv 中 D1/D2/D4 的 SSPG、DI 分布，解释图中 D4 为何呈「狭长一条」。

用法（项目根目录）：
  python scripts/analyze_d4_distribution.py
  python scripts/analyze_d4_distribution.py --csv paper1_results_v3/latent_and_gold_all.csv
"""
import os
import sys
import argparse
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Analyze D1/D2/D4 distribution in latent_and_gold CSV.")
    parser.add_argument("--csv", default=None, help="Path to latent_and_gold_all.csv (default: paper1_results_v3/latent_and_gold_all.csv)")
    args = parser.parse_args()
    csv_path = args.csv or os.path.join(REPO_ROOT, "paper1_results_v3", "latent_and_gold_all.csv")
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["subject_id"], keep="first")
    if "dataset_id" not in df.columns:
        print("No dataset_id column.")
        return

    print("=" * 60)
    print("D1 / D2 / D4 金标准分布（subject-level，去重后）")
    print("=" * 60)
    for col in ["sspg", "di", "homa_ir"]:
        if col not in df.columns:
            continue
        print(f"\n--- {col.upper()} ---")
        for did in ["D1", "D2", "D4"]:
            sub = df[df["dataset_id"] == did][col].dropna()
            if len(sub) == 0:
                print(f"  {did}: n=0 (all NaN)")
                continue
            print(f"  {did}: n={len(sub)}, mean={sub.mean():.3f}, std={sub.std():.3f}, min={sub.min():.3f}, max={sub.max():.3f}")
    print("\n说明：若 D4 的 DI 的 std 很小、范围很窄，图中 mi vs DI 上 D4 会呈狭长一条；")
    print("留一数据集出时用 D1+D2 训 Ridge 在 D4 上预测易出现外推不稳。")
    print("主结果建议仅用 D1+D2 评估（P1_GOLD_DATASETS=D1,D2）。")


if __name__ == "__main__":
    main()
