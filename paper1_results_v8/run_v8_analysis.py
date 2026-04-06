"""
V8 Route 3: 综合分析 - 整合两个专家模型的 latent，LODO-CV Ridge 得到 SSPG/DI 预测，评估并生成四象限图。
运行方式: cd paper1_results_v8 && python run_v8_analysis.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# 路径（相对于 paper1_results_v8/）
BASE = os.path.dirname(os.path.abspath(__file__))
SSPG_LATENTS_PATH = os.path.join(BASE, "route1", "sspg_specialist_latents.csv")
DI_LATENTS_PATH = os.path.join(BASE, "route2", "di_specialist_latents.csv")
OUTPUT_CSV_PATH = os.path.join(BASE, "route3", "v8_final_results.csv")
OUTPUT_PLOT_PATH = os.path.join(BASE, "route3", "v8_quadrant_plot.png")

# 26D latent 列（与 pipeline 一致）
LATENT_26D = (
    ["tau_m", "Gb", "sg", "si", "p2", "mi"]
    + [f"z_init_{i}" for i in range(4)]
    + [f"z_nonseq_{i}" for i in range(16)]
)


def get_dataset(sid: str) -> str:
    """从 subject_id 提取 dataset，如 D1_S01 -> D1"""
    if isinstance(sid, str) and (sid.startswith("D1") or sid.startswith("D2") or sid.startswith("D4")):
        for prefix in ("D1", "D2", "D4"):
            if sid.startswith(prefix):
                return prefix
    return ""


def get_lodo_predictions(df_latents: pd.DataFrame, target_col: str, feature_cols: list[str]) -> dict:
    """LODO-CV: 留一数据集出，Ridge 回归，返回每个 subject_id 的预测值。"""
    df_valid = df_latents.dropna(subset=[target_col]).copy()
    feat = [c for c in feature_cols if c in df_valid.columns]
    if not feat:
        return {}
    df_valid["dataset"] = df_valid["subject_id"].astype(str).map(get_dataset)
    datasets = sorted(df_valid["dataset"].unique())
    if "" in datasets:
        datasets = [d for d in datasets if d]
    if len(datasets) < 2:
        return {}
    predictions = {}
    for test_ds in datasets:
        train_df = df_valid[df_valid["dataset"] != test_ds]
        test_df = df_valid[df_valid["dataset"] == test_ds]
        if len(train_df) < 3 or len(test_df) < 2:
            continue
        X_train = train_df[feat].values
        y_train = train_df[target_col].values
        X_test = test_df[feat].values
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        model = RidgeCV(alphas=np.logspace(-3, 3, 100)).fit(X_tr_s, y_train)
        preds = model.predict(X_te_s)
        for sid, pred_val in zip(test_df["subject_id"], preds):
            predictions[str(sid)] = float(pred_val)
    return predictions


def main():
    os.makedirs(os.path.join(BASE, "route3"), exist_ok=True)
    if not os.path.isfile(SSPG_LATENTS_PATH):
        print(f"未找到: {SSPG_LATENTS_PATH}")
        sys.exit(1)
    if not os.path.isfile(DI_LATENTS_PATH):
        print(f"未找到: {DI_LATENTS_PATH}")
        sys.exit(1)

    df_sspg = pd.read_csv(SSPG_LATENTS_PATH)
    df_di = pd.read_csv(DI_LATENTS_PATH)

    sspg_predictions = get_lodo_predictions(df_sspg, "sspg", LATENT_26D)
    di_predictions = get_lodo_predictions(df_di, "di", LATENT_26D)

    final_df = df_sspg.copy()
    if "dataset_id" not in final_df.columns:
        final_df["dataset"] = final_df["subject_id"].astype(str).map(get_dataset)
    else:
        final_df["dataset"] = final_df["dataset_id"].fillna(final_df["subject_id"].astype(str).map(get_dataset))
    final_df["sspg_pred"] = final_df["subject_id"].astype(str).map(sspg_predictions)
    final_df["di_pred"] = final_df["subject_id"].astype(str).map(di_predictions)

    analysis_df = final_df.dropna(subset=["sspg", "di", "sspg_pred", "di_pred"]).copy()
    analysis_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"V8 最终结果已保存到: {OUTPUT_CSV_PATH} (共 {len(analysis_df)} 个样本)")

    if len(analysis_df) < 4:
        print("样本不足，无法计算 Pearson r")
    else:
        r_sspg, p_sspg = stats.pearsonr(analysis_df["sspg"], analysis_df["sspg_pred"])
        r_di, p_di = stats.pearsonr(analysis_df["di"], analysis_df["di_pred"])
        print("\n--- V8 最终性能 (LODO-CV 整合后 Pearson r) ---")
        print(f"SSPG: r = {r_sspg:.4f}  (p = {p_sspg:.4f})")
        print(f"DI:   r = {r_di:.4f}  (p = {p_di:.4f})")
        with open(os.path.join(BASE, "route3", "v8_metrics.txt"), "w") as f:
            f.write(f"SSPG Pearson r = {r_sspg:.4f}  p = {p_sspg:.4f}\n")
            f.write(f"DI   Pearson r = {r_di:.4f}  p = {p_di:.4f}\n")

    # 四象限图
    sspg_med = analysis_df["sspg"].median()
    di_med = analysis_df["di"].median()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axvline(sspg_med, color="grey", linestyle="--", alpha=0.8, label=f"SSPG median={sspg_med:.0f}")
        ax.axhline(di_med, color="grey", linestyle="--", alpha=0.8, label=f"DI median={di_med:.2f}")
        if "dataset" in analysis_df.columns and analysis_df["dataset"].notna().any():
            for ds in analysis_df["dataset"].dropna().unique():
                m = analysis_df["dataset"] == ds
                ax.scatter(analysis_df.loc[m, "sspg_pred"], analysis_df.loc[m, "di_pred"], label=str(ds), alpha=0.7)
        else:
            ax.scatter(analysis_df["sspg_pred"], analysis_df["di_pred"], alpha=0.7)
        ax.set_xlabel("Predicted SSPG (Insulin Resistance)")
        ax.set_ylabel("Predicted DI (Beta-Cell Function)")
        ax.set_title("V8 Dual Expert System: IR-Beta Quadrant")
        ax.legend()
        fig.savefig(OUTPUT_PLOT_PATH, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n四象限图已保存: {OUTPUT_PLOT_PATH}")
    except Exception as e:
        print(f"绘图跳过: {e}")


if __name__ == "__main__":
    main()
