from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import MechanisticAutoencoder

DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
V18_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v18")
V21_OPT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v21_optimization")
OUT_DIR = os.path.join(V21_OPT_ROOT, "story_plots")
STD_MEALS = ["Cornflakes", "PB_sandwich", "Protein_bar"]


def _norm_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for old, new in [("SSPG", "sspg"), ("DI", "di")]:
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    return out


def _build_d4_windows() -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "subjects.csv"))
    meals = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "meals.csv"))
    cgm = pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "cgm.csv"))
    labels = _norm_labels(pd.read_csv(os.path.join(DATA_ROOT, "D4_hall", "labels.csv")))
    label_df = labels[["subject_id", "sspg", "di"]].drop_duplicates("subject_id")

    id_map: Dict[str, str] = {}
    for _, r in subjects.dropna(subset=["subject_id", "original_id"]).iterrows():
        orig = str(r["original_id"]).strip()
        sid = str(r["subject_id"]).strip()
        id_map[orig] = sid
        id_map[f"D4_{orig}"] = sid

    meals["timestamp"] = pd.to_datetime(meals["timestamp"], errors="coerce")
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")
    if "glucose_mg_dl" in cgm.columns and "glucose_mgdl" not in cgm.columns:
        cgm = cgm.rename(columns={"glucose_mg_dl": "glucose_mgdl"})

    grid = np.arange(-30, 181, 5, dtype=np.float64)
    rows: List[Dict[str, object]] = []
    for _, meal in meals[meals["meal_type"].isin(STD_MEALS)].iterrows():
        sid_raw = str(meal["subject_id"])
        sid = id_map.get(sid_raw, sid_raw)
        t0 = meal["timestamp"]
        if pd.isna(t0):
            continue
        g = cgm[
            (cgm["subject_id"] == sid_raw)
            & (cgm["timestamp"] >= t0 + pd.Timedelta(minutes=-30))
            & (cgm["timestamp"] <= t0 + pd.Timedelta(minutes=180))
        ].copy()
        if len(g) < 10:
            continue
        t = ((g["timestamp"] - t0).dt.total_seconds() / 60.0).to_numpy(float)
        y = pd.to_numeric(g["glucose_mgdl"], errors="coerce").to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(y)
        if ok.sum() < 10:
            continue
        t, y = t[ok], y[ok]
        order = np.argsort(t)
        t, y = t[order], y[order]
        y_new = np.interp(grid, t, y).astype(np.float32)

        meal_series = np.zeros((len(grid), 6), dtype=np.float32)
        carb = float(pd.to_numeric(meal.get("carb_g", 0.0), errors="coerce") or 0.0)
        fat = float(pd.to_numeric(meal.get("fat_g", 0.0), errors="coerce") or 0.0)
        protein = float(pd.to_numeric(meal.get("protein_g", 0.0), errors="coerce") or 0.0)
        fiber = float(pd.to_numeric(meal.get("fiber_g", 0.0), errors="coerce") or 0.0)
        meal_series[:, 0] = carb + fat + protein
        meal_series[:, 1] = carb
        meal_series[:, 3] = fiber
        meal_series[:, 4] = fat
        meal_series[:, 5] = protein

        srow = subjects[subjects["subject_id"].astype(str) == sid]
        if srow.empty:
            demo = np.array([0.0, 40.0, 72.0], dtype=np.float32)
        else:
            s = srow.iloc[0]
            gender = 1.0 if str(s.get("sex", "M")).upper().startswith("F") else 0.0
            age = float(pd.to_numeric(s.get("age", 40.0), errors="coerce") or 40.0)
            weight = float(pd.to_numeric(s.get("weight_kg", np.nan), errors="coerce"))
            if not np.isfinite(weight):
                bmi = float(pd.to_numeric(s.get("bmi", 25.0), errors="coerce") or 25.0)
                weight = bmi * (1.7 ** 2)
            demo = np.array([gender, age, weight], dtype=np.float32)

        rows.append(
            {
                "subject_id": sid,
                "meal_type": str(meal["meal_type"]),
                "curve": y_new,
                "timestamps": grid.astype(np.float32),
                "meal_series": meal_series,
                "demographics": demo,
            }
        )
    return pd.DataFrame(rows), label_df


def _encode_d4_latents(windows_df: pd.DataFrame) -> pd.DataFrame:
    ckpt = torch.load(
        os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase2_finetune_head", "autoencoder_p1_full.pt"),
        map_location="cpu",
        weights_only=False,
    )
    model = MechanisticAutoencoder(
        meal_size=6, demographics_size=3, embedding_size=8, hidden_size=32, num_layers=2,
        encoder_dropout_prob=0.0, decoder_dropout_prob=0.5
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    tm, tsd = ckpt["train_mean"], ckpt["train_std"]
    rows = []
    for _, r in windows_df.iterrows():
        c = r["curve"][None, :, None].astype(np.float32)
        ts = r["timestamps"][None, :, None].astype(np.float32)
        meals = r["meal_series"][None, :, :].astype(np.float32)
        demo = r["demographics"][None, :].astype(np.float32)
        c = (c - tm[0]) / (tsd[0] + 1e-8)
        ts = (ts - tm[1]) / (tsd[1] + 1e-8)
        meals = (meals - tm[2]) / (tsd[2] + 1e-8)
        demo = (demo - tm[3]) / (tsd[3] + 1e-8)
        with torch.no_grad():
            p26, init26, z16 = model.get_all_latents(
                torch.tensor(c), torch.tensor(ts), torch.tensor(meals), torch.tensor(demo)
            )
        z = np.concatenate([p26.numpy()[0], init26.numpy()[0], z16.numpy()[0]], axis=0)
        row = {"subject_id": r["subject_id"], "meal_type": r["meal_type"]}
        for i, v in enumerate(z):
            row[f"z{i:02d}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def _scatter(ax, x, y, c, title, cmap="viridis"):
    s = ax.scatter(x, y, c=c, cmap=cmap, s=90, edgecolors="k", linewidths=0.4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(s, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    d4_windows, labels = _build_d4_windows()
    lat = _encode_d4_latents(d4_windows)

    # subject-level mean latent
    zcols = [f"z{i:02d}" for i in range(26)]
    sub_lat = lat.groupby("subject_id", as_index=False)[zcols].mean()
    pca = PCA(n_components=2, random_state=42)
    z2 = pca.fit_transform(sub_lat[zcols].to_numpy(float))
    emb = pd.DataFrame({"subject_id": sub_lat["subject_id"], "pc1": z2[:, 0], "pc2": z2[:, 1]})

    pred = pd.read_csv(os.path.join(V21_OPT_ROOT, "v21_opt_predictions_subject_level.csv"))
    p26 = pred[pred["model"] == "Ridge26D"][["subject_id", "sspg_true", "di_true", "sspg_pred", "di_pred"]].drop_duplicates("subject_id")
    plot_df = emb.merge(p26, on="subject_id", how="left")
    plot_df["ir_true"] = (plot_df["sspg_true"] >= 120).astype(int)
    plot_df["ir_pred"] = (plot_df["sspg_pred"] >= 120).astype(int)

    plot_df.to_csv(os.path.join(OUT_DIR, "v21_2d_embedding_subject_level.csv"), index=False)

    # 2D label/pred map
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=180)
    _scatter(axes[0, 0], plot_df["pc1"], plot_df["pc2"], plot_df["sspg_true"], "True SSPG on 2D latent")
    _scatter(axes[0, 1], plot_df["pc1"], plot_df["pc2"], plot_df["sspg_pred"], "Pred SSPG (Ridge26D) on 2D latent")
    _scatter(axes[1, 0], plot_df["pc1"], plot_df["pc2"], plot_df["di_true"], "True DI on 2D latent")
    _scatter(axes[1, 1], plot_df["pc1"], plot_df["pc2"], plot_df["di_pred"], "Pred DI (Ridge26D) on 2D latent")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "v21_2d_true_vs_pred_maps.png"))
    plt.close(fig)

    # true vs pred scatter
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=180)
    axes[0].scatter(plot_df["sspg_true"], plot_df["sspg_pred"], c=plot_df["pc1"], cmap="coolwarm", s=90, edgecolors="k", linewidths=0.4)
    axes[0].plot([plot_df["sspg_true"].min(), plot_df["sspg_true"].max()], [plot_df["sspg_true"].min(), plot_df["sspg_true"].max()], "k--", lw=1)
    axes[0].set_title("SSPG True vs Pred")
    axes[0].set_xlabel("True")
    axes[0].set_ylabel("Pred")
    ok_s = plot_df[["sspg_true", "sspg_pred"]].dropna()
    s_rho = stats.spearmanr(ok_s["sspg_true"], ok_s["sspg_pred"]).statistic if len(ok_s) >= 3 else np.nan
    axes[0].text(0.03, 0.95, f"Spearman={s_rho:.3f}", transform=axes[0].transAxes, va="top")

    axes[1].scatter(plot_df["di_true"], plot_df["di_pred"], c=plot_df["pc1"], cmap="coolwarm", s=90, edgecolors="k", linewidths=0.4)
    axes[1].plot([plot_df["di_true"].min(), plot_df["di_true"].max()], [plot_df["di_true"].min(), plot_df["di_true"].max()], "k--", lw=1)
    axes[1].set_title("DI True vs Pred")
    axes[1].set_xlabel("True")
    axes[1].set_ylabel("Pred")
    ok_d = plot_df[["di_true", "di_pred"]].dropna()
    d_rho = stats.spearmanr(ok_d["di_true"], ok_d["di_pred"]).statistic if len(ok_d) >= 3 else np.nan
    axes[1].text(0.03, 0.95, f"Spearman={d_rho:.3f}", transform=axes[1].transAxes, va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "v21_true_vs_pred_scatter.png"))
    plt.close(fig)

    # error map
    plot_df["sspg_abs_err"] = np.abs(plot_df["sspg_true"] - plot_df["sspg_pred"])
    plot_df["di_abs_err"] = np.abs(plot_df["di_true"] - plot_df["di_pred"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=180)
    _scatter(axes[0], plot_df["pc1"], plot_df["pc2"], plot_df["sspg_abs_err"], "SSPG absolute error on 2D latent", cmap="magma")
    _scatter(axes[1], plot_df["pc1"], plot_df["pc2"], plot_df["di_abs_err"], "DI absolute error on 2D latent", cmap="magma")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "v21_2d_error_maps.png"))
    plt.close(fig)

    summary = {
        "sspg_spearman": float(s_rho),
        "di_spearman": float(d_rho),
        "sspg_mae": float(mean_absolute_error(ok_s["sspg_true"], ok_s["sspg_pred"])) if len(ok_s) else np.nan,
        "di_mae": float(mean_absolute_error(ok_d["di_true"], ok_d["di_pred"])) if len(ok_d) else np.nan,
        "pca_explained_var_ratio": pca.explained_variance_ratio_.tolist(),
    }
    with open(os.path.join(OUT_DIR, "v21_2d_plot_summary.json"), "w") as f:
        import json
        json.dump(summary, f, indent=2)
    print("Saved:", OUT_DIR)


if __name__ == "__main__":
    main()
