"""
v30: Residual ablation at inference, 10D vs 16D context probes, stratified ICC.

Outputs under New_paper1_results_glucovector_v30_residual_probe_icc/
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

V18_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v18")
V22_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v22_locked_protocol")
D3_MERGED = os.path.join(
    REPO_ROOT, "New_paper1_results_glucovector_v20", "d3_deep_ablation", "d3_merged_GV_26D_Exp8.csv"
)
OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v30_residual_probe_icc")
SEED = 42


def _norm_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for old, new in [("SSPG", "sspg"), ("DI", "di"), ("HOMA_IR", "homa_ir"), ("HOMA_B", "homa_b")]:
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    return out


def _build_latent_train() -> pd.DataFrame:
    lat = pd.read_csv(
        os.path.join(V18_ROOT, "v18_Exp8_CorrLoss", "phase2_finetune_head", "latent_and_gold_all_26d.csv")
    )
    lat = _norm_labels(lat)
    rename_map = {
        "tau_m": "z00",
        "Gb": "z01",
        "sg": "z02",
        "si": "z03",
        "p2": "z04",
        "mi": "z05",
        "z_init_0": "z06",
        "z_init_1": "z07",
        "z_init_2": "z08",
        "z_init_3": "z09",
    }
    for k, v in rename_map.items():
        if k in lat.columns and v not in lat.columns:
            lat[v] = lat[k]
    for i in range(16):
        src = f"z_nonseq_{i}"
        dst = f"z{10 + i:02d}"
        if src in lat.columns and dst not in lat.columns:
            lat[dst] = lat[src]
    return lat.dropna(subset=[f"z{i:02d}" for i in range(26)] + ["sspg", "di"]).copy()


def _primary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 3:
        return {"n": int(len(y)), "spearman": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    return {
        "n": int(len(y)),
        "spearman": float(stats.spearmanr(y, p)[0]),
        "r2": float(r2_score(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
    }


def icc1_unbalanced(y: np.ndarray, groups: np.ndarray) -> float:
    """Shrout & Fleiss ICC(1,1) one-way random, unequal group sizes."""
    y = np.asarray(y, dtype=float)
    g = np.asarray(groups)
    ok = np.isfinite(y)
    y, g = y[ok], g[ok]
    df = pd.DataFrame({"y": y, "g": g})
    k = df["g"].nunique()
    n_total = len(df)
    if k < 2 or n_total < k + 1:
        return np.nan
    n_per = df.groupby("g", observed=True).size()
    grand = float(df["y"].mean())
    ssb = float(((df.groupby("g", observed=True)["y"].mean() - grand) ** 2 * n_per).sum())
    ssw = float(df.groupby("g", observed=True)["y"].transform(lambda s: (s - s.mean()) ** 2).sum())
    dfb = k - 1
    dfw = n_total - k
    if dfb < 1 or dfw < 1:
        return np.nan
    msb = ssb / dfb
    msw = ssw / dfw
    n0 = (n_total - float((n_per**2).sum() / n_total)) / dfb
    if n0 < 1e-8 or (msb + (n0 - 1) * msw) < 1e-12:
        return np.nan
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    return float(np.clip(icc, -1.0, 1.0))


def _subject_aggregate_preds(
    d4: pd.DataFrame, z10: List[str], z26: List[str], m_s, m_d, rng: np.random.Generator, shuffle: bool, zero_res: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X10 = d4[z10].to_numpy(float)
    X16 = d4[[f"z{i:02d}" for i in range(10, 26)]].to_numpy(float)
    if zero_res:
        X16_use = np.zeros_like(X16)
    elif shuffle:
        idx = rng.permutation(len(X16))
        X16_use = X16[idx]
    else:
        X16_use = X16
    X = np.hstack([X10, X16_use])
    ps = m_s.predict(X)
    pd_ = m_d.predict(X)
    sub = d4[["subject_id"]].copy()
    sub["ps"] = ps
    sub["pd"] = pd_
    agg = sub.groupby("subject_id", as_index=False).mean(numeric_only=True)
    return agg["ps"].to_numpy(float), agg["pd"].to_numpy(float), agg["subject_id"].astype(str).to_numpy()


def run_ablation(d4: pd.DataFrame, tr_l: pd.DataFrame, d4_labels: pd.DataFrame) -> pd.DataFrame:
    z10 = [f"z{i:02d}" for i in range(10)]
    z26 = [f"z{i:02d}" for i in range(26)]
    X26_tr = tr_l[z26].to_numpy(float)
    y_s = tr_l["sspg"].to_numpy(float)
    y_d = tr_l["di"].to_numpy(float)
    m_s = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(X26_tr, y_s)
    m_d = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))).fit(X26_tr, y_d)

    rows = []
    rng = np.random.default_rng(SEED)
    for name, zero_res, shuffle in [
        ("full_26d", False, False),
        ("zero_residual_16d", True, False),
        ("shuffle_residual_16d", False, True),
    ]:
        ps, pd_, sids = _subject_aggregate_preds(d4, z10, z26, m_s, m_d, rng, shuffle=shuffle, zero_res=zero_res)
        y_s_true = sids.astype(str)
        sspg_map = d4_labels["sspg_true"].to_dict()
        di_map = d4_labels["di_true"].to_dict()
        yt_s = np.array([float(sspg_map.get(s, np.nan)) for s in y_s_true])
        yt_d = np.array([float(di_map.get(s, np.nan)) for s in y_s_true])
        for tgt, y_true, y_pred in [("sspg", yt_s, ps), ("di", yt_d, pd_)]:
            m = _primary_metrics(y_true, y_pred)
            rows.append({"ablation": name, "target": tgt, **m})
    return pd.DataFrame(rows)


def cv_group_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> float:
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2 and y.dtype != float:
            continue
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 40)))
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        yt, pr = y[te], pred
        ok = np.isfinite(yt) & np.isfinite(pr)
        if ok.sum() < 3:
            continue
        scores.append(r2_score(yt[ok], pr[ok]))
    return float(np.mean(scores)) if scores else np.nan


def cv_group_acc_meal_type(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> float:
    classes = np.unique(y)
    if len(classes) < 2:
        return np.nan
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for tr, te in gkf.split(X, y, groups):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, multi_class="multinomial", random_state=SEED),
        )
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
    return float(np.mean(accs)) if accs else np.nan


def run_probes(df: pd.DataFrame, dataset: str, n_splits: int) -> pd.DataFrame:
    z_mech = [f"z{i:02d}" for i in range(10)]
    z_res = [f"z{i:02d}" for i in range(10, 26)]
    z_all = z_mech + z_res
    need = z_all + ["subject_id"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        return pd.DataFrame()
    d = df.dropna(subset=z_all).copy()
    if len(d) < 30:
        return pd.DataFrame()
    groups = d["subject_id"].astype(str).to_numpy()
    X_m = d[z_mech].to_numpy(float)
    X_r = d[z_res].to_numpy(float)
    X_a = d[z_all].to_numpy(float)
    rows = []
    reg_targets = [
        ("carb_g", "carb_g"),
        ("fat_g", "fat_g"),
        ("protein_g", "protein_g"),
        ("uncertainty_score", "uncertainty_score"),
    ]
    for _, col in reg_targets:
        if col not in d.columns:
            continue
        y = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 30:
            continue
        for block, Xb in [("mech10", X_m), ("res16", X_r), ("full26", X_a)]:
            r2v = cv_group_r2(Xb[ok], y[ok], groups[ok], n_splits=n_splits)
            rows.append({"dataset": dataset, "task": col, "block": block, "metric": "cv_r2_mean", "value": r2v})
    if "meal_type" in d.columns:
        yc = d["meal_type"].astype(str).to_numpy()
        for block, Xb in [("mech10", X_m), ("res16", X_r), ("full26", X_a)]:
            acc = cv_group_acc_meal_type(Xb, yc, groups, n_splits=n_splits)
            rows.append({"dataset": dataset, "task": "meal_type", "block": block, "metric": "cv_acc_mean", "value": acc})
    return pd.DataFrame(rows)


def run_stratified_icc(d4: pd.DataFrame) -> pd.DataFrame:
    z_mech = [f"z{i:02d}" for i in range(10)]
    z_res = [f"z{i:02d}" for i in range(10, 26)]
    need = z_mech + z_res + ["subject_id", "uncertainty_score", "meal_type"]
    d = d4.dropna(subset=z_mech + z_res + ["subject_id"]).copy()
    if len(d) < 20:
        return pd.DataFrame()
    d["unc_tert"] = pd.qcut(
        d["uncertainty_score"].rank(method="first"), q=3, labels=["low_unc", "mid_unc", "high_unc"], duplicates="drop"
    )
    d["z_res_norm"] = np.linalg.norm(d[z_res].to_numpy(float), axis=1)
    dims = [(f"z{i:02d}", f"z{i:02d}") for i in range(10)] + [("||z_res||16", "z_res_norm")]
    strata: List[Tuple[str, pd.DataFrame]] = [("all", d)]
    for mt, chunk in d.groupby("meal_type", observed=False):
        if len(chunk) >= 40 and chunk["subject_id"].nunique() >= 5:
            strata.append((f"meal_type={mt}", chunk))
    for ut in d["unc_tert"].dropna().unique():
        chunk = d[d["unc_tert"] == ut]
        if len(chunk) >= 40 and chunk["subject_id"].nunique() >= 5:
            strata.append((f"uncertainty={ut}", chunk))

    rows = []
    for sname, chunk in strata:
        g = chunk["subject_id"].astype(str).to_numpy()
        for dim_name, col in dims:
            if col == "z_res_norm":
                yv = chunk["z_res_norm"].to_numpy(float)
            else:
                yv = chunk[col].to_numpy(float)
            icc = icc1_unbalanced(yv, g)
            rows.append({"stratum": sname, "n_rows": len(chunk), "n_subjects": int(pd.Series(g).nunique()), "dim": dim_name, "icc1": icc})
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    latent_train = _build_latent_train()
    tr_l = latent_train.groupby("subject_id", as_index=False).median(numeric_only=True)

    d4_path = os.path.join(V22_ROOT, "v22_d4_meal_level_predictions.csv")
    d4 = pd.read_csv(d4_path)
    subj_lab = pd.read_csv(
        os.path.join(V22_ROOT, "v22_d4_subject_level_predictions.csv"), usecols=["subject_id", "sspg_true", "di_true"]
    )
    d4_labels = subj_lab.drop_duplicates("subject_id").set_index("subject_id")

    ab_df = run_ablation(d4, tr_l, d4_labels)
    ab_df.to_csv(os.path.join(OUT_ROOT, "v30_d4_residual_ablation_subject_metrics.csv"), index=False)

    n_sub = d4["subject_id"].nunique()
    n_splits = min(5, max(2, n_sub - 1))
    probe_rows = run_probes(d4, "D4", n_splits=n_splits)
    if os.path.isfile(D3_MERGED):
        d3 = pd.read_csv(D3_MERGED)
        n3 = d3["subject_id"].nunique()
        probe_rows = pd.concat(
            [probe_rows, run_probes(d3, "D3", n_splits=min(5, max(2, n3 - 1)))], ignore_index=True
        )
    probe_rows.to_csv(os.path.join(OUT_ROOT, "v30_context_probes_z10_vs_z16.csv"), index=False)

    icc_df = run_stratified_icc(d4)
    icc_df.to_csv(os.path.join(OUT_ROOT, "v30_d4_stratified_icc.csv"), index=False)

    summary = {
        "ablation_sspg_spearman": ab_df[ab_df["target"] == "sspg"]
        .set_index("ablation")["spearman"]
        .to_dict(),
        "ablation_di_spearman": ab_df[ab_df["target"] == "di"].set_index("ablation")["spearman"].to_dict(),
        "probe_file": "v30_context_probes_z10_vs_z16.csv",
        "icc_file": "v30_d4_stratified_icc.csv",
        "n_splits_d4": int(n_splits),
    }
    with open(os.path.join(OUT_ROOT, "v30_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    def _md_table(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except ImportError:
            return "```\n" + df.to_string(index=False) + "\n```"

    lines = [
        "# v30: residual ablation, context probes, stratified ICC",
        "",
        "## D4 readout ablation (subject-level mean pred vs gold; same Ridge26 trained on D1+D2)",
        _md_table(ab_df),
        "",
        "## Context probes (GroupKFold by subject; higher R² or acc ⇒ block carries more target info)",
        _md_table(probe_rows) if len(probe_rows) else "_no probe rows_",
        "",
        "## Stratified ICC(1,1) (within-subject repeatability of mech dims vs ||z_res||)",
        _md_table(icc_df.head(40)) if len(icc_df) else "_no icc rows_",
    ]
    if len(icc_df) > 40:
        lines.append(f"\n_(ICC table truncated in report; full CSV has {len(icc_df)} rows)_")
    with open(os.path.join(OUT_ROOT, "v30_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
