"""
V6 统一评估：对每条路线的特征矩阵做 100 次 5 折 CV（RidgeCV），记录 Spearman r；
两两 Wilcoxon 符号秩检验（Bonferroni p<0.003）。
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

_script_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
REPO_ROOT = os.path.dirname(_script_dir)
sys.path.insert(0, REPO_ROOT)
OUT_BASE = os.path.abspath(os.path.join(REPO_ROOT, "paper1_results_v6"))


def evaluate_route(X: np.ndarray, y_sspg: np.ndarray, y_di: np.ndarray, n_cv: int = 100, n_splits: int = 5):
    """X: (n_subjects, n_features). 返回 (rs_sspg, rs_di) 各 n_cv 个 r。"""
    rs_sspg, rs_di = [], []
    for seed in range(n_cv):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pred_s = np.full(len(y_sspg), np.nan)
        pred_d = np.full(len(y_di), np.nan)
        ok_s = np.isfinite(y_sspg)
        ok_d = np.isfinite(y_di)
        if ok_s.sum() < 10 or ok_d.sum() < 10:
            rs_sspg.append(np.nan)
            rs_di.append(np.nan)
            continue
        for tr, te in kf.split(X):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            if ok_s[tr].sum() >= 5:
                m_s = RidgeCV(alphas=np.logspace(-2, 2, 20)).fit(Xtr[ok_s[tr]], y_sspg[tr][ok_s[tr]])
                pred_s[te] = m_s.predict(Xte)
            if ok_d[tr].sum() >= 5:
                m_d = RidgeCV(alphas=np.logspace(-2, 2, 20)).fit(Xtr[ok_d[tr]], y_di[tr][ok_d[tr]])
                pred_d[te] = m_d.predict(Xte)
        r_s, _ = stats.spearmanr(pred_s[ok_s], y_sspg[ok_s], nan_policy="omit")
        r_d, _ = stats.spearmanr(pred_d[ok_d], y_di[ok_d], nan_policy="omit")
        rs_sspg.append(r_s)
        rs_di.append(r_d)
    return np.array(rs_sspg), np.array(rs_di)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default=os.path.join(OUT_BASE, "unified_eval"))
    p.add_argument("--n_cv", type=int, default=100)
    p.add_argument("--out_base", type=str, default=None, help="paper1_results_v6 目录（默认自动解析）")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_base = os.path.abspath(args.out_base) if args.out_base else OUT_BASE
    if not args.out_base:
        _check = os.path.join(out_base, "routeA_seed42", "latent_and_gold_all.csv")
        if not os.path.isfile(_check):
            for _cand in [os.path.join(os.getcwd(), "paper1_results_v6"), os.path.join(REPO_ROOT, "paper1_results_v6")]:
                _cand = os.path.abspath(_cand)
                if os.path.isfile(os.path.join(_cand, "routeA_seed42", "latent_and_gold_all.csv")):
                    out_base = _cand
                    break

    # 加载金标准（subject-level）
    from run_p1_full_pipeline import _stack_batches, GOLD_COLS
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
    gold = labels_combined.drop_duplicates(subset=["subject_id"], keep="first")
    gold = gold[["subject_id"] + [c for c in GOLD_COLS if c in labels_combined.columns]]
    if "sspg" not in gold.columns:
        gold["sspg"] = np.nan
    if "di" not in gold.columns:
        gold["di"] = np.nan
    subject_ids = gold["subject_id"].values
    y_sspg = gold["sspg"].values
    y_di = gold["di"].values

    results = {}
    # Route D: 2D CGM
    route_d_path = os.path.join(out_base, "routeD", "route_d_spearman.json")
    if os.path.isfile(route_d_path):
        with open(route_d_path) as f:
            d = json.load(f)
        results["D"] = {"sspg": np.array(d["sspg_spearman_r"]), "di": np.array(d["di_spearman_r"])}
        print("Route D loaded")
    else:
        from scripts.run_v6_route_d import main as run_d
        run_d()
        with open(route_d_path) as f:
            d = json.load(f)
        results["D"] = {"sspg": np.array(d["sspg_spearman_r"]), "di": np.array(d["di_spearman_r"])}

    # Route A/B: 聚合所有 10 个 seed 的 subject-level 特征（中位数跨 seed），再 100×5-fold
    SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    for route, n_cols, csv_name in [("A", 6, "latent_and_gold_all.csv"), ("B", 26, "latent_and_gold_all_26d.csv")]:
        per_seed_agg = []
        for seed in SEEDS:
            lat_path = os.path.join(out_base, f"route{route}_seed{seed}", csv_name)
            if not os.path.isfile(lat_path):
                continue
            df = pd.read_csv(lat_path)
            if route == "A":
                feat_cols = [c for c in ["tau_m", "Gb", "sg", "si", "p2", "mi"] if c in df.columns]
            else:
                feat_cols = [c for c in df.columns if c in ["tau_m","Gb","sg","si","p2","mi"] or c.startswith("z_init_") or c.startswith("z_nonseq_")]
            if len(feat_cols) < 2:
                feat_cols = [c for c in df.columns if c not in ["subject_id", "sample_ix", "sspg", "di", "homa_ir", "homa_b"] and (df[c].dtype == np.float64 or df[c].dtype == np.int64)][:n_cols]
            if "subject_id" not in df.columns or len(feat_cols) < 2:
                continue
            agg = df.groupby("subject_id")[feat_cols].median().reset_index()
            per_seed_agg.append(agg)
        if not per_seed_agg:
            print(f"Skip Route {route}: no seed CSVs found")
            continue
        feat_cols = [c for c in per_seed_agg[0].columns if c != "subject_id"]
        all_subs = gold["subject_id"].unique()
        rows = []
        for sid in all_subs:
            vals = []
            for a in per_seed_agg:
                r = a.loc[a["subject_id"] == sid, feat_cols]
                if len(r) > 0:
                    vals.append(r.values.flatten())
            if not vals:
                continue
            vals = np.array(vals)
            row = {"subject_id": sid, **dict(zip(feat_cols, np.nanmedian(vals, axis=0)))}
            rows.append(row)
        merged = pd.DataFrame(rows)
        merge = gold.merge(merged[["subject_id"] + feat_cols], on="subject_id", how="inner")
        ys = merge["sspg"].values if "sspg" in merge.columns else np.full(len(merge), np.nan)
        yd = merge["di"].values if "di" in merge.columns else np.full(len(merge), np.nan)
        X = merge[feat_cols].fillna(merge[feat_cols].median()).values
        rs_s, rs_d = evaluate_route(X, ys, yd, n_cv=args.n_cv)
        results[route] = {"sspg": rs_s, "di": rs_d}
        print(f"Route {route} ({len(per_seed_agg)} seeds): SSPG r median={np.nanmedian(rs_s):.4f}, DI r median={np.nanmedian(rs_d):.4f}")

    # Route C: PCA on B（取 n_components=5 的 100 次 r 作为 Route C）
    route_c_path = os.path.join(out_base, "routeC", "route_c_pca_spearman.json")
    if os.path.isfile(route_c_path):
        with open(route_c_path) as f:
            c_data = json.load(f)
        idx5 = next((i for i, n in enumerate(c_data["n_components"]) if n == 5), 0)
        results["C"] = {"sspg": np.array(c_data["sspg_spearman_r"][idx5]), "di": np.array(c_data["di_spearman_r"][idx5])}
        print("Route C (PCA n=5) loaded")
    # Route F: 38D
    route_f_path = os.path.join(out_base, "routeF", "route_f_spearman.json")
    if os.path.isfile(route_f_path):
        with open(route_f_path) as f:
            f_data = json.load(f)
        results["F"] = {"sspg": np.array(f_data["sspg_spearman_r"]), "di": np.array(f_data["di_spearman_r"])}
        print("Route F loaded")

    # Wilcoxon 两两比较（以 DI 为例）
    routes = list(results.keys())
    n_pairs = len(routes) * (len(routes) - 1) // 2
    bonferroni = 0.05 / max(n_pairs, 1)
    report = ["V6 Unified Eval", "=" * 50]
    for r in routes:
        report.append(f"Route {r} DI: median={np.nanmedian(results[r]['di']):.4f}")
    report.append("")
    for i, a in enumerate(routes):
        for b in routes[i + 1:]:
            va = results[a]["di"]
            vb = results[b]["di"]
            valid = np.isfinite(va) & np.isfinite(vb)
            if valid.sum() < 10:
                report.append(f"{a} vs {b}: N/A")
                continue
            stat, p = stats.wilcoxon(va[valid], vb[valid], alternative="two-sided")
            sig = " *" if p < bonferroni else ""
            report.append(f"{a} vs {b}: p={p:.6f}{sig}")
    text = "\n".join(report)
    print(text)
    with open(os.path.join(args.output_dir, "v6_unified_eval_report.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(args.output_dir, "v6_route_spearman.json"), "w") as f:
        out = {r: {"sspg": results[r]["sspg"].tolist(), "di": results[r]["di"].tolist()} for r in results}
        json.dump(out, f, indent=2)
    print(f"Saved {args.output_dir}/v6_unified_eval_report.txt")


if __name__ == "__main__":
    main()
