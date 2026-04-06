import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


SSPG_CUT = 120.0
DI_CUT = 1.2

# 26D latent columns as saved by run_p1_full_pipeline (ODE params + z_init + z_nonseq)
ODE_PARAMS_6 = ["tau_m", "Gb", "sg", "si", "p2", "mi"]
Z_INIT = [f"z_init_{i}" for i in range(4)]
Z_NONSEQ = [f"z_nonseq_{i}" for i in range(16)]
LATENT_26D = ODE_PARAMS_6 + Z_INIT + Z_NONSEQ


def infer_dataset_id(df: pd.DataFrame) -> pd.Series:
    if "dataset_id" in df.columns:
        return df["dataset_id"]
    if "subject_id" in df.columns:
        # Heuristic: dataset can often be inferred from subject_id prefix like "D1_", "D2_", "D4_"
        def _from_subj(s: str) -> str:
            if isinstance(s, str) and "_" in s:
                return s.split("_")[0]
            return "D?"

        return df["subject_id"].apply(_from_subj)
    raise ValueError("Cannot infer dataset_id (no dataset_id or subject_id column).")


def make_tri_class_labels(sspg: pd.Series, di: pd.Series) -> np.ndarray:
    """
    0: IS
    1: IR-compensated
    2: IR-decompensated
    -1: invalid (nan)
    """
    sspg = sspg.astype(float)
    di = di.astype(float)
    labels = np.full(len(sspg), -1, dtype=int)

    valid = (~sspg.isna()) & (~di.isna())
    hi_sspg = sspg >= SSPG_CUT
    hi_di = di >= DI_CUT

    # IS
    labels[valid & (~hi_sspg)] = 0
    # IR-compensated
    labels[valid & hi_sspg & hi_di] = 1
    # IR-decompensated
    labels[valid & hi_sspg & (~hi_di)] = 2

    return labels


def lodo_tri_class_accuracy(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "tri_class",
) -> Tuple[Dict[str, float], int]:
    ds = infer_dataset_id(df)
    datasets = sorted([d for d in ds.unique() if isinstance(d, str) and d.startswith("D")])

    if label_col in df.columns:
        y_all = df[label_col].to_numpy().astype(int)
    else:
        if "sspg" not in df.columns or "di" not in df.columns:
            raise ValueError("Need tri_class column or (sspg, di) to derive tri-class labels.")
        y_all = make_tri_class_labels(df["sspg"], df["di"])

    metrics: Dict[str, float] = {}
    total_n = 0

    for test_ds in datasets:
        train_idx = ds != test_ds
        test_idx = ds == test_ds
        if not test_idx.any() or not train_idx.any():
            continue

        X_train = df.loc[train_idx, feature_cols].to_numpy()
        y_train = y_all[train_idx.to_numpy()]
        X_test = df.loc[test_idx, feature_cols].to_numpy()
        y_test = y_all[test_idx.to_numpy()]

        # Filter invalid labels
        valid_train = y_train >= 0
        valid_test = y_test >= 0
        if valid_train.sum() < 5 or valid_test.sum() < 5:
            continue

        X_train = X_train[valid_train]
        y_train = y_train[valid_train]
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]

        clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        metrics[f"LODO_{test_ds}_acc"] = float(acc)
        total_n += len(y_test)

    if metrics:
        metrics["LODO_mean_acc"] = float(np.mean(list(metrics.values())))
    return metrics, total_n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_root",
        type=str,
        default="paper1_results_v9_cls",
        help="Root folder containing lambda_cls_* subdirs with latent_and_gold_all_26d.csv",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Optional path to save aggregated JSON metrics.",
    )
    args = ap.parse_args()

    root = args.results_root
    if not os.path.isdir(root):
        raise SystemExit(f"Results root not found: {root}")

    summary: Dict[str, Dict[str, float]] = {}

    for name in sorted(os.listdir(root)):
        subdir = os.path.join(root, name)
        if not os.path.isdir(subdir):
            continue
        csv_path = os.path.join(subdir, "latent_and_gold_all_26d.csv")
        if not os.path.isfile(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skip {name}: failed to read CSV ({e})")
            continue

        feature_cols = [c for c in LATENT_26D if c in df.columns]
        if not feature_cols:
            feature_cols = [c for c in df.columns if c.startswith("param_")]
        if not feature_cols:
            print(f"Skip {name}: no latent feature columns.")
            continue

        metrics, n = lodo_tri_class_accuracy(df, feature_cols)
        if not metrics:
            print(f"Skip {name}: no valid LODO splits.")
            continue

        summary[name] = metrics
        print(f"=== {name} ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print(f"(n_valid_test={n})")
        print()

    if args.out_json is not None and summary:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON summary to {args.out_json}")


if __name__ == "__main__":
    main()

