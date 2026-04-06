"""
Load CGM project data (D1/D2 meal-centered format) into Batch format for Hybrid VAE.

Datasets D1–D5: 见 paper1_experiment_config.py。D1/D2 为餐心格式，D3/D4/D5 为连续 CGM。

Usage:
  # 方式一：直接指定 data_dir（原文件夹名）
  batch, patient_info, labels_df = load_cgm_project_level1_level2(
      data_dir="/path/to/cgm_project/output/D1_metwally",
      num_meals_threshold=1,
  )
  # 方式二：用 D1–D5 命名 + output_base
  batch, patient_info, labels_df = load_cgm_project_level1_level2(
      data_dir=None,
      dataset_id="D1",
      output_base="/path/to/cgm_project/output",
      num_meals_threshold=1,
  )
  # 按 subject 划分 train/val/test（用于 Paper 1 Level 1）
  train_idx, val_idx, test_idx = split_by_subject(patient_info, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=21)
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

# Match data_utils.py
MEAL_COVARIATES = [
    "total_grams",
    "total_carb",
    "total_sugar",
    "total_dietary_fiber",
    "total_fat",
    "total_protein",
]
DEMOGRAPHICS_COVARIATES = ["gender", "age", "weight"]


def _ensure_numeric(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")


def _resample_to_fixed_grid(
    cgm_arrays: list,
    timestamp_arrays: list,
    meal_series_arrays: list,
    grid_mins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample each (cgm, ts, meals) to common grid. Returns (cgm, ts, meals, kept_indices)."""
    out_cgm = []
    out_ts = []
    out_meals = []
    kept = []
    for i, (cgm, ts, meal) in enumerate(zip(cgm_arrays, timestamp_arrays, meal_series_arrays)):
        t_flat = ts[:, 0].astype(np.float64)
        _, idx = np.unique(t_flat, return_index=True)
        idx = np.sort(idx)
        t_flat = t_flat[idx]
        cgm_flat = cgm[idx, 0].astype(np.float64)
        meal = meal[idx]
        if len(t_flat) < 2:
            continue
        cgm_new = np.interp(grid_mins, t_flat, cgm_flat).astype(np.float32)[:, None]
        ts_new = grid_mins.astype(np.float32)[:, None]
        meal_new = np.broadcast_to(meal[0:1], (len(grid_mins), meal.shape[1])).astype(np.float32)
        out_cgm.append(cgm_new)
        out_ts.append(ts_new)
        out_meals.append(meal_new)
        kept.append(i)
    kept = np.array(kept, dtype=np.int64)
    return np.stack(out_cgm), np.stack(out_ts), np.stack(out_meals), kept


def load_cgm_project_level1_level2(
    data_dir: Optional[str] = None,
    dataset_id: Optional[str] = None,
    output_base: Optional[str] = None,
    num_meals_threshold: int = 1,
    nan_threshold: float = 0.0,
    default_height_m: float = 1.7,
    seed: int = 21,
    resample_5min: bool = True,
    grid_min_mins: int = -30,
    grid_max_mins: int = 180,
) -> Tuple["Batch", "PatientInfo", pd.DataFrame]:
    """
    Load D1 或 D2 餐心数据为 Batch。data_dir 与 (dataset_id + output_base) 二选一。

    - dataset_id: "D1" 或 "D2"（对应 D1_metwally, D2_stanford）；需同时提供 output_base。
    - data_dir: 直接指定目录，如 output/D1_metwally。

    Returns:
        batch: Batch(cgm, timestamps, meals, demographics, diagnosis)
        patient_info: PatientInfo(patient_ids=subject_id per sample, ...)
        labels_df: subject_id, sspg, di, homa_ir, homa_b 等
    """
    from data_utils import Batch, PatientInfo

    if data_dir is None:
        if dataset_id is None or output_base is None:
            raise ValueError("Provide either data_dir or both dataset_id and output_base.")
        from paper1_experiment_config import get_data_dir
        data_dir = get_data_dir(dataset_id, output_base)

    cgm_path = os.path.join(data_dir, "cgm.csv")
    meals_path = os.path.join(data_dir, "meals.csv")
    subjects_path = os.path.join(data_dir, "subjects.csv")
    labels_path = os.path.join(data_dir, "labels.csv")

    for p in (cgm_path, meals_path, subjects_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Expected file: {p}")

    cgm_df = pd.read_csv(cgm_path)
    meals_df = pd.read_csv(meals_path)
    subjects_df = pd.read_csv(subjects_path)
    labels_df = pd.read_csv(labels_path) if os.path.isfile(labels_path) else pd.DataFrame()

    # Normalize labels column names (V2 uses SSPG, DI, HOMA_B, HOMA_IR)
    if not labels_df.empty:
        _map = {"SSPG": "sspg", "DI": "di", "HOMA_IR": "homa_ir", "HOMA_B": "homa_b"}
        for old, new in _map.items():
            if old in labels_df.columns and new not in labels_df.columns:
                labels_df[new] = labels_df[old]

    # V2 format: D1 cgm has subject_id, timepoint_mins, glucose_mg_dl (no meal_id) -> one OGTT per subject
    if "timepoint_mins" in cgm_df.columns and "glucose_mg_dl" in cgm_df.columns and "meal_id" not in cgm_df.columns:
        cgm_df = cgm_df.rename(columns={"timepoint_mins": "mins_since_meal", "glucose_mg_dl": "glucose_mgdl"})
        cgm_df["meal_id"] = cgm_df["subject_id"].astype(str)  # one meal per subject
        time_col = "mins_since_meal"
        if "meal_id" not in meals_df.columns and "subject_id" in meals_df.columns:
            meals_df = meals_df.copy()
            meals_df["meal_id"] = meals_df["subject_id"].astype(str)
    # V2 D2 format: minutes_after_meal, glucose_mg_dl, no meal_id
    elif "minutes_after_meal" in cgm_df.columns:
        cgm_df = cgm_df.rename(columns={"minutes_after_meal": "mins_since_meal"})
        if "glucose_mg_dl" in cgm_df.columns and "glucose_mgdl" not in cgm_df.columns:
            cgm_df = cgm_df.rename(columns={"glucose_mg_dl": "glucose_mgdl"})
        if "meal_id" not in cgm_df.columns and "subject_id" in cgm_df.columns:
            cgm_df = cgm_df.copy()
            rep = cgm_df["rep"] if "rep" in cgm_df.columns else pd.Series(range(len(cgm_df)))
            cgm_df["meal_id"] = cgm_df["subject_id"].astype(str) + "_" + rep.astype(str)
        cgm_df = (
            cgm_df.groupby(["subject_id", "meal_id", "mins_since_meal"], as_index=False)["glucose_mgdl"]
            .mean()
        )
        time_col = "mins_since_meal"
    # D2 may have duplicate (meal_id, mins_since_meal) -> aggregate
    elif "mins_since_meal" in cgm_df.columns:
        if "glucose_mg_dl" in cgm_df.columns and "glucose_mgdl" not in cgm_df.columns:
            cgm_df = cgm_df.rename(columns={"glucose_mg_dl": "glucose_mgdl"})
        cgm_df = (
            cgm_df.groupby(["subject_id", "meal_id", "mins_since_meal"], as_index=False)["glucose_mgdl"]
            .mean()
        )
        time_col = "mins_since_meal"
    else:
        time_col = "timestamp"
        if "glucose_mg_dl" in cgm_df.columns and "glucose_mgdl" not in cgm_df.columns:
            cgm_df = cgm_df.rename(columns={"glucose_mg_dl": "glucose_mgdl"})
        if cgm_df[time_col].dtype == object:
            cgm_df[time_col] = pd.to_datetime(cgm_df[time_col])

    # Build per-meal CGM arrays (T, 1) and timestamps
    meal_ids = cgm_df["meal_id"].unique()
    cgm_arrays = []
    timestamp_arrays = []
    meal_series_arrays = []
    demographics_arrays = []
    all_diagnosis = []
    all_patient_ids = []

    meal_covariates_src = {
        "total_grams": ["carb_g", "protein_g", "fat_g"],
        "total_carb": ["carb_g"],
        "total_sugar": [],
        "total_dietary_fiber": ["fiber_g"],
        "total_fat": ["fat_g"],
        "total_protein": ["protein_g"],
    }

    for meal_id in meal_ids:
        cgm_meal = cgm_df[cgm_df["meal_id"] == meal_id].sort_values(time_col)
        if cgm_meal.empty or len(cgm_meal) < 10:
            continue
        subject_id = cgm_meal["subject_id"].iloc[0]
        if subject_id not in subjects_df["subject_id"].values:
            continue

        glucose = _ensure_numeric(cgm_meal["glucose_mgdl"]).values
        if np.any(np.isnan(glucose)):
            if np.mean(np.isnan(glucose)) > nan_threshold:
                continue
            glucose = np.nan_to_num(glucose, nan=np.nanmean(glucose))
        glucose = glucose[:, None]

        if time_col == "mins_since_meal":
            t = _ensure_numeric(cgm_meal["mins_since_meal"]).values[:, None]
        else:
            t = (
                cgm_meal[time_col].dt.hour * 60.0 + cgm_meal[time_col].dt.minute
            ).values[:, None]
        if np.any(np.isnan(t)):
            t = np.nan_to_num(t, nan=720.0)

        meal_row = meals_df[meals_df["meal_id"] == meal_id] if "meal_id" in meals_df.columns else pd.DataFrame()
        if meal_row.empty and "subject_id" in meals_df.columns:
            sub_meals = meals_df[meals_df["subject_id"] == subject_id]
            if not sub_meals.empty:
                meal_row = sub_meals.iloc[[0]]
        if meal_row.empty:
            meal_row = pd.DataFrame({c: [np.nan] for c in ["carb_g", "protein_g", "fat_g", "fiber_g"]})
        else:
            meal_row = meal_row.iloc[0]

        meal_series = np.zeros((len(glucose), len(MEAL_COVARIATES)))
        for i, cov in enumerate(MEAL_COVARIATES):
            src = meal_covariates_src.get(cov, [])
            if src:
                val = 0.0
                for s in src:
                    if s in meal_row:
                        v = meal_row[s]
                        val += float(pd.to_numeric(v, errors="coerce") or 0)
                meal_series[:, i] = val
            else:
                meal_series[:, i] = np.nan
        meal_series = np.nan_to_num(meal_series, nan=0.0)

        demo_row = subjects_df[subjects_df["subject_id"] == subject_id].iloc[0]
        gender = 1.0 if str(demo_row.get("sex", "M")).upper().startswith("F") else 0.0
        age = float(pd.to_numeric(demo_row.get("age"), errors="coerce") or 40.0)
        bmi = float(pd.to_numeric(demo_row.get("bmi"), errors="coerce") or 25.0)
        weight = float(
            pd.to_numeric(demo_row.get("weight"), errors="coerce")
            or (bmi * default_height_m ** 2)
        )
        demographics_arrays.append(np.array([gender, age, weight]))

        diag = str(demo_row.get("diagnosis", ""))
        if "T2D" in diag or "t2d" in diag:
            diagnosis = 1.0
        elif "Pre" in diag or "pre" in diag:
            diagnosis = 0.0
        else:
            diagnosis = 0.0
        all_diagnosis.append(diagnosis)
        all_patient_ids.append(subject_id)

        cgm_arrays.append(glucose)
        timestamp_arrays.append(t)
        meal_series_arrays.append(meal_series)

    if not cgm_arrays:
        raise ValueError("No valid meal windows found in " + data_dir)

    if resample_5min:
        grid_mins = np.arange(grid_min_mins, grid_max_mins + 1, 5, dtype=np.float64)
        all_cgm, all_ts, all_meals, kept_idx = _resample_to_fixed_grid(
            cgm_arrays, timestamp_arrays, meal_series_arrays, grid_mins
        )
        all_demo = np.stack(demographics_arrays)[kept_idx]
        all_diag = np.array(all_diagnosis, dtype=np.float64)[kept_idx]
        all_pids = np.array(all_patient_ids)[kept_idx]
    else:
        all_cgm = np.stack(cgm_arrays)
        all_ts = np.stack(timestamp_arrays)
        all_meals = np.stack(meal_series_arrays)
        all_demo = np.stack(demographics_arrays)
        all_diag = np.array(all_diagnosis, dtype=np.float64)
        all_pids = np.array(all_patient_ids)

    # Filter by num_meals_threshold per patient
    from collections import Counter
    pid_counts = Counter(all_pids)
    kept_pids = {p for p, c in pid_counts.items() if c >= num_meals_threshold}
    kept_idx = np.array([i for i, p in enumerate(all_pids) if p in kept_pids])
    if len(kept_idx) == 0:
        kept_idx = np.arange(len(all_pids))
    all_cgm = all_cgm[kept_idx]
    all_ts = all_ts[kept_idx]
    all_meals = all_meals[kept_idx]
    all_demo = all_demo[kept_idx]
    all_diag = all_diag[kept_idx]
    all_pids = all_pids[kept_idx]

    batch = Batch(
        cgm=all_cgm.astype(np.float32),
        timestamps=all_ts.astype(np.float32),
        meals=all_meals.astype(np.float32),
        demographics=all_demo.astype(np.float32),
        diagnosis=all_diag,
    )
    train_ids = all_pids
    test_ids = all_pids
    patient_info = PatientInfo(
        patient_ids=all_pids,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    return batch, patient_info, labels_df


def split_by_subject(
    patient_info: "PatientInfo",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 21,
    stratify_diagnosis: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按 subject 划分样本索引，保证同一 subject 的所有 meal 只出现在 train / val / test 之一。

    Args:
        patient_info: 来自 load_cgm_project_level1_level2，patient_ids 为每样本的 subject_id
        train_frac, val_frac, test_frac: 按人数比例，需满足 train_frac + val_frac + test_frac == 1
        stratify_diagnosis: 若提供，形状 (n_samples,) 与 patient_info.patient_ids 对齐，用于分层抽样（按 subject 取众数诊断）

    Returns:
        train_idx, val_idx, test_idx: 样本下标，可用于 batch.cgm[train_idx] 等
    """
    from sklearn.model_selection import train_test_split

    pids = np.asarray(patient_info.patient_ids)
    subjects = np.unique(pids)
    n_subj = len(subjects)
    if stratify_diagnosis is not None and len(stratify_diagnosis) == len(pids):
        # 每个 subject 取众数诊断用于分层
        subj_diag = []
        for s in subjects:
            mask = pids == s
            vals, counts = np.unique(stratify_diagnosis[mask], return_counts=True)
            subj_diag.append(vals[counts.argmax()])
        subj_diag = np.array(subj_diag)
        train_sid, test_sid = train_test_split(
            subjects, test_size=test_frac, random_state=seed, stratify=subj_diag
        )
        ratio = val_frac / (1.0 - test_frac)
        stratify_train = np.array([subj_diag[subjects == s][0] for s in train_sid])
        train_sid, val_sid = train_test_split(
            train_sid, test_size=ratio, random_state=seed, stratify=stratify_train
        )
    else:
        train_sid, test_sid = train_test_split(subjects, test_size=test_frac, random_state=seed)
        ratio = val_frac / (1.0 - test_frac)
        train_sid, val_sid = train_test_split(train_sid, test_size=ratio, random_state=seed)

    train_idx = np.where(np.isin(pids, train_sid))[0]
    val_idx = np.where(np.isin(pids, val_sid))[0]
    test_idx = np.where(np.isin(pids, test_sid))[0]
    return train_idx, val_idx, test_idx


def load_cgm_project_level3(
    data_dir: Optional[str] = None,
    dataset_id: Optional[str] = None,
    output_base: Optional[str] = None,
    grid_min_mins: int = -30,
    grid_max_mins: int = 180,
    min_cgm_points: int = 10,
    default_height_m: float = 1.7,
) -> Tuple["Batch", "PatientInfo", pd.DataFrame]:
    """
    加载 D3/D4/D5（连续 CGM）：按 meals 的 timestamp 切餐心窗口，重采样到固定网格，返回与 D1/D2 相同的 Batch 格式，便于用 Level 1 训好的 encoder 做 Level 3 encode。

    - cgm.csv 需为 subject_id, timestamp, glucose_mgdl（连续）
    - meals.csv 需有 subject_id, meal_id, timestamp, carb_g, protein_g, fat_g, fiber_g 等

    data_dir 与 (dataset_id + output_base) 二选一；dataset_id 为 "D3", "D4", "D5"。
    """
    from data_utils import Batch, PatientInfo

    if data_dir is None:
        if dataset_id is None or output_base is None:
            raise ValueError("Provide either data_dir or both dataset_id and output_base.")
        from paper1_experiment_config import get_data_dir
        data_dir = get_data_dir(dataset_id, output_base)

    cgm_path = os.path.join(data_dir, "cgm.csv")
    meals_path = os.path.join(data_dir, "meals.csv")
    subjects_path = os.path.join(data_dir, "subjects.csv")
    labels_path = os.path.join(data_dir, "labels.csv")
    for p in (cgm_path, meals_path, subjects_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Expected file: {p}")

    cgm_df = pd.read_csv(cgm_path)
    if "glucose_mg_dl" in cgm_df.columns and "glucose_mgdl" not in cgm_df.columns:
        cgm_df = cgm_df.rename(columns={"glucose_mg_dl": "glucose_mgdl"})
    cgm_df["timestamp"] = pd.to_datetime(cgm_df["timestamp"], errors="coerce")
    cgm_df = cgm_df.dropna(subset=["timestamp"])
    cgm_df = cgm_df.sort_values(["subject_id", "timestamp"])

    meals_df = pd.read_csv(meals_path)
    if "meal_id" not in meals_df.columns:
        meals_df = meals_df.copy()
        meals_df["meal_id"] = meals_df["subject_id"].astype(str) + "_" + meals_df["timestamp"].astype(str)
    meals_df["timestamp"] = pd.to_datetime(meals_df["timestamp"], errors="coerce")
    meals_df = meals_df.dropna(subset=["timestamp"])

    subjects_df = pd.read_csv(subjects_path)
    labels_df = pd.read_csv(labels_path) if os.path.isfile(labels_path) else pd.DataFrame()
    if not labels_df.empty:
        for old, new in [("SSPG", "sspg"), ("DI", "di"), ("HOMA_IR", "homa_ir"), ("HOMA_B", "homa_b")]:
            if old in labels_df.columns and new not in labels_df.columns:
                labels_df[new] = labels_df[old]
        # Do NOT use insulin_rate_dd as DI: D4's insulin_rate_dd is a different quantity
        # (range ~[-0.027, 0.149]) from Bergman DI (D1/D2 range ~[0.44, 6.58]). See P1审计与行动指南 v4.0.
        # if "di" not in labels_df.columns and "insulin_rate_dd" in labels_df.columns:
        #     labels_df["di"] = labels_df["insulin_rate_dd"]

    grid_mins = np.arange(grid_min_mins, grid_max_mins + 1, 5, dtype=np.float64)
    meal_covariates_src = {
        "total_grams": ["carb_g", "protein_g", "fat_g"],
        "total_carb": ["carb_g"],
        "total_sugar": [],
        "total_dietary_fiber": ["fiber_g"],
        "total_fat": ["fat_g"],
        "total_protein": ["protein_g"],
    }

    cgm_arrays = []
    timestamp_arrays = []
    meal_series_arrays = []
    demographics_arrays = []
    all_diagnosis = []
    all_patient_ids = []

    for _, meal_row in meals_df.iterrows():
        subject_id = meal_row["subject_id"]
        meal_id = meal_row["meal_id"]
        meal_ts = meal_row["timestamp"]
        if pd.isna(meal_ts):
            continue
        cgm_subj = cgm_df[cgm_df["subject_id"] == subject_id]
        if cgm_subj.empty:
            continue
        t_min = meal_ts + pd.Timedelta(minutes=grid_min_mins)
        t_max = meal_ts + pd.Timedelta(minutes=grid_max_mins)
        cgm_subj = cgm_subj[(cgm_subj["timestamp"] >= t_min) & (cgm_subj["timestamp"] <= t_max)]
        if len(cgm_subj) < min_cgm_points:
            continue
        cgm_subj = cgm_subj.sort_values("timestamp")
        t_rel = (cgm_subj["timestamp"] - meal_ts).dt.total_seconds() / 60.0
        t_flat = t_rel.values.astype(np.float64)
        glucose_flat = _ensure_numeric(cgm_subj["glucose_mgdl"]).values.astype(np.float64)
        if np.any(np.isnan(glucose_flat)):
            glucose_flat = np.nan_to_num(glucose_flat, nan=np.nanmean(glucose_flat))
        if len(t_flat) < 2:
            continue
        cgm_new = np.interp(grid_mins, t_flat, glucose_flat).astype(np.float32)[:, None]
        ts_new = grid_mins.astype(np.float32)[:, None]
        meal_series = np.zeros((len(grid_mins), len(MEAL_COVARIATES)))
        for i, cov in enumerate(MEAL_COVARIATES):
            src = meal_covariates_src.get(cov, [])
            val = 0.0
            for s in src:
                if s in meal_row:
                    v = pd.to_numeric(meal_row[s], errors="coerce")
                    val += float(v if pd.notna(v) else 0)
            meal_series[:, i] = val
        meal_series_arrays.append(meal_series.astype(np.float32))

        demo_row = subjects_df[subjects_df["subject_id"] == subject_id]
        if demo_row.empty:
            gender, age, bmi = 0.0, 40.0, 25.0
        else:
            demo_row = demo_row.iloc[0]
            gender = 1.0 if str(demo_row.get("sex", "M")).upper().startswith("F") else 0.0
            age = float(pd.to_numeric(demo_row.get("age"), errors="coerce") or 40.0)
            bmi = float(pd.to_numeric(demo_row.get("bmi"), errors="coerce") or 25.0)
        w = pd.to_numeric(demo_row.get("weight", np.nan), errors="coerce") if not isinstance(demo_row, pd.DataFrame) else np.nan
        weight = float(w) if (isinstance(demo_row, pd.Series) and pd.notna(w)) else (bmi * default_height_m ** 2)
        demographics_arrays.append(np.array([gender, age, weight], dtype=np.float32))
        all_diagnosis.append(0.0)
        all_patient_ids.append(subject_id)
        cgm_arrays.append(cgm_new)
        timestamp_arrays.append(ts_new)

    if not cgm_arrays:
        raise ValueError("No valid meal windows found in " + data_dir)

    all_cgm = np.stack(cgm_arrays)
    all_ts = np.stack(timestamp_arrays)
    all_meals = np.stack(meal_series_arrays)
    all_demo = np.stack(demographics_arrays)
    all_diag = np.array(all_diagnosis, dtype=np.float64)
    all_pids = np.array(all_patient_ids)

    batch = Batch(
        cgm=all_cgm,
        timestamps=all_ts,
        meals=all_meals,
        demographics=all_demo,
        diagnosis=all_diag,
    )
    patient_info = PatientInfo(
        patient_ids=all_pids,
        train_ids=all_pids,
        test_ids=all_pids,
    )
    return batch, patient_info, labels_df
