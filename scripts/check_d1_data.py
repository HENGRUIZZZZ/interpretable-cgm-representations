"""
诊断 D1 数据与金标准：多少人、多少有 labels、划分后 test 里几人带金标准。
在项目根目录运行: python scripts/check_d1_data.py
"""
import os
import sys
import numpy as np

# 与 run_paper1_full 一致
OUTPUT_BASE = os.environ.get("CGM_PROJECT_OUTPUT", "/Users/hertz1030/Downloads/cgm_project/output")

def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from load_cgm_project_data import load_cgm_project_level1_level2, split_by_subject
    from paper1_experiment_config import D1_TRAIN_FRAC, D1_VAL_FRAC, D1_TEST_FRAC, SPLIT_SEED

    print("=== D1 数据与金标准诊断 ===\n")
    print(f"OUTPUT_BASE: {OUTPUT_BASE}")
    data_dir = os.path.join(OUTPUT_BASE, "D1_metwally")
    if not os.path.isdir(data_dir):
        print(f"错误: 目录不存在 {data_dir}")
        print("请设置环境变量 CGM_PROJECT_OUTPUT 或把数据解压到上述路径。")
        return

    # 先直接看 subjects / labels / cgm 的 ID 是否一致
    import pandas as pd
    subjects_path = os.path.join(data_dir, "subjects.csv")
    labels_path = os.path.join(data_dir, "labels.csv")
    cgm_path = os.path.join(data_dir, "cgm.csv")
    sub_raw = pd.read_csv(subjects_path) if os.path.isfile(subjects_path) else None
    lab_raw = pd.read_csv(labels_path) if os.path.isfile(labels_path) else None
    cgm_raw = pd.read_csv(cgm_path, nrows=0) if os.path.isfile(cgm_path) else None
    if sub_raw is not None:
        print("\n--- 0. 原始表 ID 对照（subject_id 是否一致）---")
        print("  subjects.csv columns:", list(sub_raw.columns))
        sub_ids = sorted(sub_raw["subject_id"].astype(str).unique())
        print("  subjects.subject_id 数量:", len(sub_ids), "  示例:", sub_ids[:5], "...")
        if "original_id" in sub_raw.columns:
            orig_ids = sorted(sub_raw["original_id"].dropna().astype(str).unique())
            print("  subjects.original_id 数量:", len(orig_ids), "  示例:", orig_ids[:5], "...")
    if lab_raw is not None and "subject_id" in lab_raw.columns:
        lab_ids = sorted(lab_raw["subject_id"].astype(str).unique())
        print("  labels.subject_id 数量:", len(lab_ids), "  示例:", lab_ids[:5], "...")
        sspg_col = "sspg" if "sspg" in lab_raw.columns else ("SSPG" if "SSPG" in lab_raw.columns else None)
        n_gold = lab_raw[sspg_col].notna().sum() if sspg_col else 0
        print("  labels 中有 sspg/SSPG 的行数:", n_gold)
    if sub_raw is not None and lab_raw is not None and "original_id" in sub_raw.columns:
        # 是否 labels 用的是 original_id？
        sub_orig = set(sub_raw["original_id"].dropna().astype(str))
        lab_sub = set(lab_raw["subject_id"].astype(str))
        match_orig = sub_orig & lab_sub
        match_sub = set(sub_raw["subject_id"].astype(str)) & lab_sub
        print("  labels.subject_id 与 subjects.original_id 交集:", len(match_orig))
        print("  labels.subject_id 与 subjects.subject_id 交集:", len(match_sub))
    if cgm_raw is not None and "subject_id" in cgm_raw.columns:
        cgm_df = pd.read_csv(cgm_path)
        cgm_ids = sorted(cgm_df["subject_id"].astype(str).unique())
        print("  cgm.subject_id 数量:", len(cgm_ids), "  示例:", cgm_ids[:5], "...")

    batch, info, labels_df = load_cgm_project_level1_level2(
        dataset_id="D1", output_base=OUTPUT_BASE, num_meals_threshold=1
    )

    pids = np.asarray(info.patient_ids)
    subjects_in_batch = np.unique(pids)
    n_subjects = len(subjects_in_batch)
    n_samples = len(pids)

    print("\n--- 1. Batch 与样本数 ---")
    print(f"  样本数（meal 窗口）: {n_samples}")
    print(f"  受试者数（subject）: {n_subjects}")

    print("\n--- 2. labels.csv 概况 ---")
    if labels_df is None or labels_df.empty:
        print("  labels_df 为空或未找到 labels.csv")
    else:
        print(f"  shape: {labels_df.shape}")
        print(f"  columns: {list(labels_df.columns)}")
        if "subject_id" in labels_df.columns:
            n_labels_subj = labels_df["subject_id"].nunique()
            print(f"  labels 中 subject_id 去重人数: {n_labels_subj}")
        for col in ["sspg", "di", "homa_ir", "homa_b"]:
            if col in labels_df.columns:
                n_ok = labels_df[col].notna().sum()
                print(f"  {col}: 非空行数 = {n_ok}")

    print("\n--- 3. Batch 受试者 与 labels 的交集（金标准） ---")
    if labels_df is not None and not labels_df.empty and "subject_id" in labels_df.columns:
        labels_subj = set(labels_df["subject_id"].values)
        batch_subj = set(subjects_in_batch)
        # 注意：subject_id 可能是 int 或 str，统一转成 str 比较
        labels_subj_str = set(str(s) for s in labels_subj)
        batch_subj_str = set(str(s) for s in batch_subj)
        in_both = batch_subj_str & labels_subj_str
        print(f"  Batch 中受试者数: {len(batch_subj)}")
        print(f"  Labels 中 subject_id 数: {len(labels_subj)}")
        print(f"  两者交集（按字符串 id 匹配）: {len(in_both)}")
        for col in ["sspg", "di", "homa_ir", "homa_b"]:
            if col not in labels_df.columns:
                continue
            subj_with_gold = set(labels_df.dropna(subset=[col])["subject_id"].astype(str))
            both_with_gold = batch_subj_str & subj_with_gold
            print(f"  有 {col} 且在 batch 中的受试者数: {len(both_with_gold)}")

    print("\n--- 4. 划分 (70/15/15 by subject) ---")
    train_idx, val_idx, test_idx = split_by_subject(
        info, train_frac=D1_TRAIN_FRAC, val_frac=D1_VAL_FRAC, test_frac=D1_TEST_FRAC,
        seed=SPLIT_SEED, stratify_diagnosis=batch.diagnosis
    )
    train_sid = np.unique(pids[train_idx])
    val_sid = np.unique(pids[val_idx])
    test_sid = np.unique(pids[test_idx])
    print(f"  Train 受试者数: {len(train_sid)}")
    print(f"  Val   受试者数: {len(val_sid)}")
    print(f"  Test  受试者数: {len(test_sid)}")

    print("\n--- 5. Test 受试者中，有金标准的人数（当前相关分析用的 n）---")
    if labels_df is not None and not labels_df.empty and "subject_id" in labels_df.columns:
        test_sid_str = set(str(s) for s in test_sid)
        for col in ["sspg", "di", "homa_ir", "homa_b"]:
            if col not in labels_df.columns:
                continue
            subj_with_gold = set(labels_df.dropna(subset=[col])["subject_id"].astype(str))
            test_with_gold = test_sid_str & subj_with_gold
            print(f"  Test 中有 {col} 的受试者数: {len(test_with_gold)}  {list(test_with_gold)[:5]}{'...' if len(test_with_gold)>5 else ''}")
    else:
        print("  (无 labels 或无 subject_id，无法统计)")

    print("\n--- 6. subject_id 类型检查（若交集为 0 可能是类型不一致）---")
    if labels_df is not None and not labels_df.empty and "subject_id" in labels_df.columns:
        print(f"  labels_df['subject_id'].dtype: {labels_df['subject_id'].dtype}")
        print(f"  batch patient_ids (pids) dtype: {pids.dtype}")
        print(f"  labels 前 3 个 subject_id: {list(labels_df['subject_id'].head(3))}")
        print(f"  batch 前 3 个 patient_id: {list(subjects_in_batch[:3])}")

    print("\n=== 诊断结束 ===")

if __name__ == "__main__":
    main()
