"""
P1 完整方案：合并 D1+D2+D4 训练，充分利用金标准数据，统一评估与出图。

流程：加载 D1/D2/D4 → 按 subject 合并 → 80/10/10 划分 → 训练 → 在 test 及全量金标准上
计算 latent–SSPG/DI 相关 → 保存模型、相关表、latent+金标准 CSV → 可接 plot_p1_results 出图。

用法：
  export CGM_PROJECT_OUTPUT=/path/to/cgm_project/output
  python run_p1_full_pipeline.py
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

OUTPUT_BASE = os.environ.get("CGM_PROJECT_OUTPUT", "/Users/hertz1030/Downloads/cgm_project/output")
NUM_EPOCHS = int(os.environ.get("P1_NUM_EPOCHS", "100"))
BATCH_SIZE = 32
LR = float(os.environ.get("P1_LR", "1e-2"))
# KL 权重：增大可减轻 M1 模式崩溃 (诊断 v1.0 建议 0.05–0.1)
BETA_HAT = float(os.environ.get("BETA_HAT", "0.01"))
SEED = int(os.environ.get("P1_SEED", "21"))
# IR / SSPG / DI 弱监督。M1 建议先降 λ 防模式崩溃：LAMBDA_SSPG=0.1 LAMBDA_DI=0.1
LAMBDA_IR = float(os.environ.get("LAMBDA_IR", "0.05"))
LAMBDA_SSPG = float(os.environ.get("LAMBDA_SSPG", "0.0"))
LAMBDA_DI = float(os.environ.get("LAMBDA_DI", "0.0"))
# V9 多任务学习：三分类辅助任务 (IS / IR-Comp / IR-Decomp)
LAMBDA_CLS = float(os.environ.get("LAMBDA_CLS", "0.0"))
# v10：可辨识性约束项，强制 z_init_0 贴近 Gb
LAMBDA_IDENTIFIABILITY = float(os.environ.get("P1_IDENT_LOSS_LAMBDA", "0.0"))
# 正交解耦：惩罚 si 与 mi 的相关系数平方，使二者学习不同信息 (解耦实验诊断报告 v1.0)
LAMBDA_ORTHO = float(os.environ.get("LAMBDA_ORTHO", "0.0"))
# v13 update3: 防止 si/mi 坍缩的多样性惩罚
LAMBDA_DIV = float(os.environ.get("LAMBDA_DIV", "0.0"))
# 对 SSPG/DI 真值 z-score 后再算 loss，预测时反标准化；可稳定 M1 多目标训练
P1_ZSCORE_TARGETS = os.environ.get("P1_ZSCORE_TARGETS", "").strip().lower() in ("1", "true", "yes")
# 解耦实验：SSPG 头仅用 si (第4维)，实现 si/mi 在潜在空间解耦 (P1解耦实验操作手册 v1.0)
P1_DECOUPLE_SSPG = os.environ.get("P1_DECOUPLE_SSPG", "").strip().lower() in ("1", "true", "yes")
# 系统性优化：DI 乘积约束 DI=SI×MI，去掉 DI 线性头，用 si*mi 直接预测 (P1系统性优化实验方案 v1.0)
P1_DI_PRODUCT_CONSTRAINT = os.environ.get("P1_DI_PRODUCT_CONSTRAINT", "").strip().lower() in ("1", "true", "yes")
# 尽善尽美 v2.0：对数乘积 DI = scale*(log(si)+log(mi))+bias，解决尺度不匹配
P1_DI_LOG_PRODUCT = os.environ.get("P1_DI_LOG_PRODUCT", "").strip().lower() in ("1", "true", "yes")
# 尽善尽美 v2.0：DI 头为 MLP(si, mi) 2→16→8→1
P1_DI_MLP_HEAD = os.environ.get("P1_DI_MLP_HEAD", "").strip().lower() in ("1", "true", "yes")
# V9 多任务：是否启用三分类辅助任务 (IS / IR-Comp / IR-Decomp)
P1_USE_TRI_CLASS = os.environ.get("P1_USE_TRI_CLASS", "").strip().lower() in ("1", "true", "yes")
# 尽善尽美 v2.0：学习率调度 CosineAnnealing
P1_USE_LR_SCHEDULER = os.environ.get("P1_USE_LR_SCHEDULER", "").strip().lower() in ("1", "true", "yes")
# P1 审计计划 1.2：每受试者仅保留 1 个餐窗，统一训练样本 (~127)，提高可复现性
P1_ONE_MEAL_PER_SUBJECT = os.environ.get("P1_ONE_MEAL_PER_SUBJECT", "").strip().lower() in ("1", "true", "yes")
# 实验方案 v4.0：保存 26D 全 latent (6 ODE + 4 z_init + 16 z_nonseq) 每样本，供三路对决用
P1_SAVE_26D_LATENT = os.environ.get("P1_SAVE_26D_LATENT", "").strip().lower() in ("1", "true", "yes")
# 实验方案 v5.0 终局之战：Prediction Head (z_init+z_nonseq→SSPG/DI)，与 P1_ONE_MEAL_PER_SUBJECT=0 联用
P1_V5_PREDICTION_HEAD = os.environ.get("P1_V5_PREDICTION_HEAD", "").strip().lower() in ("1", "true", "yes")
# 实验方案 V6：e2e_head 输入 26D 全 latent（6 ODE + 4 z_init + 16 z_nonseq）
P1_HEAD_USE_26D = os.environ.get("P1_HEAD_USE_26D", "").strip().lower() in ("1", "true", "yes")
# v8 Config A: e2e_head 输入仅 10D 机制特征（6 ODE + 4 init_state）
P1_V8_HEAD_10D = os.environ.get("P1_V8_HEAD_10D", "").strip().lower() in ("1", "true", "yes")
# v10：预测头输入是否 detach（切断监督梯度回流到 encoder）
P1_DETACH_HEAD_INPUT = os.environ.get("P1_DETACH_HEAD_INPUT", "").strip().lower() in ("1", "true", "yes")
# v11：当启用 detach 时，允许保留极小梯度（0=完全切断，0.01=soft detach）
P1_HEAD_GRAD_SCALE = float(os.environ.get("P1_HEAD_GRAD_SCALE", "0.0"))
# v12：在非 e2e 模式下，使用分离的 26D SSPG/DI 预测头
P1_SEPARATE_HEAD_26D = os.environ.get("P1_SEPARATE_HEAD_26D", "").strip().lower() in ("1", "true", "yes")
# V6 路线 E：仅微调 head，encoder 冻结，需先加载预训练模型
P1_FINETUNE_HEAD_ONLY = os.environ.get("P1_FINETUNE_HEAD_ONLY", "").strip().lower() in ("1", "true", "yes")
P1_PRETRAINED_MODEL = os.environ.get("P1_PRETRAINED_MODEL", "").strip()
P1_RESUME_CKPT = os.environ.get("P1_RESUME_CKPT", "").strip()
if P1_FINETUNE_HEAD_ONLY:
    P1_HEAD_USE_26D = True
RESULTS_DIR = os.environ.get("P1_RESULTS_DIR", "paper1_results")
PARAM_NAMES = ["tau_m", "Gb", "sg", "si", "p2", "mi"]
GOLD_COLS = ["sspg", "di", "homa_ir", "homa_b"]
# IR head 输入：param 的 [si, mi, tau_m] 即索引 3, 5, 0
IR_LATENT_IX = [3, 5, 0]


def _stack_batches(batch_list, info_list, labels_list, dataset_ids):
    """合并多个 (batch, info, labels) 为单一 batch + 统一 patient_ids 与 labels 表。"""
    from data_utils import Batch
    cgm_list = [b.cgm for b in batch_list]
    ts_list = [b.timestamps for b in batch_list]
    meals_list = [b.meals for b in batch_list]
    demo_list = [b.demographics for b in batch_list]
    diag_list = [b.diagnosis for b in batch_list]
    pid_list = [np.asarray(info.patient_ids) for info in info_list]
    batch = Batch(
        cgm=np.concatenate(cgm_list, axis=0),
        timestamps=np.concatenate(ts_list, axis=0),
        meals=np.concatenate(meals_list, axis=0),
        demographics=np.concatenate(demo_list, axis=0),
        diagnosis=np.concatenate(diag_list, axis=0),
    )
    pids = np.concatenate(pid_list, axis=0)
    # 统一 labels：每行 subject_id, dataset_id, sspg, di, ...
    rows = []
    for i, (lab, did) in enumerate(zip(labels_list, dataset_ids)):
        if lab is None or lab.empty:
            continue
        df = lab.copy()
        df["dataset_id"] = did
        rows.append(df)
    if not rows:
        labels_combined = pd.DataFrame(columns=["subject_id", "dataset_id"] + GOLD_COLS)
    else:
        labels_combined = pd.concat(rows, ignore_index=True)
    return batch, pids, labels_combined


def main():
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--lambda_sspg", type=float, default=None, help="Override LAMBDA_SSPG (V7 Route 1)")
    _ap.add_argument("--lambda_di", type=float, default=None, help="Override LAMBDA_DI (V7 Route 1)")
    _args, _ = _ap.parse_known_args()
    global LAMBDA_SSPG, LAMBDA_DI
    if _args.lambda_sspg is not None:
        LAMBDA_SSPG = float(_args.lambda_sspg)
    if _args.lambda_di is not None:
        LAMBDA_DI = float(_args.lambda_di)

    from load_cgm_project_data import (
        load_cgm_project_level1_level2,
        load_cgm_project_level3,
        split_by_subject,
    )
    from paper1_experiment_config import (
        get_data_dir,
        P1_FULL_TRAIN_DATASETS,
        P1_TRAIN_FRAC,
        P1_VAL_FRAC,
        P1_TEST_FRAC,
    )
    from data_utils import normalize_train_test, MEAL_COVARIATES, DEMOGRAPHICS_COVARIATES
    from models import MechanisticAutoencoder, count_params
    from utils import seed_everything

    # 可选：仅用 D1+D2 训练（P1_TRAIN_DATASETS=D1,D2），便于在 D1+D2 上得到更稳、更好的评估
    _train_ds_env = os.environ.get("P1_TRAIN_DATASETS", "").strip()
    if _train_ds_env:
        P1_TRAIN_DATASETS = [x.strip() for x in _train_ds_env.split(",") if x.strip()]
    else:
        P1_TRAIN_DATASETS = list(P1_FULL_TRAIN_DATASETS)

    seed_everything(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output base: {OUTPUT_BASE}")
    print(f"Train datasets: {P1_TRAIN_DATASETS}")

    # ---------- 1. 加载训练用数据集 ----------
    batch_list, info_list, labels_list, dataset_ids = [], [], [], []
    for did in P1_TRAIN_DATASETS:
        data_dir = get_data_dir(did, OUTPUT_BASE)
        if not os.path.isdir(data_dir):
            print(f"Skip {did}: {data_dir} not found")
            continue
        if did in ("D1", "D2"):
            b, info, lab = load_cgm_project_level1_level2(
                data_dir=data_dir, num_meals_threshold=1
            )
        else:
            try:
                b, info, lab = load_cgm_project_level3(dataset_id=did, output_base=OUTPUT_BASE)
            except Exception as e:
                print(f"Skip {did}: {e}")
                continue
        batch_list.append(b)
        info_list.append(info)
        labels_list.append(lab)
        dataset_ids.append(did)
        print(f"  {did}: {b.cgm.shape[0]} samples, {len(np.unique(info.patient_ids))} subjects")

    if len(batch_list) < 2:
        raise RuntimeError("Need at least D1 and D2 (and ideally D4) to run full pipeline.")

    batch, pids, labels_combined = _stack_batches(batch_list, info_list, labels_list, dataset_ids)
    # v5 终局之战：Prediction Head 模式强制使用全部 meal 窗口
    if P1_V5_PREDICTION_HEAD:
        print("  P1_V5_PREDICTION_HEAD=1: 使用全部 meal 窗口（忽略 P1_ONE_MEAL_PER_SUBJECT）")
    # 审计计划 1.2：每受试者仅保留一个样本（取每个 subject 首次出现的索引）
    if P1_ONE_MEAL_PER_SUBJECT and not P1_V5_PREDICTION_HEAD:
        _, first_ix = np.unique(pids, return_index=True)
        first_ix = np.sort(first_ix)
        from data_utils import Batch
        batch = Batch(
            cgm=batch.cgm[first_ix],
            timestamps=batch.timestamps[first_ix],
            meals=batch.meals[first_ix],
            demographics=batch.demographics[first_ix],
            diagnosis=batch.diagnosis[first_ix],
        )
        pids = pids[first_ix]
        print(f"  P1_ONE_MEAL_PER_SUBJECT=1: 保留每受试者 1 个餐窗 → {len(pids)} samples, {len(np.unique(pids))} subjects")
    # 每个样本对应的 subject-level 金标准（用于弱监督）；无则 NaN
    def _per_sample_from_labels(col: str) -> np.ndarray:
        arr = np.full(len(pids), np.nan, dtype=np.float32)
        if labels_combined.empty or col not in labels_combined.columns:
            return arr
        sid_str = labels_combined["subject_id"].astype(str)
        for i, sid in enumerate(pids):
            row = labels_combined[sid_str == str(sid)]
            if row.empty:
                continue
            v = row[col].iloc[0]
            if pd.notna(v) and np.isfinite(v):
                arr[i] = float(v)
        return arr

    homa_ir_per_sample = _per_sample_from_labels("homa_ir")
    sspg_per_sample = _per_sample_from_labels("sspg")
    di_per_sample = _per_sample_from_labels("di")
    # 三分类辅助标签：0=IS, 1=IR-Comp, 2=IR-Decomp, -1=无效
    def _tri_class_array(sspg_arr: np.ndarray, di_arr: np.ndarray, sspg_cut: float = 120.0, di_cut: float = 1.2) -> np.ndarray:
        tri = np.full_like(sspg_arr, -1, dtype=np.int64)
        valid = np.isfinite(sspg_arr) & np.isfinite(di_arr)
        if not valid.any():
            return tri
        sspg_v = sspg_arr.copy()
        di_v = di_arr.copy()
        # IS
        is_mask = valid & (sspg_v < sspg_cut)
        tri[is_mask] = 0
        # IR-Compensated
        comp_mask = valid & (sspg_v >= sspg_cut) & (di_v >= di_cut)
        tri[comp_mask] = 1
        # IR-Decompensated
        decomp_mask = valid & (tri < 0)
        tri[decomp_mask] = 2
        return tri
    tri_class_per_sample = _tri_class_array(sspg_per_sample, di_per_sample)
    n_ir = np.isfinite(homa_ir_per_sample).sum()
    n_sspg = np.isfinite(sspg_per_sample).sum()
    n_di = np.isfinite(di_per_sample).sum()
    print(f"  Samples with HOMA-IR for IR head: {n_ir} / {len(pids)}")
    print(f"  Samples with SSPG labels (for optional head): {n_sspg} / {len(pids)}")
    print(f"  Samples with DI labels (for optional head): {n_di} / {len(pids)}")
    # 构造 PatientInfo（train_ids/test_ids 先填全部，split 再定）
    from data_utils import PatientInfo
    info = PatientInfo(patient_ids=pids, train_ids=pids, test_ids=pids)
    n_total = len(pids)
    print(f"Pooled: {batch.cgm.shape[0]} samples, {n_total} subject-meal rows, {len(np.unique(pids))} unique subjects")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------- 2. 按 subject 划分 80/10/10 ----------
    train_idx, val_idx, test_idx = split_by_subject(
        info, train_frac=P1_TRAIN_FRAC, val_frac=P1_VAL_FRAC, test_frac=P1_TEST_FRAC,
        seed=SEED, stratify_diagnosis=batch.diagnosis
    )
    train_arrays = type(batch)(*(
        batch.cgm[train_idx], batch.timestamps[train_idx], batch.meals[train_idx],
        batch.demographics[train_idx], batch.diagnosis[train_idx],
    ))
    val_arrays = type(batch)(*(
        batch.cgm[val_idx], batch.timestamps[val_idx], batch.meals[val_idx],
        batch.demographics[val_idx], batch.diagnosis[val_idx],
    ))
    test_arrays = type(batch)(*(
        batch.cgm[test_idx], batch.timestamps[test_idx], batch.meals[test_idx],
        batch.demographics[test_idx], batch.diagnosis[test_idx],
    ))

    # ---------- 3. 归一化（仅用 train） ----------
    (train_cgm, train_ts, train_meals, train_demo), (val_cgm, val_ts, val_meals, val_demo), (train_means, train_stds) = normalize_train_test(
        (train_arrays.cgm, train_arrays.timestamps, train_arrays.meals, train_arrays.demographics),
        (val_arrays.cgm, val_arrays.timestamps, val_arrays.meals, val_arrays.demographics),
    )
    _, (test_cgm, test_ts, test_meals, test_demo), _ = normalize_train_test(
        (train_arrays.cgm, train_arrays.timestamps, train_arrays.meals, train_arrays.demographics),
        (test_arrays.cgm, test_arrays.timestamps, test_arrays.meals, test_arrays.demographics),
    )
    train_arrays = type(batch)(*(train_cgm, train_ts, train_meals, train_demo, train_arrays.diagnosis))
    val_arrays = type(batch)(*(val_cgm, val_ts, val_meals, val_demo, val_arrays.diagnosis))
    test_arrays = type(batch)(*(test_cgm, test_ts, test_meals, test_demo, test_arrays.diagnosis))
    train_mean, train_std = train_means, train_stds

    G_mean = torch.as_tensor(train_mean[0], dtype=torch.float, device=device)
    G_std = torch.as_tensor(train_std[0], dtype=torch.float, device=device)
    def remove_scale(G, mean=G_mean, std=G_std):
        return (G - mean) / std

    train_homa_ir = np.asarray(homa_ir_per_sample[train_idx], dtype=np.float32)
    train_sspg = np.asarray(sspg_per_sample[train_idx], dtype=np.float32)
    train_di = np.asarray(di_per_sample[train_idx], dtype=np.float32)
    train_tri = np.asarray(tri_class_per_sample[train_idx], dtype=np.int64)
    # SSPG/DI 训练集 z-score（仅用于 P1_ZSCORE_TARGETS；评估时反标准化）
    sspg_valid = train_sspg[np.isfinite(train_sspg)]
    di_valid = train_di[np.isfinite(train_di)]
    sspg_mean = float(np.nanmean(sspg_valid)) if len(sspg_valid) else 0.0
    sspg_std = float(np.nanstd(sspg_valid)) if len(sspg_valid) > 1 else 1.0
    if sspg_std <= 0:
        sspg_std = 1.0
    di_mean = float(np.nanmean(di_valid)) if len(di_valid) else 0.0
    di_std = float(np.nanstd(di_valid)) if len(di_valid) > 1 else 1.0
    if di_std <= 0:
        di_std = 1.0
    if P1_ZSCORE_TARGETS:
        print(f"  P1_ZSCORE_TARGETS=1: SSPG train mean={sspg_mean:.2f} std={sspg_std:.2f}, DI mean={di_mean:.4f} std={di_std:.4f}")
    if P1_DECOUPLE_SSPG:
        print("  P1_DECOUPLE_SSPG=1: SSPG 头仅用 si (1维)，解耦 si/mi")
    if LAMBDA_ORTHO > 0:
        print(f"  LAMBDA_ORTHO={LAMBDA_ORTHO}: 正交损失 (si 与 mi 相关^2)")
    train_tensors = [torch.as_tensor(x, dtype=torch.float, device=device) for x in (
        train_arrays.cgm, train_arrays.timestamps, train_arrays.meals, train_arrays.demographics, train_arrays.diagnosis
    )]
    train_tensors.append(torch.as_tensor(train_homa_ir, dtype=torch.float, device=device))
    train_tensors.append(torch.as_tensor(train_sspg, dtype=torch.float, device=device))
    train_tensors.append(torch.as_tensor(train_di, dtype=torch.float, device=device))
    # 三分类标签作为最后一个张量（int64）
    train_tensors.append(torch.as_tensor(train_tri, dtype=torch.long, device=device))
    val_tensors = [torch.as_tensor(x, dtype=torch.float, device=device) for x in (
        val_arrays.cgm, val_arrays.timestamps, val_arrays.meals, val_arrays.demographics, val_arrays.diagnosis
    )]
    test_tensors = [torch.as_tensor(x, dtype=torch.float, device=device) for x in (
        test_arrays.cgm, test_arrays.timestamps, test_arrays.meals, test_arrays.demographics, test_arrays.diagnosis
    )]
    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=min(BATCH_SIZE, len(train_tensors[0])),
        shuffle=True,
    )

    # ---------- 4. 模型与训练 ----------
    model = MechanisticAutoencoder(
        meal_size=len(MEAL_COVARIATES),
        demographics_size=len(DEMOGRAPHICS_COVARIATES),
        embedding_size=8,
        hidden_size=32,
        num_layers=2,
        encoder_dropout_prob=0.0,
        decoder_dropout_prob=0.5,
    ).to(device)
    if P1_FINETUNE_HEAD_ONLY and P1_PRETRAINED_MODEL and os.path.isfile(P1_PRETRAINED_MODEL):
        ckpt = torch.load(P1_PRETRAINED_MODEL, map_location=device, weights_only=False)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=False)
        for p in model.parameters():
            p.requires_grad = False
        print(f" [V6 Route E] 已加载预训练模型并冻结 encoder: {P1_PRETRAINED_MODEL}")
    ir_head = torch.nn.Linear(len(IR_LATENT_IX), 1).to(device)
    # v5 终局之战：使用模型内建 prediction_head，不再创建独立 sspg/di 头
    # V6 路线B：e2e_head 输入 26D 全 latent (6+4+16)
    n_ode_params = len(PARAM_NAMES)
    z_init_dim = 4
    z_nonseq_dim = 16
    e2e_head = None
    if P1_V5_PREDICTION_HEAD:
        sspg_head = None
        di_head = None
        print(" [V5] 使用模型内建 prediction_head (z_init+z_nonseq→SSPG/DI)")
    elif P1_HEAD_USE_26D:
        sspg_head = None
        di_head = None
        if P1_V8_HEAD_10D:
            head_input_dim = n_ode_params + z_init_dim
        else:
            head_input_dim = n_ode_params + z_init_dim + z_nonseq_dim
        e2e_head = torch.nn.Sequential(
            torch.nn.Linear(head_input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        ).to(device)
        if P1_V8_HEAD_10D:
            print(f" [V8 Config A] e2e_head 使用 10D 机制特征 (dim={head_input_dim})")
        else:
            print(f" [V6/V8] e2e_head 使用 26D 全 latent (dim={head_input_dim})")
    else:
        if P1_SEPARATE_HEAD_26D:
            head_input_dim = n_ode_params + z_init_dim + z_nonseq_dim
            sspg_head = torch.nn.Sequential(
                torch.nn.Linear(head_input_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(32, 1),
            ).to(device)
            di_head = torch.nn.Sequential(
                torch.nn.Linear(head_input_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(32, 1),
            ).to(device)
            print(f" [V12] 分离预测头: SSPG/DI head 输入 26D (dim={head_input_dim})")
        else:
            # 预测头：SSPG 解耦模式下仅用 si (1维)；否则 6D。DI：MLP(si,mi) / LogProduct(1,1) / 乘积无头 / 6D线性
            sspg_head = torch.nn.Linear(1 if P1_DECOUPLE_SSPG else len(PARAM_NAMES), 1).to(device)
    if not P1_V5_PREDICTION_HEAD and P1_DI_MLP_HEAD:
        di_head = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        ).to(device)
        print(" [MLP_HEAD] DI head: MLP(2→16→8→1) on [si, mi]")
    elif not P1_V5_PREDICTION_HEAD and P1_DI_LOG_PRODUCT:
        di_head = torch.nn.Linear(1, 1).to(device)
        print(" [LOG-PRODUCT] DI head: Linear(1,1) on log(si)+log(mi)")
    elif P1_DI_PRODUCT_CONSTRAINT and not P1_V5_PREDICTION_HEAD and not P1_SEPARATE_HEAD_26D:
        di_head = None
    elif not P1_V5_PREDICTION_HEAD and not P1_SEPARATE_HEAD_26D:
        di_head = torch.nn.Linear(len(PARAM_NAMES), 1).to(device)
    # 三分类辅助任务 head（基于 6D ODE latent）
    if LAMBDA_CLS > 0.0 and P1_USE_TRI_CLASS:
        cls_head = torch.nn.Sequential(
            torch.nn.Linear(len(PARAM_NAMES), 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(16, 3),
        ).to(device)
        print(" [V9] tri-class head: 6D latent -> 3 classes")
    else:
        cls_head = None

    if P1_RESUME_CKPT and os.path.isfile(P1_RESUME_CKPT):
        rckpt = torch.load(P1_RESUME_CKPT, map_location=device, weights_only=False)
        if "model_state" in rckpt:
            model.load_state_dict(rckpt["model_state"], strict=False)
        if ir_head is not None and "ir_head_state" in rckpt:
            ir_head.load_state_dict(rckpt["ir_head_state"], strict=False)
        if sspg_head is not None and "sspg_head_state" in rckpt:
            sspg_head.load_state_dict(rckpt["sspg_head_state"], strict=False)
        if di_head is not None and "di_head_state" in rckpt:
            di_head.load_state_dict(rckpt["di_head_state"], strict=False)
        if e2e_head is not None and "e2e_head_state" in rckpt:
            e2e_head.load_state_dict(rckpt["e2e_head_state"], strict=False)
        if cls_head is not None and "cls_head_state" in rckpt:
            cls_head.load_state_dict(rckpt["cls_head_state"], strict=False)
        print(f" [RESUME] 从 checkpoint 继续训练: {P1_RESUME_CKPT}")

    if P1_V5_PREDICTION_HEAD:
        opt_params = list(model.parameters())
    elif P1_HEAD_USE_26D and e2e_head is not None:
        if P1_FINETUNE_HEAD_ONLY:
            opt_params = list(e2e_head.parameters())
        else:
            opt_params = list(model.parameters()) + list(e2e_head.parameters())
    else:
        opt_params = list(model.parameters()) + list(ir_head.parameters()) + list(sspg_head.parameters())
        if di_head is not None:
            opt_params += list(di_head.parameters())
        if cls_head is not None:
            opt_params += list(cls_head.parameters())
    lr_use = 1e-3 if P1_FINETUNE_HEAD_ONLY else LR
    optimizer = torch.optim.AdamW(opt_params, lr=lr_use)
    if P1_FINETUNE_HEAD_ONLY:
        print(f" [V6 Route E] 仅优化 e2e_head, lr={lr_use}")
    if P1_USE_LR_SCHEDULER:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
        print(f" [LR_SCHEDULER] CosineAnnealing: T_max={NUM_EPOCHS}, eta_min=1e-5")
    else:
        scheduler = None
    print(f"Params: {count_params(model)} (+ IR/SSPG/DI heads)")
    sspg_mean_t = torch.tensor(sspg_mean, dtype=torch.float, device=device)
    sspg_std_t = torch.tensor(sspg_std, dtype=torch.float, device=device)
    di_mean_t = torch.tensor(di_mean, dtype=torch.float, device=device)
    di_std_t = torch.tensor(di_std, dtype=torch.float, device=device)

    def loss_fn(output, seq_q, nonseq_q, seq_p, nonseq_p, cgm):
        states = output.states
        pred_cgm = remove_scale(states[..., 0:1])
        mse = (pred_cgm - cgm).pow(2).mean()
        seq_kl = torch.distributions.kl.kl_divergence(seq_q, seq_p).sum((0, 1, 2))
        nonseq_kl = torch.distributions.kl.kl_divergence(nonseq_q, nonseq_p).sum((0, 1))
        N, T = pred_cgm.shape[:2]
        M = states.shape[-1]
        kl = seq_kl / (N * T) + M * nonseq_kl / (N * T)
        return mse + BETA_HAT * kl

    print("\n--- Training ---")
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_recon_losses = []
    epoch_sspg_losses = []
    epoch_di_losses = []
    epoch_si_cvs = []
    epoch_mi_cvs = []
    def _head_input_with_grad_control(x: torch.Tensor) -> torch.Tensor:
        if not P1_DETACH_HEAD_INPUT:
            return x
        scale = float(P1_HEAD_GRAD_SCALE)
        if scale <= 0.0:
            return x.detach()
        if scale >= 1.0:
            return x
        # Forward equals x; gradient to encoder scaled by `scale` (soft detach).
        return x.detach() + scale * (x - x.detach())

    for epoch in range(NUM_EPOCHS):
        model.train()
        if not P1_V5_PREDICTION_HEAD and not P1_HEAD_USE_26D:
            ir_head.train()
            sspg_head.train()
            if di_head is not None:
                di_head.train()
        if e2e_head is not None:
            e2e_head.train()
        train_loss = 0.0
        recon_loss_epoch = 0.0
        sspg_loss_epoch = 0.0
        di_loss_epoch = 0.0
        for tup in train_loader:
            cgm, timestamps, meals, demographics, _, homa_ir_b, sspg_b, di_b, tri_b = tup
            optimizer.zero_grad()
            output, seq_q, nonseq_q, e2e_pred = model(cgm, timestamps, meals, demographics)
            latent_all = output.param  # (batch, 6)
            # V6 路线B：e2e_head 输入 26D，在 pipeline 中计算
            if P1_HEAD_USE_26D and e2e_head is not None and hasattr(model, "get_all_latents_for_head"):
                p26, init26, z16 = model.get_all_latents_for_head(cgm, timestamps, meals, demographics)
                head_in = torch.cat([p26, init26], dim=-1) if P1_V8_HEAD_10D else torch.cat([p26, init26, z16], dim=-1)
                head_in = _head_input_with_grad_control(head_in)
                e2e_pred = e2e_head(head_in)
            loss = loss_fn(output, seq_q, nonseq_q, model.seq_p, model.nonseq_p, cgm)
            recon_part = (remove_scale(output.states[..., 0:1]) - cgm).pow(2).mean()
            batch_sspg_loss = torch.tensor(0.0, device=device)
            batch_di_loss = torch.tensor(0.0, device=device)
            if LAMBDA_IDENTIFIABILITY > 0.0:
                z_init_0 = output.states[:, 0, 0]
                gb = output.param[:, 1]
                valid_ident = torch.isfinite(z_init_0) & torch.isfinite(gb)
                if valid_ident.any():
                    ident_loss = torch.nn.functional.mse_loss(z_init_0[valid_ident], gb[valid_ident])
                    loss = loss + LAMBDA_IDENTIFIABILITY * ident_loss
            # v5 终局之战：使用模型内建 prediction_head 的 SSPG/DI 损失（仅对有标签样本）
            if P1_V5_PREDICTION_HEAD and e2e_pred is not None:
                if LAMBDA_SSPG > 0:
                    valid_sspg = torch.isfinite(sspg_b)
                    if valid_sspg.any():
                        batch_sspg_loss = (e2e_pred[valid_sspg, 0] - sspg_b[valid_sspg]).pow(2).mean()
                        loss = loss + LAMBDA_SSPG * batch_sspg_loss
                if LAMBDA_DI > 0:
                    valid_di = torch.isfinite(di_b)
                    if valid_di.any():
                        batch_di_loss = (e2e_pred[valid_di, 1] - di_b[valid_di]).pow(2).mean()
                        loss = loss + LAMBDA_DI * batch_di_loss
            elif P1_HEAD_USE_26D and e2e_pred is not None:
                if LAMBDA_SSPG > 0:
                    valid_sspg = torch.isfinite(sspg_b)
                    if valid_sspg.any():
                        target_sspg = (sspg_b[valid_sspg] - sspg_mean_t) / sspg_std_t if P1_ZSCORE_TARGETS else sspg_b[valid_sspg]
                        batch_sspg_loss = (e2e_pred[valid_sspg, 0] - target_sspg).pow(2).mean()
                        loss = loss + LAMBDA_SSPG * batch_sspg_loss
                if LAMBDA_DI > 0:
                    valid_di = torch.isfinite(di_b)
                    if valid_di.any():
                        target_di = (di_b[valid_di] - di_mean_t) / di_std_t if P1_ZSCORE_TARGETS else di_b[valid_di]
                        batch_di_loss = (e2e_pred[valid_di, 1] - target_di).pow(2).mean()
                        loss = loss + LAMBDA_DI * batch_di_loss
            else:
                # IR 弱监督：仅对 HOMA_IR 有效的样本算 MSE(log1p(HOMA_IR), ir_head(si, mi, tau_m))
                valid = torch.isfinite(homa_ir_b) & (homa_ir_b >= 0)
                if valid.any():
                    ir_in = output.param[:, IR_LATENT_IX]
                    ir_pred = ir_head(ir_in).squeeze(-1)
                    target = torch.log1p(homa_ir_b.clamp(min=0))
                    loss_ir = (ir_pred[valid] - target[valid]).pow(2).mean()
                    loss = loss + LAMBDA_IR * loss_ir
                # SSPG / DI 预测头弱监督：仅对有标签样本算 MSE
                latent_all = _head_input_with_grad_control(latent_all)
                latent_26d = None
                if P1_SEPARATE_HEAD_26D and hasattr(model, "get_all_latents_for_head"):
                    p26_h, init26_h, z16_h = model.get_all_latents_for_head(cgm, timestamps, meals, demographics)
                    latent_26d = _head_input_with_grad_control(torch.cat([p26_h, init26_h, z16_h], dim=-1))
                if LAMBDA_SSPG > 0.0 and sspg_head is not None:
                    valid_sspg = torch.isfinite(sspg_b)
                    if valid_sspg.any():
                        if latent_26d is not None:
                            sspg_in = latent_26d
                        else:
                            sspg_in = latent_all[:, 3:4] if P1_DECOUPLE_SSPG else latent_all
                        sspg_pred = sspg_head(sspg_in).squeeze(-1)
                        target_sspg = (sspg_b[valid_sspg] - sspg_mean_t) / sspg_std_t if P1_ZSCORE_TARGETS else sspg_b[valid_sspg]
                        loss_sspg = (sspg_pred[valid_sspg] - target_sspg).pow(2).mean()
                        batch_sspg_loss = loss_sspg
                        loss = loss + LAMBDA_SSPG * loss_sspg
                if LAMBDA_DI > 0.0 and di_head is not None:
                    valid_di = torch.isfinite(di_b)
                    if valid_di.any():
                        if latent_26d is not None:
                            di_pred = di_head(latent_26d).squeeze(-1)
                        elif P1_DI_MLP_HEAD:
                            si_mi_input = latent_all[:, [3, 5]]
                            di_pred = di_head(si_mi_input).squeeze(-1)
                        elif P1_DI_LOG_PRODUCT:
                            log_si = torch.log(latent_all[:, 3] + 1e-12)
                            log_mi = torch.log(latent_all[:, 5] + 1e-12)
                            log_sum = (log_si + log_mi).unsqueeze(-1)
                            di_pred = di_head(log_sum).squeeze(-1)
                        elif P1_DI_PRODUCT_CONSTRAINT:
                            di_pred = latent_all[:, 3] * latent_all[:, 5]
                        else:
                            di_pred = di_head(latent_all).squeeze(-1)
                        target_di = (di_b[valid_di] - di_mean_t) / di_std_t if P1_ZSCORE_TARGETS else di_b[valid_di]
                        loss_di = (di_pred[valid_di] - target_di).pow(2).mean()
                        batch_di_loss = loss_di
                        loss = loss + LAMBDA_DI * loss_di
            # 三分类辅助任务：Cross-Entropy(ignore_index=-1)
            if LAMBDA_CLS > 0.0 and cls_head is not None and P1_USE_TRI_CLASS:
                latent_cls = _head_input_with_grad_control(latent_all)
                valid_tri = tri_b >= 0
                if valid_tri.any():
                    logits = cls_head(latent_cls[valid_tri])
                    cls_loss = torch.nn.functional.cross_entropy(logits, tri_b[valid_tri].long())
                    loss = loss + LAMBDA_CLS * cls_loss
            if LAMBDA_DIV > 0.0:
                si_vec = output.param[:, 3]
                mi_vec = output.param[:, 5]
                si_cv_div = si_vec.std() / (si_vec.abs().mean() + 1e-8)
                mi_cv_div = mi_vec.std() / (mi_vec.abs().mean() + 1e-8)
                if torch.isfinite(si_cv_div) and torch.isfinite(mi_cv_div):
                    loss = loss - LAMBDA_DIV * (si_cv_div + mi_cv_div)
            # 正交损失：惩罚 si 与 mi 的皮尔逊相关，促进解耦
            if LAMBDA_ORTHO > 0.0:
                si_vec = latent_all[:, 3]
                mi_vec = latent_all[:, 5]
                vx = si_vec - si_vec.mean()
                vy = mi_vec - mi_vec.mean()
                denom = vx.pow(2).sum().sqrt() * vy.pow(2).sum().sqrt() + 1e-8
                corr = (vx * vy).sum() / denom
                loss_ortho = corr.pow(2)
                if torch.isfinite(loss_ortho):
                    loss = loss + LAMBDA_ORTHO * loss_ortho
            loss.backward()
            clip_params = list(model.parameters())
            if P1_HEAD_USE_26D and e2e_head is not None:
                clip_params += list(e2e_head.parameters())
            elif not P1_V5_PREDICTION_HEAD:
                clip_params += list(ir_head.parameters()) + list(sspg_head.parameters())
                if di_head is not None:
                    clip_params += list(di_head.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            optimizer.step()
            train_loss += loss.item() * len(cgm)
            recon_loss_epoch += recon_part.item() * len(cgm)
            sspg_loss_epoch += batch_sspg_loss.item() * len(cgm)
            di_loss_epoch += batch_di_loss.item() * len(cgm)
        train_loss /= len(train_tensors[0])
        recon_loss_epoch /= len(train_tensors[0])
        sspg_loss_epoch /= len(train_tensors[0])
        di_loss_epoch /= len(train_tensors[0])
        with torch.no_grad():
            model.eval()
            out_v, _, _, _ = model(*val_tensors[:4])
            pred_v = remove_scale(out_v.states[..., 0:1])
            val_loss = (pred_v - val_tensors[0]).pow(2).mean().item()
            out_tr, _, _, _ = model(*train_tensors[:4])
            si_vec = out_tr.param[:, 3]
            mi_vec = out_tr.param[:, 5]
            si_cv = (si_vec.std() / (si_vec.abs().mean() + 1e-8)).item()
            mi_cv = (mi_vec.std() / (mi_vec.abs().mean() + 1e-8)).item()
        epoch_train_losses.append(float(train_loss))
        epoch_val_losses.append(float(val_loss))
        epoch_recon_losses.append(float(recon_loss_epoch))
        epoch_sspg_losses.append(float(sspg_loss_epoch))
        epoch_di_losses.append(float(di_loss_epoch))
        epoch_si_cvs.append(float(si_cv))
        epoch_mi_cvs.append(float(mi_cv))
        if scheduler is not None:
            scheduler.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}  Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                gb_mean_dbg = float(out_tr.param[:, 1].mean().item())
                si_cv_dbg = float(si_cv)
                print(f"  [diag] epoch={epoch+1} Gb_mean={gb_mean_dbg:.1f} si_cv={si_cv_dbg:.3f} sspg_loss={sspg_loss_epoch:.4f} di_loss={di_loss_epoch:.4f}")

    # ---------- 4b. 验证集重建：每样本 MSE + 若干示例曲线（供 VAE 拟合诊断） ----------
    with torch.no_grad():
        model.eval()
        val_cgm_t = val_tensors[0]
        val_mse_list = []
        batch_size_val = min(64, len(val_cgm_t))
        for start in range(0, len(val_cgm_t), batch_size_val):
            end = min(start + batch_size_val, len(val_cgm_t))
            out_v, _, _, _ = model(
                val_tensors[0][start:end], val_tensors[1][start:end],
                val_tensors[2][start:end], val_tensors[3][start:end],
            )
            pred_v = remove_scale(out_v.states[..., 0:1])
            cgm_v = val_tensors[0][start:end]
            mse_per = ((pred_v - cgm_v).pow(2).mean(dim=(1, 2))).cpu().numpy()
            val_mse_list.append(mse_per)
    val_mse_all = np.concatenate(val_mse_list, axis=0)
    np.save(os.path.join(RESULTS_DIR, "reconstruction_val_mse.npy"), val_mse_all)
    # 保存 3 条示例：actual vs pred CGM（用于出图）
    n_ex = min(3, len(val_cgm_t))
    with torch.no_grad():
        out_ex, _, _, _ = model(val_tensors[0][:n_ex], val_tensors[1][:n_ex], val_tensors[2][:n_ex], val_tensors[3][:n_ex])
        pred_ex = remove_scale(out_ex.states[..., 0:1]).cpu().numpy()
    np.savez(
        os.path.join(RESULTS_DIR, "reconstruction_examples.npz"),
        actual=val_tensors[0][:n_ex].cpu().numpy(),
        pred=pred_ex,
    )

    # ---------- 5. Test 上 latent，与金标准对齐 ----------
    with torch.no_grad():
        model.eval()
        output_test, _, _, _ = model(*test_tensors[:4])
    param_test = output_test.param.cpu().numpy()
    test_pids = pids[test_idx]
    unique_test = np.unique(test_pids)
    param_by_subject = {}
    for sid in unique_test:
        mask = test_pids == sid
        param_by_subject[sid] = param_test[mask].mean(axis=0)

    # 合并 labels：subject_id 在 D1/D2/D4 中唯一，用 subject_id 查
    def get_gold_for_subject(sid):
        row = labels_combined[labels_combined["subject_id"].astype(str) == str(sid)]
        if row.empty:
            return None
        row = row.iloc[0]
        return {c: row.get(c) for c in GOLD_COLS if c in row.index}

    # 构建 latent + gold 表（test 及后续可全量再跑一次）
    def build_latent_gold_table(pids_subset, param_by_sid):
        rows = []
        for sid in np.unique(pids_subset):
            if sid not in param_by_sid:
                continue
            gold = get_gold_for_subject(sid)
            if gold is None:
                continue
            par = param_by_sid[sid]
            r = {"subject_id": sid, "si": par[3], "mi": par[5], "tau_m": par[0], "Gb": par[1], "sg": par[2], "p2": par[4]}
            for k, v in gold.items():
                r[k] = v
            # dataset_id
            r["dataset_id"] = labels_combined.loc[labels_combined["subject_id"].astype(str) == str(sid), "dataset_id"].iloc[0] if "dataset_id" in labels_combined.columns else ""
            # tri_class（基于 gold 的 sspg/di 计算），供多任务/分类评估使用
            if "sspg" in r and "di" in r and pd.notna(r["sspg"]) and pd.notna(r["di"]):
                r_sspg = float(r["sspg"])
                r_di = float(r["di"])
                if r_sspg < 120.0:
                    r["tri_class"] = 0
                elif r_sspg >= 120.0 and r_di >= 1.2:
                    r["tri_class"] = 1
                else:
                    r["tri_class"] = 2
            else:
                r["tri_class"] = -1
            rows.append(r)
        return pd.DataFrame(rows)

    table_test = build_latent_gold_table(test_pids, param_by_subject)
    print(f"\n--- Test set: {len(table_test)} subjects with gold ---")

    results = []
    for col in GOLD_COLS:
        if col not in table_test.columns or table_test[col].notna().sum() < 3:
            continue
        sub = table_test.dropna(subset=[col])
        if len(sub) < 3:
            continue
        for pname in ["si", "mi", "tau_m", "sg"]:
            r, pval = stats.spearmanr(sub[pname], sub[col], nan_policy="omit")
            results.append({"gold": col, "param": pname, "spearman_r": r, "p_value": pval, "n": len(sub)})
            if not np.isnan(r):
                print(f"  {pname} vs {col}:  r={r:.3f}  p={pval:.4f}  n={len(sub)}")

    # ---------- 6. 全量 encode（D1+D2+D4 全部样本）用于出图 ----------
    all_cgm = (batch.cgm - train_mean[0]) / (train_std[0] + 1e-8)
    all_ts = (batch.timestamps - train_mean[1]) / (train_std[1] + 1e-8)
    all_meals = (batch.meals - train_mean[2]) / (train_std[2] + 1e-8)
    all_demo = (batch.demographics - train_mean[3]) / (train_std[3] + 1e-8)
    all_tens = [torch.as_tensor(x, dtype=torch.float, device=device) for x in (all_cgm, all_ts, all_meals, all_demo)]
    with torch.no_grad():
        model.eval()
        if not P1_V5_PREDICTION_HEAD and not P1_HEAD_USE_26D:
            ir_head.eval()
            sspg_head.eval()
            if di_head is not None:
                di_head.eval()
        if e2e_head is not None:
            e2e_head.eval()
        out_all, _, _, e2e_all = model(*all_tens)
        if P1_HEAD_USE_26D and e2e_head is not None and hasattr(model, "get_all_latents"):
            p26, init26, z16 = model.get_all_latents(*all_tens)
            head_in = torch.cat([p26, init26], dim=-1) if P1_V8_HEAD_10D else torch.cat([p26, init26, z16], dim=-1)
            e2e_all = e2e_head(head_in)
    param_all = out_all.param.cpu().numpy()
    if P1_SAVE_26D_LATENT and hasattr(model, "get_all_latents"):
        with torch.no_grad():
            p26, init26, z16 = model.get_all_latents(*all_tens)
        param_26d = p26.cpu().numpy()
        z_init_4d = init26.cpu().numpy()
        z_nonseq_16d = z16.cpu().numpy()
    # 对全部样本计算 IR / SSPG / DI 预测（用于后续分析与可视化）
    with torch.no_grad():
        if (P1_V5_PREDICTION_HEAD or P1_HEAD_USE_26D) and e2e_all is not None:
            sspg_all = e2e_all[:, 0].cpu().numpy()
            di_all = e2e_all[:, 1].cpu().numpy()
            ir_all = np.full_like(sspg_all, np.nan)
        else:
            lat_all = torch.as_tensor(param_all, dtype=torch.float, device=device)
            ir_all = ir_head(lat_all[:, IR_LATENT_IX]).squeeze(-1).cpu().numpy()
            if P1_SEPARATE_HEAD_26D and hasattr(model, "get_all_latents"):
                p26_all, init26_all, z16_all = model.get_all_latents(*all_tens)
                lat_26_all = torch.cat([p26_all, init26_all, z16_all], dim=-1)
                sspg_all = sspg_head(lat_26_all).squeeze(-1).cpu().numpy()
                di_all = di_head(lat_26_all).squeeze(-1).cpu().numpy()
            else:
                sspg_in_all = lat_all[:, 3:4] if P1_DECOUPLE_SSPG else lat_all
                sspg_all = sspg_head(sspg_in_all).squeeze(-1).cpu().numpy()
                if P1_DI_MLP_HEAD:
                    di_all = di_head(lat_all[:, [3, 5]]).squeeze(-1).cpu().numpy()
                elif P1_DI_LOG_PRODUCT:
                    log_si_all = torch.log(lat_all[:, 3] + 1e-12)
                    log_mi_all = torch.log(lat_all[:, 5] + 1e-12)
                    log_sum_all = (log_si_all + log_mi_all).unsqueeze(-1)
                    di_all = di_head(log_sum_all).squeeze(-1).cpu().numpy()
                elif P1_DI_PRODUCT_CONSTRAINT:
                    di_all = (lat_all[:, 3] * lat_all[:, 5]).cpu().numpy()
                else:
                    di_all = di_head(lat_all).squeeze(-1).cpu().numpy()
        if not (P1_V5_PREDICTION_HEAD or P1_HEAD_USE_26D) and P1_ZSCORE_TARGETS:
            sspg_all = sspg_all * sspg_std + sspg_mean
            di_all = di_all * di_std + di_mean
    param_by_subject_all = {}
    pred_by_subject_all = {}
    for sid in np.unique(pids):
        mask = pids == sid
        param_by_subject_all[sid] = param_all[mask].mean(axis=0)
        pred_by_subject_all[sid] = {
            "homa_ir_hat": float(np.nanmean(ir_all[mask])),
            "sspg_hat_head": float(np.nanmean(sspg_all[mask])),
            "di_hat_head": float(np.nanmean(di_all[mask])),
        }

    def build_latent_gold_table_with_preds(pids_subset, param_by_sid, pred_by_sid):
        df = build_latent_gold_table(pids_subset, param_by_sid)
        if not df.empty and pred_by_sid:
            df["homa_ir_hat"] = df["subject_id"].map(lambda x: pred_by_sid.get(x, {}).get("homa_ir_hat", np.nan))
            df["sspg_hat_head"] = df["subject_id"].map(lambda x: pred_by_sid.get(x, {}).get("sspg_hat_head", np.nan))
            df["di_hat_head"] = df["subject_id"].map(lambda x: pred_by_sid.get(x, {}).get("di_hat_head", np.nan))
        return df

    table_all = build_latent_gold_table_with_preds(pids, param_by_subject_all, pred_by_subject_all)
    if "dataset_id" not in table_all.columns and not labels_combined.empty and "dataset_id" in labels_combined.columns:
        did_map = labels_combined.set_index("subject_id")["dataset_id"].to_dict()
        table_all["dataset_id"] = table_all["subject_id"].map(lambda x: did_map.get(str(x), ""))

    # ---------- 6b. 端到端预测头在测试集上的评估（M1 主结果 / v5 终局 subject-level） ----------
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scripts.evaluate_p1_metrics import compute_metrics as _compute_metrics
    print("\n--- End-to-End Head Evaluation on Test Set ---")
    if P1_V5_PREDICTION_HEAD or P1_HEAD_USE_26D:
        # v5/V6: 测试集 subject-level 预测从 pred_by_subject_all 填入，与金标准算 Spearman
        df_test = table_test.copy()
        df_test["sspg_hat_head"] = df_test["subject_id"].map(lambda x: pred_by_subject_all.get(x, {}).get("sspg_hat_head", np.nan))
        df_test["di_hat_head"] = df_test["subject_id"].map(lambda x: pred_by_subject_all.get(x, {}).get("di_hat_head", np.nan))
        if not df_test.empty:
            sspg_hat = df_test["sspg_hat_head"].values
            di_hat = df_test["di_hat_head"].values
            sspg_true = df_test["sspg"].values if "sspg" in df_test.columns else np.full_like(sspg_hat, np.nan)
            di_true = df_test["di"].values if "di" in df_test.columns else np.full_like(di_hat, np.nan)
            ok_sspg = np.isfinite(sspg_hat) & np.isfinite(sspg_true)
            ok_di = np.isfinite(di_hat) & np.isfinite(di_true)
            r_sspg, _ = stats.spearmanr(sspg_hat[ok_sspg], sspg_true[ok_sspg]) if ok_sspg.sum() > 2 else (np.nan, np.nan)
            r_di, _ = stats.spearmanr(di_hat[ok_di], di_true[ok_di]) if ok_di.sum() > 2 else (np.nan, np.nan)
            lbl = "V6 26D" if P1_HEAD_USE_26D else "V5"
            print(f"  {lbl} Prediction Head SSPG (test subject-level): Spearman r = {r_sspg:.4f}  n = {ok_sspg.sum()}")
            print(f"  {lbl} Prediction Head DI (test subject-level):   Spearman r = {r_di:.4f}  n = {ok_di.sum()}")
        else:
            r_sspg, r_di, ok_sspg, ok_di = np.nan, np.nan, np.array([], dtype=bool), np.array([], dtype=bool)
        metrics_sspg_e2e = {"spearman_r": float(r_sspg) if not np.isnan(r_sspg) else np.nan, "rmse": np.nan, "n": int(np.asarray(ok_sspg).sum())}
        metrics_di_e2e = {"spearman_r": float(r_di) if not np.isnan(r_di) else np.nan, "rmse": np.nan, "n": int(np.asarray(ok_di).sum())}
        e2e_metrics = {"sspg": metrics_sspg_e2e, "di": metrics_di_e2e}
    else:
        with torch.no_grad():
            model.eval()
            ir_head.eval()
            sspg_head.eval()
            if di_head is not None:
                di_head.eval()
            lat_test_e2e = torch.as_tensor(param_test, dtype=torch.float, device=device)
            if P1_SEPARATE_HEAD_26D and hasattr(model, "get_all_latents"):
                p26_t, init26_t, z16_t = model.get_all_latents(*test_tensors[:4])
                lat_test_26d = torch.cat([p26_t, init26_t, z16_t], dim=-1)
                sspg_pred_e2e = sspg_head(lat_test_26d).squeeze(-1).cpu().numpy()
                di_pred_e2e = di_head(lat_test_26d).squeeze(-1).cpu().numpy()
            else:
                sspg_in_e2e = lat_test_e2e[:, 3:4] if P1_DECOUPLE_SSPG else lat_test_e2e
                sspg_pred_e2e = sspg_head(sspg_in_e2e).squeeze(-1).cpu().numpy()
                if P1_DI_MLP_HEAD:
                    di_pred_e2e = di_head(lat_test_e2e[:, [3, 5]]).squeeze(-1).cpu().numpy()
                elif P1_DI_LOG_PRODUCT:
                    log_si_test = torch.log(lat_test_e2e[:, 3] + 1e-12)
                    log_mi_test = torch.log(lat_test_e2e[:, 5] + 1e-12)
                    log_sum_test = (log_si_test + log_mi_test).unsqueeze(-1)
                    di_pred_e2e = di_head(log_sum_test).squeeze(-1).cpu().numpy()
                elif P1_DI_PRODUCT_CONSTRAINT:
                    di_pred_e2e = (lat_test_e2e[:, 3] * lat_test_e2e[:, 5]).cpu().numpy()
                else:
                    di_pred_e2e = di_head(lat_test_e2e).squeeze(-1).cpu().numpy()
            if P1_ZSCORE_TARGETS:
                sspg_pred_e2e = sspg_pred_e2e * sspg_std + sspg_mean
                di_pred_e2e = di_pred_e2e * di_std + di_mean
        sspg_true_test = np.asarray(sspg_per_sample[test_idx], dtype=np.float64)
        di_true_test = np.asarray(di_per_sample[test_idx], dtype=np.float64)
        metrics_sspg_e2e = _compute_metrics(sspg_true_test, sspg_pred_e2e)
        metrics_di_e2e = _compute_metrics(di_true_test, di_pred_e2e)
        print(f"  End-to-End SSPG (test): Spearman r = {metrics_sspg_e2e['spearman_r']:.4f}  RMSE = {metrics_sspg_e2e['rmse']:.4f}  n = {metrics_sspg_e2e['n']}")
        print(f"  End-to-End DI (test):   Spearman r = {metrics_di_e2e['spearman_r']:.4f}  RMSE = {metrics_di_e2e['rmse']:.4f}  n = {metrics_di_e2e['n']}")
        e2e_metrics = {"sspg": metrics_sspg_e2e, "di": metrics_di_e2e}

    # ---------- 7. 保存 ----------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    import json
    with open(os.path.join(RESULTS_DIR, "e2e_head_metrics.json"), "w") as f:
        json.dump(e2e_metrics, f, indent=2)
    if P1_V5_PREDICTION_HEAD:
        with open(os.path.join(RESULTS_DIR, "v5_spearman.json"), "w") as f:
            json.dump({
                "sspg_spearman_r": e2e_metrics["sspg"].get("spearman_r", np.nan),
                "di_spearman_r": e2e_metrics["di"].get("spearman_r", np.nan),
                "lambda_sspg": LAMBDA_SSPG,
                "lambda_di": LAMBDA_DI,
                "seed": SEED,
            }, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "training_curves.json"), "w") as f:
        json.dump({"epoch": list(range(1, NUM_EPOCHS + 1)), "train_loss": epoch_train_losses, "val_loss": epoch_val_losses}, f, indent=0)
    with open(os.path.join(RESULTS_DIR, "training_metrics.json"), "w") as f:
        json.dump(
            {
                "epoch": list(range(1, NUM_EPOCHS + 1)),
                "recon_loss": epoch_recon_losses,
                "sspg_loss": epoch_sspg_losses,
                "di_loss": epoch_di_losses,
                "si_cv": epoch_si_cvs,
                "mi_cv": epoch_mi_cvs,
            },
            f,
            indent=0,
        )
    ckpt = {
        "model_state": model.state_dict(),
        "ir_head_state": ir_head.state_dict(),
        "IR_LATENT_IX": IR_LATENT_IX,
        "LAMBDA_IR": LAMBDA_IR,
        "LAMBDA_SSPG": LAMBDA_SSPG,
        "LAMBDA_DI": LAMBDA_DI,
        "LAMBDA_IDENTIFIABILITY": LAMBDA_IDENTIFIABILITY,
        "LAMBDA_ORTHO": LAMBDA_ORTHO,
        "LAMBDA_DIV": LAMBDA_DIV,
        "G_mean": G_mean.cpu(),
        "G_std": G_std.cpu(),
        "train_mean": [np.array(t) for t in train_mean],
        "train_std": [np.array(t) for t in train_std],
        "P1_ZSCORE_TARGETS": P1_ZSCORE_TARGETS,
        "P1_DECOUPLE_SSPG": P1_DECOUPLE_SSPG,
        "P1_DI_PRODUCT_CONSTRAINT": P1_DI_PRODUCT_CONSTRAINT,
        "P1_DI_LOG_PRODUCT": P1_DI_LOG_PRODUCT,
        "P1_DI_MLP_HEAD": P1_DI_MLP_HEAD,
        "P1_SEPARATE_HEAD_26D": P1_SEPARATE_HEAD_26D,
        "P1_DETACH_HEAD_INPUT": P1_DETACH_HEAD_INPUT,
        "P1_HEAD_GRAD_SCALE": P1_HEAD_GRAD_SCALE,
        "P1_V8_HEAD_10D": P1_V8_HEAD_10D,
        "P1_V8_RECON_CORR": os.environ.get("P1_V8_RECON_CORR", "").strip().lower() in ("1", "true", "yes"),
        "P1_V8_ODE_CORR": os.environ.get("P1_V8_ODE_CORR", "").strip().lower() in ("1", "true", "yes"),
        "sspg_mean": sspg_mean,
        "sspg_std": sspg_std,
        "di_mean": di_mean,
        "di_std": di_std,
    }
    if sspg_head is not None:
        ckpt["sspg_head_state"] = sspg_head.state_dict()
    if di_head is not None:
        ckpt["di_head_state"] = di_head.state_dict()
    if e2e_head is not None:
        ckpt["e2e_head_state"] = e2e_head.state_dict()
    if cls_head is not None:
        ckpt["cls_head_state"] = cls_head.state_dict()
    torch.save(ckpt, os.path.join(RESULTS_DIR, "autoencoder_p1_full.pt"))
    with open(os.path.join(RESULTS_DIR, "correlations.txt"), "w") as f:
        f.write("gold\tparam\tspearman_r\tp_value\tn\n")
        for r in results:
            f.write(f"{r['gold']}\t{r['param']}\t{r['spearman_r']}\t{r['p_value']}\t{r['n']}\n")
    table_test.to_csv(os.path.join(RESULTS_DIR, "latent_and_gold_test.csv"), index=False)
    table_all.to_csv(os.path.join(RESULTS_DIR, "latent_and_gold_all.csv"), index=False)
    if P1_SAVE_26D_LATENT and hasattr(model, "get_all_latents"):
        rows_26d = []
        for i in range(len(pids)):
            r = {"sample_ix": i, "subject_id": pids[i]}
            for j, name in enumerate(["tau_m", "Gb", "sg", "si", "p2", "mi"]):
                r[name] = param_26d[i, j]
            for j in range(4):
                r[f"z_init_{j}"] = z_init_4d[i, j]
            for j in range(16):
                r[f"z_nonseq_{j}"] = z_nonseq_16d[i, j]
            # 保存 V9 三分类标签（per-sample）
            if i < len(tri_class_per_sample):
                r["tri_class"] = int(tri_class_per_sample[i])
            if (P1_V5_PREDICTION_HEAD or P1_HEAD_USE_26D) and e2e_all is not None:
                sspg_h = sspg_all[i] * sspg_std + sspg_mean if P1_ZSCORE_TARGETS else sspg_all[i]
                di_h = di_all[i] * di_std + di_mean if P1_ZSCORE_TARGETS else di_all[i]
                r["sspg_hat"] = float(sspg_h)
                r["di_hat"] = float(di_h)
            gold = get_gold_for_subject(pids[i])
            if gold:
                for k, v in gold.items():
                    r[k] = v
            rows_26d.append(r)
        pd.DataFrame(rows_26d).to_csv(os.path.join(RESULTS_DIR, "latent_and_gold_all_26d.csv"), index=False)
        print(f"Saved {RESULTS_DIR}/latent_and_gold_all_26d.csv (26D latent per sample)")
    # M1 模式崩溃检查：若启用监督，打印 latent 卡上限比例（均 < 20% 为正常）
    if LAMBDA_SSPG > 0 or LAMBDA_DI > 0:
        lims = model.param_lims.cpu().numpy()
        tau_hi, gb_hi, mi_hi = float(lims[0, 1]), float(lims[1, 1]), float(lims[5, 1])
        p_tau = (table_all["tau_m"] >= tau_hi - 0.01).mean() * 100
        p_gb = (table_all["Gb"] >= gb_hi - 0.1).mean() * 100
        p_mi = (table_all["mi"] >= mi_hi - 0.01).mean() * 100
        print("\n--- M1 latent 边界检查（卡上限比例，均 < 20% 为正常） ---")
        print(f"  tau_m 卡上限({tau_hi:.0f}): {p_tau:.1f}%")
        print(f"  Gb 卡上限({gb_hi:.0f}):   {p_gb:.1f}%")
        print(f"  mi 卡上限({mi_hi:.2f}):   {p_mi:.1f}%")
    print(f"\nSaved {RESULTS_DIR}/autoencoder_p1_full.pt, correlations.txt, latent_and_gold_*.csv")
    print("Run: python scripts/plot_p1_results.py  to generate figures.")
    print("M1 与 M0 同口径 5 折评估: python scripts/evaluate_p1_metrics.py --csv <m1>/latent_and_gold_all.csv --out <m1>")
    return results, table_all


if __name__ == "__main__":
    main()
