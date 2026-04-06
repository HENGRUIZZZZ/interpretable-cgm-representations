"""
P1 M2：黑盒消融。端到端监督的黑盒 VAE（Decoder 为 LSTM，非 ODE）。

结构同 M1，但模型换为 BlackboxAutoencoder；latent 经 Linear(encoding_size, 6) 投影后接同一套 SSPG/DI/IR 头。
主结果来自测试集端到端 head 评估，保存至 e2e_head_metrics.json。

用法（项目根目录）：
  export CGM_PROJECT_OUTPUT=/path/to/cgm_project/output
  export LAMBDA_SSPG=1.0 LAMBDA_DI=1.0
  python run_p1_m2_blackbox.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_BASE = os.environ.get("CGM_PROJECT_OUTPUT", "/Users/hertz1030/Downloads/cgm_project/output")
NUM_EPOCHS = int(os.environ.get("P1_NUM_EPOCHS", "100"))
BATCH_SIZE = 32
LR = float(os.environ.get("P1_LR", "1e-2"))
BETA_HAT = float(os.environ.get("BETA_HAT", "0.01"))
SEED = int(os.environ.get("P1_SEED", "21"))
LAMBDA_IR = float(os.environ.get("LAMBDA_IR", "0.05"))
LAMBDA_SSPG = float(os.environ.get("LAMBDA_SSPG", "1.0"))
LAMBDA_DI = float(os.environ.get("LAMBDA_DI", "1.0"))
RESULTS_DIR = os.environ.get("P1_RESULTS_DIR", "paper1_results_m2")
PARAM_NAMES = ["tau_m", "Gb", "sg", "si", "p2", "mi"]
GOLD_COLS = ["sspg", "di", "homa_ir", "homa_b"]
IR_LATENT_IX = [3, 5, 0]


def main():
    from load_cgm_project_data import (
        load_cgm_project_level1_level2,
        load_cgm_project_level3,
        split_by_subject,
    )
    from paper1_experiment_config import get_data_dir, P1_FULL_TRAIN_DATASETS, P1_TRAIN_FRAC, P1_VAL_FRAC, P1_TEST_FRAC
    from data_utils import normalize_train_test, MEAL_COVARIATES, DEMOGRAPHICS_COVARIATES
    from models import BlackboxAutoencoder, count_params
    from utils import seed_everything
    from run_p1_full_pipeline import _stack_batches

    _train_ds_env = os.environ.get("P1_TRAIN_DATASETS", "").strip()
    P1_TRAIN_DATASETS = [x.strip() for x in _train_ds_env.split(",") if x.strip()] if _train_ds_env else list(P1_FULL_TRAIN_DATASETS)

    seed_everything(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("M2 Blackbox VAE (LSTM decoder). Device:", device)

    batch_list, info_list, labels_list, dataset_ids = [], [], [], []
    for did in P1_TRAIN_DATASETS:
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
    if len(batch_list) < 2:
        raise RuntimeError("Need at least D1 and D2.")
    batch, pids, labels_combined = _stack_batches(batch_list, info_list, labels_list, dataset_ids)

    def _per_sample_from_labels(col):
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
    from data_utils import PatientInfo
    info = PatientInfo(patient_ids=pids, train_ids=pids, test_ids=pids)
    train_idx, val_idx, test_idx = split_by_subject(
        info, train_frac=P1_TRAIN_FRAC, val_frac=P1_VAL_FRAC, test_frac=P1_TEST_FRAC,
        seed=SEED, stratify_diagnosis=batch.diagnosis,
    )
    train_arrays = type(batch)(*(batch.cgm[train_idx], batch.timestamps[train_idx], batch.meals[train_idx], batch.demographics[train_idx], batch.diagnosis[train_idx]))
    val_arrays = type(batch)(*(batch.cgm[val_idx], batch.timestamps[val_idx], batch.meals[val_idx], batch.demographics[val_idx], batch.diagnosis[val_idx]))
    test_arrays = type(batch)(*(batch.cgm[test_idx], batch.timestamps[test_idx], batch.meals[test_idx], batch.demographics[test_idx], batch.diagnosis[test_idx]))
    (train_cgm, train_ts, train_meals, train_demo), (val_cgm, val_ts, val_meals, val_demo), (train_means, train_stds) = normalize_train_test(
        (train_arrays.cgm, train_arrays.timestamps, train_arrays.meals, train_arrays.demographics),
        (val_arrays.cgm, val_arrays.timestamps, val_arrays.meals, val_arrays.demographics),
    )
    _, (test_cgm, test_ts, test_meals, test_demo), _ = normalize_train_test(
        (train_arrays.cgm, train_arrays.timestamps, train_arrays.meals, train_arrays.demographics),
        (test_arrays.cgm, test_arrays.timestamps, test_arrays.meals, test_arrays.demographics),
    )
    train_tensors = [torch.as_tensor(x, dtype=torch.float, device=device) for x in (train_cgm, train_ts, train_meals, train_demo)]
    train_tensors.append(torch.as_tensor(homa_ir_per_sample[train_idx], dtype=torch.float, device=device))
    train_tensors.append(torch.as_tensor(sspg_per_sample[train_idx], dtype=torch.float, device=device))
    train_tensors.append(torch.as_tensor(di_per_sample[train_idx], dtype=torch.float, device=device))
    val_tensors = [torch.as_tensor(x, dtype=torch.float, device=device) for x in (val_cgm, val_ts, val_meals, val_demo)]
    test_tensors = [torch.as_tensor(x, dtype=torch.float, device=device) for x in (test_cgm, test_ts, test_meals, test_demo)]
    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=min(BATCH_SIZE, len(train_tensors[0])), shuffle=True)

    hidden_size = 32
    model = BlackboxAutoencoder(
        meal_size=len(MEAL_COVARIATES),
        demographics_size=len(DEMOGRAPHICS_COVARIATES),
        embedding_size=8,
        hidden_size=hidden_size,
        num_layers=2,
        encoder_dropout_prob=0.0,
        decoder_dropout_prob=0.5,
    ).to(device)
    encoding_size = model.encoding_size  # = hidden_size
    param_proj = torch.nn.Linear(encoding_size, 6).to(device)
    ir_head = torch.nn.Linear(len(IR_LATENT_IX), 1).to(device)
    sspg_head = torch.nn.Linear(6, 1).to(device)
    di_head = torch.nn.Linear(6, 1).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(param_proj.parameters()) + list(ir_head.parameters()) + list(sspg_head.parameters()) + list(di_head.parameters()),
        lr=LR,
    )
    print("Params:", count_params(model), "(+ param_proj + heads)")

    for epoch in range(NUM_EPOCHS):
        model.train()
        ir_head.train()
        sspg_head.train()
        di_head.train()
        param_proj.train()
        train_loss = 0.0
        for tup in train_loader:
            cgm, timestamps, meals, demographics = tup[0], tup[1], tup[2], tup[3]
            homa_ir_b, sspg_b, di_b = tup[4], tup[5], tup[6]
            optimizer.zero_grad()
            nonseq_mean, nonseq_std = model.encode_dist(cgm, timestamps, meals, demographics)
            if model.training:
                sample = nonseq_mean + nonseq_std * torch.randn_like(nonseq_mean)
            else:
                sample = nonseq_mean
            decoding = model.decode(sample, timestamps, meals, demographics)
            param_6d = param_proj(sample)
            loss_recon = (decoding - cgm).pow(2).mean()
            std_safe = nonseq_std.clamp(min=1e-6)
            nonseq_q = torch.distributions.Normal(nonseq_mean, std_safe)
            loss_kl = torch.distributions.kl.kl_divergence(nonseq_q, model.nonseq_p).sum(dim=-1).mean()
            loss = loss_recon + BETA_HAT * loss_kl
            valid = torch.isfinite(homa_ir_b) & (homa_ir_b >= 0)
            if valid.any():
                ir_in = param_6d[:, IR_LATENT_IX]
                loss = loss + LAMBDA_IR * (ir_head(ir_in).squeeze(-1)[valid] - torch.log1p(homa_ir_b.clamp(min=0))[valid]).pow(2).mean()
            if LAMBDA_SSPG > 0:
                valid_sspg = torch.isfinite(sspg_b)
                if valid_sspg.any():
                    loss = loss + LAMBDA_SSPG * (sspg_head(param_6d).squeeze(-1)[valid_sspg] - sspg_b[valid_sspg]).pow(2).mean()
            if LAMBDA_DI > 0:
                valid_di = torch.isfinite(di_b)
                if valid_di.any():
                    loss = loss + LAMBDA_DI * (di_head(param_6d).squeeze(-1)[valid_di] - di_b[valid_di]).pow(2).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(cgm)
        train_loss /= len(train_tensors[0])
        with torch.no_grad():
            model.eval()
            nonseq_mean_v, nonseq_std_v = model.encode_dist(*val_tensors[:4])
            sample_v = nonseq_mean_v
            decoding_v = model.decode(sample_v, *val_tensors[1:4])
            val_loss = (decoding_v - val_tensors[0]).pow(2).mean().item()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}  Train loss: {train_loss:.4f}  Val recon: {val_loss:.4f}")

    with torch.no_grad():
        model.eval()
        param_proj.eval()
        sspg_head.eval()
        di_head.eval()
        nonseq_mean_te, _ = model.encode_dist(*test_tensors[:4])
        param_6d_te = param_proj(nonseq_mean_te)
        param_test = param_6d_te.cpu().numpy()
        sspg_pred_e2e = sspg_head(param_6d_te).squeeze(-1).cpu().numpy()
        di_pred_e2e = di_head(param_6d_te).squeeze(-1).cpu().numpy()
    sspg_true_test = np.asarray(sspg_per_sample[test_idx], dtype=np.float64)
    di_true_test = np.asarray(di_per_sample[test_idx], dtype=np.float64)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scripts.evaluate_p1_metrics import compute_metrics as _compute_metrics
    metrics_sspg = _compute_metrics(sspg_true_test, sspg_pred_e2e)
    metrics_di = _compute_metrics(di_true_test, di_pred_e2e)
    print("\n--- End-to-End Head Evaluation on Test Set (M2) ---")
    print(f"  SSPG: Spearman r = {metrics_sspg['spearman_r']:.4f}  RMSE = {metrics_sspg['rmse']:.4f}  n = {metrics_sspg['n']}")
    print(f"  DI:   Spearman r = {metrics_di['spearman_r']:.4f}  RMSE = {metrics_di['rmse']:.4f}  n = {metrics_di['n']}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    e2e_metrics = {"sspg": metrics_sspg, "di": metrics_di}
    with open(os.path.join(RESULTS_DIR, "e2e_head_metrics.json"), "w") as f:
        json.dump(e2e_metrics, f, indent=2)
    torch.save({
        "model_state": model.state_dict(),
        "param_proj_state": param_proj.state_dict(),
        "sspg_head_state": sspg_head.state_dict(),
        "di_head_state": di_head.state_dict(),
        "ir_head_state": ir_head.state_dict(),
    }, os.path.join(RESULTS_DIR, "m2_blackbox.pt"))
    print("Saved", RESULTS_DIR)


if __name__ == "__main__":
    main()
