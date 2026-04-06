"""
Paper 1 端到端：D1 加载 → 划分 → 归一化 → 训练 → Test 上 latent–金标准相关 → D2 encode。
需先解压 cgm_project.tar.gz 并设置 OUTPUT_BASE。
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

# 数据路径：若环境变量未设置则用默认（请按本机修改）
OUTPUT_BASE = os.environ.get("CGM_PROJECT_OUTPUT", "/Users/hertz1030/Downloads/cgm_project/output")
NUM_EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-2
BETA_HAT = 0.01
SEED = 21

def main():
    from load_cgm_project_data import load_cgm_project_level1_level2, split_by_subject
    from paper1_experiment_config import get_data_dir, D1_TRAIN_FRAC, D1_VAL_FRAC, D1_TEST_FRAC, SPLIT_SEED
    from data_utils import normalize_train_test, MEAL_COVARIATES, DEMOGRAPHICS_COVARIATES
    from models import MechanisticAutoencoder, count_params
    from utils import seed_everything

    seed_everything(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output base: {OUTPUT_BASE}")

    # ---------- 1. 加载 D1，按 subject 划分 ----------
    print("\n--- Loading D1 ---")
    batch, info, labels_df = load_cgm_project_level1_level2(
        dataset_id="D1", output_base=OUTPUT_BASE, num_meals_threshold=1
    )
    train_idx, val_idx, test_idx = split_by_subject(
        info, train_frac=D1_TRAIN_FRAC, val_frac=D1_VAL_FRAC, test_frac=D1_TEST_FRAC,
        seed=SPLIT_SEED, stratify_diagnosis=batch.diagnosis
    )
    train_arrays = type(batch)(*(batch.cgm[train_idx], batch.timestamps[train_idx], batch.meals[train_idx], batch.demographics[train_idx], batch.diagnosis[train_idx]))
    val_arrays   = type(batch)(*(batch.cgm[val_idx],   batch.timestamps[val_idx],   batch.meals[val_idx],   batch.demographics[val_idx],   batch.diagnosis[val_idx]))
    test_arrays  = type(batch)(*(batch.cgm[test_idx],  batch.timestamps[test_idx],  batch.meals[test_idx],  batch.demographics[test_idx],  batch.diagnosis[test_idx]))

    # ---------- 2. 归一化（仅用 train 的 mean/std） ----------
    (train_cgm, train_ts, train_meals, train_demo), (val_cgm, val_ts, val_meals, val_demo), (train_means, train_stds) = normalize_train_test(
        (train_arrays.cgm, train_arrays.timestamps, train_arrays.meals, train_arrays.demographics),
        (val_arrays.cgm, val_arrays.timestamps, val_arrays.meals, val_arrays.demographics),
    )
    _, (test_cgm, test_ts, test_meals, test_demo), _ = normalize_train_test(
        (train_arrays.cgm, train_arrays.timestamps, train_arrays.meals, train_arrays.demographics),
        (test_arrays.cgm, test_arrays.timestamps, test_arrays.meals, test_arrays.demographics),
    )
    train_arrays = type(batch)(*(train_cgm, train_ts, train_meals, train_demo, train_arrays.diagnosis))
    val_arrays   = type(batch)(*(val_cgm, val_ts, val_meals, val_demo, val_arrays.diagnosis))
    test_arrays  = type(batch)(*(test_cgm, test_ts, test_meals, test_demo, test_arrays.diagnosis))
    train_mean, train_std = train_means, train_stds

    G_mean = torch.as_tensor(train_mean[0], dtype=torch.float, device=device)
    G_std  = torch.as_tensor(train_std[0], dtype=torch.float, device=device)
    def remove_scale(G, mean=G_mean, std=G_std):
        return (G - mean) / std

    train_tensors = [torch.as_tensor(x, dtype=torch.float, device=device) for x in (train_arrays.cgm, train_arrays.timestamps, train_arrays.meals, train_arrays.demographics, train_arrays.diagnosis)]
    val_tensors   = [torch.as_tensor(x, dtype=torch.float, device=device) for x in (val_arrays.cgm, val_arrays.timestamps, val_arrays.meals, val_arrays.demographics, val_arrays.diagnosis)]
    test_tensors  = [torch.as_tensor(x, dtype=torch.float, device=device) for x in (test_arrays.cgm, test_arrays.timestamps, test_arrays.meals, test_arrays.demographics, test_arrays.diagnosis)]

    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=min(BATCH_SIZE, len(train_tensors[0])), shuffle=True)

    # ---------- 3. 模型与损失 ----------
    model = MechanisticAutoencoder(
        meal_size=len(MEAL_COVARIATES),
        demographics_size=len(DEMOGRAPHICS_COVARIATES),
        embedding_size=8,
        hidden_size=32,
        num_layers=2,
        encoder_dropout_prob=0.,
        decoder_dropout_prob=0.5,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    print(f"Params: {count_params(model)}")

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

    # ---------- 4. 训练 ----------
    print("\n--- Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for tup in train_loader:
            cgm, timestamps, meals, demographics, _ = tup
            optimizer.zero_grad()
            output, seq_q, nonseq_q = model(cgm, timestamps, meals, demographics)
            loss = loss_fn(output, seq_q, nonseq_q, model.seq_p, model.nonseq_p, cgm)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(cgm)
        train_loss /= len(train_tensors[0])
        with torch.no_grad():
            model.eval()
            out_v, _, _ = model(*val_tensors[:4])
            pred_v = remove_scale(out_v.states[..., 0:1])
            val_loss = (pred_v - val_tensors[0]).pow(2).mean().item()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}  Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}")

    # ---------- 5. Test 上取 latent，与金标准相关 ----------
    print("\n--- Test set: latent vs gold standard ---")
    with torch.no_grad():
        model.eval()
        output_test, _, _ = model(*test_tensors[:4])
    param_test = output_test.param.cpu().numpy()  # (n_test_samples, 6)  tau_m, Gb, sg, si, p2, mi
    test_pids = np.array(info.patient_ids)[test_idx]
    # 每 subject 平均 param
    unique_test_subjects = np.unique(test_pids)
    param_by_subject = {}
    for sid in unique_test_subjects:
        mask = test_pids == sid
        param_by_subject[sid] = param_test[mask].mean(axis=0)
    # 与 labels 对齐（labels 是 subject_id, sspg, di, homa_ir, homa_b）
    PARAM_NAMES = ["tau_m", "Gb", "sg", "si", "p2", "mi"]
    gold_cols = ["sspg", "di", "homa_ir", "homa_b"]
    results = []
    for col in gold_cols:
        if col not in labels_df.columns or labels_df[col].notna().sum() < 5:
            continue
        subj_with_gold = labels_df.dropna(subset=[col])["subject_id"].values
        param_list = []
        gold_list = []
        for sid in subj_with_gold:
            if sid not in param_by_subject:
                continue
            param_list.append(param_by_subject[sid])
            gold_list.append(labels_df.loc[labels_df["subject_id"] == sid, col].iloc[0])
        param_list = np.array(param_list)
        gold_list = np.array(gold_list)
        for i, pname in enumerate(PARAM_NAMES):
            r, pval = stats.spearmanr(param_list[:, i], gold_list, nan_policy="omit")
            results.append({"gold": col, "param": pname, "spearman_r": r, "p_value": pval, "n": len(gold_list)})
            if not np.isnan(r):
                print(f"  {pname} vs {col}:  r={r:.3f}  p={pval:.4f}  n={len(gold_list)}")

    # ---------- 6. 保存模型与结果 ----------
    os.makedirs("paper1_results", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "G_mean": G_mean.cpu(), "G_std": G_std.cpu()}, "paper1_results/autoencoder_d1.pt")
    with open("paper1_results/correlations.txt", "w") as f:
        f.write("gold\tparam\tspearman_r\tp_value\tn\n")
        for r in results:
            f.write(f"{r['gold']}\t{r['param']}\t{r['spearman_r']}\t{r['p_value']}\t{r['n']}\n")
    print("\nSaved paper1_results/autoencoder_d1.pt and paper1_results/correlations.txt")

    # ---------- 7. D2 encode（Level 2） ----------
    print("\n--- D2 encode (Level 2) ---")
    batch_d2, info_d2, _ = load_cgm_project_level1_level2(dataset_id="D2", output_base=OUTPUT_BASE, num_meals_threshold=1)
    # 用 D1 的归一化
    g_mean_np, g_std_np = train_mean[0], train_std[0]
    cgm_d2 = (batch_d2.cgm - g_mean_np) / (g_std_np + 1e-8)
    ts_d2 = (batch_d2.timestamps - train_mean[1]) / (train_std[1] + 1e-8)
    meals_d2 = (batch_d2.meals - train_mean[2]) / (train_std[2] + 1e-8)
    demo_d2 = (batch_d2.demographics - train_mean[3]) / (train_std[3] + 1e-8)
    tens_d2 = [torch.as_tensor(x, dtype=torch.float, device=device) for x in (cgm_d2, ts_d2, meals_d2, demo_d2)]
    with torch.no_grad():
        model.eval()
        out_d2, _, _ = model(*tens_d2)
    param_d2 = out_d2.param.cpu().numpy()
    np.save("paper1_results/latent_d2.npy", param_d2)
    print(f"D2 latent shape: {param_d2.shape}, mean si={param_d2[:, 3].mean():.4f}, mi={param_d2[:, 5].mean():.4f}")

    print("\n--- Done ---")
    return results

if __name__ == "__main__":
    main()
