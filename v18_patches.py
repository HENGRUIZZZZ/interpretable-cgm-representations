#!/usr/bin/env python3
"""
GlucoVector v18 Code Patches
=============================
精确修改 run_p1_full_pipeline.py，实现以下功能：
  Patch 1: 读取新环境变量（EARLY_STOPPING, CORR_LOSS_ALPHA, SEPARATE_SMALL_HEAD）
  Patch 2: 分离小头架构（26D→16→1 for SSPG, 26D→16→1 for DI）
  Patch 3: Early Stopping（patience=15, 基于 val SSPG+DI loss 选择 best model）
  Patch 4: CorrLoss（Pearson 相关性损失，与 MSE 加权混合）
  Patch 5: Phase 1 正交约束（16D 与 10D 的 Frobenius 正交惩罚）

用法：
  python v18_patches.py --apply     # 应用补丁
  python v18_patches.py --verify    # 验证补丁
  python v18_patches.py --revert    # 恢复原始文件

目标文件：run_p1_full_pipeline.py（相对于 REPO_ROOT）
"""
import os
import sys
import shutil
import argparse
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT",
    os.path.expanduser("~/interpretable-cgm-representations")))
PIPELINE = REPO_ROOT / "run_p1_full_pipeline.py"
BACKUP = PIPELINE.with_suffix(".py.v17_backup")


# ============================================================================
# Patch definitions: each is (marker_text, replacement_text, description)
# marker_text: 在原文件中精确匹配的文本片段
# replacement_text: 替换后的文本
# ============================================================================

PATCHES = []

# ---------------------------------------------------------------------------
# Patch 1: 在环境变量读取区域（约第35行 LAMBDA_ORTHO 之后）插入新变量
# ---------------------------------------------------------------------------
PATCHES.append({
    "id": "P1_NEW_ENV_VARS",
    "description": "在 LAMBDA_DIV 定义之后插入 v18 新环境变量",
    "find": 'LAMBDA_DIV = float(os.environ.get("LAMBDA_DIV", "0.0"))',
    "replace": '''LAMBDA_DIV = float(os.environ.get("LAMBDA_DIV", "0.0"))
# === v18 patches ===
# Early Stopping: patience (0=disabled)
V18_EARLY_STOPPING_PATIENCE = int(os.environ.get("V18_EARLY_STOPPING_PATIENCE", "0"))
# CorrLoss: alpha weight for Pearson correlation loss (0=pure MSE, 1=pure CorrLoss)
V18_CORR_LOSS_ALPHA = float(os.environ.get("V18_CORR_LOSS_ALPHA", "0.0"))
# Separate small heads: 26D→16→1 for SSPG and DI independently
V18_SEPARATE_SMALL_HEAD = os.environ.get("V18_SEPARATE_SMALL_HEAD", "").strip().lower() in ("1", "true", "yes")
# Phase 1 orthogonality: penalize correlation between 16D and 10D latents
V18_LAMBDA_ORTHO_P1 = float(os.environ.get("V18_LAMBDA_ORTHO_P1", "0.0"))
# === end v18 patches ===''',
})

# ---------------------------------------------------------------------------
# Patch 2: 分离小头架构 — 在 e2e_head 构建区域（约第356行）
# 当 V18_SEPARATE_SMALL_HEAD=True 且 P1_HEAD_USE_26D=True 时，
# 用两个小头替代一个大的 e2e_head
# ---------------------------------------------------------------------------
PATCHES.append({
    "id": "P2_SEPARATE_SMALL_HEAD",
    "description": "在 e2e_head 构建逻辑中加入 v18 分离小头选项",
    "find": '''    elif P1_HEAD_USE_26D:
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
            print(f" [V6/V8] e2e_head 使用 26D 全 latent (dim={head_input_dim})")''',
    "replace": '''    elif P1_HEAD_USE_26D:
        sspg_head = None
        di_head = None
        if P1_V8_HEAD_10D:
            head_input_dim = n_ode_params + z_init_dim
        else:
            head_input_dim = n_ode_params + z_init_dim + z_nonseq_dim
        if V18_SEPARATE_SMALL_HEAD and not P1_V8_HEAD_10D:
            # v18: 分离小头 — 每个头 26D→16→ReLU→Dropout→1 (参数: 26*16+16+16*1+1=449)
            e2e_head = None
            sspg_head = torch.nn.Sequential(
                torch.nn.Linear(head_input_dim, 16),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(16, 1),
            ).to(device)
            di_head = torch.nn.Sequential(
                torch.nn.Linear(head_input_dim, 16),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(16, 1),
            ).to(device)
            print(f" [V18] Separate small heads: SSPG/DI each 26D→16→1 (dim={head_input_dim})")
        else:
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
                print(f" [V6/V8] e2e_head 使用 26D 全 latent (dim={head_input_dim})")''',
})

# ---------------------------------------------------------------------------
# Patch 3: optimizer 参数列表 — 加入分离小头的参数
# ---------------------------------------------------------------------------
PATCHES.append({
    "id": "P3_OPTIMIZER_PARAMS",
    "description": "optimizer 参数列表加入 v18 分离小头",
    "find": '''    if P1_V5_PREDICTION_HEAD:
        opt_params = list(model.parameters())
    elif P1_HEAD_USE_26D and e2e_head is not None:
        if P1_FINETUNE_HEAD_ONLY:
            opt_params = list(e2e_head.parameters())
        else:
            opt_params = list(model.parameters()) + list(e2e_head.parameters())''',
    "replace": '''    if P1_V5_PREDICTION_HEAD:
        opt_params = list(model.parameters())
    elif P1_HEAD_USE_26D and V18_SEPARATE_SMALL_HEAD and sspg_head is not None:
        # v18: 分离小头模式
        if P1_FINETUNE_HEAD_ONLY:
            opt_params = list(sspg_head.parameters()) + list(di_head.parameters())
        else:
            opt_params = list(model.parameters()) + list(sspg_head.parameters()) + list(di_head.parameters())
    elif P1_HEAD_USE_26D and e2e_head is not None:
        if P1_FINETUNE_HEAD_ONLY:
            opt_params = list(e2e_head.parameters())
        else:
            opt_params = list(model.parameters()) + list(e2e_head.parameters())''',
})

# ---------------------------------------------------------------------------
# Patch 4: 训练循环中 — 分离小头的 forward + CorrLoss
# 在 e2e_head 的 SSPG loss 计算处（约第544行）
# ---------------------------------------------------------------------------
PATCHES.append({
    "id": "P4_TRAINING_LOOP_CORRLOSS",
    "description": "训练循环中加入分离小头 forward 和 CorrLoss",
    "find": '''            elif P1_HEAD_USE_26D and e2e_pred is not None:
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
                        loss = loss + LAMBDA_DI * batch_di_loss''',
    "replace": '''            elif P1_HEAD_USE_26D and V18_SEPARATE_SMALL_HEAD and sspg_head is not None:
                # v18: 分离小头 forward
                p26_h, init26_h, z16_h = model.get_all_latents_for_head(cgm, timestamps, meals, demographics)
                head_in_v18 = _head_input_with_grad_control(torch.cat([p26_h, init26_h, z16_h], dim=-1))
                sspg_pred_v18 = sspg_head(head_in_v18).squeeze(-1)
                di_pred_v18 = di_head(head_in_v18).squeeze(-1)
                if LAMBDA_SSPG > 0:
                    valid_sspg = torch.isfinite(sspg_b)
                    if valid_sspg.any():
                        target_sspg = (sspg_b[valid_sspg] - sspg_mean_t) / sspg_std_t if P1_ZSCORE_TARGETS else sspg_b[valid_sspg]
                        pred_s = sspg_pred_v18[valid_sspg]
                        mse_sspg = (pred_s - target_sspg).pow(2).mean()
                        # v18 CorrLoss: alpha * (1 - pearson_r) + (1 - alpha) * MSE
                        if V18_CORR_LOSS_ALPHA > 0 and valid_sspg.sum() > 2:
                            vx = pred_s - pred_s.mean()
                            vy = target_sspg - target_sspg.mean()
                            pearson_r = (vx * vy).sum() / (vx.pow(2).sum().sqrt() * vy.pow(2).sum().sqrt() + 1e-8)
                            corr_loss = 1.0 - pearson_r
                            batch_sspg_loss = (1 - V18_CORR_LOSS_ALPHA) * mse_sspg + V18_CORR_LOSS_ALPHA * corr_loss
                        else:
                            batch_sspg_loss = mse_sspg
                        loss = loss + LAMBDA_SSPG * batch_sspg_loss
                if LAMBDA_DI > 0:
                    valid_di = torch.isfinite(di_b)
                    if valid_di.any():
                        target_di = (di_b[valid_di] - di_mean_t) / di_std_t if P1_ZSCORE_TARGETS else di_b[valid_di]
                        pred_d = di_pred_v18[valid_di]
                        mse_di = (pred_d - target_di).pow(2).mean()
                        if V18_CORR_LOSS_ALPHA > 0 and valid_di.sum() > 2:
                            vx_d = pred_d - pred_d.mean()
                            vy_d = target_di - target_di.mean()
                            pearson_r_d = (vx_d * vy_d).sum() / (vx_d.pow(2).sum().sqrt() * vy_d.pow(2).sum().sqrt() + 1e-8)
                            corr_loss_d = 1.0 - pearson_r_d
                            batch_di_loss = (1 - V18_CORR_LOSS_ALPHA) * mse_di + V18_CORR_LOSS_ALPHA * corr_loss_d
                        else:
                            batch_di_loss = mse_di
                        loss = loss + LAMBDA_DI * batch_di_loss
            elif P1_HEAD_USE_26D and e2e_pred is not None:
                if LAMBDA_SSPG > 0:
                    valid_sspg = torch.isfinite(sspg_b)
                    if valid_sspg.any():
                        target_sspg = (sspg_b[valid_sspg] - sspg_mean_t) / sspg_std_t if P1_ZSCORE_TARGETS else sspg_b[valid_sspg]
                        pred_s = e2e_pred[valid_sspg, 0]
                        mse_sspg = (pred_s - target_sspg).pow(2).mean()
                        if V18_CORR_LOSS_ALPHA > 0 and valid_sspg.sum() > 2:
                            vx = pred_s - pred_s.mean()
                            vy = target_sspg - target_sspg.mean()
                            pearson_r = (vx * vy).sum() / (vx.pow(2).sum().sqrt() * vy.pow(2).sum().sqrt() + 1e-8)
                            batch_sspg_loss = (1 - V18_CORR_LOSS_ALPHA) * mse_sspg + V18_CORR_LOSS_ALPHA * (1.0 - pearson_r)
                        else:
                            batch_sspg_loss = mse_sspg
                        loss = loss + LAMBDA_SSPG * batch_sspg_loss
                if LAMBDA_DI > 0:
                    valid_di = torch.isfinite(di_b)
                    if valid_di.any():
                        target_di = (di_b[valid_di] - di_mean_t) / di_std_t if P1_ZSCORE_TARGETS else di_b[valid_di]
                        pred_d = e2e_pred[valid_di, 1]
                        mse_di = (pred_d - target_di).pow(2).mean()
                        if V18_CORR_LOSS_ALPHA > 0 and valid_di.sum() > 2:
                            vx_d = pred_d - pred_d.mean()
                            vy_d = target_di - target_di.mean()
                            pearson_r_d = (vx_d * vy_d).sum() / (vx_d.pow(2).sum().sqrt() * vy_d.pow(2).sum().sqrt() + 1e-8)
                            batch_di_loss = (1 - V18_CORR_LOSS_ALPHA) * mse_di + V18_CORR_LOSS_ALPHA * (1.0 - pearson_r_d)
                        else:
                            batch_di_loss = mse_di
                        loss = loss + LAMBDA_DI * batch_di_loss''',
})

# ---------------------------------------------------------------------------
# Patch 5: Phase 1 正交约束 — 在正交损失之后（约第621行）
# ---------------------------------------------------------------------------
PATCHES.append({
    "id": "P5_ORTHO_P1_16D_10D",
    "description": "Phase 1 正交约束：16D 与 10D 的 Frobenius 正交惩罚",
    "find": '''            # 正交损失：惩罚 si 与 mi 的皮尔逊相关，促进解耦
            if LAMBDA_ORTHO > 0.0:
                si_vec = latent_all[:, 3]
                mi_vec = latent_all[:, 5]
                vx = si_vec - si_vec.mean()
                vy = mi_vec - mi_vec.mean()
                denom = vx.pow(2).sum().sqrt() * vy.pow(2).sum().sqrt() + 1e-8
                corr = (vx * vy).sum() / denom
                loss_ortho = corr.pow(2)
                if torch.isfinite(loss_ortho):
                    loss = loss + LAMBDA_ORTHO * loss_ortho''',
    "replace": '''            # 正交损失：惩罚 si 与 mi 的皮尔逊相关，促进解耦
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
            # v18: Phase 1 正交约束 — 16D 与 10D 的 Frobenius 正交惩罚
            if V18_LAMBDA_ORTHO_P1 > 0.0 and hasattr(model, 'get_all_latents_for_head'):
                p26_orth, init26_orth, z16_orth = model.get_all_latents_for_head(cgm, timestamps, meals, demographics)
                ode_10d = torch.cat([p26_orth, init26_orth], dim=-1)  # (B, 10)
                # Frobenius: ||ode_10d^T @ z16_orth||_F^2 / (B^2)
                cross = torch.mm(ode_10d.T, z16_orth)  # (10, 16)
                ortho_loss_p1 = cross.pow(2).sum() / (ode_10d.shape[0] ** 2)
                if torch.isfinite(ortho_loss_p1):
                    loss = loss + V18_LAMBDA_ORTHO_P1 * ortho_loss_p1''',
})

# ---------------------------------------------------------------------------
# Patch 6: Early Stopping — 在训练循环末尾（epoch 结束后）
# ---------------------------------------------------------------------------
PATCHES.append({
    "id": "P6_EARLY_STOPPING",
    "description": "Early Stopping: 基于 val loss 保存 best model",
    "find": '''        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                gb_mean_dbg = float(out_tr.param[:, 1].mean().item())
                si_cv_dbg = float(si_cv)
                print(f"  [diag] epoch={epoch+1} Gb_mean={gb_mean_dbg:.1f} si_cv={si_cv_dbg:.3f} sspg_loss={sspg_loss_epoch:.4f} di_loss={di_loss_epoch:.4f}")''',
    "replace": '''        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                gb_mean_dbg = float(out_tr.param[:, 1].mean().item())
                si_cv_dbg = float(si_cv)
                print(f"  [diag] epoch={epoch+1} Gb_mean={gb_mean_dbg:.1f} si_cv={si_cv_dbg:.3f} sspg_loss={sspg_loss_epoch:.4f} di_loss={di_loss_epoch:.4f}")
        # v18: Early Stopping
        if V18_EARLY_STOPPING_PATIENCE > 0:
            # 用 val_loss + sspg_loss + di_loss 作为综合指标
            es_metric = val_loss + sspg_loss_epoch + di_loss_epoch
            if not hasattr(main, '_es_best'):
                main._es_best = float('inf')
                main._es_counter = 0
                main._es_best_state = {}
            if es_metric < main._es_best - 1e-6:
                main._es_best = es_metric
                main._es_counter = 0
                # 保存 best state
                main._es_best_state = {
                    'model': {k: v.clone() for k, v in model.state_dict().items()},
                    'epoch': epoch + 1,
                }
                if e2e_head is not None:
                    main._es_best_state['e2e_head'] = {k: v.clone() for k, v in e2e_head.state_dict().items()}
                if sspg_head is not None:
                    main._es_best_state['sspg_head'] = {k: v.clone() for k, v in sspg_head.state_dict().items()}
                if di_head is not None:
                    main._es_best_state['di_head'] = {k: v.clone() for k, v in di_head.state_dict().items()}
            else:
                main._es_counter += 1
                if main._es_counter >= V18_EARLY_STOPPING_PATIENCE:
                    print(f"  [V18 Early Stop] No improvement for {V18_EARLY_STOPPING_PATIENCE} epochs. "
                          f"Best at epoch {main._es_best_state.get('epoch', '?')} (metric={main._es_best:.4f})")
                    break''',
})

# ---------------------------------------------------------------------------
# Patch 7: Early Stopping — 训练循环结束后恢复 best model
# 在 "# ---------- 4b. 验证集重建" 之前
# ---------------------------------------------------------------------------
PATCHES.append({
    "id": "P7_RESTORE_BEST_MODEL",
    "description": "训练循环结束后恢复 Early Stopping 的 best model",
    "find": '''    # ---------- 4b. 验证集重建：每样本 MSE + 若干示例曲线（供 VAE 拟合诊断） ----------''',
    "replace": '''    # v18: 恢复 Early Stopping 的 best model
    if V18_EARLY_STOPPING_PATIENCE > 0 and hasattr(main, '_es_best_state') and main._es_best_state:
        best_epoch = main._es_best_state.get('epoch', NUM_EPOCHS)
        print(f"  [V18] Restoring best model from epoch {best_epoch}")
        model.load_state_dict(main._es_best_state['model'])
        if 'e2e_head' in main._es_best_state and e2e_head is not None:
            e2e_head.load_state_dict(main._es_best_state['e2e_head'])
        if 'sspg_head' in main._es_best_state and sspg_head is not None:
            sspg_head.load_state_dict(main._es_best_state['sspg_head'])
        if 'di_head' in main._es_best_state and di_head is not None:
            di_head.load_state_dict(main._es_best_state['di_head'])
        # 清理
        del main._es_best_state, main._es_best, main._es_counter

    # ---------- 4b. 验证集重建：每样本 MSE + 若干示例曲线（供 VAE 拟合诊断） ----------''',
})

# ---------------------------------------------------------------------------
# Patch 8: 全量 encode 时处理分离小头
# 在 "if P1_HEAD_USE_26D and e2e_head is not None and hasattr(model, 'get_all_latents'):"
# ---------------------------------------------------------------------------
PATCHES.append({
    "id": "P8_FULL_ENCODE_SEPARATE_HEAD",
    "description": "全量 encode 时处理 v18 分离小头",
    "find": '''        if P1_HEAD_USE_26D and e2e_head is not None and hasattr(model, "get_all_latents"):
            p26, init26, z16 = model.get_all_latents(*all_tens)
            head_in = torch.cat([p26, init26], dim=-1) if P1_V8_HEAD_10D else torch.cat([p26, init26, z16], dim=-1)
            e2e_all = e2e_head(head_in)''',
    "replace": '''        if P1_HEAD_USE_26D and V18_SEPARATE_SMALL_HEAD and sspg_head is not None and hasattr(model, "get_all_latents"):
            p26, init26, z16 = model.get_all_latents(*all_tens)
            head_in = torch.cat([p26, init26, z16], dim=-1)
            sspg_all_v18 = sspg_head(head_in).squeeze(-1)
            di_all_v18 = di_head(head_in).squeeze(-1)
            # 构造 fake e2e_all 以兼容下游代码
            e2e_all = torch.stack([sspg_all_v18, di_all_v18], dim=-1)
        elif P1_HEAD_USE_26D and e2e_head is not None and hasattr(model, "get_all_latents"):
            p26, init26, z16 = model.get_all_latents(*all_tens)
            head_in = torch.cat([p26, init26], dim=-1) if P1_V8_HEAD_10D else torch.cat([p26, init26, z16], dim=-1)
            e2e_all = e2e_head(head_in)''',
})

# ---------------------------------------------------------------------------
# Patch 9: checkpoint 保存时加入分离小头
# ---------------------------------------------------------------------------
PATCHES.append({
    "id": "P9_CKPT_SAVE_SEPARATE",
    "description": "checkpoint 保存时加入 v18 分离小头 state",
    "find": '''    if e2e_head is not None:
        ckpt["e2e_head_state"] = e2e_head.state_dict()''',
    "replace": '''    if e2e_head is not None:
        ckpt["e2e_head_state"] = e2e_head.state_dict()
    # v18: 保存分离小头 state
    ckpt["V18_SEPARATE_SMALL_HEAD"] = V18_SEPARATE_SMALL_HEAD
    ckpt["V18_CORR_LOSS_ALPHA"] = V18_CORR_LOSS_ALPHA
    ckpt["V18_EARLY_STOPPING_PATIENCE"] = V18_EARLY_STOPPING_PATIENCE''',
})


# ============================================================================
# Apply / Verify / Revert
# ============================================================================

def apply_patches():
    if not PIPELINE.exists():
        print(f"[FATAL] {PIPELINE} not found")
        sys.exit(1)

    # Backup
    if not BACKUP.exists():
        shutil.copy2(PIPELINE, BACKUP)
        print(f"[BACKUP] {PIPELINE} → {BACKUP}")
    else:
        print(f"[BACKUP] Already exists: {BACKUP}")

    code = PIPELINE.read_text()
    applied = 0
    skipped = 0

    for p in PATCHES:
        if p["find"] not in code:
            if p["replace"].split('\n')[0] in code:
                print(f"  [SKIP] {p['id']}: already applied")
                skipped += 1
                continue
            print(f"  [WARN] {p['id']}: marker not found — patch may conflict with existing modifications")
            print(f"         Looking for: {p['find'][:80]}...")
            skipped += 1
            continue
        code = code.replace(p["find"], p["replace"], 1)
        applied += 1
        print(f"  [OK] {p['id']}: {p['description']}")

    PIPELINE.write_text(code)
    print(f"\n[DONE] Applied {applied} patches, skipped {skipped}")


def verify_patches():
    if not PIPELINE.exists():
        print(f"[FATAL] {PIPELINE} not found")
        sys.exit(1)

    code = PIPELINE.read_text()
    markers = [
        ("V18_EARLY_STOPPING_PATIENCE", "Early Stopping env var"),
        ("V18_CORR_LOSS_ALPHA", "CorrLoss env var"),
        ("V18_SEPARATE_SMALL_HEAD", "Separate small head env var"),
        ("V18_LAMBDA_ORTHO_P1", "Phase 1 orthogonality env var"),
        ("[V18] Separate small heads", "Separate head architecture"),
        ("v18 CorrLoss", "CorrLoss in training loop"),
        ("[V18 Early Stop]", "Early stopping break"),
        ("[V18] Restoring best model", "Best model restoration"),
        ("ortho_loss_p1", "Phase 1 orthogonality loss"),
    ]

    ok = 0
    fail = 0
    for marker, desc in markers:
        if marker in code:
            print(f"  [OK] {desc}")
            ok += 1
        else:
            print(f"  [FAIL] {desc} — marker '{marker}' not found")
            fail += 1

    if fail == 0:
        print(f"\n[PASS] All {ok} patches verified")
    else:
        print(f"\n[FAIL] {fail}/{ok+fail} patches missing")
    return fail == 0


def revert():
    if not BACKUP.exists():
        print(f"[WARN] No backup found: {BACKUP}")
        return
    shutil.copy2(BACKUP, PIPELINE)
    print(f"[REVERTED] {BACKUP} → {PIPELINE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if args.revert:
        revert()
    elif args.apply:
        apply_patches()
        print("\nVerifying...")
        verify_patches()
    elif args.verify:
        verify_patches()
    else:
        print("Usage: python v18_patches.py --apply | --verify | --revert")
