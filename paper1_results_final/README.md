# P1 尽善尽美实验方案 (v2.0 FINAL) — 结果汇总

本文件夹按《P1 尽善尽美实验方案 (v2.0 FINAL)》执行实验，结果统一保存在 `paper1_results_final/` 下各子目录。

## 背景：尺度不匹配与对策

- **问题**：si×mi 数量级约 [1e-5, 3e-3]，而 z-score 后 DI 约 [-3, +3]，相差约 1000 倍，梯度微弱。
- **对策 1**：**对数乘积** DI = scale·(log(si)+log(mi))+bias，Linear(1,1) 头，尺度与 z-score 匹配。
- **对策 2**：**MLP(si, mi)** 2→16→8→1，仅用 si/mi 预测 DI，可学习近似乘积的非线性映射。

通用前缀：`P1_ZSCORE_TARGETS=1 BETA_HAT=0.05 P1_DECOUPLE_SSPG=1`（SSPG 头仅用 si）。

---

## 实验系列一：对数乘积约束 (Log-Product) — 8 个

| 编号 | 子目录 | λ_sspg | λ_di | λ_ortho | 说明 |
|------|--------|--------|------|---------|------|
| 1.1 | m1_logprod_base | 0.1 | 0.15 | 0.0 | 基线，无正交 |
| 1.2 | m1_logprod_ortho0.05 | 0.1 | 0.15 | 0.05 | 弱正交 |
| 1.3 | m1_logprod_ortho0.1 | 0.1 | 0.15 | 0.1 | 中正交 |
| 1.4 | m1_logprod_ortho0.2 | 0.1 | 0.15 | 0.2 | 强正交 |
| 1.5 | m1_logprod_sspg0.2 | 0.2 | 0.15 | 0.05 | 强 SSPG 锚点 |
| 1.6 | m1_logprod_sspg0.05 | 0.05 | 0.15 | 0.05 | 弱 SSPG 锚点 |
| 1.7 | m1_logprod_di0.3 | 0.1 | 0.3 | 0.05 | 强 DI 监督 |
| 1.8 | m1_logprod_di0.05 | 0.1 | 0.05 | 0.05 | 弱 DI 监督 |

启用：`P1_DI_LOG_PRODUCT=1`。

---

## 实验系列二：非线性 MLP DI 头 — 4 个

| 编号 | 子目录 | λ_sspg | λ_di | λ_ortho | 说明 |
|------|--------|--------|------|---------|------|
| 2.1 | m1_mlp_base | 0.1 | 0.15 | 0.0 | MLP 基线 |
| 2.2 | m1_mlp_ortho0.05 | 0.1 | 0.15 | 0.05 | MLP + 弱正交 |
| 2.3 | m1_mlp_ortho0.1 | 0.1 | 0.15 | 0.1 | MLP + 中正交 |
| 2.4 | m1_mlp_sspg0.2 | 0.2 | 0.15 | 0.05 | MLP + 强 SSPG |

启用：`P1_DI_MLP_HEAD=1`。

---

## 实验系列三：训练策略优化（在 1.2 配置上叠加）— 5 个

| 编号 | 子目录 | 变更 | 说明 |
|------|--------|------|------|
| 3A | m1_logprod_ortho0.05_cosine | P1_USE_LR_SCHEDULER=1 | CosineAnnealing LR |
| 3B | m1_logprod_ortho0.05_200ep | P1_NUM_EPOCHS=200 | 延长训练 |
| 3C | m1_logprod_ortho0.05_lr1e-3 | P1_LR=1e-3 | 更低学习率 |
| 3D | m1_logprod_ortho0.05_kl0.1 | BETA_HAT=0.1 | 更大 KL 权重 |
| 3E | m1_logprod_ortho0.05_kl0.01 | BETA_HAT=0.01 | 更小 KL 权重 |

基配置：1.2（logprod, λ_sspg=0.1, λ_di=0.15, λ_ortho=0.05）。

---

## E2E 头指标速览（测试集 Spearman r）

| 子目录 | SSPG r | DI r |
|--------|--------|------|
| m1_logprod_base | 0.11 | 0.15 |
| m1_logprod_ortho0.05 | -0.32 | -0.01 |
| m1_logprod_ortho0.1 | -0.27 | -0.63 |
| m1_logprod_ortho0.2 | -0.16 | -0.68 |
| m1_logprod_sspg0.2 | 0.14 | -0.34 |
| m1_logprod_sspg0.05 | 0.16 | -0.43 |
| m1_logprod_di0.3 | -0.02 | **0.47** |
| m1_logprod_di0.05 | 0.17 | -0.37 |
| m1_mlp_base | -0.14 | **0.47** |
| m1_mlp_ortho0.05 | -0.10 | 0.40 |
| m1_mlp_ortho0.1 | -0.11 | 0.31 |
| m1_mlp_sspg0.2 | -0.35 | 0.37 |
| m1_logprod_ortho0.05_cosine | 0.17 | -0.31 |
| m1_logprod_ortho0.05_200ep | 0.09 | 0.53 |
| m1_logprod_ortho0.05_lr1e-3 | **0.52** | 0.04 |
| m1_logprod_ortho0.05_kl0.1 | -0.18 | -0.28 |
| m1_logprod_ortho0.05_kl0.01 | -0.13 | **0.78** |

- **SSPG E2E 最佳**：m1_logprod_ortho0.05_lr1e-3（r=0.52）。
- **DI E2E 最佳**：m1_logprod_ortho0.05_kl0.01（r=0.78）；次之 m1_logprod_ortho0.05_200ep（0.53）、m1_logprod_di0.3 / m1_mlp_base（0.47）。

---

## 每 run 输出文件

- `latent_and_gold_all.csv`、`latent_and_gold_test.csv`
- `e2e_head_metrics.json`、`correlations.txt`
- `evaluation_metrics_summary.txt`、`evaluation_metrics.csv`、`evaluation_5fold_per_fold.csv`
- `autoencoder_p1_full.pt`、`training_curves.json`

5 折评估已对上述 17 个子目录执行。

---

## 复现命令示例

```bash
# 系列一 1.1
export P1_ZSCORE_TARGETS=1 BETA_HAT=0.05 P1_DECOUPLE_SSPG=1 P1_DI_LOG_PRODUCT=1
P1_RESULTS_DIR=paper1_results_final/m1_logprod_base \
  LAMBDA_SSPG=0.1 LAMBDA_DI=0.15 LAMBDA_ORTHO=0.0 \
  python run_p1_full_pipeline.py

# 系列二 2.1
export P1_ZSCORE_TARGETS=1 BETA_HAT=0.05 P1_DECOUPLE_SSPG=1 P1_DI_MLP_HEAD=1
P1_RESULTS_DIR=paper1_results_final/m1_mlp_base \
  LAMBDA_SSPG=0.1 LAMBDA_DI=0.15 LAMBDA_ORTHO=0.0 \
  python run_p1_full_pipeline.py

# 系列三 3C（更低学习率）
export P1_ZSCORE_TARGETS=1 BETA_HAT=0.05 P1_DECOUPLE_SSPG=1 P1_DI_LOG_PRODUCT=1 P1_LR=1e-3
P1_RESULTS_DIR=paper1_results_final/m1_logprod_ortho0.05_lr1e-3 \
  LAMBDA_SSPG=0.1 LAMBDA_DI=0.15 LAMBDA_ORTHO=0.05 \
  python run_p1_full_pipeline.py
```

评估：

```bash
python scripts/evaluate_p1_metrics.py --csv paper1_results_final/<子目录>/latent_and_gold_all.csv --out paper1_results_final/<子目录>
```

---

## 方案中未在本轮执行的实验

- **系列四**：参数范围扩展（P1_WIDE_PARAM_RANGE=1，需在 `models.py` 中实现 si/mi 范围扩展）。
- **系列五**：多种子稳定性（在最佳配置上用 P1_SEED=42/123/7 各跑一次）。

可按需在实现系列四后补跑，并在最佳配置上跑系列五。
