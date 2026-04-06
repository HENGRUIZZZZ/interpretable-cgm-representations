# P1 系统性优化实验方案 (v1.0) — 结果汇总

本文件夹按《P1 系统性优化实验方案 (v1.0)》执行全部实验，结果统一保存在 `paper1_results_systematic/` 下各子目录。

## 实验分组与配置

### 实验 1：DI 乘积约束 (DI = si × mi，无单独 DI 头)

| 子目录 | 说明 | 环境变量 |
|--------|------|----------|
| **m1_prod_base** | 1.1 基线 | `P1_DI_PRODUCT_CONSTRAINT=1` LAMBDA_SSPG=0.05 LAMBDA_DI=0.15 LAMBDA_ORTHO=0 P1_ZSCORE_TARGETS=1 |
| **m1_prod_ortho0.2** | 1.2 加正交 | 同上 + LAMBDA_ORTHO=0.2 |

### 实验 2：正交微调 (解耦 + 正交，保留 DI 线性头)

| 子目录 | 说明 | 环境变量 |
|--------|------|----------|
| **m1_decouple_ortho_0.2** | 2.1 | P1_DECOUPLE_SSPG=1 LAMBDA_SSPG=0.05 LAMBDA_DI=0.15 **LAMBDA_ORTHO=0.2** BETA_HAT=0.05 P1_ZSCORE_TARGETS=1，100 epochs |
| **m1_decouple_ortho_0.3** | 2.2 | 同上，**LAMBDA_ORTHO=0.3** |

### 实验 3：训练超参数 (长跑)

| 子目录 | 说明 | 环境变量 |
|--------|------|----------|
| **m1_decouple_ortho_1.0_longrun** | 3.1 | P1_DECOUPLE_SSPG=1 LAMBDA_ORTHO=1.0 LAMBDA_SSPG=0.05 LAMBDA_DI=0.15 **P1_NUM_EPOCHS=200** **BETA_HAT=0.1** P1_ZSCORE_TARGETS=1 |

---

## 关键指标速览

- **端到端 (E2E)**：测试集上 SSPG/DI 预测头的 Spearman r、RMSE（见各子目录 `e2e_head_metrics.json`）。
- **潜变量相关**：si vs SSPG、mi vs SSPG、mi vs DI（及实验 1 的 si×mi vs DI）见各子目录 `correlations.txt` 与 `evaluation_metrics_summary.txt`。
- **5 折评估**：`python scripts/evaluate_p1_metrics.py --csv <子目录>/latent_and_gold_all.csv --out <子目录>` 已对上述 5 个 run 执行，生成 `evaluation_metrics_summary.txt`、`evaluation_5fold_per_fold.csv` 等。

### E2E 头 (测试集)

| Run | SSPG Spearman r | SSPG RMSE | DI Spearman r | DI RMSE |
|-----|-----------------|-----------|---------------|---------|
| m1_prod_base | 0.546 | 55.17 | 0.379 | 0.97 |
| m1_prod_ortho0.2 | 0.471 | 65.55 | 0.279 | 0.97 |
| m1_decouple_ortho_0.2 | -0.100 | 68.77 | 0.379 | 1.04 |
| m1_decouple_ortho_0.3 | -0.172 | 68.79 | 0.379 | 0.56 |
| m1_decouple_ortho_1.0_longrun | 0.461 | 69.21 | -0.125 | 1.86 |

---

## 每 run 输出文件

- `latent_and_gold_all.csv`、`latent_and_gold_test.csv`：潜变量与金标准
- `e2e_head_metrics.json`：端到端 SSPG/DI 指标
- `correlations.txt`：潜变量与金标准 Spearman 相关
- `evaluation_metrics_summary.txt`、`evaluation_metrics.csv`、`evaluation_5fold_per_fold.csv`：5 折与全量评估
- `autoencoder_p1_full.pt`：模型与头权重
- `training_curves.json`：每 epoch 训练/验证损失

复现命令示例（项目根目录）：

```bash
# 实验 1.1
P1_DI_PRODUCT_CONSTRAINT=1 LAMBDA_SSPG=0.05 LAMBDA_DI=0.15 LAMBDA_ORTHO=0 \
  P1_RESULTS_DIR=paper1_results_systematic/m1_prod_base P1_ZSCORE_TARGETS=1 \
  python run_p1_full_pipeline.py

# 实验 2.1
P1_DECOUPLE_SSPG=1 LAMBDA_SSPG=0.05 LAMBDA_DI=0.15 LAMBDA_ORTHO=0.2 BETA_HAT=0.05 \
  P1_RESULTS_DIR=paper1_results_systematic/m1_decouple_ortho_0.2 P1_ZSCORE_TARGETS=1 \
  python run_p1_full_pipeline.py

# 实验 3.1
P1_DECOUPLE_SSPG=1 LAMBDA_ORTHO=1.0 LAMBDA_SSPG=0.05 LAMBDA_DI=0.15 \
  P1_NUM_EPOCHS=200 BETA_HAT=0.1 \
  P1_RESULTS_DIR=paper1_results_systematic/m1_decouple_ortho_1.0_longrun P1_ZSCORE_TARGETS=1 \
  python run_p1_full_pipeline.py
```

评估（已对上述 5 个 run 执行）：

```bash
python scripts/evaluate_p1_metrics.py --csv paper1_results_systematic/<子目录>/latent_and_gold_all.csv --out paper1_results_systematic/<子目录>
```
