# P1 全面审计与行动指南（对照 v4.0 PDF）

本文档与《P1项目全面审计与行动指南 (v4.0).pdf》对齐，汇总已执行项与金标准数据口径。

---

## 1. 金标准数据口径（数据清理后）

| 金标准 | D1 | D2 | D4 | 总计（用于训练/评估） |
|--------|----|----|-----|----------------------|
| **SSPG** | 32 | 31 | 20 | **83 subjects** |
| **DI (Bergman)** | 32 | 31 | **0** | **63 subjects** |
| **HOMA-IR** | 32 | 37 | 28 | **97 subjects** |

- **DI 污染已修复**：D4 的 `insulin_rate_dd` 不再被当作 `di`（见 `load_cgm_project_data.py` 注释）。D4 不提供 Bergman DI。
- **D4 Oral DI（22 人）**：D4 有 22 人同时有 CGM 和可计算的 Oral DI。Oral DI 与 Bergman DI 量纲不同，**不混入训练**。正确用法：用 D1+D2 的 Bergman DI 训练好模型后，在该 22 人上做**独立外部验证**（预测 DI，与真实 Oral DI 算 Spearman）；若相关显著，可作为泛化证据。
- **无监督预训练**：可使用 D1–D5 全部 CGM（约 6234 个 meal windows，270 subjects）。
- **监督训练**：使用上述有金标准标签的 subjects 所对应的 meal windows。

---

## 2. 四个模型（M0 / M1 / M2 / M3）

| 模型 | 描述 | 运行方式 |
|------|------|----------|
| **M0** | 基线：无监督/半监督 VAE + 训练后 Ridge 回归 | `LAMBDA_SSPG=0 LAMBDA_DI=0 python run_p1_full_pipeline.py`，再用 `scripts/evaluate_p1_metrics.py` 做 post-hoc 评估 |
| **M1** | 核心创新：端到端监督 VAE，loss 中直接加入 SSPG/DI 预测误差 | 建议先用 **λ=0.1** 防模式崩溃：`LAMBDA_SSPG=0.1 LAMBDA_DI=0.1`；可选 `BETA_HAT=0.05`、`P1_ZSCORE_TARGETS=1`。主结果看终端及 `e2e_head_metrics.json` |
| **M2** | 黑盒消融：端到端监督的黑盒 VAE（Decoder 为 LSTM，非 ODE） | `python run_p1_m2_blackbox.py`（见脚本内环境变量） |
| **M3** | 简单 NN 消融：仅 Encoder + 预测头，无 VAE、无 Decoder | `python run_p1_m3_direct.py`（见脚本内环境变量） |

- **M1/M2/M3** 的主结果均来自各训练脚本在测试集上的**端到端 head 评估**，不要用 `evaluate_p1_metrics.py` 作为其主结果来源（该脚本仅用于 M0）。

---

## 3. 已完成的代码修改（按 PDF 要求）

1. **load_cgm_project_data.py**：已注释掉将 `insulin_rate_dd` 当作 `di` 的代码（约 409–410 行），并加注释说明量纲不同。
2. **run_p1_full_pipeline.py**：  
   - LAMBDA_SSPG / LAMBDA_DI 通过环境变量控制（默认 0，M1 设为 1.0）。  
   - 在「7. 保存」前增加端到端 head 在测试集上的评估，并写入 `e2e_head_metrics.json`。  
3. **scripts/evaluate_p1_metrics.py**：docstring 中说明本脚本用于 M0，M1 主结果来自 pipeline 的端到端评估。
4. **run_p1_m2_blackbox.py**（M2）：BlackboxAutoencoder + Linear(latent→6) + 同一套 SSPG/DI/IR 头；loss = 重建 MSE + BETA_HAT×KL + λ_sspg×SSPG + λ_di×DI + λ_ir×IR；测试集端到端评估写入 `P1_RESULTS_DIR/e2e_head_metrics.json`（默认 `paper1_results_m2`）。
5. **run_p1_m3_direct.py**（M3）：DirectNN 训练与端到端评估，结果写入 `paper1_results_m3/e2e_head_metrics.json`。

---

## 4. M1 效果不佳时的调优建议（PDF 第四步）

- **Loss 权重 λ**：相对 CGM 重建 MSE 调整。**防模式崩溃优先**：先用 `LAMBDA_SSPG=0.1 LAMBDA_DI=0.1`，再试 0.05、0.01。
- **KL 权重 β**：`BETA_HAT` 通过环境变量设置（默认 0.01）。加重正则防坍缩可试 `BETA_HAT=0.05` 或 `0.1`。
- **学习率**：`P1_LR` 环境变量（默认 1e-2），可试 1e-3。
- **预测头**：若需非线性，可将 `sspg_head` / `di_head` 改为小 MLP，如 Linear(6, 16) → ReLU → Linear(16, 1)。
- **目标标准化**：`P1_ZSCORE_TARGETS=1` 时对 SSPG/DI 真值 z-score 后再算 loss，预测时自动反标准化；checkpoint 中会保存 mean/std。

---

## 5. M1 模式崩溃诊断与修复（诊断报告 v1.0）

**现象**：M1 在 λ_sspg=λ_di=1.0 时出现 **模式崩溃**——tau_m、Gb、mi 等大量卡在 param_lims 上限，latent 失去生理意义，DI 预测呈负相关。

**已实现修复**：
1. **放宽参数边界**（`models.py`）：`tau_m` 上限 60→120，`Gb` 范围 [80,200]→[60,250]，减轻“卡上限”压力。
2. **BETA_HAT 可配置**：`BETA_HAT` 从环境变量读取，可设为 0.05 或 0.1 加强 KL 正则。
3. **P1_ZSCORE_TARGETS**：`P1_ZSCORE_TARGETS=1` 时对 SSPG/DI 做 z-score 再算监督 loss，评估时反标准化；checkpoint 保存 `sspg_mean/std`、`di_mean/std`。
4. **推荐 M1 命令**：先降 λ 再跑，例如  
   `LAMBDA_SSPG=0.1 LAMBDA_DI=0.1 BETA_HAT=0.05 P1_ZSCORE_TARGETS=1 python run_p1_full_pipeline.py`

**备选**：两阶段训练（先 M0 预训练再冻部分 encoder 微调）、或 sspg_head/di_head 改为 2 层 MLP（见上节）。

---

## 6. 评估指标与验证策略

- **主要指标**：Spearman 相关系数 (r)。  
- **次要指标**：RMSE、R²。  
- **验证**：主体为 5-fold subject-wise 交叉验证；泛化可做留一数据集 (LODO)，例如 D1+D2 训练、D4 上测 SSPG。
