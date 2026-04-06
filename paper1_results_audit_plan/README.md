# P1 项目代码审计报告与后续实验计划 — 执行结果

本文件夹存放按《P1项目代码审计报告与后续实验计划》(v1.0, 2026-03-01) 执行后的**新实验结果**，与审计结论对应的**基线及扫描 run** 均置于 `paper1_results_audit_plan/` 下。

---

## 审计核心结论（摘要）

- **金标准映射正确**：D1/D2/D4 的 SSPG、DI 与 labels.csv 对应关系正确。
- **训练样本数不一致**：原实验或为每受试者 1 个餐窗（~127 样本），当前默认逻辑为每餐窗一样本（D2/D4 多餐窗 → 332 样本），导致复现差异。
- **参数不稳定**：si 与 SSPG、mi 与 DI 的相关系数随随机种子波动大，甚至符号反转，提示**参数不可辨识**。
- **si 范围过窄**：原 [1e-4, 1e-3] 仅 10 倍，限制受试者间 si 变异。

---

## 已实施的代码与实验

### 阶段一：基础修正与稳定性基线

| 项目 | 实施内容 |
|------|----------|
| **1.1 扩大参数范围** | `models.py`：环境变量 `P1_WIDE_PARAM_RANGE=1` 时，si 改为 [1e-5, 1e-2]，mi 改为 [0.05, 5.0]。 |
| **1.2 统一训练样本** | `run_p1_full_pipeline.py`：`P1_ONE_MEAL_PER_SUBJECT=1` 时，在堆叠 D1/D2/D4 后按受试者只保留**第一个**餐窗，训练样本数由 332 降为 **127**。 |
| **1.3 多种子基线** | 在 **无** SSPG/DI/Ortho 监督（LAMBDA_SSPG=0, LAMBDA_DI=0, LAMBDA_ORTHO=0）、宽范围 + 每受试者一餐窗下，跑 **10 个随机种子**，建立稳定性基线。 |

### 阶段二（部分）：损失权重扫描

- **2.1 损失权重扫描**：在相同设定（宽范围 + 每受试者一餐窗）下，跑若干权重组合（如仅 LAMBDA_SSPG=0.1、仅 LAMBDA_DI=0.15、以及 SSPG+DI+Ortho 组合），结果见下表及子目录。

---

## 阶段一 1.3：10 种子基线（无监督）— 测试集 si vs SSPG / mi vs DI

所有 run：`P1_WIDE_PARAM_RANGE=1`，`P1_ONE_MEAL_PER_SUBJECT=1`，LAMBDA_SSPG=0，LAMBDA_DI=0，LAMBDA_ORTHO=0。

| 子目录 | Seed | si vs SSPG (r) | mi vs DI (r) | SSPG E2E r | DI E2E r |
|--------|------|----------------|--------------|------------|----------|
| baseline_wide_onemeal_seed21  | 21   | +0.32  | +0.83 | 0.47 | 0.49 |
| baseline_wide_onemeal_seed42  | 42   | +0.37  | +0.30 | 0.09 | -0.30 |
| baseline_wide_onemeal_seed43  | 43   | +0.03  | -0.03 | -0.09 | -0.54 |
| baseline_wide_onemeal_seed44  | 44   | -0.18  | -0.02 | -0.12 | 0.07 |
| baseline_wide_onemeal_seed100 | 100  | +0.57  | +0.66 | 0.29 | -0.77 |
| baseline_wide_onemeal_seed7   | 7    | -0.09  | -0.20 | -0.03 | -0.60 |
| baseline_wide_onemeal_seed123 | 123  | -0.36  | -0.40 | -0.11 | 0.40 |
| baseline_wide_onemeal_seed2024| 2024 | +0.61  | -0.41 | -0.49 | -0.29 |
| baseline_wide_onemeal_seed2025| 2025 | -0.33  | -0.21 | 0.04 | 0.02 |
| baseline_wide_onemeal_seed99  | 99   | +0.05  | -0.26 | 0.12 | 0.49 |

**解读**：si vs SSPG 与 mi vs DI 在不同种子下符号与幅度波动大，与审计报告所述“参数不可辨识、角色互换”一致；无监督基线下单参数与金标准关系**不稳定**，需阶段二权重扫描及后续辨识度增强（2.2–2.4、阶段三）进一步优化。

---

## 阶段二 2.1：损失权重扫描（宽范围 + 每受试者一餐窗）

由 **`scripts/run_audit_plan_full.sh`** 统一跑齐，包含：

- **单变量**：scan_sspg{0.01,0.05,0.1,0.2,0.5,1.0}、scan_di{0.01,0.05,0.15,0.3}、scan_ortho{0.05,0.1,0.2}
- **组合**：scan_sspg0.1_di0.1、scan_sspg0.1_di0.15_ortho0.05、scan_sspg0.2_di0.15_ortho0.05

上述 run 均使用 P1_WIDE_PARAM_RANGE=1、P1_ONE_MEAL_PER_SUBJECT=1、P1_SEED=21、P1_ZSCORE_TARGETS=1、P1_DECOUPLE_SSPG=1。

## 阶段二 2.2：ODE 简化（固定 sg/p2）

| 子目录 | 说明 |
|--------|------|
| ode4param_sspg0.1_di0.15_ortho0.05 | P1_FIX_SG_P2=1，仅学 tau_m, Gb, si, mi；sg=0.01、p2=1/30 固定 |

## 阶段二 2.3：先验 si×mi–DI

| 子目录 | 说明 |
|--------|------|
| prior_prod_sspg0.1_di0.15_ortho0.05 | P1_DI_PRODUCT_CONSTRAINT=1（DI=si×mi） |
| prior_logprod_sspg0.1_di0.15_ortho0.05 | P1_DI_LOG_PRODUCT=1（DI 头输入 log(si)+log(mi)） |

## 阶段三：全量数据最终训练

| 子目录 | 说明 |
|--------|------|
| final_fullmeal_sspg0.1_di0.15_ortho0.05 | P1_ONE_MEAL_PER_SUBJECT=0，用全部餐窗（332 样本）+ 最佳权重训练 |

各子目录内含 `latent_and_gold_all.csv`、`e2e_head_metrics.json`、`correlations.txt`、`evaluation_metrics_summary.txt` 等。

---

## 每 run 输出文件

- `latent_and_gold_all.csv`、`latent_and_gold_test.csv`
- `e2e_head_metrics.json`、`correlations.txt`
- `evaluation_metrics_summary.txt`、`evaluation_metrics.csv`、`evaluation_5fold_per_fold.csv`（已对 10 个基线 run 执行 5 折评估）
- `autoencoder_p1_full.pt`、`training_curves.json`

---

## 复现与扩展命令

**阶段一基线（单种子，无监督）：**
```bash
export P1_WIDE_PARAM_RANGE=1 P1_ONE_MEAL_PER_SUBJECT=1
export LAMBDA_SSPG=0 LAMBDA_DI=0 LAMBDA_ORTHO=0
P1_RESULTS_DIR=paper1_results_audit_plan/baseline_wide_onemeal_seed21 P1_SEED=21 \
  python run_p1_full_pipeline.py
```

**阶段二扫描（例如仅 SSPG）：**
```bash
export P1_WIDE_PARAM_RANGE=1 P1_ONE_MEAL_PER_SUBJECT=1 P1_ZSCORE_TARGETS=1 P1_DECOUPLE_SSPG=1
P1_RESULTS_DIR=paper1_results_audit_plan/scan_sspg0.1 P1_SEED=21 \
  LAMBDA_SSPG=0.1 LAMBDA_DI=0 LAMBDA_ORTHO=0 \
  python run_p1_full_pipeline.py
```

**5 折评估：**
```bash
python scripts/evaluate_p1_metrics.py --csv paper1_results_audit_plan/<子目录>/latent_and_gold_all.csv --out paper1_results_audit_plan/<子目录>
```

---

## 已全部实现的审计项

- **2.2 ODE 结构简化**：已实现。`models.py` 支持 **P1_FIX_SG_P2=1**，固定 sg=0.01、p2=1/30，仅学习 tau_m, Gb, si, mi 四参数。run 见 `ode4param_*`。
- **2.3 先验 si×mi–DI**：在审计设定下用 **P1_DI_PRODUCT_CONSTRAINT=1**（乘积）或 **P1_DI_LOG_PRODUCT=1**（对数乘积）运行，run 见 `prior_prod_*`、`prior_logprod_*`。
- **2.1 损失权重完整扫描**：单变量（LAMBDA_SSPG 0.01–1.0、LAMBDA_DI 0.01–0.3、LAMBDA_ORTHO 0.05–0.2）及组合由 **`scripts/run_audit_plan_full.sh`** 一键执行。
- **阶段三**：留一数据集交叉验证已由 `evaluate_p1_metrics.py` 输出（Leave-one-dataset-out）；全量数据训练用 **P1_ONE_MEAL_PER_SUBJECT=0** + 最佳配置，run 见 `final_fullmeal_*`。

## 2.4 替代模型（单室等）

审计计划中的「更简单的、可能辨识度更高的生理模型（例如只有一个葡萄糖室）」需**重新设计 ODE 与接口**，作为独立开发任务保留，不在此次 pipeline 内实现。后续若实现，建议新文件（如 `models_one_compartment.py`）并接回 `run_p1_full_pipeline.py` 的选项。

## 一键跑齐阶段二+三

```bash
bash scripts/run_audit_plan_full.sh
```

跑完后对每个子目录执行 5 折与 LODO 评估：

```bash
for d in paper1_results_audit_plan/scan_* paper1_results_audit_plan/ode4param_* paper1_results_audit_plan/prior_* paper1_results_audit_plan/final_*; do
  [ -d "$d" ] && python scripts/evaluate_p1_metrics.py --csv "$d/latent_and_gold_all.csv" --out "$d"
done
```
