# P1 评估指标与可优化点

## 1. 评估参数说明

脚本 `scripts/evaluate_p1_metrics.py` 对 SSPG/DI 预测输出：

| 指标 | 含义 |
|------|------|
| **RMSE** | 均方根误差，与金标准同单位（SSPG: mg/dL 量级；DI: 无量纲） |
| **MAE** | 平均绝对误差 |
| **R²** | 决定系数；<0 表示预测比“预测均值”还差 |
| **Pearson r** | 线性相关；适合回归质量 |
| **Spearman r** | 秩相关；适合单调关系、对异常值稳健 |

验证方式：

- **full_sample**：全量拟合单潜变量→金标准，仅作描述（有过拟合）。
- **5fold_subject**：按 subject 做 5-fold，Ridge(6D latent→金标准)，报告 mean±std。
- **leave_one_dataset_out**：用两个数据集训 Ridge，在第三个上测，看跨数据集泛化。

## 2. 当前结果摘要（示例）

- **全量、单潜变量**：mi→SSPG Spearman r≈-0.41，RMSE≈70；mi→DI 几乎无相关。
- **5-fold Ridge 头**：SSPG Pearson r≈0.40，Spearman≈0.38，RMSE≈73；DI Pearson r≈0.54，Spearman≈0.65，RMSE≈1.03。
- **留一数据集出**：SSPG 在 D4 上 r≈0.60；DI 在 D2/D4 上易变（尺度/分布差异）。

## 3. 可优化方向

### 3.1 训练数据

- **划分比例**：当前 80/10/10；可试 70/15/15 或 85/5/10，看 val 早停与 test 稳定性。
- **分层**：按 `dataset_id` 或诊断分层划分，保证各 set 中 D1/D2/D4 比例接近。
- **金标准子集**：仅用“同时有 SSPG+DI”的人做评估，避免缺失模式偏倚。
- **样本量**：D4 有金标准者较少；若获 D1 的 validation cohort 金标准，可显著增加 n。

### 3.2 验证模式

- **多 seed**：固定 3–5 个 seed 重训 encoder + 同一评估脚本，报告 r / RMSE 的 mean±std 或 95% CI。
- **Subject-level CV**：已用 GroupKFold 按 subject 5-fold；可试 10-fold 或重复 5-fold 多 seed 取平均。
- **留一数据集出**：已实现；可加“留一 cohort（如仅 D4）训练、其余测试”的表格，专门看泛化。
- **Bootstrap**：对 test 或全量有金标准者做 bootstrap 抽样，报告 r / RMSE 的置信区间。

### 3.3 模型与预测头

- **Encoder**：学习率、β（KL 权重）、epoch 数、hidden_size；可做小网格搜索，用 val 重建损失或 val 上 Ridge 头的 r 选参。
- **IR 弱监督头**：已在 `run_p1_full_pipeline.py` 中实现。用 latent 的 `[si, mi, tau_m]` 经线性头预测 `log(HOMA_IR+1)`，仅对带 HOMA_IR 的样本计算 MSE，损失权重 `LAMBDA_IR=0.05`（可试 0.01、0.1）。目的：在不破坏重建的前提下拉高 latent→SSPG/HOMA-IR 相关性。checkpoint 中保存 `ir_head_state`、`IR_LATENT_IX`、`LAMBDA_IR`；加载时需构造 `nn.Linear(3, 1)` 再 `load_state_dict(ckpt["ir_head_state"])`。
- **归一化**：金标准 log 变换（如 log(DI)）有时能提升 Ridge 的 R² 和 RMSE；可对比 raw vs log。
- **预测头**：Ridge 的 alpha 已做网格搜索；可试 ElasticNet、小 MLP（1–2 层），或对 SSPG/DI 联合建模（多任务）。
- **潜变量选择**：当前用 6D；可做 L1 或相关分析只保留 si/mi/tau_m 等，减少过拟合。

### 3.4 实现建议

- 在 `run_p1_full_pipeline.py` 中支持通过环境变量或命令行传入 `SEED`，便于多 seed 批量跑。
- 将 `evaluate_p1_metrics.py` 的入口改为可指定 `--validation kfold_subject|holdout|leave_one_dataset_out` 和 `--target sspg|di|both`。
- 把 RMSE/MAE/R²/r 的汇总表写入 `paper1_results/evaluation_metrics.csv`，便于制表与画图。

## 4. 如何跑评估

```bash
# 先跑完整 pipeline 生成 latent_and_gold_all.csv
export CGM_PROJECT_OUTPUT=/path/to/cgm_project/output
python run_p1_full_pipeline.py

# 再跑评估（RMSE、MAE、R²、Pearson/Spearman r，5-fold + 留一数据集出）
python scripts/evaluate_p1_metrics.py --csv paper1_results/latent_and_gold_all.csv --out paper1_results
```

输出：`paper1_results/evaluation_metrics.csv`、`paper1_results/evaluation_metrics_summary.txt`。
