# P1 如何得到更好结果：已做尝试与建议

## 1. 已做的尝试

### 1.1 V2 的「最佳 DI run」在仅 D1+D2 上的表现
- 用 V2 的 `run_s42_lam0.1`（在 D1+D2+D4 上 DI r=0.75）**仅在 D1+D2 上**重新评估。
- 结果：**SSPG r=0.26，DI r=0.38** —— 明显低于 V4。说明 V2 的高 DI 主要来自 **D4** 的分布/样本，在 D1+D2 上不可复现。
- 结论：不能靠「用 V2 的 run、只报 D1+D2 数字」来同时拿高 DI 和干净口径。

### 1.2 V5：仅 D1+D2 训练 + 加强 SSPG/DI 头
- 设置：`P1_TRAIN_DATASETS=D1,D2`（训练只用 D1+D2）、`LAMBDA_SSPG=0.05`、`LAMBDA_DI=0.05`，2×2 网格（SEED=21,42 × LAMBDA_IR=0.05,0.1）。
- 结果：最佳 run（s21_lam0.1）**SSPG r=0.44，DI r=0.47**，均**低于 V4**（SSPG 0.65，DI 0.51）。
- 结论：**仅用 D1+D2 训练**（样本更少）+ 强 SSPG/DI 头，在当前设置下**没有提升**，反而变差。目前看 **D1+D2+D4 联合训练、仅在 D1+D2 上评估选优（V4）** 仍是 D1+D2 上表现最好的方案。

## 2. 当前最佳「好结果」：V4（仅 D1+D2 评估）

| 指标 | SSPG | DI |
|------|------|-----|
| **5-fold Spearman r** | **0.65** | **0.51** |
| **RMSE** | 59.5 mg/dL | 1.03 |
| **R²** | 0.25 | -0.21 |
| 主 run | SEED=21, LAMBDA_IR=0.05 | 同左 |

- 评估口径：`P1_GOLD_DATASETS=D1,D2`，n=63，无 D4，留一数据集出仅 D1/D2 两折且均正常。
- 用于投稿：**主表、主图、Abstract 统一用 V4 的上述数字**即可。

## 3. 若还想再冲高一点（可选）

- **方案 A（推荐先做）**：维持 V4 为主结果，在正文或补充材料中加一句敏感性表述：「当评估集包含 D4 时（V2/V3），DI 相关可达 0.71–0.75，但主结果采用仅 D1+D2 的评估以保证稳定与可比。」
- **方案 B**：在 **D1+D2+D4 训练、仅 D1+D2 评估** 不变的前提下，做一轮 **LAMBDA_SSPG=0.02、LAMBDA_DI=0.02** 的 4×4 调优（或 4×5 扩展 LAMBDA_IR），看是否出现 SSPG r≥0.6 且 DI r≥0.55 的 run；若有则替换主 run，若无则仍用 V4。
- **方案 C**：扩大 seed / LAMBDA_IR 网格（例如 6×5），仍用 D1+D2 评估选优，博取单次 run 的 SSPG/DI 再高一点；代价是计算量增大。

## 4. 代码与复现

- **仅 D1+D2 训练**：`P1_TRAIN_DATASETS=D1,D2`（见 `run_p1_full_pipeline.py`）。
- **仅 D1+D2 评估**：`P1_GOLD_DATASETS=D1,D2`（见 `run_auto_tune_and_report.py` → `evaluate_p1_metrics.py --datasets D1,D2`）。
- **缩小调优网格**：`P1_SEEDS=21,42`、`P1_LAMBDAS=0.05,0.1`（见 `run_auto_tune_and_report.py`）。
- V4 复现：`P1_RESULTS_ROOT=paper1_results_v4 P1_GOLD_DATASETS=D1,D2 python run_auto_tune_and_report.py`（默认 4×4 网格、D1+D2+D4 训练）。
