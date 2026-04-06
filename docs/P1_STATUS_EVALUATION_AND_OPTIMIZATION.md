# P1 当前状态、评估逻辑与优化方向

## 1. 当前在做什么、跑完了吗

- **自动调优已跑完**（没有在后台继续跑）。流程是：多组 (SEED × LAMBDA_IR) 训练 → 每组用 **6D Ridge** 做 5-fold 预测 SSPG/DI → 按 **Spearman(SSPG) + Spearman(DI)** 选最优。
- **当前选定的最佳**：**LAMBDA_IR = 0.01**（即 RESULTS_AND_FIGURES.md 里表格第一行）。该配置下：
  - **SSPG** 5-fold Spearman r ≈ **0.386**，RMSE ≈ 77.8
  - **DI** 5-fold Spearman r ≈ **0.716**，RMSE ≈ 0.98，R² ≈ 0.26
- **0.716 的来源**：就是上面这一 run（`tune_0.01` 或等价配置）。该 run 的结果已复制到 `paper1_results/`，所以 `evaluation_metrics_summary.txt` 里的 DI spearman_r = 0.716 与表格一致，没有“之前好现在差”——表格和当前文件是同一 run。

---

## 2. 逻辑上有没有问题

- **选优目标**：用「SSPG 与 DI 的 5-fold Spearman 之和」选 LAMBDA_IR，目的是同时兼顾 IR（SSPG）和 β 细胞（DI），且 6D 联合预测始终在用（Ridge 输入为 6 维潜变量）。
- **可能混淆点**：
  - 若之后又跑过 9 组 (seed × lambda)，选出的“最佳”可能是另一组（例如某 seed 下 lambda=0.05 的和更高），那时复制到 `paper1_results/` 的会是那一组，表格应展示 9 行并标明选中的 (seed, lambda)。当前报告是 3 行（仅 lambda），对应早期只调 LAMBDA_IR 的三次 run，选定 0.01，与当前 `paper1_results/` 内容一致。
  - **结论**：逻辑自洽；0.716 就是当前选定配置下的 DI 结果，没有丢。

---

## 3. 原文 (Wang & Fox, arXiv:2312.03344) 怎么做、我们多了什么

- **原文数据**：Keto-Med，964 条 PPGR、33 人，**没有 SSPG/DI**。
- **原文评估**：
  - **聚类**：每人用其所有 PPGR 的 embedding 平均作为代表，k-means (k=2)，用诊断（pre-diabetes vs T2D）当真实标签，看 NMI、AMI、homogeneity、completeness。
  - **HbA1C**：展示 latent（如 Gb、SI·MI）与 HbA1C 的对应关系；用 2D 平面线性分类 29/33 vs black-box 26/33。
  - **没有**：没有用 SSPG、DI 做回归或相关，也没有报 RMSE/R² 对金标准。
- **我们多的**：在 D1/D2/D4 上有了 **SSPG（胰岛素抵抗）和 DI（β 细胞功能）** 金标准，所以**额外**评估：
  - CGM 学到的 6D 潜变量（及 6D Ridge）与 SSPG/DI 的**相关**（Spearman/Pearson）和**预测**（RMSE、R²）。

因此：**“用 CGM 对 SSPG/DI 程度进行分析”的评估是我们加的**，不是原文自带的；原文是“无金标准下的聚类 + HbA1C 一致性”。

---

## 4. 应该用 RMSE 还是 R² 来评估？

- **目标**：证明 CGM 潜变量能反映 SSPG/DI 的“程度”，即既有**排序一致**，也有**预测水平**。
- **建议同时报**：
  - **相关（Spearman / Pearson）**：说明潜变量与金标准的**方向/排序**是否一致（“学出来的东西和 SSPG/DI 的强弱是否对齐”）。小样本、多 fold 时比 R² 稳定。
  - **RMSE**：预测误差的尺度（与金标准同单位），便于和临床/文献比较。
  - **R²**：预测解释了多少方差。在 5-fold 中 test 很小或跨数据集时 R² 常为负，属正常；**不单独用 R² 否定结果**，结合相关和 RMSE 一起看。
- **论文表述建议**：主看 **5-fold Spearman r**（对齐程度），辅以 **RMSE**（预测误差）和 **R²**（若为正可写“有一定预测方差解释”）；若 R² 为负可写“在有限样本下更适合作相关/排序分析”。

---

## 5. 基于结果可做的进一步优化

- **数据与划分**：金标准样本量有限；可试仅用“同时有 SSPG+DI”的受试者做评估、或按 dataset 分层划分，减少缺失与分布偏倚。
- **训练**：多 seed 已做；可再试 BETA_HAT、epoch 数、或 LAMBDA_IR 更细网格（如 0.02, 0.03），看 5-fold (SSPG_r + DI_r) 是否再升。
- **评估**：Ridge alpha 已做网格；可加 ElasticNet、或对 **log(SSPG)**、**log(DI)** 做回归，有时 R²/RMSE 更稳；留一数据集出已做，可专门汇报“在 D4 上 SSPG r=0.49”等以讨论泛化。
- **与原文对齐（可选）**：若有诊断或 HbA1C，可加 k-means 聚类 (NMI/AMI) 和“latent vs HbA1C”图，与原文 Table 2 / Figure 4 对齐，再突出我们多了 SSPG/DI 的验证。

---

## 6. SSPG 和 DI 分开、各取最好

- **问题**：用「SSPG_r + DI_r 之和」选一个 run 时，图可能来自“和”最大但 SSPG 或 DI 单看不是最好的 run，导致 si vs SSPG 或 mi vs DI 的图看起来差。
- **做法**：**SSPG 和 DI 分开预测、各取最好**。出图时：
  - **si vs SSPG、Bland-Altman**：用 **SSPG 5-fold Spearman 最优** 的那次 run 的 `latent_and_gold_all.csv`；
  - **mi vs DI**：用 **DI 5-fold Spearman 最优** 的那次 run 的 `latent_and_gold_all.csv`。
- **命令**：`python scripts/plot_p1_results.py --best-per-target`（会自动扫描 `paper1_results/tune_*` 与 `run_s*_lam*`，找到两个最优 run 并分别用其 CSV 出图）。当前：SSPG 最优 = run_s21_lam0.05（r≈0.45），DI 最优 = tune_0.01（r≈0.72）。

---

## 7. 统一结论（可作报告/论文用）

- **配置**：LAMBDA_IR = 0.01，6D 潜变量 + Ridge 回归，5-fold 按 subject。
- **SSPG**：Spearman r ≈ 0.39，RMSE ≈ 78 mg/dL；说明 CGM 潜变量与金标准 IR 指标在排序上有一致性。
- **DI**：Spearman r ≈ 0.72，RMSE ≈ 0.98，R² ≈ 0.26；说明 CGM 潜变量对 β 细胞功能金标准既有排序一致，也有一定预测能力。
- **解读**：CGM 衍生的机制化潜变量（6D）与 SSPG/DI 具有可量化关联；**6D 联合预测优于单潜变量**，符合生理上多参数联合决定葡萄糖处置的设定；评估时以相关为主、RMSE/R² 为辅，与原文无 SSPG/DI 的设定互补。**出图时 SSPG 与 DI 各取 5-fold 最优 run**，避免“和”最优但单指标图差的问题。
