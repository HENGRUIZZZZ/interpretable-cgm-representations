# P1 训练方法说明、数据利用与优化方向

## 1. 当前训练属于什么

- **类型**：**半监督 / 多任务** 的混合 VAE（Hybrid VAE）。
  - **主任务（无监督）**：用 CGM + 餐食 + 人口学输入，经 **LSTM 编码器** 得到潜变量，再经 **机制化解码器**（Bergman ODE）重建 CGM。损失 = 重建 MSE + β·KL(近似后验 || 先验)。
  - **辅助任务（弱监督）**：仅对带 HOMA_IR 的样本，用潜变量 [si, mi, tau_m] 经 **IR 头** 预测 log(HOMA_IR+1)，损失 += λ_IR·MSE；**未**对 SSPG/DI 做训练时监督。
- **评估时**：用 6D 潜变量做 **Ridge 回归** 预测 SSPG/DI（5-fold 或留一数据集出），所以 SSPG/DI 只出现在**评估阶段**，不在训练损失里。

因此：训练是 **“重建 + 轻微 IR 弱监督”**；SSPG/DI 的 r、R²、RMSE 完全来自**事后**用潜变量拟合金标准。

---

## 2. 数据集有没有充分利用

- **划分**：按 **subject** 80% train / 10% val / 10% test。VAE 只在 80% 的人上训练；val 用于早停/监控，test 用于 pipeline 内的一次相关汇报。
- **潜变量**：训练完后对 **全体** 样本（含 val/test）做 encode，得到每人每餐的 6D latent，再与金标准表按 subject_id 合并 → `latent_and_gold_all.csv`。
- **5-fold 评估**：在 **所有有金标准的 subject** 上做 GroupKFold(5)，即 5-fold 的 Ridge 是在“全量金标准”上做的，只是按人划分 fold，所以 **金标准样本在评估阶段是 100% 使用的**。
- **未用上的**：VAE 训练只用 80% 的人；若希望“更多数据学表示”，可考虑 **用 100% 人训 VAE**（不设 test），仅用 5-fold 做 Ridge 评估（见下节优化）。

---

## 3. 训练方法能不能优化


| 方向                 | 说明                                         | 实现状态/建议                                 |
| ------------------ | ------------------------------------------ | --------------------------------------- |
| **IR 弱监督**         | λ_IR·MSE(log HOMA_IR) 已用；可调 λ_IR           | 已做；自动调优会扫 0.01–0.1                      |
| **SSPG/DI 辅助头**    | 训练时加小头：潜变量→SSPG、潜变量→DI，仅在有金标准样本上算 MSE，权重很小 | 可加 LAMBDA_SSPG、LAMBDA_DI（默认 0），逐步试 0.01 |
| **更多数据训 VAE**      | 用 100% 人训 VAE（无 holdout），仅 5-fold 做 Ridge  | 可加 P1_TRAIN_ON_FULL=1：不分 test，全量训       |
| **BETA_HAT / 训练量** | 调大 β 更贴近先验、调大 epoch 更收敛                    | 已支持 P1_NUM_EPOCHS；可扫 BETA_HAT           |
| **多 seed**         | 不同随机划分/初始化会得到不同 r                          | 已做多 seed 自动调优                           |


---

## 4. 自动调优在做什么

- 对 **SEED × LAMBDA_IR** 的每个组合：跑一遍 pipeline（训练 VAE+IR 头 → 全量 encode → 写 latent_and_gold_all），再跑评估（5-fold Ridge 等），得到该 run 的 SSPG 与 DI 的 5-fold Spearman r。
- **选优**：  
  - “主 run”：按 **SSPG_r + DI_r 之和** 最大选一个 run，复制到 `paper1_results/` 作为主结果。  
  - **出图**：SSPG 相关图用 **SSPG r 最大的 run**，DI 相关图用 **DI r 最大的 run**（各取最好）。
- 所以 **DI r≈0.72** 会出现在：要么是“主 run”恰好 DI 很高，要么是“best-DI run”的图与汇报里；最终报告会列出所有 run 的表格，并标明 best-by-sum、best-SSPG、best-DI 分别对应哪一 run。

