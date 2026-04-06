# P1 运行说明与结果解读

## 1. 是否真的“全跑完了”？为什么这么快？

**是的，是完整跑完的。** 之前说的“示例”是指“给你看的是一轮真实跑出来的结果示例”，不是假数据或只跑了一部分。

- **run_p1_full_pipeline.py**：在你本机执行过，流程是：加载 D1+D2+D4 → 合并 332 条样本、127 人 → 80/10/10 划分 → **训练 80 个 epoch** → 在 test 上算 latent–金标准相关 → 全量 encode 得到 `latent_and_gold_all.csv`。终端里会看到每 20 epoch 的 train/val loss 和最后的相关系数，以及 “Saved paper1_results/autoencoder_p1_full.pt …”。
- **速度**：约 **1–2 分钟** 是合理的。模型规模小（~4 万参数）、样本量 332、batch_size=32、80 epoch，在 Mac MPS 或 CPU 上几分钟内跑完很常见。若你换成更大数据或更大模型，时间会明显变长。

所以：**你看到的数字（如 mi–SSPG r≈-0.41、Ridge DI r≈0.53）都来自这些真实运行**，不是“示例数据”。

---

## 2. SSPG/DI 是“联合”的：单参数 vs 多参数

生理上，**胰岛素抵抗（IR）和 β 细胞功能（DI）是联合决定的**：SSPG 反映外周抵抗，DI 反映处置能力，两者都受多机制影响。所以：

- **单参数相关**（如 mi vs SSPG、si vs DI）只能说明“某一个 latent 维度”和金标准的关系，可能低估模型能力。
- **多参数联合**（用 6 维 latent 一起预测 SSPG/DI）才更符合“联合”的设定。我们已经用 **Ridge(6D → SSPG)** 和 **Ridge(6D → DI)** 做了这件事，结果上：
  - 单看 mi→DI 几乎无相关（r≈-0.13），但 **6D→DI 的 Pearson r≈0.54**，说明确实是**多个参数联合**才把 DI 预测出来。
  - SSPG 上 mi 单维就有一定相关（r≈-0.41），6D 联合后 CV 的 r 约 0.40，说明 SSPG 信息较多集中在 mi 上，但其他维度也有贡献。

因此：**评估“是否捕捉到 IR/β”时，既要看单参数相关（可解释性），也要看多参数联合预测（Ridge 的 RMSE/r 和系数）**。脚本里会输出：
- **单参数**：direct_mi、direct_si 对 SSPG/DI 的 r、RMSE、R²；
- **联合**：ridge_6d 对 SSPG/DI 的 r、RMSE、R²，以及 **Ridge 系数**（哪个 latent 对 SSPG/DI 的预测权重更大），便于讨论“联合”关系。

---

## 3. 模型层面：是否存在“多个参数联合相关”？

**存在，而且我们已经用 Ridge 在量化。** 具体可以这样理解：

- **单相关**：逐一对 (si, mi, tau_m, …) 和 SSPG/DI 算 Spearman，得到“谁单独最相关”。
- **联合相关**：用线性组合 \( w_1 \cdot \text{si} + w_2 \cdot \text{mi} + \cdots \) 去拟合 SSPG（或 DI），Ridge 学到的 \( w \) 就是“联合方向”；预测的 Pearson r 和 RMSE 就是“联合相关”的强度。系数表（见 `joint_weights_*.csv`）可以看到：例如 SSPG 主要由 mi、tau_m 等共同解释，DI 由 si、mi 等共同解释。

若你希望更“模型层面”的表述，可以写：**“SSPG/DI 与 latent 的关联是联合的（multi-parameter）；单潜变量相关仅部分反映这一点，而 6D Ridge 预测及系数更完整地刻画了这种联合关系。”**
