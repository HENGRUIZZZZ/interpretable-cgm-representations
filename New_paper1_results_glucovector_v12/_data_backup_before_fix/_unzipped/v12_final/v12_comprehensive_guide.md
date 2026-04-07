# GlucoVector v12 终极实验指南：回归蓝图

## 1. 核心目标与蓝图对齐

根据 `the_grand_storyline.md`，我们的 P0 论文（开山之作）的核心 Claim 是：
> **从非侵入性的 CGM 曲线中学习到的 26D 机制表征（GlucoVector），能够同时预测侵入性的临床金标准：胰岛素抵抗（SSPG）和 β 细胞功能（DI）。**

在 v11 之前，我们偏离了蓝图：
1. **数据缺失**：模型架构虽然能接收 6D 食物特征和 3D 人口统计学特征，但数据管道导致 D2 和 D4 的这些特征全为 0。模型在"蒙眼"推断。
2. **DI 预测异常**：DI 预测结果与真实值呈负相关（r ≈ -0.65）。经过生理学分析，这**不是 Bug**，而是因为 SSPG 和 DI 在生理上本身就是强负相关的。当它们共享一个预测头时，SSPG 的梯度主导了网络，导致 DI 的预测方向被带偏。

v12 的目标是：**使用完整修复的数据集，采用分离的预测头，在半监督（0.01 弱梯度）下同时训练 SSPG 和 DI。**

---

## 2. 模型架构 (GlucoVector v12)

### 2.1 输入层 (Input)
- **CGM 时序**：`cgm` (1D), `cgm_diff` (1D), `timestamp` (1D)
- **食物特征 (6D)**：`carb_g`, `sugar_g`, `fat_g`, `protein_g`, `fiber_g`, `grams`（通过 `Conv1d` 映射为时序 embedding）
- **人口统计学 (3D)**：`age`, `sex`, `weight`（通过 `Linear` 映射并扩展为时序 embedding）

### 2.2 编码器 (Encoder)
- **结构**：2 层 Bi-LSTM (hidden=32)
- **输出**：`seq_encoding` (用于推断碳水吸收率), `nonseq_encoding` (用于推断 ODE 参数和残差)

### 2.3 潜在空间 (Latent Space - 26D)
- **10D 机制骨架 (`z_ode`)**：
  - 6D 动力学参数：`tau_m`, `Gb`, `sg`, `si`, `p2`, `mi`
  - 4D 初始状态：`G`, `X`, `G1`, `G2`
- **16D 数据驱动血肉 (`z_nonseq`)**：捕捉 ODE 无法解释的复杂生理波动（如压力、睡眠、其他激素）。

### 2.4 解码器 (Decoder)
- **结构**：Bergman Minimal Model (ODE 积分器)
- **过程**：使用 10D `z_ode` 和推断出的碳水吸收率，通过 ODE 积分重建 CGM 曲线。

### 2.5 预测头 (Prediction Heads) - **v12 关键修改**
为了解决 SSPG 和 DI 互相干扰的问题，必须**分离预测头**：
- **SSPG Head**：输入 26D 全表征 $\rightarrow$ MLP $\rightarrow$ 预测 SSPG
- **DI Head**：输入 26D 全表征 $\rightarrow$ MLP $\rightarrow$ 预测 DI

---

## 3. 数据集修复说明

v12 必须使用修复后的数据集（已打包在 `v12_experiment_package.zip` 中）：

1. **D4 (Hall 2018)**：补充了 3 种标准餐（PB_sandwich, Protein_bar, Cornflakes）的精确 `carb_g`, `fat_g`, `protein_g`, `fiber_g`。
2. **D2 (Wu 2025)**：补充了全部 24 种标准餐的精确营养成分。
3. **D3 (CGMacros)**：修复了 `weight_kg` 的单位错误（原数据为磅），修复了 `fiber_g` 的异常极大值。

**操作**：将压缩包解压，覆盖 `P1_final/` 目录下的对应文件。

---

## 4. 训练配置与环境变量

为了实现上述架构，请使用以下环境变量运行 `run_p1_full_pipeline.py`：

```bash
# 1. 基础配置
export P1_NUM_EPOCHS=100
export P1_LR=1e-2
export BETA_HAT=0.01  # KL 散度权重

# 2. 预测头配置 (分离预测头)
export P1_V5_PREDICTION_HEAD=0  # 关闭共享的 2D 预测头
export P1_HEAD_USE_26D=0        # 关闭 e2e_head
export P1_DECOUPLE_SSPG=0       # SSPG head 输入全表征 (不只是 si)
export P1_DI_PRODUCT_CONSTRAINT=0 # 关闭 DI 乘积约束
export P1_DI_LOG_PRODUCT=0      # 关闭 DI 对数乘积
export P1_DI_MLP_HEAD=0         # 关闭仅输入 si,mi 的 DI head

# 3. 损失函数权重 (半监督，0.01 弱梯度)
export LAMBDA_SSPG=0.01
export LAMBDA_DI=0.01
export LAMBDA_IR=0.0
export LAMBDA_CLS=0.0
export LAMBDA_ORTHO=0.0

# 4. 目标标准化
export P1_ZSCORE_TARGETS=1      # 必须开启，平衡 SSPG 和 DI 的尺度差异

# 5. 参数边界 (宽边界)
export P1_WIDE_PARAM_RANGE=1    # 允许 tau_m, Gb 等参数有更大的探索空间
```

### ⚠️ 代码修改要求 (极其重要)

在运行前，必须修改 `run_p1_full_pipeline.py` 中关于分离预测头的输入维度。
找到 `sspg_head` 和 `di_head` 的定义（约 358 行），修改为接收 26D 输入：

```python
# 修改前：
sspg_head = torch.nn.Linear(1 if P1_DECOUPLE_SSPG else len(PARAM_NAMES), 1).to(device)
di_head = torch.nn.Linear(len(PARAM_NAMES), 1).to(device)

# 修改后 (让它们接收 26D 全表征)：
head_input_dim = len(PARAM_NAMES) + 4 + 16  # 6 + 4 + 16 = 26
sspg_head = torch.nn.Sequential(
    torch.nn.Linear(head_input_dim, 32),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(32, 1)
).to(device)

di_head = torch.nn.Sequential(
    torch.nn.Linear(head_input_dim, 32),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(32, 1)
).to(device)
```

同时，在 `loss` 计算部分（约 480 行），确保传入的是 26D 表征：

```python
# 修改前：
sspg_in = latent_all[:, 3:4] if P1_DECOUPLE_SSPG else latent_all

# 修改后：
# 获取 26D 表征
p26, init26, z16 = model.get_all_latents_for_head(cgm, timestamps, meals, demographics)
latent_26d = torch.cat([p26, init26, z16], dim=-1)

sspg_in = latent_26d
sspg_pred = sspg_head(sspg_in).squeeze(-1)

# DI 同理
di_in = latent_26d
di_pred = di_head(di_in).squeeze(-1)
```

---

## 5. v12 实验规划

为了在论文中讲出完整的故事，我们需要跑三个实验：

### Exp1: Wang Baseline (10D 纯无监督)
- **目的**：复现 Alexander Wang 的原始架构，作为性能基线。
- **配置**：关闭 16D 残差，关闭所有预测头（`LAMBDA_SSPG=0`, `LAMBDA_DI=0`）。
- **数据**：使用修复后的完整数据。

### Exp2: GlucoVector v12 (26D 半监督) - **主推模型**
- **目的**：展示我们的最终模型在 SSPG 和 DI 预测上的强大性能，同时保持 ODE 参数的可解释性。
- **配置**：按第 4 节的配置运行（26D，分离预测头，`LAMBDA_SSPG=0.01`, `LAMBDA_DI=0.01`）。
- **数据**：使用修复后的完整数据。

### Exp3: 数据消融 (26D 半监督，旧数据)
- **目的**：证明我们补充的 6D 食物特征和 3D 人口统计学特征是模型成功的关键。
- **配置**：与 Exp2 完全相同。
- **数据**：使用**修复前**的旧数据（D2/D4 食物特征全为 0）。
- **预期**：Exp3 的 ODE 参数（特别是 `tau_m` 和 `si`）的 SHAP 贡献率和方差将显著低于 Exp2。

---

## 6. 评估指标与预期结果

运行完成后，重点关注以下指标：

1. **预测性能 (D4 独立测试集)**：
   - SSPG Pearson r > 0.65
   - DI Pearson r > 0.60（注意：由于分离了预测头，DI 的 r 值应该变为**正数**）
2. **可解释性 (SHAP)**：
   - ODE 参数（前 6 维）的总 SHAP 贡献率应 > 25%。
   - `tau_m` 应该对 SSPG 有显著贡献。
3. **生理学验证**：
   - `tau_m` 与食物中的脂肪 (`fat_g`) 和蛋白质 (`protein_g`) 应该呈正相关（高脂肪延缓胃排空）。
   - `si` 应该在早晨和晚上表现出显著的昼夜节律差异。
