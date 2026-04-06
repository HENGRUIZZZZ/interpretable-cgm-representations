# Paper 1 数据与实验设计：五个数据集 D1–D5

本文档说明 `cgm_project.tar.gz` 或 **cgm_project_v2_final.tar.gz (V2)** 解压后的**五个数据集 D1–D5** 的命名、层级、表结构，以及 **Paper 1 的完整实验设计**（训练/验证/测试划分、Level 1/2/3 协议）。配置与划分逻辑见根目录 `paper1_experiment_config.py`。

**V2 数据**：预处理报告 (V2 已修复) 与 `cgm_project_v2_final.tar.gz` 为当前推荐数据。V2 中 D1 表结构有更新（cgm 为 `subject_id, timepoint_mins, glucose_mg_dl`，labels 列为 `SSPG, DI, HOMA_IR, HOMA_B`）；本仓库的 loader 已兼容 V2 格式。

---

## 1. 数据集命名与层级（D1–D5）

| 编号 | 数据源（原文件夹名） | 层级 | 类型 | 受试者 | CGM 行数 | 餐次数 | SSPG/DI | HOMA | 用药 |
|------|----------------------|------|------|--------|----------|--------|---------|------|------|
| **D1** | D1_metwally (Metwally 2024) | Level 1 | OGTT + 金标准 | 56 | 3,230 | 83 | ✓ | ✓ | ✗ |
| **D2** | D2_stanford (Stanford CGM DB) | Level 2 | 标准餐 + CGM | 74 | 23,520 | 412 | ✓ | ✓ | ✗ |
| **D3** | D3_cgmacros (CGMacros 2025) | Level 3 | 自由生活 + 饮食 | 45 | 138,168 | 1,706 | ✗ | ✗ | ✗ |
| **D4** | D3alt_hall (Hall 2018) | Level 3 | 自由生活 CGM | 57 | 105,426 | 2,118 | ✗ | ✗ | ✗ |
| **D5** | D4_shanghai (Shanghai T2DM 2023) | Level 3 | 自由生活 + 用药 | 100 | 112,462 | 3,808 | ✗ | 部分 | ✓ |

- **合并**：约 **332 名受试者**，**38 万+** CGM 读数。
- **路径**：解压后为 `cgm_project/output/D1_metwally`, `D2_stanford`, `D3_cgmacros`, `D3alt_hall`, `D4_shanghai`；在配置与文档中统一称为 **D1, D2, D3, D4, D5**。

---

## 2. 统一表结构

各数据集均提供（或部分提供）以下表：

- **subjects.csv**：`subject_id`, `dataset_id`, `original_id`, `age`, `sex`, `bmi`, `ethnicity`, `diagnosis` 等。
- **cgm.csv**：  
  - **D1, D2**：餐心格式 `subject_id`, `meal_id`, `mins_since_meal`, `glucose_mgdl`。  
  - **D3, D4, D5**：连续格式 `subject_id`, `timestamp`, `glucose_mgdl`。
- **meals.csv**：`subject_id`, `meal_id`, `timestamp`, `meal_type`, `carb_g`, `protein_g`, `fat_g`, `fiber_g`, `energy_kcal` 等。
- **labels.csv**：`subject_id` + 金标准（D1/D2 最全）：`sspg`, `di`, `homa_ir`, `homa_b`, `hba1c`, `fasting_glucose_mgdl` 等。
- **medications.csv**：仅 **D5** 有；`subject_id`, `drug_class`, `drug_name`, `route`。

---

## 3. Paper 1 实验设计总览

- **Level 1（机制对齐）**：在 **D1** 上训练 Hybrid VAE，按 **subject** 划分 train/val/test；在 test 上评估 latent 与 SSPG/DI/HOMA 的相关与消融。
- **Level 2（换协议泛化）**：用 D1 训好的 encoder 在 **D2** 上 **仅做 encode**，评估 latent 分布/结构与 D1 的一致性；可选 D2 参与联合训练或微调。
- **Level 3（多中心/真实世界）**：用同一 encoder 在 **D3, D4, D5** 上 encode（需先将连续 CGM 按餐切窗口），评估多数据集 latent 稳定性与跨 cohort 对齐。

---

## 4. 训练 / 验证 / 测试 划分

### 4.1 划分单位与比例

- **划分单位**：**subject**（同一人的所有 meal 窗口只出现在 train / val / test 之一）。
- **分层**：若 D1 有 `diagnosis`（如 Pre-D, T2D, Healthy），按 diagnosis **分层抽样**，保证各集合中诊断比例近似。
- **比例**（可在 `paper1_experiment_config.py` 中修改）：
  - **Train**：70%
  - **Val**：15%（用于早停、超参）
  - **Test**：15%（仅用于最终报告：latent–金标准相关、消融，不参与训练与早停）

### 4.2 使用方式

- **Train**：训练 Hybrid VAE（重建 + 可选 OGTT 锚定损失）。
- **Val**：监控重建损失或下游指标，早停；可选超参搜索。
- **Test**：固定模型后，在 test subject 上取 latent（如每 meal 的 latent 平均或中位数），与 `labels.csv` 的 SSPG/DI/HOMA_IR/HOMA_B 做相关、回归或分类；并做消融（无机制 decoder、无 OGTT 锚定等）。

### 4.3 随机种子

- 划分与训练共用同一 **SPLIT_SEED**（默认 21），保证可复现。

### 4.4 D1 金标准覆盖（含 V2）

- **V2 数据**：D1 有 59 人，其中 **32 人有** SSPG/DI/HOMA_IR/HOMA_B；V2 中 subject_id 统一，**batch 与 labels 对齐后 32 人同时有 CGM 与金标准**。70/15/15 划分后 test 约 9 人，其中约 3 人有金标准，可做 latent–金标准相关。
- **旧版/非 V2**：若 batch 受试者少于 labels 行数且 ID 不一致，则「在 batch 里且 labels 有金标准」可能仅几人；test 仅 1～2 人有金标准属当时数据与划分结果，非实现错误。
- **建议**：用 **V2 数据** 跑 Level 1；可选（1）全 D1 有金标准者算相关、（2）**D2/D4 有金标准者** 做外部验证（报告称 CGM∩SSPG 共 56 人：D1:5 公开 + D2:31 + D4:20）、（3）LOO/k-fold。诊断：`python scripts/check_d1_data.py`（需设置 `CGM_PROJECT_OUTPUT` 为解压后的 output 路径）。

---

## 5. Level 1 / 2 / 3 协议摘要

| 阶段 | 数据 | 操作 | 评估内容 |
|------|------|------|----------|
| **Level 1** | D1 train/val/test | 在 D1 上训练；test 上取 latent | latent 与 SSPG/DI/HOMA 相关、可预测性；消融 |
| **Level 2** | D2 | 用 D1 的 encoder 在 D2 上 encode（不更新参数） | latent 分布与 D1 对齐、结构一致性；若 D2 有金标准则算相关 |
| **Level 3** | D3, D4, D5 | 用 `load_cgm_project_level3` 切餐心窗口后，同一 encoder encode | 多 cohort latent 分布、主轴对齐、跨数据集稳定性 |

---

## 6. 与现有代码的对接

- **D1, D2**：餐心格式，可直接用 `load_cgm_project_data.load_cgm_project_level1_level2(data_dir=...)` 得到 `Batch`；再按 `paper1_experiment_config` 的划分在 **subject** 上做 train/val/test 拆分。
- **D3, D4, D5**：CGM 为连续时间序列；使用 `load_cgm_project_level3(dataset_id="D3"|"D4"|"D5", output_base=...)` 会在内存中按餐时间切窗口并重采样到与 D1/D2 相同的网格，返回相同 Batch 格式，可直接用 Level 1 的 encoder 做 encode。
- **Demographics**：无 `weight` 时用 `weight = bmi * 1.7^2` 近似；`Batch.demographics` 对应 `[gender, age, weight]`。

---

## 7. 快速使用（D1 划分 + 加载）

```python
from load_cgm_project_data import load_cgm_project_level1_level2
from paper1_experiment_config import get_data_dir, D1_TRAIN_FRAC, D1_VAL_FRAC, D1_TEST_FRAC, SPLIT_SEED

output_base = "/path/to/cgm_project/output"
data_dir = get_data_dir("D1", output_base)

batch, patient_info, labels_df = load_cgm_project_level1_level2(data_dir=data_dir, num_meals_threshold=1)

# 按 subject 划分 train/val/test（需根据 patient_info.patient_ids 与 labels 做分层划分）
# 示例：使用 sklearn train_test_split 两次，先分出 test，再在剩余中分出 val
from sklearn.model_selection import train_test_split
subject_ids = patient_info.patient_ids
# 若需分层，可传 stratify=diagnosis_per_subject
train_sid, test_sid = train_test_split(subject_ids, test_size=D1_TEST_FRAC, random_state=SPLIT_SEED)
train_sid, val_sid = train_test_split(train_sid, test_size=D1_VAL_FRAC / (1 - D1_TEST_FRAC), random_state=SPLIT_SEED)
# 再根据 train_sid, val_sid, test_sid 过滤 batch 中对应样本
```

D2 仅加载、不划分时：

```python
data_dir_d2 = get_data_dir("D2", output_base)
batch_d2, info_d2, labels_d2 = load_cgm_project_level1_level2(data_dir=data_dir_d2, num_meals_threshold=1)
# 用 D1 训好的 encoder 对 batch_d2 做 encode，再分析 latent
```

Level 3（D3/D4/D5）按餐切窗口后加载：

```python
batch_d3, info_d3, labels_d3 = load_cgm_project_level3(dataset_id="D3", output_base=output_base, min_cgm_points=10)
# batch_d3.cgm shape (N, 43, 1)，与 D1/D2 一致，可直接用同一 encoder encode
```

---

## 8. 注意事项

- **D2 cgm**：同一 `(meal_id, mins_since_meal)` 可能多行，加载时已按平均处理。
- **D1 金标准缺失**：部分 subject 的 SSPG/DI 等为空，Level 1 相关分析时可只保留有金标准的 subject，或在文中说明缺失。
- **Level 3**：已由 `load_cgm_project_level3` 在内存中按餐时间切窗口并重采样，返回与 D1/D2 相同的 (N, T, 1) Batch，无需额外脚本。

以上设计与 `paper1_experiment_config.py`、`load_cgm_project_data.py` 配合即可完成 Paper 1 从数据到划分到 Level 1/2/3 的完整实验流程。
