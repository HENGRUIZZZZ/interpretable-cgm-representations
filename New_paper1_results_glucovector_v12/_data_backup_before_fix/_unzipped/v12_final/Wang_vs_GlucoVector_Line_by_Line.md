# Wang vs GlucoVector 逐行级别精确对比

## 结论先行：我们学了几分像？

**答案：在模型架构层面，我们几乎 100% 复刻了 Wang。但在数据管道层面，存在一个关键的"名存实亡"问题。**

---

## 一、模型架构对比（models.py）

### Encoder 输入拼接

**Wang (wang_code/models.py, line 149-154):**
```python
def encode_dist(self, cgm, timestamps, meals, demographics):
    cgm_diff = torch.diff(cgm, prepend=..., dim=-2)  # N x T x 1
    meal_embeds = self.meal_embedding(meals)           # meals → Conv1d → embed
    demo_embeds = to_seq(self.demographics_embedding(demographics), like=cgm)  # demo → Linear → repeat T次

    encoder_input = torch.cat([cgm, cgm_diff, timestamps, meal_embeds, demo_embeds], dim=-1)
```

**我们 (glucovector_code/.../models.py, line 196-201):**
```python
def encode_dist(self, cgm, timestamps, meals, demographics):
    cgm_diff = torch.diff(cgm, prepend=..., dim=-2)  # N x T x 1
    meal_embeds = self.meal_embedding(meals)           # meals → Conv1d → embed
    demo_embeds = to_seq(self.demographics_embedding(demographics), like=cgm)  # demo → Linear → repeat T次

    encoder_input = torch.cat([cgm, cgm_diff, timestamps, meal_embeds, demo_embeds], dim=-1)
```

**结论：完全一样。** 模型架构层面，我们的 Encoder 输入拼接方式与 Wang 一模一样。

### Encoder 架构

| 组件 | Wang | 我们 | 一样？ |
|:---|:---|:---|:---|
| Encoder 类型 | Bi-LSTM | Bi-LSTM | **一样** |
| hidden_size | 32 | 32 | **一样** |
| num_layers | 2 | 2 | **一样** |
| meal_embedding | NanWrapper(Conv1d) | NanWrapper(Conv1d) | **一样** |
| demographics_embedding | NanWrapper(Linear) | NanWrapper(Linear) | **一样** |
| embedding_size | 8 | 8 | **一样** |
| encoder_dropout | 0.0 | 0.0 | **一样** |
| decoder_dropout | 0.5 | 0.5 | **一样** |

### ODE Decoder

| 组件 | Wang | 我们 | 差异 |
|:---|:---|:---|:---|
| ODE 模型 | Bergman Minimal Model | Bergman Minimal Model | **一样** |
| param_lims: tau_m | [10, 60] | [10, **120**] | 我们拓宽了上界 |
| param_lims: Gb | [80, 200] | [**60**, **250**] | 我们拓宽了 |
| param_lims: sg | [5e-3, 2e-2] | [5e-3, 2e-2] | **一样** |
| param_lims: si | [1e-4, 1e-3] | [1e-4, 1e-3] (或宽版 [1e-5, 1e-2]) | 可选拓宽 |
| param_lims: p2 | [1/60, 1/15] | [1/60, 1/15] | **一样** |
| param_lims: mi | [0.1, 3.0] | [0.1, 3.0] (或宽版 [0.05, 5.0]) | 可选拓宽 |
| state_lims | 完全一样 | 完全一样 | **一样** |
| dt | 5.0 | 5.0 | **一样** |
| max_carb_per_min | 1000 | 1000 | **一样** |
| seq_encoding (u') | 有，推断碳水吸收率 | **有**，完全一样 | **一样** |
| decoder_dropout on u' | 0.5 | 0.5 | **一样** |
| bw (体重常数) | 87.5 kg | 87.5 kg | **一样** |

### 我们的创新（Wang 没有的）

| 组件 | Wang | 我们 |
|:---|:---|:---|
| z_nonseq (16D 残差) | **没有** | 有，`nonseq_to_16` 从 LSTM cells 映射 |
| Prediction Head | **没有** | 有，4D z_init + 16D z_nonseq → 2D (SSPG, DI) |
| Identifiability Loss | **没有** | 有 |
| P1_FIX_SG_P2 选项 | **没有** | 有，可固定 sg/p2 |

### 训练超参数

| 参数 | Wang | 我们 | 一样？ |
|:---|:---|:---|:---|
| lr | 0.01 | 0.01 | **一样** |
| beta_hat (KL 权重) | 0.01 | 0.01 | **一样** |
| num_epochs | 100 | 60 | 我们少 40 epochs |
| batch_size | 64 | 32 | 我们更小 |
| optimizer | AdamW | AdamW | **一样** |

---

## 二、数据管道对比（data_utils.py / load_cgm_project_data.py）

### 这里是关键差异所在！

**Wang 的数据来源 (wang_code/data_utils.py):**
```python
MEAL_COVARIATES = [
    'total_grams', 'total_carb', 'total_sugar',
    'total_dietary_fiber', 'total_fat', 'total_protein', 
]
DEMOGRAPHICS_COVARIATES = ["gender", "age", "weight"]
```
Wang 的数据来自 Keto-Med 数据集，这个数据集有**详细的营养标签**（因为是受控饮食研究），所以这 6 个食物特征都是**真实的、精确的值**。Demographics 也是真实采集的。

**我们的数据来源 (load_cgm_project_data.py, line 174-229):**
```python
meal_covariates_src = {
    "total_grams": ["carb_g", "protein_g", "fat_g"],  # 三者之和
    "total_carb": ["carb_g"],
    "total_sugar": [],                                  # ← 空！没有来源！
    "total_dietary_fiber": ["fiber_g"],
    "total_fat": ["fat_g"],
    "total_protein": ["protein_g"],
}
```

### 逐项对比：6 个食物特征的真实情况

| 特征 | Wang 的数据 | 我们的数据来源 | 实际情况 |
|:---|:---|:---|:---|
| `total_grams` | 真实食物总重量 | `carb_g + protein_g + fat_g` | **不等价！** Wang 的是食物总重量（含水分等），我们的只是三大宏量之和 |
| `total_carb` | 真实碳水总量 | `carb_g` | **取决于 D1-D4 数据集是否有 carb_g** |
| `total_sugar` | 真实糖含量 | `[]` (空列表) | **完全缺失！** 代码里 `src=[]`，所以这个特征永远是 `np.nan → 0.0` |
| `total_dietary_fiber` | 真实膳食纤维 | `fiber_g` | **取决于数据集是否有 fiber_g** |
| `total_fat` | 真实脂肪含量 | `fat_g` | **取决于数据集是否有 fat_g** |
| `total_protein` | 真实蛋白质含量 | `protein_g` | **取决于数据集是否有 protein_g** |

### 关键问题：D1-D4 数据集里到底有没有这些字段？

我们的代码从 `meals.csv` 中读取 `carb_g`, `protein_g`, `fat_g`, `fiber_g`。但这些字段是否存在，完全取决于数据集的准备情况。

**最可能的情况是：**
- **D1 (Metwally OGTT)**：OGTT 是标准化的 75g 葡萄糖溶液，所以 `carb_g=75`，其他全是 0。这意味着 6 个食物特征中只有 `total_carb=75` 有值，其余全是 0 或 NaN。
- **D2 (Stanford Diet Study)**：可能有部分宏量营养素数据（因为是受控饮食研究）。
- **D3/D4 (连续 CGM + 自由饮食)**：取决于参与者是否记录了详细的食物成分。如果只记录了碳水估计值，那 fat_g, protein_g, fiber_g 全是 NaN → 0。

### Demographics 的情况

| 特征 | Wang 的数据 | 我们的数据来源 | 实际情况 |
|:---|:---|:---|:---|
| `gender` | 真实性别 | `subjects.csv` 中的 `sex` 字段 | **有**，但如果缺失默认 M=0 |
| `age` | 真实年龄 | `subjects.csv` 中的 `age` 字段 | **有**，但如果缺失默认 40.0 |
| `weight` | 真实体重 | `subjects.csv` 中的 `weight` 字段，缺失时用 `BMI × 1.7²` 估算 | **可能是估算值** |

---

## 三、总结：学了几分像？

### 模型架构层面：**10 分像（几乎完全复刻）**

我们的 `MechanisticAutoencoder` 与 Wang 的在以下方面完全一致：
- Encoder 结构（Bi-LSTM, 2层, hidden=32）
- 输入拼接方式（cgm + cgm_diff + timestamps + meal_embed + demo_embed）
- ODE Decoder（Bergman Minimal Model, 完全相同的动力学方程）
- seq_encoding（u'，推断碳水吸收率）
- 训练目标（ELBO = MSE + β·KL, β=0.01）

我们还加了 Wang 没有的创新：16D z_nonseq 残差、Prediction Head、Identifiability Loss。

### 数据管道层面：**可能只有 3-4 分像**

虽然代码框架支持 6 维食物特征 + 3 维 Demographics，但：

1. **`total_sugar` 永远是 0**：代码里 `src=[]`，这个特征从来没有被填充过。
2. **`total_grams` 的计算方式不对**：Wang 用的是食物总重量，我们用的是三大宏量之和。
3. **D1 的食物特征几乎无信息**：OGTT 只有 75g 碳水，所有人都一样，没有区分度。
4. **D3/D4 的食物特征很可能大面积缺失**：自由饮食场景下，参与者很少能准确记录脂肪、蛋白质、纤维的摄入量。
5. **Demographics 可能有大量默认值**：缺失时 age 默认 40，weight 用 BMI 估算。

**最致命的问题**：即使代码框架支持这些特征，如果数据集里这些字段全是 NaN 或常数，那 `NanWrapper` 会把它们全部 mask 掉，等于**模型从来没有真正"看到"过食物成分和人口统计学信息**。

---

## 四、需要立即确认的事情

1. **检查 D1-D4 的 meals.csv**：`carb_g`, `protein_g`, `fat_g`, `fiber_g` 这些列是否存在？有多少比例是非零/非NaN的？
2. **检查 D1-D4 的 subjects.csv**：`age`, `sex`, `weight`, `bmi` 这些列的填充率是多少？
3. **检查训练日志**：模型在训练时，`meals` tensor 和 `demographics` tensor 的实际值分布是什么样的？是不是大面积为 0？

如果确认这些特征确实大面积缺失，那我们的模型虽然在架构上完美复刻了 Wang，但在数据层面实际上是在"空转"——Encoder 接收到的食物和 Demographics 信息几乎为零，模型只能靠 CGM 曲线本身来推断一切。
