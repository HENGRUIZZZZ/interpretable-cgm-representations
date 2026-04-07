# GlucoVector 数据管道审计报告（最终版）

## 一、模型期望的输入 vs 实际收到的输入

### 1. 食物特征 (6D MEAL_COVARIATES)

代码定义（load_cgm_project_data.py L174-181）:
```python
meal_covariates_src = {
    "total_grams": ["carb_g", "protein_g", "fat_g"],  # 错误：应该是食物总重量
    "total_carb": ["carb_g"],
    "total_sugar": [],                                  # 永远为NaN→0
    "total_dietary_fiber": ["fiber_g"],
    "total_fat": ["fat_g"],
    "total_protein": ["protein_g"],
}
```

#### D1 (Metwally 2024, OGTT only)
- meals.csv 列: subject_id, meal_type, meal_context, carb_g, description, dataset
- 只有 carb_g=75 (OGTT)
- **模型收到**: [75+0+0=75, 75, 0, 0, 0, 0] → 常数，无区分度

#### D2 (Wu 2025 / Stanford CGMDB, 24种标准餐)
- meals.csv 列: subject_id, meal_type, n_readings, description, meal_category
- **完全没有** carb_g, protein_g, fat_g, fiber_g 列！
- **模型收到**: [0, 0, 0, 0, 0, 0] → 全零！
- **真实情况**: Wu 2025 Table S6 有完整的7种基础餐营养成分，全部标准化为50g碳水
- **修复方案**: 根据meal_type查表补充

#### D3 (CGMacros 2025, 自由饮食)
- meals.csv 列: subject_id, timestamp, meal_type, calories_kcal, carb_g, protein_g, fat_g, fiber_g, amount_consumed_pct
- **唯一正常的数据集**，5个营养素列全部有值
- **但有异常值**: CGMacros-016和019的fiber_g高达2830g（可能是mg单位错误）
- **模型收到**: 基本正确，但total_grams=carb+protein+fat（错误），total_sugar=0

#### D4 (Hall 2018, 3种标准餐+OGTT)
- meals.csv 列: subject_id, timestamp, meal_type, meal_code, description, repeat, meal_category, carb_g
- carb_g: 只有OGTT行=75，标准餐全部NaN
- **完全没有** protein_g, fat_g, fiber_g 列
- **模型收到**: OGTT=[75, 75, 0, 0, 0, 0]，标准餐=[0, 0, 0, 0, 0, 0]
- **真实情况**: Hall 2018 S5 Table 有完整的3种标准餐营养成分
- **修复方案**: 根据meal_type查表补充

### 2. Demographics (3D)

代码定义（L231-239）:
```python
gender = 1.0 if str(demo_row.get("sex", "M")).upper().startswith("F") else 0.0
age = float(pd.to_numeric(demo_row.get("age"), errors="coerce") or 40.0)
weight = float(pd.to_numeric(demo_row.get("weight"), errors="coerce") or (bmi * default_height_m**2))
```

| 数据集 | gender | age | weight | 问题 |
|:---|:---|:---|:---|:---|
| D1 | ✅ 95% | ✅ 95% | ✅ 93% (weight_kg列) | 基本OK |
| D2 | ✅ 82% | ✅ 93% | ❌ 0% (无weight列，只有bmi) | weight用BMI估算 |
| D3 | ✅ 100% | ✅ 100% | **❌ 单位错误!** weight_kg实际是磅(lbs) | 模型收到的weight是真实值的2.2倍 |
| D4 | **❌ 0%** (全NaN) | ✅ 100% | ✅ 82% | gender全部默认为0(Male) |

**D3 单位错误验证**: weight_kg mean=182.5 (lbs), 转换后=82.8 kg。BMI = (lbs/inches²)×703 = 存储的BMI值，完美匹配。

**D4 sex缺失**: Hall 2018论文说32F/25M，但subjects.csv中sex列57/57全是NaN。

## 二、数据质量问题汇总

| 问题 | 严重程度 | 影响范围 | 修复难度 |
|:---|:---|:---|:---|
| D2 meals.csv 缺少所有营养素列 | **致命** | 332个训练样本 | 低（查表补充） |
| D4 标准餐 carb_g=NaN | **致命** | 176个测试样本 | 低（查表补充） |
| D4 缺少 protein_g/fat_g/fiber_g 列 | **致命** | 240个样本 | 低（查表补充） |
| D4 sex 全部缺失 | **严重** | 57个受试者 | 中（需找原始数据） |
| D3 weight_kg 单位是磅不是公斤 | **严重** | 45个受试者 | 低（÷2.205） |
| D3 height_cm 单位是英寸不是厘米 | **严重** | 45个受试者 | 低（×2.54） |
| D3 fiber_g 异常值(最高2830) | 中等 | 24行/1706行 | 低（cap或排除） |
| total_sugar 永远为0 | 中等 | 全部 | 低（加映射） |
| total_grams 计算错误 | 低 | 全部 | 低（改映射） |
| D2 无weight列 | 低 | 74个受试者 | 低（BMI估算） |

## 三、标准餐营养成分查表

### D4 (来源: Hall 2018 PLOS Biology, S5 Table)

| meal_type | carb_g | fat_g | protein_g | fiber_g | sugar_g | calories |
|:---|:---|:---|:---|:---|:---|:---|
| PB_sandwich | 51 | 20 | 18 | 12 | 12 | 430 |
| Protein_bar | 48 | 18 | 9 | 6 | 19 | 370 |
| Cornflakes | 54 | 2.5 | 11 | 3.3 | 35.2 | 280 |
| OGTT_75g | 75 | 0 | 0 | 0 | 75 | 300 |

### D2 基础餐 (来源: Wu 2025 Nature Medicine, Supplementary Table 6)

| meal_type | carb_g | fat_g | protein_g | fiber_g | sugar_g | grams |
|:---|:---|:---|:---|:---|:---|:---|
| Pasta | 50.0 | 1.5 | 9.4 | 2.9 | 0.9 | 162.0 |
| Rice | 50.0 | 0.5 | 4.8 | 0.6 | 0.1 | 177.4 |
| Potatoes | 49.9 | 0.3 | 4.3 | 5.2 | 2.5 | 148.2 |
| Grapes | 50.0 | 0.4 | 2.0 | 2.5 | 42.8 | 276.2 |
| Beans | 50.0 | 1.2 | 15.8 | 20.2 | 0.7 | 191.9 |
| Berries | 50.0 | 1.9 | 4.0 | 16.2 | 28.2 | 471.6 |
| Bread | 49.9 | 9.6 | 9.0 | 2.7 | 5.7 | 109.2 |
| Glucose | 50.0 | 0 | 0 | 0 | 50.0 | 50.0 |

### D2 Mitigator组合餐 (基础餐 + 预加载)

| meal_type | 额外成分 | 额外量 |
|:---|:---|:---|
| X+Fat | cream | +15g fat |
| X+Fiber | pea fiber | +10g fiber |
| X+Protein | egg white | +10g protein |

Quinoa: 约50g carb, ~8g protein, ~3.5g fat, ~5g fiber (标准USDA数据)
