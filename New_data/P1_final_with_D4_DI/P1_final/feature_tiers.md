# GlucoVector 特征层次组合

## Tier 0: 纯CGM（当前codebase默认）
- **输入**: CGM时间序列 (43个时间点) + 餐食信息 (6D: carb, sugar, fiber, fat, protein, total_grams)
- **Demographics**: gender, age, weight (3D) — 已在encoder中使用
- **输出特征**: ODE参数 6D (si, mi, tau_m, Gb, sg, p2) + 数据轨道 20D (z_init 4D + z_nonseq 16D) = 26D
- **预测头**: Ridge(26D → SSPG) 或 Ridge(26D → DI)

## Tier 1: CGM + 扩展Demographics（推荐的下一步）
- **在Tier 0基础上，预测头中拼接额外demographics特征**
- **额外特征**: BMI, HbA1c, FPG (空腹血糖), Fasting Insulin
- **预测头**: Ridge(26D + 4D_demo → SSPG)
- **消融组合**:
  - A: 仅Demo (4D) → Ridge → SSPG/DI
  - B: 仅CGM Latent (26D) → Ridge → SSPG/DI  [当前最佳]
  - C: CGM Latent + Demo (30D) → Ridge → SSPG/DI
  - D: 仅ODE参数 (6D) → Ridge → SSPG/DI
  - E: ODE参数 + Demo (10D) → Ridge → SSPG/DI

## Tier 2: CGM + Demographics + 血脂指标
- **在Tier 1基础上，进一步加入血脂指标**
- **额外特征**: Total Cholesterol, Triglycerides, HDL, LDL
- **可用性**: D1(部分), D2(无), D4(56人), D3_CGMacros(45人)
- **注意**: D2没有血脂数据，跨数据集使用受限

## Tier 3: CGM + 多模态（未来扩展）
- **在Tier 0基础上，加入心率、活动量等多模态信号**
- **数据源**: MultiSensor数据集 (24人, 14天, CGM+HR+HRV+Activity)
- **用途**: 验证可扩展性——加入多模态后数据轨道最佳维度是否下降

## 各数据集的特征可用性矩阵

| 特征 | D1 Metwally | D2 Stanford | D4 Hall | D3 CGMacros | MultiSensor |
|------|-------------|-------------|---------|-------------|-------------|
| CGM (餐后) | ✓ (OGTT) | ✓ (标准餐) | ✓ (标准餐) | ✗ | ✗ |
| CGM (自由生活) | ✗ | ✗ | ✓ | ✓ | ✓ |
| 餐食碳水 | ✓ (75g固定) | ✓ (已知) | ✓ (已知) | ✓ (详细) | ✓ (食物索引) |
| Age | ✓ (区间→中点) | ✓ | ✗ | ✓ | ✗ |
| Sex | ✓ | ✓ | ✗ | ✓ | ✗ |
| BMI | ✓ (区间→中点) | ✓ | ✗ | ✓ | ✗ |
| HbA1c | ✓ | ✓ | ✓ | ✓ | ✗ |
| FPG | ✓ | ✓ | ✓ | ✓ | ✗ |
| Fasting Insulin | ✓ | ✓ | ✓ | ✓ | ✗ |
| SSPG (金标准) | ✓ (32) | ✓ (43) | ✓ (42) | ✗ | ✗ |
| DI (IST-based) | ✓ (32) | ✓ (41) | ✗ | ✗ | ✗ |
| Oral DI (OGTT-based) | ✗ | ✗ | ✓ (49, 新计算) | ✗ | ✗ |
| HOMA-IR | ✓ | ✓ | ✓ | ✓ | ✗ |
| HOMA-B | ✓ | ✓ | ✓ | ✓ | ✗ |
| 血脂 | ✗ | ✗ | ✓ | ✓ | ✗ |
| HR/HRV/Activity | ✗ | ✗ | ✗ | ✗ | ✓ |
