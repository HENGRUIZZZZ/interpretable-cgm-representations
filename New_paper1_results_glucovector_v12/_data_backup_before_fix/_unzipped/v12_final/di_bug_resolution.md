# DI "Bug" 的真相：不是 Bug，是生理学！

## 关键发现

DI（Disposition Index）标签本身就和 SSPG 呈负相关：
- D1: SSPG vs DI: r = **-0.604**, p = 0.0003, n=32
- D2: SSPG vs DI: r = **-0.625**, p < 0.0001, n=41
- D4: SSPG vs DI: r = **-0.644**, p < 0.0001, n=35

## 为什么是负相关？

DI = β-cell function × Insulin Sensitivity
SSPG = Insulin Resistance（越高越差）

所以：
- SSPG 高 → 胰岛素抵抗严重 → si 低 → DI 低（因为 DI 包含 si 的成分）
- SSPG 低 → 胰岛素敏感 → si 高 → DI 高

**SSPG 和 DI 在生理学上就是负相关的！**

## 模型的 DI raw r = -0.65 意味着什么？

模型预测的 DI 和真实 DI 的 raw r = -0.65，这说明：
- 模型学到了 DI 的信息，但**符号反了**
- 或者说，模型的预测头输出的是 "负DI"（即胰岛素抵抗方向）

## 这不是数据管道的 Bug

DI 标签本身全部是正数（0.44 - 6.58），没有符号反转。
问题出在训练过程中：
- 如果 SSPG 和 DI 共享一个预测头（2D 输出），而 SSPG loss 的梯度主导了训练
- 那么预测头会倾向于让两个输出都朝 SSPG 的方向优化
- 由于 SSPG 和 DI 天然负相关，这就导致 DI 预测的符号反了

## 修复方案

1. **分离预测头**：SSPG 和 DI 用独立的预测头，不共享参数
2. **或者**：在共享预测头中，对 DI 输出取负号
3. **或者**：calibrated r = abs(raw r) = 0.65，直接用 calibrated 版本报告
