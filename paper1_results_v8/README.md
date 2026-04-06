# V8 实验方案：双模型专家系统

本目录存放 V8「双模型专家系统」的完整结果：**SSPG 专家**与 **DI 专家**分别训练，再在 Route 3 中整合两者的 latent-space，经 LODO-CV Ridge 得到最终预测并做四象限分析。

## 设计思路

- 避免 V7 中 SSPG 与 DI 监督在同一模型内竞争。
- Route 1：仅 SSPG 监督（lambda_sspg=0.1, lambda_di=0）→ SSPG 专家。
- Route 2：仅 DI 监督（lambda_sspg=0, lambda_di=1.0）→ DI 专家。
- Route 3：无训练；用两个专家的 26D latent 分别做 LODO-CV Ridge 预测 SSPG/DI，合并结果后评估并出图。

## 目录与产出

| 路线 | 内容 | 主要文件 |
|------|------|----------|
| **route1** | SSPG 专家模型 | `sspg_specialist_model.pt`, `sspg_specialist_latents.csv` |
| **route2** | DI 专家模型 | `di_specialist_model.pt`, `di_specialist_latents.csv` |
| **route3** | 综合分析 | `v8_final_results.csv`, `v8_quadrant_plot.png`, `v8_metrics.txt` |

## 运行方式

**Route 1（SSPG 专家）**：项目根目录执行  
`P1_HEAD_USE_26D=1 P1_SAVE_26D_LATENT=1 P1_ONE_MEAL_PER_SUBJECT=1 P1_RESULTS_DIR=paper1_results_v8/route1 python run_p1_full_pipeline.py --lambda_sspg 0.1 --lambda_di 0`  
再将 `route1/autoencoder_p1_full.pt` 与 `route1/latent_and_gold_all_26d.csv` 复制为 `sspg_specialist_model.pt` 与 `sspg_specialist_latents.csv`。

**Route 2（DI 专家）**：同上，`P1_RESULTS_DIR=paper1_results_v8/route2`，`--lambda_sspg 0 --lambda_di 1.0`，复制为 `di_specialist_*`。

**Route 3**：`cd paper1_results_v8 && python run_v8_analysis.py`

## 当前结果摘要

- Route 3 整合后 LODO-CV Pearson r：SSPG ≈ 0.46 (p=0.0002)，DI ≈ 0.27 (p=0.03)。
- 产出：`route3/v8_final_results.csv`（63 样本），`route3/v8_quadrant_plot.png`，`route3/v8_metrics.txt`。
