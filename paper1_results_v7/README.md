# V7 详细实验方案（修正版）结果

本目录存放基于《V7_详细实验方案（修正版）.pdf》的完整实验结果。**评价标准统一为 Pearson r**，评估为 **LODO-CV**（留一数据集出）。

## 方案要点

- **框架**：V6 Route B 的 26D 全潜在空间 + e2e_head。
- **目标 A**：突破 DI 预测瓶颈（力争 DI 的 r ≥ 0.5）。
- **目标 B**：临床四象限分层与（概念上）动态追踪。
- **评估**：LODO-CV，指标为 **Pearson r**（不再混用 R²/Spearman）。

## 四条路线与结果位置

| 路线 | 内容 | 结果目录/文件 |
|------|------|----------------|
| **Route 1** | λ 扫描：1A 仅 SSPG、1B 仅 DI，LODO-CV 选最优 λ | `route1/`，`route1/route1_lambda_scan_table.txt` |
| **Route 2** | 两阶段：26D + sspg_hat → Ridge → DI，LODO | `route2/route2_lodo_pearson.json` |
| **Route 3** | 多目标组合 A：lambda_sspg_best + lambda_di_best 联合训练 | `route3/combo_a/`，`route3/route3_combo_a_pearson.json` |
| **Route 4** | 四象限图 + 象限分类准确率 + 动态追踪说明 | `route4/route4_quadrant_report.txt`，`route4/v7_quadrant_plot.png` |

## 运行命令（项目根目录）

```bash
# Route 1：λ 扫描（1A + 1B，约 14 次 pipeline）
python scripts/run_v7_route1.py --exp all
# 仅对已有结果做 LODO 汇总
python scripts/run_v7_route1.py --exp all --skip_train

# Route 2：两阶段（需先有 1A best 的 26D CSV 且含 sspg_hat）
python scripts/run_v7_route2.py

# Route 3：组合 A（best λ 联合训练）
python scripts/run_v7_route3.py

# Route 4：四象限图与报告
python scripts/run_v7_route4.py --csv paper1_results_v7/route3/combo_a/latent_and_gold_all_26d.csv
```

Pipeline 支持命令行覆盖 λ：
```bash
python run_p1_full_pipeline.py --lambda_sspg 0.1 --lambda_di 0.05
```

## 当前结果摘要

- **Route 1**：lambda_sspg_best = **0.1**（LODO-CV Pearson r(SSPG) ≈ 0.55）；lambda_di_best = **0.05**（LODO-CV Pearson r(DI) ≈ **0.52**）。
- **Route 2**：X = [26D, sspg_hat] → DI，LODO-CV Pearson r(DI) ≈ 0.32（低于 Route 1B 端到端）。
- **Route 3 组合 A**：联合 λ_sspg=0.1、λ_di=0.05 训练，本次运行 LODO r(DI) 偏低（可能受联合训练/模式边界影响）；最佳 DI 仍来自 Route 1B。
- **Route 4**：四象限以真实 SSPG/DI 中位数分割；象限分类准确率 ≈ 41%；图见 `route4/v7_quadrant_plot.png`。

## 文件说明

- `route1/1a_lambda_sspg_*`：仅 SSPG 监督（lambda_di=0）各 λ 的 pipeline 输出。
- `route1/1b_lambda_di_*`：仅 DI 监督（lambda_sspg=0）各 λ 的 pipeline 输出。
- `route1/route1_lambda_scan_table.txt`：1A/1B 的 λ 与 LODO-CV Pearson r 汇总表。
- `route1/route1_lambda_scan.json`：同上，机器可读。
- `route2/route2_lodo_pearson.json`：两阶段 DI 的 LODO 各 fold 及 mean Pearson r。
- `route3/combo_a/`：组合 A 的 checkpoint 与 26D CSV。
- `route4/route4_quadrant_report.txt`：四象限说明与准确率；`v7_quadrant_plot.png`：SSPG_hat vs DI_hat 散点图。

## 说明

- **组合 B**（+ HOMA-IR / HOMA-B）需在 pipeline 中增加 LAMBDA_HOMA_IR / LAMBDA_HOMA_B 及对应头，当前仅实现组合 A。
- 26D CSV 自某次 pipeline 更新起已包含 `sspg_hat`、`di_hat` 列，供 Route 2 与 Route 4 使用；旧跑次需重跑对应 pipeline 才能得到这两列。
