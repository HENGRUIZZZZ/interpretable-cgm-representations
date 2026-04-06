# V6 终极实验方案：全面探索 VAE+ODE 的信息提取能力

本目录存放 V6 方案下六条路线的全部结果。

## 六条路线概览

| 路线 | 描述 | 特征 | 结果位置 |
|------|------|------|----------|
| **A** | 仅 6D ODE 潜变量 | 6D (tau_m, Gb, sg, si, p2, mi)，subject 中位数 | `routeA_seed{seed}/` |
| **B** | 26D 端到端头（6+4+16） | 26D 全潜变量训练 e2e_head 预测 SSPG/DI | `routeB_seed{seed}/` |
| **C** | Route B 的 26D 做 PCA | PCA(n=2,3,4,5) + 100×5 折 Ridge | `routeC/route_c_pca_spearman.json` |
| **D** | CGM 统计基线 | cgm_mean + ac_var，100×5 折 Ridge | `routeD/route_d_spearman.json` |
| **E** | 冻结编码器只训头 | 加载 Route A 预训练，仅训练 e2e_head | `routeE_seed{seed}/` |
| **F** | 26D + 12D CGM 混合 | 38D 特征，100×5 折 Ridge | `routeF/route_f_spearman.json` |

## 运行命令（项目根目录）

```bash
# Route A（10 seeds）
python scripts/run_v6_routes_ab.py --route A

# Route B（10 seeds）
python scripts/run_v6_routes_ab.py --route B

# Route C（基于 routeB_seed42 的 26D latent）
python scripts/run_v6_route_c.py --n_cv 100

# Route D（CGM 2D 基线）
python scripts/run_v6_route_d.py --n_cv 100

# Route E（需先有 Route A；10 seeds）
python scripts/run_v6_route_e.py

# Route F（26D + 12D CGM）
python scripts/run_v6_route_f.py --n_cv 100

# 统一评估（A/B/C/D/F 的 100×5 折 Spearman + Wilcoxon；A/B 聚合 10 seeds）
python scripts/run_v6_unified_eval.py --output_dir paper1_results_v6/unified_eval --n_cv 100
# 若结果目录不在默认路径，可指定：--out_base /path/to/paper1_results_v6
```

## 主要输出文件

- **routeA_seed42**, **routeB_seed42**: `latent_and_gold_all.csv`（6D）, `latent_and_gold_all_26d.csv`（26D）, `autoencoder_p1_full.pt`, `e2e_head_metrics.json`
- **routeC**: `route_c_pca_spearman.json`（各 n_components 的 Spearman r 与方差解释率）
- **routeD**: `route_d_spearman.json`（SSPG/DI 的 100 次 Spearman r）
- **routeE_seed{seed}**: 同 Route B 目录结构
- **routeF**: `route_f_spearman.json`（38D 的 100 次 Spearman r）
- **unified_eval**: `v6_unified_eval_report.txt`（Wilcoxon 两两比较）, `v6_route_spearman.json`

## 当前已运行结果摘要（全量跑完）

- **Route A**: 10 seeds 已跑满，结果在 `routeA_seed42` … `routeA_seed900`
- **Route B**: 10 seeds 已跑满，结果在 `routeB_seed42` … `routeB_seed900`
- **Route C**: PCA(n=2,3,4,5) 已跑，`routeC/route_c_pca_spearman.json`
- **Route D**: 100×5 折已跑，DI median ≈ 0.37
- **Route E**: 10 seeds 已跑（其中 seed 200/800 训练中曾报 grad 错误，目录已生成）
- **Route F**: 38D 已跑，DI median ≈ 0.33
- **Unified Eval**: A/B 聚合 10 seeds 中位数特征后 100×5 折；含 A/B/C/D/F，Wilcoxon 两两比较见 `unified_eval/v6_unified_eval_report.txt`
