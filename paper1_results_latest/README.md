# paper1_results_latest — 本次运行结果汇总

本目录为一次完整运行产出，所有结果已更新至此处。

## 目录结构

| 路径 | 说明 |
|------|------|
| **根目录** | M0（无监督/弱监督 VAE）主结果：模型、latent CSV、相关表、M0 端到端评估 |
| `m1/` | M1（端到端监督 VAE）**调参后**：λ=0.1, BETA_HAT=0.05, P1_ZSCORE_TARGETS=1；含 **m1/figures/**（B1 可视化） |
| `m1_a1/`～`m1_a4/` | 路线 A 调参实验 A1（只SSPG）、A2（只DI）、A3（偏SSPG）、A4（偏DI）；每目录含 5 折评估结果 |
| `m2/` | M2（黑盒 VAE）模型与 e2e head 评估 |
| `m3/` | M3（DirectNN）e2e head 评估 |
| `figures/` | M0 图表（si vs SSPG、mi vs DI、Bland-Altman、6D  pairwise/PCA/平行坐标等） |

## 主要文件

- **根目录**  
  - `autoencoder_p1_full.pt`：M0 机制 VAE + 头  
  - `latent_and_gold_all.csv` / `latent_and_gold_test.csv`：6D latent + 金标准（M0）  
  - `correlations.txt`：潜变量与金标准 Spearman 相关  
  - `e2e_head_metrics.json`：M0 测试集端到端 SSPG/DI 指标  
  - `evaluation_metrics.csv`、`evaluation_metrics_summary.txt`：M0 post-hoc Ridge 5-fold / LODO  
  - `joint_weights_*.csv`：联合 Ridge 权重  

- **m1/**  
  - `autoencoder_p1_full.pt`：M1 机制 VAE（含 sspg/di z-score mean/std）  
  - `e2e_head_metrics.json`：M1 单次划分端到端 SSPG/DI  
  - `evaluation_metrics_summary.txt`、`evaluation_metrics.csv`：**与 M0 同口径 5 折 Ridge 评估**（可比）  

- **m2/**  
  - `m2_blackbox.pt`：M2 黑盒 VAE + 头  
  - `e2e_head_metrics.json`：M2 测试集端到端 SSPG/DI  

- **m3/**  
  - `e2e_head_metrics.json`：M3 测试集端到端 SSPG/DI  

- **figures/**  
  - `p1_si_vs_sspg.png`、`p1_mi_vs_di.png`、`p1_blandaltman_si_sspg.png`  
  - `p1_6d_pairwise_by_dataset.png`、`p1_6d_pca2d.png`、`p1_6d_parallel_by_dataset.png`、`p1_6d_boxplot_by_dataset.png`  
  - `p1_correlations_summary.txt`  

## 端到端 Head 测试集摘要（Spearman r）

| 模型 | SSPG (r) | DI (r) | 备注 |
|------|----------|--------|------|
| M0（本目录 e2e_head_metrics.json） | 0.39 | 0.44 | 无监督基线 |
| M1（m1/，调参后） | **0.37** (单次) / **0.45** (5折) | **0.25** (单次) / **0.48** (5折) | λ=0.1, β=0.05, z-score；5 折见 m1/evaluation_metrics_summary.txt |
| M2（m2/e2e_head_metrics.json） | -0.15 | 0.45 | 黑盒消融 |
| M3（m3/e2e_head_metrics.json） | nan* | nan* | 预测接近常数 |

\* M3 测试集预测接近常数，Spearman 未定义。

## 复现命令

```bash
export CGM_PROJECT_OUTPUT=/path/to/cgm_project/output

# M0
P1_RESULTS_DIR=paper1_results_latest LAMBDA_SSPG=0 LAMBDA_DI=0 python run_p1_full_pipeline.py

# M1（推荐调参，防模式崩溃）
P1_RESULTS_DIR=paper1_results_latest/m1 LAMBDA_SSPG=0.1 LAMBDA_DI=0.1 BETA_HAT=0.05 P1_ZSCORE_TARGETS=1 python run_p1_full_pipeline.py

# M2
P1_RESULTS_DIR=paper1_results_latest/m2 LAMBDA_SSPG=1.0 LAMBDA_DI=1.0 python run_p1_m2_blackbox.py

# M3
P1_RESULTS_DIR=paper1_results_latest/m3 LAMBDA_SSPG=1.0 LAMBDA_DI=1.0 python run_p1_m3_direct.py

# M0 post-hoc 评估与出图
python scripts/evaluate_p1_metrics.py --csv paper1_results_latest/latent_and_gold_all.csv --out paper1_results_latest
python scripts/plot_p1_results.py --csv paper1_results_latest/latent_and_gold_all.csv --out paper1_results_latest/figures --correlations paper1_results_latest/correlations.txt
python scripts/plot_p1_6d_latent.py --csv paper1_results_latest/latent_and_gold_all.csv --out paper1_results_latest/figures

# M1 与 M0 同口径 5 折评估（训练完成后执行）
python scripts/evaluate_p1_metrics.py --csv paper1_results_latest/m1/latent_and_gold_all.csv --out paper1_results_latest/m1
```

- **P1_5FOLD_COMPARISON.md**（根目录）：M0 vs M1 五折对比、latent 边界检查结论与后续调参建议。
- **P1_A1_A4_SUMMARY.md**（根目录）：路线 A 实验 A1–A4 汇总表（λ 配置与 5 折 SSPG/DI r）、简要解读与复现命令。
- **m1/figures/**（B1）：M1 的 si vs SSPG、mi vs DI 等图，可与根目录 `figures/`（M0）对比。

生成时间：由 `run_p1_*` 与上述脚本一次完整运行得到。
