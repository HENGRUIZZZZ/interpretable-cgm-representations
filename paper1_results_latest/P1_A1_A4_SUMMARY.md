# P1 路线 A：系统性调参实验 A1–A4 汇总

所有实验均使用 **BETA_HAT=0.05、P1_ZSCORE_TARGETS=1**，与 M1 当前设置一致。评估统一为 `scripts/evaluate_p1_metrics.py` 的 **5-fold (by subject) Ridge**，指标可比。

## 汇总表（5 折 Spearman r）

| 实验 | λ_sspg | λ_di | SSPG r (5折) | DI r (5折) |
|------|--------|------|--------------|------------|
| **M0 (基线)** | 0 | 0 | 0.533 ± 0.215 | 0.347 ± 0.396 |
| **M1 当前** | 0.1 | 0.1 | 0.454 ± 0.321 | 0.483 ± 0.133 |
| **A1 (只 SSPG)** | 0.1 | 0 | 0.483 ± 0.261 | 0.491 ± 0.118 |
| **A2 (只 DI)** | 0 | 0.1 | 0.384 ± 0.308 | 0.471 ± 0.105 |
| **A3 (偏 SSPG)** | 0.2 | 0.05 | 0.396 ± 0.324 | 0.395 ± 0.154 |
| **A4 (偏 DI)** | 0.05 | 0.15 | 0.476 ± 0.239 | **0.566 ± 0.136** |

## 简要解读

- **A1（只监督 SSPG）**：SSPG r=0.483，仍低于 M0 的 0.533，未回到 0.53+，**竞争假说** 需结合 A2 一起看；DI 在无监督下仍有 0.491（latent 含 DI 信息）。
- **A2（只监督 DI）**：SSPG r=0.384 明显低于 M0；DI r=0.471，与 M1 当前接近。说明单独监督 DI 时，SSPG 的 latent 表达被削弱。
- **A3（偏 SSPG）**：λ_sspg=0.2、λ_di=0.05 时 SSPG r=0.396、DI r=0.395，双目标均未提升。
- **A4（偏 DI）**：λ_sspg=0.05、λ_di=0.15 时 **DI r=0.566** 为各实验最高，SSPG r=0.476 与 M1 当前接近；**偏 DI 的 λ 配置在 DI 上收益最大**。

## 复现命令

```bash
export CGM_PROJECT_OUTPUT=/path/to/cgm_project/output

# A1
P1_RESULTS_DIR=paper1_results_latest/m1_a1 LAMBDA_SSPG=0.1 LAMBDA_DI=0.0 BETA_HAT=0.05 P1_ZSCORE_TARGETS=1 python run_p1_full_pipeline.py
python scripts/evaluate_p1_metrics.py --csv paper1_results_latest/m1_a1/latent_and_gold_all.csv --out paper1_results_latest/m1_a1

# A2
P1_RESULTS_DIR=paper1_results_latest/m1_a2 LAMBDA_SSPG=0.0 LAMBDA_DI=0.1 BETA_HAT=0.05 P1_ZSCORE_TARGETS=1 python run_p1_full_pipeline.py
python scripts/evaluate_p1_metrics.py --csv paper1_results_latest/m1_a2/latent_and_gold_all.csv --out paper1_results_latest/m1_a2

# A3
P1_RESULTS_DIR=paper1_results_latest/m1_a3 LAMBDA_SSPG=0.2 LAMBDA_DI=0.05 BETA_HAT=0.05 P1_ZSCORE_TARGETS=1 python run_p1_full_pipeline.py
python scripts/evaluate_p1_metrics.py --csv paper1_results_latest/m1_a3/latent_and_gold_all.csv --out paper1_results_latest/m1_a3

# A4
P1_RESULTS_DIR=paper1_results_latest/m1_a4 LAMBDA_SSPG=0.05 LAMBDA_DI=0.15 BETA_HAT=0.05 P1_ZSCORE_TARGETS=1 python run_p1_full_pipeline.py
python scripts/evaluate_p1_metrics.py --csv paper1_results_latest/m1_a4/latent_and_gold_all.csv --out paper1_results_latest/m1_a4
```

## 路线 B：M1 可视化

- **B1**：M1 的 latent 图已生成在 `paper1_results_latest/m1/figures/`：
  - `p1_si_vs_sspg.png`、`p1_mi_vs_di.png`、`p1_blandaltman_si_sspg.png`、`p1_correlations_summary.txt`
- 与 M0 的 `paper1_results_latest/figures/` 中同名图对比，可直观看端到端监督是否让 latent 与金标准关系更紧。
