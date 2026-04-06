# P1 结果保留与完整重建说明

本文档说明：**哪些目录必须保留**、**如何在不重训的前提下把图表与报告全部更新到最新**、以及**如何从零做一次完整可复现流程**。便于后续任何人重建或复现结果。

---

## 1. 保留策略（请勿删除）

以下目录**全部保留**，方便后续对比与重建：

| 目录 | 用途 |
|------|------|
| `paper1_results/` | 默认/兜底结果根（可能被 copy_best 写入） |
| `paper1_results_v2/` | 某版超参/数据配置的完整 run（含 run_s*_lam*、evaluation、figures） |
| `paper1_results_v3/` | 同上 |
| `paper1_results_v4/` | **当前主用**：论文主结果与主图来源 |
| `paper1_results_v5/` | 可选（如仅 D1+D2 训练等） |
| `paper1_results_diagnostic/` | 诊断产出：全场景线性/非线性表与图、6D 副本、DIAGNOSTIC_REPORT.md |

每个 version 下通常包含：
- `run_s{seed}_lam{lam}/`：单次 pipeline 产出（模型 checkpoint、`latent_and_gold_all.csv`、若新跑则有 `training_curves.json`、`reconstruction_*.npy/npz`）
- `figures/`：该 version 的 SI/MI 图、6D 图、汇总图等
- `evaluation_metrics.csv`、`evaluation_metrics_summary.txt`、`FINAL_REPORT.md`、`RESULTS_AND_FIGURES.md`

**不要**删除上述任一 version 或 `paper1_results_diagnostic`，以便：
- 随时用 `--report-only` 或诊断脚本**只重画图、不重训**；
- 对比不同 version 或不同 seed/λ 的结果。

---

## 2. 保证「全部最新」：只更新图与报告（不训练）

在**不重新训练**的前提下，把 V4 的图、诊断目录下的表/图/报告全部刷新到当前代码与数据状态：

```bash
# 在项目根目录执行
export P1_RESULTS_ROOT=paper1_results_v4
python run_auto_tune_and_report.py --report-only

python scripts/run_p1_full_diagnostic.py \
  --results-root paper1_results_v4 \
  --results-roots paper1_results_v2,paper1_results_v3,paper1_results_v4 \
  --out paper1_results_diagnostic
```

或直接执行封装脚本（效果同上）：

```bash
bash scripts/rebuild_all_figures_and_reports.sh
```

完成后：
- **`paper1_results_v4/figures/`**：SI vs SSPG、MI vs DI、Bland-Altman、单 vs 联合、留一数据集、**6D 四张图**；若 run 目录下有 VAE 产物则还有 VAE 三张图。
- **`paper1_results_diagnostic/`**：`all_linear_vs_nonlinear.csv`、`figures/` 下全场景 SSPG/DI 条形图与热力图、6D 图、`DIAGNOSTIC_REPORT.md`。

---

## 3. 从零完整重建（含训练，可选）

若需要**从零训练 + 评估 + 出图**（例如换数据路径、换代码后重跑）：

1. **训练与评估（主用 V4 配置）**
   ```bash
   export CGM_PROJECT_OUTPUT=/path/to/cgm_project/output   # 数据根
   export P1_RESULTS_ROOT=paper1_results_v4
   python run_auto_tune_and_report.py
   ```
   会：多 seed × 多 λ 跑 pipeline、5-fold 评估、选 best run、写 V4 的 report 与 figures。

2. **刷新诊断（线性/非线性全场景 + 报告）**
   ```bash
   python scripts/run_p1_full_diagnostic.py \
     --results-root paper1_results_v4 \
     --results-roots paper1_results_v2,paper1_results_v3,paper1_results_v4 \
     --out paper1_results_diagnostic
   ```

3. **（可选）只补 VAE 拟合图**  
   若某次训练未保存 `training_curves.json` / `reconstruction_val_mse.npy` / `reconstruction_examples.npz`，可只重跑**一个** run 到该 run 目录，再执行上面「只更新图与报告」的步骤，VAE 三张图即会生成。详见 `docs/P1_FULL_DIAGNOSTIC_AND_FIGURES.md` 第 2.3 节。

---

## 4. 输出与脚本对照

| 产出 | 来源 |
|------|------|
| SI/MI vs 金标准、Bland-Altman、单 vs 联合、留一数据集、metrics 汇总 | `run_auto_tune_and_report.py --report-only` → `P1_RESULTS_ROOT/figures/` |
| 6D 图（pairwise、PCA、平行坐标、boxplot） | 同上（report-only 内调 `plot_p1_6d_latent.py`） |
| VAE 训练曲线、重建 MSE 直方图、示例 CGM | 同上（report-only 内调 `assess_vae_fit.py`，依赖 run 目录下 json/npy/npz） |
| 全场景线性 vs 非线性 CSV + 条形图 + 热力图 | `run_p1_full_diagnostic.py` → `paper1_results_diagnostic/` |
| DIAGNOSTIC_REPORT.md | `run_p1_full_diagnostic.py` → `paper1_results_diagnostic/DIAGNOSTIC_REPORT.md` |

所有脚本均位于 `scripts/`，入口与参数见各脚本顶部注释或 `docs/P1_FULL_DIAGNOSTIC_AND_FIGURES.md`。

---

## 5. 小结

- **保留**：`paper1_results*` 各 version 与 `paper1_results_diagnostic`，不删。
- **保证最新（不训练）**：执行 `rebuild_all_figures_and_reports.sh` 或其中两条命令即可。
- **完整重建（含训练）**：先 `run_auto_tune_and_report.py`，再 `run_p1_full_diagnostic.py`。
- 更多细节（图说明、VAE 图生成条件等）见 `docs/P1_FULL_DIAGNOSTIC_AND_FIGURES.md`。
