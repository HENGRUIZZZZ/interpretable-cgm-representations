# P1 完整诊断与图表说明

## 1. 你要的图都在哪

- **SI/MI vs 金标准（原有）**  
  `paper1_results_v4/figures/`：`p1_si_vs_sspg.png`、`p1_mi_vs_di.png`、Bland-Altman、单 vs 联合、留一数据集等。  
  用 `P1_RESULTS_ROOT=paper1_results_v4 python run_auto_tune_and_report.py --report-only` 会从 **V4 的 run** 读 CSV 并重画这些图。

- **6D 潜变量（新增）**  
  同一目录下：
  - `p1_6d_pairwise_by_dataset.png`：6 维两两散点（下三角），按 D1/D2/D4 着色。
  - `p1_6d_pca2d.png`：6D 做 PCA 到 2D，左图按 dataset、右图按 SSPG/DI 着色。
  - `p1_6d_parallel_by_dataset.png`：6D 平行坐标。
  - `p1_6d_boxplot_by_dataset.png`：各维度按 dataset 的分布。

- **VAE 拟合（新增，需重跑一次 pipeline 才有数据）**  
  - `p1_vae_training_curve.png`：训练/验证 loss 曲线。
  - `p1_vae_reconstruction_mse_hist.png`：验证集每样本重建 MSE 分布。
  - `p1_vae_reconstruction_examples.png`：几条 CGM 的实际 vs 重建。  
  当前 V4 的 run 是旧跑法，没有保存 `training_curves.json`、`reconstruction_val_mse.npy`、`reconstruction_examples.npz`。**重跑一次 pipeline（哪怕只跑一个 run）** 后，再执行 report-only 或诊断脚本就会自动生成上述 VAE 图。

- **线性 vs 非线性（全场景）**  
  `paper1_results_diagnostic/`：
  - `all_linear_vs_nonlinear.csv`：各 scenario（V2/V3/V4 的 best-SSPG run、best-DI run、以及 D1+D2 子集）× SSPG/DI × 方法（Ridge 线性、log、poly2、GB）的 5-fold Spearman r 与 RMSE。
  - `figures/p1_all_scenarios_sspg.png`、`p1_all_scenarios_di.png`：按场景分组的条形图。
  - `figures/p1_all_scenarios_heatmap.png`：场景 × (target_method) 的 Spearman r 热力图。

- **诊断报告**  
  `paper1_results_diagnostic/DIAGNOSTIC_REPORT.md`：VAE 拟合、6D、线性/非线性、以及「问题可能出在哪」的简要解读。

## 2. 怎么跑「完整流程」

### 2.1 只重画图（不训练）

```bash
export P1_RESULTS_ROOT=paper1_results_v4
python run_auto_tune_and_report.py --report-only
```

会：按 V4 的 best run 画 SI/MI vs 金标准、6D 图；若该 run 目录下已有 VAE 的 json/npy/npz，还会画 VAE 拟合图。

### 2.2 跑一次完整诊断（VAE + 6D + 线性 vs 非线性 + 报告）

```bash
python scripts/run_p1_full_diagnostic.py --results-root paper1_results_v4 --out paper1_results_diagnostic
```

会：用 V4 的 best run 做 6D 和 VAE 评估（无 VAE 文件则只打提示）、对 V2/V3/V4 所有 best 场景跑线性 vs 非线性、写 `paper1_results_diagnostic/DIAGNOSTIC_REPORT.md` 和上述 CSV/图。

### 2.3 补全 VAE 拟合图（训练一个 run，得到曲线与重建）

当前 V4 的 run 是之前跑的，没有保存训练曲线和重建结果。若要「先保证 VAE 拟合好」的图：

1. 只跑**一个** run（例如 SEED=21, LAMBDA_IR=0.05），并把结果写到**新目录**（避免覆盖现有 V4）：
   ```bash
   export P1_RESULTS_ROOT=paper1_results_v4
   export P1_SEEDS=21
   export P1_LAMBDAS=0.05
   python run_auto_tune_and_report.py
   ```
   或直接调 pipeline 写到一个 run 目录：
   ```bash
   export P1_RESULTS_DIR=paper1_results_v4/run_s21_lam0.05
   export P1_SEED=21
   export LAMBDA_IR=0.05
   python run_p1_full_pipeline.py
   ```
2. 该 run 目录下会出现：`training_curves.json`、`reconstruction_val_mse.npy`、`reconstruction_examples.npz`。
3. 再执行 report-only 或诊断脚本，就会在 figures 里生成 VAE 训练曲线、重建 MSE 直方图、示例 CGM 图。

## 3. 逻辑顺序（和你要求一致）

1. **VAE 要先拟合好**：看训练曲线、验证重建 MSE、示例 CGM；若拟合差，先修模型或数据。
2. **再看 6D 表示**：pairwise、PCA、平行坐标、boxplot，看是否按 dataset/金标准有合理结构。
3. **最后看回归方式与数据范围**：线性 vs 非线性、D1+D2 vs D1+D2+D4，用 `all_linear_vs_nonlinear.csv` 和热力图/条形图判断是回归方法问题、数据理解问题，还是 VAE 表示问题。

## 4. 参考

- 线性 vs 非线性单次对比：`scripts/compare_linear_vs_nonlinear.py`（可指定 `--csv`、`--datasets D1,D2`）。
- 6D 出图：`scripts/plot_p1_6d_latent.py`。
- VAE 拟合出图：`scripts/assess_vae_fit.py`。
- 全场景线性 vs 非线性：`scripts/run_all_linear_vs_nonlinear.py`。

## 5. 保留与重建

- **结果目录全部保留**（`paper1_results_v2`、`v3`、`v4`、`v5`、`paper1_results_diagnostic`），便于后续重建与对比。
- **不训练、只更新图与报告**：执行 `bash scripts/rebuild_all_figures_and_reports.sh`，或见 **`docs/P1_REBUILD_AND_RETENTION.md`** 中的完整命令与从零重建说明。
