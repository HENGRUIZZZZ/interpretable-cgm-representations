# 使用 V2 数据后的接下来步骤

## 1. 使用 V2 数据

1. **解压**：将 `cgm_project_v2_final.tar.gz` 解压到某目录，得到 `cgm_project/output/`。
2. **指向 V2**：运行实验前设置环境变量：
   ```bash
   export CGM_PROJECT_OUTPUT=/path/to/cgm_project/output
   ```
   或在 `run_paper1_full.py` 中修改 `OUTPUT_BASE`。
3. **诊断**：确认 D1 加载与金标准对齐：
   ```bash
   CGM_PROJECT_OUTPUT=/path/to/cgm_project/output python scripts/check_d1_data.py
   ```
   V2 下应看到：batch 59 人，32 人有金标准，test 约 9 人其中约 3 人有金标准。

## 2. 已做的代码适配

- **load_cgm_project_data.py**：
  - 支持 V2 D1 的 cgm 格式（`subject_id`, `timepoint_mins`, `glucose_mg_dl`，无 `meal_id`），自动转为「一人一餐」的 meal 格式。
  - 支持 V2 labels 列名（`SSPG`, `DI`, `HOMA_IR`, `HOMA_B`），自动映射为小写 `sspg`, `di`, `homa_ir`, `homa_b` 供下游使用。
- **scripts/check_d1_data.py**：原始 labels 检查兼容 `SSPG` 列名。

## 3. 建议的实验顺序

1. **Level 1（D1）**：用 V2 数据跑通 `run_paper1_full.py`，在 test 上得到 latent–金标准相关（n≈3）；可选在全 D1 有金标准者（32 人）上再算一次相关。
2. **Level 2（D2）**：脚本已包含「D1 训好后在 D2 上 encode」；若 V2 的 D2 表结构有变（如列名），需同样在 loader 中做兼容（目前 D2 可能仍为旧格式，若报错再适配）。
3. **跨数据集金标准**：报告中的 56 人 CGM∩SSPG（D1:5 公开 + D2:31 + D4:20）可用于联合相关或外部验证，需在 D2/D4 加载后与 labels 对齐，并统一 subject_id 命名空间（若需要可再加一段合并逻辑）。
4. **Level 3（D3/D4/D5）**：用 `load_cgm_project_level3` 在自由生活数据上 encode；若 V2 中 D3/D4/D5 的目录或列名有变，在 `paper1_experiment_config.py` 与 loader 中对应调整路径与列名。

## 4. 若 D2/D3/D4/D5 报错

V2 包内可能含 `D4_hall`、`D5_shanghai` 等目录；当前配置中 D4 对应 `D3alt_hall`、D5 对应 `D4_shanghai`。若你使用的 V2 只提供 `D4_hall`/`D5_shanghai`，可在 `paper1_experiment_config.py` 的 `DATASETS` 里将 `folder_name` 改为 `D4_hall` / `D5_shanghai`，或增加别名由 `get_data_dir` 解析。
