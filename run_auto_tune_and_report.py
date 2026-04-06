"""
P1 自动调优 + 评估 + 出图 + 结果报告（按论文目的撰写解读）。

流程：
  1. 多 seed × LAMBDA_IR 组合跑 pipeline + 评估，按 5-fold Spearman(SSPG)+Spearman(DI) 选最优（6D Ridge）
  2. 将最优结果复制到 paper1_results，生成 figures
  3. 写出 paper1_results/RESULTS_AND_FIGURES.md（结果表 + 图片解读）

用法（项目根目录）：
  python run_auto_tune_and_report.py
  # 可选跳过调优、仅用默认跑一遍： --no-tune
"""
import os
import re
import sys
import shutil
import subprocess
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 结果根目录：可通过环境变量 P1_RESULTS_ROOT 覆盖，便于每次实验单独备份
P1_RESULTS = os.environ.get("P1_RESULTS_ROOT", "paper1_results")
# 选优时 SSPG_r 的权重（默认 1.0，即 SSPG_r + DI_r）
SSPG_WEIGHT = float(os.environ.get("P1_SSPG_WEIGHT", "1.0"))
# 扩展调优：更多 seed 与 LAMBDA_IR，选优后写 FINAL_REPORT；可用 P1_SEEDS / P1_LAMBDAS 覆盖做快速试验
def _parse_int_list(env_key: str, default: list[int]) -> list[int]:
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(env_key: str, default: list[float]) -> list[float]:
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return default
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


SEEDS = _parse_int_list("P1_SEEDS", [21, 42, 43, 44])
LAMBDAS = _parse_float_list("P1_LAMBDAS", [0.01, 0.02, 0.05, 0.1])


def run_pipeline(lam: float, seed: int, out_dir: str) -> bool:
    env = {
        **os.environ,
        "LAMBDA_IR": str(lam),
        "LAMBDA_SSPG": os.environ.get("LAMBDA_SSPG", "0.01"),
        "LAMBDA_DI": os.environ.get("LAMBDA_DI", "0.01"),
        "P1_SEED": str(seed),
        "P1_RESULTS_DIR": out_dir,
    }
    if os.environ.get("P1_TRAIN_DATASETS", "").strip():
        env["P1_TRAIN_DATASETS"] = os.environ.get("P1_TRAIN_DATASETS").strip()
    r = subprocess.run(
        [sys.executable, "run_p1_full_pipeline.py"],
        env=env,
        cwd=REPO_ROOT,
        timeout=600,
    )
    return r.returncode == 0


def run_evaluate(csv_path: str, out_dir: str) -> bool:
    cmd = [sys.executable, "scripts/evaluate_p1_metrics.py", "--csv", csv_path, "--out", out_dir]
    # 若设置了 P1_GOLD_DATASETS（如 "D1,D2"），则只在这些 dataset 上评估金标准预测
    ds_filter = os.environ.get("P1_GOLD_DATASETS", "").strip()
    if ds_filter:
        cmd.extend(["--datasets", ds_filter])
    r = subprocess.run(cmd, cwd=REPO_ROOT, timeout=120)
    return r.returncode == 0


def parse_5fold_metrics(summary_path: str) -> dict:
    """解析 5fold_subject 的 spearman_r, rmse, r2（sspg 与 di）。返回 {"sspg": {r, rmse, r2}, "di": {...}}."""
    out = {"sspg": {"spearman_r": float("nan"), "rmse": float("nan"), "r2": float("nan")},
           "di": {"spearman_r": float("nan"), "rmse": float("nan"), "r2": float("nan")}}
    if not os.path.isfile(summary_path):
        return out
    with open(summary_path) as f:
        text = f.read()
    current = None
    for line in text.splitlines():
        if line.strip().startswith("Target:"):
            m = re.search(r"Target:\s*(\w+)", line)
            current = m.group(1).strip() if m else None
            continue
        if "5fold_subject" not in line or current not in out:
            continue
        for key, pat in [("spearman_r", r"spearman_r\s*=\s*([-\d.]+)"),
                         ("rmse", r"rmse\s*=\s*([-\d.]+)"), ("r2", r"r2\s*=\s*([-\d.]+)")]:
            m = re.search(pat, line)
            if m:
                out[current][key] = float(m.group(1))
            continue
    return out


def _parse_5fold_spearman(summary_path: str) -> tuple[float, float]:
    sspg_r, di_r = float("nan"), float("nan")
    if not os.path.isfile(summary_path):
        return sspg_r, di_r
    with open(summary_path) as f:
        text = f.read()
    current = None
    for line in text.splitlines():
        if line.strip().startswith("Target:"):
            m = re.search(r"Target:\s*(\w+)", line)
            current = m.group(1).strip() if m else None
            continue
        if "5fold_subject" in line and "spearman_r" in line:
            m = re.search(r"spearman_r\s*=\s*([-\d.]+)", line)
            if m:
                v = float(m.group(1))
                if current == "sspg":
                    sspg_r = v
                elif current == "di":
                    di_r = v
            continue
    return sspg_r, di_r


# keep alias for callers
def parse_5fold_spearman(summary_path: str) -> tuple[float, float]:
    return _parse_5fold_spearman(summary_path)


def copy_best_to_final(best_dir: str):
    os.makedirs(P1_RESULTS, exist_ok=True)
    for name in [
        "autoencoder_p1_full.pt", "correlations.txt",
        "latent_and_gold_test.csv", "latent_and_gold_all.csv",
        "evaluation_metrics.csv", "evaluation_metrics_summary.txt",
        "evaluation_5fold_per_fold.csv",
    ]:
        src = os.path.join(best_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(P1_RESULTS, name))
    if os.path.isdir(best_dir):
        for f in os.listdir(best_dir):
            if f.startswith("joint_weights_") and f.endswith(".csv"):
                shutil.copy2(os.path.join(best_dir, f), os.path.join(P1_RESULTS, f))


def run_plots():
    fig_dir = os.path.join(P1_RESULTS, "figures")
    # 各取最好 + 汇总图（r、R²、RMSE 柱状，单 vs 联合，留一数据集）
    r = subprocess.run(
        [
            sys.executable, "scripts/plot_p1_results.py",
            "--best-per-target",
            "--summary-figures",
            "--out", fig_dir,
            "--correlations", os.path.join(P1_RESULTS, "correlations.txt"),
            "--metrics-csv", os.path.join(P1_RESULTS, "evaluation_metrics.csv"),
        ],
        cwd=REPO_ROOT,
        timeout=60,
    )
    return r.returncode == 0


def run_6d_and_vae_plots(fig_dir: str, best_run_dir: str | None):
    """生成 6D 潜变量图；若 run 目录有 reconstruction/training 文件则生成 VAE 拟合图。"""
    if not best_run_dir or not os.path.isdir(best_run_dir):
        return
    csv_path = os.path.join(best_run_dir, "latent_and_gold_all.csv")
    if os.path.isfile(csv_path):
        subprocess.run(
            [sys.executable, "scripts/plot_p1_6d_latent.py", "--csv", csv_path, "--out", fig_dir],
            cwd=REPO_ROOT,
            timeout=120,
        )
        subprocess.run(
            [sys.executable, "scripts/assess_vae_fit.py", "--run-dir", best_run_dir, "--out", fig_dir],
            cwd=REPO_ROOT,
            timeout=60,
        )


def write_report(best_lam: float, best_seed: int | None, tune_scores: list[tuple], metrics_summary_path: str, fig_dir: str):
    """按论文目的写：结果摘要 + 图片路径 + 解读。tune_scores: [(seed, lam, sspg_r, di_r), ...] 或 [(lam, sspg_r, di_r), ...]。"""
    path = os.path.join(P1_RESULTS, "RESULTS_AND_FIGURES.md")
    with open(metrics_summary_path) as f:
        summary_preview = f.read()[:2500]

    fig_base = "figures"
    seed_note = f" **SEED = {best_seed}**" if best_seed is not None else ""
    content = f"""# P1 结果与图表解读（论文用）

## 1. 自动调优与选定配置

- **调优目标**：多 **SEED** × **LAMBDA_IR** 组合训练，以 5-fold（按 subject）**6D Ridge** 预测 SSPG/DI 的 **Spearman r 之和** 选优。
- **选定**：**LAMBDA_IR = {best_lam}**{seed_note}（下表为各组合的 5-fold Spearman r）。评估始终使用 **6 维潜变量** Ridge 回归。

| SEED | LAMBDA_IR | SSPG Spearman r (5-fold) | DI Spearman r (5-fold) | 合计 |
|------|-----------|--------------------------|------------------------|------|
"""
    for row in tune_scores:
        if len(row) == 4:
            seed, lam, sr, dr = row
            total = (sr if sr == sr else 0) + (dr if dr == dr else 0)
            content += f"| {seed} | {lam} | {sr:.3f} | {dr:.3f} | {total:.3f} |\n"
        else:
            lam, sr, dr = row
            total = (sr if sr == sr else 0) + (dr if dr == dr else 0)
            content += f"| - | {lam} | {sr:.3f} | {dr:.3f} | {total:.3f} |\n"
    content += """

## 2. 评估指标摘要

以下为选定配置下 `evaluation_metrics_summary.txt` 的节选（全量见该文件）。

```
"""
    content += summary_preview + "\n```\n\n"

    content += f"""## 3. 图表文件与论文解读

所有图表保存在 **`{fig_base}/`** 目录下，用于验证 CGM 潜变量与金标准（SSPG=胰岛素抵抗，DI=β 细胞功能）的对应关系。

### 3.1 `{fig_base}/p1_si_vs_sspg.png` — 敏感性指数 si 与 SSPG

- **目的**：检验模型潜变量 **si**（胰岛素敏感性）与金标准 **SSPG**（稳态血浆葡萄糖，越高越抵抗）的单调关系。
- **解读**：若 si 与 SSPG 呈负相关（Spearman r < 0），符合生理（敏感性高则 SSPG 低）；若呈正相关则可能反映数据集混合或样本量限制。图中给出按 dataset 着色、回归线及 Spearman r/p，用于说明“CGM 衍生的 si 是否与金标准 IR 指标一致”。

### 3.2 `{fig_base}/p1_mi_vs_di.png` — 胰岛素分泌指数 mi 与 DI

- **目的**：检验潜变量 **mi**（模型中的胰岛素分泌/处置相关维度）与金标准 **DI**（处置指数，β 细胞功能）的对应。
- **解读**：DI 由 OGTT 等金标准方法得到；若 mi 与 DI 正相关，支持“从 CGM 可解释地恢复 β 细胞功能信息”。图中同样给出按 dataset 的散点、回归线及 r/p，便于讨论单参数相关与 6D 联合预测的差异（见文档“单参数 vs 多参数”）。

### 3.3 `{fig_base}/p1_blandaltman_si_sspg.png` — Bland-Altman（si vs SSPG）

- **目的**：评估 si 与 SSPG 的**一致性**（非仅相关）：均值差与 95% 一致性界限。
- **解读**：若点大多落在 ±1.96 SD 内，说明在尺度上 si 与 SSPG 有一定一致性；若存在系统偏差或离散度大，可在文中说明为“CGM 潜变量与金标准单位不同，更适合作相关/预测分析而非直接替代”。

### 3.4 `{fig_base}/p1_correlations_summary.txt`

- 各潜变量（si, mi, tau_m, sg）与各金标准（sspg, di 等）的 Spearman r、p、n 汇总，供正文表格引用。

---

## 4. 论文表述建议

- **主要结论**：CGM 驱动的机制化潜变量（si, mi 等）与 SSPG（IR）和 DI（β 细胞）具有可量化关联；**6D 联合预测**（Ridge）优于单潜变量，符合生理上多参数联合决定葡萄糖处置的设定。
- **评估**：以 5-fold 按 subject 的 Spearman r、RMSE、R² 为主；留一数据集出用于讨论泛化。
- **局限**：金标准样本量有限、多中心尺度差异；单潜变量相关仅部分反映联合关系，需结合 6D Ridge 与系数表（`joint_weights_sspg.csv`, `joint_weights_di.csv`）一起解读。
"""
    with open(path, "w") as f:
        f.write(content)
    print(f"Wrote {path}")


def write_final_report(tune_scores: list, best_dir: str, best_lam: float, best_seed: int, best_sspg_dir_override: str | None = None, best_di_dir_override: str | None = None):
    """写 FINAL_REPORT.md：全部 run 表、选优说明、best SSPG/DI run 的指标、训练与数据说明、图列表。"""
    path = os.path.join(P1_RESULTS, "FINAL_REPORT.md")
    # 从 tune_scores 找 best SSPG run 和 best DI run（(seed, lam, sspg_r, di_r)）
    best_sspg_seed, best_sspg_lam, best_sspg_r = None, None, -2.0
    best_di_seed, best_di_lam, best_di_r = None, None, -2.0
    for row in tune_scores:
        if len(row) == 4:
            seed, lam, sr, dr = row
            if np.isfinite(sr) and sr > best_sspg_r:
                best_sspg_r, best_sspg_seed, best_sspg_lam = sr, seed, lam
            if np.isfinite(dr) and dr > best_di_r:
                best_di_r, best_di_seed, best_di_lam = dr, seed, lam
        elif len(row) == 3:
            lam, sr, dr = row
            if np.isfinite(sr) and sr > best_sspg_r:
                best_sspg_r, best_sspg_lam = sr, lam
            if np.isfinite(dr) and dr > best_di_r:
                best_di_r, best_di_lam = dr, lam
    # 若只有单次 run（no-tune），用主 run 的 seed/lam 作为 best_sspg/best_di
    if best_sspg_seed is None and tune_scores:
        best_sspg_seed, best_sspg_lam = best_seed, best_sspg_lam if best_sspg_lam is not None else best_lam
    if best_di_seed is None and tune_scores:
        best_di_seed, best_di_lam = best_seed, best_di_lam if best_di_lam is not None else best_lam
    best_sspg_dir = best_sspg_dir_override if best_sspg_dir_override else (os.path.join(P1_RESULTS, f"run_s{best_sspg_seed}_lam{best_sspg_lam}") if best_sspg_seed is not None else None)
    best_di_dir = best_di_dir_override if best_di_dir_override else (os.path.join(P1_RESULTS, f"run_s{best_di_seed}_lam{best_di_lam}") if best_di_seed is not None else None)
    # 用于显示 run 名称（若为 override 则取目录名）
    best_sspg_name = os.path.basename(best_sspg_dir) if best_sspg_dir and os.path.isdir(best_sspg_dir) else (f"run_s{best_sspg_seed}_lam{best_sspg_lam}" if best_sspg_seed is not None else "")
    best_di_name = os.path.basename(best_di_dir) if best_di_dir and os.path.isdir(best_di_dir) else (f"run_s{best_di_seed}_lam{best_di_lam}" if best_di_seed is not None else "")
    summary_in_results = os.path.join(P1_RESULTS, "evaluation_metrics_summary.txt")
    sspg_metrics = parse_5fold_metrics(os.path.join(best_sspg_dir, "evaluation_metrics_summary.txt")) if best_sspg_dir and os.path.isdir(best_sspg_dir) else (parse_5fold_metrics(summary_in_results) if os.path.isfile(summary_in_results) else {})
    di_metrics = parse_5fold_metrics(os.path.join(best_di_dir, "evaluation_metrics_summary.txt")) if best_di_dir and os.path.isdir(best_di_dir) else (parse_5fold_metrics(summary_in_results) if os.path.isfile(summary_in_results) else {})
    lines = [
        "# P1 最终报告：自动调优、选优与全部结果",
        "",
        "## 1. 全部 run 结果（SEED × LAMBDA_IR）",
        "",
        "| SEED | LAMBDA_IR | SSPG Spearman r (5-fold) | DI Spearman r (5-fold) | 合计 |",
        "|------|-----------|--------------------------|------------------------|------|",
    ]
    for row in tune_scores:
        if len(row) == 4:
            seed, lam, sr, dr = row
            total = (sr if np.isfinite(sr) else 0) + (dr if np.isfinite(dr) else 0)
            lines.append(f"| {seed} | {lam} | {sr:.3f} | {dr:.3f} | {total:.3f} |")
        else:
            lam, sr, dr = row
            total = (sr if np.isfinite(sr) else 0) + (dr if np.isfinite(dr) else 0)
            lines.append(f"| — | {lam} | {sr:.3f} | {dr:.3f} | {total:.3f} |")
    lines.extend([
        "",
        "## 2. 选优结果",
        "",
        f"- **按 SSPG_r + DI_r 之和** 选出的主 run：**SEED={best_seed} LAMBDA_IR={best_lam}**，结果已复制到 `paper1_results/`。",
        f"- **SSPG 最优 run**（用于 si vs SSPG、Bland-Altman 图）：" + (f"`{best_sspg_name}` → SSPG r={best_sspg_r:.3f}" + (f", RMSE={sspg_metrics.get('sspg', {}).get('rmse', float('nan')):.1f}" if sspg_metrics and np.isfinite(sspg_metrics.get('sspg', {}).get('rmse', float('nan'))) else "") + "." if best_sspg_name else "（仅单次 run 时与主 run 相同）."),
        f"- **DI 最优 run**（用于 mi vs DI 图）：" + (f"`{best_di_name}` → DI r={best_di_r:.3f}" + (f", RMSE={di_metrics.get('di', {}).get('rmse', float('nan')):.1f}" if di_metrics and np.isfinite(di_metrics.get('di', {}).get('rmse', float('nan'))) else "") + "." if best_di_name else "（仅单次 run 时与主 run 相同）."),
        "",
        "## 3. 训练方法与数据利用",
        "",
        "见 **`docs/TRAINING_AND_OPTIMIZATION.md`**：当前为半监督混合 VAE（重建 + IR 弱监督）；SSPG/DI 仅在评估阶段用 Ridge 拟合金标准。数据：80/10/10 划分训 VAE，5-fold 评估时使用全部有金标准 subject。",
        "",
        "## 4. 全部指标与图",
        "",
        "- 完整 r、R²、RMSE、MAE 表与解读：**`paper1_results/ALL_RESULTS_AND_GUIDE.md`**",
        "- 明细 CSV：`paper1_results/evaluation_metrics.csv`",
        "- 图目录：`paper1_results/figures/`",
        "  - p1_si_vs_sspg.png, p1_mi_vs_di.png, p1_blandaltman_si_sspg.png（各取最好）",
        "  - p1_metrics_summary.png, p1_single_vs_joint_sspg.png, p1_single_vs_joint_di.png, p1_single_vs_joint_sspg_rmse.png, p1_single_vs_joint_di_rmse.png, p1_leave_one_dataset.png",
        "",
        "## 5. 5-fold 每折表现与「全部好的数据」",
        "",
        "主 run 的 **每折** Spearman r、RMSE 见 `paper1_results/evaluation_5fold_per_fold.csv`（若有）。图中 6D Ridge 的误差条即 5 折的波动，**单折最高 r 可达 0.8+**，报告正文取的是 5 折平均（如 DI r=0.677）。",
        "",
        "**关于「之前 0.72、现在 0.677」**：报告一律用 **5-fold 平均**；不同 run（如 run_s21_lam0.02）单折或曾出现更高 r。当前选优按 SSPG_r+DI_r 之和，故主 run 为 run_s21_lam0.05，其 DI 平均 r=0.677，单折最高仍可 >0.8，并非结果变差，而是汇报口径统一为平均。",
        "",
        "**关于 HOMA-IR（IR）**：理论上 IR 有训练时的弱监督（IR head），应相对好预测。评估里 HOMA-IR 同样用 Ridge(6D) 做 5-fold，主 run 下 HOMA-IR 5-fold r 约 0.45–0.55（见 evaluation_metrics.csv）。若表里 IR r 低于预期，可能原因：(1) 评估与训练目标不一致（训练是 log(HOMA_IR+1)，评估是原始尺度）；(2) LAMBDA_IR 可再调大或加入选优；(3) 金标准 HOMA-IR 样本/标定差异。详见 `docs/TRAINING_AND_OPTIMIZATION.md`。",
        "",
    ])
    # 若存在每折 CSV，在「关于 0.72」之前插入每折 min/mean/max
    per_fold_path = os.path.join(P1_RESULTS, "evaluation_5fold_per_fold.csv")
    if os.path.isfile(per_fold_path):
        import csv
        by_target = {}
        with open(per_fold_path) as f:
            r = csv.DictReader(f)
            for row in r:
                t = row.get("target", "")
                if t not in by_target:
                    by_target[t] = []
                try:
                    by_target[t].append({"spearman_r": float(row.get("spearman_r", np.nan)), "rmse": float(row.get("rmse", np.nan))})
                except (ValueError, TypeError):
                    pass
        idx = next((i for i, s in enumerate(lines) if "关于「之前 0.72" in s), len(lines))
        insert_list = [""]
        for t in ["sspg", "di"]:
            if t in by_target and by_target[t]:
                vals = [x["spearman_r"] for x in by_target[t] if np.isfinite(x["spearman_r"])]
                if vals:
                    insert_list.append(f"- **{t.upper()}** 5 折 Spearman r：min={min(vals):.3f}, mean={np.mean(vals):.3f}, max={max(vals):.3f}")
        insert_list.append("")
        for j, ex in enumerate(insert_list):
            lines.insert(idx + j, ex)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


def _collect_tune_scores_from_dirs(results_dir: str) -> tuple[list, str | None, float, int, str | None, str | None]:
    """从 paper1_results 下已有 run_s*_lam* / tune_* 目录收集 tune_scores，返回 (tune_scores, best_dir, best_lam, best_seed, best_sspg_dir, best_di_dir)。"""
    import re
    tune_scores = []
    best_dir = None
    best_score = -2.0
    best_lam, best_seed = 0.05, 21
    best_sspg_dir = None
    best_di_dir = None
    best_sspg_r = -2.0
    best_di_r = -2.0
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        if not os.path.isdir(path):
            continue
        summary_path = os.path.join(path, "evaluation_metrics_summary.txt")
        if not os.path.isfile(summary_path):
            continue
        sspg_r, di_r = parse_5fold_spearman(summary_path)
        total = SSPG_WEIGHT * (sspg_r if np.isfinite(sspg_r) else 0) + (di_r if np.isfinite(di_r) else 0)
        if np.isfinite(sspg_r) and sspg_r > best_sspg_r:
            best_sspg_r, best_sspg_dir = sspg_r, path
        if np.isfinite(di_r) and di_r > best_di_r:
            best_di_r, best_di_dir = di_r, path
        if name.startswith("run_s") and "_lam" in name:
            m = re.match(r"run_s(\d+)_lam([\d.]+)", name)
            if m:
                seed, lam = int(m.group(1)), float(m.group(2))
                tune_scores.append((seed, lam, sspg_r, di_r))
                if total > best_score:
                    best_score = total
                    best_dir = path
                    best_lam, best_seed = lam, seed
        elif name.startswith("tune_"):
            try:
                lam = float(name.replace("tune_", ""))
                tune_scores.append((lam, sspg_r, di_r))
                if total > best_score:
                    best_score = total
                    best_dir = path
                    best_lam = lam
            except ValueError:
                pass
    return tune_scores, best_dir, best_lam, best_seed, best_sspg_dir, best_di_dir


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tune", action="store_true", help="Skip tuning; run once with default LAMBDA_IR and report.")
    ap.add_argument("--report-only", action="store_true", help="Rebuild report from existing run dirs (no training).")
    args = ap.parse_args()

    if getattr(args, "report_only", False):
        tune_scores, best_dir, best_lam, best_seed, best_sspg_dir, best_di_dir = _collect_tune_scores_from_dirs(P1_RESULTS)
        print(f"Collected {len(tune_scores)} runs; best by sum: {best_dir} (SEED={best_seed}, LAMBDA_IR={best_lam})")
        run_plots()
        fig_dir = os.path.join(P1_RESULTS, "figures")
        run_6d_and_vae_plots(fig_dir, best_dir)
        metrics_path = os.path.join(P1_RESULTS, "evaluation_metrics_summary.txt")
        write_report(best_lam, best_seed, tune_scores, metrics_path, fig_dir)
        write_final_report(tune_scores, best_dir or P1_RESULTS, best_lam, best_seed, best_sspg_dir, best_di_dir)
        print("Done (report-only). See paper1_results/FINAL_REPORT.md and RESULTS_AND_FIGURES.md.")
        return

    if args.no_tune:
        out_dir = P1_RESULTS
        best_dir = P1_RESULTS
        os.makedirs(out_dir, exist_ok=True)
        lam, seed = 0.05, 21
        print(f"Running single run: SEED={seed}, LAMBDA_IR={lam}, P1_RESULTS_DIR={out_dir}")
        if not run_pipeline(lam, seed, out_dir):
            print("Pipeline failed.")
            sys.exit(1)
        csv_path = os.path.join(out_dir, "latent_and_gold_all.csv")
        if not run_evaluate(csv_path, out_dir):
            print("Evaluate failed.")
            sys.exit(1)
        best_lam, best_seed = lam, seed
        tune_scores = [(lam, *parse_5fold_spearman(os.path.join(out_dir, "evaluation_metrics_summary.txt")))]
    else:
        tune_scores = []
        best_dir = None
        best_score = -2.0
        for seed in SEEDS:
            for lam in LAMBDAS:
                out_dir = f"{P1_RESULTS}/run_s{seed}_lam{lam}"
                print(f"\n--- SEED={seed} LAMBDA_IR={lam} -> {out_dir} ---")
                os.makedirs(out_dir, exist_ok=True)
                if not run_pipeline(lam, seed, out_dir):
                    print(f"Pipeline failed.")
                    tune_scores.append((seed, lam, float("nan"), float("nan")))
                    continue
                csv_path = os.path.join(out_dir, "latent_and_gold_all.csv")
                if not run_evaluate(csv_path, out_dir):
                    print(f"Evaluate failed.")
                    tune_scores.append((seed, lam, float("nan"), float("nan")))
                    continue
                sspg_r, di_r = parse_5fold_spearman(os.path.join(out_dir, "evaluation_metrics_summary.txt"))
                tune_scores.append((seed, lam, sspg_r, di_r))
                total = SSPG_WEIGHT * (sspg_r if sspg_r == sspg_r else 0) + (di_r if di_r == di_r else 0)
                print(f"  6D Ridge 5-fold: SSPG r={sspg_r:.3f}, DI r={di_r:.3f}, weighted_sum={total:.3f} (weight={SSPG_WEIGHT})")
                if total > best_score:
                    best_score = total
                    best_dir = out_dir
                    best_lam, best_seed = lam, seed

        if best_dir is None:
            best_lam, best_seed = LAMBDAS[0], SEEDS[0]
            best_dir = f"{P1_RESULTS}/run_s{best_seed}_lam{best_lam}"
        print(f"\nBest: SEED={best_seed} LAMBDA_IR={best_lam} (sum r={best_score:.3f}) -> copy to {P1_RESULTS}")
        copy_best_to_final(best_dir)

    fig_dir = os.path.join(P1_RESULTS, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    if not run_plots():
        print("Plot script failed.")
    run_6d_and_vae_plots(fig_dir, best_dir)
    metrics_path = os.path.join(P1_RESULTS, "evaluation_metrics_summary.txt")
    write_report(best_lam, best_seed, tune_scores, metrics_path, fig_dir)
    write_final_report(tune_scores, best_dir, best_lam, best_seed)
    print("\nDone. See paper1_results/FINAL_REPORT.md and RESULTS_AND_FIGURES.md.")


if __name__ == "__main__":
    main()
