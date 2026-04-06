"""
Paper 1 (CGM latent 主线) 实验配置：五个数据集 D1–D5 的命名、层级、训练/验证/测试划分。

数据目录约定：解压 cgm_project.tar.gz 后，output 下为 D1_metwally, D2_stanford, ...；
本配置中统一称 D1, D2, D3, D4, D5，通过 folder_name 映射到实际子目录。
"""

from typing import Dict, List, Optional, Tuple
import os

# ---------------------------------------------------------------------------
# D1–D5 数据集定义（命名统一为 D1, D2, D3, D4, D5）
# ---------------------------------------------------------------------------

DATASETS: Dict[str, dict] = {
    "D1": {
        "folder_name": "D1_metwally",
        "source": "Metwally et al. 2024, Nature Biomedical Engineering",
        "level": "Level1",
        "type": "OGTT + 金标准",
        "cgm_format": "meal_centered",  # meal_id + mins_since_meal + glucose_mgdl
        "has_sspg": True,
        "has_di": True,
        "has_homa_ir": True,
        "has_homa_b": True,
        "has_medications": False,
        "role": "机制对齐（latent–金标准相关）、主训练与验证",
    },
    "D2": {
        "folder_name": "D2_stanford",
        "source": "Stanford CGM Database",
        "level": "Level2",
        "type": "标准餐 + CGM",
        "cgm_format": "meal_centered",
        "has_sspg": True,
        "has_di": True,
        "has_homa_ir": True,
        "has_homa_b": True,
        "has_medications": False,
        "role": "换协议泛化（同一 encoder 在此 encode，看结构是否一致）",
    },
    "D3": {
        "folder_name": "D3_cgmacros",
        "source": "CGMacros 2025, Nature Scientific Data",
        "level": "Level3",
        "type": "自由生活 + 饮食记录",
        "cgm_format": "continuous",  # subject_id, timestamp, glucose_mgdl
        "has_sspg": False,
        "has_di": False,
        "has_homa_ir": False,
        "has_homa_b": False,
        "has_medications": False,
        "role": "真实世界/多中心泛化（latent 分布与结构稳定性）",
    },
    "D4": {
        "folder_name": "D3alt_hall",
        "source": "Hall et al. 2018, PLOS Biology",
        "level": "Level3",
        "type": "自由生活 CGM",
        "cgm_format": "continuous",
        "has_sspg": False,
        "has_di": False,
        "has_homa_ir": False,
        "has_homa_b": False,
        "has_medications": False,
        "role": "真实世界/多中心泛化",
    },
    "D5": {
        "folder_name": "D4_shanghai",
        "source": "Shanghai T2DM 2023, Nature Scientific Data",
        "level": "Level3",
        "type": "自由生活 + 用药",
        "cgm_format": "continuous",
        "has_sspg": False,
        "has_di": False,
        "has_homa_ir": True,  # 部分有
        "has_homa_b": False,
        "has_medications": True,
        "role": "真实世界/多中心泛化；后续 Paper 4 用药分析",
    },
}

# 餐心格式（可直接用 load_cgm_project_level1_level2）：D1, D2
MEAL_CENTERED_IDS: List[str] = ["D1", "D2"]
# 连续 CGM（需按餐时间切窗口或滑动窗口）：D3, D4, D5
CONTINUOUS_CGM_IDS: List[str] = ["D3", "D4", "D5"]


def get_data_dir(dataset_id: str, output_base: str, prefer_hall_naming: bool = True) -> str:
    """返回某数据集的完整路径。dataset_id 为 D1, D2, D3, D4, D5 之一。
    prefer_hall_naming: 若 True，D4 先尝试 output/D4_hall（与 cgm_all_datasets 一致），不存在再用 D3alt_hall。
    """
    if dataset_id not in DATASETS:
        raise KeyError(f"Unknown dataset_id: {dataset_id}. Use one of {list(DATASETS)}")
    path = os.path.join(output_base, DATASETS[dataset_id]["folder_name"])
    if dataset_id == "D4" and prefer_hall_naming:
        d4_hall = os.path.join(output_base, "D4_hall")
        if os.path.isdir(d4_hall):
            return d4_hall
    return path


# ---------------------------------------------------------------------------
# 训练 / 验证 / 测试 划分设计
# ---------------------------------------------------------------------------

# 默认随机种子（划分与训练一致）
SPLIT_SEED = 21

# Level 1：在 D1 上划分 train / val / test（按 subject，避免同一人既在 train 又在 test）
D1_TRAIN_FRAC = 0.70   # 70% 受试者用于训练
D1_VAL_FRAC = 0.15     # 15% 用于验证（早停、超参）
D1_TEST_FRAC = 0.15    # 15% 用于最终报告（latent–金标准相关、消融）
# 若 D1 人数少，可改为 80/10/10 或 85/0/15（无独立 val，用 train 做早停）

# Level 2：D2 可作为“域外验证”，不参与训练；或拿出部分做 train 与 D1 联合训练（可选）
D2_USE_AS: str = "encode_only"  # "encode_only" | "joint_train" | "finetune"
# encode_only：D1 训练，D2 只做 encode，看 latent 分布/结构
# joint_train：D1+D2 一起训练（可选）
# finetune：D1 训练后 D2 微调（可选）

# Level 3：D3, D4, D5 仅用于 encode（用 D1 或 D1+D2 训好的 encoder），不做训练
# 评估：多数据集 latent 分布对齐、跨 cohort 稳定性

# P1 完整方案：训练时合并使用的数据集（充分利用 D1+D2+D4 金标准）
P1_FULL_TRAIN_DATASETS: List[str] = ["D1", "D2", "D4"]
P1_TRAIN_FRAC = 0.80
P1_VAL_FRAC = 0.10
P1_TEST_FRAC = 0.10


def get_d1_split_design() -> dict:
    """返回 D1 划分设计说明（用于文档与实现）。"""
    return {
        "split_unit": "subject",
        "stratify_by": "diagnosis",  # 若有 Pre-D / T2D / Healthy，按此分层
        "train_frac": D1_TRAIN_FRAC,
        "val_frac": D1_VAL_FRAC,
        "test_frac": D1_TEST_FRAC,
        "seed": SPLIT_SEED,
        "note": "同一 subject 的所有 meal 窗口只出现在 train / val / test 之一；test 仅用于最终评估与金标准相关，不参与训练与早停。",
    }


def get_level1_protocol() -> dict:
    """Level 1 实验流程（机制对齐）。"""
    return {
        "train_data": ["D1"],  # 仅 D1 或 D1+D2
        "train_split": "D1 按 subject 70/15/15 → train/val/test",
        "validation": "val 集上重建损失 / 早停",
        "test_eval": "test 集上：latent 与 SSPG/DI/HOMA_IR/HOMA_B 相关、回归或分类；消融（无机制 decoder / 无 OGTT 锚定）",
    }


def get_level2_protocol() -> dict:
    """Level 2 实验流程（换协议泛化）。"""
    return {
        "encoder": "Level 1 在 D1 上训好的 encoder（固定或微调）",
        "data": "D2 全部或按 subject 划分",
        "eval": "D2 上 encode → latent 分布与 D1 比较、主轴对齐；若 D2 有金标准，算 latent–金标准相关",
    }


def get_level3_protocol() -> dict:
    """Level 3 实验流程（多中心/真实世界泛化）。"""
    return {
        "encoder": "Level 1 训好的 encoder",
        "data": "D3, D4, D5（需先按餐时间切窗口或滑动窗口得到餐心序列）",
        "eval": "各 cohort latent 分布、跨数据集主轴对齐、与 D1/D2 结构一致性；若有简单标签（如诊断）可做分层/预测。",
    }
