"""
二分类模型训练模块
训练 Normal vs Abnormal 分类模型
默认数据集: test-datasets (包含 Normal/ 和 Abnormal/ 子目录)
"""
import argparse
import json
import hashlib
import logging
import os
import pickle
import random
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# 忽略 NumPy 除零警告（autocorr 计算中可能出现）
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

# 限制 OpenMP 线程数，避免资源不足错误
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, rcParams
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

# 采样技术
try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    from imblearn.ensemble import BalancedBaggingClassifier
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logging.warning("未安装 imbalanced-learn，将跳过 SMOTE 采样")

SEED = 42

DEFAULT_BINARY_PARAMS: dict[str, float | int | str | bool] = {
    "objective": "binary",
    "n_estimators": 1000,          # 增加树的数量
    "learning_rate": 0.01,         # 降低学习率，更精细的收敛
    # 模型复杂度 - 适度增加以提升区分能力
    "num_leaves": 63,              # 增加叶子数
    "max_depth": 8,                # 增加树深度
    "min_child_samples": 10,       # 减少叶子节点最少样本数
    "min_child_weight": 1e-4,      # 减小叶子节点最小权重
    # 正则化 - 适度降低
    "reg_alpha": 0.01,             # L1正则化
    "reg_lambda": 0.01,            # L2正则化
    # 采样
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "subsample_freq": 3,           # 更频繁的采样
    # 类别不平衡
    "is_unbalance": True,          # 自动处理类别不平衡
}

DEFAULT_SAMPLING_CONFIG: dict[str, float | int | str] = {
    "mode": "window",
    "window_sec": 120.0,
    "step_sec": 30.0,
    "min_points": 10,
    "max_windows_per_file": 100,
    "num_workers": 0,
}

# 特征类别映射（12类）
FEATURE_CATEGORY_MAPPING: dict[str, str] = {
    # 基础统计特征
    "value_count": "基础统计特征", "value_mean": "基础统计特征", "value_min": "基础统计特征",
    "value_max": "基础统计特征", "value_median": "基础统计特征", "value_first": "基础统计特征",
    "value_last": "基础统计特征", "value_rms": "基础统计特征",
    # 离散程度特征
    "value_std": "离散程度特征", "value_range": "离散程度特征", "value_iqr": "离散程度特征", "value_cv": "离散程度特征",
    # 分布形态特征
    "skewness": "分布形态特征", "kurtosis": "分布形态特征",
    "value_q01": "分布形态特征", "value_q05": "分布形态特征", "value_q10": "分布形态特征",
    "value_q25": "分布形态特征", "value_q75": "分布形态特征", "value_q90": "分布形态特征",
    "value_q95": "分布形态特征", "value_q99": "分布形态特征",
    # 差分特征
    "diff_mean": "差分特征", "diff_std": "差分特征", "diff_abs_mean": "差分特征",
    "diff_max": "差分特征", "diff_min": "差分特征", "diff_abs_max": "差分特征",
    "diff2_mean": "差分特征", "diff2_std": "差分特征", "diff2_abs_mean": "差分特征",
    "direction_changes": "差分特征", "direction_change_rate": "差分特征",
    # 趋势特征
    "value_change": "趋势特征", "value_change_ratio": "趋势特征", "trend_slope": "趋势特征",
    "seg_mean_diff_12": "趋势特征", "seg_mean_diff_23": "趋势特征", "seg_mean_diff_13": "趋势特征",
    # 局部特征
    "seg1_mean": "局部特征", "seg2_mean": "局部特征", "seg3_mean": "局部特征",
    "seg1_std": "局部特征", "seg2_std": "局部特征", "seg3_std": "局部特征",
    "local_max_count": "局部特征", "local_min_count": "局部特征", "peak_count": "局部特征", "peak_rate": "局部特征",
    # 稳定性特征
    "rolling_var_mean": "稳定性特征", "rolling_var_std": "稳定性特征", "rolling_var_max": "稳定性特征",
    # 自相关特征
    "autocorr_lag1": "自相关特征", "autocorr_lag5": "自相关特征",
    # 信息熵特征
    "entropy": "信息熵特征", "fft_spectral_entropy": "信息熵特征",
    # 频域特征
    "fft_dc_ratio": "频域特征", "fft_low_freq_ratio": "频域特征", "fft_high_freq_ratio": "频域特征",
    "fft_mid_freq_ratio": "频域特征", "fft_dominant_freq_idx": "频域特征", "fft_dominant_power_ratio": "频域特征",
    "fft_spectral_flatness": "频域特征", "fft_spectral_centroid": "频域特征",
    # 时间间隔特征
    "dt_mean_ms": "时间间隔特征", "dt_std_ms": "时间间隔特征", "dt_max_ms": "时间间隔特征",
    "dt_min_ms": "时间间隔特征", "dt_cv": "时间间隔特征",
    # 异常指标特征
    "outlier_ratio_3std": "异常指标特征", "outlier_ratio_2std": "异常指标特征",
    "zero_ratio": "异常指标特征", "negative_ratio": "异常指标特征", "positive_ratio": "异常指标特征",
    "zero_crossing_rate": "异常指标特征",
}


def get_feature_category(feature_name: str) -> str:
    """获取特征所属类别"""
    # 处理交互特征
    if "_x_" in feature_name or "_div_" in feature_name:
        return "交互特征"
    return FEATURE_CATEGORY_MAPPING.get(feature_name, "其他特征")


# ============================================================
# 日志辅助函数
# ============================================================
def log_section(title: str) -> None:
    logging.info("")
    logging.info("=" * 60)
    logging.info(f"【{title}】")
    logging.info("=" * 60)


def log_subsection(title: str) -> None:
    logging.info(f"\n>>> {title}")
    logging.info("-" * 50)


def log_kv(key: str, value, indent: int = 2) -> None:
    prefix = " " * indent
    logging.info(f"{prefix}{key}: {value}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def setup_chinese_font() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "MS Gothic"]
    for name in candidates:
        try:
            font_path = font_manager.findfont(name, fallback_to_default=False)
        except Exception:
            continue
        if font_path:
            rcParams["font.sans-serif"] = [name]
            rcParams["axes.unicode_minus"] = False
            logging.info("  中文字体: %s", name)
            return
    logging.warning("  未找到合适的中文字体")


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train_binary.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def load_config(path: str | None) -> tuple[int, float, dict, dict, dict, str, str]:
    """加载配置文件"""
    seed = SEED
    test_size = 0.15
    binary_params = DEFAULT_BINARY_PARAMS.copy()
    sampling_config = DEFAULT_SAMPLING_CONFIG.copy()
    training_config = {
        "balance_strategy": "undersample_1to1",
        "use_focal_loss": True,
        "focal_gamma": 2.0,
        "use_feature_interactions": True,
        "target_dual_accuracy": 0.90,
        "use_pv_hash_bucket": False,
        "pv_hash_bucket_size": 4096,
        "pv_hash_salt": "pv_hash_v1",
    }
    data_dir = "test-datasets"
    output_dir = "artifacts/binary"
    
    if path and os.path.isfile(path):
        logging.info("  配置文件: %s", path)
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "seed" in cfg:
            seed = int(cfg["seed"])
        if "test_size" in cfg:
            test_size = float(cfg["test_size"])
        if "binary_params" in cfg and isinstance(cfg["binary_params"], dict):
            filtered = {k: v for k, v in cfg["binary_params"].items() if not k.startswith("_")}
            binary_params.update(filtered)
        if "sampling" in cfg and isinstance(cfg["sampling"], dict):
            filtered = {k: v for k, v in cfg["sampling"].items() if not k.startswith("_")}
            sampling_config.update(filtered)
        if "training" in cfg and isinstance(cfg["training"], dict):
            filtered = {k: v for k, v in cfg["training"].items() if not k.startswith("_")}
            training_config.update(filtered)
        if "data_dir" in cfg:
            data_dir = str(cfg["data_dir"])
        if "output_dir" in cfg:
            output_dir = str(cfg["output_dir"])
    else:
        logging.info("  配置文件: 使用默认配置")
    
    return seed, test_size, binary_params, sampling_config, training_config, data_dir, output_dir


def find_parquet_files(base_dir: str, verbose: bool = False) -> list[tuple[str, str]]:
    """
    扫描数据目录，返回 (label, file_path) 列表
    label: 'Normal' 或 'Abnormal'
    """
    results: list[tuple[str, str]] = []
    normal_dir = os.path.join(base_dir, "Normal")
    abnormal_dir = os.path.join(base_dir, "Abnormal")
    normal_count = 0
    abnormal_count = 0
    
    if os.path.isdir(normal_dir):
        for root, _, files in os.walk(normal_dir):
            for f in files:
                if f.lower().endswith(".parquet"):
                    full = os.path.join(root, f)
                    results.append(("Normal", full))
                    normal_count += 1
    
    if os.path.isdir(abnormal_dir):
        for root, _, files in os.walk(abnormal_dir):
            for f in files:
                if f.lower().endswith(".parquet"):
                    full = os.path.join(root, f)
                    results.append(("Abnormal", full))
                    abnormal_count += 1
    
    if verbose:
        logging.info("  正常样本文件: %d 个", normal_count)
        logging.info("  异常样本文件: %d 个", abnormal_count)
    
    return results


def stable_hash_bucket(text: str, bucket_size: int, salt: str) -> int:
    raw = f"{salt}|{text}".encode("utf-8", errors="ignore")
    digest = hashlib.md5(raw).digest()
    v = int.from_bytes(digest[:8], byteorder="little", signed=False)
    k = int(bucket_size)
    return int(v % k) if k > 0 else 0


def extract_features_from_series(values: pd.Series, epochs: pd.Series, use_fft_features: bool = True) -> dict:
    """从时序数据中提取统计特征（增强版）"""
    v = values.values.astype(np.float64)  # 使用 numpy 数组加速
    feats: dict[str, float] = {}
    
    # 移除 NaN
    v_valid = v[~np.isnan(v)]
    n = len(v_valid)
    feats["value_count"] = float(n)
    
    if n > 0:
        # 基础统计特征（使用 numpy 加速）
        mean_val = float(np.mean(v_valid))
        std_val = float(np.std(v_valid))
        min_val = float(np.min(v_valid))
        max_val = float(np.max(v_valid))
        median_val = float(np.median(v_valid))
        
        feats["value_mean"] = mean_val
        feats["value_std"] = std_val
        feats["value_min"] = min_val
        feats["value_max"] = max_val
        feats["value_median"] = median_val
        
        # 分位数（一次性计算）
        quantiles = np.percentile(v_valid, [1, 5, 10, 25, 75, 90, 95, 99])
        feats["value_q01"] = float(quantiles[0])
        feats["value_q05"] = float(quantiles[1])
        feats["value_q10"] = float(quantiles[2])
        feats["value_q25"] = float(quantiles[3])
        feats["value_q75"] = float(quantiles[4])
        feats["value_q90"] = float(quantiles[5])
        feats["value_q95"] = float(quantiles[6])
        feats["value_q99"] = float(quantiles[7])
        
        # 范围与离散度
        feats["value_range"] = float(max_val - min_val)
        feats["value_iqr"] = float(quantiles[4] - quantiles[3])
        feats["value_cv"] = float(std_val / (abs(mean_val) + 1e-10))
        
        # 能量特征
        feats["value_rms"] = float(np.sqrt(np.mean(v_valid ** 2)))
        
        # 异常值比例
        threshold_3std = abs(mean_val) + 3 * std_val
        threshold_2std = abs(mean_val) + 2 * std_val
        feats["outlier_ratio_3std"] = float(np.sum(np.abs(v_valid) > threshold_3std) / n)
        feats["outlier_ratio_2std"] = float(np.sum(np.abs(v_valid) > threshold_2std) / n)
        
        # 数值分布特征
        feats["zero_ratio"] = float(np.sum(v_valid == 0) / n)
        feats["negative_ratio"] = float(np.sum(v_valid < 0) / n)
        feats["positive_ratio"] = float(np.sum(v_valid > 0) / n)
        
        # ========== 新增特征: 分布形态 ==========
        # 偏度 (Skewness) - 衡量分布的不对称性
        if std_val > 1e-10:
            feats["skewness"] = float(np.mean(((v_valid - mean_val) / std_val) ** 3))
            # 峰度 (Kurtosis) - 衡量分布的尖锐程度
            feats["kurtosis"] = float(np.mean(((v_valid - mean_val) / std_val) ** 4) - 3)
        else:
            feats["skewness"] = 0.0
            feats["kurtosis"] = 0.0
        
        # ========== 新增特征: 信号变化 ==========
        # 过零率 (Zero Crossing Rate) - 信号变化频率
        if n > 1:
            zero_crossings = np.sum(np.diff(np.signbit(v_valid - mean_val)))
            feats["zero_crossing_rate"] = float(zero_crossings / (n - 1))
        else:
            feats["zero_crossing_rate"] = 0.0
        
        # 差分特征
        if n > 1:
            diffs = np.diff(v_valid)
            feats["diff_mean"] = float(np.mean(diffs))
            feats["diff_std"] = float(np.std(diffs))
            feats["diff_abs_mean"] = float(np.mean(np.abs(diffs)))
            feats["diff_max"] = float(np.max(diffs))
            feats["diff_min"] = float(np.min(diffs))
            
            # 新增: 差分绝对值最大值
            feats["diff_abs_max"] = float(np.max(np.abs(diffs)))
            
            # 新增: 符号变化次数 (方向变化)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            feats["direction_changes"] = float(sign_changes)
            feats["direction_change_rate"] = float(sign_changes / (n - 1))
            
            # ========== 新增特征: 二阶差分 (加速度) ==========
            if n > 2:
                diffs2 = np.diff(diffs)
                feats["diff2_mean"] = float(np.mean(diffs2))
                feats["diff2_std"] = float(np.std(diffs2))
                feats["diff2_abs_mean"] = float(np.mean(np.abs(diffs2)))
            else:
                feats["diff2_mean"] = 0.0
                feats["diff2_std"] = 0.0
                feats["diff2_abs_mean"] = 0.0
        else:
            feats["diff_mean"] = 0.0
            feats["diff_std"] = 0.0
            feats["diff_abs_mean"] = 0.0
            feats["diff_max"] = 0.0
            feats["diff_min"] = 0.0
            feats["diff_abs_max"] = 0.0
            feats["direction_changes"] = 0.0
            feats["direction_change_rate"] = 0.0
            feats["diff2_mean"] = 0.0
            feats["diff2_std"] = 0.0
            feats["diff2_abs_mean"] = 0.0
        
        # 首尾变化
        feats["value_first"] = float(v_valid[0])
        feats["value_last"] = float(v_valid[-1])
        feats["value_change"] = float(v_valid[-1] - v_valid[0])
        feats["value_change_ratio"] = float((v_valid[-1] - v_valid[0]) / (abs(v_valid[0]) + 1e-10))
        
        # ========== 新增特征: 分段统计 ==========
        if n >= 3:
            third = n // 3
            seg1 = v_valid[:third]
            seg2 = v_valid[third:2*third]
            seg3 = v_valid[2*third:]
            
            # 各段均值
            mean1, mean2, mean3 = np.mean(seg1), np.mean(seg2), np.mean(seg3)
            feats["seg1_mean"] = float(mean1)
            feats["seg2_mean"] = float(mean2)
            feats["seg3_mean"] = float(mean3)
            
            # 各段标准差
            feats["seg1_std"] = float(np.std(seg1))
            feats["seg2_std"] = float(np.std(seg2))
            feats["seg3_std"] = float(np.std(seg3))
            
            # 段间差异
            feats["seg_mean_diff_12"] = float(mean2 - mean1)
            feats["seg_mean_diff_23"] = float(mean3 - mean2)
            feats["seg_mean_diff_13"] = float(mean3 - mean1)
        else:
            feats["seg1_mean"] = mean_val
            feats["seg2_mean"] = mean_val
            feats["seg3_mean"] = mean_val
            feats["seg1_std"] = std_val
            feats["seg2_std"] = std_val
            feats["seg3_std"] = std_val
            feats["seg_mean_diff_12"] = 0.0
            feats["seg_mean_diff_23"] = 0.0
            feats["seg_mean_diff_13"] = 0.0
        
        # 趋势（前后半段均值差）
        if n > 1:
            mid = n // 2
            feats["trend_slope"] = float(np.mean(v_valid[mid:]) - np.mean(v_valid[:mid]))
        else:
            feats["trend_slope"] = 0.0
        
        # ========== 新增特征: 局部极值 ==========
        if n >= 3:
            # 局部最大值计数
            local_max = (v_valid[1:-1] > v_valid[:-2]) & (v_valid[1:-1] > v_valid[2:])
            local_min = (v_valid[1:-1] < v_valid[:-2]) & (v_valid[1:-1] < v_valid[2:])
            feats["local_max_count"] = float(np.sum(local_max))
            feats["local_min_count"] = float(np.sum(local_min))
            feats["peak_count"] = float(np.sum(local_max) + np.sum(local_min))
            feats["peak_rate"] = float(feats["peak_count"] / n)
        else:
            feats["local_max_count"] = 0.0
            feats["local_min_count"] = 0.0
            feats["peak_count"] = 0.0
            feats["peak_rate"] = 0.0
        
        # ========== 新增特征: 稳定性指标 ==========
        # 滑动窗口方差（衡量局部波动）
        if n >= 5:
            window_size = min(5, n // 3)
            rolling_vars = []
            for i in range(n - window_size + 1):
                rolling_vars.append(np.var(v_valid[i:i+window_size]))
            rolling_vars = np.array(rolling_vars)
            feats["rolling_var_mean"] = float(np.mean(rolling_vars))
            feats["rolling_var_std"] = float(np.std(rolling_vars))
            feats["rolling_var_max"] = float(np.max(rolling_vars))
        else:
            feats["rolling_var_mean"] = std_val ** 2
            feats["rolling_var_std"] = 0.0
            feats["rolling_var_max"] = std_val ** 2
        
        # ========== 新增特征: 自相关 ==========
        if n >= 10:
            # 滑9后的自相关
            centered = v_valid - mean_val
            autocorr_1 = np.corrcoef(centered[:-1], centered[1:])[0, 1] if n > 1 else 0.0
            feats["autocorr_lag1"] = float(autocorr_1) if not np.isnan(autocorr_1) else 0.0
            
            if n >= 5:
                autocorr_5 = np.corrcoef(centered[:-5], centered[5:])[0, 1]
                feats["autocorr_lag5"] = float(autocorr_5) if not np.isnan(autocorr_5) else 0.0
            else:
                feats["autocorr_lag5"] = 0.0
        else:
            feats["autocorr_lag1"] = 0.0
            feats["autocorr_lag5"] = 0.0
        
        # ========== 新增特征: 简单熵 (信息复杂度) ==========
        if n >= 10 and feats["value_range"] > 1e-10:
            # 将数据分档计算概率分布
            n_bins = min(10, n // 3)
            hist, _ = np.histogram(v_valid, bins=n_bins)
            hist = hist / n  # 归一化
            hist = hist[hist > 0]  # 移除零概率
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            feats["entropy"] = float(entropy)
        else:
            feats["entropy"] = 0.0
        
        # ========== 新增特征: 频域特征 (FFT) ==========
        if use_fft_features and n >= 8:
            try:
                # 计算 FFT
                fft_vals = np.fft.rfft(v_valid - mean_val)
                fft_magnitude = np.abs(fft_vals)
                fft_power = fft_magnitude ** 2
                
                # 总能量
                total_power = np.sum(fft_power) + 1e-10
                
                # 主频率成分
                if len(fft_magnitude) > 1:
                    # DC分量比例
                    feats["fft_dc_ratio"] = float(fft_power[0] / total_power)
                    # 低频能量比例 (前1/4)
                    low_freq_idx = max(1, len(fft_power) // 4)
                    feats["fft_low_freq_ratio"] = float(np.sum(fft_power[1:low_freq_idx]) / total_power)
                    # 高频能量比例 (后1/4)
                    high_freq_idx = len(fft_power) * 3 // 4
                    feats["fft_high_freq_ratio"] = float(np.sum(fft_power[high_freq_idx:]) / total_power)
                    # 中频能量比例
                    feats["fft_mid_freq_ratio"] = float(np.sum(fft_power[low_freq_idx:high_freq_idx]) / total_power)
                    # 主频率索引 (排除DC)
                    if len(fft_magnitude) > 2:
                        dominant_idx = np.argmax(fft_magnitude[1:]) + 1
                        feats["fft_dominant_freq_idx"] = float(dominant_idx / len(fft_magnitude))
                        feats["fft_dominant_power_ratio"] = float(fft_power[dominant_idx] / total_power)
                    else:
                        feats["fft_dominant_freq_idx"] = 0.0
                        feats["fft_dominant_power_ratio"] = 0.0
                    # 频谱熵
                    fft_norm = fft_power / total_power
                    fft_norm = fft_norm[fft_norm > 0]
                    feats["fft_spectral_entropy"] = float(-np.sum(fft_norm * np.log2(fft_norm + 1e-10)))
                    # 频谱平坦度
                    feats["fft_spectral_flatness"] = float(np.exp(np.mean(np.log(fft_magnitude[1:] + 1e-10))) / (np.mean(fft_magnitude[1:]) + 1e-10))
                    # 频谱质心
                    freq_indices = np.arange(len(fft_magnitude))
                    feats["fft_spectral_centroid"] = float(np.sum(freq_indices * fft_magnitude) / (np.sum(fft_magnitude) + 1e-10) / len(fft_magnitude))
                else:
                    feats["fft_dc_ratio"] = 1.0
                    feats["fft_low_freq_ratio"] = 0.0
                    feats["fft_high_freq_ratio"] = 0.0
                    feats["fft_mid_freq_ratio"] = 0.0
                    feats["fft_dominant_freq_idx"] = 0.0
                    feats["fft_dominant_power_ratio"] = 0.0
                    feats["fft_spectral_entropy"] = 0.0
                    feats["fft_spectral_flatness"] = 0.0
                    feats["fft_spectral_centroid"] = 0.0
            except Exception:
                feats["fft_dc_ratio"] = 0.0
                feats["fft_low_freq_ratio"] = 0.0
                feats["fft_high_freq_ratio"] = 0.0
                feats["fft_mid_freq_ratio"] = 0.0
                feats["fft_dominant_freq_idx"] = 0.0
                feats["fft_dominant_power_ratio"] = 0.0
                feats["fft_spectral_entropy"] = 0.0
                feats["fft_spectral_flatness"] = 0.0
                feats["fft_spectral_centroid"] = 0.0
        else:
            feats["fft_dc_ratio"] = 0.0
            feats["fft_low_freq_ratio"] = 0.0
            feats["fft_high_freq_ratio"] = 0.0
            feats["fft_mid_freq_ratio"] = 0.0
            feats["fft_dominant_freq_idx"] = 0.0
            feats["fft_dominant_power_ratio"] = 0.0
            feats["fft_spectral_entropy"] = 0.0
            feats["fft_spectral_flatness"] = 0.0
            feats["fft_spectral_centroid"] = 0.0
            
    else:
        # 空值默认 - 扩展版
        default_zero_feats = [
            "value_mean", "value_std", "value_min", "value_max", "value_median",
            "value_q01", "value_q05", "value_q10", "value_q25", "value_q75",
            "value_q90", "value_q95", "value_q99", "value_range", "value_iqr",
            "value_cv", "value_rms", "outlier_ratio_3std", "outlier_ratio_2std",
            "zero_ratio", "negative_ratio", "positive_ratio",
            "skewness", "kurtosis", "zero_crossing_rate",
            "diff_mean", "diff_std", "diff_abs_mean", "diff_max", "diff_min",
            "diff_abs_max", "direction_changes", "direction_change_rate",
            "diff2_mean", "diff2_std", "diff2_abs_mean",
            "value_first", "value_last", "value_change", "value_change_ratio",
            "seg1_mean", "seg2_mean", "seg3_mean", "seg1_std", "seg2_std", "seg3_std",
            "seg_mean_diff_12", "seg_mean_diff_23", "seg_mean_diff_13",
            "trend_slope", "local_max_count", "local_min_count", "peak_count", "peak_rate",
            "rolling_var_mean", "rolling_var_std", "rolling_var_max",
            "autocorr_lag1", "autocorr_lag5", "entropy",
            # FFT 特征
            "fft_dc_ratio", "fft_low_freq_ratio", "fft_high_freq_ratio", "fft_mid_freq_ratio",
            "fft_dominant_freq_idx", "fft_dominant_power_ratio",
            "fft_spectral_entropy", "fft_spectral_flatness", "fft_spectral_centroid",
        ]
        for feat_name in default_zero_feats:
            feats[feat_name] = 0.0
    
    # 时间间隔特征
    e = epochs.values.astype(np.int64)
    if len(e) > 1:
        delta = np.diff(e)
        feats["dt_mean_ms"] = float(np.mean(delta))
        feats["dt_std_ms"] = float(np.std(delta))
        feats["dt_max_ms"] = float(np.max(delta))
        feats["dt_min_ms"] = float(np.min(delta))
        feats["dt_cv"] = float(np.std(delta) / (np.mean(delta) + 1e-10))
    else:
        feats["dt_mean_ms"] = 0.0
        feats["dt_std_ms"] = 0.0
        feats["dt_max_ms"] = 0.0
        feats["dt_min_ms"] = 0.0
        feats["dt_cv"] = 0.0
    
    # 清理无效值
    for key in feats:
        if np.isnan(feats[key]) or np.isinf(feats[key]):
            feats[key] = 0.0
    
    return feats


def generate_window_records(
    df: pd.DataFrame,
    label: str,
    path: str,
    sampling_config: dict,
    use_fft_features: bool = True,
) -> list[dict]:
    """生成滑动窗口特征记录"""
    if "epoch_ms" not in df.columns or "value" not in df.columns:
        return []
    
    df_sorted = df
    epochs = df_sorted["epoch_ms"].astype("int64").to_numpy()
    if epochs.size == 0:
        return []
    
    window_sec = float(sampling_config.get("window_sec", 120.0))
    step_sec = float(sampling_config.get("step_sec", 30.0))
    min_points = int(sampling_config.get("min_points", 10))
    max_windows = int(sampling_config.get("max_windows_per_file", 100))
    
    window_ms = int(window_sec * 1000.0)
    step_ms = int(step_sec * 1000.0)
    start = int(epochs[0])
    end_max = int(epochs[-1]) + 1
    n = epochs.size
    idx_start = 0
    idx_end = 0
    records: list[dict] = []
    
    while start < end_max:
        if max_windows > 0 and len(records) >= max_windows:
            break
        
        win_end = start + window_ms
        while idx_start < n and epochs[idx_start] < start:
            idx_start += 1
        idx_end = idx_start
        while idx_end < n and epochs[idx_end] < win_end:
            idx_end += 1
        
        if idx_end - idx_start >= min_points:
            sub = df_sorted.iloc[idx_start:idx_end]
            feats = extract_features_from_series(sub["value"], sub["epoch_ms"], use_fft_features=use_fft_features)
            feats["file_path"] = path
            if "pv_name" in sub.columns:
                try:
                    feats["pv_name"] = str(sub["pv_name"].iloc[0])
                except Exception:
                    feats["pv_name"] = ""
            else:
                feats["pv_name"] = ""
            feats["window_start_epoch_ms"] = float(start)
            feats["window_end_epoch_ms"] = float(win_end)
            feats["label"] = 0 if label == "Normal" else 1
            records.append(feats)
        
        start += step_ms
    
    return records


def build_feature_table(
    base_dir: str,
    sampling_config: dict,
    training_config: dict,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
    """构建特征表"""
    log_subsection("数据扫描")
    logging.info("  数据目录: %s", base_dir)
    paths = find_parquet_files(base_dir, verbose=True)
    
    if not paths:
        raise RuntimeError("未找到任何 Parquet 文件")
    
    mode = str(sampling_config.get("mode", "window"))
    num_workers = int(sampling_config.get("num_workers", 0))
    
    log_subsection("特征构建")
    logging.info("  采样模式: %s", mode)
    if mode == "window":
        logging.info("  窗口长度: %.1f 秒", float(sampling_config.get("window_sec", 120.0)))
        logging.info("  滑动步长: %.1f 秒", float(sampling_config.get("step_sec", 30.0)))
        logging.info("  最小点数: %d", int(sampling_config.get("min_points", 10)))
        max_win = int(sampling_config.get("max_windows_per_file", 100))
        logging.info("  每文件最多窗口: %s", max_win if max_win > 0 else "无限制")
    logging.info("  并行线程: %s", num_workers if num_workers > 1 else "单线程")
    
    total_files = len(paths)
    logging.info("  待处理文件: %d 个", total_files)
    
    start_time = time.time()
    use_fft_features = bool(training_config.get("use_fft_features", True))
    records: list[dict] = []
    processed_files = [0]
    skipped_files = [0]
    total_samples = [0]
    last_log_time = [start_time]
    log_interval = 10
    lock = threading.Lock()

    def process_one(item: tuple[str, str]) -> list[dict]:
        label, path = item
        result = []
        try:
            df = pd.read_parquet(path, columns=["epoch_ms", "value", "pv_name"])
        except Exception:
            try:
                df = pd.read_parquet(path, columns=["epoch_ms", "value"])
            except Exception:
                with lock:
                    skipped_files[0] += 1
                return []
        
        if "value" not in df.columns or "epoch_ms" not in df.columns:
            with lock:
                skipped_files[0] += 1
            return []
        
        if mode == "window":
            result = generate_window_records(df, label, path, sampling_config, use_fft_features=use_fft_features)
        else:
            feats = extract_features_from_series(df["value"], df["epoch_ms"], use_fft_features=use_fft_features)
            feats["file_path"] = path
            if "pv_name" in df.columns:
                try:
                    feats["pv_name"] = str(df["pv_name"].iloc[0])
                except Exception:
                    feats["pv_name"] = ""
            else:
                feats["pv_name"] = ""
            feats["label"] = 0 if label == "Normal" else 1
            result = [feats]
        
        with lock:
            processed_files[0] += 1
            total_samples[0] += len(result)
            current_time = time.time()
            if current_time - last_log_time[0] >= log_interval:
                elapsed = current_time - start_time
                done = processed_files[0] + skipped_files[0]
                pct = done / total_files * 100
                speed = done / elapsed if elapsed > 0 else 0
                remaining = (total_files - done) / speed if speed > 0 else 0
                logging.info(
                    "  [进度] %d/%d (%.1f%%) | 已生成样本: %d | 耗时: %.0f秒 | 预计剩余: %.0f秒",
                    done, total_files, pct, total_samples[0], elapsed, remaining
                )
                last_log_time[0] = current_time
        
        return result

    if num_workers and num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_one, item) for item in paths]
            for fut in as_completed(futures):
                try:
                    recs = fut.result()
                except Exception:
                    with lock:
                        skipped_files[0] += 1
                    continue
                records.extend(recs)
    else:
        for item in paths:
            records.extend(process_one(item))
    
    elapsed = time.time() - start_time
    if not records:
        raise RuntimeError("未能从数据目录中提取任何有效样本")
    
    import gc
    table = pd.DataFrame(records)
    del records
    gc.collect()

    if bool(training_config.get("use_pv_hash_bucket", False)) and "pv_name" in table.columns:
        bucket_size = int(training_config.get("pv_hash_bucket_size", 2048))
        salt = str(training_config.get("pv_hash_salt", "pv_hash_v1"))
        pv_raw = table["pv_name"].fillna("").astype(str)
        pv_unique = pv_raw.unique()
        pv_map = {name: stable_hash_bucket(name, bucket_size=bucket_size, salt=salt) for name in pv_unique}
        table["pv_bucket"] = pv_raw.map(pv_map).astype(np.int32).astype("category")
        logging.info("  PV Hash Bucket: 开启 | K=%d | 唯一PV=%d", bucket_size, len(pv_unique))

    meta_cols = {"file_path", "pv_name", "window_start_epoch_ms", "window_end_epoch_ms", "label"}
    numeric_cols = table.select_dtypes(include=[np.number]).columns.tolist()
    base_cols = [c for c in numeric_cols if c not in meta_cols]
    if base_cols:
        sort_cols = ["file_path"]
        group_cols = ["file_path"]
        if "pv_name" in table.columns:
            sort_cols = ["file_path", "pv_name"]
            group_cols = ["file_path", "pv_name"]
        if "window_start_epoch_ms" in table.columns:
            sort_cols = sort_cols + ["window_start_epoch_ms"]
        table = table.sort_values(sort_cols, kind="mergesort")
        delta_df = table.groupby(group_cols, sort=False)[base_cols].diff().fillna(0.0)
        delta_df.columns = [f"delta_{c}" for c in base_cols]
        table = pd.concat([table, delta_df.astype(np.float32)], axis=1)
    
    feature_cols = [
        c for c in table.columns
        if c not in ["file_path", "pv_name", "window_start_epoch_ms", "window_end_epoch_ms", "label"]
    ]
    
    X = table[feature_cols].copy()
    if "pv_bucket" in X.columns:
        num_cols = [c for c in X.columns if c != "pv_bucket"]
        if num_cols:
            X[num_cols] = X[num_cols].astype(np.float32)
    else:
        X = X.astype(np.float32)
    y = table["label"].astype(np.int8)
    file_paths = table["file_path"].reset_index(drop=True)
    
    del table
    gc.collect()
    
    normal_samples = int((y == 0).sum())
    abnormal_samples = int((y == 1).sum())
    logging.info("")
    logging.info("  特征构建完成（耗时: %.2f 秒）", elapsed)
    logging.info("  处理文件: %d 个，跳过: %d 个", processed_files[0], skipped_files[0])
    logging.info("  生成样本: %d 条（正常: %d，异常: %d）", X.shape[0], normal_samples, abnormal_samples)
    logging.info("  特征维度: %d", len(feature_cols))
    
    return X, y, feature_cols, file_paths


def add_feature_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """添加特征交互项以提升模型区分能力（增强版）"""
    X_new = X.copy()
    cols = X.columns.tolist()
    
    # ========== 1. 关键特征交互 ==========
    key_features = ['value_mean', 'value_std', 'value_cv', 'diff_std', 'entropy', 'skewness', 'kurtosis']
    available_keys = [f for f in key_features if f in cols]
    
    for i, f1 in enumerate(available_keys):
        for f2 in available_keys[i+1:]:
            X_new[f"{f1}_x_{f2}"] = X[f1] * X[f2]
            X_new[f"{f1}_div_{f2}"] = X[f1] / (X[f2].abs() + 1e-10)
    
    # ========== 2. 稳定性特征 ==========
    if 'value_std' in cols and 'value_mean' in cols:
        X_new['std_mean_ratio'] = X['value_std'] / (X['value_mean'].abs() + 1e-10)
        X_new['mean_abs'] = X['value_mean'].abs()
    
    if 'diff_std' in cols and 'value_std' in cols:
        X_new['diff_value_std_ratio'] = X['diff_std'] / (X['value_std'] + 1e-10)
    
    if 'value_range' in cols and 'value_iqr' in cols:
        X_new['range_iqr_ratio'] = X['value_range'] / (X['value_iqr'] + 1e-10)
    
    # ========== 3. 趋势特征 ==========
    if 'seg1_mean' in cols and 'seg3_mean' in cols:
        X_new['seg_trend'] = X['seg3_mean'] - X['seg1_mean']
        X_new['seg_trend_abs'] = (X['seg3_mean'] - X['seg1_mean']).abs()
    
    if 'seg1_mean' in cols and 'seg2_mean' in cols and 'seg3_mean' in cols:
        X_new['seg_monotonic'] = ((X['seg2_mean'] - X['seg1_mean']) * (X['seg3_mean'] - X['seg2_mean']))
        X_new['seg_curvature'] = X['seg1_mean'] - 2 * X['seg2_mean'] + X['seg3_mean']
    
    # ========== 4. 复杂度/波动性特征 ==========
    if 'autocorr_lag1' in cols and 'entropy' in cols:
        X_new['autocorr_entropy'] = X['autocorr_lag1'] * X['entropy']
    
    if 'peak_rate' in cols and 'direction_change_rate' in cols:
        X_new['volatility_index'] = X['peak_rate'] + X['direction_change_rate']
        X_new['volatility_product'] = X['peak_rate'] * X['direction_change_rate']
    
    if 'rolling_var_mean' in cols and 'value_std' in cols:
        X_new['rolling_stability'] = X['rolling_var_mean'] / (X['value_std'] ** 2 + 1e-10)
    
    if 'rolling_var_std' in cols and 'rolling_var_mean' in cols:
        X_new['rolling_var_cv'] = X['rolling_var_std'] / (X['rolling_var_mean'] + 1e-10)
    
    # ========== 5. 异常检测特征 ==========
    if 'outlier_ratio_3std' in cols and 'outlier_ratio_2std' in cols:
        X_new['outlier_severity'] = X['outlier_ratio_3std'] / (X['outlier_ratio_2std'] + 1e-10)
    
    if 'value_max' in cols and 'value_q95' in cols:
        X_new['extreme_ratio_high'] = (X['value_max'] - X['value_q95']) / (X['value_range'] + 1e-10)
    
    if 'value_min' in cols and 'value_q05' in cols:
        X_new['extreme_ratio_low'] = (X['value_q05'] - X['value_min']) / (X['value_range'] + 1e-10)
    
    # ========== 6. 差分特征组合 ==========
    if 'diff_abs_mean' in cols and 'diff_std' in cols:
        X_new['diff_consistency'] = X['diff_abs_mean'] / (X['diff_std'] + 1e-10)
    
    if 'diff_abs_max' in cols and 'diff_abs_mean' in cols:
        X_new['diff_spike_ratio'] = X['diff_abs_max'] / (X['diff_abs_mean'] + 1e-10)
    
    if 'diff2_std' in cols and 'diff_std' in cols:
        X_new['acceleration_ratio'] = X['diff2_std'] / (X['diff_std'] + 1e-10)
    
    # ========== 7. 时间间隔特征 ==========
    if 'dt_cv' in cols and 'dt_std_ms' in cols:
        X_new['dt_irregularity'] = X['dt_cv'] * X['dt_std_ms']
    
    if 'dt_max_ms' in cols and 'dt_mean_ms' in cols:
        X_new['dt_max_ratio'] = X['dt_max_ms'] / (X['dt_mean_ms'] + 1e-10)
    
    # ========== 8. 分布形态特征 ==========
    if 'skewness' in cols and 'kurtosis' in cols:
        X_new['distribution_score'] = X['skewness'].abs() + X['kurtosis'].abs()
        X_new['skew_kurt_product'] = X['skewness'] * X['kurtosis']
    
    if 'zero_ratio' in cols and 'negative_ratio' in cols and 'positive_ratio' in cols:
        X_new['sign_balance'] = (X['positive_ratio'] - X['negative_ratio']).abs()
        X_new['non_zero_ratio'] = 1 - X['zero_ratio']
    
    # ========== 9. 统计矩特征 ==========
    if 'value_q75' in cols and 'value_q25' in cols and 'value_median' in cols:
        X_new['quartile_skew'] = ((X['value_q75'] - X['value_median']) - (X['value_median'] - X['value_q25'])) / (X['value_iqr'] + 1e-10)
    
    if 'value_q90' in cols and 'value_q10' in cols:
        X_new['central_80_range'] = X['value_q90'] - X['value_q10']
    
    if 'value_q99' in cols and 'value_q01' in cols:
        X_new['central_98_range'] = X['value_q99'] - X['value_q01']
    
    # ========== 10. 综合异常分数 ==========
    # 创建一个综合的异常指标
    anomaly_indicators = []
    if 'outlier_ratio_3std' in cols:
        anomaly_indicators.append(X['outlier_ratio_3std'])
    if 'value_cv' in cols:
        anomaly_indicators.append(X['value_cv'].clip(0, 10) / 10)  # 归一化
    if 'diff_abs_max' in cols and 'value_range' in cols:
        spike_score = X['diff_abs_max'] / (X['value_range'] + 1e-10)
        anomaly_indicators.append(spike_score.clip(0, 1))
    if 'entropy' in cols:
        entropy_norm = X['entropy'] / 3.5  # 归一化 (最大约 log2(10) = 3.32)
        anomaly_indicators.append(entropy_norm.clip(0, 1))
    
    if anomaly_indicators:
        X_new['composite_anomaly_score'] = sum(anomaly_indicators) / len(anomaly_indicators)
    
    # 清理无效值
    num_cols = X_new.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X_new[num_cols] = X_new[num_cols].replace([np.inf, -np.inf], 0).fillna(0)
    
    return X_new


# 集成学习模型封装类（模块级别，支持 pickle）
class EnsembleLGBMClassifier:
    """LightGBM 集成学习分类器"""
    def __init__(self, models, feature_names=None):
        self.models = models
        self.feature_names_in_ = feature_names or []
    
    def predict_proba(self, X):
        probas = [m.predict_proba(X) for m in self.models]
        return np.mean(probas, axis=0)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    @property
    def feature_importances_(self):
        return np.mean([m.feature_importances_ for m in self.models], axis=0)


# Focal Loss 模型封装类（模块级别，支持 pickle）
class FocalLossLGBMClassifier:
    """Focal Loss LightGBM 分类器"""
    def __init__(self, booster, feature_names=None):
        self.booster = booster
        self.feature_names_in_ = feature_names or []
    
    def predict_proba(self, X):
        raw_pred = self.booster.predict(X)
        prob_1 = 1.0 / (1.0 + np.exp(-raw_pred))
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    @property
    def feature_importances_(self):
        return self.booster.feature_importance()


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    seed: int,
    params: dict,
    training_config: dict,
    file_paths: pd.Series,
) -> tuple[lgb.LGBMClassifier, dict, pd.DataFrame, pd.Series, pd.Series]:
    """训练二分类模型"""
    log_subsection("二分类模型训练 (Normal vs Abnormal)")
    logging.info("  样本总数: %d", X.shape[0])
    logging.info("  验证集比例: %.1f%%", test_size * 100)
    logging.info("  随机种子: %d", seed)
    
    # 获取训练配置
    balance_strategy = training_config.get("balance_strategy", "undersample_1to1")
    use_focal_loss = training_config.get("use_focal_loss", True)
    focal_gamma = training_config.get("focal_gamma", 2.0)
    use_feature_interactions = training_config.get("use_feature_interactions", True)
    target_accuracy = training_config.get("target_dual_accuracy", 0.90)
    use_ensemble = training_config.get("use_ensemble", False)
    ensemble_models = training_config.get("ensemble_models", 5)
    normal_class_weight = training_config.get("normal_class_weight", 1.0)
    smote_ratio = training_config.get("smote_ratio", 0.8)
    # 硬样本挖掘配置
    use_hard_mining = training_config.get("use_hard_mining", False)
    hard_mining_rounds = training_config.get("hard_mining_rounds", 2)
    hard_margin = training_config.get("hard_margin", 0.3)  # 边界样本阈值

    pv_hash_bundle = None
    if bool(training_config.get("use_pv_hash_bucket", False)) and "pv_bucket" in X.columns:
        pv_hash_bundle = {
            "type": "hash_bucket",
            "feature": "pv_bucket",
            "bucket_size": int(training_config.get("pv_hash_bucket_size", 2048)),
            "salt": str(training_config.get("pv_hash_salt", "pv_hash_v1")),
        }
        logging.info("  PV Hash Bucket: %s", f"K={pv_hash_bundle['bucket_size']} feature=pv_bucket")
    
    logging.info("  平衡策略: %s", balance_strategy)
    logging.info("  Focal Loss: %s (gamma=%.1f)", use_focal_loss, focal_gamma)
    if use_ensemble:
        logging.info("  集成学习: %d 个模型", ensemble_models)
    if normal_class_weight != 1.0:
        logging.info("  正常类权重: %.1fx", normal_class_weight)
    if use_hard_mining:
        logging.info("  硬样本挖掘: %d 轮, margin=%.2f", hard_mining_rounds, hard_margin)
    
    # 添加特征交互
    if use_feature_interactions:
        logging.info("  添加特征交互项...")
        original_cols = X.shape[1]
        X = add_feature_interactions(X)
        logging.info("  特征维度: %d -> %d (+%d 交互特征)", 
                    original_cols, X.shape[1], X.shape[1] - original_cols)
    
    X_train, X_valid, y_train, y_valid, paths_train, paths_valid = train_test_split(
        X, y, file_paths, test_size=test_size, random_state=seed, stratify=y,
    )
    
    # 计算类别分布
    train_normal = int((y_train == 0).sum())
    train_abnormal = int((y_train == 1).sum())
    logging.info("  训练集: %d 条（正常: %d，异常: %d）", X_train.shape[0], train_normal, train_abnormal)
    logging.info("  验证集: %d 条", X_valid.shape[0])
    
    # 根据平衡策略处理数据
    if balance_strategy == "undersample_1to1":
        # 1:1 下采样 - 让正常和异常样本数量相等
        logging.info("  应用 1:1 下采样...")
        target_count = train_normal  # 异常类采样到与正常类相同
        
        normal_idx = y_train[y_train == 0].index
        abnormal_idx = y_train[y_train == 1].index
        
        np.random.seed(seed)
        sampled_abnormal_idx = np.random.choice(abnormal_idx, size=target_count, replace=False)
        
        balanced_idx = np.concatenate([normal_idx.values, sampled_abnormal_idx])
        np.random.shuffle(balanced_idx)
        
        X_train = X_train.loc[balanced_idx].reset_index(drop=True)
        y_train = y_train.loc[balanced_idx].reset_index(drop=True)
        
        train_normal_new = int((y_train == 0).sum())
        train_abnormal_new = int((y_train == 1).sum())
        logging.info("  下采样后: %d 条（正常: %d，异常: %d，比例 1:1）", 
                    X_train.shape[0], train_normal_new, train_abnormal_new)
    
    elif balance_strategy == "undersample_1to2":
        # 1:2 下采样
        logging.info("  应用 1:2 下采样...")
        target_abnormal = int(train_normal * 2)
        
        normal_idx = y_train[y_train == 0].index
        abnormal_idx = y_train[y_train == 1].index
        
        np.random.seed(seed)
        sampled_abnormal_idx = np.random.choice(abnormal_idx, size=target_abnormal, replace=False)
        
        balanced_idx = np.concatenate([normal_idx.values, sampled_abnormal_idx])
        np.random.shuffle(balanced_idx)
        
        X_train = X_train.loc[balanced_idx].reset_index(drop=True)
        y_train = y_train.loc[balanced_idx].reset_index(drop=True)
        
        train_normal_new = int((y_train == 0).sum())
        train_abnormal_new = int((y_train == 1).sum())
        logging.info("  下采样后: %d 条（正常: %d，异常: %d，比例 1:%.1f）", 
                    X_train.shape[0], train_normal_new, train_abnormal_new,
                    train_abnormal_new / train_normal_new if train_normal_new > 0 else 0)
    
    elif balance_strategy == "class_weight":
        # 使用类别权重，不采样
        logging.info("  使用类别权重处理不平衡...")
        # scale_pos_weight 将在后面计算
    
    elif balance_strategy == "smote" and IMBLEARN_AVAILABLE:
        logging.info("  应用 SMOTETomek 混合采样...")
        try:
            smote_tomek = SMOTETomek(
                smote=SMOTE(sampling_strategy=0.8, random_state=seed, k_neighbors=5),
                random_state=seed,
            )
            X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
            
            resampled_normal = int((y_train_resampled == 0).sum())
            resampled_abnormal = int((y_train_resampled == 1).sum())
            logging.info("  采样后: %d 条（正常: %d，异常: %d）", 
                        len(y_train_resampled), resampled_normal, resampled_abnormal)
            
            X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
            y_train = pd.Series(y_train_resampled)
        except Exception as e:
            logging.warning("  SMOTETomek 采样失败: %s，使用原始数据", str(e))
    
    elif balance_strategy == "smote_oversample" and IMBLEARN_AVAILABLE:
        # SMOTE 过采样正常类（少数类）
        logging.info("  应用 SMOTE 过采样正常类 (ratio=%.2f)...", smote_ratio)
        try:
            # 先对异常类进行下采样，降低内存压力
            target_abnormal = int(train_normal * 3)  # 先将异常类减少到正常类的3倍
            if train_abnormal > target_abnormal:
                logging.info("  先对异常类下采样: %d -> %d", train_abnormal, target_abnormal)
                normal_idx = y_train[y_train == 0].index
                abnormal_idx = y_train[y_train == 1].index
                np.random.seed(seed)
                sampled_abnormal_idx = np.random.choice(abnormal_idx, size=target_abnormal, replace=False)
                balanced_idx = np.concatenate([normal_idx.values, sampled_abnormal_idx])
                np.random.shuffle(balanced_idx)
                X_train = X_train.loc[balanced_idx].reset_index(drop=True)
                y_train = y_train.loc[balanced_idx].reset_index(drop=True)
                train_normal = int((y_train == 0).sum())
                train_abnormal = int((y_train == 1).sum())
            
            # 然后 SMOTE 过采样正常类
            smote = SMOTE(sampling_strategy=smote_ratio, random_state=seed, k_neighbors=min(5, train_normal-1))
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            resampled_normal = int((y_train_resampled == 0).sum())
            resampled_abnormal = int((y_train_resampled == 1).sum())
            logging.info("  SMOTE后: %d 条（正常: %d，异常: %d，比例 1:%.2f）", 
                        len(y_train_resampled), resampled_normal, resampled_abnormal,
                        resampled_abnormal / resampled_normal if resampled_normal > 0 else 0)
            
            X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
            y_train = pd.Series(y_train_resampled)
        except Exception as e:
            logging.warning("  SMOTE 过采样失败: %s，回退到1:1下采样", str(e))
            # 回退到 1:1 下采样
            target_count = train_normal
            normal_idx = y_train[y_train == 0].index
            abnormal_idx = y_train[y_train == 1].index
            np.random.seed(seed)
            sampled_abnormal_idx = np.random.choice(abnormal_idx, size=min(target_count, len(abnormal_idx)), replace=False)
            balanced_idx = np.concatenate([normal_idx.values, sampled_abnormal_idx])
            np.random.shuffle(balanced_idx)
            X_train = X_train.loc[balanced_idx].reset_index(drop=True)
            y_train = y_train.loc[balanced_idx].reset_index(drop=True)
    
    # 重新计算类别比例以设置 scale_pos_weight
    final_normal = int((y_train == 0).sum())
    final_abnormal = int((y_train == 1).sum())
    scale_pos_weight = final_normal / final_abnormal if final_abnormal > 0 else 1.0
    
    merged_params = params.copy()
    if "random_state" not in merged_params:
        merged_params["random_state"] = seed
    if "n_jobs" not in merged_params:
        merged_params["n_jobs"] = -1

    boosting_type = str(merged_params.get("boosting_type", "gbdt")).lower()
    if boosting_type == "goss":
        if merged_params.get("subsample", 1.0) != 1.0:
            merged_params["subsample"] = 1.0
        if merged_params.get("subsample_freq", 0) != 0:
            merged_params["subsample_freq"] = 0
        if merged_params.get("bagging_fraction", 1.0) != 1.0:
            merged_params["bagging_fraction"] = 1.0
        if merged_params.get("bagging_freq", 0) != 0:
            merged_params["bagging_freq"] = 0
        logging.info("  GOSS 已启用：强制 subsample=1.0, subsample_freq=0")
    
    # 设置 scale_pos_weight（只在非平衡数据时有效）
    if balance_strategy == "class_weight":
        merged_params["scale_pos_weight"] = scale_pos_weight
        logging.info("  scale_pos_weight: %.4f", scale_pos_weight)
    
    logging.info("  模型参数:")
    for k, v in merged_params.items():
        if k not in ["objective", "random_state", "n_jobs"]:
            logging.info("    - %s: %s", k, v)
    
    logging.info("  开始训练...")
    start_time = time.time()
    
    # 计算样本权重
    sample_weights = None
    if normal_class_weight != 1.0:
        sample_weights = np.ones(len(y_train))
        sample_weights[y_train == 0] = normal_class_weight
        logging.info("  应用正常类权重: %.1fx", normal_class_weight)
    
    # 创建模型
    model = lgb.LGBMClassifier(**merged_params)
    
    # 集成学习模式
    if use_ensemble and not use_focal_loss:
        logging.info("  使用集成学习 (%d 个模型)...", ensemble_models)
        models = []
        for i in range(ensemble_models):
            # 每个模型使用不同的随机种子和采样
            model_params = merged_params.copy()
            model_params["random_state"] = seed + i
            model_params["colsample_bytree"] = max(0.6, merged_params.get("colsample_bytree", 0.8) - 0.05 * i)
            
            m = lgb.LGBMClassifier(**model_params)
            
            # bootstrap 采样
            np.random.seed(seed + i)
            bootstrap_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_boot = X_train.iloc[bootstrap_idx].reset_index(drop=True)
            y_boot = y_train.iloc[bootstrap_idx].reset_index(drop=True)
            w_boot = sample_weights[bootstrap_idx] if sample_weights is not None else None
            
            m.fit(
                X_boot, y_boot,
                sample_weight=w_boot,
                eval_set=[(X_valid, y_valid)],
                eval_metric="binary_logloss",
            )
            models.append(m)
            logging.info("    模型 %d/%d 完成", i + 1, ensemble_models)
        
        # 使用模块级别的 EnsembleLGBMClassifier
        model = EnsembleLGBMClassifier(models, feature_names=X_train.columns.tolist())
    
    # 训练模型 - 使用 Focal Loss 或普通损失
    elif use_focal_loss:
        # Focal Loss 自定义目标函数
        def focal_loss_lgb(y_pred, dtrain):
            """Focal Loss for LightGBM"""
            y_true = dtrain.get_label()
            gamma = focal_gamma
            # sigmoid 转换
            p = 1.0 / (1.0 + np.exp(-y_pred))
            # 梯度计算
            grad = p - y_true
            hess = p * (1 - p)
            # Focal 权重调整
            focal_weight = np.where(y_true == 1, (1 - p) ** gamma, p ** gamma)
            grad = grad * focal_weight
            hess = hess * focal_weight + 1e-10  # 避免除零
            return grad, hess
        
        logging.info("  使用 Focal Loss (gamma=%.1f)...", focal_gamma)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        
        # LightGBM 4.x: 通过 params['objective'] 传递自定义目标函数
        lgb_params = merged_params.copy()
        lgb_params.pop("n_estimators", None)
        lgb_params.pop("random_state", None)
        lgb_params.pop("n_jobs", None)
        lgb_params.pop("class_weight", None)
        lgb_params["objective"] = focal_loss_lgb
        lgb_params["metric"] = "binary_logloss"
        lgb_params["verbose"] = -1
        lgb_params["seed"] = seed
        lgb_params["num_threads"] = -1
        
        num_boost_round = merged_params.get("n_estimators", 1000)
        
        try:
            booster = lgb.train(
                params=lgb_params,
                train_set=train_data,
                num_boost_round=num_boost_round,
                valid_sets=[valid_data],
                valid_names=["valid"],
                callbacks=[lgb.log_evaluation(period=0)],
            )
            
            # 使用模块级别的 FocalLossLGBMClassifier
            model = FocalLossLGBMClassifier(booster, feature_names=X_train.columns.tolist())
        
        except Exception as e:
            # 降级到标准训练
            logging.warning("  Focal Loss 不可用 (%s)，使用标准二分类...", str(e))
            model = lgb.LGBMClassifier(**merged_params)
            model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_valid, y_valid)],
                eval_metric="binary_logloss",
            )
    else:
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_valid, y_valid)],
            eval_metric="binary_logloss",
        )
    
    # ========== 硬样本挖掘 (Hard Example Mining) ==========
    if use_hard_mining and not use_focal_loss:
        logging.info("")
        logging.info("  [硬样本挖掘] 开始 %d 轮迭代训练...", hard_mining_rounds)
        
        # 保存原始训练数据
        X_train_original = X_train.copy()
        y_train_original = y_train.copy()
        
        for mining_round in range(hard_mining_rounds):
            logging.info("    第 %d/%d 轮硬样本挖掘...", mining_round + 1, hard_mining_rounds)
            
            # 用当前模型预测训练集
            train_proba = model.predict_proba(X_train_original)[:, 1]
            train_pred = (train_proba >= 0.5).astype(int)
            
            # 识别硬样本:
            # 1. 预测错误的样本
            wrong_mask = (train_pred != y_train_original.values)
            # 2. 置信度低的边界样本 (0.5-margin, 0.5+margin)
            boundary_mask = (train_proba >= 0.5 - hard_margin) & (train_proba <= 0.5 + hard_margin)
            # 3. 正常类被错分为异常的样本 (特别重要)
            fn_normal = (y_train_original.values == 0) & (train_pred == 1)
            
            # 组合硬样本
            hard_mask = wrong_mask | boundary_mask | fn_normal
            
            n_hard = hard_mask.sum()
            n_wrong = wrong_mask.sum()
            n_boundary = boundary_mask.sum()
            logging.info("      硬样本: %d 条 (错分: %d, 边界: %d)", n_hard, n_wrong, n_boundary)
            
            if n_hard < 100:
                logging.info("      硬样本太少，跳过本轮")
                continue
            
            # 硬样本
            X_hard = X_train_original[hard_mask].reset_index(drop=True)
            y_hard = y_train_original[hard_mask].reset_index(drop=True)
            
            # 随机采样一部分简单样本保持多样性
            easy_mask = ~hard_mask
            n_easy_sample = min(n_hard, easy_mask.sum())  # 硬样本和简单样本1:1
            if n_easy_sample > 0:
                easy_indices = np.where(easy_mask)[0]
                np.random.seed(seed + mining_round)
                sampled_easy = np.random.choice(easy_indices, size=n_easy_sample, replace=False)
                X_easy = X_train_original.iloc[sampled_easy].reset_index(drop=True)
                y_easy = y_train_original.iloc[sampled_easy].reset_index(drop=True)
                
                # 合并硬样本和简单样本
                X_combined = pd.concat([X_hard, X_easy], ignore_index=True)
                y_combined = pd.concat([y_hard, y_easy], ignore_index=True)
            else:
                X_combined = X_hard
                y_combined = y_hard
            
            # 计算硬样本的样本权重 (硬样本权重更高，不再额外叠加正常类权重)
            combined_weights = np.ones(len(y_combined))
            # 硬样本部分给更高权重
            combined_weights[:len(y_hard)] = 2.0
            # 不再额外叠加正常类权重，避免过度偏向正常类
            
            logging.info("      训练样本: %d 条 (硬: %d, 简单: %d)", 
                        len(y_combined), len(y_hard), n_easy_sample)
            
            # 重新训练模型
            if use_ensemble:
                # 集成学习模式下的硬样本训练
                models_new = []
                for i in range(ensemble_models):
                    model_params = merged_params.copy()
                    model_params["random_state"] = seed + mining_round * 100 + i
                    model_params["n_estimators"] = max(500, merged_params.get("n_estimators", 1000) // 2)
                    
                    m = lgb.LGBMClassifier(**model_params)
                    
                    # bootstrap
                    np.random.seed(seed + mining_round * 100 + i)
                    boot_idx = np.random.choice(len(X_combined), size=len(X_combined), replace=True)
                    m.fit(
                        X_combined.iloc[boot_idx], y_combined.iloc[boot_idx],
                        sample_weight=combined_weights[boot_idx],
                        eval_set=[(X_valid, y_valid)],
                        eval_metric="binary_logloss",
                    )
                    models_new.append(m)
                
                # 融合旧模型和新模型 (新模型权重更高)
                if isinstance(model, EnsembleLGBMClassifier):
                    all_models = model.models + models_new
                else:
                    all_models = models_new
                model = EnsembleLGBMClassifier(all_models, feature_names=X_train.columns.tolist())
                logging.info("      集成模型数: %d", len(all_models))
            else:
                # 单模型模式
                model_params = merged_params.copy()
                model_params["random_state"] = seed + mining_round * 100
                model_params["n_estimators"] = max(500, merged_params.get("n_estimators", 1000) // 2)
                
                model = lgb.LGBMClassifier(**model_params)
                model.fit(
                    X_combined, y_combined,
                    sample_weight=combined_weights,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric="binary_logloss",
                )
            
            # 评估本轮效果
            valid_proba = model.predict_proba(X_valid)[:, 1]
            valid_pred = (valid_proba >= 0.5).astype(int)
            acc = accuracy_score(y_valid, valid_pred)
            
            # 计算两类正确率
            cm = confusion_matrix(y_valid, valid_pred)
            normal_acc = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
            abnormal_acc = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
            logging.info("      第%d轮结果: 正常类=%.1f%%, 异常类=%.1f%%, 总体=%.1f%%", 
                        mining_round + 1, normal_acc * 100, abnormal_acc * 100, acc * 100)
        
        logging.info("  [硬样本挖掘] 完成")
    
    elapsed = time.time() - start_time
    logging.info("  训练完成（耗时: %.2f 秒）", elapsed)
    
    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]
    
    metrics: dict[str, float | str] = {}
    metrics["accuracy"] = float(accuracy_score(y_valid, y_pred))
    metrics["precision"] = float(precision_score(y_valid, y_pred))
    metrics["recall"] = float(recall_score(y_valid, y_pred))
    metrics["f1"] = float(f1_score(y_valid, y_pred))
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_valid, y_proba))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    
    metrics["classification_report"] = classification_report(
        y_valid, y_pred, target_names=["Normal", "Abnormal"],
    )
    metrics["confusion_matrix"] = confusion_matrix(y_valid, y_pred).tolist()
    
    fpr, tpr, _ = roc_curve(y_valid, y_proba)
    metrics["roc_curve_fpr"] = fpr.tolist()
    metrics["roc_curve_tpr"] = tpr.tolist()
    
    logging.info("")
    logging.info("  二分类评估指标:")
    logging.info("    - 准确率 (Accuracy):  %.4f", metrics["accuracy"])
    logging.info("    - 精确率 (Precision): %.4f", metrics["precision"])
    logging.info("    - 召回率 (Recall):    %.4f", metrics["recall"])
    logging.info("    - F1 分数:            %.4f", metrics["f1"])
    logging.info("    - ROC-AUC:            %.4f", metrics["roc_auc"])
    
    # ============================================================
    # 全局特征重要性分析
    # ============================================================
    log_subsection("全局特征重要性 Top20（基于 Gain）")
    
    feature_names = X.columns.tolist()
    try:
        # 尝试获取 booster
        if hasattr(model, 'booster_'):
            booster = model.booster_
            importance_gain = booster.feature_importance(importance_type='gain')
        elif hasattr(model, 'booster'):
            importance_gain = model.booster.feature_importance(importance_type='gain')
        else:
            importance_gain = model.feature_importances_
    except Exception:
        importance_gain = model.feature_importances_
    
    # 确保长度匹配
    if len(importance_gain) != len(feature_names):
        logging.warning("  特征重要性长度不匹配，跳过详细分析")
    else:
        # 构建特征重要性表
        feat_imp_gain = sorted(zip(feature_names, importance_gain), key=lambda x: x[1], reverse=True)
        
        # 输出 Top20 (Gain)
        top20_gain = []
        logging.info("  %-4s %-40s %-15s %-15s", "排名", "特征名称", "重要性(Gain)", "特征类别")
        logging.info("  " + "-" * 80)
        for i, (fname, imp) in enumerate(feat_imp_gain[:20]):
            category = get_feature_category(fname)
            logging.info("  %-4d %-40s %-15.2f %-15s", i + 1, fname[:38], imp, category)
            top20_gain.append({
                "rank": i + 1,
                "feature": fname,
                "importance_gain": float(imp),
                "category": category
            })
        metrics["top20_features_gain"] = top20_gain
        
        # 特征类别贡献分析
        log_subsection("特征类别贡献分析")
        
        category_stats: dict[str, dict] = {}
        total_importance = sum(importance_gain)
        
        for fname, imp in zip(feature_names, importance_gain):
            cat = get_feature_category(fname)
            if cat not in category_stats:
                category_stats[cat] = {"count": 0, "total_importance": 0.0}
            category_stats[cat]["count"] += 1
            category_stats[cat]["total_importance"] += imp
        
        category_list = []
        for cat, stats in sorted(category_stats.items(), key=lambda x: x[1]["total_importance"], reverse=True):
            ratio = stats["total_importance"] / (total_importance + 1e-10) * 100
            category_list.append({
                "category": cat,
                "feature_count": stats["count"],
                "total_importance": float(stats["total_importance"]),
                "ratio_percent": float(ratio)
            })
        
        logging.info("  %-15s %-10s %-15s %-10s", "特征类别", "特征数量", "累计重要性", "占比")
        logging.info("  " + "-" * 55)
        for item in category_list:
            logging.info("  %-15s %-10d %-15.2f %.2f%%", 
                        item["category"], item["feature_count"], 
                        item["total_importance"], item["ratio_percent"])
        metrics["feature_category_importance"] = category_list
    
    # ============================================================
    # Top 10 影响最大的数据文件（模型最确信的样本）
    # ============================================================
    log_subsection("Top 10 影响最大的数据文件（模型最确信的样本）")
    
    # 计算每个样本的置信度（离 0.5 越远越确信）
    confidence_score = np.abs(y_proba - 0.5) * 2  # 归一化到 0-1
    # 按置信度降序排列，取 TOP10
    top_confident_idx = np.argsort(-confidence_score)[:10]
    
    top10_influential = []
    logging.info("  %-4s %-12s %-12s %s", "排名", "预测类别", "置信度", "文件名")
    logging.info("  " + "-" * 100)
    for rank, idx in enumerate(top_confident_idx):
        pred_label = int(y_pred[idx])
        prob = float(y_proba[idx])
        conf = float(confidence_score[idx])
        pred_name = "Normal" if pred_label == 0 else "Abnormal"
        file_path = paths_valid.iloc[idx]
        file_name = os.path.basename(file_path)
        logging.info("  %-4d %-12s %.2f%%       %s", 
                    rank + 1, pred_name, conf * 100, file_name)
        top10_influential.append({
            "rank": rank + 1,
            "pred_label": pred_name,
            "probability": prob,
            "confidence": conf,
            "file_path": file_path,
            "file_name": file_name
        })
    metrics["top10_influential_files"] = top10_influential

    if pv_hash_bundle is not None:
        metrics["pv_hash"] = pv_hash_bundle
        try:
            setattr(model, "pv_hash_", pv_hash_bundle)
        except Exception:
            pass
    
    return model, metrics, X_valid, y_valid, paths_valid


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "f1",
    target_recall: float = 0.95,
    target_both: float = 0.90,
) -> tuple[float, dict, pd.DataFrame]:
    """
    寻找最优分类阈值
    
    策略:
    - 'f1': 最大化 F1 分数
    - 'balanced': 平衡精确率和召回率（几何平均）
    - 'youden': 最大化 Youden's J 统计量 (TPR - FPR)
    - 'target_recall': 在保证目标召回率的前提下最大化精确率
    - 'dual90': 寻找让两类都达到目标正确率的阈值
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 正常类召回率
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        balanced = np.sqrt(precision * recall) if precision > 0 and recall > 0 else 0
        youden = recall + specificity - 1
        # 两类最小值（用于 dual90 策略）
        min_both = min(recall, specificity)
        
        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'balanced': balanced,
            'youden': youden,
            'min_both': min_both,
        })
    
    df_results = pd.DataFrame(results)
    
    if strategy == 'f1':
        best_idx = df_results['f1'].idxmax()
    elif strategy == 'balanced':
        best_idx = df_results['balanced'].idxmax()
    elif strategy == 'youden':
        best_idx = df_results['youden'].idxmax()
    elif strategy == 'target_recall':
        valid = df_results[df_results['recall'] >= target_recall]
        if len(valid) > 0:
            best_idx = valid['precision'].idxmax()
        else:
            best_idx = df_results['recall'].idxmax()
    elif strategy == 'dual90':
        # 寻找让两类都达到目标的阈值
        valid = df_results[(df_results['recall'] >= target_both) & (df_results['specificity'] >= target_both)]
        if len(valid) > 0:
            # 在满足条件的阈值中，选择两类最小值最大的
            best_idx = valid['min_both'].idxmax()
        else:
            # 如果没有满足条件的，选择两类最小值最大的
            best_idx = df_results['min_both'].idxmax()
    else:
        best_idx = df_results['f1'].idxmax()
    
    best_threshold = df_results.loc[best_idx, 'threshold']
    best_metrics = df_results.loc[best_idx].to_dict()
    
    return best_threshold, best_metrics, df_results


def evaluate_with_threshold(
    model: lgb.LGBMClassifier,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    threshold: float,
) -> dict:
    """使用指定阈值评估模型"""
    y_proba = model.predict_proba(X_valid)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics: dict[str, float | str] = {}
    metrics["threshold"] = threshold
    metrics["accuracy"] = float(accuracy_score(y_valid, y_pred))
    metrics["precision"] = float(precision_score(y_valid, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_valid, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_valid, y_pred, zero_division=0))
    
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_valid, y_proba))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    
    metrics["classification_report"] = classification_report(
        y_valid, y_pred, target_names=["Normal", "Abnormal"], zero_division=0,
    )
    metrics["confusion_matrix"] = confusion_matrix(y_valid, y_pred).tolist()
    
    return metrics


def optimize_threshold_and_report(
    model: lgb.LGBMClassifier,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    output_dir: str,
    target_accuracy: float = 0.90,
) -> tuple[float, dict]:
    """阈值优化并生成对比报告"""
    log_subsection("分类阈值优化")
    
    y_proba = model.predict_proba(X_valid)[:, 1]
    y_true = y_valid.values
    
    strategies = ['f1', 'balanced', 'youden', 'dual90']
    all_results = {}
    
    logging.info("  默认阈值 (0.5) 评估:")
    default_metrics = evaluate_with_threshold(model, X_valid, y_valid, 0.5)
    cm_default = np.array(default_metrics["confusion_matrix"])
    cm_pct = cm_default.astype(float) / cm_default.sum(axis=1, keepdims=True) * 100
    logging.info("    - 正常类正确率: %.1f%%", cm_pct[0, 0])
    logging.info("    - 异常类正确率: %.1f%%", cm_pct[1, 1])
    logging.info("    - F1 分数: %.4f", default_metrics["f1"])
    
    logging.info("")
    logging.info("  各策略最优阈值搜索:")
    
    for strategy in strategies:
        best_thresh, best_metrics, _ = find_optimal_threshold(
            y_true, y_proba, strategy=strategy, target_both=target_accuracy
        )
        all_results[strategy] = {
            'threshold': best_thresh,
            'metrics': best_metrics,
        }
        logging.info("    [%s] 阈值=%.2f, F1=%.4f, 精确率=%.4f, 召回率=%.4f, 特异度=%.4f",
                     strategy, best_thresh, best_metrics['f1'], 
                     best_metrics['precision'], best_metrics['recall'],
                     best_metrics['specificity'])
    
    # 检查 dual90 策略是否达到目标
    dual90_metrics = all_results['dual90']['metrics']
    dual90_achievable = (dual90_metrics['recall'] >= target_accuracy and 
                         dual90_metrics['specificity'] >= target_accuracy)
    
    if dual90_achievable:
        recommended_strategy = 'dual90'
        logging.info("")
        logging.info("  ✅ dual90 策略达标！两类正确率均 >= %.0f%%", target_accuracy * 100)
    else:
        # 如果 dual90 未达标，使用 youden
        recommended_strategy = 'youden'
        logging.info("")
        logging.info("  ⚠️ dual90 未达标，当前模型无法让两类都达到 %.0f%%", target_accuracy * 100)
        logging.info("  最佳可达: 正常类=%.1f%%, 异常类=%.1f%%", 
                     dual90_metrics['specificity'] * 100, dual90_metrics['recall'] * 100)
    
    recommended_threshold = all_results[recommended_strategy]['threshold']
    
    logging.info("")
    logging.info("  推荐阈值: %.2f (策略: %s)", recommended_threshold, recommended_strategy)
    
    optimized_metrics = evaluate_with_threshold(model, X_valid, y_valid, recommended_threshold)
    cm_optimized = np.array(optimized_metrics["confusion_matrix"])
    cm_opt_pct = cm_optimized.astype(float) / cm_optimized.sum(axis=1, keepdims=True) * 100
    
    logging.info("")
    logging.info("  优化后评估指标:")
    logging.info("    - 准确率 (Accuracy):  %.4f", optimized_metrics["accuracy"])
    logging.info("    - 精确率 (Precision): %.4f", optimized_metrics["precision"])
    logging.info("    - 召回率 (Recall):    %.4f", optimized_metrics["recall"])
    logging.info("    - F1 分数:            %.4f", optimized_metrics["f1"])
    logging.info("    - 正常类正确率: %.1f%% (提升 %.1f%%)", 
                 cm_opt_pct[0, 0], cm_opt_pct[0, 0] - cm_pct[0, 0])
    logging.info("    - 异常类正确率: %.1f%% (变化 %.1f%%)", 
                 cm_opt_pct[1, 1], cm_opt_pct[1, 1] - cm_pct[1, 1])
    
    # 保存对比图
    save_threshold_comparison_plot(cm_default, cm_optimized, recommended_threshold, output_dir)
    
    return recommended_threshold, optimized_metrics


def save_threshold_comparison_plot(
    cm_default: np.ndarray,
    cm_optimized: np.ndarray,
    threshold: float,
    output_dir: str,
) -> None:
    """保存阈值对比图"""
    os.makedirs(output_dir, exist_ok=True)
    setup_chinese_font()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    cm_default_pct = cm_default.astype(float) / cm_default.sum(axis=1, keepdims=True) * 100
    cm_optimized_pct = cm_optimized.astype(float) / cm_optimized.sum(axis=1, keepdims=True) * 100
    
    labels = ["正常", "异常"]
    
    # 默认阈值混淆矩阵
    sns.heatmap(cm_default_pct, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 12})
    for t in axes[0].texts:
        t.set_text(t.get_text() + "%")
    axes[0].set_xlabel("预测类别")
    axes[0].set_ylabel("真实类别")
    axes[0].set_title("默认阈值 (0.5)")
    
    # 优化阈值混淆矩阵
    sns.heatmap(cm_optimized_pct, annot=True, fmt=".1f", cmap="Greens", cbar=False, ax=axes[1],
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 12})
    for t in axes[1].texts:
        t.set_text(t.get_text() + "%")
    axes[1].set_xlabel("预测类别")
    axes[1].set_ylabel("真实类别")
    axes[1].set_title(f"优化阈值 ({threshold:.2f})")
    
    plt.suptitle("分类阈值优化对比", fontsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, "threshold_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    logging.info("  阈值对比图: %s", fig_path)


def save_metrics_and_plots(metrics: dict, output_dir: str) -> None:
    """保存评估报告和可视化图"""
    log_subsection("保存评估报告和可视化图")
    os.makedirs(output_dir, exist_ok=True)
    setup_chinese_font()
    
    metrics_path = os.path.join(output_dir, "metrics_binary.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logging.info("  评估指标: %s", metrics_path)
    
    cm = np.array(metrics.get("confusion_matrix", []))
    if cm.size > 0:
        cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        for lang in ("en", "zh"):
            fig, ax = plt.subplots(figsize=(4, 4))
            if lang == "en":
                labels = ["Normal", "Abnormal"]
                title = "Binary Confusion Matrix (%)"
                filename = "confusion_binary_en.png"
            else:
                labels = ["正常", "异常"]
                title = "二分类混淆矩阵 (%)"
                filename = "confusion_binary_zh.png"
            
            sns.heatmap(
                cm_percent, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=ax,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 12},
            )
            for t in ax.texts:
                t.set_text(t.get_text() + "%")
            ax.set_xlabel("Predicted" if lang == "en" else "预测类别")
            ax.set_ylabel("True" if lang == "en" else "真实类别")
            plt.title(title)
            plt.tight_layout()
            fig_path = os.path.join(output_dir, filename)
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
            logging.info("  混淆矩阵(%s): %s", lang.upper(), fig_path)
    
    fpr = metrics.get("roc_curve_fpr")
    tpr = metrics.get("roc_curve_tpr")
    if fpr and tpr:
        for lang in ("en", "zh"):
            fig, ax = plt.subplots(figsize=(4, 4))
            if lang == "en":
                xlabel = "False Positive Rate"
                ylabel = "True Positive Rate"
                title = "ROC Curve (Binary)"
                filename = "roc_binary_en.png"
                random_label = "Random"
            else:
                xlabel = "假阳性率"
                ylabel = "真阳性率"
                title = "ROC 曲线（二分类）"
                filename = "roc_binary_zh.png"
                random_label = "随机"
            ax.plot(fpr, tpr, label="ROC")
            ax.plot([0, 1], [0, 1], "k--", label=random_label)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            plt.tight_layout()
            fig_path = os.path.join(output_dir, filename)
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
            logging.info("  ROC曲线(%s): %s", lang.upper(), fig_path)


def save_binary_visualizations(
    model,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    output_dir: str,
    optimal_threshold: float,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    setup_chinese_font()

    y_true = y_valid.values.astype(int)
    y_proba = model.predict_proba(X_valid)[:, 1].astype(float)
    y_pred_opt = (y_proba >= float(optimal_threshold)).astype(int)

    def save_placeholder(base_name: str, title_zh: str, title_en: str) -> None:
        for lang in ("zh", "en"):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.axis("off")
            ax.text(0.5, 0.5, "不可用" if lang == "zh" else "N/A", ha="center", va="center", fontsize=14)
            ax.set_title(title_zh if lang == "zh" else title_en)
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"{base_name}_{lang}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)

    try:
        evals_result = None
        if hasattr(model, "evals_result_"):
            evals_result = getattr(model, "evals_result_", None)
        elif isinstance(model, EnsembleLGBMClassifier):
            for m in model.models:
                if hasattr(m, "evals_result_"):
                    evals_result = getattr(m, "evals_result_", None)
                    break
        metric_series = None
        metric_name = None
        if isinstance(evals_result, dict):
            key0 = next(iter(evals_result.keys()))
            metric0 = next(iter(evals_result[key0].keys()))
            metric_series = np.asarray(evals_result[key0][metric0], dtype=float)
            metric_name = str(metric0)
        if metric_series is not None and metric_series.size > 1:
            x = np.arange(1, metric_series.size + 1)
            for lang in ("zh", "en"):
                fig, ax = plt.subplots(figsize=(6.2, 4.2))
                ax.plot(x, metric_series, label=metric_name or "loss")
                ax.set_xlabel("迭代轮数" if lang == "zh" else "Iteration")
                ax.set_ylabel("损失" if lang == "zh" else "Loss")
                ax.set_title(("损失曲线" if lang == "zh" else "Loss Curve") + f" ({'LightGBM' if lang=='en' else 'LightGBM'})")
                ax.legend()
                plt.tight_layout()
                fig_path = os.path.join(output_dir, f"loss_curve_{lang}.png")
                plt.savefig(fig_path, dpi=150)
                plt.close(fig)
        else:
            raise RuntimeError("no evals")
    except Exception:
        try:
            ll = float(log_loss(y_true, y_proba, eps=1e-15))
            for lang in ("zh", "en"):
                fig, ax = plt.subplots(figsize=(6.2, 4.2))
                ax.plot([1], [ll], marker="o")
                ax.set_xlabel("迭代轮数" if lang == "zh" else "Iteration")
                ax.set_ylabel("损失" if lang == "zh" else "Loss")
                ax.set_title(("损失曲线" if lang == "zh" else "Loss Curve") + " (LightGBM)")
                plt.tight_layout()
                fig_path = os.path.join(output_dir, f"loss_curve_{lang}.png")
                plt.savefig(fig_path, dpi=150)
                plt.close(fig)
        except Exception:
            save_placeholder("loss_curve", "损失曲线（LightGBM）", "Loss Curve (LightGBM)")

    cm = confusion_matrix(y_true, y_pred_opt, labels=[0, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cm_pct = np.nan_to_num(cm_pct, nan=0.0, posinf=0.0, neginf=0.0)
    for lang in ("zh", "en"):
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        labels = ["正常", "异常"] if lang == "zh" else ["Normal", "Abnormal"]
        title = "二分类混淆矩阵(%)" if lang == "zh" else "Binary Confusion Matrix (%)"
        sns.heatmap(
            cm_pct,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 12},
        )
        for t in ax.texts:
            t.set_text(t.get_text() + "%")
        ax.set_xlabel("预测类别" if lang == "zh" else "Predicted")
        ax.set_ylabel("真实类别" if lang == "zh" else "True")
        ax.set_title(title + (f"\n阈值={optimal_threshold:.2f}" if lang == "zh" else f"\nThreshold={optimal_threshold:.2f}"))
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"confusion_percent_{lang}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_val = roc_auc_score(y_true, y_proba)
        for lang in ("zh", "en"):
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            ax.plot(fpr, tpr, label=f"AUC={auc_val:.4f}")
            ax.plot([0, 1], [0, 1], "k--", label=("随机" if lang == "zh" else "Random"))
            ax.set_xlabel("假阳性率" if lang == "zh" else "False Positive Rate")
            ax.set_ylabel("真阳性率" if lang == "zh" else "True Positive Rate")
            ax.set_title(("ROC 曲线" if lang == "zh" else "ROC Curve") + " (LightGBM)")
            ax.legend()
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"roc_{lang}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
    except Exception:
        save_placeholder("roc", "ROC 曲线（LightGBM）", "ROC Curve (LightGBM)")

    try:
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        for lang in ("zh", "en"):
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            ax.plot(rec, prec, label=f"AP={ap:.4f}")
            ax.set_xlabel("召回率" if lang == "zh" else "Recall")
            ax.set_ylabel("精确率" if lang == "zh" else "Precision")
            ax.set_title(("PR 曲线" if lang == "zh" else "PR Curve") + " (LightGBM)")
            ax.legend()
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"pr_curve_{lang}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
    except Exception:
        save_placeholder("pr_curve", "PR 曲线（LightGBM）", "PR Curve (LightGBM)")

    try:
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
        brier = brier_score_loss(y_true, y_proba)
        for lang in ("zh", "en"):
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            ax.plot([0, 1], [0, 1], "k--", label=("理想校准" if lang == "zh" else "Perfect"))
            ax.plot(prob_pred, prob_true, marker="o", label=f"Brier={brier:.4f}")
            ax.set_xlabel("预测概率" if lang == "zh" else "Predicted Probability")
            ax.set_ylabel("真实正例比例" if lang == "zh" else "Fraction of Positives")
            ax.set_title(("校准曲线" if lang == "zh" else "Calibration Curve") + " (LightGBM)")
            ax.legend()
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"calibration_curve_{lang}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
    except Exception:
        save_placeholder("calibration_curve", "校准曲线（LightGBM）", "Calibration Curve (LightGBM)")

    for lang in ("zh", "en"):
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.hist(y_proba[y_true == 0], bins=40, alpha=0.6, label=("正常" if lang == "zh" else "Normal"))
        ax.hist(y_proba[y_true == 1], bins=40, alpha=0.6, label=("异常" if lang == "zh" else "Abnormal"))
        ax.axvline(float(optimal_threshold), color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("预测为异常的概率" if lang == "zh" else "Predicted Abnormal Probability")
        ax.set_ylabel("样本数" if lang == "zh" else "Count")
        ax.set_title(("预测分数分布" if lang == "zh" else "Prediction Score Distribution") + " (LightGBM)")
        ax.legend()
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"score_distribution_{lang}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)

    thresholds = np.linspace(0.0, 1.0, 201)
    precision_list = []
    recall_list = []
    specificity_list = []
    f1_list = []
    accuracy_list = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)
        f1_list.append(f1)
        accuracy_list.append(accuracy)

    for lang in ("zh", "en"):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(thresholds, accuracy_list, label=("准确率" if lang == "zh" else "Accuracy"))
        ax.plot(thresholds, precision_list, label=("精确率" if lang == "zh" else "Precision"))
        ax.plot(thresholds, recall_list, label=("召回率" if lang == "zh" else "Recall"))
        ax.plot(thresholds, specificity_list, label=("特异度" if lang == "zh" else "Specificity"))
        ax.plot(thresholds, f1_list, label=("F1" if lang == "zh" else "F1"))
        ax.axvline(float(optimal_threshold), color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("阈值" if lang == "zh" else "Threshold")
        ax.set_ylabel("指标值" if lang == "zh" else "Metric")
        ax.set_title(("阈值对比曲线" if lang == "zh" else "Metrics vs Threshold") + " (LightGBM)")
        ax.legend(ncol=3, fontsize=9)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"threshold_curve_{lang}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)

    summary = {}
    summary["threshold_optimal"] = float(optimal_threshold)
    summary["accuracy_opt"] = float(accuracy_score(y_true, y_pred_opt))
    summary["precision_opt"] = float(precision_score(y_true, y_pred_opt, zero_division=0))
    summary["recall_opt"] = float(recall_score(y_true, y_pred_opt, zero_division=0))
    summary["f1_opt"] = float(f1_score(y_true, y_pred_opt, zero_division=0))
    try:
        summary["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        summary["roc_auc"] = float("nan")
    try:
        summary["ap"] = float(average_precision_score(y_true, y_proba))
    except Exception:
        summary["ap"] = float("nan")
    try:
        summary["logloss"] = float(log_loss(y_true, y_proba, eps=1e-15))
    except Exception:
        summary["logloss"] = float("nan")
    try:
        summary["brier"] = float(brier_score_loss(y_true, y_proba))
    except Exception:
        summary["brier"] = float("nan")

    keys = ["threshold_optimal", "accuracy_opt", "precision_opt", "recall_opt", "f1_opt", "roc_auc", "ap", "logloss", "brier"]
    for lang in ("zh", "en"):
        fig, ax = plt.subplots(figsize=(6.2, 2.8))
        ax.axis("off")
        rows = []
        for k in keys:
            v = summary.get(k, float("nan"))
            rows.append([k, f"{v:.6f}" if isinstance(v, float) else str(v)])
        col_labels = ["指标", "数值"] if lang == "zh" else ["Metric", "Value"]
        table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="left", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.3)
        ax.set_title(("二分类指标汇总" if lang == "zh" else "Binary Metrics Summary") + " (LightGBM)")
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"metrics_summary_{lang}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)


def save_model(
    model: lgb.LGBMClassifier,
    feature_names: list[str],
    output_dir: str,
    seed: int,
    sampling_config: dict,
) -> str:
    """保存模型文件"""
    log_subsection("保存模型文件")
    os.makedirs(output_dir, exist_ok=True)
    
    pv_hash = getattr(model, "pv_hash_", None)

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "label_mapping": {0: "Normal", 1: "Abnormal"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "sampling_config": sampling_config,
        "pv_hash": pv_hash,
    }
    
    path = os.path.join(output_dir, "binary_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logging.info("  模型文件: %s", path)
    
    meta_path = os.path.join(output_dir, "binary_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": path,
            "feature_names": feature_names,
            "label_mapping": bundle["label_mapping"],
            "created_at": bundle["created_at"],
            "pv_hash": pv_hash,
        }, f, ensure_ascii=False, indent=2)
    logging.info("  元信息: %s", meta_path)
    
    return path


def run_training(
    data_dir: str,
    output_dir: str,
    test_size: float,
    seed: int,
    binary_params: dict,
    sampling_config: dict,
    training_config: dict,
) -> None:
    """执行训练流程"""
    total_start = time.time()
    set_global_seed(seed)
    setup_logging(output_dir)
    
    log_section("二分类模型训练")
    log_subsection("配置信息")
    logging.info("  数据目录: %s", data_dir)
    logging.info("  输出目录: %s", output_dir)
    logging.info("  随机种子: %d", seed)
    logging.info("  验证集比例: %.1f%%", test_size * 100)
    logging.info("  平衡策略: %s", training_config.get("balance_strategy", "undersample_1to1"))
    logging.info("  目标双类准确率: %.0f%%", training_config.get("target_dual_accuracy", 0.90) * 100)
    
    X, y, feature_names, file_paths = build_feature_table(data_dir, sampling_config, training_config)
    
    model, metrics, X_valid, y_valid, paths_valid = train_model(
        X, y, test_size, seed, binary_params, training_config, file_paths
    )
    
    reports_dir = os.path.join(output_dir, "reports")
    save_metrics_and_plots(metrics, reports_dir)
    
    # 阈值优化
    target_accuracy = training_config.get("target_dual_accuracy", 0.90)
    optimal_threshold, optimized_metrics = optimize_threshold_and_report(
        model, X_valid, y_valid, reports_dir, target_accuracy=target_accuracy
    )

    save_binary_visualizations(model, X_valid, y_valid, reports_dir, optimal_threshold)
    
    # 保存优化后的指标
    optimized_metrics_path = os.path.join(reports_dir, "metrics_binary_optimized.json")
    with open(optimized_metrics_path, "w", encoding="utf-8") as f:
        json.dump(optimized_metrics, f, ensure_ascii=False, indent=2)
    logging.info("  优化后指标: %s", optimized_metrics_path)
    
    # 保存训练配置
    training_config_path = os.path.join(reports_dir, "training_config.json")
    with open(training_config_path, "w", encoding="utf-8") as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)
    
    models_dir = os.path.join(output_dir, "models")
    model_path = save_model(model, feature_names, models_dir, seed, sampling_config)
    
    # 保存最优阈值
    threshold_path = os.path.join(models_dir, "optimal_threshold.json")
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({"optimal_threshold": optimal_threshold}, f, indent=2)
    logging.info("  最优阈值: %s", threshold_path)
    
    total_elapsed = time.time() - total_start
    log_section("训练完成")
    logging.info("  总耗时: %.2f 秒", total_elapsed)
    logging.info("  模型路径: %s", model_path)
    logging.info("  报告目录: %s", reports_dir)
    logging.info("  推荐分类阈值: %.2f", optimal_threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="二分类模型训练 (Normal vs Abnormal)")
    parser.add_argument("--data-dir", type=str, default=None, help="数据目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--test-size", type=float, default=None, help="验证集比例")
    parser.add_argument("--config", type=str, default=os.path.join("config", "binary_config.json"), help="配置文件路径")
    
    args = parser.parse_args()
    
    seed, cfg_test_size, binary_params, sampling_config, training_config, cfg_data_dir, cfg_output_dir = load_config(args.config)
    
    test_size = cfg_test_size if args.test_size is None else args.test_size
    data_dir = args.data_dir if args.data_dir is not None else cfg_data_dir
    output_dir = args.output_dir if args.output_dir is not None else cfg_output_dir
    
    run_training(data_dir, output_dir, test_size, seed, binary_params, sampling_config, training_config)


if __name__ == "__main__":
    main()
