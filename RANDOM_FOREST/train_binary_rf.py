"""
随机森林二分类模型训练模块
训练 Normal vs Abnormal 分类模型
用于与 LightGBM 和 XGBoost 进行对比实验
"""

import argparse
import json
import logging
import os
import pickle
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, rcParams
from sklearn.ensemble import RandomForestClassifier
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
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

SEED = 42

DEFAULT_BINARY_PARAMS: dict[str, float | int | str | bool] = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "oob_score": True,
    "random_state": SEED,
    "n_jobs": -1,
}

DEFAULT_SAMPLING_CONFIG: dict[str, float | int | str] = {
    "mode": "window",
    "window_sec": 120.0,
    "step_sec": 30.0,
    "min_points": 10,
    "max_windows_per_file": 100,
    "num_workers": 0,
}

FEATURE_CATEGORY_MAPPING: dict[str, str] = {
    "value_count": "基础统计特征", "value_mean": "基础统计特征", "value_min": "基础统计特征",
    "value_max": "基础统计特征", "value_median": "基础统计特征", "value_first": "基础统计特征",
    "value_last": "基础统计特征", "value_rms": "基础统计特征",
    "value_std": "离散程度特征", "value_range": "离散程度特征", "value_iqr": "离散程度特征", "value_cv": "离散程度特征",
    "skewness": "分布形态特征", "kurtosis": "分布形态特征",
    "value_q01": "分布形态特征", "value_q05": "分布形态特征", "value_q10": "分布形态特征",
    "value_q25": "分布形态特征", "value_q75": "分布形态特征", "value_q90": "分布形态特征",
    "value_q95": "分布形态特征", "value_q99": "分布形态特征",
    "diff_mean": "差分特征", "diff_std": "差分特征", "diff_abs_mean": "差分特征",
    "diff_max": "差分特征", "diff_min": "差分特征", "diff_abs_max": "差分特征",
    "diff2_mean": "差分特征", "diff2_std": "差分特征", "diff2_abs_mean": "差分特征",
    "direction_changes": "差分特征", "direction_change_rate": "差分特征",
    "value_change": "趋势特征", "value_change_ratio": "趋势特征", "trend_slope": "趋势特征",
    "seg_mean_diff_12": "趋势特征", "seg_mean_diff_23": "趋势特征", "seg_mean_diff_13": "趋势特征",
    "seg1_mean": "局部特征", "seg2_mean": "局部特征", "seg3_mean": "局部特征",
    "seg1_std": "局部特征", "seg2_std": "局部特征", "seg3_std": "局部特征",
    "local_max_count": "局部特征", "local_min_count": "局部特征", "peak_count": "局部特征", "peak_rate": "局部特征",
    "rolling_var_mean": "稳定性特征", "rolling_var_std": "稳定性特征", "rolling_var_max": "稳定性特征",
    "autocorr_lag1": "自相关特征", "autocorr_lag5": "自相关特征",
    "entropy": "信息熵特征", "fft_spectral_entropy": "信息熵特征",
    "fft_dc_ratio": "频域特征", "fft_low_freq_ratio": "频域特征", "fft_high_freq_ratio": "频域特征",
    "fft_mid_freq_ratio": "频域特征", "fft_dominant_freq_idx": "频域特征", "fft_dominant_power_ratio": "频域特征",
    "fft_spectral_flatness": "频域特征", "fft_spectral_centroid": "频域特征",
    "dt_mean_ms": "时间间隔特征", "dt_std_ms": "时间间隔特征", "dt_max_ms": "时间间隔特征",
    "dt_min_ms": "时间间隔特征", "dt_cv": "时间间隔特征",
    "outlier_ratio_3std": "异常指标特征", "outlier_ratio_2std": "异常指标特征",
    "zero_ratio": "异常指标特征", "negative_ratio": "异常指标特征", "positive_ratio": "异常指标特征",
    "zero_crossing_rate": "异常指标特征",
}


def get_feature_category(feature_name: str) -> str:
    if "_x_" in feature_name or "_div_" in feature_name:
        return "交互特征"
    return FEATURE_CATEGORY_MAPPING.get(feature_name, "其他特征")


def log_section(title: str) -> None:
    logging.info("")
    logging.info("=" * 60)
    logging.info(f"【{title}】")
    logging.info("=" * 60)


def log_subsection(title: str) -> None:
    logging.info(f"\n>>> {title}")
    logging.info("-" * 50)


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
            return


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train_binary_rf.log")
    
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
    seed = SEED
    test_size = 0.15
    binary_params = DEFAULT_BINARY_PARAMS.copy()
    sampling_config = DEFAULT_SAMPLING_CONFIG.copy()
    training_config = {
        "balance_strategy": "undersample_1to1",
        "use_feature_interactions": True,
        "target_dual_accuracy": 0.90,
    }
    data_dir = "test-datasets"
    output_dir = "artifacts/random_forest/binary"
    
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


def extract_features_from_series(values: pd.Series, epochs: pd.Series) -> dict:
    """从时序数据中提取统计特征（与 LightGBM 版本完全一致）"""
    v = values.values.astype(np.float64)
    feats: dict[str, float] = {}
    
    v_valid = v[~np.isnan(v)]
    n = len(v_valid)
    feats["value_count"] = float(n)
    
    if n > 0:
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
        
        quantiles = np.percentile(v_valid, [1, 5, 10, 25, 75, 90, 95, 99])
        feats["value_q01"] = float(quantiles[0])
        feats["value_q05"] = float(quantiles[1])
        feats["value_q10"] = float(quantiles[2])
        feats["value_q25"] = float(quantiles[3])
        feats["value_q75"] = float(quantiles[4])
        feats["value_q90"] = float(quantiles[5])
        feats["value_q95"] = float(quantiles[6])
        feats["value_q99"] = float(quantiles[7])
        
        feats["value_range"] = float(max_val - min_val)
        feats["value_iqr"] = float(quantiles[4] - quantiles[3])
        feats["value_cv"] = float(std_val / (abs(mean_val) + 1e-10))
        feats["value_rms"] = float(np.sqrt(np.mean(v_valid ** 2)))
        
        threshold_3std = abs(mean_val) + 3 * std_val
        threshold_2std = abs(mean_val) + 2 * std_val
        feats["outlier_ratio_3std"] = float(np.sum(np.abs(v_valid) > threshold_3std) / n)
        feats["outlier_ratio_2std"] = float(np.sum(np.abs(v_valid) > threshold_2std) / n)
        
        feats["zero_ratio"] = float(np.sum(v_valid == 0) / n)
        feats["negative_ratio"] = float(np.sum(v_valid < 0) / n)
        feats["positive_ratio"] = float(np.sum(v_valid > 0) / n)
        
        if std_val > 1e-10:
            feats["skewness"] = float(np.mean(((v_valid - mean_val) / std_val) ** 3))
            feats["kurtosis"] = float(np.mean(((v_valid - mean_val) / std_val) ** 4) - 3)
        else:
            feats["skewness"] = 0.0
            feats["kurtosis"] = 0.0
        
        if n > 1:
            zero_crossings = np.sum(np.diff(np.signbit(v_valid - mean_val)))
            feats["zero_crossing_rate"] = float(zero_crossings / (n - 1))
        else:
            feats["zero_crossing_rate"] = 0.0
        
        if n > 1:
            diffs = np.diff(v_valid)
            feats["diff_mean"] = float(np.mean(diffs))
            feats["diff_std"] = float(np.std(diffs))
            feats["diff_abs_mean"] = float(np.mean(np.abs(diffs)))
            feats["diff_max"] = float(np.max(diffs))
            feats["diff_min"] = float(np.min(diffs))
            feats["diff_abs_max"] = float(np.max(np.abs(diffs)))
            
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            feats["direction_changes"] = float(sign_changes)
            feats["direction_change_rate"] = float(sign_changes / (n - 1))
            
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
        
        feats["value_first"] = float(v_valid[0])
        feats["value_last"] = float(v_valid[-1])
        feats["value_change"] = float(v_valid[-1] - v_valid[0])
        feats["value_change_ratio"] = float((v_valid[-1] - v_valid[0]) / (abs(v_valid[0]) + 1e-10))
        
        if n >= 3:
            third = n // 3
            seg1, seg2, seg3 = v_valid[:third], v_valid[third:2*third], v_valid[2*third:]
            mean1, mean2, mean3 = np.mean(seg1), np.mean(seg2), np.mean(seg3)
            feats["seg1_mean"] = float(mean1)
            feats["seg2_mean"] = float(mean2)
            feats["seg3_mean"] = float(mean3)
            feats["seg1_std"] = float(np.std(seg1))
            feats["seg2_std"] = float(np.std(seg2))
            feats["seg3_std"] = float(np.std(seg3))
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
        
        if n > 1:
            mid = n // 2
            feats["trend_slope"] = float(np.mean(v_valid[mid:]) - np.mean(v_valid[:mid]))
        else:
            feats["trend_slope"] = 0.0
        
        if n >= 3:
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
        
        if n >= 10:
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
        
        if n >= 10 and feats["value_range"] > 1e-10:
            n_bins = min(10, n // 3)
            hist, _ = np.histogram(v_valid, bins=n_bins)
            hist = hist / n
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            feats["entropy"] = float(entropy)
        else:
            feats["entropy"] = 0.0
        
        if n >= 8:
            try:
                fft_vals = np.fft.rfft(v_valid - mean_val)
                fft_magnitude = np.abs(fft_vals)
                fft_power = fft_magnitude ** 2
                total_power = np.sum(fft_power) + 1e-10
                
                if len(fft_magnitude) > 1:
                    feats["fft_dc_ratio"] = float(fft_power[0] / total_power)
                    low_freq_idx = max(1, len(fft_power) // 4)
                    feats["fft_low_freq_ratio"] = float(np.sum(fft_power[1:low_freq_idx]) / total_power)
                    high_freq_idx = len(fft_power) * 3 // 4
                    feats["fft_high_freq_ratio"] = float(np.sum(fft_power[high_freq_idx:]) / total_power)
                    feats["fft_mid_freq_ratio"] = float(np.sum(fft_power[low_freq_idx:high_freq_idx]) / total_power)
                    if len(fft_magnitude) > 2:
                        dominant_idx = np.argmax(fft_magnitude[1:]) + 1
                        feats["fft_dominant_freq_idx"] = float(dominant_idx / len(fft_magnitude))
                        feats["fft_dominant_power_ratio"] = float(fft_power[dominant_idx] / total_power)
                    else:
                        feats["fft_dominant_freq_idx"] = 0.0
                        feats["fft_dominant_power_ratio"] = 0.0
                    fft_norm = fft_power / total_power
                    fft_norm = fft_norm[fft_norm > 0]
                    feats["fft_spectral_entropy"] = float(-np.sum(fft_norm * np.log2(fft_norm + 1e-10)))
                    feats["fft_spectral_flatness"] = float(np.exp(np.mean(np.log(fft_magnitude[1:] + 1e-10))) / (np.mean(fft_magnitude[1:]) + 1e-10))
                    freq_indices = np.arange(len(fft_magnitude))
                    feats["fft_spectral_centroid"] = float(np.sum(freq_indices * fft_magnitude) / (np.sum(fft_magnitude) + 1e-10) / len(fft_magnitude))
                else:
                    for key in ["fft_dc_ratio", "fft_low_freq_ratio", "fft_high_freq_ratio", "fft_mid_freq_ratio",
                                "fft_dominant_freq_idx", "fft_dominant_power_ratio", "fft_spectral_entropy",
                                "fft_spectral_flatness", "fft_spectral_centroid"]:
                        feats[key] = 0.0
            except Exception:
                for key in ["fft_dc_ratio", "fft_low_freq_ratio", "fft_high_freq_ratio", "fft_mid_freq_ratio",
                            "fft_dominant_freq_idx", "fft_dominant_power_ratio", "fft_spectral_entropy",
                            "fft_spectral_flatness", "fft_spectral_centroid"]:
                    feats[key] = 0.0
        else:
            for key in ["fft_dc_ratio", "fft_low_freq_ratio", "fft_high_freq_ratio", "fft_mid_freq_ratio",
                        "fft_dominant_freq_idx", "fft_dominant_power_ratio", "fft_spectral_entropy",
                        "fft_spectral_flatness", "fft_spectral_centroid"]:
                feats[key] = 0.0
    else:
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
            "fft_dc_ratio", "fft_low_freq_ratio", "fft_high_freq_ratio", "fft_mid_freq_ratio",
            "fft_dominant_freq_idx", "fft_dominant_power_ratio",
            "fft_spectral_entropy", "fft_spectral_flatness", "fft_spectral_centroid",
        ]
        for feat_name in default_zero_feats:
            feats[feat_name] = 0.0
    
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
    
    for key in feats:
        if np.isnan(feats[key]) or np.isinf(feats[key]):
            feats[key] = 0.0
    
    return feats


def generate_window_records(df: pd.DataFrame, label: str, path: str, sampling_config: dict) -> list[dict]:
    if "epoch_ms" not in df.columns or "value" not in df.columns:
        return []
    
    df_sorted = df.sort_values("epoch_ms")
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
            feats = extract_features_from_series(sub["value"], sub["epoch_ms"])
            feats["file_path"] = path
            feats["pv_name"] = str(sub["pv_name"].iloc[0]) if "pv_name" in sub.columns else ""
            feats["window_start_epoch_ms"] = float(start)
            feats["window_end_epoch_ms"] = float(win_end)
            feats["label"] = 0 if label == "Normal" else 1
            records.append(feats)
        
        start += step_ms
    
    return records


def build_feature_table(base_dir: str, sampling_config: dict) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
    log_subsection("数据扫描")
    logging.info("  数据目录: %s", base_dir)
    paths = find_parquet_files(base_dir, verbose=True)
    
    if not paths:
        raise RuntimeError("未找到任何 Parquet 文件")
    
    mode = str(sampling_config.get("mode", "window"))
    num_workers = int(sampling_config.get("num_workers", 0))
    
    log_subsection("特征构建")
    logging.info("  采样模式: %s", mode)
    logging.info("  并行线程: %s", num_workers if num_workers > 1 else "单线程")
    
    total_files = len(paths)
    start_time = time.time()
    records: list[dict] = []
    processed_files = [0]
    skipped_files = [0]
    lock = threading.Lock()

    def process_one(item: tuple[str, str]) -> list[dict]:
        label, path = item
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
            result = generate_window_records(df, label, path, sampling_config)
        else:
            feats = extract_features_from_series(df["value"], df["epoch_ms"])
            feats["file_path"] = path
            feats["pv_name"] = str(df["pv_name"].iloc[0]) if "pv_name" in df.columns else ""
            feats["label"] = 0 if label == "Normal" else 1
            result = [feats]
        
        with lock:
            processed_files[0] += 1
        return result

    if num_workers and num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_one, item) for item in paths]
            for fut in as_completed(futures):
                try:
                    records.extend(fut.result())
                except Exception:
                    with lock:
                        skipped_files[0] += 1
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
    
    feature_cols = [c for c in table.columns if c not in 
                   ["file_path", "pv_name", "window_start_epoch_ms", "window_end_epoch_ms", "label"]]
    
    X = table[feature_cols].astype(np.float32)
    y = table["label"].astype(np.int8)
    file_paths = table["file_path"].reset_index(drop=True)
    
    logging.info("  特征构建完成（耗时: %.2f 秒）", elapsed)
    logging.info("  生成样本: %d 条（正常: %d，异常: %d）", X.shape[0], int((y == 0).sum()), int((y == 1).sum()))
    logging.info("  特征维度: %d", len(feature_cols))
    
    return X, y, feature_cols, file_paths


def add_feature_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """添加特征交互项（与 LightGBM 版本一致）"""
    X_new = X.copy()
    cols = X.columns.tolist()
    
    key_features = ['value_mean', 'value_std', 'value_cv', 'diff_std', 'entropy', 'skewness', 'kurtosis']
    available_keys = [f for f in key_features if f in cols]
    
    for i, f1 in enumerate(available_keys):
        for f2 in available_keys[i+1:]:
            X_new[f"{f1}_x_{f2}"] = X[f1] * X[f2]
            X_new[f"{f1}_div_{f2}"] = X[f1] / (X[f2].abs() + 1e-10)
    
    if 'value_std' in cols and 'value_mean' in cols:
        X_new['std_mean_ratio'] = X['value_std'] / (X['value_mean'].abs() + 1e-10)
    
    if 'diff_std' in cols and 'value_std' in cols:
        X_new['diff_value_std_ratio'] = X['diff_std'] / (X['value_std'] + 1e-10)
    
    if 'seg1_mean' in cols and 'seg3_mean' in cols:
        X_new['seg_trend'] = X['seg3_mean'] - X['seg1_mean']
    
    if 'outlier_ratio_3std' in cols and 'outlier_ratio_2std' in cols:
        X_new['outlier_severity'] = X['outlier_ratio_3std'] / (X['outlier_ratio_2std'] + 1e-10)
    
    if 'skewness' in cols and 'kurtosis' in cols:
        X_new['distribution_score'] = X['skewness'].abs() + X['kurtosis'].abs()
    
    X_new = X_new.replace([np.inf, -np.inf], 0).fillna(0)
    return X_new


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    seed: int,
    params: dict,
    training_config: dict,
    file_paths: pd.Series,
) -> tuple[RandomForestClassifier, dict, pd.DataFrame, pd.Series, pd.Series]:
    """训练随机森林二分类模型"""
    log_subsection("随机森林二分类模型训练 (Normal vs Abnormal)")
    logging.info("  样本总数: %d", X.shape[0])
    logging.info("  验证集比例: %.1f%%", test_size * 100)
    
    balance_strategy = training_config.get("balance_strategy", "undersample_1to1")
    use_feature_interactions = training_config.get("use_feature_interactions", True)
    
    logging.info("  平衡策略: %s", balance_strategy)
    
    if use_feature_interactions:
        logging.info("  添加特征交互项...")
        original_cols = X.shape[1]
        X = add_feature_interactions(X)
        logging.info("  特征维度: %d -> %d", original_cols, X.shape[1])
    
    X_train, X_valid, y_train, y_valid, paths_train, paths_valid = train_test_split(
        X, y, file_paths, test_size=test_size, random_state=seed, stratify=y,
    )
    
    train_normal = int((y_train == 0).sum())
    train_abnormal = int((y_train == 1).sum())
    logging.info("  训练集: %d 条（正常: %d，异常: %d）", X_train.shape[0], train_normal, train_abnormal)
    
    # 平衡策略
    if balance_strategy == "undersample_1to1":
        logging.info("  应用 1:1 下采样...")
        target_count = train_normal
        normal_idx = y_train[y_train == 0].index
        abnormal_idx = y_train[y_train == 1].index
        np.random.seed(seed)
        sampled_abnormal_idx = np.random.choice(abnormal_idx, size=target_count, replace=False)
        balanced_idx = np.concatenate([normal_idx.values, sampled_abnormal_idx])
        np.random.shuffle(balanced_idx)
        X_train = X_train.loc[balanced_idx].reset_index(drop=True)
        y_train = y_train.loc[balanced_idx].reset_index(drop=True)
        logging.info("  下采样后: %d 条", X_train.shape[0])
    
    # 更新参数
    merged_params = params.copy()
    merged_params["random_state"] = seed
    merged_params["n_jobs"] = -1
    
    logging.info("  模型参数:")
    for k, v in merged_params.items():
        logging.info("    - %s: %s", k, v)
    
    logging.info("  开始训练...")
    start_time = time.time()
    
    model = RandomForestClassifier(**merged_params)
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    logging.info("  训练完成（耗时: %.2f 秒）", elapsed)
    logging.info("  OOB Score: %.4f", model.oob_score_)
    
    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]
    
    metrics: dict = {}
    metrics["accuracy"] = float(accuracy_score(y_valid, y_pred))
    metrics["precision"] = float(precision_score(y_valid, y_pred))
    metrics["recall"] = float(recall_score(y_valid, y_pred))
    metrics["f1"] = float(f1_score(y_valid, y_pred))
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_valid, y_proba))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    
    metrics["classification_report"] = classification_report(y_valid, y_pred, target_names=["Normal", "Abnormal"])
    metrics["confusion_matrix"] = confusion_matrix(y_valid, y_pred).tolist()
    
    fpr, tpr, _ = roc_curve(y_valid, y_proba)
    metrics["roc_curve_fpr"] = fpr.tolist()
    metrics["roc_curve_tpr"] = tpr.tolist()
    
    logging.info("")
    logging.info("  随机森林二分类评估指标:")
    logging.info("    - 准确率 (Accuracy):  %.4f", metrics["accuracy"])
    logging.info("    - 精确率 (Precision): %.4f", metrics["precision"])
    logging.info("    - 召回率 (Recall):    %.4f", metrics["recall"])
    logging.info("    - F1 分数:            %.4f", metrics["f1"])
    logging.info("    - ROC-AUC:            %.4f", metrics["roc_auc"])
    
    # 特征重要性
    log_subsection("Top 20 特征重要性")
    feature_names = X.columns.tolist()
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    
    top20 = []
    for i, (fname, imp) in enumerate(feat_imp[:20]):
        category = get_feature_category(fname)
        logging.info("  %2d. %-35s %.4f  [%s]", i + 1, fname[:33], imp, category)
        top20.append({"rank": i + 1, "feature": fname, "importance": float(imp), "category": category})
    metrics["top20_features"] = top20
    
    return model, metrics, X_valid, y_valid, paths_valid


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, strategy: str = "f1") -> tuple[float, dict]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh, best_score = 0.5, 0
    best_metrics = {}
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if strategy == "f1" and f1 > best_score:
            best_score = f1
            best_thresh = thresh
            best_metrics = {"precision": precision, "recall": recall, "specificity": specificity, "f1": f1}
        elif strategy == "dual90":
            min_both = min(recall, specificity)
            if min_both > best_score:
                best_score = min_both
                best_thresh = thresh
                best_metrics = {"precision": precision, "recall": recall, "specificity": specificity, "f1": f1}
    
    return best_thresh, best_metrics


def _save_binary_visualizations(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_dir: str,
    model_name_zh: str,
    model_name_en: str,
    optimal_threshold: float,
) -> None:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

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

    def confusion_percent(y_pred: np.ndarray) -> np.ndarray:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        return np.nan_to_num(cm_pct, nan=0.0, posinf=0.0, neginf=0.0)

    y_pred_opt = (y_proba >= float(optimal_threshold)).astype(int)

    cm_pct = confusion_percent(y_pred_opt)
    for lang in ("zh", "en"):
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        labels = ["正常", "异常"] if lang == "zh" else ["Normal", "Abnormal"]
        title = f"{model_name_zh} 混淆矩阵(%)" if lang == "zh" else f"{model_name_en} Confusion Matrix (%)"
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
            ax.set_title(("ROC 曲线" if lang == "zh" else "ROC Curve") + f" ({model_name_zh if lang=='zh' else model_name_en})")
            ax.legend()
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"roc_{lang}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
    except Exception:
        save_placeholder("roc", f"ROC 曲线（{model_name_zh}）", f"ROC Curve ({model_name_en})")

    try:
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        for lang in ("zh", "en"):
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            ax.plot(rec, prec, label=f"AP={ap:.4f}")
            ax.set_xlabel("召回率" if lang == "zh" else "Recall")
            ax.set_ylabel("精确率" if lang == "zh" else "Precision")
            ax.set_title(("PR 曲线" if lang == "zh" else "PR Curve") + f" ({model_name_zh if lang=='zh' else model_name_en})")
            ax.legend()
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"pr_curve_{lang}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
    except Exception:
        save_placeholder("pr_curve", f"PR 曲线（{model_name_zh}）", f"PR Curve ({model_name_en})")

    try:
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
        brier = brier_score_loss(y_true, y_proba)
        for lang in ("zh", "en"):
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            ax.plot([0, 1], [0, 1], "k--", label=("理想校准" if lang == "zh" else "Perfect"))
            ax.plot(prob_pred, prob_true, marker="o", label=f"Brier={brier:.4f}")
            ax.set_xlabel("预测概率" if lang == "zh" else "Predicted Probability")
            ax.set_ylabel("真实正例比例" if lang == "zh" else "Fraction of Positives")
            ax.set_title(("校准曲线" if lang == "zh" else "Calibration Curve") + f" ({model_name_zh if lang=='zh' else model_name_en})")
            ax.legend()
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"calibration_curve_{lang}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
    except Exception:
        save_placeholder("calibration_curve", f"校准曲线（{model_name_zh}）", f"Calibration Curve ({model_name_en})")

    for lang in ("zh", "en"):
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.hist(y_proba[y_true == 0], bins=40, alpha=0.6, label=("正常" if lang == "zh" else "Normal"))
        ax.hist(y_proba[y_true == 1], bins=40, alpha=0.6, label=("异常" if lang == "zh" else "Abnormal"))
        ax.axvline(float(optimal_threshold), color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("预测为异常的概率" if lang == "zh" else "Predicted Abnormal Probability")
        ax.set_ylabel("样本数" if lang == "zh" else "Count")
        ax.set_title(("预测分数分布" if lang == "zh" else "Prediction Score Distribution") + f" ({model_name_zh if lang=='zh' else model_name_en})")
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
        ax.set_title(("阈值对比曲线" if lang == "zh" else "Metrics vs Threshold") + f" ({model_name_zh if lang=='zh' else model_name_en})")
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
        ax.set_title(("二分类指标汇总" if lang == "zh" else "Binary Metrics Summary") + f" ({model_name_zh if lang=='zh' else model_name_en})")
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"metrics_summary_{lang}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)

    try:
        ll = float(log_loss(y_true, y_proba, eps=1e-15))
        for lang in ("zh", "en"):
            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            ax.plot([1], [ll], marker="o")
            ax.set_xlabel("迭代轮数" if lang == "zh" else "Iteration")
            ax.set_ylabel("损失" if lang == "zh" else "Loss")
            ax.set_title(("损失曲线" if lang == "zh" else "Loss Curve") + f" ({model_name_zh if lang=='zh' else model_name_en})")
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"loss_curve_{lang}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
    except Exception:
        save_placeholder("loss_curve", f"损失曲线（{model_name_zh}）", f"Loss Curve ({model_name_en})")


def save_metrics_and_plots(
    metrics: dict,
    output_dir: str,
    y_true: np.ndarray | None = None,
    y_proba: np.ndarray | None = None,
    optimal_threshold: float | None = None,
) -> None:
    log_subsection("保存评估报告")
    os.makedirs(output_dir, exist_ok=True)
    setup_chinese_font()
    
    metrics_path = os.path.join(output_dir, "metrics_binary_rf.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logging.info("  评估指标: %s", metrics_path)
    
    cm = np.array(metrics.get("confusion_matrix", []))
    if cm.size > 0:
        cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        fig, ax = plt.subplots(figsize=(4, 4))
        labels = ["正常", "异常"]
        sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels, annot_kws={"size": 12})
        for t in ax.texts:
            t.set_text(t.get_text() + "%")
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        plt.title("随机森林二分类混淆矩阵 (%)")
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "confusion_binary_rf.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        logging.info("  混淆矩阵: %s", fig_path)
    
    fpr = metrics.get("roc_curve_fpr")
    tpr = metrics.get("roc_curve_tpr")
    if fpr and tpr:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(fpr, tpr, label="ROC")
        ax.plot([0, 1], [0, 1], "k--", label="随机")
        ax.set_xlabel("假阳性率")
        ax.set_ylabel("真阳性率")
        ax.set_title("随机森林 ROC 曲线")
        ax.legend()
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "roc_binary_rf.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        logging.info("  ROC曲线: %s", fig_path)

    if y_true is not None and y_proba is not None:
        thr = 0.5 if optimal_threshold is None else float(optimal_threshold)
        _save_binary_visualizations(
            y_true=np.asarray(y_true),
            y_proba=np.asarray(y_proba),
            output_dir=output_dir,
            model_name_zh="随机森林",
            model_name_en="Random Forest",
            optimal_threshold=thr,
        )


def save_model(model: RandomForestClassifier, feature_names: list[str], output_dir: str, seed: int, sampling_config: dict) -> str:
    log_subsection("保存模型文件")
    os.makedirs(output_dir, exist_ok=True)
    
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "label_mapping": {0: "Normal", 1: "Abnormal"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "sampling_config": sampling_config,
        "model_type": "RandomForest",
    }
    
    path = os.path.join(output_dir, "binary_model_rf.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logging.info("  模型文件: %s", path)
    
    return path


def run_training(data_dir: str, output_dir: str, test_size: float, seed: int,
                 binary_params: dict, sampling_config: dict, training_config: dict) -> None:
    total_start = time.time()
    set_global_seed(seed)
    setup_logging(output_dir)
    
    log_section("随机森林二分类模型训练")
    log_subsection("配置信息")
    logging.info("  数据目录: %s", data_dir)
    logging.info("  输出目录: %s", output_dir)
    logging.info("  随机种子: %d", seed)
    
    X, y, feature_names, file_paths = build_feature_table(data_dir, sampling_config)
    
    model, metrics, X_valid, y_valid, paths_valid = train_model(
        X, y, test_size, seed, binary_params, training_config, file_paths
    )
    
    y_proba = model.predict_proba(X_valid)[:, 1]
    optimal_threshold, opt_metrics = find_optimal_threshold(y_valid.values, y_proba, strategy="f1")
    logging.info("  推荐阈值: %.2f (F1=%.4f)", optimal_threshold, opt_metrics.get("f1", 0))

    reports_dir = os.path.join(output_dir, "reports")
    save_metrics_and_plots(metrics, reports_dir, y_true=y_valid.values, y_proba=y_proba, optimal_threshold=optimal_threshold)
    
    models_dir = os.path.join(output_dir, "models")
    save_model(model, feature_names, models_dir, seed, sampling_config)
    
    # 保存最优阈值
    threshold_path = os.path.join(models_dir, "optimal_threshold.json")
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({"optimal_threshold": optimal_threshold}, f, indent=2)
    
    total_elapsed = time.time() - total_start
    log_section("训练完成")
    logging.info("  总耗时: %.2f 秒", total_elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="随机森林二分类模型训练")
    parser.add_argument("--data-dir", type=str, default=None, help="数据目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--test-size", type=float, default=None, help="验证集比例")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    
    args = parser.parse_args()
    
    seed, cfg_test_size, binary_params, sampling_config, training_config, cfg_data_dir, cfg_output_dir = load_config(args.config)
    
    test_size = cfg_test_size if args.test_size is None else args.test_size
    data_dir = args.data_dir if args.data_dir is not None else cfg_data_dir
    output_dir = args.output_dir if args.output_dir is not None else cfg_output_dir
    
    run_training(data_dir, output_dir, test_size, seed, binary_params, sampling_config, training_config)


if __name__ == "__main__":
    main()
