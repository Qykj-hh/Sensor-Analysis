"""
SVM多分类模型训练模块
训练异常场景分类模型（区分不同类型的异常）
用于与 LightGBM、XGBoost 和 Random Forest 进行对比实验
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
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

DEFAULT_MULTICLASS_PARAMS: dict[str, float | int | str] = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "probability": True,  # 启用概率预测
    "random_state": SEED,
    "max_iter": 10000,  # 增加最大迭代次数
    "decision_function_shape": "ovr",  # one-vs-rest策略
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
    "value_max": "基础统计特征", "value_median": "基础统计特征", "value_abs_mean": "基础统计特征",
    "value_abs_max": "基础统计特征", "value_first": "基础统计特征", "value_last": "基础统计特征",
    "value_rms": "基础统计特征", "value_energy": "基础统计特征",
    "value_std": "离散程度特征", "value_range": "离散程度特征", "value_iqr": "离散程度特征", "value_cv": "离散程度特征",
    "value_skew": "分布形态特征", "value_kurt": "分布形态特征",
    "value_q01": "分布形态特征", "value_q05": "分布形态特征", "value_q10": "分布形态特征",
    "value_q25": "分布形态特征", "value_q75": "分布形态特征", "value_q90": "分布形态特征",
    "value_q95": "分布形态特征", "value_q99": "分布形态特征",
    "diff_mean": "差分特征", "diff_std": "差分特征", "diff_min": "差分特征", "diff_max": "差分特征",
    "diff_abs_mean": "差分特征", "diff_abs_max": "差分特征", "diff_range": "差分特征",
    "diff_skew": "差分特征", "diff_kurt": "差分特征",
    "diff2_mean": "差分特征", "diff2_std": "差分特征", "diff2_abs_mean": "差分特征", "sign_change_rate": "差分特征",
    "value_change": "趋势特征", "value_change_rate": "趋势特征", "trend_slope": "趋势特征",
    "linear_trend_slope": "趋势特征", "linear_trend_intercept": "趋势特征",
    "trend_residual_std": "趋势特征", "trend_r2": "趋势特征",
    "segment_mean_first": "局部特征", "segment_mean_mid": "局部特征", "segment_mean_last": "局部特征",
    "segment_std_first": "局部特征", "segment_std_last": "局部特征",
    "local_peaks_ratio": "局部特征", "local_valleys_ratio": "局部特征",
    "max_jump": "局部特征", "max_jump_idx_ratio": "局部特征", "jump_count_ratio": "局部特征", "direction_change_ratio": "局部特征",
    "rolling_std_mean": "稳定性特征", "rolling_std_max": "稳定性特征", "rolling_std_range": "稳定性特征",
    "value_autocorr_1": "自相关特征", "value_autocorr_2": "自相关特征",
    "value_autocorr_5": "自相关特征", "value_autocorr_10": "自相关特征",
    "fft_spectral_entropy": "信息熵特征",
    "fft_peak_freq": "频域特征", "fft_peak_mag": "频域特征", "fft_total_power": "频域特征",
    "fft_low_high_ratio": "频域特征", "fft_spectral_centroid": "频域特征",
    "fft_spectral_spread": "频域特征", "fft_spectral_flatness": "频域特征", "fft_spectral_rolloff": "频域特征",
    "fft_band0_power": "频域特征", "fft_band1_power": "频域特征",
    "fft_band2_power": "频域特征", "fft_band3_power": "频域特征",
    "dt_mean_ms": "时间间隔特征", "dt_std_ms": "时间间隔特征", "dt_min_ms": "时间间隔特征",
    "dt_max_ms": "时间间隔特征", "dt_cv": "时间间隔特征", "dt_range_ms": "时间间隔特征",
    "outlier_ratio_3std": "异常指标特征", "zero_ratio": "异常指标特征", "negative_ratio": "异常指标特征",
}


def get_feature_category(feature_name: str) -> str:
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
    log_path = os.path.join(output_dir, "train_multiclass_svm.log")
    
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


def load_config(path: str | None) -> tuple[int, float, dict, dict, str, str]:
    seed = SEED
    test_size = 0.15
    multiclass_params = DEFAULT_MULTICLASS_PARAMS.copy()
    sampling_config = DEFAULT_SAMPLING_CONFIG.copy()
    data_dir = "test-datasets/Abnormal"
    output_dir = "artifacts/svm/multiclass"
    
    if path and os.path.isfile(path):
        logging.info("  配置文件: %s", path)
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "seed" in cfg:
            seed = int(cfg["seed"])
        if "test_size" in cfg:
            test_size = float(cfg["test_size"])
        if "multiclass_params" in cfg and isinstance(cfg["multiclass_params"], dict):
            filtered = {k: v for k, v in cfg["multiclass_params"].items() if not k.startswith("_")}
            multiclass_params.update(filtered)
        if "sampling" in cfg and isinstance(cfg["sampling"], dict):
            filtered = {k: v for k, v in cfg["sampling"].items() if not k.startswith("_")}
            sampling_config.update(filtered)
        if "data_dir" in cfg:
            data_dir = str(cfg["data_dir"])
        if "output_dir" in cfg:
            output_dir = str(cfg["output_dir"])
    else:
        logging.info("  配置文件: 使用默认配置")
    
    return seed, test_size, multiclass_params, sampling_config, data_dir, output_dir


def find_parquet_files(base_dir: str, verbose: bool = False) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    scenario_counts: dict[str, int] = {}
    
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"数据目录不存在: {base_dir}")
    
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".parquet"):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, base_dir)
                parts = rel.split(os.sep)
                scenario = parts[0] if len(parts) > 1 else os.path.basename(base_dir)
                results.append((scenario, full))
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    
    if verbose:
        logging.info("  总文件数: %d 个", len(results))
        if scenario_counts:
            logging.info("  场景分布:")
            for sc, cnt in sorted(scenario_counts.items()):
                logging.info("    - %s: %d 个文件", sc, cnt)
    
    return results


def extract_features_from_series(values: pd.Series, epochs: pd.Series) -> dict:
    """从时序数据中提取统计特征（与 LightGBM 多分类版本一致）"""
    v = values.astype(float)
    feats: dict[str, float] = {}
    feats["value_count"] = float(v.shape[0])
    v_valid = v.dropna()
    
    if not v_valid.empty:
        feats["value_mean"] = float(v_valid.mean())
        feats["value_std"] = float(v_valid.std())
        feats["value_min"] = float(v_valid.min())
        feats["value_max"] = float(v_valid.max())
        feats["value_median"] = float(v_valid.median())
        feats["value_q01"] = float(v_valid.quantile(0.01))
        feats["value_q05"] = float(v_valid.quantile(0.05))
        feats["value_q95"] = float(v_valid.quantile(0.95))
        feats["value_q99"] = float(v_valid.quantile(0.99))
        feats["value_skew"] = float(v_valid.skew())
        feats["value_kurt"] = float(v_valid.kurt())
        feats["value_abs_mean"] = float(v_valid.abs().mean())
        feats["value_abs_max"] = float(v_valid.abs().max())
        
        feats["value_range"] = float(v_valid.max() - v_valid.min())
        feats["value_iqr"] = float(v_valid.quantile(0.75) - v_valid.quantile(0.25))
        feats["value_cv"] = float(v_valid.std() / (abs(v_valid.mean()) + 1e-10))
        feats["value_q10"] = float(v_valid.quantile(0.10))
        feats["value_q25"] = float(v_valid.quantile(0.25))
        feats["value_q75"] = float(v_valid.quantile(0.75))
        feats["value_q90"] = float(v_valid.quantile(0.90))
        
        threshold_3std = abs(v_valid.mean()) + 3 * v_valid.std()
        feats["outlier_ratio_3std"] = float((v_valid.abs() > threshold_3std).sum() / len(v_valid))
        feats["zero_ratio"] = float((v_valid == 0).sum() / len(v_valid))
        feats["negative_ratio"] = float((v_valid < 0).sum() / len(v_valid))
        
        feats["value_rms"] = float(np.sqrt((v_valid ** 2).mean()))
        feats["value_energy"] = float((v_valid ** 2).sum())
        
        # 频域特征
        arr = v_valid.values
        n_pts = len(arr)
        if n_pts >= 8:
            try:
                fft_vals = np.fft.rfft(arr - arr.mean())
                fft_mag = np.abs(fft_vals)
                freqs = np.fft.rfftfreq(n_pts)
                
                if len(fft_mag) > 1:
                    peak_idx = np.argmax(fft_mag[1:]) + 1
                    feats["fft_peak_freq"] = float(freqs[peak_idx])
                    feats["fft_peak_mag"] = float(fft_mag[peak_idx])
                else:
                    feats["fft_peak_freq"] = 0.0
                    feats["fft_peak_mag"] = 0.0
                
                total_power = float(np.sum(fft_mag ** 2))
                feats["fft_total_power"] = total_power
                
                power_norm = fft_mag ** 2 / (total_power + 1e-10)
                feats["fft_spectral_entropy"] = float(-np.sum(power_norm * np.log(power_norm + 1e-10)))
                
                mid = len(fft_mag) // 2
                low_power = float(np.sum(fft_mag[:mid] ** 2))
                high_power = float(np.sum(fft_mag[mid:] ** 2))
                feats["fft_low_high_ratio"] = low_power / (high_power + 1e-10)
                
                feats["fft_spectral_centroid"] = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10))
                
                centroid = feats["fft_spectral_centroid"]
                feats["fft_spectral_spread"] = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_mag) / (np.sum(fft_mag) + 1e-10)))
                
                geo_mean = np.exp(np.mean(np.log(fft_mag + 1e-10)))
                arith_mean = np.mean(fft_mag)
                feats["fft_spectral_flatness"] = float(geo_mean / (arith_mean + 1e-10))
                
                n_bands = 4
                band_size = len(fft_mag) // n_bands
                for bi in range(n_bands):
                    start_idx = bi * band_size
                    end_idx = (bi + 1) * band_size if bi < n_bands - 1 else len(fft_mag)
                    band_power = float(np.sum(fft_mag[start_idx:end_idx] ** 2))
                    feats[f"fft_band{bi}_power"] = band_power / (total_power + 1e-10)
                
                cumsum = np.cumsum(fft_mag ** 2)
                rolloff_thresh = 0.85 * total_power
                rolloff_idx = np.searchsorted(cumsum, rolloff_thresh)
                feats["fft_spectral_rolloff"] = float(rolloff_idx / len(fft_mag))
            except Exception:
                for key in ["fft_peak_freq", "fft_peak_mag", "fft_total_power", "fft_spectral_entropy",
                           "fft_low_high_ratio", "fft_spectral_centroid", "fft_spectral_spread",
                           "fft_spectral_flatness", "fft_spectral_rolloff"]:
                    feats[key] = 0.0
                for bi in range(4):
                    feats[f"fft_band{bi}_power"] = 0.0
        else:
            for key in ["fft_peak_freq", "fft_peak_mag", "fft_total_power", "fft_spectral_entropy",
                       "fft_low_high_ratio", "fft_spectral_centroid", "fft_spectral_spread",
                       "fft_spectral_flatness", "fft_spectral_rolloff"]:
                feats[key] = 0.0
            for bi in range(4):
                feats[f"fft_band{bi}_power"] = 0.0
        
        # 局部特征
        n_pts = len(v_valid)
        if n_pts >= 10:
            q1_end = n_pts // 4
            q3_start = 3 * n_pts // 4
            feats["segment_mean_first"] = float(v_valid.iloc[:q1_end].mean())
            feats["segment_mean_mid"] = float(v_valid.iloc[q1_end:q3_start].mean())
            feats["segment_mean_last"] = float(v_valid.iloc[q3_start:].mean())
            feats["segment_std_first"] = float(v_valid.iloc[:q1_end].std())
            feats["segment_std_last"] = float(v_valid.iloc[q3_start:].std())
            
            arr_local = v_valid.values
            peaks, valleys = 0, 0
            for i in range(1, len(arr_local) - 1):
                if arr_local[i] > arr_local[i-1] and arr_local[i] > arr_local[i+1]:
                    peaks += 1
                elif arr_local[i] < arr_local[i-1] and arr_local[i] < arr_local[i+1]:
                    valleys += 1
            feats["local_peaks_ratio"] = float(peaks / n_pts)
            feats["local_valleys_ratio"] = float(valleys / n_pts)
        else:
            feats["segment_mean_first"] = float(v_valid.mean())
            feats["segment_mean_mid"] = float(v_valid.mean())
            feats["segment_mean_last"] = float(v_valid.mean())
            feats["segment_std_first"] = 0.0
            feats["segment_std_last"] = 0.0
            feats["local_peaks_ratio"] = 0.0
            feats["local_valleys_ratio"] = 0.0
        
        # 滚动窗口统计
        if n_pts >= 20:
            local_window = n_pts // 5
            rolling_std = v_valid.rolling(window=local_window, min_periods=1).std()
            feats["rolling_std_mean"] = float(rolling_std.mean())
            feats["rolling_std_max"] = float(rolling_std.max())
            feats["rolling_std_range"] = float(rolling_std.max() - rolling_std.min())
        else:
            feats["rolling_std_mean"] = 0.0
            feats["rolling_std_max"] = 0.0
            feats["rolling_std_range"] = 0.0
        
        diffs = v_valid.diff().dropna()
    else:
        default_feats = ["value_mean", "value_std", "value_min", "value_max", "value_median",
                        "value_q01", "value_q05", "value_q95", "value_q99", "value_skew", "value_kurt",
                        "value_abs_mean", "value_abs_max", "value_range", "value_iqr", "value_cv",
                        "value_q10", "value_q25", "value_q75", "value_q90",
                        "outlier_ratio_3std", "zero_ratio", "negative_ratio", "value_rms", "value_energy",
                        "fft_peak_freq", "fft_peak_mag", "fft_total_power", "fft_spectral_entropy",
                        "fft_low_high_ratio", "fft_spectral_centroid", "fft_spectral_spread",
                        "fft_spectral_flatness", "fft_spectral_rolloff",
                        "segment_mean_first", "segment_mean_mid", "segment_mean_last",
                        "segment_std_first", "segment_std_last", "local_peaks_ratio", "local_valleys_ratio",
                        "rolling_std_mean", "rolling_std_max", "rolling_std_range"]
        for key in default_feats:
            feats[key] = 0.0
        for bi in range(4):
            feats[f"fft_band{bi}_power"] = 0.0
        diffs = pd.Series(dtype=float)
    
    # 差分特征
    if not diffs.empty:
        feats["diff_mean"] = float(diffs.mean())
        feats["diff_std"] = float(diffs.std())
        feats["diff_min"] = float(diffs.min())
        feats["diff_max"] = float(diffs.max())
        feats["diff_abs_mean"] = float(diffs.abs().mean())
        feats["diff_abs_max"] = float(diffs.abs().max())
        feats["diff_range"] = float(diffs.max() - diffs.min())
        feats["diff_skew"] = float(diffs.skew()) if len(diffs) > 2 else 0.0
        feats["diff_kurt"] = float(diffs.kurt()) if len(diffs) > 3 else 0.0
        
        if len(diffs) > 1:
            sign_changes = ((diffs.iloc[:-1].values * diffs.iloc[1:].values) < 0).sum()
            feats["sign_change_rate"] = float(sign_changes / (len(diffs) - 1))
        else:
            feats["sign_change_rate"] = 0.0
        
        diffs2 = diffs.diff().dropna()
        if not diffs2.empty:
            feats["diff2_mean"] = float(diffs2.mean())
            feats["diff2_std"] = float(diffs2.std())
            feats["diff2_abs_mean"] = float(diffs2.abs().mean())
        else:
            feats["diff2_mean"] = 0.0
            feats["diff2_std"] = 0.0
            feats["diff2_abs_mean"] = 0.0
    else:
        for key in ["diff_mean", "diff_std", "diff_min", "diff_max", "diff_abs_mean", "diff_abs_max",
                   "diff_range", "diff_skew", "diff_kurt", "sign_change_rate",
                   "diff2_mean", "diff2_std", "diff2_abs_mean"]:
            feats[key] = 0.0
    
    # 时间间隔特征
    e = epochs.astype("int64")
    delta = e.diff().dropna()
    if not delta.empty:
        feats["dt_mean_ms"] = float(delta.mean())
        feats["dt_std_ms"] = float(delta.std())
        feats["dt_min_ms"] = float(delta.min())
        feats["dt_max_ms"] = float(delta.max())
        feats["dt_cv"] = float(delta.std() / (delta.mean() + 1e-10))
        feats["dt_range_ms"] = float(delta.max() - delta.min())
    else:
        for key in ["dt_mean_ms", "dt_std_ms", "dt_min_ms", "dt_max_ms", "dt_cv", "dt_range_ms"]:
            feats[key] = 0.0
    
    # 首尾变化特征
    if v_valid.shape[0] > 1:
        feats["value_first"] = float(v_valid.iloc[0])
        feats["value_last"] = float(v_valid.iloc[-1])
        feats["value_change"] = float(v_valid.iloc[-1] - v_valid.iloc[0])
        feats["value_change_rate"] = float(feats["value_change"] / (abs(feats["value_first"]) + 1e-10))
        
        n = len(v_valid)
        mid = n // 2
        first_half_mean = float(v_valid.iloc[:mid].mean())
        second_half_mean = float(v_valid.iloc[mid:].mean())
        feats["trend_slope"] = float(second_half_mean - first_half_mean)
        
        # 自相关特征
        if n > 2 and v_valid.std() > 1e-10:
            try:
                ac = v_valid.autocorr(lag=1)
                feats["value_autocorr_1"] = float(ac) if not np.isnan(ac) else 0.0
            except Exception:
                feats["value_autocorr_1"] = 0.0
            
            for lag in [2, 5, 10]:
                if n > lag:
                    try:
                        ac_lag = v_valid.autocorr(lag=lag)
                        feats[f"value_autocorr_{lag}"] = float(ac_lag) if not np.isnan(ac_lag) else 0.0
                    except Exception:
                        feats[f"value_autocorr_{lag}"] = 0.0
                else:
                    feats[f"value_autocorr_{lag}"] = 0.0
        else:
            for lag in [1, 2, 5, 10]:
                feats[f"value_autocorr_{lag}"] = 0.0
        
        # 线性回归趋势
        if n >= 5:
            x = np.arange(n)
            try:
                slope, intercept = np.polyfit(x, v_valid.values, 1)
                feats["linear_trend_slope"] = float(slope)
                feats["linear_trend_intercept"] = float(intercept)
                predicted = slope * x + intercept
                residuals = v_valid.values - predicted
                feats["trend_residual_std"] = float(np.std(residuals))
                feats["trend_r2"] = float(1 - np.var(residuals) / (np.var(v_valid.values) + 1e-10))
            except Exception:
                feats["linear_trend_slope"] = 0.0
                feats["linear_trend_intercept"] = 0.0
                feats["trend_residual_std"] = 0.0
                feats["trend_r2"] = 0.0
        else:
            feats["linear_trend_slope"] = 0.0
            feats["linear_trend_intercept"] = 0.0
            feats["trend_residual_std"] = 0.0
            feats["trend_r2"] = 0.0
        
        # 突变检测特征
        if n >= 10:
            arr_v = v_valid.values
            abs_diffs = np.abs(np.diff(arr_v))
            feats["max_jump"] = float(np.max(abs_diffs))
            feats["max_jump_idx_ratio"] = float(np.argmax(abs_diffs) / n)
            jump_threshold = 3 * np.std(abs_diffs)
            feats["jump_count_ratio"] = float(np.sum(abs_diffs > jump_threshold) / n)
            signs = np.sign(np.diff(arr_v))
            sign_changes = np.sum(signs[:-1] != signs[1:])
            feats["direction_change_ratio"] = float(sign_changes / n)
        else:
            feats["max_jump"] = 0.0
            feats["max_jump_idx_ratio"] = 0.0
            feats["jump_count_ratio"] = 0.0
            feats["direction_change_ratio"] = 0.0
    else:
        for key in ["value_first", "value_last", "value_change", "value_change_rate", "trend_slope",
                   "value_autocorr_1", "value_autocorr_2", "value_autocorr_5", "value_autocorr_10",
                   "linear_trend_slope", "linear_trend_intercept", "trend_residual_std", "trend_r2",
                   "max_jump", "max_jump_idx_ratio", "jump_count_ratio", "direction_change_ratio"]:
            feats[key] = 0.0
    
    # 清理无效值
    for key in feats:
        if isinstance(feats[key], float) and (np.isnan(feats[key]) or np.isinf(feats[key])):
            feats[key] = 0.0
    
    return feats


def generate_window_records(df: pd.DataFrame, scenario: str, label_idx: int, path: str, sampling_config: dict) -> list[dict]:
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
            feats["label"] = label_idx
            feats["scenario"] = scenario
            records.append(feats)
        
        start += step_ms
    
    return records


def build_feature_table(base_dir: str, sampling_config: dict) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, int], pd.Series]:
    log_subsection("数据扫描")
    logging.info("  数据目录: %s", base_dir)
    paths = find_parquet_files(base_dir, verbose=True)
    
    if not paths:
        raise RuntimeError("未找到任何 Parquet 文件")
    
    scenarios = sorted({scenario for scenario, _ in paths})
    if len(scenarios) < 2:
        raise RuntimeError(f"至少需要 2 个异常场景类别，当前只有: {scenarios}")
    
    label_mapping: dict[str, int] = {scenario: idx for idx, scenario in enumerate(scenarios)}
    logging.info("  场景类别数: %d", len(scenarios))
    for scenario, idx in label_mapping.items():
        logging.info("    - %s -> %d", scenario, idx)
    
    mode = str(sampling_config.get("mode", "window"))
    num_workers = int(sampling_config.get("num_workers", 0))
    
    log_subsection("特征构建")
    logging.info("  采样模式: %s", mode)
    
    total_files = len(paths)
    start_time = time.time()
    records: list[dict] = []
    processed_files = [0]
    skipped_files = [0]
    lock = threading.Lock()

    def process_one(item: tuple[str, str]) -> list[dict]:
        scenario, path = item
        label_idx = label_mapping[scenario]
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
            result = generate_window_records(df, scenario, label_idx, path, sampling_config)
        else:
            feats = extract_features_from_series(df["value"], df["epoch_ms"])
            feats["file_path"] = path
            feats["pv_name"] = str(df["pv_name"].iloc[0]) if "pv_name" in df.columns else ""
            feats["label"] = label_idx
            feats["scenario"] = scenario
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
                   ["file_path", "pv_name", "window_start_epoch_ms", "window_end_epoch_ms", "label", "scenario"]]
    
    X = table[feature_cols].astype(np.float32)
    y = table["label"].astype(np.int8)
    file_paths = table["file_path"].reset_index(drop=True)
    
    logging.info("  特征构建完成（耗时: %.2f 秒）", elapsed)
    logging.info("  生成样本: %d 条", X.shape[0])
    logging.info("  特征维度: %d", len(feature_cols))
    
    class_counts = y.value_counts().sort_index()
    logging.info("  各类别样本分布:")
    for label_idx, count in class_counts.items():
        label_name = [k for k, v in label_mapping.items() if v == label_idx]
        label_name = label_name[0] if label_name else str(label_idx)
        logging.info("    - %s: %d 条", label_name, count)
    
    return X, y, feature_cols, label_mapping, file_paths


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    label_mapping: dict[str, int],
    test_size: float,
    seed: int,
    params: dict,
    file_paths: pd.Series,
) -> tuple[SVC, StandardScaler, dict, pd.DataFrame, pd.Series, np.ndarray, pd.Series]:
    """训练SVM多分类模型"""
    log_subsection("SVM多分类模型训练 (异常场景识别)")
    logging.info("  样本总数: %d", X.shape[0])
    logging.info("  场景类别数: %d", len(label_mapping))
    logging.info("  验证集比例: %.1f%%", test_size * 100)
    
    X_train, X_valid, y_train, y_valid, paths_train, paths_valid = train_test_split(
        X, y, file_paths, test_size=test_size, random_state=seed, stratify=y,
    )
    logging.info("  训练集: %d 条，验证集: %d 条", X_train.shape[0], X_valid.shape[0])
    
    # 特征缩放
    logging.info("  应用特征标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    merged_params = params.copy()
    merged_params["random_state"] = seed
    
    logging.info("  模型参数:")
    for k, v in merged_params.items():
        logging.info("    - %s: %s", k, v)
    
    logging.info("  开始训练...")
    start_time = time.time()
    
    model = SVC(**merged_params)
    model.fit(X_train_scaled, y_train)
    
    elapsed = time.time() - start_time
    logging.info("  训练完成（耗时: %.2f 秒）", elapsed)
    
    y_pred = model.predict(X_valid_scaled)
    y_pred_proba = model.predict_proba(X_valid_scaled)
    
    metrics: dict = {}
    metrics["accuracy"] = float(accuracy_score(y_valid, y_pred))
    metrics["macro_precision"] = float(precision_score(y_valid, y_pred, average="macro"))
    metrics["macro_recall"] = float(recall_score(y_valid, y_pred, average="macro"))
    metrics["macro_f1"] = float(f1_score(y_valid, y_pred, average="macro"))
    
    labels_sorted = sorted(label_mapping.items(), key=lambda x: x[1])
    target_names = [name for name, _ in labels_sorted]
    metrics["classification_report"] = classification_report(
        y_valid, y_pred, labels=list(range(len(target_names))), target_names=target_names,
    )
    metrics["confusion_matrix"] = confusion_matrix(y_valid, y_pred).tolist()
    
    logging.info("")
    logging.info("  SVM多分类评估指标:")
    logging.info("    - 准确率 (Accuracy):      %.4f", metrics["accuracy"])
    logging.info("    - 宏精确率 (Macro-P):    %.4f", metrics["macro_precision"])
    logging.info("    - 宏召回率 (Macro-R):    %.4f", metrics["macro_recall"])
    logging.info("    - 宏 F1 分数 (Macro-F1): %.4f", metrics["macro_f1"])
    
    # Top 20 特征重要性（如果可用）
    log_subsection("Top 20 特征重要性")
    feature_names = X.columns.tolist()
    top20 = []
    if hasattr(model, 'coef_') and model.coef_ is not None:
        # 对于多分类问题，coef_ 是一个形状为 (n_classes, n_features) 的数组
        # 我们可以计算每个特征在所有类别中的平均重要性
        coef_abs = np.abs(model.coef_)
        avg_coef = np.mean(coef_abs, axis=0)  # 按类别平均
        feat_imp = sorted(zip(feature_names, avg_coef), key=lambda x: x[1], reverse=True)
        
        for i, (fname, imp) in enumerate(feat_imp[:20]):
            category = get_feature_category(fname)
            logging.info("  %2d. %-35s %.4f  [%s]", i + 1, fname[:33], imp, category)
            top20.append({"rank": i + 1, "feature": fname, "importance": float(imp), "category": category})
    else:
        # 如果模型没有系数（例如使用非线性核），则无法提供有意义的特征重要性
        logging.info("  SVM模型无法提供特征重要性（核函数非线性）")
        for i, fname in enumerate(feature_names[:20]):
            category = get_feature_category(fname)
            top20.append({"rank": i + 1, "feature": fname, "importance": 0.0, "category": category})
    metrics["top20_features"] = top20
    
    return model, scaler, metrics, X_valid, y_valid, y_pred_proba, paths_valid


def save_metrics_and_plots(metrics: dict, label_mapping: dict[str, int], output_dir: str) -> None:
    log_subsection("保存评估报告")
    os.makedirs(output_dir, exist_ok=True)
    setup_chinese_font()
    
    metrics_path = os.path.join(output_dir, "metrics_multiclass_svm.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logging.info("  评估指标: %s", metrics_path)
    
    cm = np.array(metrics.get("confusion_matrix", []))
    if cm.size > 0:
        cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        labels_sorted = sorted(label_mapping.items(), key=lambda x: x[1])
        target_names = [name for name, _ in labels_sorted]
        
        fig, ax = plt.subplots(figsize=(max(6, len(target_names)), max(5, len(target_names) * 0.8)))
        sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=target_names, yticklabels=target_names, annot_kws={"size": 10})
        for t in ax.texts:
            t.set_text(t.get_text() + "%")
        plt.title("SVM多分类混淆矩阵 (%)")
        plt.xlabel("预测类别")
        plt.ylabel("真实类别")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "confusion_multiclass_svm.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        logging.info("  混淆矩阵: %s", fig_path)


def save_model(model: SVC, scaler: StandardScaler, feature_names: list[str], label_mapping: dict[str, int],
               output_dir: str, seed: int, sampling_config: dict) -> str:
    log_subsection("保存模型文件")
    os.makedirs(output_dir, exist_ok=True)
    
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "label_mapping": label_mapping,
        "inv_label_mapping": {v: k for k, v in label_mapping.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "sampling_config": sampling_config,
        "model_type": "SVM",
    }
    
    path = os.path.join(output_dir, "multiclass_model_svm.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logging.info("  模型文件: %s", path)
    
    return path


def run_training(data_dir: str, output_dir: str, test_size: float, seed: int,
                 multiclass_params: dict, sampling_config: dict) -> None:
    total_start = time.time()
    set_global_seed(seed)
    setup_logging(output_dir)
    
    log_section("SVM多分类模型训练")
    log_subsection("配置信息")
    logging.info("  数据目录: %s", data_dir)
    logging.info("  输出目录: %s", output_dir)
    logging.info("  随机种子: %d", seed)
    
    X, y, feature_names, label_mapping, file_paths = build_feature_table(data_dir, sampling_config)
    
    model, scaler, metrics, X_valid, y_valid, y_pred_proba, paths_valid = train_model(
        X, y, label_mapping, test_size, seed, multiclass_params, file_paths
    )
    
    reports_dir = os.path.join(output_dir, "reports")
    save_metrics_and_plots(metrics, label_mapping, reports_dir)
    
    models_dir = os.path.join(output_dir, "models")
    save_model(model, scaler, feature_names, label_mapping, models_dir, seed, sampling_config)
    
    total_elapsed = time.time() - total_start
    log_section("训练完成")
    logging.info("  总耗时: %.2f 秒", total_elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="SVM多分类模型训练")
    parser.add_argument("--data-dir", type=str, default=None, help="数据目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--test-size", type=float, default=None, help="验证集比例")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    
    args = parser.parse_args()
    
    seed, cfg_test_size, multiclass_params, sampling_config, cfg_data_dir, cfg_output_dir = load_config(args.config)
    
    test_size = cfg_test_size if args.test_size is None else args.test_size
    data_dir = args.data_dir if args.data_dir is not None else cfg_data_dir
    output_dir = args.output_dir if args.output_dir is not None else cfg_output_dir
    
    run_training(data_dir, output_dir, test_size, seed, multiclass_params, sampling_config)


if __name__ == "__main__":
    main()