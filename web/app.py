"""
Beam Abort Prediction Visualization System
Flask backend for loading real sensor data and LightGBM model prediction
"""

from flask import Flask, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
from scipy import stats
from scipy.fft import fft
import lightgbm as lgb

app = Flask(__name__, static_folder='.', template_folder='.')


# Custom classes for model loading (must match training code)
class EnsembleLGBMClassifier:
    """LightGBM ensemble classifier"""
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


class FocalLossLGBMClassifier:
    """LightGBM classifier with Focal Loss"""
    def __init__(self, booster, feature_names=None, threshold=0.5):
        self.booster = booster
        self.feature_names_in_ = feature_names or []
        self.threshold = threshold
        self.classes_ = np.array([0, 1])
    
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        raw_preds = self.booster.predict(X)
        proba = 1.0 / (1.0 + np.exp(-raw_preds))
        return np.column_stack([1 - proba, proba])
    
    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

# 20 specified data files
DATA_FILES = [
    'monitor_BMH_MQF2E31_POS.PYP_2024',
    'BT_BTpBPM_QMD11P_K_2_XPOS_2024',
    'others_CGHOBT_PMD12H1_Y_2024',
    'BT_BTpBPM_QAF8P_K_2_XPOS_2024',
    'monitor_BMH_MQEAE28_POS.PYP_2024',
    'BT_BTpBPM_QXF1P_A_1_NC_2024',
    'monitor_BMD_QF6_XM_2024',
    'BT_BTpBPM_QCD6P_A_1_XPOS_2024',
    'monitor_BMH_MQT4FOE2_POS.PYP_2024',
    'BT_BTnBPM_QMD8N_1_NC_2024',
    'monitor_BMH_MQF6E5_POS.PYP_2024',
    'BT_BTpBPM_QAF2P_K_2_NC_2024',
    'monitor_BMH_MQTANFE2_POS.PYP_2024',
    'BT_BTpBPM_QTD4P_K_2_YPOS_2024',
    'monitor_BMH_MQX6RE_POS.PYP_2024',
    'BT_BTpBPM_QMD11P_K_1_NC_2024',
    'monitor_BMH_MQF2E6_POS.PYP_2024',
    'BT_BTpBPM_QCF3P_K_1_XPOS_2024',
    'monitor_BMH_MQEAE24_POS.PYP_2024',
    'BT_BTpBPM_QBD2P_K_2_YPOS_2024'
]

DATA_DIR = 'dataset'
MODEL_DIR = 'artifacts copy'

# Cache
sensor_data_cache = None
abort_events_cache = None
binary_model = None
multiclass_model = None
prediction_cache = None

# Class labels for multiclass
CLASS_LABELS = ['Normal', 'Belle2 CLAWS', 'Belle2 VXD diamond', 'CLAWS D05V1', 
                'High THR VXD diamond', 'Loss Monitor', 'Low THR VXD diamond', 
                'Manual Abort', 'RF']

# Warning threshold for abnormal probability
WARNING_THRESHOLD = 0.85
CRITICAL_THRESHOLD = 0.9

# Feature names required by the binary model
FEATURE_NAMES = [
    "value_count", "value_mean", "value_std", "value_min", "value_max",
    "value_median", "value_q01", "value_q05", "value_q10", "value_q25",
    "value_q75", "value_q90", "value_q95", "value_q99", "value_range",
    "value_iqr", "value_cv", "value_rms", "outlier_ratio_3std", "outlier_ratio_2std",
    "zero_ratio", "negative_ratio", "positive_ratio", "skewness", "kurtosis",
    "zero_crossing_rate", "diff_mean", "diff_std", "diff_abs_mean", "diff_max",
    "diff_min", "diff_abs_max", "direction_changes", "direction_change_rate",
    "diff2_mean", "diff2_std", "diff2_abs_mean", "value_first", "value_last",
    "value_change", "value_change_ratio", "seg1_mean", "seg2_mean", "seg3_mean",
    "seg1_std", "seg2_std", "seg3_std", "seg_mean_diff_12", "seg_mean_diff_23",
    "seg_mean_diff_13", "trend_slope", "local_max_count", "local_min_count",
    "peak_count", "peak_rate", "rolling_var_mean", "rolling_var_std", "rolling_var_max",
    "autocorr_lag1", "autocorr_lag5", "entropy", "fft_dc_ratio", "fft_low_freq_ratio",
    "fft_high_freq_ratio", "fft_mid_freq_ratio", "fft_dominant_freq_idx",
    "fft_dominant_power_ratio", "fft_spectral_entropy", "fft_spectral_flatness",
    "fft_spectral_centroid", "dt_mean_ms", "dt_std_ms", "dt_max_ms", "dt_min_ms", "dt_cv"
]


def load_model():
    """Load the binary classification model"""
    global binary_model
    if binary_model is not None:
        return binary_model
    
    model_path = os.path.join(MODEL_DIR, 'binary/models/binary_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            binary_model = pickle.load(f)
        print(f"Loaded binary model from {model_path}")
    return binary_model


def load_multiclass_model():
    """Load the multiclass classification model"""
    global multiclass_model
    if multiclass_model is not None:
        return multiclass_model
    
    model_path = os.path.join(MODEL_DIR, 'multiclass/models/multiclass_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            multiclass_model = pickle.load(f)
        print(f"Loaded multiclass model from {model_path}")
    return multiclass_model


def extract_features(values, timestamps_ms):
    """Extract features from a window of sensor values"""
    values = np.array(values, dtype=float)
    timestamps_ms = np.array(timestamps_ms, dtype=float)
    
    if len(values) < 10:
        return None
    
    features = {}
    
    # Basic statistics
    features['value_count'] = len(values)
    features['value_mean'] = np.mean(values)
    features['value_std'] = np.std(values) if len(values) > 1 else 0
    features['value_min'] = np.min(values)
    features['value_max'] = np.max(values)
    features['value_median'] = np.median(values)
    
    # Quantiles
    features['value_q01'] = np.percentile(values, 1)
    features['value_q05'] = np.percentile(values, 5)
    features['value_q10'] = np.percentile(values, 10)
    features['value_q25'] = np.percentile(values, 25)
    features['value_q75'] = np.percentile(values, 75)
    features['value_q90'] = np.percentile(values, 90)
    features['value_q95'] = np.percentile(values, 95)
    features['value_q99'] = np.percentile(values, 99)
    
    # Range and IQR
    features['value_range'] = features['value_max'] - features['value_min']
    features['value_iqr'] = features['value_q75'] - features['value_q25']
    features['value_cv'] = features['value_std'] / abs(features['value_mean']) if features['value_mean'] != 0 else 0
    features['value_rms'] = np.sqrt(np.mean(values ** 2))
    
    # Outlier ratios
    if features['value_std'] > 0:
        z_scores = np.abs((values - features['value_mean']) / features['value_std'])
        features['outlier_ratio_3std'] = np.mean(z_scores > 3)
        features['outlier_ratio_2std'] = np.mean(z_scores > 2)
    else:
        features['outlier_ratio_3std'] = 0
        features['outlier_ratio_2std'] = 0
    
    # Value ratios
    features['zero_ratio'] = np.mean(values == 0)
    features['negative_ratio'] = np.mean(values < 0)
    features['positive_ratio'] = np.mean(values > 0)
    
    # Shape statistics
    features['skewness'] = stats.skew(values) if len(values) > 2 else 0
    features['kurtosis'] = stats.kurtosis(values) if len(values) > 3 else 0
    
    # Zero crossing
    sign_changes = np.sum(np.diff(np.sign(values)) != 0)
    features['zero_crossing_rate'] = sign_changes / len(values) if len(values) > 1 else 0
    
    # Diff features
    diff = np.diff(values)
    if len(diff) > 0:
        features['diff_mean'] = np.mean(diff)
        features['diff_std'] = np.std(diff)
        features['diff_abs_mean'] = np.mean(np.abs(diff))
        features['diff_max'] = np.max(diff)
        features['diff_min'] = np.min(diff)
        features['diff_abs_max'] = np.max(np.abs(diff))
        
        # Direction changes
        direction_changes = np.sum(np.diff(np.sign(diff)) != 0)
        features['direction_changes'] = direction_changes
        features['direction_change_rate'] = direction_changes / len(diff) if len(diff) > 1 else 0
    else:
        features['diff_mean'] = features['diff_std'] = features['diff_abs_mean'] = 0
        features['diff_max'] = features['diff_min'] = features['diff_abs_max'] = 0
        features['direction_changes'] = features['direction_change_rate'] = 0
    
    # Second order diff
    diff2 = np.diff(values, n=2)
    if len(diff2) > 0:
        features['diff2_mean'] = np.mean(diff2)
        features['diff2_std'] = np.std(diff2)
        features['diff2_abs_mean'] = np.mean(np.abs(diff2))
    else:
        features['diff2_mean'] = features['diff2_std'] = features['diff2_abs_mean'] = 0
    
    # First and last values
    features['value_first'] = values[0]
    features['value_last'] = values[-1]
    features['value_change'] = values[-1] - values[0]
    features['value_change_ratio'] = features['value_change'] / abs(values[0]) if values[0] != 0 else 0
    
    # Segment statistics
    n = len(values)
    seg1 = values[:n//3]
    seg2 = values[n//3:2*n//3]
    seg3 = values[2*n//3:]
    
    features['seg1_mean'] = np.mean(seg1) if len(seg1) > 0 else 0
    features['seg2_mean'] = np.mean(seg2) if len(seg2) > 0 else 0
    features['seg3_mean'] = np.mean(seg3) if len(seg3) > 0 else 0
    features['seg1_std'] = np.std(seg1) if len(seg1) > 1 else 0
    features['seg2_std'] = np.std(seg2) if len(seg2) > 1 else 0
    features['seg3_std'] = np.std(seg3) if len(seg3) > 1 else 0
    
    features['seg_mean_diff_12'] = features['seg2_mean'] - features['seg1_mean']
    features['seg_mean_diff_23'] = features['seg3_mean'] - features['seg2_mean']
    features['seg_mean_diff_13'] = features['seg3_mean'] - features['seg1_mean']
    
    # Trend
    x = np.arange(len(values))
    if len(values) > 1:
        slope, _, _, _, _ = stats.linregress(x, values)
        features['trend_slope'] = slope
    else:
        features['trend_slope'] = 0
    
    # Local extrema
    local_max = np.sum((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])) if len(values) > 2 else 0
    local_min = np.sum((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])) if len(values) > 2 else 0
    features['local_max_count'] = local_max
    features['local_min_count'] = local_min
    features['peak_count'] = local_max + local_min
    features['peak_rate'] = features['peak_count'] / len(values) if len(values) > 0 else 0
    
    # Rolling variance
    window_size = min(10, len(values) // 3)
    if window_size > 1:
        rolling_var = pd.Series(values).rolling(window=window_size).var().dropna()
        if len(rolling_var) > 0:
            features['rolling_var_mean'] = np.mean(rolling_var)
            features['rolling_var_std'] = np.std(rolling_var)
            features['rolling_var_max'] = np.max(rolling_var)
        else:
            features['rolling_var_mean'] = features['rolling_var_std'] = features['rolling_var_max'] = 0
    else:
        features['rolling_var_mean'] = features['rolling_var_std'] = features['rolling_var_max'] = 0
    
    # Autocorrelation
    if len(values) > 5:
        autocorr = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
        features['autocorr_lag1'] = autocorr[1] if len(autocorr) > 1 else 0
        features['autocorr_lag5'] = autocorr[5] if len(autocorr) > 5 else 0
    else:
        features['autocorr_lag1'] = features['autocorr_lag5'] = 0
    
    # Entropy
    hist, _ = np.histogram(values, bins=10)
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    features['entropy'] = -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0
    
    # FFT features
    if len(values) >= 8:
        fft_vals = np.abs(fft(values))
        n_fft = len(fft_vals) // 2
        fft_vals = fft_vals[:n_fft]
        total_power = np.sum(fft_vals ** 2)
        
        if total_power > 0:
            features['fft_dc_ratio'] = (fft_vals[0] ** 2) / total_power
            low_end = max(1, n_fft // 4)
            mid_end = n_fft // 2
            features['fft_low_freq_ratio'] = np.sum(fft_vals[1:low_end] ** 2) / total_power
            features['fft_mid_freq_ratio'] = np.sum(fft_vals[low_end:mid_end] ** 2) / total_power
            features['fft_high_freq_ratio'] = np.sum(fft_vals[mid_end:] ** 2) / total_power
            features['fft_dominant_freq_idx'] = np.argmax(fft_vals[1:]) + 1
            features['fft_dominant_power_ratio'] = (np.max(fft_vals[1:]) ** 2) / total_power
            
            # Spectral features
            power_spectrum = fft_vals ** 2 / total_power
            power_spectrum = power_spectrum[power_spectrum > 0]
            features['fft_spectral_entropy'] = -np.sum(power_spectrum * np.log(power_spectrum))
            features['fft_spectral_flatness'] = np.exp(np.mean(np.log(fft_vals + 1e-10))) / (np.mean(fft_vals) + 1e-10)
            freq_bins = np.arange(n_fft)
            features['fft_spectral_centroid'] = np.sum(freq_bins * fft_vals) / (np.sum(fft_vals) + 1e-10)
        else:
            for k in ['fft_dc_ratio', 'fft_low_freq_ratio', 'fft_mid_freq_ratio', 'fft_high_freq_ratio',
                      'fft_dominant_freq_idx', 'fft_dominant_power_ratio', 'fft_spectral_entropy',
                      'fft_spectral_flatness', 'fft_spectral_centroid']:
                features[k] = 0
    else:
        for k in ['fft_dc_ratio', 'fft_low_freq_ratio', 'fft_mid_freq_ratio', 'fft_high_freq_ratio',
                  'fft_dominant_freq_idx', 'fft_dominant_power_ratio', 'fft_spectral_entropy',
                  'fft_spectral_flatness', 'fft_spectral_centroid']:
            features[k] = 0
    
    # Time interval features
    dt = np.diff(timestamps_ms)
    if len(dt) > 0:
        features['dt_mean_ms'] = np.mean(dt)
        features['dt_std_ms'] = np.std(dt)
        features['dt_max_ms'] = np.max(dt)
        features['dt_min_ms'] = np.min(dt)
        features['dt_cv'] = features['dt_std_ms'] / features['dt_mean_ms'] if features['dt_mean_ms'] > 0 else 0
    else:
        features['dt_mean_ms'] = features['dt_std_ms'] = features['dt_max_ms'] = features['dt_min_ms'] = features['dt_cv'] = 0
    
    return features


def load_sensor_data():
    """Load all 20 sensor files and filter October data"""
    global sensor_data_cache
    
    if sensor_data_cache is not None:
        return sensor_data_cache
    
    all_data = {}
    
    for file_name in DATA_FILES:
        file_path = os.path.join(DATA_DIR, f'{file_name}.parquet')
        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                df['ts'] = pd.to_datetime(df['timestamp_jst'], format='ISO8601')
                # Filter October data
                oct_df = df[df['ts'].dt.month == 10].copy()
                
                if len(oct_df) > 0:
                    # Sort by time
                    oct_df = oct_df.sort_values('ts')
                    # Extract sensor name from file name
                    sensor_name = file_name.replace('_2024', '')
                    all_data[sensor_name] = {
                        'timestamps': oct_df['timestamp_jst'].tolist(),
                        'epoch_ms': oct_df['epoch_ms'].tolist(),
                        'values': oct_df['value'].tolist(),
                        'count': len(oct_df)
                    }
                    print(f"Loaded {sensor_name}: {len(oct_df)} October records")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    
    sensor_data_cache = all_data
    return all_data


def load_abort_events():
    """Load abort events from aborts.json"""
    global abort_events_cache
    
    if abort_events_cache is not None:
        return abort_events_cache
    
    abort_file = 'aborts.json'
    if not os.path.exists(abort_file):
        return []
    
    with open(abort_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter October events
    oct_events = []
    for d in data:
        ts = d.get('timestamp', '')
        if ts.startswith('2024-10'):
            trigger = d.get('critical_trigger', '')
            # Simplify trigger type
            if 'Belle2 CLAWS' in trigger:
                event_type = 'Belle2 CLAWS'
            elif 'Manual Abort' in trigger:
                event_type = 'Manual Abort'
            elif 'Loss Monitor' in trigger:
                event_type = 'Loss Monitor'
            elif 'High THR VXD' in trigger:
                event_type = 'High THR VXD diamond'
            elif 'Low THR VXD' in trigger:
                event_type = 'Low THR VXD diamond'
            elif 'CLAWS D05' in trigger:
                event_type = 'CLAWS D05V1'
            elif 'RF' in trigger:
                event_type = 'RF'
            else:
                event_type = 'Other'
            
            oct_events.append({
                'timestamp': ts[:19],
                'type': event_type
            })
    
    abort_events_cache = oct_events
    print(f"Loaded {len(oct_events)} October abort events")
    return oct_events


@app.route('/')
def index():
    """Serve the visualization page"""
    return send_from_directory('.', 'visualization_demo.html')


@app.route('/api/sensor-data')
def get_sensor_data():
    """API endpoint to get all sensor data"""
    data = load_sensor_data()
    
    # Calculate statistics
    stats = {
        'total_sensors': len(data),
        'sensors': []
    }
    
    for name, sensor in data.items():
        stats['sensors'].append({
            'name': name,
            'count': sensor['count']
        })
    
    return jsonify({
        'status': 'success',
        'stats': stats,
        'data': data
    })


@app.route('/api/abort-events')
def get_abort_events():
    """API endpoint to get abort events"""
    events = load_abort_events()
    return jsonify({
        'status': 'success',
        'count': len(events),
        'events': events
    })


@app.route('/api/combined-data')
def get_combined_data():
    """API endpoint to get combined sensor and abort data for visualization"""
    sensor_data = load_sensor_data()
    abort_events = load_abort_events()
    
    # Get time range from sensor data
    all_timestamps = []
    for sensor in sensor_data.values():
        all_timestamps.extend(sensor['timestamps'])
    
    if all_timestamps:
        all_timestamps.sort()
        time_range = {
            'start': all_timestamps[0],
            'end': all_timestamps[-1]
        }
    else:
        time_range = {'start': None, 'end': None}
    
    # Prepare merged timeline data
    # Sample data points for visualization (every 5 minutes)
    timeline_data = []
    
    # Get all unique timestamps from all sensors
    all_times = set()
    for sensor in sensor_data.values():
        for ts in sensor['timestamps']:
            all_times.add(ts)
    
    sorted_times = sorted(list(all_times))
    
    # Sample at regular intervals (approximately every 5 minutes = 300 seconds)
    if len(sorted_times) > 0:
        sample_interval = max(1, len(sorted_times) // 3000)  # ~3000 data points
        sampled_times = sorted_times[::sample_interval]
        
        for ts in sampled_times:
            point = {'timestamp': ts, 'values': {}}
            for name, sensor in sensor_data.items():
                # Find closest value
                idx = None
                for i, t in enumerate(sensor['timestamps']):
                    if t >= ts:
                        idx = i
                        break
                if idx is not None:
                    point['values'][name] = sensor['values'][idx]
            timeline_data.append(point)
    
    return jsonify({
        'status': 'success',
        'time_range': time_range,
        'sensor_count': len(sensor_data),
        'event_count': len(abort_events),
        'abort_events': abort_events,
        'timeline_sample_count': len(timeline_data),
        'timeline_data': timeline_data[:500]  # Limit to 500 points for initial load
    })


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('.', filename)


def generate_predictions():
    """Pre-generate predictions for all time windows - optimized version"""
    global prediction_cache, binary_model, multiclass_model
    
    if prediction_cache is not None:
        return prediction_cache
    
    # Load both models
    binary_model = load_model()
    multiclass_model = load_multiclass_model()
    
    if binary_model is None:
        print("Warning: Binary model not loaded, cannot generate predictions")
        return None
    
    if multiclass_model is None:
        print("Warning: Multiclass model not loaded")
    
    sensor_data = load_sensor_data()
    abort_events = load_abort_events()
    
    print("Generating predictions...")
    
    # Pre-convert all sensor timestamps to datetime (optimization)
    sensor_df_cache = {}
    for name, sensor in sensor_data.items():
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(sensor['timestamps'], format='ISO8601'),
            'value': sensor['values'],
            'epoch_ms': sensor['epoch_ms']
        })
        df = df.sort_values('timestamp')
        sensor_df_cache[name] = df
    
    # Get time range
    all_times = []
    for df in sensor_df_cache.values():
        all_times.extend(df['timestamp'].tolist())
    
    if not all_times:
        return None
    
    start_time = min(all_times)
    end_time = max(all_times)
    
    # Generate prediction points: 120s window, 30s step
    window_size = pd.Timedelta(seconds=120)
    step_size = pd.Timedelta(seconds=30)
    
    predictions = []
    current_time = start_time
    step_count = 0
    skipped_count = 0
    
    while current_time <= end_time:
        window_start = current_time - window_size
        window_end = current_time
        
        # Collect features from all sensors for this window
        all_features = []
        
        for sensor_name, df in sensor_df_cache.items():
            # Filter data in time window
            mask = (df['timestamp'] >= window_start) & (df['timestamp'] <= window_end)
            window_df = df[mask]
            
            if len(window_df) >= 10:
                features = extract_features(
                    window_df['value'].tolist(),
                    window_df['epoch_ms'].tolist()
                )
                if features is not None:
                    all_features.append(features)
        
        # Only generate prediction when sufficient data exists
        if len(all_features) > 0:
            # Create feature vector
            feature_vector = []
            for fname in FEATURE_NAMES:
                vals = [f.get(fname, 0) for f in all_features]
                feature_vector.append(np.mean(vals))
            
            # Binary prediction (no fallback - skip on error)
            try:
                X = np.array([feature_vector])
                binary_prob = float(binary_model.predict_proba(X)[0][1])
            except Exception as e:
                current_time += step_size
                step_count += 1
                skipped_count += 1
                continue
            
            # Multiclass prediction (no fallback - skip on error)
            try:
                X = np.array([feature_vector])
                mc_probs = multiclass_model.predict_proba(X)[0]
                normal_prob = 1 - binary_prob
                class_probs = [float(normal_prob)] + [float(p * binary_prob) for p in mc_probs]
            except Exception as e:
                current_time += step_size
                step_count += 1
                skipped_count += 1
                continue
            
            # Determine warning status and predicted class (backend logic)
            if binary_prob >= CRITICAL_THRESHOLD:
                alert_level = 'danger'
            elif binary_prob >= WARNING_THRESHOLD:
                alert_level = 'warning'
            else:
                alert_level = 'normal'
            
            is_warning = binary_prob >= WARNING_THRESHOLD
            predicted_class_idx = int(np.argmax(class_probs))
            predicted_class = CLASS_LABELS[predicted_class_idx]
            
            predictions.append({
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'abnormal_prob': float(binary_prob),
                'normal_prob': float(1 - binary_prob),
                'class_probs': class_probs,
                'is_warning': is_warning,
                'alert_level': alert_level,
                'predicted_class': predicted_class,
                'predicted_class_idx': predicted_class_idx
            })
        else:
            skipped_count += 1
        
        current_time += step_size
        step_count += 1
        
        if step_count % 1000 == 0:
            print(f"  Processed {step_count} steps, generated {len(predictions)} predictions...")
    
    print(f"Skipped {skipped_count} steps due to insufficient data")
    
    prediction_cache = {
        'predictions': predictions,
        'count': len(predictions),
        'time_range': {
            'start': predictions[0]['timestamp'] if predictions else None,
            'end': predictions[-1]['timestamp'] if predictions else None
        }
    }
    
    print(f"Generated {len(predictions)} predictions")
    return prediction_cache


@app.route('/api/predictions')
def get_predictions():
    """API endpoint to get pre-computed predictions"""
    predictions = generate_predictions()
    
    if predictions is None:
        return jsonify({
            'status': 'error',
            'message': 'Predictions not available'
        }), 500
    
    return jsonify({
        'status': 'success',
        'class_labels': CLASS_LABELS,
        **predictions
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Beam Abort Prediction Visualization System")
    print("=" * 60)
    
    # Pre-load data
    print("\nLoading sensor data...")
    sensor_data = load_sensor_data()
    print(f"Loaded {len(sensor_data)} sensors")
    
    print("\nLoading abort events...")
    abort_events = load_abort_events()
    print(f"Loaded {len(abort_events)} events")
    
    print("\nLoading LightGBM models...")
    bin_model = load_model()
    mc_model = load_multiclass_model()
    if bin_model:
        print("Binary model loaded successfully")
    else:
        print("Warning: Binary model not found")
    if mc_model:
        print("Multiclass model loaded successfully")
    else:
        print("Warning: Multiclass model not found")
    
    print("\nPre-generating predictions (this may take a moment)...")
    predictions = generate_predictions()
    if predictions:
        print(f"Generated {predictions['count']} prediction points")
    
    print("\n" + "=" * 60)
    print("Starting server at http://127.0.0.1:5000")
    print("=" * 60)
    
    app.run(debug=True, port=5000)
