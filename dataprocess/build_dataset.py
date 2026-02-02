#!/usr/bin/env python3
"""
数据集构建工具 - 最终版
特性：分批处理、极低内存、断点续传、多进程并行

处理流程：
1. 将时间戳分批（每批约1000个）
2. 每批独立处理：遍历TSV文件提取数据
3. 处理完一批立即生成Parquet并释放内存
"""

import os
import json
import pickle
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import pytz
import pandas as pd
import logging
import gc

# ==================== 配置 ====================
CONFIG = {
    "tsv_directory": "/gpfs/group/belle2/group/accelerator/pv/2024/AA_tsv/",
    "abnormal_json": "./aborts_classified.json",
    "normal_json": "./normal_timestamps.json",
    "output_dir": "./dataset",
    "checkpoint_file": "./dataset/.checkpoint.pkl",
    "max_workers": 32,
    "time_before_min": 35,
    "time_after_min": 5,
    "timezone": "Asia/Tokyo",
    "batch_size": 1000,  # 每批时间戳数量（可根据内存调整为500-1000）
    "tsv_cache_file": "./.tsv_list.pkl",
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ==================== 工具函数 ====================

def process_tsv_for_batch(args: Tuple) -> Tuple[str, Dict]:
    """处理单个TSV文件，提取批次内所有时间戳的数据"""
    tsv_path, ranges_info = args
    # ranges_info: [(ts_ms, start_ms, end_ms), ...]
    
    try:
        with open(tsv_path, 'r') as f:
            pv_name = None
            t0 = None
            nval = None
            
            for _ in range(4):
                line = f.readline()
                if '# PV name' in line:
                    pv_name = line.split()[-1]
                elif '# t0' in line:
                    t0 = int(line.split()[2])
                elif '# Nval' in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        nval = int(parts[2])
            
            if nval != 1 or not pv_name or t0 is None:
                return None, {}
            
            # 转换为相对时间戳并过滤，按结束时间排序
            rel_ranges = []
            for ts_ms, start_ms, end_ms in ranges_info:
                rel_s = start_ms - t0
                rel_e = end_ms - t0
                if rel_e >= 0:  # 只保留可能有数据的范围
                    rel_ranges.append((ts_ms, rel_s, rel_e))
            
            if not rel_ranges:
                return pv_name, {}
            
            # 按起始时间排序
            rel_ranges.sort(key=lambda x: x[1])
            result = defaultdict(list)
            
            # 计算全局最小起始和最大结束时间，用于快速跳过
            global_start = min(r[1] for r in rel_ranges)
            global_end = max(r[2] for r in rel_ranges)
            
            # 流式读取
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                try:
                    rel_ts = int(parts[0])
                except:
                    continue
                
                # 快速跳过全局范围之外的数据
                if rel_ts < global_start:
                    continue
                if rel_ts > global_end:
                    break
                
                try:
                    val = float(parts[1])
                except:
                    continue
                
                # 检查属于哪些时间范围（一个数据点可能属于多个范围）
                for ts_ms, rs, re in rel_ranges:
                    if rel_ts < rs:
                        break  # 后续范围起始更大，无需继续
                    if rel_ts <= re:
                        result[ts_ms].append((t0 + rel_ts, val))
            
            return pv_name, dict(result)
    except:
        return None, {}

def load_records(json_path: str, tz, label: str = None) -> List[Dict]:
    """加载JSON记录"""
    if not os.path.exists(json_path):
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    out = []
    for item in data:
        ts = item.get('timestamp', '')
        trigger = label or item.get('critical_trigger', 'Unknown')
        try:
            dt = datetime.strptime(ts[:19], '%Y-%m-%d %H:%M:%S')
            dt = tz.localize(dt)
            out.append({'ts_ms': int(dt.timestamp() * 1000), 'ts_str': ts, 'trigger': trigger})
        except:
            continue
    return out

def scan_tsv(directory: str, cache_file: str = None) -> List[str]:
    """
    扫描TSV文件，只返回单维度(Nval=1)的文件
    支持缓存：首次扫描后保存列表，后续直接加载
    """
    # 尝试加载缓存
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            log.info(f"从TSV缓存加载: {len(cached)} 个文件")
            return cached
        except:
            log.info("缓存加载失败，重新扫描")
    
    # 扫描文件
    log.info("扫描TSV文件中...")
    files = glob.glob(os.path.join(directory, "*.tsv"))
    log.info(f"总文件数: {len(files)}")
    
    valid = []
    for i, fpath in enumerate(files):
        try:
            with open(fpath, 'r') as fp:
                for _ in range(4):
                    line = fp.readline()
                    if line.startswith('# Nval'):
                        parts = line.split()
                        if len(parts) >= 3 and parts[2] == '1':
                            valid.append(fpath)
                        break
        except:
            continue
        
        # 进度提示
        if (i + 1) % 5000 == 0:
            log.info(f"扫描进度: {i+1}/{len(files)}, 有效: {len(valid)}")
    
    # 保存缓存
    if cache_file and valid:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(valid, f)
            log.info(f"TSV列表已缓存: {cache_file}")
        except Exception as e:
            log.info(f"缓存保存失败: {e}")
    
    return valid

def save_parquet(data: Dict, path: str) -> bool:
    """保存Parquet"""
    ts_set = set()
    for pts in data.values():
        ts_set.update(t for t, _ in pts)
    
    if not ts_set:
        return False
    
    ts_list = sorted(ts_set)
    pvs = sorted(data.keys())
    
    df_data = {'timestamp': ts_list}
    for pv in pvs:
        m = {t: v for t, v in data.get(pv, [])}
        df_data[pv] = [m.get(t, 0.0) for t in ts_list]
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(df_data).to_parquet(path, index=False, compression='snappy')
    return True

def load_checkpoint(path: str) -> Set[str]:
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return set()

def save_checkpoint(path: str, done: Set[str]):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(done, f)
    except:
        pass

# ==================== 主流程 ====================

def main():
    cfg = CONFIG
    tz = pytz.timezone(cfg['timezone'])
    
    log.info("=" * 50)
    log.info("数据集构建工具 - 分批处理版")
    log.info(f"时间窗口: -{cfg['time_before_min']}min ~ -{cfg['time_after_min']}min")
    log.info(f"并行进程: {cfg['max_workers']}, 批大小: {cfg['batch_size']}")
    log.info("=" * 50)
    
    # 1. 加载记录
    log.info("[1] 加载时间戳...")
    abnormal = load_records(cfg['abnormal_json'], tz)
    normal = load_records(cfg['normal_json'], tz, 'Normal')
    all_rec = abnormal + normal
    log.info(f"异常: {len(abnormal)}, 正常: {len(normal)}")
    
    # 2. 检查点
    done = load_checkpoint(cfg['checkpoint_file'])
    pending = [r for r in all_rec if r['ts_str'] not in done]
    log.info(f"已完成: {len(done)}, 待处理: {len(pending)}")
    
    if not pending:
        log.info("全部完成!")
        return
    
    # 3. 扫描TSV（支持缓存）
    log.info("[2] 扫描TSV文件...")
    tsv_files = scan_tsv(cfg['tsv_directory'], cfg.get('tsv_cache_file'))
    log.info(f"有效TSV: {len(tsv_files)}")
    
    if not tsv_files:
        log.error("无TSV文件")
        return
    
    # 4. 分批处理
    time_before_ms = cfg['time_before_min'] * 60 * 1000
    time_after_ms = cfg['time_after_min'] * 60 * 1000
    batch_size = cfg['batch_size']
    total_batches = (len(pending) + batch_size - 1) // batch_size
    
    log.info(f"[3] 分批处理: {total_batches} 批")
    
    total_success = 0
    
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(pending))
        batch = pending[start:end]
        
        log.info(f"\n--- 批次 {batch_idx+1}/{total_batches} ({len(batch)}条) ---")
        
        # 构建时间范围
        ranges = []
        rec_map = {}
        for r in batch:
            ts_ms = r['ts_ms']
            ranges.append((ts_ms, ts_ms - time_before_ms, ts_ms - time_after_ms))
            rec_map[ts_ms] = r
        
        # 并行处理TSV
        agg = defaultdict(dict)
        tasks = [(f, ranges) for f in tsv_files]
        
        with ProcessPoolExecutor(max_workers=cfg['max_workers']) as exe:
            futures = {exe.submit(process_tsv_for_batch, t): t[0] for t in tasks}
            
            for future in as_completed(futures):
                try:
                    pv, ts_data = future.result()
                    if pv and ts_data:
                        for ts_ms, pts in ts_data.items():
                            if pts:
                                agg[ts_ms][pv] = pts
                except:
                    pass
        
        # 生成Parquet
        batch_success = 0
        for ts_ms, pv_data in agg.items():
            if ts_ms not in rec_map:
                continue
            
            rec = rec_map[ts_ms]
            trigger = rec['trigger']
            ts_name = rec['ts_str'][:19].replace(':', '').replace('-', '').replace(' ', '_')
            
            if trigger == 'Normal':
                out_dir = os.path.join(cfg['output_dir'], 'Normal')
            else:
                safe = trigger.replace(':', '_').replace('/', '_').replace(' ', '_')
                out_dir = os.path.join(cfg['output_dir'], 'Abnormal', safe)
            
            out_path = os.path.join(out_dir, f"{ts_name}.parquet")
            
            try:
                if save_parquet(pv_data, out_path):
                    done.add(rec['ts_str'])
                    batch_success += 1
            except:
                pass
        
        total_success += batch_success
        log.info(f"批次完成: {batch_success}/{len(batch)}")
        
        # 保存检查点并清理内存
        save_checkpoint(cfg['checkpoint_file'], done)
        del agg, tasks, rec_map
        gc.collect()
    
    # 清理
    try:
        os.remove(cfg['checkpoint_file'])
    except:
        pass
    
    log.info("\n" + "=" * 50)
    log.info(f"全部完成! 成功: {total_success}")
    log.info("=" * 50)

if __name__ == "__main__":
    main()
