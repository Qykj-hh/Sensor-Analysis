#!/usr/bin/env python3
"""
TSV文件Nval维度类型分析
目的：分析各个Nval值对应的数据可能是什么类型

输出：
1. 各Nval的文件数量统计
2. 各Nval的PV名称特征分析
3. 推断各Nval对应的数据类型
"""

import os
import glob
from collections import defaultdict
import logging
import json
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ==================== 配置 ====================
CONFIG = {
    "tsv_directory": "/gpfs/group/belle2/group/accelerator/pv/2024/AA_tsv/",
    "output_dir": "./analysis_results",
    "max_files_to_scan": None,  # None表示扫描全部
}

# 数据类型关键词匹配规则
TYPE_PATTERNS = {
    "波形数据": ["WAVE", "WF", "WAVEFORM"],
    "BPM位置": ["BPM", "POSITION", "POS"],
    "掩码/开关": ["MASK", "FLAG", "ENABLE", "SW"],
    "温度": ["TEMP", "TEMPERATURE"],
    "电流": ["CURR", "CURRENT", "I_"],
    "电压": ["VOLT", "VOLTAGE", "V_"],
    "功率": ["POWER", "PWR"],
    "真空": ["VA_", "VACUUM", "PRESSURE"],
    "磁场": ["MAG", "FIELD", "B_"],
    "射频": ["RF", "LLRF", "CAV"],
    "反馈系统": ["FB", "FEEDBACK", "DRIVE"],
    "注入系统": ["INJ", "INJECT"],
    "监控": ["MON", "MONITOR"],
    "采集": ["ACQ", "ACQUIRE", "SRAM", "BRAM"],
    "损失监控": ["LM:", "LOSS"],
    "填充模式": ["FILL", "BKT", "BUCKET"],
}

# ==================== 工具函数 ====================

def parse_tsv_header(path: str) -> dict:
    """解析TSV头部元数据"""
    try:
        with open(path, 'r') as f:
            meta = {'path': path, 'filename': os.path.basename(path)}
            for _ in range(4):
                line = f.readline()
                if '# PV name' in line:
                    meta['pv'] = line.split()[-1]
                elif '# Nval' in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        meta['nval'] = int(parts[2])
            return meta
    except:
        return None

def infer_data_type(pv_name: str) -> list:
    """根据PV名称推断数据类型"""
    pv_upper = pv_name.upper()
    types = []
    for dtype, keywords in TYPE_PATTERNS.items():
        for kw in keywords:
            if kw in pv_upper:
                types.append(dtype)
                break
    return types if types else ["未知类型"]

def extract_prefix(pv_name: str, level: int = 1) -> str:
    """提取PV名称前缀"""
    parts = pv_name.split(':')
    return ':'.join(parts[:level]) if len(parts) >= level else pv_name

# ==================== 主分析流程 ====================

def main():
    cfg = CONFIG
    os.makedirs(cfg['output_dir'], exist_ok=True)
    
    log.info("=" * 60)
    log.info("TSV文件Nval维度类型分析")
    log.info("=" * 60)
    
    # 1. 扫描所有TSV文件
    log.info("\n[1] 扫描TSV文件...")
    files = glob.glob(os.path.join(cfg['tsv_directory'], "*.tsv"))
    
    if cfg['max_files_to_scan']:
        files = files[:cfg['max_files_to_scan']]
    
    log.info(f"待扫描文件数: {len(files)}")
    
    # 按Nval分组存储
    by_nval = defaultdict(list)
    
    for i, path in enumerate(files):
        meta = parse_tsv_header(path)
        if meta and 'nval' in meta and 'pv' in meta:
            by_nval[meta['nval']].append(meta)
        
        if (i + 1) % 10000 == 0:
            log.info(f"扫描进度: {i+1}/{len(files)}")
    
    total_valid = sum(len(v) for v in by_nval.values())
    log.info(f"有效文件: {total_valid}")
    
    # 2. 按Nval分析
    log.info("\n" + "=" * 60)
    log.info("[2] 各Nval维度分析")
    log.info("=" * 60)
    
    results = {}
    
    for nval in sorted(by_nval.keys()):
        metas = by_nval[nval]
        count = len(metas)
        ratio = count / total_valid * 100
        
        log.info(f"\n{'='*50}")
        log.info(f"Nval={nval}: {count}个文件 ({ratio:.2f}%)")
        log.info(f"{'='*50}")
        
        # 统计前缀分布
        prefix_count = defaultdict(int)
        type_count = defaultdict(int)
        
        for meta in metas:
            pv = meta['pv']
            prefix = extract_prefix(pv, 1)
            prefix_count[prefix] += 1
            
            # 推断类型
            types = infer_data_type(pv)
            for t in types:
                type_count[t] += 1
        
        # 输出前缀TOP5
        log.info("\n前缀分布 (TOP5):")
        sorted_prefixes = sorted(prefix_count.items(), key=lambda x: -x[1])
        for prefix, cnt in sorted_prefixes[:5]:
            log.info(f"  {prefix}: {cnt} ({cnt/count*100:.1f}%)")
        
        # 输出推断类型
        log.info("\n推断数据类型:")
        sorted_types = sorted(type_count.items(), key=lambda x: -x[1])
        for dtype, cnt in sorted_types:
            log.info(f"  {dtype}: {cnt} ({cnt/count*100:.1f}%)")
        
        # 主要类型判断
        main_type = sorted_types[0][0] if sorted_types else "未知"
        main_type_ratio = sorted_types[0][1] / count if sorted_types else 0
        
        # 示例PV名称
        log.info("\n示例PV名称:")
        for meta in metas[:5]:
            log.info(f"  - {meta['pv']}")
        if len(metas) > 5:
            log.info(f"  ... 及其他 {len(metas)-5} 个")
        
        # 数据用途建议
        if nval == 1:
            suggestion = "标准单值时序数据，适合直接用于特征提取"
        elif nval <= 10:
            suggestion = "低维多值数据，可能是多通道传感器，可考虑展开或取均值"
        elif nval <= 100:
            suggestion = "中维数据，可能是频谱或位置数组，需评估是否有用"
        elif nval <= 1000:
            suggestion = "高维数据，可能是采集缓冲区或掩码，建议跳过"
        else:
            suggestion = "超高维波形数据，不适合作为特征，建议跳过"
        
        log.info(f"\n建议: {suggestion}")
        
        results[nval] = {
            'count': count,
            'ratio': ratio,
            'main_type': main_type,
            'main_type_ratio': main_type_ratio,
            'prefix_distribution': dict(sorted_prefixes[:10]),
            'type_distribution': dict(sorted_types),
            'examples': [m['pv'] for m in metas[:10]],
            'suggestion': suggestion,
        }
    
    # 3. 总结
    log.info("\n" + "=" * 60)
    log.info("[3] 总结")
    log.info("=" * 60)
    
    log.info("\n各Nval分类汇总:")
    log.info(f"{'Nval':<10} {'数量':<10} {'占比':<10} {'主要类型':<15} {'建议':<20}")
    log.info("-" * 65)
    
    for nval in sorted(results.keys()):
        r = results[nval]
        short_suggestion = "可用" if nval == 1 else ("评估" if nval <= 100 else "跳过")
        log.info(f"{nval:<10} {r['count']:<10} {r['ratio']:.2f}%     {r['main_type']:<15} {short_suggestion:<20}")
    
    # 单维度占比
    single_dim = results.get(1, {}).get('count', 0)
    log.info(f"\n单维度(Nval=1)占比: {single_dim}/{total_valid} ({single_dim/total_valid*100:.2f}%)")
    
    if single_dim / total_valid > 0.9:
        log.info("结论: 绝大多数为单维度数据，只使用Nval=1的策略合理")
    elif single_dim / total_valid > 0.7:
        log.info("结论: 单维度数据占主导，可考虑仅使用Nval=1")
    else:
        log.info("结论: 多维度数据占比较大，需仔细评估各类型价值")
    
    # 4. 保存结果
    result_path = os.path.join(cfg['output_dir'], 'nval_type_analysis.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    log.info(f"\n详细结果已保存: {result_path}")

if __name__ == "__main__":
    main()
