#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
信号成功案例扫描器 (Signal Success Scanner)
================================================================================

功能描述:
    本脚本用于自动化扫描股票历史数据，验证特定技术买入信号的有效性，
    并筛选出成功案例供后续模式分析使用。

主要功能:
    1. 扫描 /data/day/ 目录下所有股票CSV文件
    2. 识别三类买入信号：
       - 六脉神剑信号（4红以上）
       - 缠论5类买点（一买、二买、三买、强二买、类二买）
       - 买卖点信号（买点1、买点2）
    3. 计算信号后15个交易日的涨幅
    4. 筛选涨幅>5%的成功案例
    5. 按信号类型和六脉神剑组合进行统计

输出文件:
    - report/all_signal_records.csv           : 所有信号记录（包含成功和失败）
    - report/signal_success_cases.csv         : 仅成功案例
    - report/signal_summary.json              : 统计摘要（报告目录备份）
    - web/client/src/data/signal_summary.json : 统计摘要（供Web前端使用）

使用方法:
    cd ~/tdx-strategy-backtest/py_file
    python3 a4_signal_success_scanner.py

依赖模块:
    - pandas, numpy: 数据处理
    - indicators: 技术指标计算（项目内部模块）
    - multiprocessing: 多进程并行处理

作者: TradeGuide System
版本: 2.0.0
更新日期: 2026-01-15
================================================================================
"""

try:
    from a99_logger import log
except ImportError:
    def log(msg, level="INFO"): print(f"[{level}] {msg}")

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings

# 忽略警告信息，保持输出整洁
warnings.filterwarnings('ignore')

# ==============================================================================
# 路径配置
# ==============================================================================

# 获取项目根目录（py_file的上级目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加项目路径到系统路径，以便导入项目内部模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 数据目录：存放股票历史数据的CSV文件
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')

# 报告输出目录：存放分析报告和中间结果
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')

# Web数据目录：存放供前端展示的JSON数据文件
# 注意：脚本会自动将数据输出到此目录，无需手动复制
WEB_DATA_DIR = os.path.join(PROJECT_ROOT, 'web', 'client', 'src', 'data')

# ==============================================================================
# 回测参数配置
# ==============================================================================

# 持仓天数：信号触发后持有的交易日数量
# 说明：15天是一个较为常用的短期持仓周期，可根据需要调整
HOLDING_DAYS = 15

# 成功阈值：涨幅超过此百分比视为成功
# 说明：5%是一个相对保守的阈值，可根据风险偏好调整
SUCCESS_THRESHOLD = 5.0

# 六脉神剑最小红色数量：只统计达到此数量的信号
# 说明：4红以上被认为是较强的多头信号
MIN_RED_COUNT = 4

# ==============================================================================
# 导入技术指标计算模块
# ==============================================================================

from a99_indicators import (
    calculate_six_veins,       # 六脉神剑指标计算
    calculate_buy_sell_points, # 买卖点指标计算
    calculate_chan_theory      # 缠论买点指标计算
)

# 六脉神剑指标列和名称
red_cols = ['macd_red', 'kdj_red', 'rsi_red', 'lwr_red', 'bbi_red', 'mtm_red']
indicator_names = ['MACD', 'KDJ', 'RSI', 'LWR', 'BBI', 'MTM']

# ==============================================================================
# 扫描与统计函数
# ==============================================================================

def scan_single_stock(file_path: str) -> pd.DataFrame:
    """
    扫描单只股票的信号，并返回信号记录DataFrame。
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8', parse_dates=['date'])
        # 基础列重命名
        df.rename(columns=lambda c: c.strip().lower(), inplace=True)
        if 'date' not in df.columns:
            log(f"文件缺少日期列: {file_path}", level="ERROR")
            return pd.DataFrame()
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        # 数据不足以判断15日涨幅的无需扫描
        if len(df) < HOLDING_DAYS + 1:
            return pd.DataFrame()

        # 计算所需指标列
        df = calculate_six_veins(df)
        df = calculate_buy_sell_points(df)
        df = calculate_chan_theory(df)

        # 计算未来HOLDING_DAYS日的涨幅百分比
        df['future_return'] = (df['close'].shift(-HOLDING_DAYS) - df['close']) / df['close'] * 100

        records = []
        for i in range(len(df) - HOLDING_DAYS):
            # 逐日检查信号
            has_six_veins_signal = df.at[i, 'six_veins_count'] >= MIN_RED_COUNT
            has_chan_signal = df.at[i, 'chan_any_buy']
            has_buy1 = bool(df.at[i, 'buy1'])
            has_buy2 = bool(df.at[i, 'buy2'])
            if has_six_veins_signal or has_chan_signal or has_buy1 or has_buy2:
                record = {
                    'date': df.at[i, 'date'],
                    'stock': os.path.basename(file_path).replace('.csv', ''),
                    'six_veins_count': int(df.at[i, 'six_veins_count']),
                    'buy1': int(has_buy1),
                    'buy2': int(has_buy2),
                    'chan_any_buy': int(bool(df.at[i, 'chan_any_buy'])),
                    'future_return': float(df.at[i, 'future_return'])
                }
                # 信号类型分类
                signal_types = []
                if has_six_veins_signal: signal_types.append('six_veins')
                if has_chan_signal: signal_types.append('chan_buy')
                if has_buy1 or has_buy2: signal_types.append('buy_sell')
                record['signal_type'] = '+'.join(signal_types) if signal_types else 'none'
                # 成功标记
                record['is_success'] = int(record['future_return'] > SUCCESS_THRESHOLD)
                records.append(record)
        if not records:
            return pd.DataFrame()
        df_records = pd.DataFrame(records)
        # 生成六脉神剑红色组合列 (性能优化)
        if not df_records.empty:
            combo_list = []
            # 组合名称：六脉神剑红色指标列表
            for vals in df_records[red_cols].to_numpy(dtype=bool):
                names = [name for val, name in zip(vals, indicator_names) if val]
                combo_list.append('+'.join(names) if names else '')
            df_records['six_veins_combo'] = combo_list
        else:
            df_records['six_veins_combo'] = []
        return df_records
    except Exception as e:
        log(f"扫描股票失败: {file_path}, 错误: {e}", level="ERROR")
        return pd.DataFrame()

def aggregate_results(all_records: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    汇总所有股票的信号记录，拆分成功/失败案例，并计算统计摘要。
    """
    if all_records.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    # 所有信号记录按日期排序保存
    all_records.sort_values('date', inplace=True)
    all_records.reset_index(drop=True, inplace=True)
    # 成功案例筛选
    success_cases = all_records[all_records['is_success'] == 1].copy()
    success_cases.reset_index(drop=True, inplace=True)
    # 统计摘要
    summary = {
        'total_signals': len(all_records),
        'success_signals': int(all_records['is_success'].sum()),
        'success_rate': round(all_records['is_success'].mean() * 100, 2),
        'signal_type_counts': all_records['signal_type'].value_counts().to_dict(),
        'six_veins_success_by_count': success_cases['six_veins_count'].value_counts().to_dict()
    }
    return all_records, success_cases, summary

def save_results(all_records: pd.DataFrame, success_cases: pd.DataFrame, summary: Dict):
    """
    保存扫描结果至文件。
    """
    # 保存所有信号记录
    all_records_path = os.path.join(REPORT_DIR, 'all_signal_records.csv')
    all_records.to_csv(all_records_path, index=False, encoding='utf-8-sig')
    log(f"所有信号记录已保存: {all_records_path}")
    # 保存成功案例
    success_path = os.path.join(REPORT_DIR, 'signal_success_cases.csv')
    success_cases.to_csv(success_path, index=False, encoding='utf-8-sig')
    log(f"成功案例列表已保存: {success_path}")
    # 保存统计摘要（同时写入前端数据目录）
    summary_path = os.path.join(REPORT_DIR, 'signal_summary.json')
    web_summary_path = os.path.join(WEB_DATA_DIR, 'signal_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(web_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"统计摘要已保存: {summary_path}")
    log(f"前端统计摘要已更新: {web_summary_path}")

# ==============================================================================
# 主流程
# ==============================================================================

def main():
    stock_files = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.csv'):
                stock_files.append(os.path.join(root, file))
    if not stock_files:
        log("未找到任何股票CSV文件，请检查 data/day 目录。", level="ERROR")
        return

    log(f"开始扫描 {len(stock_files)} 只股票的买入信号...")
    results = []
    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() - 1)) as executor:
        futures = {executor.submit(scan_single_stock, f): f for f in stock_files}
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                df_res = future.result()
            except Exception as e:
                log(f"股票扫描出错: {file_path}, 错误: {e}", level="ERROR")
            else:
                if not df_res.empty:
                    results.append(df_res)
    if not results:
        log("未检测到任何买入信号。", level="WARNING")
        return

    all_records_df = pd.concat(results, ignore_index=True)
    all_records_df, success_cases_df, summary = aggregate_results(all_records_df)
    save_results(all_records_df, success_cases_df, summary)
    log("扫描完成！成功案例数: %d, 成功率: %.2f%%" % (summary.get('success_signals', 0), summary.get('success_rate', 0.0)))

if __name__ == "__main__":
    main()

print("a4_signal_success_scanner.py 脚本执行完毕")
