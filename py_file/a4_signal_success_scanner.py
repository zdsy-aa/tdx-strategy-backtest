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
    python3 signal_success_scanner.py

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
from typing import Dict, List, Tuple, Optional
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
    calculate_chan_theory      # 缠论指标计算
)


# ==============================================================================
# 数据加载函数
# ==============================================================================

def load_stock_data(filepath: str) -> pd.DataFrame:
    """
    加载单只股票的CSV数据
    
    功能说明:
        读取CSV文件，统一列名格式，并按日期排序。
        支持中文列名和英文列名两种格式。
    
    参数:
        filepath (str): CSV文件的完整路径
    
    返回:
        pd.DataFrame: 标准化后的股票数据，包含以下列：
            - name: 股票名称
            - date: 交易日期
            - open: 开盘价
            - close: 收盘价
            - high: 最高价
            - low: 最低价
            - volume: 成交量
            - amount: 成交额
            - amplitude: 振幅
            - pct_change: 涨跌幅
            - change: 涨跌额
            - turnover: 换手率
        如果文件无法读取，返回空DataFrame
    
    示例:
        >>> df = load_stock_data('/path/to/000001.csv')
        >>> log(df.columns.tolist())
        ['name', 'date', 'open', 'close', 'high', 'low', ...]
    """
    try:
        # 读取CSV文件，使用UTF-8-SIG编码以处理BOM
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        
        # 列名映射：将中文列名转换为英文列名
        column_mapping = {
            '名称': 'name',
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover'
        }
        df = df.rename(columns=column_mapping)
        
        # 日期处理：转换为datetime类型
        df['date'] = pd.to_datetime(df['date'])
        
        # 按日期升序排序，确保时间序列正确
        df = df.sort_values('date').reset_index(drop=True)
        
        # 过滤无效数据：收盘价必须大于0
        df = df[df['close'] > 0]
        
        return df
        
    except Exception as e:
        # 文件读取失败时返回空DataFrame，不中断整体处理流程
        return pd.DataFrame()


def get_all_stock_files() -> List[str]:
    """
    获取所有股票数据文件路径
    
    功能说明:
        遍历数据目录，收集所有CSV文件的完整路径。
        支持上海(sh)、深圳(sz)、北京(bj)三个市场。
    
    返回:
        List[str]: CSV文件路径列表
    
    目录结构示例:
        data/day/
        ├── sh/           # 上海市场
        │   ├── 600000.csv
        │   └── 600001.csv
        ├── sz/           # 深圳市场
        │   ├── 000001.csv
        │   └── 000002.csv
        └── bj/           # 北京市场
            ├── 430002.csv
            └── 430003.csv
    """
    stock_files = []
    
    # 遍历三个市场目录
    for market in ['sh', 'sz', 'bj']:
        market_dir = os.path.join(DATA_DIR, market)
        if os.path.exists(market_dir):
            for filename in os.listdir(market_dir):
                if filename.endswith('.csv'):
                    stock_files.append(os.path.join(market_dir, filename))
    
    return stock_files


# ==============================================================================
# 信号检测函数
# ==============================================================================

def detect_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    检测所有买入信号
    
    功能说明:
        对股票数据计算各类技术指标，并标记买入信号。
        包括六脉神剑、缠论买点和买卖点三类信号。
    
    参数:
        df (pd.DataFrame): 股票历史数据
    
    返回:
        pd.DataFrame: 添加了信号标记列的数据框，新增列包括：
            - six_veins_count: 六脉神剑红色指标数量（0-6）
            - six_veins_4plus: 是否达到4红以上
            - six_veins_combo: 红色指标组合名称（如 "MACD+KDJ+RSI+BBI"）
            - chan_buy1~chan_like_buy2: 缠论5类买点信号
            - buy1, buy2: 买卖点信号
    
    信号检测规则:
        1. 六脉神剑：统计MACD、KDJ、RSI、LWR、BBI、MTM六个指标中
           处于红色（多头）状态的数量，4个及以上触发信号
        2. 缠论买点：基于缠论理论的5类标准买点
        3. 买卖点：基于主力资金流向的买点信号
    """
    # 数据量检查：至少需要30条记录才能计算技术指标
    if len(df) < 30:
        return df
    
    try:
        # 计算六脉神剑指标
        # 返回的df会包含 macd_red, kdj_red 等布尔列，以及 six_veins_count 计数列
        df = calculate_six_veins(df)
        
        # 计算买卖点指标
        # 返回的df会包含 buy1, buy2 等布尔列
        df = calculate_buy_sell_points(df)
        
        # 计算缠论买点指标
        # 返回的df会包含 chan_buy1, chan_buy2, chan_buy3, chan_strong_buy2, chan_like_buy2 等布尔列
        df = calculate_chan_theory(df)
        
        # 标记六脉神剑4红以上信号
        df['six_veins_4plus'] = df['six_veins_count'] >= MIN_RED_COUNT
        
        # 生成六脉神剑组合名称
        # 例如：如果MACD、KDJ、RSI、BBI四个指标为红，则组合名称为 "MACD+KDJ+RSI+BBI"
        red_cols = ['macd_red', 'kdj_red', 'rsi_red', 'lwr_red', 'bbi_red', 'mtm_red']
        indicator_names = ['MACD', 'KDJ', 'RSI', 'LWR', 'BBI', 'MTM']
        
        def get_red_combo(row):
            """根据各指标红色状态生成组合名称"""
            combo = []
            for col, name in zip(red_cols, indicator_names):
                if row.get(col, False):
                    combo.append(name)
            return '+'.join(combo) if combo else ''
        
        df['six_veins_combo'] = df.apply(get_red_combo, axis=1)
        
        return df
        
    except Exception as e:
        # 信号检测失败时返回原始数据
        return df


def calculate_future_return(df: pd.DataFrame, signal_idx: int, days: int = HOLDING_DAYS) -> Tuple[float, float]:
    """
    计算信号触发后的未来收益
    
    功能说明:
        以信号触发日的收盘价为基准，计算持仓期间的最大涨幅和最终涨幅。
    
    参数:
        df (pd.DataFrame): 股票历史数据
        signal_idx (int): 信号触发日在数据框中的索引位置
        days (int): 持仓天数，默认为HOLDING_DAYS
    
    返回:
        Tuple[float, float]: (最大涨幅百分比, 最终涨幅百分比)
            - 最大涨幅：持仓期间最高价相对于入场价的涨幅
            - 最终涨幅：持仓结束日收盘价相对于入场价的涨幅
            如果数据不足，返回 (np.nan, np.nan)
    
    计算公式:
        最大涨幅 = (期间最高价 - 入场价) / 入场价 * 100
        最终涨幅 = (结束日收盘价 - 入场价) / 入场价 * 100
    """
    # 检查是否有足够的未来数据
    if signal_idx + days >= len(df):
        return np.nan, np.nan
    
    # 获取入场价格（信号触发日的收盘价）
    entry_price = df.iloc[signal_idx]['close']
    
    # 价格有效性检查
    if entry_price <= 0:
        return np.nan, np.nan
    
    # 获取持仓期间的数据切片（从信号日后一天开始）
    future_data = df.iloc[signal_idx + 1: signal_idx + days + 1]
    
    if len(future_data) < days:
        return np.nan, np.nan
    
    # 计算最大涨幅（基于期间最高价）
    max_high = future_data['high'].max()
    max_return = (max_high - entry_price) / entry_price * 100
    
    # 计算最终涨幅（基于持仓结束日收盘价）
    end_price = future_data.iloc[-1]['close']
    final_return = (end_price - entry_price) / entry_price * 100
    
    return max_return, final_return


# ==============================================================================
# 核心扫描函数
# ==============================================================================

def scan_single_stock(filepath: str) -> List[Dict]:
    """
    扫描单只股票的所有买入信号
    
    功能说明:
        对单只股票进行完整的信号扫描和绩效验证。
        这是多进程并行处理的核心函数。
    
    参数:
        filepath (str): 股票CSV文件路径
    
    返回:
        List[Dict]: 信号记录列表，每条记录包含：
            - stock_code: 股票代码（格式：市场_代码，如 sz_000001）
            - stock_name: 股票名称
            - date: 信号日期
            - signal_type: 信号类型
            - signal_detail: 信号详情
            - six_veins_count: 六脉神剑红色数量
            - entry_price: 入场价格
            - max_return_pct: 最大涨幅百分比
            - final_return_pct: 最终涨幅百分比
            - is_success: 是否成功
    """
    results = []
    
    # 从文件路径提取股票代码和市场
    stock_code = os.path.basename(filepath).replace('.csv', '')
    market = os.path.basename(os.path.dirname(filepath))
    
    # 加载数据
    df = load_stock_data(filepath)
    
    # 数据有效性检查：至少需要50条记录
    if df.empty or len(df) < 50:
        return results
    
    # 检测信号
    df = detect_signals(df)
    
    if df.empty:
        return results
    
    # 获取股票名称
    stock_name = df['name'].iloc[0] if 'name' in df.columns else stock_code
    
    # 定义需要检测的信号类型及其对应的列名和显示名称
    signal_types = {
        'six_veins_4plus': '六脉神剑',      # 六脉神剑4红以上
        'chan_buy1': '缠论一买',            # 缠论一买点
        'chan_buy2': '缠论二买',            # 缠论二买点
        'chan_buy3': '缠论三买',            # 缠论三买点
        'chan_strong_buy2': '缠论强二买',   # 缠论强二买
        'chan_like_buy2': '缠论类二买',     # 缠论类二买
        'buy1': '买卖点-买点1',             # 买卖点买点1
        'buy2': '买卖点-买点2',             # 买卖点买点2
    }
    
    # 遍历每个交易日检测信号
    for idx in range(len(df) - HOLDING_DAYS):
        row = df.iloc[idx]
        
        for signal_col, signal_name in signal_types.items():
            # 检查信号列是否存在
            if signal_col not in df.columns:
                continue
            
            # 检查是否触发信号
            if row.get(signal_col, False):
                # 计算未来收益
                max_return, final_return = calculate_future_return(df, idx, HOLDING_DAYS)
                
                # 跳过无效收益
                if pd.isna(final_return):
                    continue
                
                # 判断是否成功（最大涨幅超过阈值）
                is_success = max_return >= SUCCESS_THRESHOLD
                
                # 获取六脉神剑组合信息
                six_veins_combo = ''
                six_veins_count = 0
                if signal_col == 'six_veins_4plus':
                    six_veins_combo = row.get('six_veins_combo', '')
                    six_veins_count = row.get('six_veins_count', 0)
                
                # 构建信号记录
                record = {
                    'stock_code': f"{market}_{stock_code}",
                    'stock_name': stock_name,
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'signal_type': signal_name,
                    'signal_detail': six_veins_combo if signal_col == 'six_veins_4plus' else signal_name,
                    'six_veins_count': int(six_veins_count) if signal_col == 'six_veins_4plus' else 0,
                    'entry_price': float(row['close']),
                    'max_return_pct': round(max_return, 2),
                    'final_return_pct': round(final_return, 2),
                    'is_success': is_success,
                }
                
                results.append(record)
    
    return results


def scan_all_stocks(max_workers: int = None) -> pd.DataFrame:
    """
    扫描所有股票数据
    
    功能说明:
        使用多进程并行处理所有股票文件，提高扫描效率。
    
    参数:
        max_workers (int): 最大并行进程数，默认为CPU核心数-1
    
    返回:
        pd.DataFrame: 所有信号记录的数据框
    """
    # 获取所有股票文件
    stock_files = get_all_stock_files()
    total_files = len(stock_files)
    
    log(f"发现 {total_files} 个股票文件")
    log("-" * 60)
    
    # 设置并行进程数 (cpu核数-1)
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    all_results = []
    processed = 0
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(scan_single_stock, f): f for f in stock_files}
        
        # 收集结果
        for future in as_completed(futures):
            processed += 1
            filepath = futures[future]
            
            try:
                results = future.result()
                all_results.extend(results)
                
                # 每处理100个文件打印一次进度
                if processed % 100 == 0:
                    log(f"进度: {processed}/{total_files} ({processed/total_files*100:.1f}%)")
                    
            except Exception as e:
                log(f"处理失败 {filepath}: {str(e)}")
    
    log(f"\n扫描完成，共发现 {len(all_results)} 条信号记录")
    
    if all_results:
        return pd.DataFrame(all_results)
    else:
        return pd.DataFrame()


# ==============================================================================
# 统计分析函数
# ==============================================================================

def generate_summary(df: pd.DataFrame) -> Dict:
    """
    生成统计摘要
    
    功能说明:
        对所有信号记录进行统计分析，生成供Web前端展示的摘要数据。
    
    参数:
        df (pd.DataFrame): 信号记录DataFrame
    
    返回:
        Dict: 统计摘要，包含：
            - scan_date: 扫描时间
            - total_signals: 总信号数
            - success_signals: 成功信号数
            - overall_success_rate: 总体成功率
            - holding_days: 持仓天数
            - success_threshold: 成功阈值
            - by_signal_type: 按信号类型统计
            - six_veins_combos: 六脉神剑组合统计
    """
    if df.empty:
        return {}
    
    # 基础统计
    summary = {
        'scan_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_signals': len(df),
        'success_signals': int(df['is_success'].sum()),
        'overall_success_rate': round(df['is_success'].mean() * 100, 2),
        'holding_days': HOLDING_DAYS,
        'success_threshold': SUCCESS_THRESHOLD,
        'by_signal_type': {},
        'six_veins_combos': {},
    }
    
    # 按信号类型统计
    for signal_type in df['signal_type'].unique():
        type_df = df[df['signal_type'] == signal_type]
        success_df = type_df[type_df['is_success']]
        
        summary['by_signal_type'][signal_type] = {
            'total': len(type_df),
            'success': len(success_df),
            'success_rate': round(len(success_df) / len(type_df) * 100, 2) if len(type_df) > 0 else 0,
            'avg_max_return': round(type_df['max_return_pct'].mean(), 2),
            'avg_final_return': round(type_df['final_return_pct'].mean(), 2),
        }
    
    # 六脉神剑组合统计
    six_veins_df = df[df['signal_type'] == '六脉神剑']
    if not six_veins_df.empty:
        for combo in six_veins_df['signal_detail'].unique():
            if combo:  # 跳过空组合
                combo_df = six_veins_df[six_veins_df['signal_detail'] == combo]
                success_combo_df = combo_df[combo_df['is_success']]
                
                summary['six_veins_combos'][combo] = {
                    'total': len(combo_df),
                    'success': len(success_combo_df),
                    'success_rate': round(len(success_combo_df) / len(combo_df) * 100, 2) if len(combo_df) > 0 else 0,
                    'avg_max_return': round(combo_df['max_return_pct'].mean(), 2),
                    'red_count': int(combo.count('+')) + 1 if combo else 0,
                }
    
    return summary


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    """
    主函数：执行完整的信号扫描流程
    
    执行流程:
        1. 创建输出目录
        2. 获取所有股票文件列表
        3. 使用多进程并行扫描所有股票
        4. 汇总结果并生成统计
        5. 输出CSV和JSON文件到report目录和web目录
    
    输出文件:
        - report/all_signal_records.csv: 所有信号记录
        - report/signal_success_cases.csv: 成功案例
        - report/signal_summary.json: 统计摘要（报告目录）
        - web/client/src/data/signal_summary.json: 统计摘要（Web前端）
    """
    log("=" * 70)
    log("信号成功案例扫描器 v2.0")
    log("=" * 70)
    log(f"扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"持仓天数: {HOLDING_DAYS} 天")
    log(f"成功阈值: {SUCCESS_THRESHOLD}%")
    log("=" * 70)
    
    # 创建输出目录
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(WEB_DATA_DIR, exist_ok=True)
    
    # 扫描所有股票
    df = scan_all_stocks()
    
    if df.empty:
        log("未发现任何信号记录")
        return
    
    # 保存所有信号记录到CSV
    all_signals_path = os.path.join(REPORT_DIR, 'all_signal_records.csv')
    df.to_csv(all_signals_path, index=False, encoding='utf-8-sig')
    log(f"\n已保存所有信号记录: {all_signals_path}")
    
    # 筛选并保存成功案例
    success_df = df[df['is_success']]
    success_path = os.path.join(REPORT_DIR, 'signal_success_cases.csv')
    success_df.to_csv(success_path, index=False, encoding='utf-8-sig')
    log(f"已保存成功案例 ({len(success_df)} 条): {success_path}")
    
    # 生成统计摘要
    summary = generate_summary(df)
    
    # 保存统计摘要到report目录
    report_summary_path = os.path.join(REPORT_DIR, 'signal_summary.json')
    with open(report_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"已保存统计摘要: {report_summary_path}")
    
    # 保存统计摘要到Web数据目录（供前端使用）
    web_summary_path = os.path.join(WEB_DATA_DIR, 'signal_summary.json')
    with open(web_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"已保存统计摘要到Web目录: {web_summary_path}")
    
    # 打印统计摘要
    log("\n" + "=" * 70)
    log("统计结果")
    log("=" * 70)
    log(f"总信号数: {summary['total_signals']}")
    log(f"成功信号: {summary['success_signals']}")
    log(f"总体成功率: {summary['overall_success_rate']}%")
    log("-" * 70)
    log("按信号类型统计:")
    for signal_type, stats in summary['by_signal_type'].items():
        log(f"  {signal_type}: {stats['total']}个信号, "
              f"成功率{stats['success_rate']}%, "
              f"平均最大涨幅{stats['avg_max_return']}%")
    log("=" * 70)
    log("扫描完成！")


if __name__ == "__main__":
    main()
