#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
信号成功案例扫描器 (signal_success_scanner.py)
================================================================================

功能说明:
    本脚本用于自动化扫描股票历史数据，验证特定技术买入信号的有效性：
    1. 识别三类买入信号：六脉神剑(4红以上)、缠论5类买点、买卖点(买点1/2)
    2. 计算信号后15个交易日涨幅
    3. 筛选涨幅>5%的成功案例
    4. 输出中间结果供模式分析脚本使用

数据输入:
    - /data/day/ 目录下的股票CSV文件
    - 字段：名称、日期、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌幅、涨跌额、换手率

信号检测规则:
    - 六脉神剑信号："4红以上"状态
    - 缠论买点信号：5类标准买点信号
    - 买卖点："买点1"和"买点2"信号

绩效验证条件:
    - 成功标准：信号出现日收盘价为基准，15个交易日后涨幅>5%

输出:
    - report/signal_success_cases.csv: 所有成功案例详细记录
    - report/signal_summary.json: 信号统计摘要

作者: TradeGuide System
版本: 1.0.0
创建日期: 2026-01-15
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入指标计算模块
from indicators import (
    calculate_six_veins, 
    calculate_buy_sell_points, 
    calculate_chan_theory,
    REF, MA, EMA, SMA, HHV, LLV, CROSS, COUNT, ABS, MAX, IF
)


# ==============================================================================
# 配置常量
# ==============================================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')

# 报告输出目录
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')

# 绩效验证参数
HOLDING_DAYS = 15  # 持仓天数
SUCCESS_THRESHOLD = 5.0  # 成功阈值（涨幅百分比）

# 六脉神剑最小红色数量
MIN_RED_COUNT = 4


# ==============================================================================
# 数据加载函数
# ==============================================================================

def load_stock_data(filepath: str) -> pd.DataFrame:
    """
    加载单只股票的CSV数据
    
    参数:
        filepath: CSV文件路径
        
    返回:
        pd.DataFrame: 标准化后的股票数据
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        
        # 标准化列名映射
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
        
        # 确保日期列是datetime类型
        df['date'] = pd.to_datetime(df['date'])
        
        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 过滤无效数据
        df = df[df['close'] > 0]
        
        return df
        
    except Exception as e:
        print(f"加载文件失败 {filepath}: {str(e)}")
        return pd.DataFrame()


def get_all_stock_files() -> List[str]:
    """
    获取所有股票CSV文件路径
    
    返回:
        List[str]: CSV文件路径列表
    """
    stock_files = []
    
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
    
    参数:
        df: 股票数据DataFrame
        
    返回:
        pd.DataFrame: 添加了信号列的DataFrame
    """
    if len(df) < 30:  # 数据太少无法计算指标
        return df
    
    try:
        # 计算六脉神剑指标
        df = calculate_six_veins(df)
        
        # 计算买卖点指标
        df = calculate_buy_sell_points(df)
        
        # 计算缠论指标
        df = calculate_chan_theory(df)
        
        # 六脉神剑4红以上信号
        df['six_veins_4plus'] = df['six_veins_count'] >= MIN_RED_COUNT
        
        # 生成六脉神剑组合标识
        red_cols = ['macd_red', 'kdj_red', 'rsi_red', 'lwr_red', 'bbi_red', 'mtm_red']
        indicator_names = ['MACD', 'KDJ', 'RSI', 'LWR', 'BBI', 'MTM']
        
        def get_red_combo(row):
            """获取红色指标组合名称"""
            combo = []
            for col, name in zip(red_cols, indicator_names):
                if row.get(col, False):
                    combo.append(name)
            return '+'.join(combo) if combo else ''
        
        df['six_veins_combo'] = df.apply(get_red_combo, axis=1)
        
        return df
        
    except Exception as e:
        print(f"信号检测失败: {str(e)}")
        return df


def calculate_future_return(df: pd.DataFrame, signal_idx: int, days: int = HOLDING_DAYS) -> Tuple[float, float]:
    """
    计算信号后N天的收益率
    
    参数:
        df: 股票数据DataFrame
        signal_idx: 信号出现的行索引
        days: 持仓天数
        
    返回:
        Tuple[float, float]: (最大涨幅, N天后涨幅)
    """
    if signal_idx + days >= len(df):
        return np.nan, np.nan
    
    entry_price = df.iloc[signal_idx]['close']
    
    if entry_price <= 0:
        return np.nan, np.nan
    
    # 计算未来N天的数据
    future_data = df.iloc[signal_idx + 1: signal_idx + days + 1]
    
    if len(future_data) < days:
        return np.nan, np.nan
    
    # 最大涨幅（期间最高价）
    max_high = future_data['high'].max()
    max_return = (max_high - entry_price) / entry_price * 100
    
    # N天后涨幅（收盘价）
    end_price = future_data.iloc[-1]['close']
    final_return = (end_price - entry_price) / entry_price * 100
    
    return max_return, final_return


# ==============================================================================
# 核心扫描函数
# ==============================================================================

def scan_single_stock(filepath: str) -> List[Dict]:
    """
    扫描单只股票的所有信号
    
    参数:
        filepath: 股票CSV文件路径
        
    返回:
        List[Dict]: 信号记录列表
    """
    results = []
    
    # 提取股票代码
    stock_code = os.path.basename(filepath).replace('.csv', '')
    market = os.path.basename(os.path.dirname(filepath))
    
    # 加载数据
    df = load_stock_data(filepath)
    
    if df.empty or len(df) < 50:
        return results
    
    # 检测信号
    df = detect_signals(df)
    
    if df.empty:
        return results
    
    # 获取股票名称
    stock_name = df['name'].iloc[0] if 'name' in df.columns else stock_code
    
    # 定义信号类型映射
    signal_types = {
        # 六脉神剑信号（4红以上）
        'six_veins_4plus': '六脉神剑',
        # 缠论买点
        'chan_buy1': '缠论一买',
        'chan_buy2': '缠论二买',
        'chan_buy3': '缠论三买',
        'chan_strong_buy2': '缠论强二买',
        'chan_like_buy2': '缠论类二买',
        # 买卖点
        'buy1': '买卖点-买点1',
        'buy2': '买卖点-买点2',
    }
    
    # 遍历每个交易日检测信号
    for idx in range(len(df) - HOLDING_DAYS):
        row = df.iloc[idx]
        
        for signal_col, signal_name in signal_types.items():
            if signal_col not in df.columns:
                continue
                
            if row.get(signal_col, False):
                # 计算未来收益
                max_return, final_return = calculate_future_return(df, idx, HOLDING_DAYS)
                
                if pd.isna(final_return):
                    continue
                
                # 判断是否成功
                is_success = max_return >= SUCCESS_THRESHOLD
                
                # 获取六脉神剑组合信息
                six_veins_combo = ''
                six_veins_count = 0
                if signal_col == 'six_veins_4plus':
                    six_veins_combo = row.get('six_veins_combo', '')
                    six_veins_count = row.get('six_veins_count', 0)
                
                # 记录信号
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
                    # 保存原始数据用于后续分析
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'amount': float(row.get('amount', 0)),
                    'turnover': float(row.get('turnover', 0)),
                    'pct_change': float(row.get('pct_change', 0)),
                }
                
                results.append(record)
    
    return results


def scan_all_stocks(max_workers: int = None) -> pd.DataFrame:
    """
    扫描所有股票数据
    
    参数:
        max_workers: 最大并行进程数，默认为CPU核心数
        
    返回:
        pd.DataFrame: 所有信号记录
    """
    stock_files = get_all_stock_files()
    total_files = len(stock_files)
    
    print(f"发现 {total_files} 个股票文件")
    print("=" * 60)
    
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 2)
    
    all_results = []
    processed = 0
    
    # 使用多进程加速
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_single_stock, f): f for f in stock_files}
        
        for future in as_completed(futures):
            processed += 1
            filepath = futures[future]
            
            try:
                results = future.result()
                all_results.extend(results)
                
                if processed % 100 == 0:
                    print(f"进度: {processed}/{total_files} ({processed/total_files*100:.1f}%)")
                    
            except Exception as e:
                print(f"处理失败 {filepath}: {str(e)}")
    
    print(f"\n扫描完成，共发现 {len(all_results)} 条信号记录")
    
    if all_results:
        return pd.DataFrame(all_results)
    else:
        return pd.DataFrame()


# ==============================================================================
# 统计分析函数
# ==============================================================================

def generate_summary(df: pd.DataFrame) -> Dict:
    """
    生成信号统计摘要
    
    参数:
        df: 信号记录DataFrame
        
    返回:
        Dict: 统计摘要
    """
    if df.empty:
        return {}
    
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
            if combo:
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
    主程序入口
    """
    print("=" * 60)
    print("信号成功案例扫描器")
    print("=" * 60)
    print(f"扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"持仓天数: {HOLDING_DAYS} 天")
    print(f"成功阈值: {SUCCESS_THRESHOLD}%")
    print("=" * 60)
    
    # 确保报告目录存在
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 扫描所有股票
    df = scan_all_stocks()
    
    if df.empty:
        print("未发现任何信号记录")
        return
    
    # 保存所有信号记录
    all_signals_path = os.path.join(REPORT_DIR, 'all_signal_records.csv')
    df.to_csv(all_signals_path, index=False, encoding='utf-8-sig')
    print(f"\n所有信号记录已保存: {all_signals_path}")
    
    # 筛选成功案例
    success_df = df[df['is_success']]
    success_path = os.path.join(REPORT_DIR, 'signal_success_cases.csv')
    success_df.to_csv(success_path, index=False, encoding='utf-8-sig')
    print(f"成功案例已保存: {success_path}")
    
    # 生成统计摘要
    summary = generate_summary(df)
    summary_path = os.path.join(REPORT_DIR, 'signal_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"统计摘要已保存: {summary_path}")
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"总信号数: {summary['total_signals']}")
    print(f"成功信号数: {summary['success_signals']}")
    print(f"总体成功率: {summary['overall_success_rate']}%")
    
    print("\n按信号类型统计:")
    print("-" * 60)
    for signal_type, stats in summary['by_signal_type'].items():
        print(f"  {signal_type}:")
        print(f"    总数: {stats['total']}, 成功: {stats['success']}, 成功率: {stats['success_rate']}%")
        print(f"    平均最大涨幅: {stats['avg_max_return']}%, 平均最终涨幅: {stats['avg_final_return']}%")
    
    if summary['six_veins_combos']:
        print("\n六脉神剑组合统计 (按成功率排序):")
        print("-" * 60)
        sorted_combos = sorted(
            summary['six_veins_combos'].items(), 
            key=lambda x: x[1]['success_rate'], 
            reverse=True
        )
        for combo, stats in sorted_combos[:20]:  # 显示前20个组合
            print(f"  {combo} ({stats['red_count']}红):")
            print(f"    总数: {stats['total']}, 成功: {stats['success']}, 成功率: {stats['success_rate']}%")
    
    print("\n" + "=" * 60)
    print("扫描完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
