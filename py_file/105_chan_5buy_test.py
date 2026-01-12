#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
缠论5个买点独立测试脚本 (105_chan_5buy_test.py)
================================================================================

功能说明:
    本脚本对缠论5个买点进行独立回测，分析每个买点的胜率和收益表现。
    
    测试的5个买点:
    1. 一买 (chan_buy1) - 底背驰买点
    2. 二买 (chan_buy2) - 回踩确认买点
    3. 三买 (chan_buy3) - 中枢突破买点
    4. 强二买 (chan_strong_buy2) - 强势二买
    5. 类二买 (chan_like_buy2) - 类似二买

使用方法:
    python 105_chan_5buy_test.py

输出:
    - 控制台: 各买点的回测统计结果
    - 文件: web/client/src/data/chan_backtest_results.json

作者: TradeGuide System
版本: 1.0.0
更新日期: 2026-01-12
================================================================================
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import calculate_chan_theory, calculate_six_veins, calculate_buy_sell_points


# ==============================================================================
# 配置参数
# ==============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'web', 'client', 'src', 'data')

# 回测参数
HOLD_DAYS = 5  # 持有天数
STOP_LOSS = -0.05  # 止损比例 (-5%)
TAKE_PROFIT = 0.10  # 止盈比例 (+10%)

# 缠论5个买点
CHAN_BUY_SIGNALS = [
    'chan_buy1',        # 一买
    'chan_buy2',        # 二买
    'chan_buy3',        # 三买
    'chan_strong_buy2', # 强二买
    'chan_like_buy2',   # 类二买
]

# 买点中文名称映射
SIGNAL_NAMES = {
    'chan_buy1': '缠论一买',
    'chan_buy2': '缠论二买',
    'chan_buy3': '缠论三买',
    'chan_strong_buy2': '缠论强二买',
    'chan_like_buy2': '缠论类二买',
}


# ==============================================================================
# 数据加载函数
# ==============================================================================

def load_stock_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    加载单只股票数据
    
    参数:
        file_path: CSV文件路径
        
    返回:
        pd.DataFrame 或 None
    """
    try:
        df = pd.read_csv(file_path, encoding='gbk')
        
        # 标准化列名
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '名称': 'name'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # 确保必要列存在
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None
        
        # 转换日期
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # 过滤无效数据
        df = df[df['close'] > 0]
        
        if len(df) < 50:
            return None
            
        return df
        
    except Exception as e:
        return None


def get_all_stock_files() -> List[str]:
    """
    获取所有股票CSV文件路径
    
    返回:
        List[str]: 文件路径列表
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
# 回测函数
# ==============================================================================

def backtest_single_signal(df: pd.DataFrame, signal_col: str) -> Dict:
    """
    对单个信号进行回测
    
    参数:
        df: 包含信号的DataFrame
        signal_col: 信号列名
        
    返回:
        Dict: 回测结果
    """
    results = {
        'total_trades': 0,
        'win_trades': 0,
        'loss_trades': 0,
        'total_return': 0.0,
        'returns': []
    }
    
    if signal_col not in df.columns:
        return results
    
    signal_indices = df[df[signal_col] == True].index.tolist()
    
    for idx in signal_indices:
        if idx + HOLD_DAYS >= len(df):
            continue
        
        buy_price = df.loc[idx, 'close']
        if buy_price <= 0:
            continue
        
        # 计算持有期收益
        best_return = 0
        final_return = 0
        
        for hold_day in range(1, HOLD_DAYS + 1):
            if idx + hold_day >= len(df):
                break
            
            sell_price = df.loc[idx + hold_day, 'close']
            daily_return = (sell_price - buy_price) / buy_price
            
            # 检查止损止盈
            if daily_return <= STOP_LOSS:
                final_return = STOP_LOSS
                break
            elif daily_return >= TAKE_PROFIT:
                final_return = TAKE_PROFIT
                break
            
            best_return = max(best_return, daily_return)
            final_return = daily_return
        
        results['total_trades'] += 1
        results['returns'].append(final_return)
        results['total_return'] += final_return
        
        if final_return > 0:
            results['win_trades'] += 1
        else:
            results['loss_trades'] += 1
    
    return results


def process_single_stock(file_path: str) -> Dict:
    """
    处理单只股票的回测
    
    参数:
        file_path: 股票CSV文件路径
        
    返回:
        Dict: 各信号的回测结果
    """
    df = load_stock_data(file_path)
    if df is None:
        return {}
    
    # 计算缠论信号
    try:
        df = calculate_chan_theory(df)
    except Exception as e:
        return {}
    
    # 对每个买点信号进行回测
    results = {}
    for signal in CHAN_BUY_SIGNALS:
        results[signal] = backtest_single_signal(df, signal)
    
    return results


def aggregate_results(all_results: List[Dict]) -> Dict:
    """
    汇总所有股票的回测结果
    
    参数:
        all_results: 所有股票的回测结果列表
        
    返回:
        Dict: 汇总后的结果
    """
    aggregated = {}
    
    for signal in CHAN_BUY_SIGNALS:
        aggregated[signal] = {
            'name': SIGNAL_NAMES.get(signal, signal),
            'total_trades': 0,
            'win_trades': 0,
            'loss_trades': 0,
            'total_return': 0.0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'all_returns': []
        }
    
    for result in all_results:
        for signal in CHAN_BUY_SIGNALS:
            if signal in result:
                sig_result = result[signal]
                aggregated[signal]['total_trades'] += sig_result['total_trades']
                aggregated[signal]['win_trades'] += sig_result['win_trades']
                aggregated[signal]['loss_trades'] += sig_result['loss_trades']
                aggregated[signal]['total_return'] += sig_result['total_return']
                aggregated[signal]['all_returns'].extend(sig_result['returns'])
    
    # 计算胜率和平均收益
    for signal in CHAN_BUY_SIGNALS:
        total = aggregated[signal]['total_trades']
        if total > 0:
            aggregated[signal]['win_rate'] = round(
                aggregated[signal]['win_trades'] / total * 100, 2
            )
            aggregated[signal]['avg_return'] = round(
                aggregated[signal]['total_return'] / total * 100, 2
            )
        
        # 移除详细收益列表（太大）
        del aggregated[signal]['all_returns']
    
    return aggregated


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    """主程序入口"""
    print("=" * 60)
    print("缠论5个买点独立测试")
    print("=" * 60)
    
    # 获取所有股票文件
    stock_files = get_all_stock_files()
    print(f"\n找到 {len(stock_files)} 只股票数据")
    
    if len(stock_files) == 0:
        print("错误: 未找到股票数据文件")
        return
    
    # 使用多进程并行处理
    num_workers = max(1, cpu_count() - 1)
    print(f"使用 {num_workers} 个进程并行处理...")
    
    with Pool(num_workers) as pool:
        all_results = pool.map(process_single_stock, stock_files)
    
    # 过滤空结果
    all_results = [r for r in all_results if r]
    print(f"成功处理 {len(all_results)} 只股票")
    
    # 汇总结果
    aggregated = aggregate_results(all_results)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("回测结果汇总")
    print("=" * 60)
    print(f"{'买点类型':<15} {'交易次数':>10} {'胜率':>10} {'平均收益':>10}")
    print("-" * 60)
    
    for signal in CHAN_BUY_SIGNALS:
        data = aggregated[signal]
        print(f"{data['name']:<15} {data['total_trades']:>10} {data['win_rate']:>9.2f}% {data['avg_return']:>9.2f}%")
    
    # 保存结果到JSON
    output_data = {
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hold_days': HOLD_DAYS,
        'stop_loss': STOP_LOSS,
        'take_profit': TAKE_PROFIT,
        'signals': aggregated
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, 'chan_backtest_results.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    print("\n测试完成!")


if __name__ == "__main__":
    main()
