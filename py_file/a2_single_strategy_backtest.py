#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
单指标策略回测系统 (Single Strategy Backtest)
================================================================================

功能描述:
    本脚本整合了原101-105五个回测脚本的功能，提供统一的单指标策略回测能力。
    支持六脉神剑、买卖点、缠论买点等多种技术指标策略的全量股票回测。

整合的原脚本:
    - 101_six_veins_test.py      : 六脉神剑策略回测
    - 102_buy_sell_points_test.py : 买卖点策略回测
    - 103_chan_buy_point_test.py  : 缠论买点策略回测（基础3买点）
    - 104_test_sell_points.py     : 卖出点优化测试
    - 105_chan_5buy_test.py       : 缠论5买点策略回测

主要功能:
    1. 六脉神剑策略：测试单指标和组合（4红以上）的胜率
    2. 买卖点策略：测试买点1（吸筹）和买点2（低位金叉）
    3. 缠论买点策略：测试5类买点（一买、二买、三买、强二买、类二买）
    4. 卖出点优化：测试不同持仓周期对收益的影响

输出文件:
    - report/total/single_strategy_report.md       : 综合回测报告（Markdown格式）
    - report/total/single_strategy_summary.csv     : 回测结果汇总（CSV格式）
    - web/client/src/data/backtest_single.json     : 回测数据（供Web前端使用）

使用方法:
    cd ~/tdx-strategy-backtest/py_file
    
    # 运行所有策略回测
    python3 single_strategy_backtest.py
    
    # 运行指定策略回测
    python3 single_strategy_backtest.py --strategy six_veins
    python3 single_strategy_backtest.py --strategy buy_sell
    python3 single_strategy_backtest.py --strategy chan
    python3 single_strategy_backtest.py --strategy sell_optimize

命令行参数:
    --strategy : 指定要运行的策略类型
                 可选值: all, six_veins, buy_sell, chan, sell_optimize
                 默认值: all（运行所有策略）
    --limit    : 限制处理的股票数量（用于测试）
                 默认值: 0（不限制，处理所有股票）

依赖模块:
    - pandas, numpy: 数据处理
    - indicators: 技术指标计算（项目内部模块）
    - multiprocessing: 多进程并行处理

作者: TradeGuide System
版本: 1.0.0
创建日期: 2026-01-15
================================================================================
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from itertools import combinations
try:
    from a99_logger import log, check_memory
except ImportError:
    def log(msg, level="INFO"): print(f"[{level}] {msg}")
    def check_memory(t=0.9): pass

# ==============================================================================
# 路径配置
# ==============================================================================

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加项目路径到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')

# 报告输出目录
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report', 'total')

# Web数据目录
WEB_DATA_DIR = os.path.join(PROJECT_ROOT, 'web', 'client', 'src', 'data')

# ==============================================================================
# 回测参数配置
# ==============================================================================

# 六脉神剑指标列表
SIX_VEINS_INDICATORS = ['macd_red', 'kdj_red', 'rsi_red', 'lwr_red', 'bbi_red', 'mtm_red']

# 六脉神剑指标名称映射
INDICATOR_NAMES = {
    'macd_red': 'MACD',
    'kdj_red': 'KDJ',
    'rsi_red': 'RSI',
    'lwr_red': 'LWR',
    'bbi_red': 'BBI',
    'mtm_red': 'MTM'
}

# 缠论买点信号列表
CHAN_BUY_SIGNALS = ['chan_buy1', 'chan_buy2', 'chan_buy3', 'chan_strong_buy2', 'chan_like_buy2']

# 缠论买点名称映射
CHAN_SIGNAL_NAMES = {
    'chan_buy1': '缠论一买',
    'chan_buy2': '缠论二买',
    'chan_buy3': '缠论三买',
    'chan_strong_buy2': '缠论强二买',
    'chan_like_buy2': '缠论类二买'
}

# 买卖点信号映射
BUY_SELL_SIGNALS = {
    'buy1': '买点1 (吸筹)',
    'buy2': '买点2 (低位金叉)'
}

# 默认持仓周期
DEFAULT_HOLD_PERIODS = [5, 10, 20]

# 卖出优化持仓周期
SELL_OPTIMIZE_PERIODS = [5, 10, 15, 20, 30]

# ==============================================================================
# 导入技术指标计算模块
# ==============================================================================

from a99_indicators import (
    calculate_six_veins,
    calculate_buy_sell_points,
    calculate_chan_theory,
    calculate_all_signals
)

from a99_backtest_utils import get_all_stock_files, aggregate_results


# ==============================================================================
# 数据加载函数
# ==============================================================================

def load_stock_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    加载单只股票的CSV数据
    
    功能说明:
        读取CSV文件，统一列名格式，并按日期排序。
    
    参数:
        filepath (str): CSV文件的完整路径
    
    返回:
        pd.DataFrame: 标准化后的股票数据
        如果数据不足100条或读取失败，返回None
    """
    try:
        df = pd.read_csv(filepath)
        
        # 数据量检查
        if len(df) < 100:
            return None
        
        # 列名映射
        name_map = {
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
        df = df.rename(columns=name_map)
        
        # 日期处理
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        return None


# ==============================================================================
# 收益计算函数
# ==============================================================================

def calculate_returns(df: pd.DataFrame, signal_col: str, hold_period: int) -> Dict:
    """
    计算指定信号的回测收益
    
    功能说明:
        基于信号列计算持仓期间的胜率和平均收益。
    
    参数:
        df (pd.DataFrame): 股票数据（包含信号列）
        signal_col (str): 信号列名
        hold_period (int): 持仓天数
    
    返回:
        Dict: 回测统计结果
            - signal_count: 信号总数
            - trade_count: 有效交易数
            - win_rate: 胜率（百分比）
            - avg_return: 平均收益（百分比）
    """
    # 获取信号触发点
    signals = df[df[signal_col] == True].copy()
    
    if len(signals) == 0:
        return {
            'signal_count': 0,
            'trade_count': 0,
            'win_rate': np.nan,
            'avg_return': np.nan
        }
    
    returns = []
    
    # 遍历每个信号计算收益
    for idx in signals.index:
        # 检查是否有足够的未来数据
        if idx + hold_period < len(df):
            entry_price = df.loc[idx, 'close']
            exit_price = df.loc[idx + hold_period, 'close']
            
            if entry_price > 0:
                ret = (exit_price - entry_price) / entry_price * 100
                returns.append(ret)
    
    if len(returns) == 0:
        return {
            'signal_count': len(signals),
            'trade_count': 0,
            'win_rate': np.nan,
            'avg_return': np.nan
        }
    
    returns = np.array(returns)
    win_rate = np.sum(returns > 0) / len(returns) * 100
    
    return {
        'signal_count': len(signals),
        'trade_count': len(returns),
        'win_rate': win_rate,
        'avg_return': np.mean(returns)
    }


# ==============================================================================
# 六脉神剑策略回测
# ==============================================================================

def backtest_six_veins_single(filepath: str) -> Optional[pd.DataFrame]:
    """
    六脉神剑策略 - 单只股票回测
    
    功能说明:
        测试六脉神剑各单指标和组合（4红以上）的表现。
    
    参数:
        filepath (str): 股票CSV文件路径
    
    返回:
        pd.DataFrame: 回测结果，如果失败返回None
    """
    df = load_stock_data(filepath)
    if df is None:
        return None
    
    try:
        # 计算六脉神剑指标
        df = calculate_six_veins(df)
        
        results = []
        
        # 1. 测试单指标
        for indicator in SIX_VEINS_INDICATORS:
            # 生成信号列（当天变红）
            signal_col = f'{indicator}_signal'
            df[signal_col] = df[indicator] & ~df[indicator].shift(1, fill_value=False)
            
            for period in DEFAULT_HOLD_PERIODS:
                stats = calculate_returns(df, signal_col, period)
                if stats['trade_count'] > 0:
                    results.append({
                        'strategy': '六脉神剑',
                        'type': '单指标',
                        'name': INDICATOR_NAMES[indicator],
                        'hold_period': period,
                        **stats
                    })
        
        # 2. 测试N红组合（4红、5红、6红）
        for n in [4, 5, 6]:
            signal_col = f'{n}_red_signal'
            df[f'{n}_red'] = df['six_veins_count'] >= n
            df[signal_col] = df[f'{n}_red'] & ~df[f'{n}_red'].shift(1, fill_value=False)
            
            for period in DEFAULT_HOLD_PERIODS:
                stats = calculate_returns(df, signal_col, period)
                if stats['trade_count'] > 0:
                    results.append({
                        'strategy': '六脉神剑',
                        'type': '组合',
                        'name': f'≥{n}红',
                        'hold_period': period,
                        **stats
                    })
        
        return pd.DataFrame(results) if results else None
        
    except Exception as e:
        return None


# ==============================================================================
# 买卖点策略回测
# ==============================================================================

def backtest_buy_sell_single(filepath: str) -> Optional[pd.DataFrame]:
    """
    买卖点策略 - 单只股票回测
    
    功能说明:
        测试买点1（吸筹）和买点2（低位金叉）的表现。
    
    参数:
        filepath (str): 股票CSV文件路径
    
    返回:
        pd.DataFrame: 回测结果，如果失败返回None
    """
    df = load_stock_data(filepath)
    if df is None:
        return None
    
    try:
        # 计算买卖点指标
        df = calculate_buy_sell_points(df)
        
        results = []
        
        for signal_col, signal_name in BUY_SELL_SIGNALS.items():
            for period in DEFAULT_HOLD_PERIODS:
                stats = calculate_returns(df, signal_col, period)
                if stats['trade_count'] > 0:
                    results.append({
                        'strategy': '买卖点',
                        'type': '买点',
                        'name': signal_name,
                        'hold_period': period,
                        **stats
                    })
        
        return pd.DataFrame(results) if results else None
        
    except Exception as e:
        return None


# ==============================================================================
# 缠论买点策略回测
# ==============================================================================

def backtest_chan_single(filepath: str) -> Optional[pd.DataFrame]:
    """
    缠论买点策略 - 单只股票回测
    
    功能说明:
        测试缠论5类买点（一买、二买、三买、强二买、类二买）的表现。
    
    参数:
        filepath (str): 股票CSV文件路径
    
    返回:
        pd.DataFrame: 回测结果，如果失败返回None
    """
    df = load_stock_data(filepath)
    if df is None:
        return None
    
    try:
        # 计算缠论指标
        df = calculate_chan_theory(df)
        
        results = []
        
        for signal_col in CHAN_BUY_SIGNALS:
            signal_name = CHAN_SIGNAL_NAMES[signal_col]
            
            for period in DEFAULT_HOLD_PERIODS:
                stats = calculate_returns(df, signal_col, period)
                if stats['trade_count'] > 0:
                    results.append({
                        'strategy': '缠论买点',
                        'type': '买点',
                        'name': signal_name,
                        'hold_period': period,
                        **stats
                    })
        
        return pd.DataFrame(results) if results else None
        
    except Exception as e:
        return None


# ==============================================================================
# 卖出点优化回测
# ==============================================================================

def backtest_sell_optimize_single(filepath: str) -> Optional[pd.DataFrame]:
    """
    卖出点优化 - 单只股票回测
    
    功能说明:
        以六脉6红作为买入基准，测试不同持仓周期对收益的影响。
    
    参数:
        filepath (str): 股票CSV文件路径
    
    返回:
        pd.DataFrame: 回测结果，如果失败返回None
    """
    df = load_stock_data(filepath)
    if df is None:
        return None
    
    try:
        # 计算六脉神剑指标
        df = calculate_six_veins(df)
        
        # 以六脉6红作为买入信号
        df['buy_signal'] = (df['six_veins_count'] == 6) & (df['six_veins_count'].shift(1) < 6)
        
        results = []
        
        for period in SELL_OPTIMIZE_PERIODS:
            stats = calculate_returns(df, 'buy_signal', period)
            if stats['trade_count'] > 0:
                results.append({
                    'strategy': '卖出优化',
                    'type': '固定周期',
                    'name': f'{period}天卖出',
                    'hold_period': period,
                    **stats
                })
        
        return pd.DataFrame(results) if results else None
        
    except Exception as e:
        return None


# ==============================================================================
# 综合回测函数
# ==============================================================================

def run_backtest(strategy: str, stock_files: List[str]) -> pd.DataFrame:
    """
    运行指定策略的回测
    
    参数:
        strategy (str): 策略类型
        stock_files (List[str]): 股票文件列表
    
    返回:
        pd.DataFrame: 汇总后的回测结果
    """
    # 选择回测函数
    backtest_funcs = {
        'six_veins': backtest_six_veins_single,
        'buy_sell': backtest_buy_sell_single,
        'chan': backtest_chan_single,
        'sell_optimize': backtest_sell_optimize_single
    }
    
    backtest_func = backtest_funcs.get(strategy)
    if backtest_func is None:
        log(f"未知策略类型: {strategy}")
        return pd.DataFrame()
    
    strategy_names = {
        'six_veins': '六脉神剑',
        'buy_sell': '买卖点',
        'chan': '缠论买点',
        'sell_optimize': '卖出优化'
    }
    
    log(f"\n开始回测: {strategy_names[strategy]}")
    log(f"处理 {len(stock_files)} 个股票文件...")
    
    # 多进程并行处理 (cpu核数-1)
    num_processes = max(1, cpu_count() - 1)
    with Pool(num_processes) as pool:
        all_results = pool.map(backtest_func, stock_files)
    
    # 过滤空结果
    all_results = [r for r in all_results if r is not None and not r.empty]
    
    if not all_results:
        log(f"  {strategy_names[strategy]}: 没有有效的回测结果")
        return pd.DataFrame()
    
    # 汇总结果
    combined = pd.concat(all_results, ignore_index=True)
    
    # 按策略、类型、名称、持仓周期分组汇总
    group_cols = ['strategy', 'type', 'name', 'hold_period']
    summary = combined.groupby(group_cols).agg({
        'signal_count': 'sum',
        'trade_count': 'sum',
        'win_rate': 'mean',
        'avg_return': 'mean'
    }).reset_index()
    
    log(f"  {strategy_names[strategy]}: 完成，共 {len(summary)} 条汇总记录")
    
    return summary


# ==============================================================================
# 报告生成函数
# ==============================================================================

def generate_markdown_report(results: pd.DataFrame, stock_count: int) -> str:
    """
    生成Markdown格式的回测报告
    
    参数:
        results (pd.DataFrame): 回测结果
        stock_count (int): 测试股票数量
    
    返回:
        str: Markdown格式的报告内容
    """
    report = []
    report.append("# 单指标策略回测报告\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**测试股票数量**: {stock_count}\n\n")
    
    # 按策略分组输出
    for strategy in results['strategy'].unique():
        strategy_df = results[results['strategy'] == strategy]
        
        report.append(f"## {strategy}\n\n")
        report.append("| 类型 | 名称 | 持仓周期 | 信号数 | 交易数 | 胜率 | 平均收益 |\n")
        report.append("| --- | --- | --- | --- | --- | --- | --- |\n")
        
        for _, row in strategy_df.sort_values(['type', 'win_rate'], ascending=[True, False]).iterrows():
            report.append(
                f"| {row['type']} | {row['name']} | {row['hold_period']}天 | "
                f"{int(row['signal_count'])} | {int(row['trade_count'])} | "
                f"{row['win_rate']:.2f}% | {row['avg_return']:.2f}% |\n"
            )
        
        report.append("\n")
    
    return ''.join(report)


def generate_json_data(results: pd.DataFrame, stock_count: int) -> Dict:
    """
    生成供Web前端使用的JSON数据
    
    参数:
        results (pd.DataFrame): 回测结果
        stock_count (int): 测试股票数量
    
    返回:
        Dict: JSON格式的数据
    """
    data = {
        'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock_count': stock_count,
        'strategies': {}
    }
    
    for strategy in results['strategy'].unique():
        strategy_df = results[results['strategy'] == strategy]
        
        data['strategies'][strategy] = {
            'summary': [],
            'best_performers': []
        }
        
        # 添加所有结果
        for _, row in strategy_df.iterrows():
            data['strategies'][strategy]['summary'].append({
                'type': row['type'],
                'name': row['name'],
                'hold_period': int(row['hold_period']),
                'signal_count': int(row['signal_count']),
                'trade_count': int(row['trade_count']),
                'win_rate': round(row['win_rate'], 2),
                'avg_return': round(row['avg_return'], 2)
            })
        
        # 添加最佳表现（按胜率排序前5）
        top5 = strategy_df.nlargest(5, 'win_rate')
        for _, row in top5.iterrows():
            data['strategies'][strategy]['best_performers'].append({
                'name': f"{row['type']}-{row['name']}",
                'hold_period': int(row['hold_period']),
                'win_rate': round(row['win_rate'], 2),
                'avg_return': round(row['avg_return'], 2)
            })
    
    return data


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    """
    主函数：执行单指标策略回测
    
    支持通过命令行参数选择要运行的策略类型。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='单指标策略回测系统')
    parser.add_argument(
        '--strategy',
        type=str,
        default='all',
        choices=['all', 'six_veins', 'buy_sell', 'chan', 'sell_optimize'],
        help='要运行的策略类型 (默认: all)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='限制处理的股票数量，0表示不限制 (默认: 0)'
    )
    
    args = parser.parse_args()
    
    log("=" * 70)
    log("单指标策略回测系统 v1.0")
    log("=" * 70)
    log(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"策略选择: {args.strategy}")
    log("=" * 70)
    
    # 创建输出目录
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(WEB_DATA_DIR, exist_ok=True)
    
    # 获取所有股票文件
    stock_files = get_all_stock_files(DATA_DIR)
    
    # 限制股票数量（用于测试）
    if args.limit > 0:
        stock_files = stock_files[:args.limit]
        log(f"注意: 已限制处理 {args.limit} 个股票文件")
    
    log(f"发现 {len(stock_files)} 个股票文件")
    
    if not stock_files:
        log("错误: 未找到股票数据文件")
        return
    
    # 确定要运行的策略
    if args.strategy == 'all':
        strategies = ['six_veins', 'buy_sell', 'chan', 'sell_optimize']
    else:
        strategies = [args.strategy]
    
    # 运行回测
    all_results = []
    for strategy in strategies:
        result = run_backtest(strategy, stock_files)
        if not result.empty:
            all_results.append(result)
    
    if not all_results:
        log("\n没有生成任何回测结果")
        return
    
    # 合并所有结果
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # 生成报告
    log("\n" + "-" * 70)
    log("生成报告...")
    
    # 保存CSV
    csv_path = os.path.join(REPORT_DIR, 'single_strategy_summary.csv')
    combined_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    log(f"已保存CSV汇总: {csv_path}")
    
    # 生成并保存Markdown报告
    md_report = generate_markdown_report(combined_results, len(stock_files))
    md_path = os.path.join(REPORT_DIR, 'single_strategy_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    log(f"已保存Markdown报告: {md_path}")
    
    # 生成并保存JSON数据（供Web前端使用）
    json_data = generate_json_data(combined_results, len(stock_files))
    
    # 保存到report目录
    json_path = os.path.join(REPORT_DIR, 'backtest_single.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    log(f"已保存JSON数据: {json_path}")
    
    # 保存到Web数据目录
    web_json_path = os.path.join(WEB_DATA_DIR, 'backtest_single.json')
    with open(web_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    log(f"已保存JSON数据到Web目录: {web_json_path}")
    
    # 打印摘要
    log("\n" + "=" * 70)
    log("回测完成！")
    log("=" * 70)
    log(f"测试股票数: {len(stock_files)}")
    log(f"策略数量: {len(strategies)}")
    log(f"汇总记录: {len(combined_results)}")
    log("=" * 70)


if __name__ == "__main__":
    main()
