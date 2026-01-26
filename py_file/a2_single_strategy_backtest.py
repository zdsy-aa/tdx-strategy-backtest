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
    python3 a2_single_strategy_backtest.py --strategy all
    
    # 仅运行六脉神剑策略回测
    python3 a2_single_strategy_backtest.py --strategy six_veins
"""

try:
    from a99_logger import log
except ImportError:
    def log(msg, level="INFO"): print(f"[{level}] {msg}")

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Optional

# 添加项目路径，以便导入内部模块
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

from a99_indicators import (
    calculate_six_veins,
    calculate_buy_sell_points,
    calculate_chan_theory,
    calculate_all_signals
)
from a99_backtest_utils import (
    get_all_stock_files,
    run_backtest_on_all_stocks,
    backtest_trades_fixed_hold,
    summarize_trades,
)

# 回测参数
COMMISSION_RATE = 0.00008  # 佣金费率
STAMP_TAX_RATE = 0.0005    # 印花税率

# 默认持仓周期列表
DEFAULT_HOLD_PERIODS = [5, 10, 20]

# 定义六脉神剑指标列（对应 calculate_six_veins 输出的列）
SIX_VEINS_INDICATORS = ['macd_red', 'kdj_red', 'rsi_red', 'lwr_red', 'bbi_red', 'mtm_red']

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
        # 列名映射：将中文列名转换为英文列名
        column_map = {
            '日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'
        }
        df.rename(columns={c: column_map.get(c, c) for c in df.columns}, inplace=True)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        # 数据量检查
        if len(df) < 100:
            return None
        # 排序并reset索引
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        log(f"数据加载失败: {filepath}, 错误: {e}", level="ERROR")
        return None

def calculate_returns(df: pd.DataFrame, signal_col: str, hold_period: int) -> Dict:
    """固定持有回测口径。"""
    if df is None or df.empty or signal_col not in df.columns:
        return {
            'signal_count': 0,
            'trade_count': 0,
            'win_count': 0,
            'win_rate': np.nan,
            'avg_return': np.nan,
            'sum_return': 0.0,
            'sum_profit_return': 0.0,
            'sum_loss_return': 0.0,
            'max_return': np.nan,
            'min_return': np.nan,
        }

    trades = backtest_trades_fixed_hold(
        df=df,
        signal_col=signal_col,
        hold_period=hold_period,
        entry_lag=1,
        entry_price_col='open',
        exit_price_col='open',
        commission_rate=COMMISSION_RATE,
        stamp_tax_rate=STAMP_TAX_RATE,
    )
    return summarize_trades(trades, signal_count=len(trades))

def backtest_six_veins_single(filepath: str) -> Optional[pd.DataFrame]:
    """六脉神剑策略 - 单只股票回测"""
    df = load_stock_data(filepath)
    if df is None: return None
    df = calculate_six_veins(df)
    results = []
    for indicator in SIX_VEINS_INDICATORS:
        signal_col = f'{indicator}_signal'
        df[signal_col] = df[indicator] & ~df[indicator].shift(1, fill_value=False)
        for period in DEFAULT_HOLD_PERIODS:
            stats = calculate_returns(df, signal_col, period)
            stats.update({'strategy': indicator, 'hold_period': period})
            results.append(stats)
    signal_col = 'four_red_signal'
    df[signal_col] = (df['six_veins_count'] >= 4) & ~(df['six_veins_count'].shift(1, fill_value=0) >= 4)
    for period in DEFAULT_HOLD_PERIODS:
        stats = calculate_returns(df, signal_col, period)
        stats.update({'strategy': 'four_red_plus', 'hold_period': period})
        results.append(stats)
    return pd.DataFrame(results) if results else None

def backtest_buy_sell_single(filepath: str) -> Optional[pd.DataFrame]:
    """买卖点策略 - 单只股票回测"""
    df = load_stock_data(filepath)
    if df is None: return None
    df = calculate_buy_sell_points(df)
    results = []
    for signal_col in ['buy1', 'buy2']:
        df[f'{signal_col}_sig'] = df[signal_col] & ~df[signal_col].shift(1, fill_value=False)
        for period in DEFAULT_HOLD_PERIODS:
            stats = calculate_returns(df, f'{signal_col}_sig', period)
            stats.update({'strategy': signal_col, 'hold_period': period})
            results.append(stats)
    return pd.DataFrame(results) if results else None

def backtest_chan_single(filepath: str) -> Optional[pd.DataFrame]:
    """缠论买点策略 - 单只股票回测"""
    df = load_stock_data(filepath)
    if df is None: return None
    df = calculate_chan_theory(df)
    results = []
    chan_signals = ['chan_buy1', 'chan_buy2', 'chan_buy3', 'chan_strong_buy2', 'chan_like_buy2']
    for signal_col in chan_signals:
        df[f'{signal_col}_sig'] = df[signal_col] & ~df[signal_col].shift(1, fill_value=False)
        for period in DEFAULT_HOLD_PERIODS:
            stats = calculate_returns(df, f'{signal_col}_sig', period)
            stats.update({'strategy': signal_col, 'hold_period': period})
            results.append(stats)
    return pd.DataFrame(results) if results else None

def backtest_sell_optimize_single(filepath: str) -> Optional[pd.DataFrame]:
    """卖出点优化 - 单只股票回测"""
    df = load_stock_data(filepath)
    if df is None: return None
    df = calculate_all_signals(df)
    if 'six_veins_buy' not in df.columns: return None
    df['entry_signal'] = df['six_veins_buy'] & ~df['six_veins_buy'].shift(1, fill_value=False)
    results = []
    for hold_period in range(1, 21):
        stats = calculate_returns(df, 'entry_signal', hold_period)
        stats.update({'strategy': 'sell_opt', 'hold_period': hold_period})
        results.append(stats)
    return pd.DataFrame(results) if results else None

def run_backtest(strategy: str, stock_files: List[str]) -> pd.DataFrame:
    """运行指定策略的回测"""
    backtest_funcs = {
        'six_veins': backtest_six_veins_single,
        'buy_sell': backtest_buy_sell_single,
        'chan': backtest_chan_single,
        'sell_opt': backtest_sell_optimize_single,
    }
    results_list = []
    if strategy == 'all':
        for strat_key, func in backtest_funcs.items():
            log(f"开始回测策略: {strat_key}")
            res = run_backtest_on_all_stocks(stock_files, func)
            if res:
                df_res = pd.DataFrame(res)
                df_res['strategy_type'] = strat_key
                results_list.append(df_res)
        return pd.concat(results_list, ignore_index=True) if results_list else pd.DataFrame()
    else:
        if strategy not in backtest_funcs:
            log(f"无效的策略类型: {strategy}", level="ERROR")
            return pd.DataFrame()
        log(f"开始回测策略: {strategy}")
        res = run_backtest_on_all_stocks(stock_files, backtest_funcs[strategy])
        if not res: return pd.DataFrame()
        df_res = pd.DataFrame(res)
        df_res['strategy_type'] = strategy
        return df_res

def generate_markdown_report(results: pd.DataFrame, stock_count: int) -> str:
    """生成Markdown格式的回测报告"""
    report = ["# 单指标策略回测报告\n", f"- 测试股票数量: **{stock_count}**", f"- 测试时间: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**\n"]
    for strat_type, df in results.groupby('strategy_type'):
        report.append(f"## 策略类型: {strat_type}\n")
        for strategy, sdf in df.groupby('strategy'):
            avg_win_rate = sdf['win_rate'].mean()
            avg_return = sdf['avg_return'].mean()
            report.append(f"**策略 {strategy}:** 平均胜率 {avg_win_rate:.2f}%, 平均收益率 {avg_return:.2f}%")
        report.append("")
    return "\n".join(report)

def generate_json_data(results: pd.DataFrame, stock_count: int) -> Dict:
    """生成供Web前端使用的JSON数据"""
    data = {'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'stock_count': stock_count, 'strategies': {}}
    for strategy, sdf in results.groupby('strategy'):
        strat_data = {}
        best_idx = sdf['win_rate'].idxmax()
        best = sdf.loc[best_idx]
        strat_data['optimal_period_win'] = str(int(best['hold_period']))
        strat_data['win_rate'] = round(float(sdf['win_rate'].mean()), 2)
        strat_data['avg_return'] = round(float(sdf['avg_return'].mean()), 2)
        strat_data['trades'] = int(sdf['trade_count'].sum())
        data['strategies'][strategy] = strat_data
    return data

def main():
    parser = argparse.ArgumentParser(description='单指标策略回测系统')
    parser.add_argument('--strategy', type=str, default='all', choices=['all', 'six_veins', 'buy_sell', 'chan', 'sell_opt'], help='策略类型')
    args = parser.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'day')
    stock_files = get_all_stock_files(data_dir)
    if not stock_files:
        log(f"未找到股票数据文件: {data_dir}", level="ERROR")
        return

    results_df = run_backtest(args.strategy, stock_files)
    if results_df.empty:
        log("回测无有效结果。", level="WARNING")
        return

    os.makedirs(os.path.join(PROJECT_ROOT, 'report', 'total'), exist_ok=True)
    summary_csv_path = os.path.join(PROJECT_ROOT, 'report', 'total', 'single_strategy_summary.csv')
    results_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    log(f"回测结果汇总已保存: {summary_csv_path}")

    md_report = generate_markdown_report(results_df, len(stock_files))
    md_path = os.path.join(PROJECT_ROOT, 'report', 'total', 'single_strategy_report.md')
    with open(md_path, 'w', encoding='utf-8') as f: f.write(md_report)
    log(f"已保存Markdown报告: {md_path}")
    
    json_data = generate_json_data(results_df, len(stock_files))
    json_dir = os.path.join(PROJECT_ROOT, 'web', 'client', 'src', 'data')
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, 'backtest_single.json')
    with open(json_path, 'w', encoding='utf-8') as f: json.dump(json_data, f, ensure_ascii=False, indent=2)
    log(f"Web前端数据已保存: {json_path}")

if __name__ == "__main__":
    main()
