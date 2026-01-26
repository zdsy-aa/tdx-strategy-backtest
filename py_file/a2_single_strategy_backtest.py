#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a2_single_strategy_backtest.py
功能描述: 单指标策略回测系统
使用方法: python3 a2_single_strategy_backtest.py --strategy all
依赖库: pandas, numpy, a99_indicators, a99_backtest_utils
安装命令: pip install pandas numpy
================================================================================
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Optional

# 尝试导入日志模块
try:
    from a99_logger import log
except ImportError:
    def log(msg, level="INFO"):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}")

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

# 定义六脉神剑指标列
SIX_VEINS_INDICATORS = ['macd_red', 'kdj_red', 'rsi_red', 'lwr_red', 'bbi_red', 'mtm_red']

def load_stock_data(filepath: str) -> Optional[pd.DataFrame]:
    """加载单只股票的CSV数据并处理编码问题。"""
    try:
        # 尝试多种编码加载数据
        df = None
        for enc in ['utf-8', 'gbk', 'utf-8-sig']:
            try:
                df = pd.read_csv(filepath, encoding=enc)
                break
            except (UnicodeDecodeError, Exception):
                continue
        
        if df is None:
            log(f"无法读取文件(编码错误): {filepath}", level="ERROR")
            return None
            
        # 列名映射：将中文列名转换为英文列名
        column_map = {
            '日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'
        }
        df.rename(columns={c: column_map.get(c, c) for c in df.columns}, inplace=True)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        # 数据量检查
        if len(df) < 30: # 降低门槛以便测试，生产环境可调回100
            return None
            
        # 排序并reset索引
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        log(f"数据加载异常: {filepath}, 错误: {e}", level="ERROR")
        return None

def calculate_returns(df: pd.DataFrame, signal_col: str, hold_period: int) -> Dict:
    """固定持有回测口径。"""
    if df is None or df.empty or signal_col not in df.columns:
        return {
            'signal_count': 0, 'trade_count': 0, 'win_count': 0,
            'win_rate': 0.0, 'avg_return': 0.0, 'sum_return': 0.0
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
    """六脉神剑策略回测"""
    df = load_stock_data(filepath)
    if df is None: return None
    df = calculate_six_veins(df)
    results = []
    # 1. 测试单个指标红柱
    for indicator in SIX_VEINS_INDICATORS:
        # 信号触发：当前红且前一周期不为红
        sig_col = f'{indicator}_sig'
        df[sig_col] = df[indicator] & ~df[indicator].shift(1, fill_value=False)
        for period in DEFAULT_HOLD_PERIODS:
            stats = calculate_returns(df, sig_col, period)
            stats.update({'strategy': indicator, 'hold_period': period})
            results.append(stats)
    # 2. 测试4红以上共振
    df['four_red_sig'] = (df['six_veins_count'] >= 4) & ~(df['six_veins_count'].shift(1, fill_value=0) >= 4)
    for period in DEFAULT_HOLD_PERIODS:
        stats = calculate_returns(df, 'four_red_sig', period)
        stats.update({'strategy': 'four_red_plus', 'hold_period': period})
        results.append(stats)
    return pd.DataFrame(results) if results else None

def backtest_buy_sell_single(filepath: str) -> Optional[pd.DataFrame]:
    """买卖点策略回测"""
    df = load_stock_data(filepath)
    if df is None: return None
    df = calculate_buy_sell_points(df)
    results = []
    for signal in ['buy1', 'buy2']:
        sig_col = f'{signal}_sig'
        df[sig_col] = df[signal] & ~df[signal].shift(1, fill_value=False)
        for period in DEFAULT_HOLD_PERIODS:
            stats = calculate_returns(df, sig_col, period)
            stats.update({'strategy': signal, 'hold_period': period})
            results.append(stats)
    return pd.DataFrame(results) if results else None

def backtest_chan_single(filepath: str) -> Optional[pd.DataFrame]:
    """缠论策略回测"""
    df = load_stock_data(filepath)
    if df is None: return None
    df = calculate_chan_theory(df)
    results = []
    for signal in ['chan_buy1']:
        sig_col = f'{signal}_sig'
        df[sig_col] = df[signal] & ~df[signal].shift(1, fill_value=False)
        for period in DEFAULT_HOLD_PERIODS:
            stats = calculate_returns(df, sig_col, period)
            stats.update({'strategy': signal, 'hold_period': period})
            results.append(stats)
    return pd.DataFrame(results) if results else None

def run_backtest(strategy: str, stock_files: List[str]) -> pd.DataFrame:
    """运行回测控制器"""
    funcs = {
        'six_veins': backtest_six_veins_single,
        'buy_sell': backtest_buy_sell_single,
        'chan': backtest_chan_single,
    }
    
    all_results = []
    strats_to_run = funcs.keys() if strategy == 'all' else [strategy]
    
    for s_name in strats_to_run:
        if s_name not in funcs: continue
        log(f"开始回测策略: {s_name}")
        res = run_backtest_on_all_stocks(stock_files, funcs[s_name])
        if res:
            df_res = pd.DataFrame(res)
            df_res['strategy_type'] = s_name
            all_results.append(df_res)
            
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description='单指标策略回测系统')
    parser.add_argument('--strategy', type=str, default='all', help='策略类型')
    args = parser.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'day')
    stock_files = get_all_stock_files(data_dir)
    if not stock_files:
        log(f"未找到数据文件: {data_dir}", level="ERROR")
        return

    results_df = run_backtest(args.strategy, stock_files)
    if results_df.empty:
        log("回测无有效结果。", level="WARNING")
        return

    # 保存结果
    out_dir = os.path.join(PROJECT_ROOT, 'report', 'total')
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(os.path.join(out_dir, 'single_strategy_summary.csv'), index=False, encoding='utf-8-sig')
    
    # 保存前端数据
    json_dir = os.path.join(PROJECT_ROOT, 'web', 'client', 'src', 'data')
    os.makedirs(json_dir, exist_ok=True)
    with open(os.path.join(json_dir, 'backtest_single.json'), 'w', encoding='utf-8') as f:
        json.dump({'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'count': len(stock_files)}, f)
    
    log("回测任务全部完成。")

if __name__ == "__main__":
    main()
