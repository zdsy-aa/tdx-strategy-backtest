#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a99_backtest_utils.py
功能描述: 回测工具函数库
使用方法: 被其他脚本导入使用
依赖库: pandas, numpy
安装命令: pip install pandas numpy
================================================================================
"""

import os
import json
import logging
import multiprocessing
from typing import List, Dict, Callable, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("backtest_utils")

def get_all_stock_files(data_dir: str) -> List[str]:
    """获取目录下所有股票 CSV 文件路径。"""
    stock_files = []
    if not os.path.exists(data_dir):
        logger.warning(f"目录不存在: {data_dir}")
        return []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                stock_files.append(os.path.join(root, file))
    return stock_files

def run_backtest_on_all_stocks(
    stock_files: List[str], 
    backtest_func: Callable, 
    num_processes: Optional[int] = None
) -> List[Dict]:
    """并行执行回测。"""
    if not stock_files: return []
    
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    results = []
    if num_processes == 1:
        for f in stock_files:
            res = backtest_func(f)
            if res is not None and not res.empty: results.append(res)
    else:
        with multiprocessing.Pool(num_processes) as pool:
            for res in pool.imap_unordered(backtest_func, stock_files):
                if res is not None and not res.empty: results.append(res)
    return results

def backtest_trades_fixed_hold(
    df: pd.DataFrame,
    signal_col: str,
    hold_period: int,
    commission_rate: float = 0.00008,
    stamp_tax_rate: float = 0.0005
) -> List[Dict]:
    """执行固定持仓回测。"""
    if df is None or df.empty or signal_col not in df.columns:
        return []
    
    trades = []
    # 简单实现：信号确认后次日开盘买入，持有 N 天后开盘卖出
    signals = df[df[signal_col] == True].index
    
    for idx in signals:
        entry_idx = idx + 1
        exit_idx = entry_idx + hold_period
        
        if exit_idx >= len(df): continue
        
        entry_p = df.at[entry_idx, 'open']
        exit_p = df.at[exit_idx, 'open']
        
        if entry_p <= 0 or exit_p <= 0: continue
        
        buy_cost = entry_p * (1 + commission_rate)
        sell_rev = exit_p * (1 - commission_rate - stamp_tax_rate)
        net_ret = (sell_rev - buy_cost) / buy_cost
        
        trades.append({
            'entry_date': df.at[entry_idx, 'date'],
            'exit_date': df.at[exit_idx, 'date'],
            'net_return': net_ret,
            'profit': 1 if net_ret > 0 else 0
        })
    return trades

def summarize_trades(trades: List[Dict]) -> Dict:
    """汇总交易结果。"""
    if not trades:
        return {'trade_count': 0, 'win_rate': 0, 'avg_return': 0}
    
    rets = [t['net_return'] * 100 for t in trades]
    win_count = sum(1 for t in trades if t['profit'] == 1)
    
    return {
        'trade_count': len(trades),
        'win_count': win_count,
        'win_rate': round(win_count / len(trades) * 100, 2),
        'avg_return': round(np.mean(rets), 2),
        'sum_return': round(np.sum(rets), 2)
    }
