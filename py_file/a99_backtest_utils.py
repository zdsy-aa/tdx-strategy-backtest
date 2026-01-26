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
) -> List:
    """并行执行回测，支持返回 DataFrame 或 Dict。"""
    if not stock_files: return []
    
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    results = []
    if num_processes == 1:
        for f in stock_files:
            res = backtest_func(f)
            if res is not None:
                if isinstance(res, pd.DataFrame):
                    results.extend(res.to_dict(orient='records'))
                elif isinstance(res, list):
                    results.extend(res)
                elif isinstance(res, dict):
                    results.append(res)
    else:
        with multiprocessing.Pool(num_processes) as pool:
            for res in pool.imap_unordered(backtest_func, stock_files):
                if res is not None:
                    if isinstance(res, pd.DataFrame):
                        results.extend(res.to_dict(orient='records'))
                    elif isinstance(res, list):
                        results.extend(res)
                    elif isinstance(res, dict):
                        results.append(res)
    return results

def backtest_trades_fixed_hold(
    df: pd.DataFrame,
    signal_col: str,
    hold_period: int,
    entry_lag: int = 1,
    entry_price_col: str = 'open',
    exit_price_col: str = 'open',
    commission_rate: float = 0.00008,
    stamp_tax_rate: float = 0.0005
) -> List[Dict]:
    """
    执行固定持仓回测。
    
    参数:
        df: 股票数据 DataFrame
        signal_col: 信号列名
        hold_period: 持仓天数
        entry_lag: 信号确认后多少天成交（默认1=次日开盘）
        entry_price_col: 买入价格列名（默认'open'）
        exit_price_col: 卖出价格列名（默认'open'）
        commission_rate: 佣金费率
        stamp_tax_rate: 印花税率
    """
    if df is None or df.empty or signal_col not in df.columns:
        return []
    
    if entry_price_col not in df.columns or exit_price_col not in df.columns:
        return []
    
    trades = []
    # 获取所有信号为 True 的索引
    signal_indices = df[df[signal_col] == True].index.tolist()
    
    for idx in signal_indices:
        entry_idx = idx + entry_lag
        exit_idx = entry_idx + hold_period
        
        if exit_idx >= len(df): continue
        
        try:
            entry_p = float(df.at[entry_idx, entry_price_col])
            exit_p = float(df.at[exit_idx, exit_price_col])
        except:
            continue
        
        if entry_p <= 0 or exit_p <= 0 or pd.isna(entry_p) or pd.isna(exit_p): 
            continue
        
        buy_cost = entry_p * (1 + commission_rate)
        sell_rev = exit_p * (1 - commission_rate - stamp_tax_rate)
        net_ret = (sell_rev - buy_cost) / buy_cost
        
        trade = {
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'entry_price': entry_p,
            'exit_price': exit_p,
            'hold_days': hold_period,
            'net_return': net_ret,
            'profit': 1 if net_ret > 0 else 0
        }
        
        # 添加日期信息（如果存在）
        if 'date' in df.columns:
            trade['entry_date'] = df.at[entry_idx, 'date']
            trade['exit_date'] = df.at[exit_idx, 'date']
            
        trades.append(trade)
    return trades

def summarize_trades(trades: List[Dict], signal_count: int = None) -> Dict:
    """汇总交易结果。"""
    if not trades:
        return {
            'signal_count': signal_count if signal_count else 0,
            'trade_count': 0, 
            'win_count': 0,
            'win_rate': 0.0, 
            'avg_return': 0.0,
            'sum_return': 0.0
        }
    
    rets = [t['net_return'] * 100 for t in trades]
    win_count = sum(1 for t in trades if t['profit'] == 1)
    
    return {
        'signal_count': signal_count if signal_count else len(trades),
        'trade_count': len(trades),
        'win_count': win_count,
        'win_rate': round(win_count / len(trades) * 100, 2) if trades else 0.0,
        'avg_return': round(np.mean(rets), 2) if rets else 0.0,
        'sum_return': round(np.sum(rets), 2) if rets else 0.0
    }
