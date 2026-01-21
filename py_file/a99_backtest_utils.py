#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
回测工具函数库 (Backtest Utilities)
================================================================================

功能描述:
    本模块提供回测系统所需的通用工具函数，包括：
    - 股票文件遍历
    - 多进程并行回测
    - 固定持仓回测执行
    - 交易统计汇总

模块函数:
    - get_all_stock_files(): 获取指定目录下所有股票CSV文件
    - run_backtest_on_all_stocks(): 多进程并行执行回测函数
    - backtest_trades_fixed_hold(): 执行固定持仓周期的回测
    - summarize_trades(): 汇总交易统计结果

使用示例:
    from a99_backtest_utils import get_all_stock_files, run_backtest_on_all_stocks
    
    # 获取所有股票文件
    stock_files = get_all_stock_files('/path/to/data')
    
    # 定义回测函数
    def my_backtest(filepath):
        # ... 回测逻辑 ...
        return result_dict
    
    # 运行回测
    results = run_backtest_on_all_stocks(stock_files, my_backtest)
"""

try:
    from a99_logger import log
except ImportError:
    def log(msg, level="INFO"): print(f"[{level}] {msg}")

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Optional
import multiprocessing

def get_all_stock_files(data_dir: str, incremental: bool = False, status_file: str = None) -> List[str]:
    """
    获取指定目录下所有股票数据文件的完整路径
    
    参数:
        data_dir (str): 数据根目录
        incremental (bool): 是否启用增量模式
        status_file (str): 状态记录文件路径
    """
    stock_files = []
    if not os.path.exists(data_dir):
        log(f"警告: 数据目录不存在: {data_dir}")
        return stock_files
    
    # 加载状态
    status_data = {}
    if incremental and status_file and os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status_data = json.load(f)
        except Exception as e:
            log(f"加载状态文件失败: {e}", level="ERROR")

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                
                if incremental:
                    # 增量逻辑：检查文件修改时间或最后一行日期
                    # 这里使用文件修改时间作为快速判断，或者您可以根据需要读取CSV最后一行
                    last_mod = os.path.getmtime(full_path)
                    prev_mod = status_data.get(file)
                    if prev_mod and last_mod <= prev_mod:
                        continue
                    status_data[file] = last_mod
                stock_files.append(full_path)
    # 增量模式下更新状态文件
    if incremental and status_file:
        try:
            with open(status_file, 'w') as f:
                json.dump(status_data, f)
        except Exception as e:
            log(f"保存状态文件失败: {e}", level="ERROR")
    return stock_files

def run_backtest_on_all_stocks(
    stock_files: List[str], 
    backtest_func: Callable, 
    num_processes: Optional[int] = None,
    show_progress: bool = True
) -> List[Dict]:
    """
    在所有股票上并行运行回测函数
    
    功能说明:
        使用多进程池并行处理多只股票的回测，充分利用多核CPU提升效率。
        自动过滤掉返回None的无效结果。
    
    参数:
        stock_files (List[str]): 股票文件路径列表
                                 由 get_all_stock_files() 函数获取
        
        backtest_func (Callable): 回测函数
                                  函数签名: func(filepath: str) -> Optional[Dict]
                                  接收单个股票文件路径，返回回测结果字典或None
        
        num_processes (int, optional): 并行进程数
                                       默认为None，自动使用CPU核心数
                                       建议设置为 cpu_count() - 1 以保留系统资源
        
        show_progress (bool): 是否显示进度信息
                              默认为True
    
    返回:
        List[Dict]: 所有有效回测结果的列表
                    无效结果（None）会被自动过滤
    
    使用示例:
        >>> def my_backtest(filepath):
        ...     df = pd.read_csv(filepath)
        ...     # ... 回测逻辑 ...
        ...     return {'win_rate': 0.65, 'avg_return': 2.5}
        >>> stock_files = get_all_stock_files(data_dir)
        >>> results = run_backtest_on_all_stocks(stock_files, my_backtest)
    """
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    results: List[Dict] = []
    if not stock_files:
        return results
    if num_processes == 1:
        # 单进程顺序执行（调试或CPU仅1核时）
        for filepath in stock_files:
            res = backtest_func(filepath)
            if res is not None:
                if isinstance(res, pd.DataFrame):
                    # DataFrame结果集则转为dict列表
                    results.extend(res.to_dict(orient='records'))
                elif isinstance(res, list):
                    results.extend(res)
                elif isinstance(res, dict):
                    results.append(res)
    else:
        # 多进程并行执行
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
    执行固定持仓周期的回测，返回每笔交易的详细信息。
    
    交易口径（统一标准）：
      - 信号确认：t日收盘后确认
      - 成交时间：t+1日开盘（entry_lag=1）
      - 持有周期：hold_period个交易日
      - 卖出时间：t+1+hold_period日开盘
      - 成本计算：
        * 买入成本 = 成交价 × (1 + 佣金率)
        * 卖出收入 = 成交价 × (1 - 佣金率 - 印花税率)
        * 收益率 = (卖出收入 - 买入成本) / 买入成本
    
    参数:
        df (pd.DataFrame): 股票数据，必须包含date、signal_col、entry_price_col、exit_price_col列
        signal_col (str): 信号列名（True表示买入信号）
        hold_period (int): 持仓天数（交易日）
        entry_lag (int): 信号确认后多少天成交（默认1=次日开盘）
        entry_price_col (str): 买入价格列名（默认'open'）
        exit_price_col (str): 卖出价格列名（默认'open'）
        commission_rate (float): 佣金费率（双边，默认0.00008=万0.8）
        stamp_tax_rate (float): 印花税率（仅卖出，默认0.0005=0.05%）
    
    返回:
        List[Dict]: 每笔交易的详细信息，包括：
            - entry_idx: 买入行索引
            - exit_idx: 卖出行索引
            - entry_date: 买入日期
            - exit_date: 卖出日期
            - entry_price: 买入价格
            - exit_price: 卖出价格
            - hold_days: 实际持仓天数
            - gross_return: 毛收益率（不含成本）
            - net_return: 净收益率（含成本）
            - profit: 是否盈利（1=盈利，0=亏损）
    
    使用示例:
        >>> df = pd.read_csv('stock.csv')
        >>> df['signal'] = df['indicator'] & ~df['indicator'].shift(1, fill_value=False)
        >>> trades = backtest_trades_fixed_hold(
        ...     df=df,
        ...     signal_col='signal',
        ...     hold_period=5,
        ...     entry_lag=1,
        ...     commission_rate=0.00008,
        ...     stamp_tax_rate=0.0005
        ... )
        >>> print(f"总交易数: {len(trades)}")
        >>> print(f"胜率: {sum(t['profit'] for t in trades) / len(trades):.2%}")
    """
    if df is None or df.empty or signal_col not in df.columns:
        return []
    
    if entry_price_col not in df.columns or exit_price_col not in df.columns:
        return []
    
    trades = []
    in_position = False
    entry_idx = None
    
    # 遍历所有行，寻找信号
    for i in range(len(df)):
        # 检查是否有买入信号（当前行为True，前一行为False或不存在）
        if not in_position and df.at[i, signal_col]:
            # 确认买入时间（entry_lag天后）
            entry_idx = i + entry_lag
            # 如果买入时间超出数据范围，跳过
            if entry_idx >= len(df):
                break
            in_position = True
        
        # 如果已持仓，检查是否到达卖出时间
        if in_position and entry_idx is not None:
            exit_idx = entry_idx + hold_period
            # 如果卖出时间超出数据范围，跳过
            if exit_idx >= len(df):
                break
            
            # 获取买入和卖出价格
            entry_price = df.at[entry_idx, entry_price_col]
            exit_price = df.at[exit_idx, exit_price_col]
            
            # 检查价格有效性
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0 or exit_price <= 0:
                in_position = False
                entry_idx = None
                continue
            
            # 计算成本
            # 买入成本 = 成交价 × (1 + 佣金率)
            buy_cost = entry_price * (1 + commission_rate)
            
            # 卖出收入 = 成交价 × (1 - 佣金率 - 印花税率)
            sell_revenue = exit_price * (1 - commission_rate - stamp_tax_rate)
            
            # 毛收益率（不含成本）
            gross_return = (exit_price - entry_price) / entry_price
            
            # 净收益率（含成本）
            net_return = (sell_revenue - buy_cost) / buy_cost
            
            # 实际持仓天数
            hold_days = exit_idx - entry_idx
            
            # 记录交易
            trade = {
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'entry_date': df.at[entry_idx, 'date'] if 'date' in df.columns else None,
                'exit_date': df.at[exit_idx, 'date'] if 'date' in df.columns else None,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'hold_days': hold_days,
                'gross_return': gross_return,
                'net_return': net_return,
                'profit': 1 if net_return > 0 else 0,
            }
            trades.append(trade)
            
            # 重置持仓状态
            in_position = False
            entry_idx = None
    
    return trades

def summarize_trades(trades: List[Dict], signal_count: int = None) -> Dict:
    """
    汇总交易统计结果，计算胜率、平均收益等关键指标。
    
    参数:
        trades (List[Dict]): backtest_trades_fixed_hold()返回的交易列表
        signal_count (int): 信号总数（用于计算信号命中率）
                           如果为None，则使用len(trades)
    
    返回:
        Dict: 包含以下统计指标的字典：
            - signal_count: 信号总数
            - trade_count: 实际交易数
            - win_count: 盈利交易数
            - win_rate: 胜率（%）
            - avg_return: 平均收益率（%）
            - sum_return: 总收益率（%）
            - sum_profit_return: 盈利交易的总收益率（%）
            - sum_loss_return: 亏损交易的总收益率（%）
            - max_return: 最大单笔收益率（%）
            - min_return: 最小单笔收益率（%）
    
    使用示例:
        >>> trades = backtest_trades_fixed_hold(df, 'signal', 5)
        >>> stats = summarize_trades(trades, signal_count=100)
        >>> print(f"胜率: {stats['win_rate']:.2f}%")
        >>> print(f"平均收益: {stats['avg_return']:.2f}%")
    """
    if not trades:
        return {
            'signal_count': signal_count if signal_count is not None else 0,
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
    
    # 如果signal_count未指定，使用交易数
    if signal_count is None:
        signal_count = len(trades)
    
    # 提取收益率（转换为百分比）
    returns = [t['net_return'] * 100 for t in trades]
    
    # 计算统计指标
    trade_count = len(trades)
    win_count = sum(1 for t in trades if t['profit'] == 1)
    win_rate = (win_count / trade_count * 100) if trade_count > 0 else np.nan
    
    avg_return = np.mean(returns) if returns else np.nan
    sum_return = np.sum(returns) if returns else 0.0
    
    # 分别统计盈利和亏损
    profit_returns = [r for t, r in zip(trades, returns) if t['profit'] == 1]
    loss_returns = [r for t, r in zip(trades, returns) if t['profit'] == 0]
    
    sum_profit_return = np.sum(profit_returns) if profit_returns else 0.0
    sum_loss_return = np.sum(loss_returns) if loss_returns else 0.0
    
    max_return = np.max(returns) if returns else np.nan
    min_return = np.min(returns) if returns else np.nan
    
    return {
        'signal_count': signal_count,
        'trade_count': trade_count,
        'win_count': win_count,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'sum_return': sum_return,
        'sum_profit_return': sum_profit_return,
        'sum_loss_return': sum_loss_return,
        'max_return': max_return,
        'min_return': min_return,
    }

def aggregate_results(results: List[Dict]) -> pd.DataFrame:
    """
    将回测结果字典列表汇总为DataFrame，并计算附加统计列（如胜率、平均收益等）。
    
    参数:
        results (List[Dict]): 回测结果列表，每个元素为单个股票的结果字典
    
    返回:
        pd.DataFrame: 汇总后的DataFrame，包括附加统计列
    """
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    # 示例：添加胜率统计列（如有需要）
    if 'win_count' in df.columns and 'trade_count' in df.columns:
        df['win_rate'] = (df['win_count'] / df['trade_count'] * 100).round(2)
    return df
