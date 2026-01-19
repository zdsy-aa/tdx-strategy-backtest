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
    - 结果汇总统计

模块函数:
    - get_all_stock_files(): 获取指定目录下所有股票CSV文件
    - run_backtest_on_all_stocks(): 多进程并行执行回测函数
    - aggregate_results(): 汇总多只股票的回测结果

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
