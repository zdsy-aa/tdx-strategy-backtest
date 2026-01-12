#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测工具函数库
提供全量股票遍历、胜率汇总等通用功能
"""

import os
import pandas as pd
import numpy as np
from typing import List, Callable, Dict
from multiprocessing import Pool, cpu_count
from functools import partial

def get_all_stock_files(data_dir: str) -> List[str]:
    """获取所有股票数据文件路径"""
    stock_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                stock_files.append(os.path.join(root, file))
    return stock_files

def run_backtest_on_all_stocks(
    stock_files: List[str], 
    backtest_func: Callable, 
    num_processes: int = None
) -> List[Dict]:
    """
    在所有股票上运行回测函数
    
    参数:
        stock_files: 股票文件路径列表
        backtest_func: 回测函数，接收文件路径，返回结果字典或DataFrame
        num_processes: 进程数，默认为CPU核心数
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"开始全量回测，使用 {num_processes} 个进程处理 {len(stock_files)} 个文件...")
    
    with Pool(num_processes) as pool:
        results = pool.map(backtest_func, stock_files)
    
    # 过滤掉空结果
    return [r for r in results if r is not None]

def aggregate_results(results: List[pd.DataFrame], group_by_cols: List[str]) -> pd.DataFrame:
    """
    汇总回测结果并计算平均值
    """
    if not results:
        return pd.DataFrame()
    
    # 合并所有结果
    combined_df = pd.concat(results, ignore_index=True)
    
    # 按指定列分组并计算平均值
    # 注意：signal_count 和 trade_count 应该求和，win_rate 和 avg_return 应该求平均
    agg_dict = {
        'signal_count': 'sum',
        'trade_count': 'sum',
        'win_rate': 'mean',
        'avg_return': 'mean'
    }
    
    # 检查是否存在其他需要汇总的列
    for col in combined_df.columns:
        if col not in group_by_cols and col not in agg_dict:
            if combined_df[col].dtype in [np.float64, np.int64]:
                agg_dict[col] = 'mean'
    
    summary = combined_df.groupby(group_by_cols).agg(agg_dict).reset_index()
    return summary
