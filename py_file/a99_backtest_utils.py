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

作者: TradeGuide System
版本: 2.0.0
更新日期: 2026-01-15
================================================================================
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Callable, Dict, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
try:
    from a99_logger import log
except ImportError:
    # 兼容性处理
    def log(msg, level="INFO"): print(f"[{level}] {msg}")


# ==============================================================================
# 交易级回测核心（口径A：确认日触发，次日开盘成交；单票单仓；计入交易成本）
# ==============================================================================

def _safe_float(x) -> float:
    try:
        if pd.isna(x):
            return float('nan')
        return float(x)
    except Exception:
        return float('nan')


def calc_net_return(
    entry_price: float,
    exit_price: float,
    commission_rate: float = 0.00008,
    stamp_tax_rate: float = 0.0005,
) -> float:
    """计算单笔交易净收益率（不含仓位杠杆）。

    成本模型（与你给定口径一致）：
      - 佣金：万0.8（0.00008），买卖双边收取
      - 印花税：0.05%（0.0005），仅卖出收取

    返回：净收益率（小数，例如 0.0123 表示 +1.23%）。
    """
    entry_price = _safe_float(entry_price)
    exit_price = _safe_float(exit_price)
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
        return float('nan')

    buy_rate = commission_rate
    sell_rate = commission_rate + stamp_tax_rate

    # 使用“按成交额计费”的近似：
    # 买入实际成本 = entry_price * (1 + buy_rate)
    # 卖出实际到手 = exit_price * (1 - sell_rate)
    net = (exit_price * (1 - sell_rate)) / (entry_price * (1 + buy_rate)) - 1
    return float(net)


def backtest_trades_fixed_hold(
    df: pd.DataFrame,
    signal_col: str,
    hold_period: int,
    entry_lag: int = 1,
    entry_price_col: str = 'open',
    exit_price_col: str = 'open',
    commission_rate: float = 0.00008,
    stamp_tax_rate: float = 0.0005,
) -> pd.DataFrame:
    """逐笔回测：固定持有N天。

    交易口径：
      - 信号在 t 日收盘后确认（口径A），在 t+1 日开盘买入（entry_lag=1，entry_price_col='open'）。
      - 固定持有 hold_period 天后，在对应出场日开盘卖出（exit_price_col='open'）。
      - 单票单仓：持仓期间忽略后续信号。

    返回：每笔交易一行的 DataFrame（空则返回空DF）。
    """
    if df is None or df.empty or signal_col not in df.columns:
        return pd.DataFrame()
    if hold_period <= 0:
        raise ValueError('hold_period must be positive')

    # 统一索引为 0..n-1，避免外部传入非连续索引导致定位异常
    work = df.reset_index(drop=True)

    records = []
    i = 0
    n = len(work)

    while i < n:
        sig = bool(work.at[i, signal_col]) if i < n else False
        if not sig:
            i += 1
            continue

        entry_i = i + entry_lag
        if entry_i >= n:
            break

        entry_price = work.at[entry_i, entry_price_col] if entry_price_col in work.columns else np.nan
        entry_price = _safe_float(entry_price)
        if not np.isfinite(entry_price) or entry_price <= 0:
            i += 1
            continue

        exit_i = entry_i + hold_period
        if exit_i >= n:
            break

        exit_price = work.at[exit_i, exit_price_col] if exit_price_col in work.columns else np.nan
        exit_price = _safe_float(exit_price)
        if not np.isfinite(exit_price) or exit_price <= 0:
            i += 1
            continue

        net_ret = calc_net_return(
            entry_price=entry_price,
            exit_price=exit_price,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
        )

        records.append({
            'signal_index': i,
            'entry_index': entry_i,
            'exit_index': exit_i,
            'signal_date': work.at[i, 'date'] if 'date' in work.columns else pd.NaT,
            'entry_date': work.at[entry_i, 'date'] if 'date' in work.columns else pd.NaT,
            'exit_date': work.at[exit_i, 'date'] if 'date' in work.columns else pd.NaT,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'net_return': net_ret,  # 小数
        })

        # 单票单仓：出场次日才允许再入场
        i = exit_i + 1

    return pd.DataFrame.from_records(records)


def summarize_trades(trades: pd.DataFrame, signal_count: int) -> Dict:
    """将逐笔交易明细汇总为统计字段（加权口径需要的原子量）。"""
    if trades is None or trades.empty:
        return {
            'signal_count': int(signal_count),
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

    r = trades['net_return'].astype(float)
    r = r[np.isfinite(r)]
    if r.empty:
        return {
            'signal_count': int(signal_count),
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

    trade_count = int(len(r))
    win_mask = r > 0
    win_count = int(win_mask.sum())

    sum_return = float(r.sum())
    sum_profit = float(r[win_mask].sum()) if win_count > 0 else 0.0
    sum_loss = float(r[~win_mask].sum()) if win_count < trade_count else 0.0

    return {
        'signal_count': int(signal_count),
        'trade_count': trade_count,
        'win_count': win_count,
        'win_rate': (win_count / trade_count * 100.0) if trade_count > 0 else np.nan,
        'avg_return': (float(r.mean()) * 100.0) if trade_count > 0 else np.nan,  # 百分比
        'sum_return': sum_return,
        'sum_profit_return': sum_profit,
        'sum_loss_return': sum_loss,
        'max_return': float(r.max()) * 100.0,
        'min_return': float(r.min()) * 100.0,
    }


# ==============================================================================
# 文件操作函数
# ==============================================================================

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
                    mtime = os.path.getmtime(full_path)
                    last_mtime = status_data.get(full_path, 0)
                    if mtime > last_mtime:
                        stock_files.append(full_path)
                else:
                    stock_files.append(full_path)
    
    stock_files.sort()
    return stock_files

def update_backtest_status(files: List[str], status_file: str):
    """更新回测状态文件"""
    status_data = {}
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status_data = json.load(f)
        except:
            pass
            
    for f in files:
        status_data[f] = os.path.getmtime(f)
        
    try:
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
    except Exception as e:
        log(f"保存状态文件失败: {e}", level="ERROR")


# ==============================================================================
# 并行回测函数
# ==============================================================================

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
        >>> 
        >>> files = get_all_stock_files('/data/day')
        >>> results = run_backtest_on_all_stocks(files, my_backtest)
        开始全量回测，使用 8 个进程处理 5789 个文件...
        >>> print(f"有效结果: {len(results)} 条")
    
    注意事项:
        - 回测函数必须是可序列化的（不能使用lambda或闭包）
        - 回测函数内部应处理异常，避免中断整个进程池
        - 大量文件时建议监控内存使用情况
    """
    # 确定进程数
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    
    # 显示进度信息
    if show_progress:
        print(f"开始全量回测，使用 {num_processes} 个进程处理 {len(stock_files)} 个文件...")
    
    # 在开始前检查内存
    
    # 创建进程池并执行回测
    with Pool(num_processes) as pool:
        # 在并行任务中，我们可以通过包装函数来定期检查内存
        results = pool.map(backtest_func, stock_files)
    
    # 完成后再次检查
    
    # 过滤掉空结果
    valid_results = [r for r in results if r is not None]
    
    if show_progress:
        print(f"回测完成，有效结果: {len(valid_results)}/{len(stock_files)}")
    
    return valid_results


# ==============================================================================
# 结果汇总函数
# ==============================================================================

def aggregate_results(
    results: List[pd.DataFrame], 
    group_by_cols: List[str],
    sum_cols: Optional[List[str]] = None,
    mean_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    汇总多只股票的回测结果
    
    功能说明:
        将多只股票的回测结果DataFrame合并，并按指定列分组汇总。
        支持自定义哪些列求和、哪些列求平均。
    
    参数:
        results (List[pd.DataFrame]): 回测结果DataFrame列表
                                      每个DataFrame包含单只股票的回测结果
        
        group_by_cols (List[str]): 分组列名列表
                                   例如: ['strategy', 'type', 'name', 'hold_period']
        
        sum_cols (List[str], optional): 需要求和的列名列表
                                        默认为 ['signal_count', 'trade_count']
        
        mean_cols (List[str], optional): 需要求平均的列名列表
                                         默认为 ['win_rate', 'avg_return']
    
    返回:
        pd.DataFrame: 汇总后的结果DataFrame
                      如果输入为空，返回空DataFrame
    
    使用示例:
        >>> results = [df1, df2, df3]  # 多只股票的回测结果
        >>> summary = aggregate_results(
        ...     results,
        ...     group_by_cols=['strategy', 'hold_period'],
        ...     sum_cols=['signal_count', 'trade_count'],
        ...     mean_cols=['win_rate', 'avg_return']
        ... )
        >>> print(summary)
           strategy  hold_period  signal_count  trade_count  win_rate  avg_return
        0  六脉神剑          5         12345         10234     62.35        3.21
        1  六脉神剑         10         12345         10234     65.12        4.56
    
    注意事项:
        - 空列表输入返回空DataFrame
        - 自动检测数值列并应用默认聚合方式
    """
    # 处理空输入
    if not results:
        return pd.DataFrame()
    
    # 过滤掉空DataFrame
    valid_results = [r for r in results if r is not None and not r.empty]
    if not valid_results:
        return pd.DataFrame()
    
    # 合并所有结果
    combined_df = pd.concat(valid_results, ignore_index=True)
    
    # 设置默认的求和列
    if sum_cols is None:
        sum_cols = ['signal_count', 'trade_count']
    
    # 设置默认的求平均列
    if mean_cols is None:
        mean_cols = ['win_rate', 'avg_return']
    
    # 构建聚合字典
    agg_dict = {}
    
    # 添加求和列
    for col in sum_cols:
        if col in combined_df.columns:
            agg_dict[col] = 'sum'
    
    # 添加求平均列
    for col in mean_cols:
        if col in combined_df.columns:
            agg_dict[col] = 'mean'
    
    # 自动检测其他数值列
    for col in combined_df.columns:
        if col not in group_by_cols and col not in agg_dict:
            if combined_df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                # 默认对其他数值列求平均
                agg_dict[col] = 'mean'
    
    # 执行分组汇总
    if agg_dict:
        summary = combined_df.groupby(group_by_cols).agg(agg_dict).reset_index()
    else:
        summary = combined_df.drop_duplicates(subset=group_by_cols)
    
    return summary


def aggregate_trade_results(
    combined_df: pd.DataFrame,
    group_by_cols: List[str],
) -> pd.DataFrame:
    """对“交易级统计字段”进行加权汇总（推荐使用）。

    预期 combined_df 至少包含：
      - signal_count, trade_count, win_count, sum_return（小数）
    可选包含：
      - sum_profit_return, sum_loss_return, max_return, min_return

    输出会生成：win_rate（%）, avg_return（%）等派生列。
    """
    if combined_df is None or combined_df.empty:
        return pd.DataFrame()

    sum_cols = [
        'signal_count',
        'trade_count',
        'win_count',
        'sum_return',
        'sum_profit_return',
        'sum_loss_return',
    ]
    existing_sum_cols = [c for c in sum_cols if c in combined_df.columns]

    # max/min 对收益分布有意义，但这里按整体最大/最小聚合（不是均值）
    agg_dict = {c: 'sum' for c in existing_sum_cols}
    if 'max_return' in combined_df.columns:
        agg_dict['max_return'] = 'max'
    if 'min_return' in combined_df.columns:
        agg_dict['min_return'] = 'min'

    out = combined_df.groupby(group_by_cols).agg(agg_dict).reset_index()

    # 派生指标
    if 'trade_count' in out.columns:
        tc = out['trade_count'].replace(0, np.nan)
        if 'win_count' in out.columns:
            out['win_rate'] = (out['win_count'] / tc) * 100.0
        if 'sum_return' in out.columns:
            out['avg_return'] = (out['sum_return'] / tc) * 100.0

    return out


# ==============================================================================
# 辅助函数
# ==============================================================================

def calculate_win_rate(returns: List[float]) -> float:
    """
    计算胜率
    
    功能说明:
        计算收益列表中正收益的比例。
    
    参数:
        returns (List[float]): 收益率列表（百分比形式）
    
    返回:
        float: 胜率（百分比形式，0-100）
               如果列表为空，返回NaN
    
    使用示例:
        >>> returns = [5.2, -2.1, 3.5, -1.0, 8.3]
        >>> win_rate = calculate_win_rate(returns)
        >>> print(f"胜率: {win_rate:.2f}%")
        胜率: 60.00%
    """
    if not returns:
        return np.nan
    
    returns_array = np.array(returns)
    win_count = np.sum(returns_array > 0)
    
    return win_count / len(returns_array) * 100


def calculate_sharpe_ratio(
    returns: List[float], 
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    计算夏普比率
    
    功能说明:
        夏普比率衡量每承担一单位风险所获得的超额收益。
        值越高表示风险调整后的收益越好。
    
    参数:
        returns (List[float]): 收益率列表（百分比形式）
        risk_free_rate (float): 无风险利率（年化，百分比形式）
                                默认为0
        periods_per_year (int): 每年的交易周期数
                                默认为252（交易日数）
    
    返回:
        float: 夏普比率
               如果标准差为0或列表为空，返回NaN
    
    使用示例:
        >>> returns = [1.2, -0.5, 2.1, 0.8, -0.3, 1.5]
        >>> sharpe = calculate_sharpe_ratio(returns, risk_free_rate=3.0)
        >>> print(f"夏普比率: {sharpe:.2f}")
    """
    if not returns or len(returns) < 2:
        return np.nan
    
    returns_array = np.array(returns)
    
    # 计算平均收益和标准差
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array, ddof=1)
    
    if std_return == 0:
        return np.nan
    
    # 年化处理
    annual_return = mean_return * periods_per_year
    annual_std = std_return * np.sqrt(periods_per_year)
    
    # 计算夏普比率
    sharpe = (annual_return - risk_free_rate) / annual_std
    
    return sharpe


def calculate_max_drawdown(cumulative_returns: List[float]) -> float:
    """
    计算最大回撤
    
    功能说明:
        最大回撤是指从历史最高点到最低点的最大跌幅。
        用于衡量策略的最大潜在损失。
    
    参数:
        cumulative_returns (List[float]): 累计收益率列表（百分比形式）
    
    返回:
        float: 最大回撤（百分比形式，负值）
               如果列表为空，返回NaN
    
    使用示例:
        >>> cum_returns = [0, 5, 8, 6, 10, 7, 12, 9, 15]
        >>> max_dd = calculate_max_drawdown(cum_returns)
        >>> print(f"最大回撤: {max_dd:.2f}%")
    """
    if not cumulative_returns:
        return np.nan
    
    cum_array = np.array(cumulative_returns)
    
    # 计算累计最大值
    running_max = np.maximum.accumulate(cum_array)
    
    # 计算回撤
    drawdowns = cum_array - running_max
    
    # 返回最大回撤
    return np.min(drawdowns)


# ==============================================================================
# 模块测试
# ==============================================================================

if __name__ == "__main__":
    # 模块自测试
    print("=" * 60)
    print("回测工具函数库 - 模块测试")
    print("=" * 60)
    
    # 测试胜率计算
    test_returns = [5.2, -2.1, 3.5, -1.0, 8.3, -0.5, 2.1]
    win_rate = calculate_win_rate(test_returns)
    print(f"\n测试胜率计算:")
    print(f"  收益列表: {test_returns}")
    print(f"  胜率: {win_rate:.2f}%")
    
    # 测试夏普比率计算
    sharpe = calculate_sharpe_ratio(test_returns)
    print(f"\n测试夏普比率计算:")
    print(f"  夏普比率: {sharpe:.2f}")
    
    # 测试最大回撤计算
    cum_returns = [0, 5, 8, 6, 10, 7, 12, 9, 15]
    max_dd = calculate_max_drawdown(cum_returns)
    print(f"\n测试最大回撤计算:")
    print(f"  累计收益: {cum_returns}")
    print(f"  最大回撤: {max_dd:.2f}%")
    
    print("\n" + "=" * 60)
    print("模块测试完成")
    print("=" * 60)
