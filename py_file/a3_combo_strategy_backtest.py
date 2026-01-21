#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
组合策略回测系统 (Combo Strategy Backtest)
================================================================================

功能描述:
    本脚本提供组合策略（稳健和激进方案）的全市场回测能力。
    - 稳健组合：六脉≥4红 + 买点2 + (缠论二买或三买)
    - 激进组合：包含三种激进子策略的组合测试

主要功能:
    1. 稳健组合策略回测
    2. 激进组合策略回测（包括激进1、激进2、激进3）
    3. 输出汇总报告和供前端使用的数据

输出文件:
    - report/total/combo_strategy_report.md        : 组合策略回测报告（Markdown格式）
    - report/total/combo_strategy_summary.csv      : 回测结果汇总（CSV格式）
    - web/client/src/data/backtest_combo.json      : 回测数据（供Web前端使用）
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
from datetime import datetime
from typing import List, Dict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a99_indicators import calculate_all_signals
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

def load_stock_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    加载单只股票的CSV数据
    
    功能说明:
        读取CSV文件并按日期排序，确保包含必要的列。
    
    参数:
        filepath (str): CSV文件路径
    
    返回:
        pd.DataFrame: 格式化后的股票数据，若失败返回None
    """
    try:
        df = pd.read_csv(filepath)
        # 列名标准化
        df.rename(columns=lambda x: x.strip().lower(), inplace=True)
        # 日期列转换
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif '日期' in df.columns:
            df['date'] = pd.to_datetime(df['日期'])
        else:
            log(f"文件缺少日期列: {filepath}", level="ERROR")
            return None
        # 缺失列处理
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = 0.0
        # 数据量检查
        if len(df) < 100:
            return None
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        log(f"加载股票数据失败: {filepath}, 错误: {e}", level="ERROR")
        return None

def calculate_returns(df: pd.DataFrame, signal_col: str, hold_period: int) -> Dict:
    """
    固定持有回测统计：买入信号持有 hold_period 天后卖出。
    返回每笔交易的收益率，用于进一步汇总统计。
    """
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
    # 使用实际交易数作为signal_count，避免信号密集时被稀释
    stats = summarize_trades(trades, signal_count=len(trades))
    return stats

def backtest_steady_single(filepath: str) -> Optional[pd.DataFrame]:
    """
    稳健组合策略 - 单只股票回测
    
    功能说明:
        稳健组合买入条件：六脉≥4红 + 买点2 + (缠论二买或三买)
        该组合要求多个条件同时满足，信号较少但质量较高。
    
    参数:
        filepath (str): 股票CSV文件路径
    
    返回:
        pd.DataFrame: 回测结果，如果失败返回None
    """
    df = load_stock_data(filepath)
    if df is None:
        return None
    df = calculate_all_signals(df)
    if df.empty or 'combo_steady' not in df.columns:
        return None
    # 信号：稳健组合触发点（从False变True的点）
    df['steady_sig'] = df['combo_steady'] & ~df['combo_steady'].shift(1, fill_value=False)
    results = []
    for period in DEFAULT_HOLD_PERIODS:
        stats = calculate_returns(df, 'steady_sig', period)
        stats.update({
            'strategy': 'steady',
            'hold_period': period
        })
        results.append(stats)
    return pd.DataFrame(results) if results else None

def backtest_aggressive_single(filepath: str) -> Optional[pd.DataFrame]:
    """
    激进组合策略 - 单只股票回测
    
    功能说明:
        激进组合包含三种子策略：
        - 激进1：六脉≥5红 + 买点2
        - 激进2：六脉6红 + 摇钱树
        - 激进3：六脉6红 + 缠论任意买点
        将上述条件的信号合并为一个综合策略。
    
    参数:
        filepath (str): 股票CSV文件路径
    
    返回:
        pd.DataFrame: 回测结果，如果失败返回None
    """
    df = load_stock_data(filepath)
    if df is None:
        return None
    df = calculate_all_signals(df)
    if df.empty:
        return None
    # 子策略信号定义
    df['aggr1'] = ((df['six_veins_count'] >= 5) & df['buy2'])
    df['aggr2'] = (df['six_veins_count'] == 6) & df['money_tree']
    df['aggr3'] = (df['six_veins_count'] == 6) & df['chan_any_buy']
    # 综合激进信号：子策略1或2或3触发
    df['aggressive_sig'] = (df['aggr1'] | df['aggr2'] | df['aggr3']) & ~(df['aggr1'] | df['aggr2'] | df['aggr3']).shift(1, fill_value=False)
    results = []
    for period in DEFAULT_HOLD_PERIODS:
        stats = calculate_returns(df, 'aggressive_sig', period)
        stats.update({
            'strategy': 'aggressive',
            'hold_period': period
        })
        results.append(stats)
    return pd.DataFrame(results) if results else None

def run_backtest(strategy: str, stock_files: List[str]) -> pd.DataFrame:
    """
    运行指定组合策略的回测
    
    参数:
        strategy (str): 'steady' 或 'aggressive'
        stock_files (List[str]): 股票文件路径列表
    
    返回:
        pd.DataFrame: 回测结果汇总
    """
    backtest_funcs = {
        'steady': backtest_steady_single,
        'aggressive': backtest_aggressive_single
    }
    results_list = []
    if strategy == 'all':
        for strat, func in backtest_funcs.items():
            log(f"开始回测组合策略: {strat}")
            res = run_backtest_on_all_stocks(stock_files, func)
            if res:
                df_res = pd.concat(res, ignore_index=True)
                df_res['strategy'] = strat
                results_list.append(df_res)
        return pd.concat(results_list, ignore_index=True) if results_list else pd.DataFrame()
    else:
        if strategy not in backtest_funcs:
            log(f"无效的策略参数: {strategy}", level="ERROR")
            return pd.DataFrame()
        log(f"开始回测组合策略: {strategy}")
        res = run_backtest_on_all_stocks(stock_files, backtest_funcs[strategy])
        return pd.concat(res, ignore_index=True) if res else pd.DataFrame()

def generate_markdown_report(results: pd.DataFrame, stock_count: int) -> str:
    """
    生成Markdown格式的组合策略回测报告
    """
    report = []
    report.append("# 组合策略回测报告\n")
    report.append(f"- 测试股票数量: **{stock_count}**")
    report.append(f"- 测试时间: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**\n")
    if results.empty:
        report.append("*(无有效回测结果)*")
        return "\n".join(report)
    for strat, df in results.groupby('strategy'):
        avg_win_rate = df['win_rate'].mean()
        avg_return = df['avg_return'].mean()
        report.append(f"## 策略: {strat}")
        report.append(f"- 平均胜率: **{avg_win_rate:.2f}%**")
        report.append(f"- 平均收益率: **{avg_return:.2f}%**")
        # 可根据需要添加更多详细统计
        report.append("")  # 空行
    return "\n".join(report)

def generate_json_data(results: pd.DataFrame, stock_count: int) -> Dict:
    """
    生成组合策略回测的JSON数据 (供前端)
    """
    data = {
        'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock_count': stock_count,
        'strategies': {}
    }
    for strat, df in results.groupby('strategy'):
        strat_data = {}
        if df.empty:
            continue
        best = df.loc[df['win_rate'].idxmax()]
        strat_data['optimal_period_win'] = str(int(best['hold_period']))
        strat_data['win_rate'] = round(float(df['win_rate'].mean()), 2)
        strat_data['avg_return'] = round(float(df['avg_return'].mean()), 2)
        strat_data['trades'] = int(df['trade_count'].sum())
        data['strategies'][strat] = strat_data
    return data

def main():
    parser = argparse.ArgumentParser(description='组合策略回测系统')
    parser.add_argument(
        '--strategy',
        type=str,
        default='all',
        choices=['all', 'steady', 'aggressive'],
        help='要运行的组合策略类型 (all=全部)'
    )
    args = parser.parse_args()

    stock_files = get_all_stock_files(os.path.join(os.path.dirname(PROJECT_ROOT), 'data', 'day'))
    if not stock_files:
        log("未找到股票数据文件。", level="ERROR")
        return

    results_df = run_backtest(args.strategy, stock_files)
    if results_df.empty:
        log("回测没有产生有效结果。", level="WARNING")
        return

    # 保存结果到CSV
    summary_csv = os.path.join(PROJECT_ROOT, 'report', 'total', 'combo_strategy_summary.csv')
    results_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    log(f"组合策略回测结果汇总已保存: {summary_csv}")

    # 生成Markdown报告并保存
    md_report = generate_markdown_report(results_df, stock_count=len(stock_files))
    report_md_path = os.path.join(PROJECT_ROOT, 'report', 'total', 'combo_strategy_report.md')
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    log(f"组合策略Markdown报告已保存: {report_md_path}")

    # 生成JSON数据并保存
    json_data = generate_json_data(results_df, stock_count=len(stock_files))
    json_path = os.path.join(PROJECT_ROOT, 'web', 'client', 'src', 'data', 'backtest_combo.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    log(f"Web前端组合数据已保存: {json_path}")

if __name__ == "__main__":
    main()

print("a3_combo_strategy_backtest.py 脚本执行完毕")
