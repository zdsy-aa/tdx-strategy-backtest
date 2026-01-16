#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
组合策略回测系统 (Combo Strategy Backtest)
================================================================================

功能描述:
    本脚本整合了原201-202两个回测脚本的功能，提供统一的组合策略回测能力。
    支持稳健型和激进型两种组合策略的全量股票回测。

整合的原脚本:
    - 201_steady_combo_test.py     : 稳健组合策略回测
    - 202_aggressive_combo_test.py : 激进组合策略回测

组合策略说明:
    
    【稳健组合】
    适合风险偏好较低的投资者，追求稳定收益。
    买入条件：六脉≥4红 + 买点2 + (缠论二买或三买)
    持仓周期：10, 20, 30天
    
    【激进组合】
    适合风险偏好较高的投资者，追求高收益。
    包含三种子策略：
    - 激进1：六脉≥5红 + 买点2
    - 激进2：六脉6红 + 摇钱树
    - 激进3：缠论一买 + 六脉≥4红
    持仓周期：5, 10, 15天

输出文件:
    - report/total/combo_strategy_report.md       : 综合回测报告（Markdown格式）
    - report/total/combo_strategy_summary.csv     : 回测结果汇总（CSV格式）
    - web/client/src/data/backtest_combo.json     : 回测数据（供Web前端使用）

使用方法:
    cd ~/tdx-strategy-backtest/py_file
    
    # 运行所有组合策略回测
    python3 combo_strategy_backtest.py
    
    # 运行指定组合策略回测
    python3 combo_strategy_backtest.py --strategy steady
    python3 combo_strategy_backtest.py --strategy aggressive

命令行参数:
    --strategy : 指定要运行的策略类型
                 可选值: all, steady, aggressive
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

# 稳健组合持仓周期
STEADY_HOLD_PERIODS = [10, 20, 30]

# 激进组合持仓周期
AGGRESSIVE_HOLD_PERIODS = [5, 10, 15]

# ==============================================================================
# 导入技术指标计算模块
# ==============================================================================

from indicators_99 import calculate_all_signals
from backtest_utils_99 import get_all_stock_files, aggregate_results


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
            '成交额': 'amount'
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
            - max_return: 最大收益（百分比）
            - min_return: 最小收益（百分比）
    """
    # 获取信号触发点
    signals = df[df[signal_col] == True].copy()
    
    if len(signals) == 0:
        return {
            'signal_count': 0,
            'trade_count': 0,
            'win_rate': np.nan,
            'avg_return': np.nan,
            'max_return': np.nan,
            'min_return': np.nan
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
            'avg_return': np.nan,
            'max_return': np.nan,
            'min_return': np.nan
        }
    
    returns = np.array(returns)
    win_rate = np.sum(returns > 0) / len(returns) * 100
    
    return {
        'signal_count': len(signals),
        'trade_count': len(returns),
        'win_rate': win_rate,
        'avg_return': np.mean(returns),
        'max_return': np.max(returns),
        'min_return': np.min(returns)
    }


# ==============================================================================
# 稳健组合策略回测
# ==============================================================================

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
    
    try:
        # 计算所有指标
        df = calculate_all_signals(df)
        
        # 稳健组合信号条件
        # 条件1：六脉神剑达到4红以上
        # 条件2：买点2信号（低位金叉）
        # 条件3：缠论二买或三买信号
        df['steady_signal'] = (
            (df['six_veins_count'] >= 4) & 
            df['buy2'] & 
            (df['chan_buy2'] | df['chan_buy3'])
        )
        
        results = []
        
        for period in STEADY_HOLD_PERIODS:
            stats = calculate_returns(df, 'steady_signal', period)
            if stats['trade_count'] > 0:
                results.append({
                    'strategy': '稳健组合',
                    'type': '组合策略',
                    'name': '六脉≥4红+买点2+缠论二/三买',
                    'hold_period': period,
                    **stats
                })
        
        return pd.DataFrame(results) if results else None
        
    except Exception as e:
        return None


# ==============================================================================
# 激进组合策略回测
# ==============================================================================

def backtest_aggressive_single(filepath: str) -> Optional[pd.DataFrame]:
    """
    激进组合策略 - 单只股票回测
    
    功能说明:
        激进组合包含三种子策略：
        - 激进1：六脉≥5红 + 买点2
        - 激进2：六脉6红 + 摇钱树
        - 激进3：缠论一买 + 六脉≥4红
        
        这些组合追求更高的收益，但风险也相对较高。
    
    参数:
        filepath (str): 股票CSV文件路径
    
    返回:
        pd.DataFrame: 回测结果，如果失败返回None
    """
    df = load_stock_data(filepath)
    if df is None:
        return None
    
    try:
        # 计算所有指标
        df = calculate_all_signals(df)
        
        # 定义激进组合策略
        combos = {
            '激进1 (六脉≥5红+买点2)': (df['six_veins_count'] >= 5) & df['buy2'],
            '激进2 (六脉6红+摇钱树)': (df['six_veins_count'] == 6) & df.get('money_tree', False),
            '激进3 (缠论一买+六脉≥4红)': df['chan_buy1'] & (df['six_veins_count'] >= 4)
        }
        
        results = []
        
        for combo_name, signal_series in combos.items():
            # 处理可能的布尔值（当money_tree不存在时）
            if isinstance(signal_series, bool):
                continue
                
            df[combo_name] = signal_series
            
            for period in AGGRESSIVE_HOLD_PERIODS:
                stats = calculate_returns(df, combo_name, period)
                if stats['trade_count'] > 0:
                    results.append({
                        'strategy': '激进组合',
                        'type': '组合策略',
                        'name': combo_name,
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
        strategy (str): 策略类型 ('steady' 或 'aggressive')
        stock_files (List[str]): 股票文件列表
    
    返回:
        pd.DataFrame: 汇总后的回测结果
    """
    # 选择回测函数
    backtest_funcs = {
        'steady': backtest_steady_single,
        'aggressive': backtest_aggressive_single
    }
    
    backtest_func = backtest_funcs.get(strategy)
    if backtest_func is None:
        print(f"未知策略类型: {strategy}")
        return pd.DataFrame()
    
    strategy_names = {
        'steady': '稳健组合',
        'aggressive': '激进组合'
    }
    
    print(f"\n开始回测: {strategy_names[strategy]}")
    print(f"处理 {len(stock_files)} 个股票文件...")
    
    # 多进程并行处理 (cpu核数-1)
    num_processes = max(1, cpu_count() - 1)
    with Pool(num_processes) as pool:
        all_results = pool.map(backtest_func, stock_files)
    
    # 过滤空结果
    all_results = [r for r in all_results if r is not None and not r.empty]
    
    if not all_results:
        print(f"  {strategy_names[strategy]}: 没有有效的回测结果")
        return pd.DataFrame()
    
    # 汇总结果
    combined = pd.concat(all_results, ignore_index=True)
    
    # 按策略、类型、名称、持仓周期分组汇总
    group_cols = ['strategy', 'type', 'name', 'hold_period']
    summary = combined.groupby(group_cols).agg({
        'signal_count': 'sum',
        'trade_count': 'sum',
        'win_rate': 'mean',
        'avg_return': 'mean',
        'max_return': 'max',
        'min_return': 'min'
    }).reset_index()
    
    print(f"  {strategy_names[strategy]}: 完成，共 {len(summary)} 条汇总记录")
    
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
    report.append("# 组合策略回测报告\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**测试股票数量**: {stock_count}\n\n")
    
    # 策略说明
    report.append("## 策略说明\n\n")
    report.append("### 稳健组合\n")
    report.append("- **买入条件**: 六脉≥4红 + 买点2 + (缠论二买或三买)\n")
    report.append("- **适合人群**: 风险偏好较低的投资者\n")
    report.append("- **持仓周期**: 10, 20, 30天\n\n")
    
    report.append("### 激进组合\n")
    report.append("- **激进1**: 六脉≥5红 + 买点2\n")
    report.append("- **激进2**: 六脉6红 + 摇钱树\n")
    report.append("- **激进3**: 缠论一买 + 六脉≥4红\n")
    report.append("- **适合人群**: 风险偏好较高的投资者\n")
    report.append("- **持仓周期**: 5, 10, 15天\n\n")
    
    # 按策略分组输出
    for strategy in results['strategy'].unique():
        strategy_df = results[results['strategy'] == strategy]
        
        report.append(f"## {strategy}回测结果\n\n")
        report.append("| 策略名称 | 持仓周期 | 信号数 | 交易数 | 胜率 | 平均收益 | 最大收益 | 最小收益 |\n")
        report.append("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        
        for _, row in strategy_df.sort_values('win_rate', ascending=False).iterrows():
            report.append(
                f"| {row['name']} | {row['hold_period']}天 | "
                f"{int(row['signal_count'])} | {int(row['trade_count'])} | "
                f"{row['win_rate']:.2f}% | {row['avg_return']:.2f}% | "
                f"{row['max_return']:.2f}% | {row['min_return']:.2f}% |\n"
            )
        
        report.append("\n")
    
    # 结论
    report.append("## 结论与建议\n\n")
    
    if not results.empty:
        best = results.loc[results['win_rate'].idxmax()]
        report.append(f"**最佳策略**: {best['name']} ({best['hold_period']}天持仓)\n")
        report.append(f"- 胜率: {best['win_rate']:.2f}%\n")
        report.append(f"- 平均收益: {best['avg_return']:.2f}%\n\n")
    
    report.append("**风险提示**: 历史回测结果不代表未来收益，请根据自身风险承受能力谨慎投资。\n")
    
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
        'strategies': {
            'steady': {
                'name': '稳健组合',
                'description': '六脉≥4红 + 买点2 + (缠论二买或三买)',
                'risk_level': '低',
                'results': []
            },
            'aggressive': {
                'name': '激进组合',
                'description': '多种高收益组合策略',
                'risk_level': '高',
                'results': []
            }
        },
        'comparison': []
    }
    
    # 填充策略结果
    for _, row in results.iterrows():
        strategy_key = 'steady' if row['strategy'] == '稳健组合' else 'aggressive'
        
        data['strategies'][strategy_key]['results'].append({
            'name': row['name'],
            'hold_period': int(row['hold_period']),
            'signal_count': int(row['signal_count']),
            'trade_count': int(row['trade_count']),
            'win_rate': round(row['win_rate'], 2),
            'avg_return': round(row['avg_return'], 2),
            'max_return': round(row['max_return'], 2),
            'min_return': round(row['min_return'], 2)
        })
    
    # 生成对比数据（按胜率排序）
    for _, row in results.sort_values('win_rate', ascending=False).iterrows():
        data['comparison'].append({
            'strategy': row['strategy'],
            'name': row['name'],
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
    主函数：执行组合策略回测
    
    支持通过命令行参数选择要运行的策略类型。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='组合策略回测系统')
    parser.add_argument(
        '--strategy',
        type=str,
        default='all',
        choices=['all', 'steady', 'aggressive'],
        help='要运行的策略类型 (默认: all)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='限制处理的股票数量，0表示不限制 (默认: 0)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("组合策略回测系统 v1.0")
    print("=" * 70)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"策略选择: {args.strategy}")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(WEB_DATA_DIR, exist_ok=True)
    
    # 获取所有股票文件
    stock_files = get_all_stock_files(DATA_DIR)
    
    # 限制股票数量（用于测试）
    if args.limit > 0:
        stock_files = stock_files[:args.limit]
        print(f"注意: 已限制处理 {args.limit} 个股票文件")
    
    print(f"发现 {len(stock_files)} 个股票文件")
    
    if not stock_files:
        print("错误: 未找到股票数据文件")
        return
    
    # 确定要运行的策略
    if args.strategy == 'all':
        strategies = ['steady', 'aggressive']
    else:
        strategies = [args.strategy]
    
    # 运行回测
    all_results = []
    for strategy in strategies:
        result = run_backtest(strategy, stock_files)
        if not result.empty:
            all_results.append(result)
    
    if not all_results:
        print("\n没有生成任何回测结果")
        return
    
    # 合并所有结果
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # 生成报告
    print("\n" + "-" * 70)
    print("生成报告...")
    
    # 保存CSV
    csv_path = os.path.join(REPORT_DIR, 'combo_strategy_summary.csv')
    combined_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存CSV汇总: {csv_path}")
    
    # 生成并保存Markdown报告
    md_report = generate_markdown_report(combined_results, len(stock_files))
    md_path = os.path.join(REPORT_DIR, 'combo_strategy_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"已保存Markdown报告: {md_path}")
    
    # 生成并保存JSON数据（供Web前端使用）
    json_data = generate_json_data(combined_results, len(stock_files))
    
    # 保存到report目录
    json_path = os.path.join(REPORT_DIR, 'backtest_combo.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"已保存JSON数据: {json_path}")
    
    # 保存到Web数据目录
    web_json_path = os.path.join(WEB_DATA_DIR, 'backtest_combo.json')
    with open(web_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"已保存JSON数据到Web目录: {web_json_path}")
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("回测完成！")
    print("=" * 70)
    print(f"测试股票数: {len(stock_files)}")
    print(f"策略数量: {len(strategies)}")
    print(f"汇总记录: {len(combined_results)}")
    
    # 打印最佳策略
    if not combined_results.empty:
        best = combined_results.loc[combined_results['win_rate'].idxmax()]
        print(f"\n最佳策略: {best['name']}")
        print(f"  持仓周期: {best['hold_period']}天")
        print(f"  胜率: {best['win_rate']:.2f}%")
        print(f"  平均收益: {best['avg_return']:.2f}%")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
