#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
102_买卖点单指标测试 (102_buy_sell_points_test.py)
================================================================================

测试内容:
    本脚本专门测试买卖点指标的效果，包括：
    1. 买点1 (吸筹信号) 的胜率和收益
    2. 买点2 (庄家上穿散户) 的胜率和收益
    3. 卖点1和卖点2的效果验证
    4. 不同持有周期的收益对比

买卖点指标说明:
    - 买点1: 吸筹指标上穿14，主力开始吸筹
    - 买点2: 庄家线上穿散户线且庄家<50，低位金叉
    - 卖点1: 庄家线下穿88，高位见顶
    - 卖点2: 散户线上穿庄家线，趋势反转

输出文件:
    - /report/total/buy_sell_points_report.md: 综合报告
    - /report/total/buy_sell_points_detail.csv: 详细数据

作者: TradeGuide System
版本: 2.0.0
更新日期: 2026-01-07
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import calculate_buy_sell_points


# ==============================================================================
# 配置常量
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report', 'total')

# 测试的持有周期列表
HOLD_PERIODS = [1, 2, 3, 5, 10, 15, 20, 30]


# ==============================================================================
# 回测核心函数
# ==============================================================================

def load_market_data(filename: str = 'sh000300.csv') -> pd.DataFrame:
    """加载市场数据"""
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"错误: 数据文件不存在 - {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"已加载数据: {filename}")
    print(f"时间范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"数据条数: {len(df)}")
    
    return df


def calculate_returns(df: pd.DataFrame, signal_col: str, hold_period: int) -> Dict:
    """计算指定信号的收益统计"""
    signals = df[df[signal_col] == True].copy()
    
    if len(signals) == 0:
        return {
            'signal_count': 0,
            'trade_count': 0,
            'win_rate': 0,
            'avg_return': 0,
            'max_return': 0,
            'min_return': 0,
            'total_return': 0,
            'std_return': 0
        }
    
    returns = []
    for idx in signals.index:
        if idx + hold_period < len(df):
            entry_price = df.loc[idx, 'close']
            exit_price = df.loc[idx + hold_period, 'close']
            ret = (exit_price - entry_price) / entry_price * 100
            returns.append(ret)
    
    if len(returns) == 0:
        return {
            'signal_count': len(signals),
            'trade_count': 0,
            'win_rate': 0,
            'avg_return': 0,
            'max_return': 0,
            'min_return': 0,
            'total_return': 0,
            'std_return': 0
        }
    
    returns = np.array(returns)
    win_count = np.sum(returns > 0)
    
    return {
        'signal_count': len(signals),
        'trade_count': len(returns),
        'win_rate': win_count / len(returns) * 100,
        'avg_return': np.mean(returns),
        'max_return': np.max(returns),
        'min_return': np.min(returns),
        'total_return': np.sum(returns),
        'std_return': np.std(returns)
    }


def test_buy_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    测试买入信号效果
    
    参数:
        df: 已计算买卖点指标的DataFrame
        
    返回:
        pd.DataFrame: 买入信号测试结果
    """
    results = []
    
    buy_signals = {
        'buy1': '买点1 (吸筹信号)',
        'buy2': '买点2 (庄家上穿散户)'
    }
    
    for signal_col, signal_name in buy_signals.items():
        for period in HOLD_PERIODS:
            stats = calculate_returns(df, signal_col, period)
            results.append({
                'signal': signal_name,
                'signal_code': signal_col,
                'hold_period': period,
                **stats
            })
    
    return pd.DataFrame(results)


def test_sell_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    测试卖出信号效果
    
    卖出信号的测试方式与买入相反：
    - 在信号出现时卖出（假设已持有）
    - 计算如果继续持有会产生的亏损
    
    参数:
        df: 已计算买卖点指标的DataFrame
        
    返回:
        pd.DataFrame: 卖出信号测试结果
    """
    results = []
    
    sell_signals = {
        'sell1': '卖点1 (庄家下穿88)',
        'sell2': '卖点2 (散户上穿庄家)'
    }
    
    for signal_col, signal_name in sell_signals.items():
        for period in HOLD_PERIODS:
            # 计算如果不卖出继续持有的收益（负值表示卖出正确）
            signals = df[df[signal_col] == True].copy()
            
            if len(signals) == 0:
                results.append({
                    'signal': signal_name,
                    'signal_code': signal_col,
                    'hold_period': period,
                    'signal_count': 0,
                    'trade_count': 0,
                    'correct_rate': 0,
                    'avg_avoided_loss': 0
                })
                continue
            
            avoided_losses = []
            for idx in signals.index:
                if idx + period < len(df):
                    entry_price = df.loc[idx, 'close']
                    exit_price = df.loc[idx + period, 'close']
                    ret = (exit_price - entry_price) / entry_price * 100
                    avoided_losses.append(ret)
            
            if len(avoided_losses) == 0:
                results.append({
                    'signal': signal_name,
                    'signal_code': signal_col,
                    'hold_period': period,
                    'signal_count': len(signals),
                    'trade_count': 0,
                    'correct_rate': 0,
                    'avg_avoided_loss': 0
                })
                continue
            
            avoided_losses = np.array(avoided_losses)
            correct_count = np.sum(avoided_losses < 0)  # 卖出后价格下跌，说明卖出正确
            
            results.append({
                'signal': signal_name,
                'signal_code': signal_col,
                'hold_period': period,
                'signal_count': len(signals),
                'trade_count': len(avoided_losses),
                'correct_rate': correct_count / len(avoided_losses) * 100,
                'avg_avoided_loss': np.mean(avoided_losses)
            })
    
    return pd.DataFrame(results)


def test_combined_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    测试买卖点组合策略
    
    策略: 买点2买入，卖点1或卖点2卖出
    
    参数:
        df: 已计算买卖点指标的DataFrame
        
    返回:
        pd.DataFrame: 组合策略测试结果
    """
    results = []
    
    # 找出所有买点2信号
    buy_signals = df[df['buy2'] == True].index.tolist()
    
    trades = []
    for buy_idx in buy_signals:
        # 从买入点开始寻找卖出信号
        for sell_idx in range(buy_idx + 1, min(buy_idx + 60, len(df))):  # 最多持有60天
            if df.loc[sell_idx, 'sell1'] or df.loc[sell_idx, 'sell2']:
                entry_price = df.loc[buy_idx, 'close']
                exit_price = df.loc[sell_idx, 'close']
                ret = (exit_price - entry_price) / entry_price * 100
                hold_days = sell_idx - buy_idx
                trades.append({
                    'entry_date': df.loc[buy_idx, 'date'],
                    'exit_date': df.loc[sell_idx, 'date'],
                    'hold_days': hold_days,
                    'return': ret
                })
                break
    
    if len(trades) == 0:
        return pd.DataFrame()
    
    trades_df = pd.DataFrame(trades)
    
    # 统计结果
    win_count = (trades_df['return'] > 0).sum()
    
    results.append({
        'strategy': '买点2买入 + 卖点信号卖出',
        'trade_count': len(trades_df),
        'win_rate': win_count / len(trades_df) * 100,
        'avg_return': trades_df['return'].mean(),
        'avg_hold_days': trades_df['hold_days'].mean(),
        'total_return': trades_df['return'].sum(),
        'max_return': trades_df['return'].max(),
        'min_return': trades_df['return'].min()
    })
    
    return pd.DataFrame(results), trades_df


def generate_report(
    buy_results: pd.DataFrame,
    sell_results: pd.DataFrame,
    combo_results: pd.DataFrame,
    trades_df: pd.DataFrame,
    df: pd.DataFrame
) -> str:
    """生成Markdown格式的测试报告"""
    report = []
    report.append("# 买卖点指标测试报告")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**测试数据**: 沪深300指数")
    report.append(f"**数据范围**: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
    report.append("")
    
    # 1. 买入信号测试
    report.append("## 一、买入信号测试")
    report.append("")
    
    for signal in buy_results['signal'].unique():
        signal_data = buy_results[buy_results['signal'] == signal]
        best_period = signal_data.loc[signal_data['win_rate'].idxmax()]
        
        report.append(f"### {signal}")
        report.append("")
        report.append("| 持有周期 | 信号次数 | 交易次数 | 胜率 | 平均收益 | 最大收益 | 最大亏损 |")
        report.append("|----------|----------|----------|------|----------|----------|----------|")
        for _, row in signal_data.iterrows():
            report.append(f"| {row['hold_period']}天 | {row['signal_count']} | {row['trade_count']} | {row['win_rate']:.2f}% | {row['avg_return']:.2f}% | {row['max_return']:.2f}% | {row['min_return']:.2f}% |")
        report.append("")
        report.append(f"**最优持有周期**: {best_period['hold_period']}天，胜率 {best_period['win_rate']:.2f}%")
        report.append("")
    
    # 2. 卖出信号测试
    report.append("## 二、卖出信号测试")
    report.append("")
    report.append("卖出信号的正确率表示：信号出现后，如果继续持有会产生亏损的概率。")
    report.append("")
    
    for signal in sell_results['signal'].unique():
        signal_data = sell_results[sell_results['signal'] == signal]
        
        report.append(f"### {signal}")
        report.append("")
        report.append("| 持有周期 | 信号次数 | 正确率 | 平均避免亏损 |")
        report.append("|----------|----------|--------|--------------|")
        for _, row in signal_data.iterrows():
            report.append(f"| {row['hold_period']}天 | {row['signal_count']} | {row['correct_rate']:.2f}% | {row['avg_avoided_loss']:.2f}% |")
        report.append("")
    
    # 3. 组合策略测试
    if not combo_results.empty:
        report.append("## 三、组合策略测试")
        report.append("")
        report.append("策略: 买点2买入，卖点1或卖点2卖出")
        report.append("")
        
        for _, row in combo_results.iterrows():
            report.append(f"- **交易次数**: {row['trade_count']}")
            report.append(f"- **胜率**: {row['win_rate']:.2f}%")
            report.append(f"- **平均收益**: {row['avg_return']:.2f}%")
            report.append(f"- **平均持有天数**: {row['avg_hold_days']:.1f}天")
            report.append(f"- **累计收益**: {row['total_return']:.2f}%")
        report.append("")
    
    # 4. 结论
    report.append("## 四、结论与建议")
    report.append("")
    
    # 找出买点2的最优周期
    buy2_data = buy_results[buy_results['signal_code'] == 'buy2']
    best_buy2 = buy2_data.loc[buy2_data['win_rate'].idxmax()]
    
    report.append("### 核心发现")
    report.append("")
    report.append(f"1. **买点2最优持有周期**: {best_buy2['hold_period']}天")
    report.append(f"2. **买点2预期胜率**: {best_buy2['win_rate']:.2f}%")
    report.append(f"3. **买点2平均收益**: {best_buy2['avg_return']:.2f}%")
    report.append("")
    report.append("### 使用建议")
    report.append("")
    report.append("1. **买点2** 是最重要的买入信号，建议在庄家线<50时介入")
    report.append("2. **卖点1** 适合作为止盈信号，高位出现时减仓")
    report.append("3. **卖点2** 适合作为止损信号，趋势反转时清仓")
    report.append("")
    
    return '\n'.join(report)


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    """主程序入口"""
    print("=" * 60)
    print("买卖点指标测试")
    print("=" * 60)
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 1. 加载数据
    df = load_market_data()
    if df.empty:
        print("无法加载数据，退出测试")
        return
    
    # 2. 计算买卖点指标
    print("\n计算买卖点指标...")
    df = calculate_buy_sell_points(df)
    
    # 3. 测试买入信号
    print("\n测试买入信号...")
    buy_results = test_buy_signals(df)
    
    # 4. 测试卖出信号
    print("\n测试卖出信号...")
    sell_results = test_sell_signals(df)
    
    # 5. 测试组合策略
    print("\n测试组合策略...")
    combo_results, trades_df = test_combined_strategy(df)
    
    # 6. 生成报告
    print("\n生成测试报告...")
    report_content = generate_report(buy_results, sell_results, combo_results, trades_df, df)
    
    report_path = os.path.join(REPORT_DIR, 'buy_sell_points_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"报告已保存: {report_path}")
    
    # 保存详细数据
    detail_path = os.path.join(REPORT_DIR, 'buy_sell_points_detail.csv')
    buy_results.to_csv(detail_path, index=False, encoding='utf-8')
    print(f"详细数据已保存: {detail_path}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
