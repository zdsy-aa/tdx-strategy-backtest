#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
202_激进组合测试 (202_aggressive_combo_test.py)
================================================================================

测试内容:
    本脚本专门测试激进型组合策略的效果，包括：
    1. 六脉≥5红 + 买点2
    2. 六脉6红 + 摇钱树
    3. 缠论一买 + 六脉≥4红
    4. 5天偏移值组合信号测试

激进组合特点:
    - 信号条件相对宽松
    - 适合短线操作
    - 追求高收益
    - 风险相对较高

输出文件:
    - /report/total/aggressive_combo_report.md: 综合报告
    - /report/total/aggressive_combo_detail.csv: 详细数据

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
from indicators import calculate_all_signals


# ==============================================================================
# 配置常量
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report', 'total')

HOLD_PERIODS = [1, 2, 3, 5, 10, 15, 20, 30]
OFFSET_DAYS = 5


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
            'std_return': 0,
            'sharpe_ratio': 0
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
            'std_return': 0,
            'sharpe_ratio': 0
        }
    
    returns = np.array(returns)
    win_count = np.sum(returns > 0)
    std = np.std(returns) if len(returns) > 1 else 0
    sharpe = np.mean(returns) / std if std > 0 else 0
    
    return {
        'signal_count': len(signals),
        'trade_count': len(returns),
        'win_rate': win_count / len(returns) * 100,
        'avg_return': np.mean(returns),
        'max_return': np.max(returns),
        'min_return': np.min(returns),
        'total_return': np.sum(returns),
        'std_return': std,
        'sharpe_ratio': sharpe
    }


def create_offset_signal(df: pd.DataFrame, signal_cols: List[str], offset_days: int = 5) -> pd.Series:
    """创建偏移值组合信号"""
    result = pd.Series(False, index=df.index)
    
    for i in range(offset_days, len(df)):
        all_signals_found = True
        for col in signal_cols:
            if not df[col].iloc[i-offset_days:i+1].any():
                all_signals_found = False
                break
        
        if all_signals_found:
            prev_all_found = True
            for col in signal_cols:
                if not df[col].iloc[i-offset_days-1:i].any():
                    prev_all_found = False
                    break
            
            if not prev_all_found:
                result.iloc[i] = True
    
    return result


def test_aggressive_combos(df: pd.DataFrame) -> pd.DataFrame:
    """
    测试激进组合策略
    
    激进组合定义:
    1. 六脉≥5红 + 买点2
    2. 六脉6红 + 摇钱树
    3. 缠论一买 + 六脉≥4红
    
    参数:
        df: 已计算所有信号的DataFrame
        
    返回:
        pd.DataFrame: 测试结果
    """
    results = []
    
    # 组合1: 六脉≥5红 + 买点2
    df['aggressive1_same'] = (df['six_veins_count'] >= 5) & df['buy2']
    df['six_ge5'] = df['six_veins_count'] >= 5
    df['aggressive1_offset'] = create_offset_signal(df, ['six_ge5', 'buy2'], OFFSET_DAYS)
    
    # 组合2: 六脉6红 + 摇钱树
    df['aggressive2_same'] = (df['six_veins_count'] == 6) & df['money_tree']
    df['six_eq6'] = df['six_veins_count'] == 6
    df['aggressive2_offset'] = create_offset_signal(df, ['six_eq6', 'money_tree'], OFFSET_DAYS)
    
    # 组合3: 缠论一买 + 六脉≥4红
    df['aggressive3_same'] = df['chan_buy1'] & (df['six_veins_count'] >= 4)
    df['six_ge4'] = df['six_veins_count'] >= 4
    df['aggressive3_offset'] = create_offset_signal(df, ['chan_buy1', 'six_ge4'], OFFSET_DAYS)
    
    combos = [
        ('激进1: 六脉≥5红+买点2 (同日)', 'aggressive1_same', 0),
        ('激进1: 六脉≥5红+买点2 (5日偏移)', 'aggressive1_offset', 5),
        ('激进2: 六脉6红+摇钱树 (同日)', 'aggressive2_same', 0),
        ('激进2: 六脉6红+摇钱树 (5日偏移)', 'aggressive2_offset', 5),
        ('激进3: 缠论一买+六脉≥4红 (同日)', 'aggressive3_same', 0),
        ('激进3: 缠论一买+六脉≥4红 (5日偏移)', 'aggressive3_offset', 5),
    ]
    
    for combo_name, signal_col, offset in combos:
        for period in HOLD_PERIODS:
            stats = calculate_returns(df, signal_col, period)
            results.append({
                'combo_type': combo_name,
                'offset_days': offset,
                'hold_period': period,
                **stats
            })
    
    return pd.DataFrame(results)


def test_short_term_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    测试短线表现
    
    激进策略更适合短线操作，重点测试1-5天的表现
    
    参数:
        df: 已计算所有信号的DataFrame
        
    返回:
        pd.DataFrame: 短线测试结果
    """
    results = []
    short_periods = [1, 2, 3, 4, 5]
    
    # 最激进组合: 六脉6红
    df['most_aggressive'] = df['six_veins_count'] == 6
    
    for period in short_periods:
        stats = calculate_returns(df, 'most_aggressive', period)
        results.append({
            'strategy': '六脉6红',
            'hold_period': period,
            **stats
        })
    
    # 次激进组合: 六脉≥5红 + 买点2
    df['second_aggressive'] = (df['six_veins_count'] >= 5) & df['buy2']
    
    for period in short_periods:
        stats = calculate_returns(df, 'second_aggressive', period)
        results.append({
            'strategy': '六脉≥5红+买点2',
            'hold_period': period,
            **stats
        })
    
    return pd.DataFrame(results)


def generate_report(
    combo_results: pd.DataFrame,
    short_results: pd.DataFrame,
    df: pd.DataFrame
) -> str:
    """生成Markdown格式的测试报告"""
    report = []
    report.append("# 激进组合策略测试报告")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**测试数据**: 沪深300指数")
    report.append(f"**数据范围**: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
    report.append("")
    
    # 1. 策略定义
    report.append("## 一、策略定义")
    report.append("")
    report.append("### 激进组合条件")
    report.append("")
    report.append("| 组合 | 条件 | 风险等级 |")
    report.append("|------|------|----------|")
    report.append("| 激进1 | 六脉≥5红 + 买点2 | 中高 |")
    report.append("| 激进2 | 六脉6红 + 摇钱树 | 高 |")
    report.append("| 激进3 | 缠论一买 + 六脉≥4红 | 中 |")
    report.append("")
    
    # 2. 组合测试结果
    report.append("## 二、组合测试结果")
    report.append("")
    
    for combo_type in combo_results['combo_type'].unique():
        combo_data = combo_results[combo_results['combo_type'] == combo_type]
        
        report.append(f"### {combo_type}")
        report.append("")
        report.append("| 持有周期 | 信号次数 | 交易次数 | 胜率 | 平均收益 | 最大收益 | 最大亏损 |")
        report.append("|----------|----------|----------|------|----------|----------|----------|")
        for _, row in combo_data.iterrows():
            report.append(f"| {row['hold_period']}天 | {row['signal_count']} | {row['trade_count']} | {row['win_rate']:.2f}% | {row['avg_return']:.2f}% | {row['max_return']:.2f}% | {row['min_return']:.2f}% |")
        report.append("")
        
        if combo_data['trade_count'].sum() > 0:
            best = combo_data.loc[combo_data['win_rate'].idxmax()]
            report.append(f"**最优持有周期**: {best['hold_period']}天，胜率 {best['win_rate']:.2f}%")
        report.append("")
    
    # 3. 短线表现
    report.append("## 三、短线表现分析")
    report.append("")
    report.append("激进策略更适合短线操作，以下是1-5天持有期的表现：")
    report.append("")
    
    if not short_results.empty:
        for strategy in short_results['strategy'].unique():
            strategy_data = short_results[short_results['strategy'] == strategy]
            
            report.append(f"### {strategy}")
            report.append("")
            report.append("| 持有天数 | 信号次数 | 胜率 | 平均收益 |")
            report.append("|----------|----------|------|----------|")
            for _, row in strategy_data.iterrows():
                report.append(f"| {row['hold_period']}天 | {row['signal_count']} | {row['win_rate']:.2f}% | {row['avg_return']:.2f}% |")
            report.append("")
    
    # 4. 结论
    report.append("## 四、结论与建议")
    report.append("")
    
    # 找出整体最优
    valid_results = combo_results[combo_results['trade_count'] > 0]
    if not valid_results.empty:
        best_overall = valid_results.loc[valid_results['win_rate'].idxmax()]
        
        report.append("### 核心发现")
        report.append("")
        report.append(f"1. **最优组合类型**: {best_overall['combo_type']}")
        report.append(f"2. **最优持有周期**: {best_overall['hold_period']}天")
        report.append(f"3. **预期胜率**: {best_overall['win_rate']:.2f}%")
        report.append(f"4. **平均收益**: {best_overall['avg_return']:.2f}%")
    
    report.append("")
    report.append("### 使用建议")
    report.append("")
    report.append("1. **适用人群**: 风险偏好型投资者，追求高收益")
    report.append("2. **仓位建议**: 单次建仓不超过20%，严格止损")
    report.append("3. **止损设置**: 建议设置3%止损位")
    report.append("4. **持有周期**: 建议1-5天短线操作")
    report.append("5. **风险提示**: 激进策略波动大，需要较强的心理承受能力")
    report.append("")
    
    return '\n'.join(report)


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    """主程序入口"""
    print("=" * 60)
    print("激进组合策略测试")
    print("=" * 60)
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 1. 加载数据
    df = load_market_data()
    if df.empty:
        print("无法加载数据，退出测试")
        return
    
    # 2. 计算所有指标
    print("\n计算所有指标...")
    df = calculate_all_signals(df)
    
    # 3. 测试激进组合
    print("\n测试激进组合...")
    combo_results = test_aggressive_combos(df)
    
    # 4. 测试短线表现
    print("\n测试短线表现...")
    short_results = test_short_term_performance(df)
    
    # 5. 生成报告
    print("\n生成测试报告...")
    report_content = generate_report(combo_results, short_results, df)
    
    report_path = os.path.join(REPORT_DIR, 'aggressive_combo_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"报告已保存: {report_path}")
    
    # 保存详细数据
    detail_path = os.path.join(REPORT_DIR, 'aggressive_combo_detail.csv')
    combo_results.to_csv(detail_path, index=False, encoding='utf-8')
    print(f"详细数据已保存: {detail_path}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
