#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
201_稳健组合测试 (201_steady_combo_test.py)
================================================================================

测试内容:
    本脚本专门测试稳健型组合策略的效果，包括：
    1. 六脉≥4红 + 买点2 + 缠论二买/三买
    2. 5天偏移值组合信号测试
    3. 不同市场环境下的表现

稳健组合特点:
    - 多重信号确认，降低假信号
    - 适合中长线持有
    - 风险相对较低
    - 追求稳定收益而非暴利

输出文件:
    - /report/total/steady_combo_report.md: 综合报告
    - /report/total/steady_combo_detail.csv: 详细数据

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
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import calculate_all_signals


# ==============================================================================
# 配置常量
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report', 'total')

# 测试的持有周期列表
HOLD_PERIODS = [1, 2, 3, 5, 10, 15, 20, 30]

# 偏移值天数 (组合信号在N天内出现即视为有效)
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
    """
    创建偏移值组合信号
    
    在offset_days天内，如果所有指定信号都出现过，则视为组合信号触发。
    
    参数:
        df: DataFrame
        signal_cols: 信号列名列表
        offset_days: 偏移天数
        
    返回:
        pd.Series: 组合信号序列
    """
    result = pd.Series(False, index=df.index)
    
    for i in range(offset_days, len(df)):
        # 检查过去offset_days天内是否所有信号都出现过
        all_signals_found = True
        for col in signal_cols:
            if not df[col].iloc[i-offset_days:i+1].any():
                all_signals_found = False
                break
        
        if all_signals_found:
            # 检查是否是新触发（前一天不满足条件）
            prev_all_found = True
            for col in signal_cols:
                if not df[col].iloc[i-offset_days-1:i].any():
                    prev_all_found = False
                    break
            
            if not prev_all_found:
                result.iloc[i] = True
    
    return result


def test_steady_combo(df: pd.DataFrame) -> pd.DataFrame:
    """
    测试稳健组合策略
    
    稳健组合定义:
    - 六脉≥4红 + 买点2 + 缠论二买或三买
    
    参数:
        df: 已计算所有信号的DataFrame
        
    返回:
        pd.DataFrame: 测试结果
    """
    results = []
    
    # 1. 原始稳健组合 (同一天出现)
    df['steady_same_day'] = (
        (df['six_veins_count'] >= 4) & 
        df['buy2'] & 
        (df['chan_buy2'] | df['chan_buy3'])
    )
    
    for period in HOLD_PERIODS:
        stats = calculate_returns(df, 'steady_same_day', period)
        results.append({
            'combo_type': '稳健组合 (同日)',
            'offset_days': 0,
            'hold_period': period,
            **stats
        })
    
    # 2. 5天偏移值稳健组合
    # 六脉≥4红
    df['six_ge4'] = df['six_veins_count'] >= 4
    # 缠论二买或三买
    df['chan_23'] = df['chan_buy2'] | df['chan_buy3']
    
    df['steady_offset5'] = create_offset_signal(
        df, 
        ['six_ge4', 'buy2', 'chan_23'], 
        offset_days=OFFSET_DAYS
    )
    
    for period in HOLD_PERIODS:
        stats = calculate_returns(df, 'steady_offset5', period)
        results.append({
            'combo_type': '稳健组合 (5日偏移)',
            'offset_days': OFFSET_DAYS,
            'hold_period': period,
            **stats
        })
    
    return pd.DataFrame(results)


def test_market_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    测试不同市场环境下的表现
    
    市场环境分类:
    - 牛市: 20日均线 > 60日均线
    - 熊市: 20日均线 < 60日均线
    - 震荡: 20日均线与60日均线接近
    
    参数:
        df: 已计算所有信号的DataFrame
        
    返回:
        pd.DataFrame: 不同市场环境下的测试结果
    """
    from indicators import MA
    
    results = []
    
    # 计算均线
    df['ma20'] = MA(df['close'], 20)
    df['ma60'] = MA(df['close'], 60)
    
    # 定义市场环境
    df['market_bull'] = df['ma20'] > df['ma60'] * 1.02  # 牛市
    df['market_bear'] = df['ma20'] < df['ma60'] * 0.98  # 熊市
    df['market_range'] = ~df['market_bull'] & ~df['market_bear']  # 震荡
    
    # 稳健组合信号
    df['steady_signal'] = (
        (df['six_veins_count'] >= 4) & 
        df['buy2'] & 
        (df['chan_buy2'] | df['chan_buy3'])
    )
    
    markets = {
        '牛市': 'market_bull',
        '熊市': 'market_bear',
        '震荡': 'market_range'
    }
    
    for market_name, market_col in markets.items():
        # 筛选该市场环境下的信号
        df_market = df[df[market_col]].copy()
        
        for period in [5, 10, 15, 20]:
            stats = calculate_returns(df_market, 'steady_signal', period)
            results.append({
                'market': market_name,
                'hold_period': period,
                **stats
            })
    
    return pd.DataFrame(results)


def generate_report(
    combo_results: pd.DataFrame,
    market_results: pd.DataFrame,
    df: pd.DataFrame
) -> str:
    """生成Markdown格式的测试报告"""
    report = []
    report.append("# 稳健组合策略测试报告")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**测试数据**: 沪深300指数")
    report.append(f"**数据范围**: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
    report.append("")
    
    # 1. 策略定义
    report.append("## 一、策略定义")
    report.append("")
    report.append("### 稳健组合条件")
    report.append("")
    report.append("| 条件 | 说明 |")
    report.append("|------|------|")
    report.append("| 六脉≥4红 | 六脉神剑中至少4个指标显示红色 |")
    report.append("| 买点2 | 庄家线上穿散户线且庄家<50 |")
    report.append("| 缠论二买/三买 | 缠论结构中的二买或三买信号 |")
    report.append("")
    report.append("**触发条件**: 以上三个条件同时满足（或在5天内依次满足）")
    report.append("")
    
    # 2. 组合测试结果
    report.append("## 二、组合测试结果")
    report.append("")
    
    for combo_type in combo_results['combo_type'].unique():
        combo_data = combo_results[combo_results['combo_type'] == combo_type]
        
        report.append(f"### {combo_type}")
        report.append("")
        report.append("| 持有周期 | 信号次数 | 交易次数 | 胜率 | 平均收益 | 夏普比率 |")
        report.append("|----------|----------|----------|------|----------|----------|")
        for _, row in combo_data.iterrows():
            report.append(f"| {row['hold_period']}天 | {row['signal_count']} | {row['trade_count']} | {row['win_rate']:.2f}% | {row['avg_return']:.2f}% | {row['sharpe_ratio']:.2f} |")
        report.append("")
        
        # 找出最优周期
        best = combo_data.loc[combo_data['win_rate'].idxmax()]
        report.append(f"**最优持有周期**: {best['hold_period']}天，胜率 {best['win_rate']:.2f}%，平均收益 {best['avg_return']:.2f}%")
        report.append("")
    
    # 3. 市场环境分析
    report.append("## 三、不同市场环境表现")
    report.append("")
    
    if not market_results.empty:
        for market in market_results['market'].unique():
            market_data = market_results[market_results['market'] == market]
            
            report.append(f"### {market}环境")
            report.append("")
            report.append("| 持有周期 | 信号次数 | 胜率 | 平均收益 |")
            report.append("|----------|----------|------|----------|")
            for _, row in market_data.iterrows():
                report.append(f"| {row['hold_period']}天 | {row['signal_count']} | {row['win_rate']:.2f}% | {row['avg_return']:.2f}% |")
            report.append("")
    
    # 4. 结论
    report.append("## 四、结论与建议")
    report.append("")
    
    # 找出整体最优
    best_overall = combo_results.loc[combo_results['win_rate'].idxmax()]
    
    report.append("### 核心发现")
    report.append("")
    report.append(f"1. **最优组合类型**: {best_overall['combo_type']}")
    report.append(f"2. **最优持有周期**: {best_overall['hold_period']}天")
    report.append(f"3. **预期胜率**: {best_overall['win_rate']:.2f}%")
    report.append(f"4. **平均收益**: {best_overall['avg_return']:.2f}%")
    report.append("")
    report.append("### 使用建议")
    report.append("")
    report.append("1. **适用人群**: 风险厌恶型投资者，追求稳定收益")
    report.append("2. **仓位建议**: 单次建仓不超过30%，分批进场")
    report.append("3. **止损设置**: 建议设置5%止损位")
    report.append("4. **市场选择**: 在牛市和震荡市中效果更佳，熊市慎用")
    report.append("")
    
    return '\n'.join(report)


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    """主程序入口"""
    print("=" * 60)
    print("稳健组合策略测试")
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
    
    # 3. 测试稳健组合
    print("\n测试稳健组合...")
    combo_results = test_steady_combo(df)
    
    # 4. 测试市场环境
    print("\n测试不同市场环境...")
    market_results = test_market_conditions(df)
    
    # 5. 生成报告
    print("\n生成测试报告...")
    report_content = generate_report(combo_results, market_results, df)
    
    report_path = os.path.join(REPORT_DIR, 'steady_combo_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"报告已保存: {report_path}")
    
    # 保存详细数据
    detail_path = os.path.join(REPORT_DIR, 'steady_combo_detail.csv')
    combo_results.to_csv(detail_path, index=False, encoding='utf-8')
    print(f"详细数据已保存: {detail_path}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
