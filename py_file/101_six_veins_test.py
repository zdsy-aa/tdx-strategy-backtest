#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
101_六脉神剑单指标测试 (101_six_veins_test.py)
================================================================================

测试内容:
    本脚本专门测试六脉神剑指标的各种组合效果，包括：
    1. 单个指标红色状态的胜率
    2. 任意N个指标同时变红的胜率 (N=2,3,4,5,6)
    3. 最优4红组合的筛选
    4. 不同持有周期的收益对比

六脉神剑六大指标:
    - MACD: 移动平均收敛发散
    - KDJ: 随机指标
    - RSI: 相对强弱指数
    - LWR: 威廉指标变种
    - BBI: 多空分界线
    - MTM: 动量指标

输出文件:
    - /report/total/six_veins_report.md: 综合报告
    - /report/total/six_veins_detail.csv: 详细数据

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
from itertools import combinations
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import calculate_six_veins, MA


# ==============================================================================
# 配置常量
# ==============================================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')

# 报告输出目录
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report', 'total')

# 六脉神剑的六个指标名称
SIX_INDICATORS = ['macd_red', 'kdj_red', 'rsi_red', 'lwr_red', 'bbi_red', 'mtm_red']

# 指标中文名称映射
INDICATOR_NAMES = {
    'macd_red': 'MACD',
    'kdj_red': 'KDJ',
    'rsi_red': 'RSI',
    'lwr_red': 'LWR',
    'bbi_red': 'BBI',
    'mtm_red': 'MTM'
}

# 测试的持有周期列表
HOLD_PERIODS = [1, 2, 3, 5, 10, 15, 20, 30]


# ==============================================================================
# 回测核心函数
# ==============================================================================

def load_market_data(filename: str = 'sh000300.csv') -> pd.DataFrame:
    """
    加载市场数据
    
    参数:
        filename: 数据文件名，默认使用沪深300指数
        
    返回:
        pd.DataFrame: 市场数据
    """
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"错误: 数据文件不存在 - {filepath}")
        print("请先运行 data_fetcher.py 下载数据")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"已加载数据: {filename}")
    print(f"时间范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"数据条数: {len(df)}")
    
    return df


def calculate_returns(df: pd.DataFrame, signal_col: str, hold_period: int) -> Dict:
    """
    计算指定信号的收益统计
    
    参数:
        df: 包含信号列的DataFrame
        signal_col: 信号列名
        hold_period: 持有周期 (天)
        
    返回:
        Dict: 包含胜率、平均收益等统计数据
    """
    # 获取信号触发点
    signals = df[df[signal_col] == True].copy()
    
    if len(signals) == 0:
        return {
            'signal_count': 0,
            'win_rate': 0,
            'avg_return': 0,
            'max_return': 0,
            'min_return': 0,
            'total_return': 0
        }
    
    # 计算每次信号后的收益
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
            'win_rate': 0,
            'avg_return': 0,
            'max_return': 0,
            'min_return': 0,
            'total_return': 0
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


def test_single_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    测试单个指标的效果
    
    参数:
        df: 已计算六脉神剑指标的DataFrame
        
    返回:
        pd.DataFrame: 单指标测试结果
    """
    results = []
    
    for indicator in SIX_INDICATORS:
        # 创建信号: 指标从非红变红
        signal_col = f'{indicator}_signal'
        df[signal_col] = df[indicator] & ~df[indicator].shift(1).fillna(False)
        
        for period in HOLD_PERIODS:
            stats = calculate_returns(df, signal_col, period)
            results.append({
                'indicator': INDICATOR_NAMES[indicator],
                'hold_period': period,
                **stats
            })
    
    return pd.DataFrame(results)


def test_n_red_combinations(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    测试N个指标同时变红的效果
    
    参数:
        df: 已计算六脉神剑指标的DataFrame
        n: 要求同时变红的指标数量
        
    返回:
        pd.DataFrame: N红组合测试结果
    """
    results = []
    
    # 生成所有N个指标的组合
    combos = list(combinations(SIX_INDICATORS, n))
    
    for combo in combos:
        combo_name = '+'.join([INDICATOR_NAMES[ind] for ind in combo])
        
        # 创建组合信号: 所有指定指标同时为红
        signal_col = f'combo_{n}red_{"_".join(combo)}'
        df[signal_col] = df[list(combo)].all(axis=1)
        
        # 信号触发: 从非全红变为全红
        df[f'{signal_col}_trigger'] = df[signal_col] & ~df[signal_col].shift(1).fillna(False)
        
        for period in HOLD_PERIODS:
            stats = calculate_returns(df, f'{signal_col}_trigger', period)
            results.append({
                'combination': combo_name,
                'n_indicators': n,
                'hold_period': period,
                **stats
            })
    
    return pd.DataFrame(results)


def find_best_4red_combination(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    寻找最优的4红组合
    
    参数:
        df: 已计算六脉神剑指标的DataFrame
        
    返回:
        Tuple[str, pd.DataFrame]: (最优组合名称, 详细结果DataFrame)
    """
    results_4red = test_n_red_combinations(df, 4)
    
    # 按胜率排序，找出最优组合
    best_by_period = results_4red.loc[results_4red.groupby('hold_period')['win_rate'].idxmax()]
    
    # 综合评分: 胜率 * 0.6 + 平均收益 * 0.4
    results_4red['score'] = results_4red['win_rate'] * 0.6 + results_4red['avg_return'] * 10 * 0.4
    best_overall = results_4red.loc[results_4red.groupby('combination')['score'].mean().idxmax()]
    
    return best_overall, results_4red


def test_count_based_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    测试基于红色数量的信号效果
    
    参数:
        df: 已计算六脉神剑指标的DataFrame
        
    返回:
        pd.DataFrame: 基于数量的测试结果
    """
    results = []
    
    for min_count in range(1, 7):
        # 创建信号: 红色数量 >= min_count
        signal_col = f'count_ge_{min_count}'
        df[signal_col] = df['six_veins_count'] >= min_count
        
        # 信号触发
        df[f'{signal_col}_trigger'] = df[signal_col] & ~df[signal_col].shift(1).fillna(False)
        
        for period in HOLD_PERIODS:
            stats = calculate_returns(df, f'{signal_col}_trigger', period)
            results.append({
                'min_red_count': min_count,
                'hold_period': period,
                **stats
            })
    
    return pd.DataFrame(results)


# ==============================================================================
# 报告生成函数
# ==============================================================================

def generate_report(
    single_results: pd.DataFrame,
    count_results: pd.DataFrame,
    best_4red: str,
    results_4red: pd.DataFrame,
    df: pd.DataFrame
) -> str:
    """
    生成Markdown格式的测试报告
    
    参数:
        single_results: 单指标测试结果
        count_results: 基于数量的测试结果
        best_4red: 最优4红组合
        results_4red: 4红组合详细结果
        df: 原始数据
        
    返回:
        str: Markdown格式的报告内容
    """
    report = []
    report.append("# 六脉神剑指标测试报告")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**测试数据**: 沪深300指数")
    report.append(f"**数据范围**: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
    report.append(f"**数据条数**: {len(df)}")
    report.append("")
    
    # 1. 单指标测试结果
    report.append("## 一、单指标测试结果")
    report.append("")
    report.append("测试每个指标单独变红时的买入效果。")
    report.append("")
    
    # 找出每个指标的最优持有周期
    best_single = single_results.loc[single_results.groupby('indicator')['win_rate'].idxmax()]
    report.append("| 指标 | 最优周期 | 信号次数 | 胜率 | 平均收益 |")
    report.append("|------|----------|----------|------|----------|")
    for _, row in best_single.iterrows():
        report.append(f"| {row['indicator']} | {row['hold_period']}天 | {row['signal_count']} | {row['win_rate']:.2f}% | {row['avg_return']:.2f}% |")
    report.append("")
    
    # 2. 基于红色数量的测试结果
    report.append("## 二、基于红色数量的测试结果")
    report.append("")
    report.append("测试不同红色指标数量阈值的买入效果。")
    report.append("")
    
    best_count = count_results.loc[count_results.groupby('min_red_count')['win_rate'].idxmax()]
    report.append("| 最少红色数 | 最优周期 | 信号次数 | 胜率 | 平均收益 |")
    report.append("|------------|----------|----------|------|----------|")
    for _, row in best_count.iterrows():
        report.append(f"| ≥{row['min_red_count']}红 | {row['hold_period']}天 | {row['signal_count']} | {row['win_rate']:.2f}% | {row['avg_return']:.2f}% |")
    report.append("")
    
    # 3. 最优4红组合
    report.append("## 三、最优4红组合分析")
    report.append("")
    report.append("从15种4红组合中筛选出最优组合。")
    report.append("")
    
    # 按综合得分排序
    results_4red['score'] = results_4red['win_rate'] * 0.6 + results_4red['avg_return'] * 10 * 0.4
    combo_scores = results_4red.groupby('combination').agg({
        'win_rate': 'mean',
        'avg_return': 'mean',
        'signal_count': 'sum',
        'score': 'mean'
    }).sort_values('score', ascending=False)
    
    report.append("### 4红组合综合排名 (Top 10)")
    report.append("")
    report.append("| 排名 | 组合 | 平均胜率 | 平均收益 | 综合得分 |")
    report.append("|------|------|----------|----------|----------|")
    for i, (combo, row) in enumerate(combo_scores.head(10).iterrows(), 1):
        report.append(f"| {i} | {combo} | {row['win_rate']:.2f}% | {row['avg_return']:.2f}% | {row['score']:.2f} |")
    report.append("")
    
    # 4. 最优组合详情
    best_combo_name = combo_scores.index[0]
    best_combo_detail = results_4red[results_4red['combination'] == best_combo_name]
    
    report.append(f"### 最优组合详情: {best_combo_name}")
    report.append("")
    report.append("| 持有周期 | 信号次数 | 胜率 | 平均收益 | 最大收益 | 最大亏损 |")
    report.append("|----------|----------|------|----------|----------|----------|")
    for _, row in best_combo_detail.iterrows():
        report.append(f"| {row['hold_period']}天 | {row['trade_count']} | {row['win_rate']:.2f}% | {row['avg_return']:.2f}% | {row['max_return']:.2f}% | {row['min_return']:.2f}% |")
    report.append("")
    
    # 5. 结论与建议
    report.append("## 四、结论与建议")
    report.append("")
    report.append("### 核心发现")
    report.append("")
    report.append(f"1. **最优4红组合**: {best_combo_name}")
    report.append(f"2. **推荐持有周期**: {best_combo_detail.loc[best_combo_detail['win_rate'].idxmax(), 'hold_period']}天")
    report.append(f"3. **预期胜率**: {best_combo_detail['win_rate'].max():.2f}%")
    report.append("")
    report.append("### 使用建议")
    report.append("")
    report.append("1. **稳健型投资者**: 建议等待≥5红信号，持有周期10-15天")
    report.append("2. **激进型投资者**: 可在4红组合出现时介入，持有周期3-5天")
    report.append("3. **风险控制**: 无论何种策略，建议设置5%止损位")
    report.append("")
    
    return '\n'.join(report)


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    """
    主程序入口
    """
    print("=" * 60)
    print("六脉神剑指标测试")
    print("=" * 60)
    
    # 确保报告目录存在
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 1. 加载数据
    df = load_market_data()
    if df.empty:
        print("无法加载数据，退出测试")
        return
    
    # 2. 计算六脉神剑指标
    print("\n计算六脉神剑指标...")
    df = calculate_six_veins(df)
    
    # 3. 单指标测试
    print("\n测试单指标效果...")
    single_results = test_single_indicator(df)
    
    # 4. 基于数量的测试
    print("\n测试基于红色数量的效果...")
    count_results = test_count_based_signals(df)
    
    # 5. 4红组合测试
    print("\n测试4红组合效果...")
    best_4red, results_4red = find_best_4red_combination(df)
    
    # 6. 生成报告
    print("\n生成测试报告...")
    report_content = generate_report(single_results, count_results, best_4red, results_4red, df)
    
    # 保存报告
    report_path = os.path.join(REPORT_DIR, 'six_veins_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"报告已保存: {report_path}")
    
    # 保存详细数据
    detail_path = os.path.join(REPORT_DIR, 'six_veins_detail.csv')
    results_4red.to_csv(detail_path, index=False, encoding='utf-8')
    print(f"详细数据已保存: {detail_path}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
