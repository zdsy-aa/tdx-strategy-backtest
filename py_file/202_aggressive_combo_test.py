#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
202_激进组合全量股票测试
修改版：支持遍历所有股票数据并计算平均胜率
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from multiprocessing import Pool, cpu_count

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import calculate_all_signals
from backtest_utils import get_all_stock_files, aggregate_results

# ==============================================================================
# 配置常量
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report', 'total')
HOLD_PERIODS = [5, 10, 15]

# ==============================================================================
# 回测核心函数
# ==============================================================================

def calculate_returns(df: pd.DataFrame, signal_col: str, hold_period: int) -> Dict:
    signals = df[df[signal_col] == True].copy()
    if len(signals) == 0:
        return {'signal_count': 0, 'trade_count': 0, 'win_rate': np.nan, 'avg_return': np.nan}
    
    returns = []
    for idx in signals.index:
        if idx + hold_period < len(df):
            entry_price = df.loc[idx, 'close']
            exit_price = df.loc[idx + hold_period, 'close']
            if entry_price > 0:
                ret = (exit_price - entry_price) / entry_price * 100
                returns.append(ret)
    
    if len(returns) == 0:
        return {'signal_count': len(signals), 'trade_count': 0, 'win_rate': np.nan, 'avg_return': np.nan}
    
    returns = np.array(returns)
    return {
        'signal_count': len(signals),
        'trade_count': len(returns),
        'win_rate': np.sum(returns > 0) / len(returns) * 100,
        'avg_return': np.mean(returns)
    }

def process_single_stock(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        if len(df) < 100: return None
        
        name_map = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}
        df = df.rename(columns=name_map)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
        df = calculate_all_signals(df)
        
        # 定义激进组合
        combos = {
            '激进1 (六脉≥5红+买点2)': (df['six_veins_count'] >= 5) & df['buy2'],
            '激进2 (六脉6红+摇钱树)': (df['six_veins_count'] == 6) & df['money_tree'],
            '激进3 (缠论一买+六脉≥4红)': df['chan_buy1'] & (df['six_veins_count'] >= 4)
        }
        
        results = []
        for name, signal_series in combos.items():
            df[name] = signal_series
            for period in HOLD_PERIODS:
                stats = calculate_returns(df, name, period)
                if stats['trade_count'] > 0:
                    results.append({
                        'strategy': name,
                        'hold_period': period,
                        **stats
                    })
        return pd.DataFrame(results)
    except Exception:
        return None

def main():
    print("=" * 60)
    print("激进组合全量股票回测 (计算平均胜率)")
    print("=" * 60)
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    stock_files = get_all_stock_files(DATA_DIR)
    
    with Pool(cpu_count()) as pool:
        all_results = pool.map(process_single_stock, stock_files)
    
    all_results = [r for r in all_results if r is not None and not r.empty]
    if not all_results:
        print("没有有效的回测结果")
        return
    
    summary = aggregate_results(all_results, ['strategy', 'hold_period'])
    
    # 生成报告
    report_path = os.path.join(REPORT_DIR, 'aggressive_combo_all_stocks_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 激进组合全量股票回测报告\n\n")
        f.write(f"**测试股票数量**: {len(all_results)}\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 汇总统计 (平均胜率)\n\n")
        f.write("| 策略名称 | 持有周期 | 总信号数 | 总交易数 | 平均胜率 | 平均收益 |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for _, row in summary.sort_values('win_rate', ascending=False).iterrows():
            f.write(f"| {row['strategy']} | {row['hold_period']}天 | {int(row['signal_count'])} | {int(row['trade_count'])} | {row['win_rate']:.2f}% | {row['avg_return']:.2f}% |\n")
            
    print(f"报告已生成: {report_path}")
    summary.to_csv(os.path.join(REPORT_DIR, 'aggressive_combo_all_stocks_summary.csv'), index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()
