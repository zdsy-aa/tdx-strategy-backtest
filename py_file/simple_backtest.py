#!/usr/bin/env python3
"""
简化版回测脚本 - 快速生成回测结果
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import calculate_six_veins, calculate_buy_sell_points

DATA_DIR = Path(__file__).parent.parent / "data" / "day"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "backtest_results"

def load_and_process_stock(stock_code):
    """加载并处理股票数据"""
    file_path = DATA_DIR / f"{stock_code}.csv"
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif '日期' in df.columns:
            df['date'] = pd.to_datetime(df['日期'])
        else:
            return None
        
        if '收盘' in df.columns:
            df = df.rename(columns={
                '开盘': 'open', '收盘': 'close', '最高': 'high', 
                '最低': 'low', '成交量': 'volume'
            })
        
        df = df[df['close'] > 0].copy()
        df = df.sort_values('date').reset_index(drop=True)
        df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)  # 只用最近5年数据
        
        if len(df) < 100:
            return None
            
        df = calculate_six_veins(df)
        df = calculate_buy_sell_points(df)
        return df
    except:
        return None

def backtest_signal(df, signal_col, hold_days=14):
    """回测信号"""
    if signal_col not in df.columns:
        return []
    
    trades = []
    signals = df[df[signal_col] == True].copy()
    
    for idx in signals.index:
        buy_price = df.loc[idx, 'close']
        buy_date = df.loc[idx, 'date']
        
        if buy_price <= 0 or pd.isna(buy_price):
            continue
        
        sell_idx = min(idx + hold_days, len(df) - 1)
        if sell_idx <= idx:
            continue
            
        sell_price = df.loc[sell_idx, 'close']
        if sell_price <= 0 or pd.isna(sell_price):
            continue
        
        return_pct = (sell_price - buy_price) / buy_price * 100
        
        trades.append({
            'buy_date': buy_date,
            'return': return_pct,
            'win': return_pct > 0,
            'year': buy_date.year,
            'month': buy_date.month
        })
    
    return trades

def calculate_stats(trades):
    """计算统计数据"""
    if not trades:
        return {'total': {'trades': 0, 'win_rate': 0, 'avg_return': 0}, 'yearly': {}, 'monthly': {}}
    
    wins = sum(1 for t in trades if t['win'])
    returns = [t['return'] for t in trades]
    
    total_stats = {
        'trades': len(trades),
        'win_rate': round(wins / len(trades) * 100, 1),
        'avg_return': round(np.mean(returns), 2),
        'total_return': round(np.sum(returns), 2),
        'max_return': round(np.max(returns), 2),
        'min_return': round(np.min(returns), 2)
    }
    
    # 年度统计
    yearly_stats = {}
    trades_by_year = defaultdict(list)
    for t in trades:
        trades_by_year[t['year']].append(t)
    
    for year, year_trades in sorted(trades_by_year.items()):
        wins = sum(1 for t in year_trades if t['win'])
        returns = [t['return'] for t in year_trades]
        yearly_stats[str(year)] = {
            'trades': len(year_trades),
            'win_rate': round(wins / len(year_trades) * 100, 1),
            'avg_return': round(np.mean(returns), 2)
        }
    
    # 月度统计
    monthly_stats = {}
    trades_by_month = defaultdict(list)
    for t in trades:
        trades_by_month[t['month']].append(t)
    
    for month, month_trades in sorted(trades_by_month.items()):
        wins = sum(1 for t in month_trades if t['win'])
        returns = [t['return'] for t in month_trades]
        monthly_stats[f"{month}月"] = {
            'trades': len(month_trades),
            'win_rate': round(wins / len(month_trades) * 100, 1),
            'avg_return': round(np.mean(returns), 2)
        }
    
    return {'total': total_stats, 'yearly': yearly_stats, 'monthly': monthly_stats}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    stock_files = list(DATA_DIR.glob("*.csv"))[:100]  # 只处理前100只股票
    print(f"处理 {len(stock_files)} 只股票...")
    
    # 信号类型
    signal_configs = [
        ('six_veins_6red', '六脉6红', lambda df: (df['six_veins_count'] == 6) & (df['six_veins_count'].shift(1) != 6), 14),
        ('six_veins_5red', '六脉5红', lambda df: (df['six_veins_count'] >= 5) & (df['six_veins_count'].shift(1) < 5), 14),
        ('buy_point_2', '买点2', lambda df: df.get('buy2', pd.Series(False, index=df.index)), 16),
    ]
    
    all_results = {}
    
    for signal_id, signal_name, signal_func, hold_days in signal_configs:
        print(f"\n测试: {signal_name}")
        all_trades = []
        
        for i, stock_file in enumerate(stock_files):
            df = load_and_process_stock(stock_file.stem)
            if df is None:
                continue
            
            df['signal'] = signal_func(df)
            trades = backtest_signal(df, 'signal', hold_days)
            all_trades.extend(trades)
            
            if (i + 1) % 20 == 0:
                print(f"  已处理 {i + 1}/{len(stock_files)}...")
        
        stats = calculate_stats(all_trades)
        all_results[signal_id] = {'name': signal_name, 'stats': stats, 'hold_days': hold_days}
        print(f"  交易: {stats['total']['trades']}, 胜率: {stats['total']['win_rate']}%, 收益: {stats['total']['avg_return']}%")
    
    # 测试卖出点
    print("\n测试卖出点...")
    sell_results = {}
    for days in [1, 3, 5, 10, 14, 20, 30]:
        all_trades = []
        for stock_file in stock_files:
            df = load_and_process_stock(stock_file.stem)
            if df is None:
                continue
            df['signal'] = (df['six_veins_count'] >= 5) & (df['six_veins_count'].shift(1) < 5)
            trades = backtest_signal(df, 'signal', days)
            all_trades.extend(trades)
        
        stats = calculate_stats(all_trades)
        sell_results[f'{days}天后卖出'] = stats['total']
        print(f"  {days}天: 胜率 {stats['total']['win_rate']}%, 收益 {stats['total']['avg_return']}%")
    
    # 生成股票报告
    print("\n生成股票报告...")
    stock_reports = []
    for stock_file in stock_files:
        df = load_and_process_stock(stock_file.stem)
        if df is None:
            continue
        
        df['signal'] = (df['six_veins_count'] >= 5) & (df['six_veins_count'].shift(1) < 5)
        trades = backtest_signal(df, 'signal', 14)
        
        if not trades:
            continue
        
        total_wins = sum(1 for t in trades if t['win'])
        total_returns = [t['return'] for t in trades]
        
        # 2025年数据
        year_trades = [t for t in trades if t['year'] == 2025]
        year_wins = sum(1 for t in year_trades if t['win'])
        year_returns = [t['return'] for t in year_trades]
        
        # 最近一个月
        month_trades = [t for t in trades if t['year'] == 2025 and t['month'] == 1]
        month_wins = sum(1 for t in month_trades if t['win'])
        month_returns = [t['return'] for t in month_trades]
        
        stock_reports.append({
            'code': stock_file.stem,
            'name': stock_file.stem,
            'totalReturn': f"{sum(total_returns):.1f}%",
            'yearReturn': f"{sum(year_returns):.1f}%" if year_returns else "0%",
            'monthReturn': f"{sum(month_returns):.1f}%" if month_returns else "0%",
            'totalWinRate': f"{total_wins/len(trades)*100:.1f}%" if trades else "0%",
            'yearWinRate': f"{year_wins/len(year_trades)*100:.1f}%" if year_trades else "0%",
            'monthWinRate': f"{month_wins/len(month_trades)*100:.1f}%" if month_trades else "0%",
            'totalTrades': len(trades),
            'yearTrades': len(year_trades),
            'monthTrades': len(month_trades),
        })
    
    # 保存结果
    with open(OUTPUT_DIR / "backtest_results.json", 'w', encoding='utf-8') as f:
        json.dump({'strategies': all_results, 'sell_points': sell_results}, f, ensure_ascii=False, indent=2, default=str)
    
    with open(OUTPUT_DIR / "stock_reports.json", 'w', encoding='utf-8') as f:
        json.dump(stock_reports, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到 {OUTPUT_DIR}")
    print(f"股票报告: {len(stock_reports)} 只")
    
    return all_results, sell_results, stock_reports

if __name__ == "__main__":
    main()
