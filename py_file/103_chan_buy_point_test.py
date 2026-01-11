#!/usr/bin/env python3
"""
单指标回测脚本：缠论买点测试
测试缠论一买、二买、三买的独立表现
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import calculate_chan_theory

# 数据目录
DATA_DIR = Path(__file__).parent.parent / "data" / "day"

def load_stock_data(stock_code):
    """加载股票数据"""
    file_path = None
    for market in ['sh', 'sz', 'bj']:
        temp_path = DATA_DIR / market / f"{stock_code}.csv"
        if temp_path.exists():
            file_path = temp_path
            break
    
    if file_path is None:
        file_path = DATA_DIR / f"{stock_code}.csv"
        if not file_path.exists():
            return None
    
    df = pd.read_csv(file_path)
    if '日期' in df.columns:
        df['date'] = pd.to_datetime(df['日期'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        return None
    
    if '收盘' in df.columns:
        df = df.rename(columns={
            '开盘': 'open', '收盘': 'close', '最高': 'high', 
            '最低': 'low', '成交量': 'volume'
        })
    
    df = df.sort_values('date').reset_index(drop=True)
    return df

def test_chan_signals(hold_days=14):
    """测试缠论各个买点信号"""
    results = {
        'chan_buy1': [],
        'chan_buy2': [],
        'chan_buy3': []
    }
    
    stock_files = list(DATA_DIR.rglob("*.csv"))
    print(f"开始运行缠论买点测试，共 {len(stock_files)} 只股票，持有期 {hold_days} 天...")
    
    for i, stock_file in enumerate(stock_files):
        stock_code = stock_file.stem
        df = load_stock_data(stock_code)
        
        if df is None or len(df) < 100:
            continue
            
        df = calculate_chan_theory(df)
        
        for signal_name in results.keys():
            buy_indices = df[df[signal_name]].index.tolist()
            
            for idx in buy_indices:
                if idx + hold_days >= len(df):
                    continue
                    
                buy_price = df.loc[idx, 'close']
                sell_price = df.loc[idx + hold_days, 'close']
                
                if buy_price == 0 or pd.isna(buy_price):
                    continue
                    
                return_pct = (sell_price - buy_price) / buy_price * 100
                results[signal_name].append({
                    'return': return_pct,
                    'win': return_pct > 0,
                    'year': df.loc[idx, 'date'].year
                })
                
        if (i + 1) % 500 == 0:
            print(f"已处理 {i + 1} 只股票...")
            
    return results

def print_summary(all_results):
    print("\n" + "="*50)
    print("缠论买点单指标回测汇总")
    print("="*50)
    
    for signal, trades in all_results.items():
        if not trades:
            print(f"\n信号 {signal}: 无交易数据")
            continue
            
        df = pd.DataFrame(trades)
        win_rate = (df['win'].sum() / len(df)) * 100
        avg_return = df['return'].mean()
        
        print(f"\n信号: {signal}")
        print(f"  总交易次数: {len(df)}")
        print(f"  总胜率: {win_rate:.2f}%")
        print(f"  平均收益率: {avg_return:.2f}%")
        
        # 年度胜率
        yearly_win = df.groupby('year')['win'].mean() * 100
        print(f"  年度胜率: {yearly_win.to_dict()}")

if __name__ == "__main__":
    results = test_chan_signals()
    print_summary(results)
