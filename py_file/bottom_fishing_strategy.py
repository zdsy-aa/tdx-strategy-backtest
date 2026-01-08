#!/usr/bin/env python3
"""
抄底方案回测脚本
买入信号：缠论一买信号(chan_buy1) + 六脉神剑 ≥ 3红
卖出条件：
1. 缠论二买(chan_buy2)出现后减仓50%
2. 缠论三买(chan_buy3)出现后清仓
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import calculate_six_veins, calculate_chan_theory

# 数据目录
DATA_DIR = Path(__file__).parent.parent / "data" / "day"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "backtest_results"

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

def run_bottom_fishing_backtest():
    """运行抄底方案回测"""
    all_trades = []
    stock_files = list(DATA_DIR.rglob("*.csv"))
    
    print(f"开始运行抄底方案回测，共 {len(stock_files)} 只股票...")
    
    for i, stock_file in enumerate(stock_files):
        stock_code = stock_file.stem
        df = load_stock_data(stock_code)
        
        if df is None or len(df) < 100:
            continue
            
        # 计算指标
        df = calculate_six_veins(df)
        df = calculate_chan_theory(df)
        
        # 买入信号：缠论一买 + 六脉神剑 >= 3红
        df['buy_signal'] = df['chan_buy1'] & (df['six_veins_count'] >= 3)
        
        buy_indices = df[df['buy_signal']].index.tolist()
        
        for buy_idx in buy_indices:
            buy_price = df.loc[buy_idx, 'close']
            buy_date = df.loc[buy_idx, 'date']
            
            # 查找卖出点
            # 1. 缠论二买减仓50%
            # 2. 缠论三买清仓
            
            sell_half_idx = None
            sell_all_idx = None
            
            # 在买入点之后查找
            future_df = df.loc[buy_idx+1:]
            
            # 查找二买
            chan2_signals = future_df[future_df['chan_buy2']].index.tolist()
            if chan2_signals:
                sell_half_idx = chan2_signals[0]
                
            # 查找三买
            chan3_signals = future_df[future_df['chan_buy3']].index.tolist()
            if chan3_signals:
                sell_all_idx = chan3_signals[0]
            
            # 如果没有二买或三买，则以最后一天清仓（或者设定一个最大持有天数，这里假设直到最后）
            if sell_half_idx is None and sell_all_idx is None:
                continue # 忽略未完成的交易
                
            # 计算收益
            # 逻辑：50% 在二买卖出，50% 在三买卖出
            # 如果只有三买没有二买，则全部在三买卖出
            # 如果只有二买没有三买，则50%在二买卖出，剩余50%在最后一天卖出
            
            results = []
            
            # 验证买入价
            if buy_price <= 0 or pd.isna(buy_price):
                continue

            results = []
            
            # 第一部分：50% 仓位
            if sell_half_idx:
                p1_sell_price = df.loc[sell_half_idx, 'close']
                if p1_sell_price > 0:
                    p1_return = (p1_sell_price - buy_price) / buy_price * 100
                    results.append(p1_return * 0.5)
            elif sell_all_idx:
                p1_sell_price = df.loc[sell_all_idx, 'close']
                if p1_sell_price > 0:
                    p1_return = (p1_sell_price - buy_price) / buy_price * 100
                    results.append(p1_return * 0.5)
            
            if not results: # 如果第一部分没卖成，跳过
                continue
                
            # 第二部分：剩余 50% 仓位
            if sell_all_idx:
                p2_sell_price = df.loc[sell_all_idx, 'close']
                if p2_sell_price > 0:
                    p2_return = (p2_sell_price - buy_price) / buy_price * 100
                    results.append(p2_return * 0.5)
            else:
                # 如果没有三买，最后一天卖出
                p2_sell_price = df.iloc[-1]['close']
                if p2_sell_price > 0:
                    p2_return = (p2_sell_price - buy_price) / buy_price * 100
                    results.append(p2_return * 0.5)
                
            total_return = sum(results)
            
            all_trades.append({
                'stock_code': stock_code,
                'buy_date': buy_date,
                'buy_price': buy_price,
                'sell_half_date': df.loc[sell_half_idx, 'date'] if sell_half_idx else None,
                'sell_all_date': df.loc[sell_all_idx, 'date'] if sell_all_idx else None,
                'return': total_return,
                'win': total_return > 0,
                'year': buy_date.year
            })
            
        if (i + 1) % 500 == 0:
            print(f"已处理 {i + 1} 只股票...")
            
    return all_trades

def print_stats(trades):
    if not trades:
        print("没有产生交易信号。")
        return
        
    df_trades = pd.DataFrame(trades)
    win_rate = (df_trades['win'].sum() / len(df_trades)) * 100
    avg_return = df_trades['return'].mean()
    
    print("\n" + "="*40)
    print("抄底方案回测统计")
    print("="*40)
    print(f"总交易次数: {len(df_trades)}")
    print(f"总胜率: {win_rate:.2f}%")
    print(f"平均收益率: {avg_return:.2f}%")
    
    print("\n年度统计:")
    yearly = df_trades.groupby('year').agg({
        'return': ['count', 'mean'],
        'win': 'sum'
    })
    for year, row in yearly.iterrows():
        y_win_rate = (row[('win', 'sum')] / row[('return', 'count')]) * 100
        print(f"  {year}年: 次数={int(row[('return', 'count')])}, 胜率={y_win_rate:.2f}%, 平均收益={row[('return', 'mean')]:.2f}%")

if __name__ == "__main__":
    trades = run_bottom_fishing_backtest()
    print_stats(trades)
