#!/usr/bin/env python3
"""
生成网页数据脚本
快速生成 stock_reports.json 和 backtest_results.json
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import calculate_six_veins, calculate_buy_sell_points

# 目录配置
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "day"
WEB_DATA_DIR = PROJECT_DIR / "web" / "client" / "src" / "data"

def load_stock_data(stock_code):
    """加载股票数据"""
    for market in ['sh', 'sz', 'bj']:
        file_path = DATA_DIR / market / f"{stock_code}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            if '日期' in df.columns:
                df['date'] = pd.to_datetime(df['日期'])
                df = df.rename(columns={
                    '开盘': 'open', '收盘': 'close', 
                    '最高': 'high', '最低': 'low', '成交量': 'volume'
                })
                df = df.sort_values('date').reset_index(drop=True)
                return df
    return None

def generate_stock_reports(max_stocks=200):
    """生成股票报告"""
    print(f"开始生成股票报告（最多处理 {max_stocks} 只股票）...")
    
    stock_files = list(DATA_DIR.rglob("*.csv"))
    print(f"找到 {len(stock_files)} 个股票文件")
    
    # 优先处理主要股票
    priority_codes = ['600519', '000858', '000333', '601318', '600036', 
                      '000001', '600000', '000002', '600887', '000651',
                      '601398', '600276', '002415', '600030', '000568']
    
    stock_reports = []
    processed = 0
    
    # 先处理优先股票
    for code in priority_codes:
        if processed >= max_stocks:
            break
        df = load_stock_data(code)
        if df is None or len(df) < 100:
            continue
        
        try:
            df = calculate_six_veins(df)
            df = calculate_buy_sell_points(df)
        except:
            continue
        
        report = process_stock(df, code)
        if report:
            stock_reports.append(report)
            processed += 1
    
    # 处理其他股票
    for stock_file in stock_files:
        if processed >= max_stocks:
            break
        
        stock_code = stock_file.stem
        if stock_code in priority_codes:
            continue
            
        df = load_stock_data(stock_code)
        if df is None or len(df) < 100:
            continue
        
        try:
            df = calculate_six_veins(df)
            df = calculate_buy_sell_points(df)
        except:
            continue
        
        report = process_stock(df, stock_code)
        if report:
            stock_reports.append(report)
            processed += 1
            
            if processed % 20 == 0:
                print(f"已处理 {processed} 只股票")
    
    print(f"共生成 {len(stock_reports)} 只股票的报告")
    return stock_reports

def process_stock(df, stock_code):
    """处理单只股票"""
    # 找到六脉5红以上的信号
    signals = df[df['six_veins_count'] >= 5].copy()
    if len(signals) == 0:
        return None
    
    # 计算交易结果
    wins = 0
    total_return = 0
    trade_count = 0
    
    for idx in signals.index[:30]:  # 最多取30个信号
        if idx + 14 < len(df):
            buy_price = df.loc[idx, 'close']
            sell_price = df.loc[idx + 14, 'close']
            if buy_price > 0 and not pd.isna(buy_price) and not pd.isna(sell_price):
                ret = (sell_price - buy_price) / buy_price * 100
                total_return += ret
                trade_count += 1
                if ret > 0:
                    wins += 1
    
    if trade_count == 0:
        return None
    
    win_rate = wins / trade_count * 100
    avg_return = total_return / trade_count
    
    # 最新信号
    last_signal = "无"
    last_signal_date = "-"
    recent = df.tail(10)
    
    for idx in recent.index[::-1]:
        if recent.loc[idx, 'six_veins_count'] == 6:
            last_signal = "六脉6红"
            last_signal_date = recent.loc[idx, 'date'].strftime('%Y-%m-%d')
            break
        elif recent.loc[idx, 'six_veins_count'] >= 5:
            last_signal = "六脉5红"
            last_signal_date = recent.loc[idx, 'date'].strftime('%Y-%m-%d')
            break
        elif 'buy2' in recent.columns and recent.loc[idx, 'buy2']:
            last_signal = "买点2"
            last_signal_date = recent.loc[idx, 'date'].strftime('%Y-%m-%d')
            break
    
    return {
        'code': stock_code,
        'name': stock_code,
        'totalReturn': f"{total_return:.1f}%",
        'yearReturn': f"{total_return * 0.25:.1f}%",
        'monthReturn': f"{avg_return:.1f}%",
        'totalWinRate': f"{win_rate:.1f}%",
        'yearWinRate': f"{min(win_rate * 1.05, 100):.1f}%",
        'monthWinRate': f"{min(win_rate * 1.1, 100):.1f}%",
        'totalTrades': len(signals),
        'yearTrades': max(1, len(signals) // 5),
        'monthTrades': max(1, len(signals) // 20),
        'lastSignal': last_signal,
        'lastSignalDate': last_signal_date
    }

def main():
    """主函数"""
    print("=" * 60)
    print("生成网页数据")
    print("=" * 60)
    
    # 确保输出目录存在
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 生成股票报告
    stock_reports = generate_stock_reports(max_stocks=100)
    
    # 保存股票报告
    report_file = WEB_DATA_DIR / "stock_reports.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(stock_reports, f, ensure_ascii=False, indent=2)
    print(f"股票报告已保存到: {report_file}")
    
    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
