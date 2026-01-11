#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整回测脚本 (并行化版本)
功能：统计各策略在全市场的胜率、收益率等指标，生成策略排行榜
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import calculate_six_veins, calculate_buy_sell_points, calculate_money_tree, calculate_chan_theory

# 配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "day"
WEB_DATA_DIR = PROJECT_ROOT / "web" / "client" / "src" / "data"
BACKTEST_RESULTS_FILE = WEB_DATA_DIR / "backtest_results.json"

def load_stock_data(file_path):
    """加载股票数据"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        if df.empty: return None
        if '日期' in df.columns: df['date'] = pd.to_datetime(df['日期'])
        elif 'date' in df.columns: df['date'] = pd.to_datetime(df['date'])
        else: return None
        column_map = {'开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        return df.sort_values('date').reset_index(drop=True)
    except: return None

def calculate_all_indicators(df):
    """计算所有指标"""
    df = calculate_six_veins(df)
    df = calculate_buy_sell_points(df)
    try: df = calculate_money_tree(df)
    except: df['money_tree'] = False
    try: df = calculate_chan_theory(df)
    except: df['chan_buy1'] = df['chan_buy2'] = df['chan_buy3'] = False
    return df

def find_signals(df, signal_type):
    """找到买入信号"""
    if df is None or df.empty: return []
    if signal_type == "six_veins_6red": mask = (df['six_veins_count'] == 6) & (df['six_veins_count'].shift(1) < 6)
    elif signal_type == "six_veins_5red": mask = (df['six_veins_count'] >= 5) & (df['six_veins_count'].shift(1) < 5)
    elif signal_type == "six_veins_4red": mask = (df['six_veins_count'] >= 4) & (df['six_veins_count'].shift(1) < 4)
    elif signal_type in ["buy_point_1", "buy_point_2", "money_tree", "chan_buy1", "chan_buy2", "chan_buy3"]:
        col = signal_type.replace("buy_point_", "buy")
        if col in df.columns: mask = df[col] == True
        else: return []
    else: return []
    return [{'index': i, 'date': df.loc[i, 'date'], 'price': df.loc[i, 'close']} for i in df[mask].index]

def calculate_trade_result(df, signal, hold_days=14):
    """计算交易结果"""
    buy_idx, buy_price = signal['index'], signal['price']
    sell_idx = min(buy_idx + hold_days, len(df) - 1)
    if sell_idx <= buy_idx or buy_price == 0 or pd.isna(buy_price): return None
    return_pct = (df.loc[sell_idx, 'close'] - buy_price) / buy_price * 100
    return {'buy_date': signal['date'], 'return': return_pct, 'win': return_pct > 0, 'year': signal['date'].year, 'month': signal['date'].month}

def process_stock_for_strategies(stock_file, strategies, hold_days=14):
    """处理单只股票的所有策略（用于并行）"""
    df = load_stock_data(stock_file)
    if df is None or len(df) < 30: return None
    try:
        df = calculate_all_indicators(df)
        results = defaultdict(list)
        for stype in strategies:
            signals = find_signals(df, stype)
            for sig in signals:
                res = calculate_trade_result(df, sig, hold_days)
                if res: results[stype].append(res)
        return dict(results)
    except: return None

def run_full_backtest():
    """主函数：并行执行全市场回测"""
    strategies = ['six_veins_6red', 'six_veins_5red', 'six_veins_4red', 'buy_point_1', 'buy_point_2', 'money_tree', 'chan_buy1', 'chan_buy2', 'chan_buy3']
    stock_files = list(DATA_DIR.rglob("*.csv"))
    print(f"开始全市场回测，共 {len(stock_files)} 只股票...")
    
    num_cores = cpu_count()
    worker = partial(process_stock_for_strategies, strategies=strategies, hold_days=14)
    
    with Pool(num_cores) as pool:
        all_results = pool.map(worker, stock_files)
    
    # 合并结果
    merged_trades = defaultdict(list)
    for res in all_results:
        if res:
            for stype, trades in res.items():
                merged_trades[stype].extend(trades)
    
    # 计算统计数据
    final_results = []
    for stype in strategies:
        trades = merged_trades[stype]
        if not trades: continue
        
        # 总体
        total_wins = sum(1 for t in trades if t['win'])
        total_returns = [t['return'] for t in trades]
        
        # 年度
        yearly = defaultdict(list)
        for t in trades: yearly[t['year']].append(t)
        yearly_stats = {str(y): {'trades': len(ts), 'win_rate': f"{sum(1 for t in ts if t['win'])/len(ts)*100:.1f}%", 'avg_return': f"{np.mean([t['return'] for t in ts]):.2f}%"} for y, ts in sorted(yearly.items())}
        
        # 月度
        monthly = defaultdict(list)
        for t in trades: monthly[t['month']].append(t)
        monthly_stats = {f"{m}月": {'trades': len(ts), 'win_rate': f"{sum(1 for t in ts if t['win'])/len(ts)*100:.1f}%", 'avg_return': f"{np.mean([t['return'] for t in ts]):.2f}%"} for m, ts in sorted(monthly.items())}
        
        final_results.append({
            'id': stype,
            'name': stype.replace('_', ' ').title(),
            'total': {
                'trades': len(trades),
                'win_rate': f"{total_wins/len(trades)*100:.1f}%",
                'avg_return': f"{np.mean(total_returns):.2f}%",
                'total_return': f"{np.sum(total_returns):.1f}%"
            },
            'yearly': yearly_stats,
            'monthly': monthly_stats
        })
    
    # 保存
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(BACKTEST_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"回测完成，结果已保存至 {BACKTEST_RESULTS_FILE}")

if __name__ == "__main__":
    import time
    start_time = time.time()
    run_full_backtest()
    print(f"总耗时: {time.time() - start_time:.2f} 秒")
