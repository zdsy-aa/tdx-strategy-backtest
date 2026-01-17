#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成股票报告明细 (并行化版本)
功能：遍历所有股票，计算各策略表现，生成汇总 JSON 报告
"""

try:
    from a99_logger import log
except ImportError:
    def log(msg, level="INFO"): print(f"[{level}] {msg}")

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a99_indicators import calculate_six_veins, calculate_buy_sell_points, calculate_money_tree, calculate_chan_theory

# 配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "day"
WEB_DATA_DIR = PROJECT_ROOT / "web" / "client" / "src" / "data"
STOCK_REPORTS_FILE = WEB_DATA_DIR / "stock_reports.json"

# 缓存股票名称
STOCK_NAMES = {}

def load_stock_data(file_path):
    """加载股票数据"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        if df.empty:
            return None
            
        # 统一日期列
        if '日期' in df.columns:
            df['date'] = pd.to_datetime(df['日期'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            return None
            
        # 统一价格列名
        column_map = {
            '名称': 'name',
            '开盘': 'open', 
            '收盘': 'close', 
            '最高': 'high', 
            '最低': 'low', 
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover'
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        return df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        return None

def calculate_all_indicators(df):
    """计算所有指标"""
    df = calculate_six_veins(df)
    df = calculate_buy_sell_points(df)
    try:
        df = calculate_money_tree(df)
    except:
        df['money_tree'] = False
    try:
        df = calculate_chan_theory(df)
    except:
        df['chan_buy1'] = df['chan_buy2'] = df['chan_buy3'] = False
    return df

def find_signals(df, signal_type):
    """找到买入信号"""
    signals = []
    if df is None or df.empty:
        return signals
        
    if signal_type == "six_veins_6red":
        mask = (df['six_veins_count'] == 6) & (df['six_veins_count'].shift(1) < 6)
    elif signal_type == "six_veins_5red":
        mask = (df['six_veins_count'] >= 5) & (df['six_veins_count'].shift(1) < 5)
    elif signal_type == "six_veins_4red":
        mask = (df['six_veins_count'] >= 4) & (df['six_veins_count'].shift(1) < 4)
    elif signal_type in ["buy_point_1", "buy_point_2", "money_tree", "chan_buy1", "chan_buy2", "chan_buy3", "chan_strong_buy2", "chan_like_buy2"]:
        col = signal_type.replace("buy_point_", "buy")
        if col in df.columns:
            mask = df[col] == True
        else:
            return signals
    else:
        return signals
        
    buy_indices = df[mask].index.tolist()
    for idx in buy_indices:
        signals.append({
            'index': idx,
            'date': df.loc[idx, 'date'],
            'price': df.loc[idx, 'close']
        })
    return signals

def calculate_trade_result(df, signal, hold_days=14):
    """计算交易结果"""
    buy_idx = signal['index']
    sell_idx = min(buy_idx + hold_days, len(df) - 1)
    
    if sell_idx <= buy_idx:
        return None
    
    buy_price = signal['price']
    sell_price = df.loc[sell_idx, 'close']
    
    # 修复除零错误
    if buy_price == 0 or pd.isna(buy_price):
        return None
        
    return_pct = (sell_price - buy_price) / buy_price * 100
    
    return {
        'buy_date': signal['date'],
        'return': return_pct,
        'win': return_pct > 0
    }

def get_market_info(file_path):
    """从路径获取市场信息"""
    path_str = str(file_path)
    if '/sh/' in path_str: return 'sh', '沪市'
    if '/sz/' in path_str: return 'sz', '深市'
    if '/bj/' in path_str: return 'bj', '北交所'
    return 'unknown', '未知'

def process_single_stock(stock_file, end_dt, year_start, month_start):
    """处理单只股票的逻辑（用于并行）"""
    stock_code = stock_file.stem
    market, market_name = get_market_info(stock_file)
    
    df = load_stock_data(stock_file)
    if df is None or len(df) < 30:
        return None
        
    try:
        # 获取股票名称
        stock_name = df['名称'].iloc[0] if '名称' in df.columns else stock_code
        
        df = calculate_all_indicators(df)
        
        # 统计各策略信号（包含缠论5个买点）
        all_trades = []
        chan_trades = []  # 缠论买点交易
        
        # 基础策略
        for stype in ['six_veins_6red', 'six_veins_5red', 'six_veins_4red', 'buy_point_1', 'buy_point_2']:
            signals = find_signals(df, stype)
            for sig in signals:
                res = calculate_trade_result(df, sig, 14)
                if res:
                    all_trades.append(res)
        
        # 缠论5个买点
        for stype in ['chan_buy1', 'chan_buy2', 'chan_buy3', 'chan_strong_buy2', 'chan_like_buy2']:
            signals = find_signals(df, stype)
            for sig in signals:
                res = calculate_trade_result(df, sig, 14)
                if res:
                    res['signal_type'] = stype
                    chan_trades.append(res)
                    all_trades.append(res)
        
        if not all_trades:
            return None
            
        # 统计计算
        total_returns = [t['return'] for t in all_trades]
        total_wins = sum(1 for t in all_trades if t['win'])
        
        year_trades = [t for t in all_trades if t['buy_date'] >= year_start]
        year_returns = [t['return'] for t in year_trades]
        year_wins = sum(1 for t in year_trades if t['win'])
        
        month_trades = [t for t in all_trades if t['buy_date'] >= month_start]
        month_returns = [t['return'] for t in month_trades]
        month_wins = sum(1 for t in month_trades if t['win'])
        
        # 最新信号
        last_signal = "无"
        last_date = "-"
        df_recent = df.tail(5)
        for idx in df_recent.index[::-1]:
            if df_recent.loc[idx, 'six_veins_count'] == 6:
                last_signal, last_date = "六脉6红", df_recent.loc[idx, 'date'].strftime('%Y-%m-%d')
                break
            elif df_recent.loc[idx, 'six_veins_count'] >= 5:
                last_signal, last_date = "六脉5红", df_recent.loc[idx, 'date'].strftime('%Y-%m-%d')
                break
        
        return {
            'code': stock_code,
            'name': stock_name,
            'market': market,
            'marketName': market_name,
            'totalReturn': f"{np.sum(total_returns):.1f}%",
            'yearReturn': f"{np.sum(year_returns):.1f}%",
            'monthReturn': f"{np.sum(month_returns):.1f}%",
            'totalWinRate': f"{total_wins/len(all_trades)*100:.1f}%" if all_trades else "0.0%",
            'yearWinRate': f"{year_wins/len(year_trades)*100:.1f}%" if year_trades else "0.0%",
            'monthWinRate': f"{month_wins/len(month_trades)*100:.1f}%" if month_trades else "0.0%",
            'totalTrades': len(all_trades),
            'yearTrades': len(year_trades),
            'monthTrades': len(month_trades),
            'lastSignal': last_signal,
            'lastSignalDate': last_date
        }
    except Exception as e:
        return None

def generate_reports(end_date=None):
    """主函数：并行生成报告"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    end_dt = pd.to_datetime(end_date)
    year_start = pd.to_datetime(f"{end_dt.year}-01-01")
    month_start = pd.to_datetime(f"{end_dt.year}-{end_dt.month:02d}-01")
    
    log(f"开始生成报告，截止日期: {end_date}")
    stock_files = list(DATA_DIR.rglob("*.csv"))
    log(f"找到 {len(stock_files)} 只股票数据文件")
    
    # 使用进程池并行处理
    num_cores = 6
    log(f"使用 {num_cores} 个核心进行并行计算...")
    
    worker = partial(process_single_stock, end_dt=end_dt, year_start=year_start, month_start=month_start)
    
    with Pool(num_cores) as pool:
        results = pool.map(worker, stock_files)
    
    # 过滤掉 None 结果
    final_reports = [r for r in results if r is not None]
    
    # 保存结果
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(STOCK_REPORTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_reports, f, ensure_ascii=False, indent=2)
    
    log(f"报告生成完成，共计 {len(final_reports)} 只股票，已保存至 {STOCK_REPORTS_FILE}")

if __name__ == "__main__":
    import time
    start_time = time.time()
    generate_reports()
    log(f"总耗时: {time.time() - start_time:.2f} 秒")
