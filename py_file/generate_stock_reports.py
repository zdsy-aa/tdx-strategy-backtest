#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成股票报告明细数据 (stock_reports.json)

功能：
1. 遍历 data/day 目录下所有股票数据
2. 正确识别市场（沪市/深市/北交所）
3. 计算各维度的回测统计
4. 生成完整的报告明细数据
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from indicators import (
    calculate_six_veins, 
    calculate_buy_sell_points, 
    calculate_money_tree,
    calculate_chan_theory
)

# 路径配置
DATA_DIR = Path(__file__).parent.parent / "data" / "day"
WEB_DATA_DIR = Path(__file__).parent.parent / "web" / "client" / "src" / "data"

# 股票名称缓存
STOCK_NAMES = {}


def load_stock_names():
    """加载股票名称映射"""
    global STOCK_NAMES
    
    # 尝试从股票列表文件加载
    stock_list_file = Path(__file__).parent.parent / "data" / "stock_list.json"
    if stock_list_file.exists():
        try:
            with open(stock_list_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if 'code' in item and 'name' in item:
                        STOCK_NAMES[item['code']] = item['name']
            print(f"已加载 {len(STOCK_NAMES)} 个股票名称")
        except:
            pass


def get_stock_name(code, df=None):
    """获取股票名称"""
    # 优先从缓存获取
    if code in STOCK_NAMES:
        return STOCK_NAMES[code]
    
    # 从数据文件获取
    if df is not None and '名称' in df.columns:
        name = df['名称'].iloc[0]
        if pd.notna(name) and name != code:
            STOCK_NAMES[code] = name
            return name
    
    # 返回代码作为名称
    return code


def get_market_from_path(file_path):
    """根据文件路径获取市场"""
    path_str = str(file_path)
    if '/sh/' in path_str:
        return 'sh'
    elif '/sz/' in path_str:
        return 'sz'
    elif '/bj/' in path_str:
        return 'bj'
    return 'unknown'


def get_market_name(market):
    """获取市场中文名称"""
    names = {
        'sh': '沪市',
        'sz': '深市',
        'bj': '北交所',
        'unknown': '未知'
    }
    return names.get(market, '未知')


def load_stock_data(file_path):
    """加载股票数据"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # 确保日期列正确
        if '日期' in df.columns:
            df['date'] = pd.to_datetime(df['日期'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            return None
        
        # 确保列名正确
        if '收盘' in df.columns:
            df = df.rename(columns={
                '开盘': 'open', '收盘': 'close', '最高': 'high', 
                '最低': 'low', '成交量': 'volume'
            })
        
        df = df.sort_values('date').reset_index(drop=True)
        return df
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
        df['chan_buy1'] = False
        df['chan_buy2'] = False
        df['chan_buy3'] = False
    return df


def find_signals(df, signal_type):
    """找到买入信号"""
    signals = []
    
    if signal_type == 'six_veins_6red':
        mask = df['six_veins_count'] == 6
    elif signal_type == 'six_veins_5red':
        mask = df['six_veins_count'] == 5
    elif signal_type == 'six_veins_4red':
        mask = df['six_veins_count'] == 4
    elif signal_type == 'buy_point_1':
        mask = df.get('buy1', pd.Series([False]*len(df)))
    elif signal_type == 'buy_point_2':
        mask = df.get('buy2', pd.Series([False]*len(df)))
    elif signal_type == 'money_tree':
        mask = df.get('money_tree', pd.Series([False]*len(df)))
    elif signal_type == 'chan_buy1':
        mask = df.get('chan_buy1', pd.Series([False]*len(df)))
    elif signal_type == 'chan_buy2':
        mask = df.get('chan_buy2', pd.Series([False]*len(df)))
    elif signal_type == 'chan_buy3':
        mask = df.get('chan_buy3', pd.Series([False]*len(df)))
    else:
        return signals
    
    for idx in df[mask].index:
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
    
    return_pct = (sell_price - buy_price) / buy_price * 100
    
    return {
        'buy_date': signal['date'],
        'buy_price': buy_price,
        'sell_price': sell_price,
        'return': round(return_pct, 2),
        'win': return_pct > 0,
        'year': signal['date'].year,
        'month': signal['date'].month
    }


def generate_stock_reports(end_date='2026-01-09'):
    """生成股票报告明细"""
    print("="*60)
    print("开始生成股票报告明细")
    print("="*60)
    
    # 加载股票名称
    load_stock_names()
    
    end_dt = pd.to_datetime(end_date)
    year_start = pd.to_datetime(f"{end_dt.year}-01-01")
    month_start = pd.to_datetime(f"{end_dt.year}-{end_dt.month:02d}-01")
    
    stock_reports = []
    skipped_stocks = []  # 记录被跳过的股票
    
    # 遍历所有市场目录
    total_files = 0
    processed = 0
    success = 0
    
    for market in ['sh', 'sz', 'bj']:
        market_dir = DATA_DIR / market
        if not market_dir.exists():
            print(f"警告: 目录不存在 {market_dir}")
            continue
        
        stock_files = list(market_dir.glob('*.csv'))
        total_files += len(stock_files)
        print(f"\n处理 {get_market_name(market)} ({market}): {len(stock_files)} 个文件")
        
        for i, stock_file in enumerate(stock_files):
            processed += 1
            stock_code = stock_file.stem
            
            # 进度显示
            if processed % 500 == 0:
                print(f"  进度: {processed}/{total_files} ({processed*100//total_files}%)")
            
            # 加载数据
            df = load_stock_data(stock_file)
            if df is None:
                skipped_stocks.append({'code': stock_code, 'market': market, 'reason': '加载失败'})
                continue
            
            # 数据不足的股票记录但不处理
            if len(df) < 100:
                skipped_stocks.append({
                    'code': stock_code, 
                    'market': market, 
                    'reason': f'数据不足({len(df)}行)',
                    'rows': len(df)
                })
                continue
            
            # 计算指标
            try:
                df = calculate_all_indicators(df)
            except Exception as e:
                skipped_stocks.append({'code': stock_code, 'market': market, 'reason': f'指标计算失败: {str(e)}'})
                continue
            
            # 获取股票名称
            stock_name = get_stock_name(stock_code, df)
            
            # 找到所有买入信号
            all_signals = []
            for signal_type in ['six_veins_6red', 'six_veins_5red', 'six_veins_4red', 
                               'buy_point_1', 'buy_point_2', 'chan_buy1', 'chan_buy2', 'chan_buy3']:
                signals = find_signals(df, signal_type)
                all_signals.extend(signals)
            
            # 计算所有交易
            all_trades = []
            for signal in all_signals:
                result = calculate_trade_result(df, signal, 14)
                if result:
                    all_trades.append(result)
            
            # 总体统计
            total_trades = all_trades
            total_wins = sum(1 for t in total_trades if t['win']) if total_trades else 0
            total_returns = [t['return'] for t in total_trades] if total_trades else []
            
            # 年度统计
            year_trades = [t for t in all_trades if t['buy_date'] >= year_start]
            year_wins = sum(1 for t in year_trades if t['win']) if year_trades else 0
            year_returns = [t['return'] for t in year_trades] if year_trades else []
            
            # 月度统计
            month_trades = [t for t in all_trades if t['buy_date'] >= month_start]
            month_wins = sum(1 for t in month_trades if t['win']) if month_trades else 0
            month_returns = [t['return'] for t in month_trades] if month_trades else []
            
            # 最新信号
            df_recent = df[df['date'] <= end_dt].tail(10)
            last_signal = "无"
            last_signal_date = "-"
            
            if len(df_recent) > 0:
                for idx in df_recent.index[::-1]:
                    if df_recent.loc[idx, 'six_veins_count'] == 6:
                        last_signal = "六脉6红"
                        last_signal_date = df_recent.loc[idx, 'date'].strftime('%Y-%m-%d')
                        break
                    elif df_recent.loc[idx, 'six_veins_count'] >= 5:
                        last_signal = "六脉5红"
                        last_signal_date = df_recent.loc[idx, 'date'].strftime('%Y-%m-%d')
                        break
                    elif 'buy2' in df_recent.columns and df_recent.loc[idx, 'buy2']:
                        last_signal = "买点2"
                        last_signal_date = df_recent.loc[idx, 'date'].strftime('%Y-%m-%d')
                        break
                    elif 'buy1' in df_recent.columns and df_recent.loc[idx, 'buy1']:
                        last_signal = "买点1"
                        last_signal_date = df_recent.loc[idx, 'date'].strftime('%Y-%m-%d')
                        break
            
            # 构建报告
            report = {
                'code': stock_code,
                'name': stock_name,
                'market': market,
                'marketName': get_market_name(market),
                'totalReturn': f"{np.sum(total_returns):.1f}%" if total_returns else "0.0%",
                'yearReturn': f"{np.sum(year_returns):.1f}%" if year_returns else "0.0%",
                'monthReturn': f"{np.sum(month_returns):.1f}%" if month_returns else "0.0%",
                'totalWinRate': f"{total_wins/len(total_trades)*100:.1f}%" if total_trades else "0.0%",
                'yearWinRate': f"{year_wins/len(year_trades)*100:.1f}%" if year_trades else "0.0%",
                'monthWinRate': f"{month_wins/len(month_trades)*100:.1f}%" if month_trades else "0.0%",
                'totalTrades': len(total_trades),
                'yearTrades': len(year_trades),
                'monthTrades': len(month_trades),
                'lastSignal': last_signal,
                'lastSignalDate': last_signal_date,
                'dataRows': len(df)
            }
            
            stock_reports.append(report)
            success += 1
    
    print(f"\n{'='*60}")
    print(f"处理完成:")
    print(f"  总文件数: {total_files}")
    print(f"  成功处理: {success}")
    print(f"  跳过: {len(skipped_stocks)}")
    
    # 按市场统计
    market_stats = defaultdict(int)
    for report in stock_reports:
        market_stats[report['market']] += 1
    
    print(f"\n各市场统计:")
    for market, count in sorted(market_stats.items()):
        print(f"  {get_market_name(market)} ({market}): {count}")
    
    # 保存报告
    if not WEB_DATA_DIR.exists():
        WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    output_file = WEB_DATA_DIR / "stock_reports.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stock_reports, f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存到: {output_file}")
    
    # 保存跳过的股票列表
    skipped_file = WEB_DATA_DIR / "skipped_stocks.json"
    with open(skipped_file, 'w', encoding='utf-8') as f:
        json.dump(skipped_stocks, f, ensure_ascii=False, indent=2)
    
    print(f"跳过的股票列表: {skipped_file}")
    
    return stock_reports


if __name__ == "__main__":
    generate_stock_reports()
