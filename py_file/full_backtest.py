#!/usr/bin/env python3
"""
完整回测脚本
包含总数据、年度数据、月度数据的胜率和收益统计
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import calculate_six_veins, calculate_buy_sell_points, calculate_money_tree

# 数据目录
DATA_DIR = Path(__file__).parent.parent / "data" / "day"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "backtest_results"

def load_stock_data(stock_code):
    """加载股票数据"""
    file_path = DATA_DIR / f"{stock_code}.csv"
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path)
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

def calculate_all_indicators(df):
    """计算所有指标"""
    df = calculate_six_veins(df)
    df = calculate_buy_sell_points(df)
    try:
        df = calculate_money_tree(df)
    except:
        df['money_tree'] = False
    return df

def find_signals(df, signal_type):
    """找到买入信号"""
    signals = []
    
    if signal_type == "six_veins_6red":
        df['signal'] = (df['six_veins_count'] == 6).astype(int)
        df['prev_signal'] = df['signal'].shift(1).fillna(0)
        buy_mask = (df['signal'] == 1) & (df['prev_signal'] == 0)
        
    elif signal_type == "six_veins_5red":
        df['signal'] = (df['six_veins_count'] >= 5).astype(int)
        df['prev_signal'] = df['signal'].shift(1).fillna(0)
        buy_mask = (df['signal'] == 1) & (df['prev_signal'] == 0)
        
    elif signal_type == "six_veins_4red":
        df['signal'] = (df['six_veins_count'] >= 4).astype(int)
        df['prev_signal'] = df['signal'].shift(1).fillna(0)
        buy_mask = (df['signal'] == 1) & (df['prev_signal'] == 0)
        
    elif signal_type == "buy_point_1":
        if 'buy1' in df.columns:
            buy_mask = df['buy1'] == True
        else:
            return signals
            
    elif signal_type == "buy_point_2":
        if 'buy2' in df.columns:
            buy_mask = df['buy2'] == True
        else:
            return signals
            
    elif signal_type == "money_tree":
        if 'money_tree' in df.columns:
            buy_mask = df['money_tree'] == True
        else:
            return signals
    else:
        return signals
    
    buy_indices = df[buy_mask].index.tolist()
    for idx in buy_indices:
        signals.append({
            'index': idx,
            'date': df.loc[idx, 'date'],
            'price': df.loc[idx, 'close']
        })
    
    return signals

def calculate_trade_result(df, buy_signal, hold_days=14):
    """计算交易结果"""
    buy_idx = buy_signal['index']
    buy_price = buy_signal['price']
    buy_date = buy_signal['date']
    
    if buy_idx >= len(df) - 1:
        return None
    
    # 跳过无效价格
    if buy_price is None or buy_price == 0 or pd.isna(buy_price):
        return None
    
    sell_idx = min(buy_idx + hold_days, len(df) - 1)
    sell_price = df.loc[sell_idx, 'close']
    sell_date = df.loc[sell_idx, 'date']
    
    # 跳过无效卖出价格
    if sell_price is None or sell_price == 0 or pd.isna(sell_price):
        return None
    
    return_pct = (sell_price - buy_price) / buy_price * 100
    
    return {
        'buy_date': buy_date,
        'buy_price': buy_price,
        'sell_date': sell_date,
        'sell_price': sell_price,
        'return': return_pct,
        'hold_days': sell_idx - buy_idx,
        'win': return_pct > 0,
        'year': buy_date.year,
        'month': buy_date.month
    }

def run_backtest(signal_type, hold_days=14):
    """运行回测"""
    all_trades = []
    
    stock_files = list(DATA_DIR.glob("*.csv"))
    
    for stock_file in stock_files:
        stock_code = stock_file.stem
        df = load_stock_data(stock_code)
        
        if df is None or len(df) < 100:
            continue
        
        try:
            df = calculate_all_indicators(df)
        except Exception as e:
            continue
        
        signals = find_signals(df, signal_type)
        
        for signal in signals:
            result = calculate_trade_result(df, signal, hold_days)
            if result:
                result['stock_code'] = stock_code
                all_trades.append(result)
    
    return all_trades

def calculate_statistics(trades):
    """计算统计数据，区分总/年/月"""
    if not trades:
        return {
            'total': {'trades': 0, 'win_rate': 0, 'avg_return': 0},
            'yearly': {},
            'monthly': {}
        }
    
    # 总体统计
    wins = sum(1 for t in trades if t['win'])
    returns = [t['return'] for t in trades]
    
    total_stats = {
        'trades': len(trades),
        'win_rate': round(wins / len(trades) * 100, 1) if trades else 0,
        'avg_return': round(np.mean(returns), 2) if returns else 0,
        'total_return': round(np.sum(returns), 2) if returns else 0,
        'max_return': round(np.max(returns), 2) if returns else 0,
        'min_return': round(np.min(returns), 2) if returns else 0
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
            'win_rate': round(wins / len(year_trades) * 100, 1) if year_trades else 0,
            'avg_return': round(np.mean(returns), 2) if returns else 0
        }
    
    # 月度统计
    monthly_stats = {}
    trades_by_month = defaultdict(list)
    for t in trades:
        trades_by_month[t['month']].append(t)
    
    for month, month_trades in sorted(trades_by_month.items()):
        wins = sum(1 for t in month_trades if t['win'])
        returns = [t['return'] for t in month_trades]
        month_name = f"{month}月"
        monthly_stats[month_name] = {
            'trades': len(month_trades),
            'win_rate': round(wins / len(month_trades) * 100, 1) if month_trades else 0,
            'avg_return': round(np.mean(returns), 2) if returns else 0
        }
    
    return {
        'total': total_stats,
        'yearly': yearly_stats,
        'monthly': monthly_stats
    }

def find_optimal_hold_period(signal_type, max_days=30):
    """找到最优持有周期"""
    best_win_rate = 0
    best_return = -999
    optimal_days_win = 1
    optimal_days_return = 1
    
    for days in [1, 2, 3, 5, 7, 10, 14, 20, 30]:
        if days > max_days:
            break
        trades = run_backtest(signal_type, days)
        if not trades:
            continue
        
        wins = sum(1 for t in trades if t['win'])
        win_rate = wins / len(trades) * 100 if trades else 0
        avg_return = np.mean([t['return'] for t in trades]) if trades else 0
        
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            optimal_days_win = days
        
        if avg_return > best_return:
            best_return = avg_return
            optimal_days_return = days
    
    return optimal_days_win, optimal_days_return

def generate_stock_report(end_date="2025-01-06"):
    """生成股票报告明细"""
    end_dt = pd.to_datetime(end_date)
    year_start = pd.to_datetime(f"{end_dt.year}-01-01")
    month_start = pd.to_datetime(f"{end_dt.year}-{end_dt.month:02d}-01")
    
    stock_reports = []
    stock_files = list(DATA_DIR.glob("*.csv"))
    
    for stock_file in stock_files:
        stock_code = stock_file.stem
        df = load_stock_data(stock_code)
        
        if df is None or len(df) < 50:
            continue
        
        try:
            df = calculate_all_indicators(df)
        except:
            continue
        
        # 获取股票名称（从数据中或使用代码）
        stock_name = df.get('名称', pd.Series([stock_code])).iloc[0] if '名称' in df.columns else stock_code
        
        # 找到所有买入信号
        all_signals = []
        for signal_type in ['six_veins_6red', 'six_veins_5red', 'buy_point_2']:
            signals = find_signals(df, signal_type)
            all_signals.extend(signals)
        
        if not all_signals:
            continue
        
        # 计算所有交易
        all_trades = []
        for signal in all_signals:
            result = calculate_trade_result(df, signal, 14)
            if result:
                all_trades.append(result)
        
        if not all_trades:
            continue
        
        # 总体统计
        total_trades = all_trades
        total_wins = sum(1 for t in total_trades if t['win'])
        total_returns = [t['return'] for t in total_trades]
        
        # 年度统计
        year_trades = [t for t in all_trades if t['buy_date'] >= year_start]
        year_wins = sum(1 for t in year_trades if t['win'])
        year_returns = [t['return'] for t in year_trades]
        
        # 月度统计
        month_trades = [t for t in all_trades if t['buy_date'] >= month_start]
        month_wins = sum(1 for t in month_trades if t['win'])
        month_returns = [t['return'] for t in month_trades]
        
        # 最新信号
        df_recent = df[df['date'] <= end_dt].tail(5)
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
        
        report = {
            'code': stock_code,
            'name': str(stock_name),
            'totalReturn': f"{sum(total_returns):.1f}%" if total_returns else "0%",
            'yearReturn': f"{sum(year_returns):.1f}%" if year_returns else "0%",
            'monthReturn': f"{sum(month_returns):.1f}%" if month_returns else "0%",
            'totalWinRate': f"{total_wins/len(total_trades)*100:.1f}%" if total_trades else "0%",
            'yearWinRate': f"{year_wins/len(year_trades)*100:.1f}%" if year_trades else "0%",
            'monthWinRate': f"{month_wins/len(month_trades)*100:.1f}%" if month_trades else "0%",
            'totalTrades': len(total_trades),
            'yearTrades': len(year_trades),
            'monthTrades': len(month_trades),
            'lastSignal': last_signal,
            'lastSignalDate': last_signal_date
        }
        
        stock_reports.append(report)
    
    return stock_reports

def main():
    """主函数"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 定义要测试的信号类型
    signal_types = [
        ('six_veins_6red', '六脉6红', 14),
        ('six_veins_5red', '六脉5红', 14),
        ('six_veins_4red', '六脉4红', 10),
        ('buy_point_1', '买点1', 20),
        ('buy_point_2', '买点2', 16),
        ('money_tree', '摇钱树', 7),
    ]
    
    all_results = {}
    
    print("="*80)
    print("开始完整回测...")
    print("="*80)
    
    for signal_id, signal_name, default_hold in signal_types:
        print(f"\n正在测试: {signal_name} ({signal_id})")
        
        # 运行回测
        trades = run_backtest(signal_id, default_hold)
        
        if not trades:
            print(f"  {signal_name}: 无交易信号")
            continue
        
        # 计算统计
        stats = calculate_statistics(trades)
        
        # 找到最优持有周期
        optimal_win, optimal_return = find_optimal_hold_period(signal_id)
        
        all_results[signal_id] = {
            'name': signal_name,
            'stats': stats,
            'optimal_period_win': optimal_win,
            'optimal_period_return': optimal_return,
            'default_hold_days': default_hold
        }
        
        print(f"  总交易: {stats['total']['trades']}")
        print(f"  总胜率: {stats['total']['win_rate']}%")
        print(f"  平均收益: {stats['total']['avg_return']}%")
        print(f"  最优持有周期(胜率): {optimal_win}天")
    
    # 保存回测结果
    output_file = OUTPUT_DIR / "full_backtest_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n回测结果已保存到: {output_file}")
    
    # 生成股票报告明细
    print("\n正在生成股票报告明细...")
    stock_reports = generate_stock_report()
    
    report_file = OUTPUT_DIR / "stock_reports.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(stock_reports, f, ensure_ascii=False, indent=2)
    print(f"股票报告已保存到: {report_file}")
    print(f"共生成 {len(stock_reports)} 只股票的报告")
    
    # 打印汇总
    print("\n" + "="*80)
    print("回测汇总")
    print("="*80)
    
    print(f"\n{'策略名称':<15} {'总交易':>8} {'总胜率':>10} {'平均收益':>10} {'最优周期':>10}")
    print("-"*60)
    
    for signal_id, data in all_results.items():
        stats = data['stats']['total']
        print(f"{data['name']:<15} {stats['trades']:>8} {stats['win_rate']:>9.1f}% {stats['avg_return']:>9.2f}% {data['optimal_period_win']:>8}天")
    
    return all_results, stock_reports

if __name__ == "__main__":
    main()
