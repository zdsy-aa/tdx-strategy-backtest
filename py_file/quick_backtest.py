#!/usr/bin/env python3
"""
快速回测脚本 - 优化版本
使用向量化操作和更高效的数据处理
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

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import calculate_six_veins, calculate_buy_sell_points

# 数据目录
DATA_DIR = Path(__file__).parent.parent / "data" / "day"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "backtest_results"

def load_stock_data(stock_code):
    """加载股票数据"""
    file_path = DATA_DIR / f"{stock_code}.csv"
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        # 确保日期列正确
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif '日期' in df.columns:
            df['date'] = pd.to_datetime(df['日期'])
        else:
            return None
        
        # 确保列名正确
        if '收盘' in df.columns:
            df = df.rename(columns={
                '开盘': 'open', '收盘': 'close', '最高': 'high', 
                '最低': 'low', '成交量': 'volume'
            })
        
        # 过滤掉无效数据
        df = df[df['close'] > 0].copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # 只使用2015年以后的数据进行快速回测
        df = df[df['date'] >= '2015-01-01'].reset_index(drop=True)
        
        return df
    except Exception as e:
        return None

def calculate_all_indicators(df):
    """计算所有指标"""
    try:
        df = calculate_six_veins(df)
        df = calculate_buy_sell_points(df)
    except Exception as e:
        pass
    return df

def vectorized_backtest(df, signal_col, hold_days=14):
    """向量化回测"""
    if signal_col not in df.columns:
        return []
    
    # 找到信号点
    signals = df[df[signal_col] == True].copy()
    if len(signals) == 0:
        return []
    
    trades = []
    for idx in signals.index:
        buy_price = df.loc[idx, 'close']
        buy_date = df.loc[idx, 'date']
        
        if buy_price <= 0 or pd.isna(buy_price):
            continue
        
        sell_idx = min(idx + hold_days, len(df) - 1)
        if sell_idx <= idx:
            continue
            
        sell_price = df.loc[sell_idx, 'close']
        sell_date = df.loc[sell_idx, 'date']
        
        if sell_price <= 0 or pd.isna(sell_price):
            continue
        
        return_pct = (sell_price - buy_price) / buy_price * 100
        
        trades.append({
            'buy_date': buy_date,
            'buy_price': buy_price,
            'sell_date': sell_date,
            'sell_price': sell_price,
            'return': return_pct,
            'hold_days': sell_idx - idx,
            'win': return_pct > 0,
            'year': buy_date.year,
            'month': buy_date.month
        })
    
    return trades

def run_backtest_for_signal(signal_type, hold_days=14):
    """运行特定信号的回测"""
    all_trades = []
    
    stock_files = list(DATA_DIR.glob("*.csv"))
    processed = 0
    
    for stock_file in stock_files:
        stock_code = stock_file.stem
        df = load_stock_data(stock_code)
        
        if df is None or len(df) < 100:
            continue
        
        try:
            df = calculate_all_indicators(df)
        except:
            continue
        
        # 根据信号类型创建信号列
        if signal_type == "six_veins_6red":
            df['signal'] = (df['six_veins_count'] == 6) & (df['six_veins_count'].shift(1) != 6)
        elif signal_type == "six_veins_5red":
            df['signal'] = (df['six_veins_count'] >= 5) & (df['six_veins_count'].shift(1) < 5)
        elif signal_type == "six_veins_4red":
            df['signal'] = (df['six_veins_count'] >= 4) & (df['six_veins_count'].shift(1) < 4)
        elif signal_type == "buy_point_1":
            df['signal'] = df.get('buy1', False)
        elif signal_type == "buy_point_2":
            df['signal'] = df.get('buy2', False)
        else:
            continue
        
        trades = vectorized_backtest(df, 'signal', hold_days)
        for t in trades:
            t['stock_code'] = stock_code
        all_trades.extend(trades)
        
        processed += 1
        if processed % 50 == 0:
            print(f"  已处理 {processed} 只股票...")
    
    return all_trades

def calculate_statistics(trades):
    """计算统计数据"""
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

def generate_stock_report(end_date="2025-01-06"):
    """生成股票报告明细"""
    end_dt = pd.to_datetime(end_date)
    year_start = pd.to_datetime(f"{end_dt.year}-01-01")
    month_start = pd.to_datetime(f"{end_dt.year}-{end_dt.month:02d}-01")
    
    stock_reports = []
    stock_files = list(DATA_DIR.glob("*.csv"))
    
    print(f"正在生成 {len(stock_files)} 只股票的报告明细...")
    
    for i, stock_file in enumerate(stock_files):
        stock_code = stock_file.stem
        df = load_stock_data(stock_code)
        
        if df is None or len(df) < 50:
            continue
        
        try:
            df = calculate_all_indicators(df)
        except:
            continue
        
        # 获取股票名称
        stock_name = stock_code
        
        # 创建信号
        df['signal'] = (df['six_veins_count'] >= 5) & (df['six_veins_count'].shift(1) < 5)
        
        # 计算所有交易
        all_trades = vectorized_backtest(df, 'signal', 14)
        
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
        
        report = {
            'code': stock_code,
            'name': stock_name,
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
        
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(stock_files)} 只股票...")
    
    return stock_reports

def test_sell_points():
    """测试不同卖出点的效果"""
    print("\n测试不同卖出点的效果...")
    
    sell_conditions = [
        {'name': '1天后卖出', 'days': 1},
        {'name': '3天后卖出', 'days': 3},
        {'name': '5天后卖出', 'days': 5},
        {'name': '10天后卖出', 'days': 10},
        {'name': '14天后卖出', 'days': 14},
        {'name': '20天后卖出', 'days': 20},
        {'name': '30天后卖出', 'days': 30},
    ]
    
    results = {}
    
    for cond in sell_conditions:
        print(f"  测试 {cond['name']}...")
        trades = run_backtest_for_signal('six_veins_5red', cond['days'])
        stats = calculate_statistics(trades)
        results[cond['name']] = stats['total']
    
    return results

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
    ]
    
    all_results = {}
    
    print("="*80)
    print("开始快速回测...")
    print(f"数据目录: {DATA_DIR}")
    print(f"股票数量: {len(list(DATA_DIR.glob('*.csv')))}")
    print("="*80)
    
    for signal_id, signal_name, default_hold in signal_types:
        print(f"\n正在测试: {signal_name} ({signal_id})")
        
        # 运行回测
        trades = run_backtest_for_signal(signal_id, default_hold)
        
        if not trades:
            print(f"  {signal_name}: 无交易信号")
            continue
        
        # 计算统计
        stats = calculate_statistics(trades)
        
        all_results[signal_id] = {
            'name': signal_name,
            'stats': stats,
            'default_hold_days': default_hold
        }
        
        print(f"  总交易: {stats['total']['trades']}")
        print(f"  总胜率: {stats['total']['win_rate']}%")
        print(f"  平均收益: {stats['total']['avg_return']}%")
    
    # 测试卖出点
    sell_point_results = test_sell_points()
    
    # 保存回测结果
    output_file = OUTPUT_DIR / "backtest_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'strategies': all_results,
            'sell_points': sell_point_results
        }, f, ensure_ascii=False, indent=2, default=str)
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
    
    print(f"\n{'策略名称':<15} {'总交易':>8} {'总胜率':>10} {'平均收益':>10}")
    print("-"*50)
    
    for signal_id, data in all_results.items():
        stats = data['stats']['total']
        print(f"{data['name']:<15} {stats['trades']:>8} {stats['win_rate']:>9.1f}% {stats['avg_return']:>9.2f}%")
    
    # 打印卖出点测试结果
    print("\n" + "="*80)
    print("卖出点测试结果 (六脉5红信号)")
    print("="*80)
    
    print(f"\n{'卖出条件':<15} {'交易次数':>10} {'胜率':>10} {'平均收益':>10}")
    print("-"*50)
    
    for cond_name, stats in sell_point_results.items():
        print(f"{cond_name:<15} {stats['trades']:>10} {stats['win_rate']:>9.1f}% {stats['avg_return']:>9.2f}%")
    
    # 生成数据下载报告
    print("\n" + "="*80)
    print("数据下载报告")
    print("="*80)
    
    stock_files = list(DATA_DIR.glob("*.csv"))
    if stock_files:
        # 获取数据时间范围
        sample_df = pd.read_csv(stock_files[0])
        if 'date' in sample_df.columns:
            dates = pd.to_datetime(sample_df['date'])
            print(f"数据时间范围: {dates.min().strftime('%Y-%m-%d')} 至 {dates.max().strftime('%Y-%m-%d')}")
        print(f"股票数量: {len(stock_files)}")
    
    return all_results, stock_reports

if __name__ == "__main__":
    main()
