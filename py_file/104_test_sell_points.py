#!/usr/bin/env python3
"""
单指标最佳卖出点测试脚本
测试不同卖出条件（固定天数、盈利百分比、指标信号）对收益的影响
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

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
    
    df = pd.read_csv(file_path)
    # 确保日期列正确
    if '日期' in df.columns:
        df['date'] = pd.to_datetime(df['日期'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        return None
    
    df = df.sort_values('date').reset_index(drop=True)
    return df

def calculate_indicators(df):
    """计算所有指标"""
    # 确保列名正确
    if '收盘' in df.columns:
        df = df.rename(columns={
            '开盘': 'open', '收盘': 'close', '最高': 'high', 
            '最低': 'low', '成交量': 'volume'
        })
    
    # 计算六脉神剑
    df = calculate_six_veins(df)
    
    # 计算买卖点
    df = calculate_buy_sell_points(df)
    
    return df

def find_buy_signals(df, signal_type):
    """找到买入信号"""
    signals = []
    
    if signal_type == "six_veins_6red":
        # 六脉6红首次出现
        df['six_red'] = (df['six_veins_count'] == 6).astype(int)
        df['prev_six_red'] = df['six_red'].shift(1).fillna(0)
        buy_mask = (df['six_red'] == 1) & (df['prev_six_red'] == 0)
        
    elif signal_type == "six_veins_5red":
        # 六脉5红首次出现
        df['five_red'] = (df['six_veins_count'] >= 5).astype(int)
        df['prev_five_red'] = df['five_red'].shift(1).fillna(0)
        buy_mask = (df['five_red'] == 1) & (df['prev_five_red'] == 0)
        
    elif signal_type == "buy_point_2":
        # 买点2信号
        if 'buy_point_2' in df.columns:
            buy_mask = df['buy_point_2'] == 1
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

def test_sell_condition(df, buy_signal, sell_condition):
    """测试特定卖出条件的收益"""
    buy_idx = buy_signal['index']
    buy_price = buy_signal['price']
    buy_date = buy_signal['date']
    
    # 确保有足够的数据
    if buy_idx >= len(df) - 1:
        return None
    
    result = {
        'buy_date': buy_date,
        'buy_price': buy_price,
        'sell_date': None,
        'sell_price': None,
        'return': None,
        'hold_days': None,
        'win': None
    }
    
    if sell_condition['type'] == 'fixed_days':
        # 固定天数后卖出
        hold_days = sell_condition['days']
        sell_idx = min(buy_idx + hold_days, len(df) - 1)
        result['sell_date'] = df.loc[sell_idx, 'date']
        result['sell_price'] = df.loc[sell_idx, 'close']
        result['hold_days'] = sell_idx - buy_idx
        
    elif sell_condition['type'] == 'profit_target':
        # 盈利目标卖出
        target_pct = sell_condition['target']
        target_price = buy_price * (1 + target_pct / 100)
        
        # 查找达到目标的第一天
        for i in range(buy_idx + 1, min(buy_idx + 60, len(df))):  # 最多持有60天
            if df.loc[i, 'high'] >= target_price:
                result['sell_date'] = df.loc[i, 'date']
                result['sell_price'] = target_price
                result['hold_days'] = i - buy_idx
                break
        
        # 如果60天内未达到目标，按60天后价格卖出
        if result['sell_date'] is None:
            sell_idx = min(buy_idx + 60, len(df) - 1)
            result['sell_date'] = df.loc[sell_idx, 'date']
            result['sell_price'] = df.loc[sell_idx, 'close']
            result['hold_days'] = sell_idx - buy_idx
            
    elif sell_condition['type'] == 'signal_reverse':
        # 信号反转卖出（六脉转非6红）
        for i in range(buy_idx + 1, min(buy_idx + 60, len(df))):
            if df.loc[i, 'six_veins_count'] < 6:
                result['sell_date'] = df.loc[i, 'date']
                result['sell_price'] = df.loc[i, 'close']
                result['hold_days'] = i - buy_idx
                break
        
        if result['sell_date'] is None:
            sell_idx = min(buy_idx + 60, len(df) - 1)
            result['sell_date'] = df.loc[sell_idx, 'date']
            result['sell_price'] = df.loc[sell_idx, 'close']
            result['hold_days'] = sell_idx - buy_idx
    
    # 计算收益
    if result['sell_price'] is not None:
        result['return'] = (result['sell_price'] - buy_price) / buy_price * 100
        result['win'] = result['return'] > 0
    
    return result

def run_backtest_for_signal(signal_type, sell_conditions):
    """对特定信号类型运行回测"""
    results = {cond['name']: [] for cond in sell_conditions}
    
    # 获取所有股票文件
    stock_files = list(DATA_DIR.glob("*.csv"))
    
    print(f"\n测试信号类型: {signal_type}")
    print(f"共有 {len(stock_files)} 只股票")
    
    for stock_file in stock_files:
        stock_code = stock_file.stem
        df = load_stock_data(stock_code)
        
        if df is None or len(df) < 100:
            continue
        
        try:
            df = calculate_indicators(df)
        except Exception as e:
            continue
        
        # 找到买入信号
        buy_signals = find_buy_signals(df, signal_type)
        
        # 对每个买入信号测试不同卖出条件
        for signal in buy_signals:
            for cond in sell_conditions:
                result = test_sell_condition(df, signal, cond)
                if result and result['return'] is not None:
                    result['stock_code'] = stock_code
                    results[cond['name']].append(result)
    
    return results

def calculate_statistics(results):
    """计算统计数据"""
    stats = {}
    
    for cond_name, trades in results.items():
        if not trades:
            stats[cond_name] = {
                'trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0
            }
            continue
        
        wins = sum(1 for t in trades if t['win'])
        returns = [t['return'] for t in trades]
        
        stats[cond_name] = {
            'trades': len(trades),
            'win_rate': wins / len(trades) * 100 if trades else 0,
            'avg_return': np.mean(returns) if returns else 0,
            'total_return': np.sum(returns) if returns else 0,
            'max_return': np.max(returns) if returns else 0,
            'min_return': np.min(returns) if returns else 0,
            'avg_hold_days': np.mean([t['hold_days'] for t in trades]) if trades else 0
        }
    
    return stats

def main():
    """主函数"""
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 定义卖出条件
    sell_conditions = [
        {'name': '固定1天后卖出', 'type': 'fixed_days', 'days': 1},
        {'name': '固定3天后卖出', 'type': 'fixed_days', 'days': 3},
        {'name': '固定5天后卖出', 'type': 'fixed_days', 'days': 5},
        {'name': '固定10天后卖出', 'type': 'fixed_days', 'days': 10},
        {'name': '固定15天后卖出', 'type': 'fixed_days', 'days': 15},
        {'name': '固定20天后卖出', 'type': 'fixed_days', 'days': 20},
        {'name': '固定30天后卖出', 'type': 'fixed_days', 'days': 30},
        {'name': '盈利3%卖出', 'type': 'profit_target', 'target': 3},
        {'name': '盈利5%卖出', 'type': 'profit_target', 'target': 5},
        {'name': '盈利10%卖出', 'type': 'profit_target', 'target': 10},
        {'name': '六脉转非6红卖出', 'type': 'signal_reverse'},
    ]
    
    # 测试不同信号类型
    signal_types = ['six_veins_6red', 'six_veins_5red', 'buy_point_2']
    
    all_results = {}
    
    for signal_type in signal_types:
        print(f"\n{'='*60}")
        print(f"测试信号: {signal_type}")
        print('='*60)
        
        results = run_backtest_for_signal(signal_type, sell_conditions)
        stats = calculate_statistics(results)
        
        all_results[signal_type] = {
            'stats': stats,
            'raw_results': results
        }
        
        # 打印结果
        print(f"\n{signal_type} 回测结果:")
        print("-" * 80)
        print(f"{'卖出条件':<20} {'交易次数':>10} {'胜率':>10} {'平均收益':>12} {'平均持有天数':>12}")
        print("-" * 80)
        
        for cond_name, stat in stats.items():
            print(f"{cond_name:<20} {stat['trades']:>10} {stat['win_rate']:>9.1f}% {stat['avg_return']:>11.2f}% {stat['avg_hold_days']:>12.1f}")
    
    # 保存结果到JSON
    output_file = OUTPUT_DIR / "sell_point_test_results.json"
    
    # 转换为可序列化格式
    serializable_results = {}
    for signal_type, data in all_results.items():
        serializable_results[signal_type] = {
            'stats': data['stats'],
            'trade_count': {k: len(v) for k, v in data['raw_results'].items()}
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 生成最佳卖出点建议
    print("\n" + "="*60)
    print("最佳卖出点建议")
    print("="*60)
    
    for signal_type, data in all_results.items():
        stats = data['stats']
        # 找到胜率最高的卖出条件
        best_win_rate = max(stats.items(), key=lambda x: x[1]['win_rate'] if x[1]['trades'] > 5 else 0)
        # 找到平均收益最高的卖出条件
        best_return = max(stats.items(), key=lambda x: x[1]['avg_return'] if x[1]['trades'] > 5 else -999)
        
        print(f"\n{signal_type}:")
        print(f"  最高胜率: {best_win_rate[0]} ({best_win_rate[1]['win_rate']:.1f}%)")
        print(f"  最高收益: {best_return[0]} ({best_return[1]['avg_return']:.2f}%)")
    
    return all_results

if __name__ == "__main__":
    main()
