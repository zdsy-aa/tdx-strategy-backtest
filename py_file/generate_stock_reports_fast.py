#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速生成股票报告明细数据 (stock_reports.json)

简化版：只生成基本信息和简单统计，不计算复杂指标
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# 路径配置
DATA_DIR = Path(__file__).parent.parent / "data" / "day"
WEB_DATA_DIR = Path(__file__).parent.parent / "web" / "client" / "src" / "data"


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


def calculate_simple_stats(df, end_date):
    """计算简单统计（不依赖复杂指标）"""
    if df is None or len(df) < 10:
        return None
    
    end_dt = pd.to_datetime(end_date)
    
    # 筛选有效数据
    df = df[df['date'] <= end_dt].copy()
    if len(df) < 10:
        return None
    
    # 计算收益率
    df['return'] = df['close'].pct_change() * 100
    
    # 总体统计（最近一年）
    year_start = end_dt - pd.DateOffset(years=1)
    df_year = df[df['date'] >= year_start]
    
    # 月度统计（最近一个月）
    month_start = end_dt - pd.DateOffset(months=1)
    df_month = df[df['date'] >= month_start]
    
    # 计算胜率（收益为正的天数比例）
    total_wins = (df['return'] > 0).sum()
    total_days = len(df[df['return'].notna()])
    
    year_wins = (df_year['return'] > 0).sum() if len(df_year) > 0 else 0
    year_days = len(df_year[df_year['return'].notna()]) if len(df_year) > 0 else 0
    
    month_wins = (df_month['return'] > 0).sum() if len(df_month) > 0 else 0
    month_days = len(df_month[df_month['return'].notna()]) if len(df_month) > 0 else 0
    
    # 计算累计收益
    total_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100 if len(df) > 1 else 0
    year_return = ((df_year['close'].iloc[-1] / df_year['close'].iloc[0]) - 1) * 100 if len(df_year) > 1 else 0
    month_return = ((df_month['close'].iloc[-1] / df_month['close'].iloc[0]) - 1) * 100 if len(df_month) > 1 else 0
    
    return {
        'totalReturn': f"{total_return:.1f}%",
        'yearReturn': f"{year_return:.1f}%",
        'monthReturn': f"{month_return:.1f}%",
        'totalWinRate': f"{total_wins/total_days*100:.1f}%" if total_days > 0 else "0.0%",
        'yearWinRate': f"{year_wins/year_days*100:.1f}%" if year_days > 0 else "0.0%",
        'monthWinRate': f"{month_wins/month_days*100:.1f}%" if month_days > 0 else "0.0%",
        'totalTrades': total_days,
        'yearTrades': year_days,
        'monthTrades': month_days,
        'lastDate': df['date'].iloc[-1].strftime('%Y-%m-%d'),
        'dataRows': len(df)
    }


def generate_stock_reports(end_date='2026-01-09'):
    """生成股票报告明细"""
    print("="*60)
    print("快速生成股票报告明细")
    print("="*60)
    
    stock_reports = []
    skipped_stocks = []
    
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
            
            # 数据不足的股票记录
            if len(df) < 100:
                skipped_stocks.append({
                    'code': stock_code, 
                    'market': market, 
                    'reason': f'数据不足({len(df)}行)',
                    'rows': len(df)
                })
                continue
            
            # 计算统计
            stats = calculate_simple_stats(df, end_date)
            if stats is None:
                skipped_stocks.append({'code': stock_code, 'market': market, 'reason': '统计计算失败'})
                continue
            
            # 获取股票名称（优先从 CSV 中读取，否则使用代码）
            if '名称' in df.columns and pd.notna(df['名称'].iloc[0]):
                stock_name = df['名称'].iloc[0]
            else:
                # 使用市场前缀 + 代码作为临时名称
                stock_name = f"{get_market_name(market)}{stock_code}"
            
            # 构建报告
            report = {
                'code': stock_code,
                'name': stock_name,
                'market': market,
                'marketName': get_market_name(market),
                'totalReturn': stats['totalReturn'],
                'yearReturn': stats['yearReturn'],
                'monthReturn': stats['monthReturn'],
                'totalWinRate': stats['totalWinRate'],
                'yearWinRate': stats['yearWinRate'],
                'monthWinRate': stats['monthWinRate'],
                'totalTrades': stats['totalTrades'],
                'yearTrades': stats['yearTrades'],
                'monthTrades': stats['monthTrades'],
                'lastSignal': '无',
                'lastSignalDate': stats['lastDate'],
                'dataRows': stats['dataRows']
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
    
    # 更新 backtest_results.json 的 last_update 字段
    backtest_file = WEB_DATA_DIR / "backtest_results.json"
    if backtest_file.exists():
        try:
            with open(backtest_file, 'r', encoding='utf-8') as f:
                backtest_data = json.load(f)
            backtest_data['last_update'] = datetime.now().strftime('%Y-%m-%d')
            with open(backtest_file, 'w', encoding='utf-8') as f:
                json.dump(backtest_data, f, ensure_ascii=False, indent=2)
            print(f"已更新 backtest_results.json 的 last_update 字段")
        except:
            pass
    
    return stock_reports


if __name__ == "__main__":
    generate_stock_reports()
