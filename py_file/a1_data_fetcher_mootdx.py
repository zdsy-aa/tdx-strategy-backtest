#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
基于 mootdx 的日线数据下载模块 (a1_data_fetcher_mootdx.py)
================================================================================

功能说明:
    本模块使用优化后的 mootdx 库从通达信行情服务器获取 A 股日线数据。
    相比 AKShare，通达信接口通常更稳定且速度更快。

数据来源:
    - 通达信行情服务器 (通过 mootdx 接口)

存储路径:
    - 日线数据: /data/day/{market}/{stock_code}.csv

================================================================================
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime
from mootdx.quotes import Quotes
from mootdx.logger import logger as mootdx_logger
import logging

# 禁用 mootdx 的调试日志，保持界面整洁
mootdx_logger.setLevel(logging.INFO)

# ==============================================================================
# 日志模块
# ==============================================================================
try:
    from a99_logger import log
except ImportError:
    def log(msg, level="INFO"):
        print(f"[{level}] {msg}")

# ==============================================================================
# 项目路径配置
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "day")

# ==============================================================================
# 字段映射 (mootdx -> backtest 格式)
# ==============================================================================
# backtest 要求的格式：名称, 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
# mootdx 返回的字段：open, close, high, low, vol, amount, datetime 等

def get_market_folder(stock_code: str) -> str:
    """根据股票代码判断市场"""
    if stock_code.startswith(("60", "68", "90")):
        return "sh"
    elif stock_code.startswith(("00", "30", "20")):
        return "sz"
    elif stock_code.startswith(("8", "4")):
        return "bj"
    return "sh"

class MootdxFetcher:
    def __init__(self):
        log("正在初始化 mootdx 行情接口 (自动选择最优服务器)...")
        try:
            self.client = Quotes.factory(market='std', multithread=True, heartbeat=True)
            log(f"成功连接至服务器: {self.client.server}")
        except Exception as e:
            log(f"初始化 mootdx 失败: {e}", level="ERROR")
            sys.exit(1)

    def fetch_daily(self, symbol: str, name: str, offset: int = 800):
        """获取单只股票的日线数据"""
        try:
            # frequency=9 为日线
            df = self.client.bars(symbol=symbol, frequency=9, offset=offset)
            if df is None or df.empty:
                return None
            
            # 转换日期格式 (mootdx 的 bars 返回的 index 通常就是 datetime)
            if 'datetime' not in df.columns:
                df = df.reset_index()
            
            df['日期'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')
            
            # 计算涨跌额和涨跌幅 (mootdx 原始数据不含这些，需要计算)
            df['涨跌额'] = df['close'].diff()
            df['涨跌幅'] = (df['涨跌额'] / df['close'].shift(1)) * 100
            
            # 计算振幅
            df['振幅'] = ((df['high'] - df['low']) / df['close'].shift(1)) * 100
            
            # 填充名称
            df['名称'] = name
            
            # 映射列名
            column_map = {
                'open': '开盘',
                'close': '收盘',
                'high': '最高',
                'low': '最低',
                'vol': '成交量',
                'amount': '成交额'
            }
            df = df.rename(columns=column_map)
            
            # 换手率在行情接口中通常不直接提供，设为 0 或从其他接口获取
            # 这里为了兼容性设为 0
            if '换手率' not in df.columns:
                df['换手率'] = 0.0
            
            # 整理列顺序
            expected_columns = ["名称", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
            
            # 过滤掉计算产生的 NaN (第一行)
            df = df.dropna(subset=['涨跌额'])
            
            return df[expected_columns]
        except Exception as e:
            log(f"获取 {symbol} 数据失败: {e}", level="ERROR")
            return None

    def save_data(self, symbol: str, df: pd.DataFrame):
        """保存数据到项目目录"""
        market = get_market_folder(symbol)
        market_dir = os.path.join(DATA_DIR, market)
        os.makedirs(market_dir, exist_ok=True)
        
        filepath = os.path.join(market_dir, f"{symbol}.csv")
        
        # 如果文件已存在，进行合并
        if os.path.exists(filepath):
            try:
                old_df = pd.read_csv(filepath)
                combined_df = pd.concat([old_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['日期'], keep='last')
                combined_df = combined_df.sort_values('日期')
                combined_df.to_csv(filepath, index=False, encoding="utf-8-sig")
            except Exception as e:
                log(f"合并 {symbol} 数据失败: {e}", level="ERROR")
                df.to_csv(filepath, index=False, encoding="utf-8-sig")
        else:
            df.to_csv(filepath, index=False, encoding="utf-8-sig")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="使用 mootdx 下载股票数据")
    parser.add_argument("--symbol", type=str, help="股票代码，如 600036")
    parser.add_argument("--name", type=str, default="未知", help="股票名称")
    parser.add_argument("--limit", type=int, default=100, help="获取的天数")
    args = parser.parse_args()

    if not args.symbol:
        log("请提供股票代码 --symbol", level="ERROR")
        return

    fetcher = MootdxFetcher()
    df = fetcher.fetch_daily(args.symbol, args.name, offset=args.limit)
    
    if df is not None:
        fetcher.save_data(args.symbol, df)
        log(f"成功下载并保存 {args.symbol} ({args.name}) 的数据，共 {len(df)} 条。")
    else:
        log(f"未能获取 {args.symbol} 的数据。", level="ERROR")

if __name__ == "__main__":
    main()
