#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
基于 mootdx 的日线数据下载模块 (a1_data_fetcher_mootdx.py)
================================================================================

功能说明:
    本模块使用优化后的 mootdx 库从通达信行情服务器获取 A 股日线数据。
    支持自动加载项目 external 目录下的 mootdx 源码。

================================================================================
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime
import logging

# ==============================================================================
# 1. 核心：优先加载本地源码逻辑
# ==============================================================================
def load_local_modules():
    """
    自动检测并加载项目内置的模块路径
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    
    # 1. 加载 external/mootdx
    local_mootdx_path = os.path.join(base_dir, "external", "mootdx")
    if os.path.exists(local_mootdx_path):
        if local_mootdx_path not in sys.path:
            sys.path.insert(0, local_mootdx_path)
            
    # 2. 加载 py_file (为了 a99_logger 等)
    if current_dir not in sys.path:
        sys.path.append(current_dir)

load_local_modules()

# 现在可以安全导入了
try:
    from mootdx.quotes import Quotes
    from mootdx.logger import logger as mootdx_logger
    # 禁用 mootdx 的调试日志
    mootdx_logger.setLevel(logging.INFO)
except ImportError as e:
    print(f"[ERROR] 导入 mootdx 失败: {e}")
    print("[INFO] 请确保项目 external/mootdx 目录下存在源码，或执行 pip install mootdx")
    sys.exit(1)

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
            df = self.client.bars(symbol=symbol, frequency=9, offset=offset)
            if df is None or df.empty:
                return None
            
            if 'datetime' not in df.columns:
                df = df.reset_index()
            
            df['日期'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')
            df['涨跌额'] = df['close'].diff()
            df['涨跌幅'] = (df['涨跌额'] / df['close'].shift(1)) * 100
            df['振幅'] = ((df['high'] - df['low']) / df['close'].shift(1)) * 100
            df['名称'] = name
            
            column_map = {
                'open': '开盘',
                'close': '收盘',
                'high': '最高',
                'low': '最低',
                'vol': '成交量',
                'amount': '成交额'
            }
            df = df.rename(columns=column_map)
            
            if '换手率' not in df.columns:
                df['换手率'] = 0.0
            
            expected_columns = ["名称", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
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
