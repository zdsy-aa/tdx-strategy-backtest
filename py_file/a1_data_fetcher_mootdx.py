# -*- coding: utf-8 -*-
"""
================================================================================
基于 mootdx 的日线数据下载模块 (a1_data_fetcher_mootdx.py)
================================================================================

功能说明:
    本模块使用优化后的 mootdx 库从通达信行情服务器获取 A 股日线数据。
    支持自动加载项目 external 目录下的 mootdx 和 tdxpy 源码。

================================================================================
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime
import logging
import argparse

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
            
    # 2. 加载 external/tdxpy (支持离线环境)
    local_tdxpy_path = os.path.join(base_dir, "external")
    if os.path.exists(os.path.join(local_tdxpy_path, "tdxpy")):
        if local_tdxpy_path not in sys.path:
            sys.path.insert(0, local_tdxpy_path)
            
    # 3. 加载 py_file (为了 a99_logger 等)
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
    print(f"[ERROR] 导入失败: {e}")
    print("[INFO] 请确保项目 external 目录下存在 mootdx 和 tdxpy 源码")
    sys.exit(1)

def log(msg, level="INFO"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [{level}] {msg}")

class MootdxFetcher:
    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        log("正在初始化 mootdx 行情接口 (自动选择最优服务器)...")
        try:
            self.client = Quotes.factory(market='std')
            log(f"成功连接至服务器: {self.client.bestip}")
        except Exception as e:
            log(f"初始化失败: {e}", level="ERROR")
            self.client = None

    def fetch_daily(self, symbol, name="未知", offset=100):
        if not self.client:
            return None
        
        try:
            df = self.client.bars(symbol=symbol, frequency='day', offset=offset)
            if df is not None and not df.empty:
                # 如果 datetime 已经在列中，不要 reset_index() 否则会冲突
                if 'datetime' not in df.columns:
                    df = df.reset_index()
                # 统一列名
                df['名称'] = name
                return df
            return None
        except Exception as e:
            log(f"获取 {symbol} 数据失败: {e}", level="ERROR")
            return None

    def save_data(self, symbol, df):
        if df is None or df.empty:
            return
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        
        # 根据代码判断市场
        market = "sh" if symbol.startswith(('6', '9')) else "sz"
        save_dir = os.path.join(base_dir, "data", "day", market)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        file_path = os.path.join(save_dir, f"{symbol}.csv")
        df.to_csv(file_path, index=False, encoding='utf-8-sig')

def main():
    parser = argparse.ArgumentParser(description="Mootdx 数据抓取工具")
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
