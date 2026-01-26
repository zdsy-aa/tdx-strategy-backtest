# -*- coding: utf-8 -*-
"""
================================================================================
基于 mootdx 的日线数据下载模块 (a1_data_fetcher_mootdx.py)
================================================================================

功能说明:
    本模块使用优化后的 mootdx 库从通达信行情服务器获取 A 股日线数据。
    支持自动加载项目 external 目录下的所有依赖源码（mootdx, tdxpy, httpx 等）。

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
    
    # 1. 加载 external 目录下的所有源码包
    local_ext_path = os.path.join(base_dir, "external")
    if os.path.exists(local_ext_path):
        # 将 external 目录加入 sys.path，这样可以直接导入其中的子文件夹作为包
        if local_ext_path not in sys.path:
            sys.path.insert(0, local_ext_path)
            
        # 针对 mootdx 的特殊路径处理（如果它在 external/mootdx/mootdx 结构下）
        mootdx_nested = os.path.join(local_ext_path, "mootdx")
        if os.path.exists(mootdx_nested) and mootdx_nested not in sys.path:
            sys.path.insert(0, mootdx_nested)

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
    print(f"[ERROR] 导入失败: {e}")
    print("[INFO] 请确保项目 external 目录下存在必要的依赖源码 (mootdx, tdxpy, httpx 等)")
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
                # 1. 基础字段处理
                if 'datetime' not in df.columns:
                    df = df.reset_index()
                
                # 2. 字段映射与计算
                # mootdx 默认字段: datetime, open, close, high, low, vol, amount
                df = df.rename(columns={
                    'datetime': '日期',
                    'open': '开盘',
                    'close': '收盘',
                    'high': '最高',
                    'low': '最低',
                    'vol': '成交量',
                    'amount': '成交额'
                })
                
                # 处理日期格式
                df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y%m%d')
                df['名称'] = name
                
                # 3. 计算额外指标 (模仿通达信导出格式)
                # 涨跌额 = 今日收盘 - 昨日收盘
                df['涨跌额'] = df['收盘'].diff()
                # 涨跌幅 = (今日收盘 - 昨日收盘) / 昨日收盘 * 100
                df['涨跌幅'] = (df['收盘'].diff() / df['收盘'].shift(1)) * 100
                # 振幅 = (最高 - 最低) / 昨日收盘 * 100
                df['振幅'] = ((df['最高'] - df['最低']) / df['收盘'].shift(1)) * 100
                
                # 换手率: mootdx 接口通常不直接提供日线换手率，如果需要精准值需从其他接口获取
                # 这里先填充 0.0 或 NaN，或者如果有总股本可以计算
                if 'turnover' in df.columns:
                    df = df.rename(columns={'turnover': '换手率'})
                else:
                    df['换手率'] = 0.0
                
                # 填充 NaN (第一行 diff 会产生 NaN)
                df = df.fillna(0.0)
                
                # 4. 按照要求的顺序重新排序列
                # 要求的顺序：名称,日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
                target_cols = ['名称', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                
                # 确保所有目标列都存在
                for col in target_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                
                return df[target_cols]
            return None
        except Exception as e:
            log(f"获取 {symbol} 数据失败: {e}", level="ERROR")
            import traceback
            traceback.print_exc()
            return None

    def save_data(self, symbol, df):
        if df is None or df.empty:
            return
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        
        # 根据代码判断市场
        # 6/9开头为上海，其他通常为深圳或北京
        if symbol.startswith(('6', '9')):
            market = "sh"
        elif symbol.startswith('8') or symbol.startswith('4'):
            market = "bj"
        else:
            market = "sz"
            
        save_dir = os.path.join(base_dir, "data", "day", market)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        file_path = os.path.join(save_dir, f"{symbol}.csv")
        # 写入 CSV，不包含索引
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
