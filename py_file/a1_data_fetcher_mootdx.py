# -*- coding: utf-8 -*-
"""
================================================================================
基于 mootdx 的日线数据下载模块 (a1_data_fetcher_mootdx.py)
================================================================================

功能说明:
    本模块使用优化后的 mootdx 库从通达信行情服务器获取 A 股日线数据。
    支持全量下载、日期范围下载、指定多代码下载。

用法示例:
    1. 下载单个股票: python a1_data_fetcher_mootdx.py --symbol 600036
    2. 下载多个股票: python a1_data_fetcher_mootdx.py --symbol 600036,000001
    3. 全量下载:     python a1_data_fetcher_mootdx.py --all
    4. 日期范围:     python a1_data_fetcher_mootdx.py --symbol 600036 --start 20230101 --end 20231231

================================================================================
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        if local_ext_path not in sys.path:
            sys.path.insert(0, local_ext_path)
            
        # 针对 mootdx 的特殊路径处理
        mootdx_nested = os.path.join(local_ext_path, "mootdx")
        if os.path.exists(mootdx_nested) and mootdx_nested not in sys.path:
            sys.path.insert(0, mootdx_nested)

    if current_dir not in sys.path:
        sys.path.append(current_dir)

load_local_modules()

# 现在可以安全导入了
try:
    from mootdx.quotes import Quotes
    from mootdx.logger import logger as mootdx_logger
    mootdx_logger.setLevel(logging.WARNING)
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

    def get_stock_list(self):
        """获取沪深全市场股票列表"""
        if not self.client: return []
        try:
            log("正在获取全市场股票列表...")
            # market=1 为上海, market=0 为深圳
            sh = self.client.stocks(market=1)
            sz = self.client.stocks(market=0)
            
            all_stocks = []
            if sh is not None:
                sh['market_type'] = 'sh'
                all_stocks.append(sh)
            if sz is not None:
                sz['market_type'] = 'sz'
                all_stocks.append(sz)
            
            if not all_stocks: return []
            
            df = pd.concat(all_stocks)
            # 过滤掉指数和非 A 股（简单过滤，实际可根据代码规则精细化）
            # 过滤掉退市或指数类代码 (通常 A 股为 60, 00, 30, 68, 8, 4 开头)
            df = df[df['code'].str.startswith(('60', '00', '30', '68', '8', '4'))]
            
            return df[['code', 'name', 'market_type']].to_dict('records')
        except Exception as e:
            log(f"获取股票列表失败: {e}", level="ERROR")
            return []

    def fetch_daily(self, symbol, name="未知", offset=800, start_date=None, end_date=None):
        if not self.client:
            return None
        
        try:
            # 获取数据
            df = self.client.bars(symbol=symbol, frequency='day', offset=offset)
            if df is not None and not df.empty:
                if 'datetime' not in df.columns:
                    df = df.reset_index()
                
                # 字段映射
                df = df.rename(columns={
                    'datetime': '日期',
                    'open': '开盘',
                    'close': '收盘',
                    'high': '最高',
                    'low': '最低',
                    'vol': '成交量',
                    'amount': '成交额'
                })
                
                # 处理日期
                df['日期'] = pd.to_datetime(df['日期'])
                
                # 日期范围过滤
                if start_date:
                    start_ts = pd.to_datetime(start_date)
                    df = df[df['日期'] >= start_ts]
                if end_date:
                    end_ts = pd.to_datetime(end_date)
                    df = df[df['日期'] <= end_ts]
                
                if df.empty: return None

                df['日期'] = df['日期'].dt.strftime('%Y%m%d')
                df['名称'] = name
                
                # 计算指标
                df['涨跌额'] = df['收盘'].diff()
                df['涨跌幅'] = (df['收盘'].diff() / df['收盘'].shift(1)) * 100
                df['振幅'] = ((df['最高'] - df['最低']) / df['收盘'].shift(1)) * 100
                df['换手率'] = 0.0 # 默认填充
                
                df = df.fillna(0.0)
                
                # 排序与列选择
                target_cols = ['名称', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                for col in target_cols:
                    if col not in df.columns: df[col] = 0.0
                
                return df[target_cols]
            return None
        except Exception as e:
            log(f"获取 {symbol} 数据失败: {e}", level="ERROR")
            return None

    def save_data(self, symbol, df):
        if df is None or df.empty: return
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        
        if symbol.startswith(('6', '9')): market = "sh"
        elif symbol.startswith(('8', '4')): market = "bj"
        else: market = "sz"
            
        save_dir = os.path.join(base_dir, "data", "day", market)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
            
        file_path = os.path.join(save_dir, f"{symbol}.csv")
        df.to_csv(file_path, index=False, encoding='utf-8-sig')

def process_task(fetcher, stock, offset, start, end):
    code = stock['code']
    name = stock['name']
    df = fetcher.fetch_daily(code, name, offset=offset, start_date=start, end_date=end)
    if df is not None:
        fetcher.save_data(code, df)
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Mootdx 数据抓取工具 (增强版)")
    parser.add_argument("--symbol", type=str, help="股票代码，支持多个逗号分隔，如 600036,000001")
    parser.add_argument("--all", action="store_true", help="全量下载所有 A 股")
    parser.add_argument("--start", type=str, help="开始日期 (YYYYMMDD)")
    parser.add_argument("--end", type=str, help="结束日期 (YYYYMMDD)")
    parser.add_argument("--limit", type=int, default=800, help="单只股票获取的最大天数 (默认 800)")
    parser.add_argument("--workers", type=int, default=4, help="并行下载线程数 (默认 4)")
    args = parser.parse_args()

    fetcher = MootdxFetcher()
    stocks_to_download = []

    if args.all:
        stocks_to_download = fetcher.get_stock_list()
    elif args.symbol:
        symbols = args.symbol.split(',')
        for s in symbols:
            stocks_to_download.append({'code': s.strip(), 'name': '未知'})
    else:
        log("请提供 --symbol 或 --all 参数", level="ERROR")
        return

    if not stocks_to_download:
        log("没有需要下载的股票列表", level="WARNING")
        return

    log(f"准备下载 {len(stocks_to_download)} 只股票的数据...")
    
    success_count = 0
    # 使用线程池加速全量下载
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_stock = {executor.submit(process_task, fetcher, s, args.limit, args.start, args.end): s for s in stocks_to_download}
        
        for i, future in enumerate(as_completed(future_to_stock)):
            stock = future_to_stock[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                log(f"处理 {stock['code']} 时发生异常: {e}", level="ERROR")
            
            if (i + 1) % 50 == 0 or (i + 1) == len(stocks_to_download):
                log(f"进度: {i+1}/{len(stocks_to_download)}, 成功: {success_count}")

    log(f"下载任务结束。总计: {len(stocks_to_download)}, 成功: {success_count}")

if __name__ == "__main__":
    main()
