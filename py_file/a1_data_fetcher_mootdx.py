# -*- coding: utf-8 -*-
"""
================================================================================
基于 mootdx 的日线数据下载模块 (a1_data_fetcher_mootdx.py)
================================================================================

功能说明:
    本模块使用优化后的 mootdx 库从通达信行情服务器获取 A 股日线数据。
    支持全量下载、智能增量更新、指定多代码下载。

用法示例:
    1. 下载单个股票全量: python a1_data_fetcher_mootdx.py --symbol 600036
    2. 全量下载所有股票: python a1_data_fetcher_mootdx.py --all
    3. 增量更新指定日期: python a1_data_fetcher_mootdx.py --all --start 20240101
    4. 默认开启 4 线程并行下载。

================================================================================
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# 1. 核心：优先加载本地源码逻辑
# ==============================================================================
def load_local_modules():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    local_ext_path = os.path.join(base_dir, "external")
    if os.path.exists(local_ext_path):
        if local_ext_path not in sys.path:
            sys.path.insert(0, local_ext_path)
        mootdx_nested = os.path.join(local_ext_path, "mootdx")
        if os.path.exists(mootdx_nested) and mootdx_nested not in sys.path:
            sys.path.insert(0, mootdx_nested)
    if current_dir not in sys.path:
        sys.path.append(current_dir)

load_local_modules()

try:
    from mootdx.quotes import Quotes
    from mootdx.logger import logger as mootdx_logger
    mootdx_logger.setLevel(logging.WARNING)
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    sys.exit(1)

def log(msg, level="INFO"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [{level}] {msg}")

class MootdxFetcher:
    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            self.client = Quotes.factory(market='std')
        except Exception as e:
            log(f"初始化失败: {e}", level="ERROR")
            self.client = None

    def get_stock_list(self):
        if not self.client: return []
        try:
            log("正在获取全市场股票列表...")
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
            df = df[df['code'].str.startswith(('60', '00', '30', '68', '8', '4'))]
            return df[['code', 'name', 'market_type']].to_dict('records')
        except Exception as e:
            log(f"获取股票列表失败: {e}", level="ERROR")
            return []

    def get_local_last_date(self, symbol):
        """获取本地文件的最后一条数据日期"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        market = "sh" if symbol.startswith(('6', '9')) else ("bj" if symbol.startswith(('8', '4')) else "sz")
        file_path = os.path.join(base_dir, "data", "day", market, f"{symbol}.csv")
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, usecols=['日期'], encoding='utf-8-sig')
                if not df.empty:
                    last_date_str = str(df['日期'].iloc[-1])
                    if '/' in last_date_str:
                        return datetime.strptime(last_date_str, '%Y/%m/%d').strftime('%Y%m%d'), file_path
                    return last_date_str, file_path
            except Exception:
                pass
        return None, file_path

    def fetch_daily(self, symbol, name="未知", offset=800, start_date=None, end_date=None, is_incremental=False):
        if not self.client: return None
        
        local_last_date, _ = self.get_local_last_date(symbol)
        actual_start = start_date
        
        if is_incremental and not actual_start and local_last_date:
            try:
                last_dt = datetime.strptime(local_last_date, '%Y%m%d')
                actual_start = (last_dt + timedelta(days=1)).strftime('%Y%m%d')
                offset = 100 
            except:
                pass

        try:
            df_new = self.client.bars(symbol=symbol, frequency='day', offset=offset)
            if df_new is not None and not df_new.empty:
                if 'datetime' not in df_new.columns: df_new = df_new.reset_index()
                
                # 核心修复：显式过滤掉不属于当前 symbol 的数据
                if 'code' in df_new.columns:
                    df_new = df_new[df_new['code'] == symbol].copy()
                
                if df_new.empty: return None

                df_new = df_new.rename(columns={'datetime':'日期','open':'开盘','close':'收盘','high':'最高','low':'最低','vol':'成交量','amount':'成交额'})
                df_new['日期'] = pd.to_datetime(df_new['日期'])
                
                # 过滤掉 1970 年等异常日期
                df_new = df_new[df_new['日期'] > pd.to_datetime('1990-01-01')]
                
                if actual_start:
                    df_new = df_new[df_new['日期'] >= pd.to_datetime(actual_start)]
                if end_date:
                    df_new = df_new[df_new['日期'] <= pd.to_datetime(end_date)]
                
                if df_new.empty: return None

                df_new['日期'] = df_new['日期'].dt.strftime('%Y/%m/%d')
                df_new['名称'] = name
                return df_new
            return None
        except Exception as e:
            log(f"获取 {symbol} 数据失败: {e}", level="ERROR")
            return None

    def save_data(self, symbol, df_new):
        if df_new is None or df_new.empty: return
        
        _, file_path = self.get_local_last_date(symbol)
        
        if os.path.exists(file_path):
            try:
                df_old = pd.read_csv(file_path, encoding='utf-8-sig')
                df_old['日期'] = pd.to_datetime(df_old['日期']).dt.strftime('%Y/%m/%d')
                df_new['日期'] = pd.to_datetime(df_new['日期']).dt.strftime('%Y/%m/%d')
                df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['日期'], keep='last').sort_values('日期')
            except Exception as e:
                log(f"合并 {symbol} 本地数据失败，将覆盖: {e}", level="WARNING")
                df_final = df_new.sort_values('日期')
        else:
            df_final = df_new.sort_values('日期')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 确保数值类型正确
        for col in ['开盘', '收盘', '最高', '最低', '成交量', '成交额']:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        
        df_final = df_final.dropna(subset=['收盘'])
        
        # 重新计算指标
        df_final['涨跌额'] = df_final['收盘'].diff()
        df_final['涨跌幅'] = (df_final['收盘'].diff() / df_final['收盘'].shift(1)) * 100
        df_final['振幅'] = ((df_final['最高'] - df_final['最低']) / df_final['收盘'].shift(1)) * 100
        if '换手率' not in df_final.columns: df_final['换手率'] = 0.0
        
        df_final = df_final.fillna(0.0)
        target_cols = ['名称', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        for col in target_cols:
            if col not in df_final.columns: df_final[col] = 0.0
            
        df_final[target_cols].to_csv(file_path, index=False, encoding='utf-8-sig')

def process_task(stock, offset, start, end, is_incremental):
    try:
        fetcher = MootdxFetcher()
        df = fetcher.fetch_daily(stock['code'], stock['name'], offset=offset, start_date=start, end_date=end, is_incremental=is_incremental)
        if df is not None:
            fetcher.save_data(stock['code'], df)
            return True
    except Exception as e:
        log(f"线程处理 {stock['code']} 失败: {e}", level="ERROR")
    return False

def main():
    parser = argparse.ArgumentParser(description="Mootdx 数据抓取工具 (彻底修复版)")
    parser.add_argument("--symbol", type=str, help="股票代码，支持多个逗号分隔")
    parser.add_argument("--all", action="store_true", help="全量下载模式")
    parser.add_argument("--start", type=str, help="开始日期 (YYYYMMDD)")
    parser.add_argument("--end", type=str, help="结束日期 (YYYYMMDD)")
    parser.add_argument("--limit", type=int, default=None, help="获取天数限制")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数 (默认 4)")
    args = parser.parse_args()

    initial_fetcher = MootdxFetcher()
    stocks_to_download = []
    
    is_incremental = True if (args.start or not args.limit) else False
    if args.limit is None:
        limit = 4000 if not is_incremental else 800
    else:
        limit = args.limit

    if args.all:
        stocks_to_download = initial_fetcher.get_stock_list()
    elif args.symbol:
        is_incremental = False
        limit = args.limit if args.limit else 4000
        for s in args.symbol.split(','):
            stocks_to_download.append({'code': s.strip(), 'name': '未知'})
    else:
        log("请提供 --symbol 或 --all 参数", level="ERROR")
        return

    log(f"模式: {'增量更新' if is_incremental else '全量下载'}, 线程数: {args.workers}, 目标数: {len(stocks_to_download)}")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_stock = {executor.submit(process_task, s, limit, args.start, args.end, is_incremental): s for s in stocks_to_download}
        for i, future in enumerate(as_completed(future_to_stock)):
            if future.result(): success_count += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(stocks_to_download):
                log(f"进度: {i+1}/{len(stocks_to_download)}, 成功: {success_count}")

    log(f"任务结束。成功: {success_count}/{len(stocks_to_download)}")

if __name__ == "__main__":
    main()
