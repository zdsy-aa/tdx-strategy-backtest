#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
基于 mootdx 的日线数据下载模块 (a1_data_fetcher_mootdx.py)
================================================================================

功能说明:
    本脚本利用 mootdx 接口抓取沪深京市场的股票日线 K 线数据。采用多进程架构
    以彻底隔离底层库的全局状态冲突，确保数据的纯净性和准确性。

主要功能:
    1. 支持全量下载、增量更新及指定股票代码下载。
    2. 智能增量合并：新下载的数据将与本地历史数据合并，按日期去重并排序，绝不覆盖历史数据。
    3. 自动计算指标：合并后重新计算涨跌额、涨跌幅、振幅等技术指标。
    4. 严格的数据归属校验，防止不同股票数据串扰。
    5. 统一输出格式为 YYYY/MM/DD 的 CSV 文件。
    6. 默认开启 4 进程并行下载，提高效率。

使用方法:
    python a1_data_fetcher_mootdx.py [options]
    示例:
      全量下载所有历史: python a1_data_fetcher_mootdx.py --all
      下载指定股票全量: python a1_data_fetcher_mootdx.py --symbol 600036,000001
      指定日期范围增量: python a1_data_fetcher_mootdx.py --symbol 600036 --start 20240101 --end 20241231

依赖库:
    pandas, mootdx, tdxpy

安装命令:
    pip install pandas
    # mootdx 和 tdxpy 已集成在项目 external 目录下，或使用:
    pip install mootdx tdxpy

================================================================================
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==============================================================================
# 1. 环境配置与模块加载
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
    pid = os.getpid()
    print(f"[{now}] [{level}] [PID:{pid}] {msg}")

class MootdxFetcher:
    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            # 增加随机延迟，避免多进程同时冲击服务器
            time.sleep(random.uniform(0.1, 0.5))
            self.client = Quotes.factory(market='std')
        except Exception as e:
            log(f"初始化行情服务器失败: {e}", level="ERROR")
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
            # 过滤 A 股代码
            df = df[df['code'].str.startswith(('60', '00', '30', '68', '8', '4'))]
            return df[['code', 'name', 'market_type']].to_dict('records')
        except Exception as e:
            log(f"获取股票列表失败: {e}", level="ERROR")
            return []

    def get_local_info(self, symbol):
        """获取本地文件路径及最后日期"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        market = "sh" if symbol.startswith(('6', '9')) else ("bj" if symbol.startswith(('8', '4')) else "sz")
        file_path = os.path.join(base_dir, "data", "day", market, f"{symbol}.csv")
        
        last_date = None
        if os.path.exists(file_path):
            try:
                # 仅读取最后一行日期
                df = pd.read_csv(file_path, usecols=['日期'], encoding='utf-8-sig')
                if not df.empty:
                    last_date_str = str(df['日期'].iloc[-1]).replace('/', '')
                    last_date = last_date_str
            except Exception:
                pass
        return last_date, file_path

    def fetch_daily(self, symbol, name="未知", offset=8000, start_date=None, end_date=None, is_incremental=False):
        if not self.client: return None
        
        local_last_date, _ = self.get_local_info(symbol)
        
        # 增量模式逻辑：
        # 如果是增量模式，且未显式指定开始日期，且本地有数据，则从本地最后一天的后一天开始抓取
        actual_start = start_date
        if is_incremental and not actual_start and local_last_date:
            try:
                last_dt = datetime.strptime(local_last_date, '%Y%m%d')
                actual_start = (last_dt + timedelta(days=1)).strftime('%Y%m%d')
                # 增量模式下，如果 offset 过大，缩减抓取长度以提高效率
                if offset > 200:
                    offset = 200 
            except:
                pass

        try:
            df_new = self.client.bars(symbol=symbol, frequency='day', offset=offset)
            
            if df_new is not None and not df_new.empty:
                if 'datetime' not in df_new.columns:
                    df_new = df_new.reset_index()
                
                # 严格校验：只保留当前进程负责的 symbol
                if 'code' in df_new.columns:
                    df_new = df_new[df_new['code'].astype(str) == str(symbol)].copy()
                
                if df_new.empty: return None

                df_new = df_new.rename(columns={
                    'datetime': '日期', 'open': '开盘', 'close': '收盘', 
                    'high': '最高', 'low': '最低', 'vol': '成交量', 'amount': '成交额'
                })
                
                df_new['日期'] = pd.to_datetime(df_new['日期'])
                # 基础过滤：排除非法日期
                df_new = df_new[df_new['日期'] > pd.to_datetime('1990-01-01')]
                
                # 日期范围过滤
                if actual_start:
                    df_new = df_new[df_new['日期'] >= pd.to_datetime(actual_start)]
                if end_date:
                    df_new = df_new[df_new['日期'] <= pd.to_datetime(end_date)]
                
                if df_new.empty: return None

                # 统一格式化为 YYYY/MM/DD
                df_new['日期'] = df_new['日期'].dt.strftime('%Y/%m/%d')
                df_new['名称'] = name
                return df_new
            return None
        except Exception as e:
            log(f"获取 {symbol} 数据失败: {e}", level="ERROR")
            return None

    def save_data(self, symbol, df_new):
        """核心合并逻辑：读取旧数据 -> 合并新数据 -> 去重排序 -> 重算指标 -> 保存"""
        if df_new is None or df_new.empty: return
        
        _, file_path = self.get_local_info(symbol)
        
        if os.path.exists(file_path):
            try:
                # 读取本地所有历史数据
                df_old = pd.read_csv(file_path, encoding='utf-8-sig')
                # 确保旧数据日期格式统一，便于合并去重
                df_old['日期'] = pd.to_datetime(df_old['日期']).dt.strftime('%Y/%m/%d')
                
                # 合并新旧数据
                # drop_duplicates 确保日期唯一，keep='last' 优先保留新下载的数据
                df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['日期'], keep='last').sort_values('日期')
            except Exception as e:
                log(f"合并 {symbol} 失败，为保护数据，本次不执行保存: {e}", level="ERROR")
                return
        else:
            df_final = df_new.sort_values('日期')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 数据清洗与数值化
        for col in ['开盘', '收盘', '最高', '最低', '成交量', '成交额']:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        
        df_final = df_final.dropna(subset=['收盘'])
        
        # 重新计算技术指标，确保合并后的连续性
        df_final['涨跌额'] = df_final['收盘'].diff()
        df_final['涨跌幅'] = (df_final['收盘'].diff() / df_final['收盘'].shift(1)) * 100
        df_final['振幅'] = ((df_final['最高'] - df_final['最低']) / df_final['收盘'].shift(1)) * 100
        if '换手率' not in df_final.columns: df_final['换手率'] = 0.0
        
        df_final = df_final.fillna(0.0)
        
        # 严格按要求的列顺序保存
        target_cols = ['名称', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        df_final[target_cols].to_csv(file_path, index=False, encoding='utf-8-sig')

def process_single_stock(stock, offset, start, end, is_incremental):
    """独立的进程处理函数"""
    try:
        fetcher = MootdxFetcher()
        df = fetcher.fetch_daily(stock['code'], stock['name'], offset, start, end, is_incremental)
        if df is not None:
            fetcher.save_data(stock['code'], df)
            return True
        return False
    except Exception as e:
        log(f"进程异常 {stock['code']}: {e}", level="ERROR")
    return False

def main():
    parser = argparse.ArgumentParser(description="Mootdx 数据抓取工具 (多进程增量合并版)")
    parser.add_argument("--symbol", type=str, help="股票代码，支持多个逗号分隔")
    parser.add_argument("--all", action="store_true", help="全量下载模式")
    parser.add_argument("--start", type=str, help="开始日期 (YYYYMMDD)")
    parser.add_argument("--end", type=str, help="结束日期 (YYYYMMDD)")
    parser.add_argument("--limit", type=int, default=None, help="获取天数限制")
    parser.add_argument("--workers", type=int, default=4, help="并行进程数 (默认 4)")
    args = parser.parse_args()

    # 初始化获取列表
    init_fetcher = MootdxFetcher()
    stocks_to_download = []
    
    # --- 参数识别逻辑修正 ---
    # 1. 如果用户明确输入了 --start 或 --end，则一定是增量/范围模式
    # 2. 如果用户没有输入 --limit，且没有输入 --all (即指定了 --symbol)，默认执行全量抓取
    # 3. 如果用户输入了 --all，且没有指定日期范围，应强制执行全量下载
    
    if args.start or args.end:
        # 增量/范围模式
        is_incremental = True
        limit = args.limit if args.limit else 800
    elif args.all:
        # 全量模式：抓取所有日期 (8000天)
        is_incremental = False
        limit = args.limit if args.limit else 8000
    elif args.symbol:
        # 指定代码全量模式
        is_incremental = False
        limit = args.limit if args.limit else 8000
    else:
        log("请提供 --symbol 或 --all 参数", level="ERROR")
        return

    if args.all:
        stocks_to_download = init_fetcher.get_stock_list()
    elif args.symbol:
        for s in args.symbol.split(','):
            stocks_to_download.append({'code': s.strip(), 'name': '未知'})

    log(f"启动多进程抓取 | 模式: {'增量更新' if is_incremental else '全量下载'}, 进程数: {args.workers}, 目标数: {len(stocks_to_download)}, 抓取长度: {limit}")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_stock, s, limit, args.start, args.end, is_incremental): s 
            for s in stocks_to_download
        }
        
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                success_count += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(stocks_to_download):
                log(f"进度: {i+1}/{len(stocks_to_download)}, 成功更新: {success_count}")

    log(f"任务结束。成功处理: {success_count}/{len(stocks_to_download)}")

if __name__ == "__main__":
    main()
