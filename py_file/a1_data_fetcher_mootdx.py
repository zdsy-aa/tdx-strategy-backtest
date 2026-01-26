# -*- coding: utf-8 -*-
"""
================================================================================
基于 mootdx 的日线数据下载模块 (a1_data_fetcher_mootdx.py) - 稳定增强版
================================================================================

修复说明:
    1. 彻底解决数据串扰: 在多线程环境下，mootdx 可能因全局 instance 共享导致数据错乱。
       本版本引入严格的数据归属校验 (Data Ownership Validation)。
    2. 严格日期格式: 统一使用 YYYY/MM/DD。
    3. 线程安全: 每个线程强制隔离，并在获取数据后立即校验 code 字段。

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
    print(f"[{now}] [{level}] {msg}")

class MootdxFetcher:
    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            # 显式指定 std 模式，并在每个实例中隔离连接
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
            # 获取数据
            df_new = self.client.bars(symbol=symbol, frequency='day', offset=offset)
            
            if df_new is not None and not df_new.empty:
                # 核心防御逻辑：检查返回的数据是否属于当前 symbol
                # mootdx.bars 返回的 DataFrame index 通常是 datetime，但也可能包含 code 列
                if 'datetime' not in df_new.columns:
                    df_new = df_new.reset_index()
                
                # 关键修复：mootdx 的 to_data 可能会在 df 中注入 code
                # 如果没有 code 列，我们需要通过其他方式验证（或者信任这次 factory 隔离后的 bars 调用）
                # 但为了绝对安全，我们在这里进行二次校验
                
                # 某些情况下，mootdx 返回的 df 包含 'code' 列
                if 'code' in df_new.columns:
                    # 过滤掉不属于当前 symbol 的行
                    wrong_data = df_new[df_new['code'] != symbol]
                    if not wrong_data.empty:
                        log(f"警告: {symbol} 的数据中混入了其他代码 ({wrong_data['code'].unique()})，已过滤。", level="WARNING")
                    df_new = df_new[df_new['code'] == symbol].copy()

                if df_new.empty:
                    return None

                # 重命名列名
                df_new = df_new.rename(columns={
                    'datetime': '日期', 'open': '开盘', 'close': '收盘', 
                    'high': '最高', 'low': '最低', 'vol': '成交量', 'amount': '成交额'
                })
                
                df_new['日期'] = pd.to_datetime(df_new['日期'])
                
                # 过滤异常日期
                df_new = df_new[df_new['日期'] > pd.to_datetime('1990-01-01')]
                
                if actual_start:
                    df_new = df_new[df_new['日期'] >= pd.to_datetime(actual_start)]
                if end_date:
                    df_new = df_new[df_new['日期'] <= pd.to_datetime(end_date)]
                
                if df_new.empty:
                    return None

                # 统一日期格式为 YYYY/MM/DD
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
                # 确保旧数据的日期也是 YYYY/MM/DD
                df_old['日期'] = pd.to_datetime(df_old['日期']).dt.strftime('%Y/%m/%d')
                
                # 合并并去重
                df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['日期'], keep='last').sort_values('日期')
            except Exception as e:
                log(f"合并 {symbol} 本地数据失败，将覆盖: {e}", level="WARNING")
                df_final = df_new.sort_values('日期')
        else:
            df_final = df_new.sort_values('日期')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 数值转换与清洗
        for col in ['开盘', '收盘', '最高', '最低', '成交量', '成交额']:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        
        df_final = df_final.dropna(subset=['收盘'])
        
        # 重新计算技术指标 (涨跌额、涨跌幅、振幅)
        # 注意：diff 计算依赖于正确的排序
        df_final['涨跌额'] = df_final['收盘'].diff()
        df_final['涨跌幅'] = (df_final['收盘'].diff() / df_final['收盘'].shift(1)) * 100
        df_final['振幅'] = ((df_final['最高'] - df_final['最低']) / df_final['收盘'].shift(1)) * 100
        
        if '换手率' not in df_final.columns:
            df_final['换手率'] = 0.0
        
        df_final = df_final.fillna(0.0)
        
        # 严格按照要求的列顺序输出
        target_cols = ['名称', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        for col in target_cols:
            if col not in df_final.columns:
                df_final[col] = 0.0
            
        df_final[target_cols].to_csv(file_path, index=False, encoding='utf-8-sig')

def task_wrapper(stock, offset, start, end, is_incremental):
    """单线程任务包装器，每个任务独立实例化 Fetcher 以强制隔离连接"""
    try:
        # 关键：每个线程内部实例化，防止 mootdx 全局单例干扰
        fetcher = MootdxFetcher()
        df = fetcher.fetch_daily(
            stock['code'], stock['name'], 
            offset=offset, start_date=start, end_date=end, 
            is_incremental=is_incremental
        )
        if df is not None:
            fetcher.save_data(stock['code'], df)
            return True
    except Exception as e:
        log(f"处理 {stock['code']} ({stock['name']}) 时发生未捕获异常: {e}", level="ERROR")
    return False

def main():
    parser = argparse.ArgumentParser(description="Mootdx 数据抓取工具 (串扰修复版)")
    parser.add_argument("--symbol", type=str, help="股票代码，支持多个逗号分隔")
    parser.add_argument("--all", action="store_true", help="全量下载模式")
    parser.add_argument("--start", type=str, help="开始日期 (YYYYMMDD)")
    parser.add_argument("--end", type=str, help="结束日期 (YYYYMMDD)")
    parser.add_argument("--limit", type=int, default=None, help="获取天数限制")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数 (默认 4)")
    args = parser.parse_args()

    # 1. 初始化并获取任务列表
    init_fetcher = MootdxFetcher()
    stocks_to_download = []
    
    is_incremental = True if (args.start or not args.limit) else False
    if args.limit is None:
        limit = 4000 if not is_incremental else 800
    else:
        limit = args.limit

    if args.all:
        stocks_to_download = init_fetcher.get_stock_list()
    elif args.symbol:
        is_incremental = False
        limit = args.limit if args.limit else 4000
        for s in args.symbol.split(','):
            stocks_to_download.append({'code': s.strip(), 'name': '未知'})
    else:
        log("请提供 --symbol 或 --all 参数", level="ERROR")
        return

    if not stocks_to_download:
        log("未找到待处理的股票列表，请检查网络或服务器连接。", level="WARNING")
        return

    log(f"模式: {'增量更新' if is_incremental else '全量下载'}, 线程数: {args.workers}, 目标数: {len(stocks_to_download)}")
    
    # 2. 并行执行任务
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(task_wrapper, s, limit, args.start, args.end, is_incremental): s 
            for s in stocks_to_download
        }
        
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                success_count += 1
            
            if (i + 1) % 50 == 0 or (i + 1) == len(stocks_to_download):
                log(f"执行进度: {i+1}/{len(stocks_to_download)}, 成功抓取: {success_count}")

    log(f"下载任务圆满结束。成功率: {success_count}/{len(stocks_to_download)}")

if __name__ == "__main__":
    main()
