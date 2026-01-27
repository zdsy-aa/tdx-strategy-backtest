#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
基于 mootdx 的日线数据下载模块 (a1_data_fetcher_mootdx.py) - v3 (注释增强版)
================================================================================

功能说明:
    本脚本利用 mootdx 接口抓取沪深京市场的股票日线 K 线数据。采用多进程架构
    以彻底隔离底层库的全局状态冲突，确保数据下载的稳定、高效和准确。

主要功能:
    1. 多模式下载：支持全量历史下载、增量更新及指定代码下载。
    2. 智能增量合并：新下载的数据将与本地历史数据合并，按日期去重并排序，确保数据连续性。
    3. 自动计算指标：数据保存时，会自动计算涨跌额、涨跌幅、振幅等常用技术指标。
    4. 股票名称精准获取：在主进程统一获取股票代码与名称的映射，确保多进程环境下名称准确无误。
    5. 数据格式统一：所有输出的 CSV 文件均采用统一的列序和 `YYYY/MM/DD` 日期格式。
    6. 高效并行处理：默认开启 4 进程并行下载，可根据机器性能调整进程数。

--------------------------------------------------------------------------------
使用方法:
    在终端或命令行中，使用 `python a1_data_fetcher_mootdx.py [参数]` 来运行脚本。

    常用命令示例:

    1. 【全量下载】下载所有 A 股的完整历史数据 (首次运行时使用):
       python a1_data_fetcher_mootdx.py --all

    2. 【增量更新】对本地已有的所有股票数据进行增量更新 (推荐每日收盘后运行):
       # 脚本会自动查找每个股票的本地最后日期，并从下一天开始下载
       python a1_data_fetcher_mootdx.py --all --start

    3. 【下载指定股】下载单只或多只股票的完整历史数据:
       # 下载单只 (贵州茅台)
       python a1_data_fetcher_mootdx.py --symbol 600519
       # 下载多只 (用逗号分隔)
       python a1_data_fetcher_mootdx.py --symbol 600519,000001,300750

    4. 【指定日期范围】下载指定股票在特定日期范围内的数据:
       python a1_data_fetcher_mootdx.py --symbol 600036 --start 20230101 --end 20231231

    5. 【调整进程数】使用 8 个进程进行全量下载以提高速度:
       # 进程数建议不超过 CPU 核心数的两倍
       python a1_data_fetcher_mootdx.py --all --workers 8

--------------------------------------------------------------------------------
依赖库:
    pandas, mootdx, tdxpy

安装命令:
    pip install pandas
    # mootdx 和 tdxpy 已集成在项目 external 目录下，或通过 pip 安装:
    pip install mootdx tdxpy
================================================================================
"""

# ==============================================================================
# 1. 系统与环境配置
# ==============================================================================
import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_local_modules():
    """
    动态加载本地 `external` 目录下的依赖库。
    这使得脚本可以打包分发，无需用户手动安装特定版本的库。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    local_ext_path = os.path.join(base_dir, "external")
    if os.path.exists(local_ext_path):
        if local_ext_path not in sys.path:
            sys.path.insert(0, local_ext_path)
        # 兼容嵌套的 mootdx 目录结构
        mootdx_nested = os.path.join(local_ext_path, "mootdx")
        if os.path.exists(mootdx_nested) and mootdx_nested not in sys.path:
            sys.path.insert(0, mootdx_nested)
    if current_dir not in sys.path:
        sys.path.append(current_dir)

# 执行加载，以便后续导入 `mootdx`
load_local_modules()

# 尝试导入核心依赖库，如果失败则退出
try:
    from mootdx.quotes import Quotes
    from mootdx.logger import logger as mootdx_logger
    # 将 mootdx 的日志级别调高，避免在控制台输出过多的调试信息
    mootdx_logger.setLevel(logging.WARNING)
except ImportError as e:
    print(f"[ERROR] 核心依赖库 `mootdx` 导入失败: {e}")
    print("[INFO] 请确保已安装 `mootdx` 或 `external` 目录存在且结构正确。")
    sys.exit(1)

def log(msg, level="INFO"):
    """
    自定义的日志函数，用于在控制台输出带有时间戳和进程ID的信息。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pid = os.getpid()
    print(f"[{now}] [{level}] [PID:{pid}] {msg}")


# ==============================================================================
# 2. 数据获取核心类 (MootdxFetcher)
# ==============================================================================
class MootdxFetcher:
    """
    封装了与 `mootdx` 交互的所有操作，如连接、获取股票列表、下载日线数据等。
    """
    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        """
        初始化 `mootdx` 行情客户端。
        增加随机延迟，以减轻多进程同时初始化时对服务器的冲击。
        """
        try:
            # 随机休眠 0.1 到 0.5 秒
            time.sleep(random.uniform(0.1, 0.5))
            self.client = Quotes.factory(market='std')
        except Exception as e:
            log(f"初始化行情服务器失败: {e}", level="ERROR")
            self.client = None

    def get_stock_list(self):
        """
        从服务器获取完整的沪深京A股列表（代码、名称、市场）。
        这是唯一需要主进程调用的函数，用于获取股票全貌。
        """
        if not self.client: return []
        try:
            log("正在从服务器获取全市场A股列表...")
            # 分别获取上海和深圳市场的股票
            sh = self.client.stocks(market=1) # 1: 上海
            sz = self.client.stocks(market=0) # 0: 深圳
            
            all_stocks = []
            if sh is not None:
                sh['market_type'] = 'sh'
                all_stocks.append(sh)
            if sz is not None:
                sz['market_type'] = 'sz'
                all_stocks.append(sz)
            
            if not all_stocks: return []
            
            # 合并两个市场的 DataFrame
            df = pd.concat(all_stocks)
            # 根据A股代码规则进行筛选
            df = df[df['code'].str.startswith(('60', '00', '30', '68', '8', '4'))]
            
            log(f"已成功获取 {len(df)} 只A股股票信息。")
            # 将 DataFrame 转换为字典列表，方便后续处理
            return df[['code', 'name', 'market_type']].to_dict('records')
        except Exception as e:
            log(f"获取股票列表失败: {e}", level="ERROR")
            return []

    def get_local_info(self, symbol):
        """
        获取指定股票在本地的存储路径和最后一条数据的日期。
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        # 根据股票代码判断所属市场，以确定存储子目录
        market = "sh" if symbol.startswith(('6', '9')) else ("bj" if symbol.startswith(('8', '4')) else "sz")
        file_path = os.path.join(base_dir, "data", "day", market, f"{symbol}.csv")
        
        last_date = None
        if os.path.exists(file_path):
            try:
                # 高效读取：仅读取最后一行的日期列，避免加载整个文件
                df = pd.read_csv(file_path, usecols=['日期'], encoding='utf-8-sig')
                if not df.empty:
                    last_date_str = str(df['日期'].iloc[-1]).replace('/', '')
                    last_date = last_date_str
            except Exception:
                # 如果文件损坏或格式不正确，则忽略
                pass
        return last_date, file_path

    def fetch_daily(self, symbol, stock_name, offset=None, start_date=None, end_date=None, is_incremental=False):
        """
        获取单只股票的日线数据。
        """
        if not self.client: return None
        
        # 如果未指定 `offset` (获取天数)，则根据模式设置默认值
        if offset is None:
            offset = 800 if is_incremental else 8000

        local_last_date, _ = self.get_local_info(symbol)
        
        actual_start = start_date
        # 智能增量逻辑：如果是增量模式，且用户未指定开始日期，则自动从本地最后日期的后一天开始
        if is_incremental and not actual_start and local_last_date:
            try:
                last_dt = datetime.strptime(local_last_date, '%Y%m%d')
                actual_start = (last_dt + timedelta(days=1)).strftime('%Y%m%d')
                # 增量模式下，将抓取长度缩短以提高效率
                if offset > 200:
                    offset = 200 
            except:
                pass

        try:
            # 调用 mootdx API 获取日线数据
            df_new = self.client.bars(symbol=symbol, frequency='day', offset=offset)
            
            if df_new is not None and not df_new.empty:
                # 数据清洗与格式化
                if 'datetime' not in df_new.columns:
                    df_new = df_new.reset_index()
                
                # 严格校验数据归属，防止API返回其他股票的数据
                if 'code' in df_new.columns:
                    df_new = df_new[df_new['code'].astype(str) == str(symbol)].copy()
                
                if df_new.empty: return None

                # 重命名列为中文
                df_new = df_new.rename(columns={
                    'datetime': '日期', 'open': '开盘', 'close': '收盘', 
                    'high': '最高', 'low': '最低', 'vol': '成交量', 'amount': '成交额'
                })
                
                df_new['日期'] = pd.to_datetime(df_new['日期'])
                # 过滤掉无效的早期数据
                df_new = df_new[df_new['日期'] > pd.to_datetime('1990-01-01')]
                
                # 根据指定的开始和结束日期进行过滤
                if actual_start:
                    df_new = df_new[df_new['日期'] >= pd.to_datetime(actual_start)]
                if end_date:
                    df_new = df_new[df_new['日期'] <= pd.to_datetime(end_date)]
                
                if df_new.empty: return None

                # 统一日期格式并添加股票名称列
                df_new['日期'] = df_new['日期'].dt.strftime('%Y/%m/%d')
                df_new['名称'] = stock_name
                return df_new
            return None
        except Exception as e:
            log(f"获取 {symbol} ({stock_name}) 数据时发生异常: {e}", level="ERROR")
            return None

    def save_data(self, symbol, stock_name, df_new):
        """
        核心数据合并与保存逻辑。
        读取旧数据 -> 合并新数据 -> 去重排序 -> 重算指标 -> 保存到 CSV。
        """
        if df_new is None or df_new.empty: return
        
        _, file_path = self.get_local_info(symbol)
        
        # 如果本地已存在历史数据文件
        if os.path.exists(file_path):
            try:
                # 读取本地所有历史数据
                df_old = pd.read_csv(file_path, encoding='utf-8-sig')
                # 统一日期格式，为合并做准备
                df_old['日期'] = pd.to_datetime(df_old['日期']).dt.strftime('%Y/%m/%d')
                
                # 确保新旧数据中的名称列都是最新的正确名称
                df_new['名称'] = stock_name
                df_old['名称'] = stock_name
                
                # 合并新旧数据，并基于日期去重，保留最新下载的数据 (`keep='last'`)
                df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['日期'], keep='last').sort_values('日期')
            except Exception as e:
                log(f"合并 {symbol} 数据失败，为保护本地数据，本次不执行保存: {e}", level="ERROR")
                return
        else:
            # 如果是新股票，直接使用下载的数据
            df_new['名称'] = stock_name
            df_final = df_new.sort_values('日期')
            # 创建存储目录
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # --- 数据清洗与指标计算 ---
        # 将核心数据列转换为数值类型
        for col in ['开盘', '收盘', '最高', '最低', '成交量', '成交额']:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        
        # 删除收盘价为空的无效行
        df_final = df_final.dropna(subset=['收盘'])
        
        # 重新计算技术指标，确保合并后数据的连续性和准确性
        df_final['涨跌额'] = df_final['收盘'].diff()
        df_final['涨跌幅'] = (df_final['收盘'].diff() / df_final['收盘'].shift(1)) * 100
        df_final['振幅'] = ((df_final['最高'] - df_final['最低']) / df_final['收盘'].shift(1)) * 100
        # 如果不存在换手率列，则填充为0
        if '换手率' not in df_final.columns: df_final['换手率'] = 0.0
        
        # 将计算产生的 NaN 值（如第一天的涨跌幅）填充为 0
        df_final = df_final.fillna(0.0)
        
        # 再次确保名称列正确
        df_final['名称'] = stock_name
        
        # 按照指定的列顺序组织 DataFrame
        target_cols = ['名称', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        # 保存到 CSV 文件，不包含索引列
        df_final[target_cols].to_csv(file_path, index=False, encoding='utf-8-sig')


# ==============================================================================
# 3. 多进程执行与主函数
# ==============================================================================
def process_single_stock(stock, offset, start, end, is_incremental):
    """
    独立的进程工作函数。每个子进程会执行这个函数。
    它接收包含代码和名称的 `stock` 字典，并完成该股票的数据获取和保存。
    """
    try:
        # 每个子进程都创建自己的 Fetcher 实例，以避免多进程共享一个网络连接
        fetcher = MootdxFetcher()
        stock_code = stock['code']
        stock_name = stock['name']
        
        df = fetcher.fetch_daily(stock_code, stock_name, offset, start, end, is_incremental)
        if df is not None:
            fetcher.save_data(stock_code, stock_name, df)
            return True # 返回 True 表示成功
        return False # 返回 False 表示没有下载到新数据
    except Exception as e:
        log(f"进程处理 {stock.get('code', '未知代码')} 时发生严重异常: {e}", level="ERROR")
    return False

def main():
    """
    脚本的主入口函数。
    负责解析命令行参数、获取待下载的股票列表、分发任务给子进程。
    """
    # --- 1. 解析命令行参数 ---
    parser = argparse.ArgumentParser(
        description="Mootdx 数据抓取工具 (多进程增量合并版)",
        formatter_class=argparse.RawTextHelpFormatter # 保持说明文本格式
    )
    parser.add_argument("--symbol", type=str, help="股票代码，支持多个逗号分隔 (例如: 600519,000001)")
    parser.add_argument("--all", action="store_true", help="选择所有A股作为目标")
    parser.add_argument("--start", type=str, help="开始日期 (格式: YYYYMMDD)，若与--all连用则代表增量更新模式")
    parser.add_argument("--end", type=str, help="结束日期 (格式: YYYYMMDD)")
    parser.add_argument("--limit", type=int, default=None, help="指定抓取的数据天数 (offset)")
    parser.add_argument("--workers", type=int, default=4, help="并行下载的进程数 (默认: 4)")
    args = parser.parse_args()

    # --- 2. 准备待下载的股票列表 ---
    # 仅在主进程中初始化一次 Fetcher，用于获取股票列表
    log("主进程：初始化并准备获取股票列表...")
    init_fetcher = MootdxFetcher()
    
    stocks_to_download = []
    
    # 判断是增量模式还是全量模式
    # 如果用户提供了 --start 或 --end，则认为是增量或范围模式
    is_incremental = bool(args.start or args.end)
    # 根据模式设置默认的抓取天数
    limit = args.limit if args.limit is not None else (800 if is_incremental else 8000)

    # 如果用户既没有指定 --all 也没有指定 --symbol，则打印帮助信息并退出
    if not args.all and not args.symbol:
        log("错误：必须提供 --all 或 --symbol 参数之一。", level="ERROR")
        parser.print_help()
        return

    # 在主进程中准备好所有需要下载的股票信息（代码+名称）
    if args.all:
        stocks_to_download = init_fetcher.get_stock_list()
    elif args.symbol:
        # 即使用户只指定了几个代码，我们仍然获取全量列表，以便从中准确匹配名称
        full_stock_list = init_fetcher.get_stock_list()
        stock_dict = {str(stock['code']): stock for stock in full_stock_list}
        
        for s in args.symbol.split(','):
            symbol = s.strip()
            if symbol in stock_dict:
                stocks_to_download.append(stock_dict[symbol])
            else:
                log(f"警告：股票 {symbol} 未在A股列表中找到，将使用代码作为名称。", level="WARNING")
                stocks_to_download.append({'code': symbol, 'name': symbol, 'market_type': 'unknown'})

    if not stocks_to_download:
        log("未能获取到任何待下载的股票列表，任务终止。", level="ERROR")
        return

    # --- 3. 使用多进程执行下载任务 ---
    log(f"任务启动 -> 模式: {'增量/范围' if is_incremental else '全量'}, 进程数: {args.workers}, 股票数: {len(stocks_to_download)}, 抓取天数: {limit}")
    
    success_count = 0
    # 使用进程池来管理子进程
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务到进程池
        futures = {
            executor.submit(process_single_stock, s, limit, args.start, args.end, is_incremental): s 
            for s in stocks_to_download
        }
        
        # 迭代已完成的任务，获取结果并打印进度
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                success_count += 1
            # 每处理50个或处理完最后一个时，打印一次进度日志
            if (i + 1) % 50 == 0 or (i + 1) == len(stocks_to_download):
                log(f"处理进度: {i+1}/{len(stocks_to_download)}, 已成功: {success_count}")

    log(f"所有任务已完成。总计成功处理: {success_count}/{len(stocks_to_download)}")

# ==============================================================================
# 4. 脚本执行入口
# ==============================================================================
if __name__ == "__main__":
    # 当脚本作为主程序运行时，调用 main() 函数
    main()
