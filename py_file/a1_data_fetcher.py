#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
日线数据下载模块 (a1_data_fetcher.py)
================================================================================

功能说明:
    本模块用于从 AKShare 获取 A 股日线数据，支持：
    1. 全量下载所有股票历史数据 (--full)
    2. 增量更新当天最新数据 (--today)
    3. 下载指定日期范围数据 (--date)
    4. 按市场分类存储 (sh/sz/bj)
    5. 包含字段：名称、日期、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌幅、涨跌额、换手率

数据来源:
    - AKShare: 开源财经数据接口库 (https://akshare.xyz/)

存储路径:
    - 日线数据: /data/day/{market}/{stock_code}.csv

作者: Manus AI
版本: 3.0.0
更新日期: 2026-01-16
================================================================================
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# 尝试导入 akshare
try:
    import akshare as ak
except ImportError:
    print("错误: 请先安装 akshare 库 (pip install akshare)")
    sys.exit(1)

# ==============================================================================
# 配置常量
# ==============================================================================

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据存储根目录
DATA_DIR = os.path.join(BASE_DIR, 'data', 'day')

# 字段顺序定义
COLUMN_ORDER = [
    '名称', '日期', '开盘', '收盘', '最高', '最低', 
    '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率'
]

# ==============================================================================
# 辅助函数
# ==============================================================================

def get_market_folder(stock_code: str) -> str:
    """根据股票代码判断市场并返回对应文件夹名称"""
    if stock_code.startswith(('60', '68', '90')):
        return 'sh'
    elif stock_code.startswith(('00', '30', '20')):
        return 'sz'
    elif stock_code.startswith(('8', '4')):
        return 'bj'
    else:
        return 'sh'  # 默认

def ensure_dir(path: str):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def get_all_stock_info() -> pd.DataFrame:
    """获取所有股票的代码和名称映射"""
    try:
        print("正在获取全市场股票列表...")
        df = ak.stock_zh_a_spot_em()
        return df[['代码', '名称']]
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return pd.DataFrame()

# ==============================================================================
# 核心下载函数
# ==============================================================================

def download_single_stock(stock_info: Dict, start_date: str, end_date: str, mode: str = 'update') -> bool:
    """
    下载单只股票数据并保存
    mode: 'full' (覆盖), 'update' (追加)
    """
    stock_code = stock_info['代码']
    stock_name = stock_info['名称']
    market = get_market_folder(stock_code)
    market_dir = os.path.join(DATA_DIR, market)
    ensure_dir(market_dir)
    filepath = os.path.join(market_dir, f"{stock_code}.csv")

    try:
        # 获取历史数据
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        if df.empty:
            return False

        # 统一列名
        df = df.rename(columns={
            '日期': '日期',
            '开盘': '开盘',
            '最高': '最高',
            '最低': '最低',
            '收盘': '收盘',
            '成交量': '成交量',
            '成交额': '成交额',
            '振幅': '振幅',
            '涨跌幅': '涨跌幅',
            '涨跌额': '涨跌额',
            '换手率': '换手率'
        })
        
        # 添加名称列
        df['名称'] = stock_name
        
        # 整理列顺序
        df = df[COLUMN_ORDER]

        if mode == 'update' and os.path.exists(filepath):
            old_df = pd.read_csv(filepath)
            # 合并并去重
            combined_df = pd.concat([old_df, df], ignore_index=True)
            combined_df['日期'] = pd.to_datetime(combined_df['日期'])
            combined_df = combined_df.drop_duplicates(subset=['日期'], keep='last')
            combined_df = combined_df.sort_values('日期')
            combined_df['日期'] = combined_df['日期'].dt.strftime('%Y-%m-%d')
            combined_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        return True
    except Exception as e:
        # print(f"下载 {stock_code} 失败: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='A股日线数据下载工具')
    parser.add_argument('--full', action='store_true', help='全量下载所有历史数据')
    parser.add_argument('--today', action='store_true', help='仅下载当天数据')
    parser.add_argument('--date', nargs=2, metavar=('START', 'END'), help='下载指定日期范围数据 (YYYYMMDD)')
    parser.add_argument('--limit', type=int, help='限制下载股票数量 (用于测试)')
    
    args = parser.parse_args()

    # 获取股票列表
    stock_df = get_all_stock_info()
    if stock_df.empty:
        return
    
    if args.limit:
        stock_df = stock_df.head(args.limit)
    
    stocks = stock_df.to_dict('records')
    total = len(stocks)
    
    # 确定日期范围和模式
    today_str = datetime.now().strftime('%Y%m%d')
    if args.full:
        start_date = '19900101'
        end_date = today_str
        mode = 'full'
        print(f"开始全量下载 {total} 只股票历史数据...")
    elif args.today:
        start_date = today_str
        end_date = today_str
        mode = 'update'
        print(f"开始更新 {total} 只股票当日数据...")
    elif args.date:
        start_date, end_date = args.date
        mode = 'update'
        print(f"开始下载 {total} 只股票日期范围 {start_date} - {end_date} 数据...")
    else:
        parser.print_help()
        return

    # 使用多进程下载
    # 并行数量为 CPU核心数 - 1
    num_workers = max(1, cpu_count() - 1)
    
    # 包装下载函数
    download_func = partial(download_single_stock, start_date=start_date, end_date=end_date, mode=mode)
    
    start_time = time.time()
    success_count = 0
    
    with Pool(num_workers) as pool:
        results = pool.map(download_func, stocks)
        success_count = sum(results)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*50)
    print(f"下载任务完成!")
    print(f"总数: {total}")
    print(f"成功: {success_count}")
    print(f"失败: {total - success_count}")
    print(f"耗时: {duration:.2f} 秒")
    print(f"数据存储目录: {DATA_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()
