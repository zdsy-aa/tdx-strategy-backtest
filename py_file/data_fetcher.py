#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
日线数据下载模块 (data_fetcher.py)
================================================================================

功能说明:
    本模块用于从 AKShare 获取 A 股日线数据，支持：
    1. 下载指定股票的历史日线数据
    2. 下载当天最新日线数据
    3. 批量下载多只股票数据
    4. 数据增量更新

数据来源:
    - AKShare: 开源财经数据接口库 (https://akshare.xyz/)

存储路径:
    - 日线数据: /data/day/{stock_code}.csv

依赖库:
    - akshare: A股数据获取
    - pandas: 数据处理

作者: TradeGuide System
版本: 2.0.0
更新日期: 2026-01-07
================================================================================
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import time

# 尝试导入 akshare，如果失败则给出提示
try:
    import akshare as ak
except ImportError:
    print("错误: 请先安装 akshare 库")
    print("安装命令: pip install akshare")
    sys.exit(1)


# ==============================================================================
# 配置常量
# ==============================================================================

# 数据存储根目录 (相对于项目根目录)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'day')

# 默认股票列表 (沪深300成分股的部分代表)
DEFAULT_STOCKS = [
    '000001',  # 平安银行
    '000002',  # 万科A
    '000063',  # 中兴通讯
    '000333',  # 美的集团
    '000651',  # 格力电器
    '000858',  # 五粮液
    '600000',  # 浦发银行
    '600036',  # 招商银行
    '600519',  # 贵州茅台
    '600887',  # 伊利股份
]

# 指数列表
DEFAULT_INDICES = [
    'sh000001',  # 上证指数
    'sh000300',  # 沪深300
    'sz399001',  # 深证成指
    'sz399006',  # 创业板指
]


# ==============================================================================
# 核心函数
# ==============================================================================

def ensure_data_dir() -> str:
    """
    确保数据目录存在
    
    返回:
        str: 数据目录的绝对路径
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def get_stock_data(
    stock_code: str,
    start_date: str = '20160101',
    end_date: Optional[str] = None,
    adjust: str = 'qfq'
) -> pd.DataFrame:
    """
    获取单只股票的日线数据
    
    参数:
        stock_code: 股票代码 (如 '000001', '600519')
        start_date: 开始日期，格式 'YYYYMMDD'，默认 '20160101'
        end_date: 结束日期，格式 'YYYYMMDD'，默认为今天
        adjust: 复权类型
            - 'qfq': 前复权 (推荐，用于技术分析)
            - 'hfq': 后复权
            - '': 不复权
            
    返回:
        pd.DataFrame: 包含以下列的日线数据
            - date: 日期
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量
            - amount: 成交额
            
    异常:
        如果获取失败，返回空 DataFrame
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    try:
        # 使用 akshare 获取日线数据
        # 注意: akshare 的股票代码不需要市场前缀
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )
        
        if df.empty:
            print(f"警告: 股票 {stock_code} 没有数据")
            return pd.DataFrame()
        
        # 标准化列名
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover'
        })
        
        # 确保日期列是 datetime 类型
        df['date'] = pd.to_datetime(df['date'])
        
        # 添加股票代码列
        df['code'] = stock_code
        
        return df
        
    except Exception as e:
        print(f"错误: 获取股票 {stock_code} 数据失败 - {str(e)}")
        return pd.DataFrame()


def get_index_data(
    index_code: str,
    start_date: str = '20160101',
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取指数日线数据
    
    参数:
        index_code: 指数代码 (如 'sh000001', 'sh000300')
        start_date: 开始日期，格式 'YYYYMMDD'
        end_date: 结束日期，格式 'YYYYMMDD'
        
    返回:
        pd.DataFrame: 指数日线数据
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    try:
        # 使用 akshare 获取指数数据
        df = ak.stock_zh_index_daily(symbol=index_code)
        
        if df.empty:
            print(f"警告: 指数 {index_code} 没有数据")
            return pd.DataFrame()
        
        # 标准化列名
        df = df.rename(columns={
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        # 确保日期列是 datetime 类型
        df['date'] = pd.to_datetime(df['date'])
        
        # 过滤日期范围
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        
        # 添加代码列
        df['code'] = index_code
        
        return df
        
    except Exception as e:
        print(f"错误: 获取指数 {index_code} 数据失败 - {str(e)}")
        return pd.DataFrame()


def save_data(df: pd.DataFrame, filename: str) -> bool:
    """
    保存数据到 CSV 文件
    
    参数:
        df: 要保存的 DataFrame
        filename: 文件名 (不含路径)
        
    返回:
        bool: 保存是否成功
    """
    ensure_data_dir()
    filepath = os.path.join(DATA_DIR, filename)
    
    try:
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"数据已保存: {filepath}")
        return True
    except Exception as e:
        print(f"错误: 保存文件失败 - {str(e)}")
        return False


def load_data(filename: str) -> pd.DataFrame:
    """
    从 CSV 文件加载数据
    
    参数:
        filename: 文件名 (不含路径)
        
    返回:
        pd.DataFrame: 加载的数据，如果文件不存在则返回空 DataFrame
    """
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"警告: 文件不存在 - {filepath}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath, parse_dates=['date'])
        return df
    except Exception as e:
        print(f"错误: 加载文件失败 - {str(e)}")
        return pd.DataFrame()


def update_stock_data(stock_code: str) -> Tuple[bool, int]:
    """
    增量更新股票数据
    
    如果本地已有数据，只下载缺失的部分；
    如果本地没有数据，下载全部历史数据。
    
    参数:
        stock_code: 股票代码
        
    返回:
        Tuple[bool, int]: (是否成功, 新增数据条数)
    """
    filename = f"{stock_code}.csv"
    existing_df = load_data(filename)
    
    if existing_df.empty:
        # 没有历史数据，下载全部
        print(f"下载 {stock_code} 全部历史数据...")
        new_df = get_stock_data(stock_code)
        if not new_df.empty:
            save_data(new_df, filename)
            return True, len(new_df)
        return False, 0
    
    # 有历史数据，增量更新
    last_date = existing_df['date'].max()
    start_date = (last_date + timedelta(days=1)).strftime('%Y%m%d')
    today = datetime.now().strftime('%Y%m%d')
    
    if start_date > today:
        print(f"股票 {stock_code} 数据已是最新")
        return True, 0
    
    print(f"更新 {stock_code} 数据: {start_date} 至 {today}...")
    new_df = get_stock_data(stock_code, start_date=start_date)
    
    if new_df.empty:
        print(f"股票 {stock_code} 没有新数据")
        return True, 0
    
    # 合并数据
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
    combined_df = combined_df.sort_values('date')
    
    save_data(combined_df, filename)
    return True, len(new_df)


def batch_download(
    stock_list: Optional[List[str]] = None,
    include_indices: bool = True,
    delay: float = 0.5
) -> dict:
    """
    批量下载股票数据
    
    参数:
        stock_list: 股票代码列表，默认使用 DEFAULT_STOCKS
        include_indices: 是否包含指数数据
        delay: 每次请求之间的延迟 (秒)，避免被限流
        
    返回:
        dict: 下载结果统计
            - success: 成功数量
            - failed: 失败数量
            - total_rows: 总数据行数
    """
    if stock_list is None:
        stock_list = DEFAULT_STOCKS
    
    results = {
        'success': 0,
        'failed': 0,
        'total_rows': 0
    }
    
    print("=" * 60)
    print("开始批量下载股票数据")
    print("=" * 60)
    
    # 下载股票数据
    for i, code in enumerate(stock_list, 1):
        print(f"\n[{i}/{len(stock_list)}] 处理股票: {code}")
        success, rows = update_stock_data(code)
        
        if success:
            results['success'] += 1
            results['total_rows'] += rows
        else:
            results['failed'] += 1
        
        time.sleep(delay)
    
    # 下载指数数据
    if include_indices:
        print("\n" + "=" * 60)
        print("开始下载指数数据")
        print("=" * 60)
        
        for i, code in enumerate(DEFAULT_INDICES, 1):
            print(f"\n[{i}/{len(DEFAULT_INDICES)}] 处理指数: {code}")
            df = get_index_data(code)
            
            if not df.empty:
                save_data(df, f"{code}.csv")
                results['success'] += 1
                results['total_rows'] += len(df)
            else:
                results['failed'] += 1
            
            time.sleep(delay)
    
    print("\n" + "=" * 60)
    print("下载完成!")
    print(f"成功: {results['success']}, 失败: {results['failed']}")
    print(f"总数据行数: {results['total_rows']}")
    print("=" * 60)
    
    return results


def download_today_data(stock_list: Optional[List[str]] = None) -> dict:
    """
    下载当天日线数据 (用于定时任务)
    
    此函数专门用于每日收盘后更新数据，
    只下载当天的数据并追加到历史文件中。
    
    参数:
        stock_list: 股票代码列表，默认使用 DEFAULT_STOCKS
        
    返回:
        dict: 更新结果统计
    """
    if stock_list is None:
        stock_list = DEFAULT_STOCKS + DEFAULT_INDICES
    
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"开始更新 {today} 的日线数据...")
    
    results = {
        'updated': 0,
        'skipped': 0,
        'failed': 0
    }
    
    for code in stock_list:
        success, rows = update_stock_data(code)
        
        if success and rows > 0:
            results['updated'] += 1
        elif success:
            results['skipped'] += 1
        else:
            results['failed'] += 1
        
        time.sleep(0.3)
    
    print(f"\n更新完成: 更新 {results['updated']}, 跳过 {results['skipped']}, 失败 {results['failed']}")
    return results


# ==============================================================================
# 命令行接口
# ==============================================================================

def main():
    """
    命令行入口函数
    
    用法:
        python data_fetcher.py                  # 批量下载默认股票
        python data_fetcher.py --today          # 只更新当天数据
        python data_fetcher.py --stock 000001   # 下载指定股票
        python data_fetcher.py --index sh000300 # 下载指定指数
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='A股日线数据下载工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python data_fetcher.py                  # 批量下载默认股票
    python data_fetcher.py --today          # 只更新当天数据
    python data_fetcher.py --stock 000001   # 下载指定股票
    python data_fetcher.py --index sh000300 # 下载指定指数
        """
    )
    
    parser.add_argument('--today', action='store_true', help='只更新当天数据')
    parser.add_argument('--stock', type=str, help='下载指定股票代码')
    parser.add_argument('--index', type=str, help='下载指定指数代码')
    parser.add_argument('--all', action='store_true', help='下载所有默认股票和指数')
    
    args = parser.parse_args()
    
    if args.today:
        download_today_data()
    elif args.stock:
        success, rows = update_stock_data(args.stock)
        if success:
            print(f"股票 {args.stock} 下载成功，共 {rows} 条数据")
        else:
            print(f"股票 {args.stock} 下载失败")
    elif args.index:
        df = get_index_data(args.index)
        if not df.empty:
            save_data(df, f"{args.index}.csv")
            print(f"指数 {args.index} 下载成功，共 {len(df)} 条数据")
        else:
            print(f"指数 {args.index} 下载失败")
    else:
        batch_download()


if __name__ == "__main__":
    main()
