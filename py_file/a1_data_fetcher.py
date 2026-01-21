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

修复日志:
    - 2026-01-21: 增强API调用健壮性，添加重试机制，改进错误处理和日志

作者: Manus AI
版本: 3.1.0
更新日期: 2026-01-21
================================================================================
"""

try:
    from a99_logger import log
except ImportError:
    def log(msg, level="INFO"): print(f"[{level}] {msg}")

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

warnings.filterwarnings('ignore')

# 尝试导入 akshare
try:
    import akshare as ak
except ImportError:
    log("错误: 请先安装 akshare 库 (pip install akshare)")
    sys.exit(1)

# ==============================================================================
# 配置常量
# ==============================================================================

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据存储根目录
DATA_DIR = os.path.join(BASE_DIR, 'data', 'day')

# 字段顺序定义（标准化列名）
COLUMN_ORDER = [
    '名称', '日期', '开盘', '收盘', '最高', '最低', 
    '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率'
]

# 重试配置
MAX_RETRIES = 3
RETRY_WAIT_BASE = 2  # 指数退避基数

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

def validate_date_format(date_str: str) -> bool:
    """验证日期格式 (YYYYMMDD)"""
    if len(date_str) != 8 or not date_str.isdigit():
        return False
    try:
        datetime.strptime(date_str, '%Y%m%d')
        return True
    except ValueError:
        return False

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化列名，支持多种AKShare API返回的列名格式。
    
    功能说明:
        AKShare库的API经常变更，返回的列名可能不同。
        本函数支持多种列名格式的自动识别和转换。
    """
    if df.empty:
        return df
    
    # 定义可能的列名映射（从AKShare可能的列名到标准列名）
    column_mapping = {
        # 日期
        '日期': '日期', 'date': '日期', 'Date': '日期', '时间': '日期',
        # 开盘
        '开盘': '开盘', 'open': '开盘', 'Open': '开盘',
        # 最高
        '最高': '最高', 'high': '最高', 'High': '最高',
        # 最低
        '最低': '最低', 'low': '最低', 'Low': '最低',
        # 收盘
        '收盘': '收盘', 'close': '收盘', 'Close': '收盘',
        # 成交量
        '成交量': '成交量', 'volume': '成交量', 'Volume': '成交量', 'vol': '成交量',
        # 成交额
        '成交额': '成交额', 'amount': '成交额', 'Amount': '成交额', 'money': '成交额',
        # 振幅
        '振幅': '振幅', 'amplitude': '振幅', 'Amplitude': '振幅',
        # 涨跌幅
        '涨跌幅': '涨跌幅', 'change_pct': '涨跌幅', 'Change_pct': '涨跌幅', 'pct_change': '涨跌幅',
        # 涨跌额
        '涨跌额': '涨跌额', 'change': '涨跌额', 'Change': '涨跌额',
        # 换手率
        '换手率': '换手率', 'turnover': '换手率', 'Turnover': '换手率', 'turnover_rate': '换手率'
    }
    
    # 重命名存在的列
    rename_dict = {}
    for old_col in df.columns:
        if old_col in column_mapping:
            rename_dict[old_col] = column_mapping[old_col]
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    return df

def get_all_stock_info() -> pd.DataFrame:
    """
    获取所有股票的代码和名称映射。
    
    功能说明:
        从AKShare获取全市场股票列表，支持多种列名格式。
    
    返回:
        pd.DataFrame: 包含'代码'和'名称'两列的DataFrame
    """
    try:
        log("正在获取全市场股票列表...")
        df = ak.stock_zh_a_spot_em()
        
        if df.empty:
            log("警告：获取的股票列表为空")
            return pd.DataFrame()
        
        # 尝试多种列名组合
        if '代码' in df.columns and '名称' in df.columns:
            return df[['代码', '名称']].copy()
        elif 'symbol' in df.columns and 'name' in df.columns:
            result = df[['symbol', 'name']].copy()
            result.columns = ['代码', '名称']
            return result
        elif '代码' in df.columns and '股票简称' in df.columns:
            result = df[['代码', '股票简称']].copy()
            result.columns = ['代码', '名称']
            return result
        else:
            log(f"警告：股票列表列名不匹配")
            log(f"实际列名: {df.columns.tolist()}")
            log(f"尝试自动识别...")
            
            # 尝试自动识别
            code_col = None
            name_col = None
            
            for col in df.columns:
                if any(x in col.lower() for x in ['code', 'symbol', '代码']):
                    code_col = col
                if any(x in col.lower() for x in ['name', 'title', '名称', '简称']):
                    name_col = col
            
            if code_col and name_col:
                result = df[[code_col, name_col]].copy()
                result.columns = ['代码', '名称']
                log(f"自动识别成功: {code_col} -> 代码, {name_col} -> 名称")
                return result
            else:
                log("自动识别失败")
                return pd.DataFrame()
            
    except Exception as e:
        log(f"获取股票列表失败: {e}", level="ERROR")
        return pd.DataFrame()

def download_with_retry(stock_code: str, start_date: str, end_date: str, max_retries: int = MAX_RETRIES) -> Optional[pd.DataFrame]:
    """
    带重试机制的数据下载函数。
    
    功能说明:
        使用指数退避策略重试失败的下载请求。
    
    参数:
        stock_code (str): 股票代码
        start_date (str): 开始日期 (YYYYMMDD)
        end_date (str): 结束日期 (YYYYMMDD)
        max_retries (int): 最大重试次数
    
    返回:
        Optional[pd.DataFrame]: 下载的数据，失败返回None
    """
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            
            if df is not None and not df.empty:
                return df
            else:
                return None
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = RETRY_WAIT_BASE ** attempt  # 指数退避: 2, 4, 8...
                log(f"下载{stock_code}失败，{wait_time}秒后重试... (尝试 {attempt+1}/{max_retries}): {str(e)[:50]}")
                time.sleep(wait_time)
            else:
                log(f"下载{stock_code}失败，已达最大重试次数: {str(e)[:100]}", level="ERROR")
                return None
    
    return None

# ==============================================================================
# 核心下载函数
# ==============================================================================

def download_single_stock(stock_info: Dict, start_date: str, end_date: str, mode: str = 'update') -> bool:
    """
    下载单只股票数据并保存。
    
    功能说明:
        从AKShare下载单只股票的日线数据，支持全量覆盖和增量更新两种模式。
        包含完整的错误处理、数据验证和日志记录。
    
    参数:
        stock_info (Dict): 包含'代码'和'名称'的字典
        start_date (str): 开始日期 (YYYYMMDD)
        end_date (str): 结束日期 (YYYYMMDD)
        mode (str): 'full' (覆盖) 或 'update' (追加)
    
    返回:
        bool: 下载成功返回True，失败返回False
    """
    stock_code = stock_info.get('代码')
    stock_name = stock_info.get('名称')
    
    if not stock_code or not stock_name:
        log(f"错误：股票信息不完整: {stock_info}", level="ERROR")
        return False
    
    market = get_market_folder(stock_code)
    market_dir = os.path.join(DATA_DIR, market)
    ensure_dir(market_dir)
    filepath = os.path.join(market_dir, f"{stock_code}.csv")

    try:
        # 获取历史数据（带重试）
        df = download_with_retry(stock_code, start_date, end_date)
        
        if df is None or df.empty:
            log(f"警告: {stock_code} 返回空数据")
            return False

        # 标准化列名
        df = standardize_columns(df)
        
        # 检查必需的列
        required_cols = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            log(f"错误: {stock_code} 缺少列: {missing_cols}，实际列: {df.columns.tolist()}", level="ERROR")
            return False
        
        # 添加名称列
        df['名称'] = stock_name
        
        # 选择需要的列（只选择存在的列）
        available_cols = ['名称', '日期'] + [col for col in COLUMN_ORDER[2:] if col in df.columns]
        df = df[available_cols]
        
        # 数据验证：检查价格和成交量的有效性
        try:
            df['开盘'] = pd.to_numeric(df['开盘'], errors='coerce')
            df['收盘'] = pd.to_numeric(df['收盘'], errors='coerce')
            df['最高'] = pd.to_numeric(df['最高'], errors='coerce')
            df['最低'] = pd.to_numeric(df['最低'], errors='coerce')
            df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce')
            
            # 移除无效数据行
            df = df.dropna(subset=['开盘', '收盘', '最高', '最低', '成交量'])
            
            if df.empty:
                log(f"警告: {stock_code} 数据验证后为空")
                return False
                
        except Exception as e:
            log(f"错误: {stock_code} 数据验证失败: {e}", level="ERROR")
            return False

        # 保存数据
        if mode == 'update' and os.path.exists(filepath):
            try:
                old_df = pd.read_csv(filepath)
                combined_df = pd.concat([old_df, df], ignore_index=True)
                combined_df['日期'] = pd.to_datetime(combined_df['日期'], errors='coerce')
                combined_df = combined_df.drop_duplicates(subset=['日期'], keep='last')
                combined_df = combined_df.sort_values('日期')
                combined_df['日期'] = combined_df['日期'].dt.strftime('%Y-%m-%d')
                combined_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            except Exception as e:
                log(f"警告: {stock_code} 合并旧数据失败，使用新数据覆盖: {e}")
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        return True
        
    except Exception as e:
        log(f"下载 {stock_code} 失败: {e}", level="ERROR")
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
        log("错误：无法获取股票列表，程序退出", level="ERROR")
        return
    
    log(f"成功获取 {len(stock_df)} 只股票信息")
    
    if args.limit:
        stock_df = stock_df.head(args.limit)
        log(f"限制为前 {args.limit} 只股票")
    
    stocks = stock_df.to_dict('records')
    total = len(stocks)
    
    # 确定日期范围和模式
    today_str = datetime.now().strftime('%Y%m%d')
    if args.full:
        start_date = '19900101'
        end_date = today_str
        mode = 'full'
        log(f"开始全量下载 {total} 只股票历史数据...")
    elif args.today:
        start_date = today_str
        end_date = today_str
        mode = 'update'
        log(f"开始更新 {total} 只股票当日数据...")
    elif args.date:
        start_date, end_date = args.date
        
        # 验证日期格式
        if not validate_date_format(start_date) or not validate_date_format(end_date):
            log(f"错误：日期格式不正确，应为YYYYMMDD格式", level="ERROR")
            return
        
        mode = 'update'
        log(f"开始下载 {total} 只股票日期范围 {start_date} - {end_date} 数据...")
    else:
        parser.print_help()
        return

    # 使用多进程下载
    num_workers = max(1, cpu_count() - 1)
    log(f"使用 {num_workers} 个进程并行下载...")
    
    # 包装下载函数
    download_func = partial(download_single_stock, start_date=start_date, end_date=end_date, mode=mode)
    
    start_time = time.time()
    success_count = 0
    
    try:
        with Pool(num_workers) as pool:
            results = pool.map(download_func, stocks)
            success_count = sum(results)
    except Exception as e:
        log(f"多进程下载出错: {e}", level="ERROR")
        return
    
    end_time = time.time()
    duration = end_time - start_time
    
    # 输出统计结果
    log("\n" + "="*60)
    log("下载任务完成!")
    log(f"总数: {total}")
    log(f"成功: {success_count}")
    log(f"失败: {total - success_count}")
    log(f"成功率: {success_count/total*100:.1f}%")
    log(f"耗时: {duration:.2f} 秒")
    log(f"平均速度: {total/duration:.1f} 只/秒")
    log(f"数据存储目录: {DATA_DIR}")
    log("="*60 + "\n")

if __name__ == "__main__":
    main()
