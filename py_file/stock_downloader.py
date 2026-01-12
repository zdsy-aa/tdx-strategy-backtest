#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
股票数据下载工具 (stock_downloader.py)
================================================================================

功能说明:
    1. 支持全量下载 (--full): 下载所有A股股票从上市至今的完整历史数据
    2. 支持增量下载 (--incremental): 仅下载现有数据之后的更新数据
    3. 支持指定股票下载 (--stocks): 下载指定的股票代码列表
    4. 自动停止机制: 如果某只股票在最近10个交易日内没有交易记录，后续将不再下载该股票
    5. 指数下载: 支持下载主要指数数据

数据来源:
    - AKShare: 开源财经数据接口库

存储路径:
    - 日线数据: /data/day/{stock_code}.csv

作者: Manus
版本: 3.2.0
更新日期: 2026-01-12
================================================================================
"""

import os
import sys
import time
import json
import argparse
import threading
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------------------------
# akshare
# ------------------------------------------------------------------------------
try:
    import akshare as ak
except ImportError:
    print("错误: 请先安装 akshare 库")
    print("安装命令: pip install akshare")
    sys.exit(1)

# ==============================================================================
# 配置常量
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
SKIP_LIST_FILE = os.path.join(CONFIG_DIR, 'skip_stocks.json')

EARLIEST_DATE = '19900101'
DOWNLOAD_DELAY = 0.2
MAX_RETRY = 3
DEFAULT_WORKERS = 4

# CSV 固定前置列（顺序锁死）
FIXED_COLUMNS = ['名称', '日期', '开盘', '收盘', '最高', '最低', '成交量']

lock = threading.Lock()

# ==============================================================================
# 基础工具
# ==============================================================================

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

def load_skip_list() -> List[str]:
    if os.path.exists(SKIP_LIST_FILE):
        try:
            with open(SKIP_LIST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return []

def save_skip_list(skip_list: List[str]):
    with open(SKIP_LIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(set(skip_list)), f, ensure_ascii=False, indent=2)

def get_code_name_map() -> Dict[str, str]:
    df = ak.stock_zh_a_spot_em()
    return dict(zip(df['代码'], df['名称']))

def get_all_stock_codes() -> List[str]:
    df = ak.stock_zh_a_spot_em()
    return df['代码'].tolist()

def check_inactive(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    last_date = pd.to_datetime(df['日期']).max()
    return (datetime.now() - last_date).days > 15

# ==============================================================================
# 核心逻辑
# ==============================================================================

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """固定前7列，其余字段自动追加"""
    other_cols = [c for c in df.columns if c not in FIXED_COLUMNS]
    return df[FIXED_COLUMNS + other_cols]

def download_stock_data(
    code: str,
    name: str,
    start_date: str,
    end_date: Optional[str] = None
) -> pd.DataFrame:

    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )

    if df.empty:
        return df

    # 中文字段标准化
    df = df.rename(columns={
        '日期': '日期',
        '开盘': '开盘',
        '收盘': '收盘',
        '最高': '最高',
        '最低': '最低',
        '成交量': '成交量'
    })

    df['日期'] = pd.to_datetime(df['日期'])
    df['名称'] = name

    return df

def process_stock(
    idx: int,
    total: int,
    code: str,
    name: str,
    mode: str,
    skip_list: List[str]
) -> None:

    if code in skip_list:
        return

    filepath = os.path.join(DATA_DIR, f"{code}.csv")
    start_date = EARLIEST_DATE
    old_df = pd.DataFrame()

    if mode == 'incremental' and os.path.exists(filepath):
        try:
            old_df = pd.read_csv(filepath)
            old_df['日期'] = pd.to_datetime(old_df['日期'])
            start_date = old_df['日期'].max().strftime('%Y%m%d')
        except:
            pass

    for attempt in range(1, MAX_RETRY + 1):
        try:
            df = download_stock_data(code, name, start_date)
            if df.empty:
                raise RuntimeError("无返回数据")

            if not old_df.empty:
                df = pd.concat([old_df, df])
                df = df.drop_duplicates(subset=['日期']).sort_values('日期')

            df = reorder_columns(df)
            df.to_csv(filepath, index=False, encoding='utf-8')

            if check_inactive(df):
                with lock:
                    skip_list.append(code)

            print(f"[{idx:>5}/{total}] [OK] {code} {name} rows={len(df)}")
            return

        except Exception as e:
            if attempt < MAX_RETRY:
                print(f"[{idx:>5}/{total}] [RETRY {attempt}] {code} {name} error={e}")
                time.sleep(1)
            else:
                print(f"[{idx:>5}/{total}] [FAIL] {code} {name} error={e}")

# ==============================================================================
# 主入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='股票数据下载工具')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--full', action='store_true')
    group.add_argument('--incremental', action='store_true')
    group.add_argument('--stocks', type=str)
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    ensure_dirs()
    skip_list = load_skip_list()

    code_name_map = get_code_name_map()

    if args.stocks:
        stock_list = args.stocks.split(',')
        mode = 'incremental'
    else:
        stock_list = get_all_stock_codes()
        mode = 'full' if args.full else 'incremental'

    total = len(stock_list)
    print(f"开始下载：模式={mode} 并行={args.workers} 股票数={total}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks = []
        for i, code in enumerate(stock_list, 1):
            name = code_name_map.get(code, '')
            tasks.append(
                executor.submit(
                    process_stock, i, total, code, name, mode, skip_list
                )
            )
            time.sleep(DOWNLOAD_DELAY)

        for _ in as_completed(tasks):
            pass

    save_skip_list(skip_list)
    print("全部任务完成")

if __name__ == "__main__":
    main()
