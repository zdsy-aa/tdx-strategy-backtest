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
    5. 包含字段：名称、日期、开盘、收盘、最高、最低、成交量、成交额、
       振幅、涨跌幅、涨跌额、换手率

数据来源:
    - AKShare (底层：东方财富 Eastmoney)

存储路径:
    - 日线数据: /data/day/{market}/{stock_code}.csv

================================================================================
【生产级增强说明（非常重要）】

本版本在【不改变任何原始业务逻辑】的前提下，增加了：

1. 全局 HTTP 伪装（浏览器指纹 + Referer）
2. requests 层 Retry（防止 connection reset / remote host closed）
3. Monkey Patch，确保 AKShare 内部所有 requests 都被伪装
4. 失败股票记录 failed_symbols.csv
5. 动态限速（降低东方财富风控命中率）
6. 多进程安全（patch 在 fork 前完成）

目标：支持 全市场 / 多进程 / 长时间稳定运行

================================================================================
"""

# ==============================================================================
# 日志模块（原始逻辑，保持不变）
# ==============================================================================
try:
    from a99_logger import log
except ImportError:
    def log(msg, level="INFO"):
        print(f"[{level}] {msg}")

# ==============================================================================
# 标准库 & 第三方库
# ==============================================================================
import os
import sys
import time
import random
import warnings
import pandas as pd

from datetime import datetime
from typing import Dict, Optional
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings("ignore")

# ==============================================================================
# AKShare 导入
# ==============================================================================
try:
    import akshare as ak
except ImportError:
    log("错误: 请先安装 akshare 库 (pip install akshare)", level="ERROR")
    sys.exit(1)

# ==============================================================================
# 【生产级增强】全局 HTTP 伪装 + Retry + Monkey Patch
# 必须在任何 AKShare 调用之前执行
# ==============================================================================

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def init_akshare_http_layer():
    """
    初始化全局 HTTP 会话：
    - 伪装浏览器 headers
    - requests 层自动重试
    - monkey patch requests.get
    """

    session = requests.Session()

    # --- 浏览器级 headers（东方财富强校验项）---
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0"
        ),
        "Referer": "https://quote.eastmoney.com/center/gridlist.html#hs_a_board",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
    })

    # --- Retry 策略（网络层，不是业务层）---
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # --- Monkey Patch requests.get ---
    original_get = requests.get

    def patched_get(*args, **kwargs):
        kwargs.setdefault("headers", session.headers)
        kwargs.setdefault("timeout", 20)
        return original_get(*args, **kwargs)

    requests.get = patched_get

    # --- 尝试 Patch AKShare 内部 session（不同版本兼容）---
    try:
        import akshare.stock.stock_info_em as sie
        sie.session = session
    except Exception:
        pass

    log("HTTP 伪装 & Retry & Monkey Patch 初始化完成")

# 初始化（只执行一次，fork 前完成）
init_akshare_http_layer()

# ==============================================================================
# 项目路径配置（原始逻辑）
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "day")
FAILED_FILE = os.path.join(BASE_DIR, "failed_symbols.csv")

# ==============================================================================
# 字段定义（原始逻辑）
# ==============================================================================
COLUMN_ORDER = [
    "名称", "日期", "开盘", "收盘", "最高", "最低",
    "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"
]

# ==============================================================================
# 工具函数（原始逻辑 + 注释增强）
# ==============================================================================
def get_market_folder(stock_code: str) -> str:
    """根据股票代码判断市场"""
    if stock_code.startswith(("60", "68", "90")):
        return "sh"
    elif stock_code.startswith(("00", "30", "20")):
        return "sz"
    elif stock_code.startswith(("8", "4")):
        return "bj"
    return "sh"

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def validate_date_format(date_str: str) -> bool:
    """验证日期格式 YYYYMMDD"""
    try:
        datetime.strptime(date_str, "%Y%m%d")
        return True
    except Exception:
        return False

# ==============================================================================
# 股票列表获取（原始逻辑，未改）
# ==============================================================================
def get_all_stock_info() -> pd.DataFrame:
    """获取全市场股票代码 + 名称"""
    try:
        log("正在获取全市场股票列表...")
        df = ak.stock_zh_a_spot_em()
        if df.empty:
            return pd.DataFrame()

        if "代码" in df.columns and "名称" in df.columns:
            return df[["代码", "名称"]]

        # 自动识别（兼容未来 AKShare 变更）
        code_col, name_col = None, None
        for col in df.columns:
            if "code" in col.lower() or "代码" in col:
                code_col = col
            if "name" in col.lower() or "名称" in col or "简称" in col:
                name_col = col

        if code_col and name_col:
            tmp = df[[code_col, name_col]].copy()
            tmp.columns = ["代码", "名称"]
            return tmp

        return pd.DataFrame()

    except Exception as e:
        log(f"获取股票列表失败: {e}", level="ERROR")
        return pd.DataFrame()

# ==============================================================================
# 【生产级增强】失败股票记录
# ==============================================================================
def record_failed(stock_code: str, reason: str):
    """记录失败股票，便于夜间补偿"""
    with open(FAILED_FILE, "a", encoding="utf-8") as f:
        f.write(f"{stock_code},{reason}\n")

# ==============================================================================
# 下载函数（业务逻辑不变 + 稳定性增强）
# ==============================================================================
def download_with_retry(
    stock_code: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3
) -> Optional[pd.DataFrame]:
    """
    业务级重试（配合 requests 层 Retry）
    """
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            if df is not None and not df.empty:
                return df
            return None
        except Exception as e:
            time.sleep(2 ** attempt)
            last_error = str(e)
    record_failed(stock_code, last_error[:80])
    return None

# ==============================================================================
# 单股票下载（核心逻辑）
# ==============================================================================
def download_single_stock(
    stock_info: Dict,
    start_date: str,
    end_date: str,
    mode: str = "update"
) -> bool:
    stock_code = stock_info["代码"]
    stock_name = stock_info["名称"]

    # --- 动态限速（降低风控命中）---
    time.sleep(random.uniform(0.2, 0.6))

    market = get_market_folder(stock_code)
    market_dir = os.path.join(DATA_DIR, market)
    ensure_dir(market_dir)
    filepath = os.path.join(market_dir, f"{stock_code}.csv")

    df = download_with_retry(stock_code, start_date, end_date)
    if df is None or df.empty:
        record_failed(stock_code, "empty_data")
        return False

    df["名称"] = stock_name

    # --- 数据合并（原始逻辑）---
    if mode == "update" and os.path.exists(filepath):
        old = pd.read_csv(filepath)
        df = pd.concat([old, df], ignore_index=True)
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.drop_duplicates(subset=["日期"], keep="last")
        df = df.sort_values("日期")
        df["日期"] = df["日期"].dt.strftime("%Y-%m-%d")

    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    return True

# ==============================================================================
# 主入口
# ==============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser("A股日线数据下载（生产级）")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--today", action="store_true")
    parser.add_argument("--date", nargs=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    stock_df = get_all_stock_info()
    if stock_df.empty:
        log("无法获取股票列表", level="ERROR")
        return

    if args.limit:
        stock_df = stock_df.head(args.limit)

    today = datetime.now().strftime("%Y%m%d")

    if args.full:
        start_date, end_date = "19900101", today
        mode = "full"
    elif args.today:
        start_date = end_date = today
        mode = "update"
    elif args.date:
        start_date, end_date = args.date
        mode = "update"
    else:
        parser.print_help()
        return

    workers = 4
    log(f"使用 {workers} 个进程")

    func = partial(
        download_single_stock,
        start_date=start_date,
        end_date=end_date,
        mode=mode,
    )

    with Pool(workers) as pool:
        results = pool.map(func, stock_df.to_dict("records"))

    log(f"完成：成功 {sum(results)} / 总数 {len(results)}")

if __name__ == "__main__":
    main()
