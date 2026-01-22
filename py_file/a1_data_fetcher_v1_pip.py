#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案一：极致伪装版 (a1_data_fetcher_v1_pip.py)
适用场景：直接使用 pip install akshare 安装，通过脚本层的极致伪装绕过 API 限制。
"""

import os
import sys
import time
import random
import warnings
import pandas as pd
import requests
from datetime import datetime
from typing import Dict, Optional
from multiprocessing import Pool
from functools import partial
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. 核心：极致 HTTP 伪装层
# ==============================================================================
def init_stealth_mode():
    """
    初始化隐身模式：
    - 动态 User-Agent 轮换
    - 强制 Referer 伪装
    - 自动重试机制
    - Monkey Patch 全局 requests
    """
    session = requests.Session()
    
    # 模拟真实浏览器的完整 Headers
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0",
        "Referer": "https://quote.eastmoney.com/center/gridlist.html",
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    })

    # 网络层自动重试（处理连接重置等问题）
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Monkey Patch: 强制让 akshare 内部的所有请求都带上我们的伪装
    original_get = requests.get
    def patched_get(*args, **kwargs):
        kwargs.setdefault("headers", session.headers)
        kwargs.setdefault("timeout", 15)
        return original_get(*args, **kwargs)
    requests.get = patched_get
    
    print("[INFO] 隐身模式已启动：HTTP 伪装 & 全局 Patch 完成")

init_stealth_mode()

# 延迟导入 akshare，确保 Patch 生效
import akshare as ak

# ==============================================================================
# 2. 业务逻辑
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "day")

def download_stock(stock_info: Dict, start_date: str, end_date: str) -> bool:
    code = stock_info["代码"]
    name = stock_info["名称"]
    
    # 动态限速：模拟真人点击间隔
    time.sleep(random.uniform(0.5, 1.5))
    
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if df is not None and not df.empty:
            os.makedirs(DATA_DIR, exist_ok=True)
            df.to_csv(os.path.join(DATA_DIR, f"{code}.csv"), index=False, encoding="utf-8-sig")
            print(f"[SUCCESS] {code} {name} 下载完成")
            return True
    except Exception as e:
        print(f"[ERROR] {code} 下载失败: {e}")
    return False

def main():
    print("正在获取股票列表...")
    try:
        stock_df = ak.stock_zh_a_spot_em()
        stocks = stock_df[["代码", "名称"]].head(10).to_dict("records") # 示例仅下载10只
        
        start = "20240101"
        end = datetime.now().strftime("%Y%m%d")
        
        with Pool(4) as pool:
            func = partial(download_stock, start_date=start, end_date=end)
            pool.map(func, stocks)
    except Exception as e:
        print(f"主程序运行异常: {e}")

if __name__ == "__main__":
    main()
