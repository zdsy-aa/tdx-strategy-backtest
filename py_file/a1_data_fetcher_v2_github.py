#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案二：源码集成版 (a1_data_fetcher_v2_github.py)
适用场景：当 pip 版本失效时，通过 git clone 下载 akshare 源码到本地，脚本会自动优先加载源码。
"""

import os
import sys
import time
import random
import warnings
import pandas as pd

# ==============================================================================
# 1. 核心：优先加载本地源码逻辑
# ==============================================================================
def load_local_akshare():
    """
    自动检测并加载本地 akshare 源码。
    请确保您已执行：git clone https://github.com/akfamily/akshare.git akshare_lib
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设您将源码克隆到了脚本同级目录下的 akshare_lib 文件夹
    local_path = os.path.join(current_dir, "akshare_lib")
    
    if os.path.exists(local_path):
        sys.path.insert(0, local_path)
        print(f"[INFO] 检测到本地源码，已优先加载：{local_path}")
    else:
        print("[WARN] 未检测到本地 akshare_lib 文件夹，将尝试使用系统 pip 安装的版本")

load_local_akshare()

# 现在导入的 akshare 将优先使用本地源码
try:
    import akshare as ak
    print(f"[INFO] 当前使用的 AKShare 版本: {getattr(ak, '__version__', '未知')}")
except ImportError:
    print("[ERROR] 未找到 akshare，请先 pip install akshare 或克隆源码到 akshare_lib")
    sys.exit(1)

# ==============================================================================
# 2. 业务逻辑（保持与您的原始逻辑一致）
# ==============================================================================
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "day")

def download_stock(stock_info: dict, start_date: str, end_date: str):
    code = stock_info["代码"]
    try:
        # 增加随机延迟，降低被封风险
        time.sleep(random.uniform(0.3, 0.8))
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if df is not None and not df.empty:
            os.makedirs(DATA_DIR, exist_ok=True)
            df.to_csv(os.path.join(DATA_DIR, f"{code}.csv"), index=False, encoding="utf-8-sig")
            return True
    except Exception as e:
        print(f"下载 {code} 失败: {e}")
    return False

def main():
    print("正在获取股票列表...")
    stock_df = ak.stock_zh_a_spot_em()
    stocks = stock_df[["代码", "名称"]].head(5).to_dict("records") # 示例下载5只
    
    today = time.strftime("%Y%m%d")
    for s in stocks:
        success = download_stock(s, "20240101", today)
        status = "成功" if success else "失败"
        print(f"股票 {s['代码']} {s['名称']} 下载{status}")

if __name__ == "__main__":
    main()
