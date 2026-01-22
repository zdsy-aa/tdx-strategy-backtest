#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成版数据获取脚本 (a1_data_fetcher_v2_github.py)
1. 自动加载与项目文件夹并列的 akshare 源码。
2. 下载数据并保存为指定格式：名称 日期 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
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
    自动检测并加载与项目文件夹并列的 akshare 源码。
    结构：
    /parent_dir/
       ├── akshare/
       └── tdx-strategy-backtest/
           └── a1_data_fetcher_v2_github.py
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 父目录
    parent_dir = os.path.dirname(current_dir)
    # akshare 源码路径
    local_akshare_path = os.path.join(parent_dir, "akshare")
    
    if os.path.exists(local_akshare_path):
        # 将父目录加入 sys.path，这样 import akshare 就能找到并列的 akshare 文件夹
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        print(f"[INFO] 检测到并列目录下的本地源码，已优先加载：{local_akshare_path}")
    else:
        print(f"[WARN] 未检测到并列的 akshare 文件夹 ({local_akshare_path})，将尝试使用系统 pip 安装的版本")

load_local_akshare()

# 现在导入的 akshare 将优先使用本地源码
try:
    import akshare as ak
    print(f"[INFO] 当前使用的 AKShare 版本: {getattr(ak, '__version__', '未知')}")
except ImportError:
    print("[ERROR] 未找到 akshare，请确保项目文件夹并列位置存在 akshare 文件夹")
    sys.exit(1)

# ==============================================================================
# 2. 业务逻辑
# ==============================================================================
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "day")

def download_stock(stock_info: dict, start_date: str, end_date: str):
    code = stock_info["代码"]
    name = stock_info["名称"]
    try:
        # 增加随机延迟，降低被封风险
        time.sleep(random.uniform(0.1, 0.5))
        
        # 获取历史行情数据
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        
        if df is not None and not df.empty:
            # 插入“名称”列到第一列
            df.insert(0, "名称", name)
            
            # 用户要求的格式：名称 日期 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
            # ak.stock_zh_a_hist 返回的默认列名通常是：日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
            
            expected_columns = ["名称", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
            
            # 确保列顺序正确
            df = df[expected_columns]
            
            os.makedirs(DATA_DIR, exist_ok=True)
            # 保存为 CSV，使用 utf-8-sig 以便 Excel 直接打开
            df.to_csv(os.path.join(DATA_DIR, f"{code}.csv"), index=False, encoding="utf-8-sig")
            return True
    except Exception as e:
        print(f"下载 {code} ({name}) 失败: {e}")
    return False

def main():
    print("正在获取股票列表...")
    try:
        stock_df = ak.stock_zh_a_spot_em()
        # 选取前 5 只作为示例
        stocks = stock_df[["代码", "名称"]].head(5).to_dict("records") 
        
        today = time.strftime("%Y%m%d")
        print(f"开始下载数据到 {DATA_DIR}...")
        
        for s in stocks:
            success = download_stock(s, "20240101", today)
            status = "成功" if success else "失败"
            print(f"股票 {s['代码']} {s['名称']} 下载{status}")
            
    except Exception as e:
        print(f"获取股票列表失败: {e}")

if __name__ == "__main__":
    main()