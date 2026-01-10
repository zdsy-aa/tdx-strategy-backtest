#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
股票数据下载脚本 (stock_downloader_script.py)
================================================================================
功能说明:
    统一的股票数据下载脚本，支持下载全部A股（沪市、深市、北交所）的历史日线数据。
    整合了多种数据源，自动切换以确保数据获取成功。

使用方法:
    python stock_downloader_script.py                    # 下载全部A股
    python stock_downloader_script.py --update           # 只更新已有数据（增量更新）
    python stock_downloader_script.py --market sh        # 只下载沪市
    python stock_downloader_script.py --market sz        # 只下载深市
    python stock_downloader_script.py --market bj        # 只下载北交所
    python stock_downloader_script.py --limit 100        # 只下载前100只
    python stock_downloader_script.py --resume           # 断点续传模式

数据来源（按优先级）:
    1. efinance - 东方财富数据接口
    2. akshare - 开源财经数据接口库

存储路径:
    - 沪市: data/day/sh/{stock_code}.csv
    - 深市: data/day/sz/{stock_code}.csv
    - 北交所: data/day/bj/{stock_code}.csv

作者: TradeGuide System
版本: 3.0.0
更新日期: 2026-01-08
================================================================================
"""

import os
import sys
import pandas as pd
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 配置常量
# ==============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "day"
PROGRESS_FILE = PROJECT_DIR / "download_progress.json"

# 下载配置
MAX_WORKERS = 10          # 并发线程数
REQUEST_DELAY = 0.3       # 请求间隔（秒）
MAX_RETRIES = 3           # 最大重试次数

# ==============================================================================
# 数据源模块
# ==============================================================================

# 尝试导入数据源库
EFINANCE_AVAILABLE = False
AKSHARE_AVAILABLE = False

try:
    import efinance as ef
    EFINANCE_AVAILABLE = True
except ImportError:
    pass

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    pass

def check_dependencies():
    """检查依赖库"""
    if not EFINANCE_AVAILABLE and not AKSHARE_AVAILABLE:
        print("错误: 请至少安装 efinance 或 akshare 库")
        print("安装命令:")
        print("  pip install efinance")
        print("  pip install akshare")
        sys.exit(1)
    
    print("数据源可用性:")
    print(f"  - efinance: {'✓' if EFINANCE_AVAILABLE else '✗'}")
    print(f"  - akshare:  {'✓' if AKSHARE_AVAILABLE else '✗'}")

# ==============================================================================
# 获取股票列表
# ==============================================================================

def get_all_stock_codes():
    """获取所有A股股票代码和名称"""
    stocks = []
    
    # 方法1: 使用 efinance
    if EFINANCE_AVAILABLE:
        try:
            print("正在使用 efinance 获取股票列表...")
            df = ef.stock.get_realtime_quotes()
            for _, row in df.iterrows():
                code = str(row['股票代码'])
                name = row['股票名称']
                # 判断市场
                if code.startswith('6'):
                    market = 'sh'
                elif code.startswith(('0', '3')):
                    market = 'sz'
                elif code.startswith(('8', '4')):
                    market = 'bj'
                else:
                    continue
                stocks.append({'code': code, 'name': name, 'market': market})
            print(f"efinance: 成功获取 {len(stocks)} 只股票")
            return stocks
        except Exception as e:
            print(f"efinance 获取失败: {e}")
    
    # 方法2: 使用 akshare
    if AKSHARE_AVAILABLE:
        try:
            print("正在使用 akshare 获取股票列表...")
            df = ak.stock_zh_a_spot_em()
            for _, row in df.iterrows():
                code = str(row['代码'])
                name = row['名称']
                if code.startswith('6'):
                    market = 'sh'
                elif code.startswith(('0', '3')):
                    market = 'sz'
                elif code.startswith(('8', '4')):
                    market = 'bj'
                else:
                    continue
                stocks.append({'code': code, 'name': name, 'market': market})
            print(f"akshare: 成功获取 {len(stocks)} 只股票")
            return stocks
        except Exception as e:
            print(f"akshare 获取失败: {e}")
    
    print("警告: 无法获取股票列表")
    return stocks

# ==============================================================================
# 下载单只股票数据
# ==============================================================================

def download_from_efinance(stock_code):
    """从 efinance 下载数据"""
    if not EFINANCE_AVAILABLE:
        return None
    try:
        df = ef.stock.get_quote_history(stock_code, klt=101)
        if df is None or df.empty:
            return None
        # 标准化列名
        df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量']]
        return df
    except Exception:
        return None

def download_from_akshare(stock_code):
    """从 akshare 下载数据"""
    if not AKSHARE_AVAILABLE:
        return None
    try:
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                                start_date="19900101", 
                                end_date=datetime.now().strftime("%Y%m%d"),
                                adjust="qfq")
        if df is None or df.empty:
            return None
        # 标准化列名
        df = df.rename(columns={
            '日期': '日期',
            '开盘': '开盘',
            '收盘': '收盘',
            '最高': '最高',
            '最低': '最低',
            '成交量': '成交量'
        })
        df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量']]
        return df
    except Exception:
        return None

def download_single_stock(stock_info, update_mode=False):
    """下载单只股票数据"""
    code = stock_info['code']
    market = stock_info['market']
    
    # 确定保存路径
    market_dir = DATA_DIR / market
    market_dir.mkdir(parents=True, exist_ok=True)
    file_path = market_dir / f"{code}.csv"
    
    # 更新模式：如果文件已存在且较新，跳过
    if update_mode and file_path.exists():
        # 检查文件修改时间
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        if (datetime.now() - mtime).days < 1:
            return True, code, "跳过（已是最新）"
    
    # 尝试多个数据源
    df = None
    source = None
    name = stock_info.get('name', '')
    
    for attempt in range(MAX_RETRIES):
        # 优先使用 efinance
        df = download_from_efinance(code)
        if df is not None and not df.empty:
            source = "efinance"
            break
        
        # 备用 akshare
        df = download_from_akshare(code)
        if df is not None and not df.empty:
            source = "akshare"
            break
        
        time.sleep(REQUEST_DELAY)
    
    if df is None or df.empty:
        return False, code, "下载失败"
    
    # 插入名称列
    if '名称' not in df.columns:
        df.insert(0, '名称', name)
    else:
        df['名称'] = name
        
    # 保存数据
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    return True, code, f"成功（{source}）"

# ==============================================================================
# 批量下载
# ==============================================================================

def download_all_stocks(stocks, update_mode=False):
    """批量下载所有股票数据"""
    total = len(stocks)
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    print(f"\n开始下载，共 {total} 只股票...")
    print(f"数据保存目录: {DATA_DIR.resolve()}")
    print("-" * 60)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_stock = {
            executor.submit(download_single_stock, stock, update_mode): stock
            for stock in stocks
        }
        
        for future in as_completed(future_to_stock):
            stock = future_to_stock[future]
            try:
                success, code, msg = future.result()
                if success:
                    if "跳过" in msg:
                        skip_count += 1
                    else:
                        success_count += 1
                else:
                    fail_count += 1
                
                # 进度显示
                done = success_count + fail_count + skip_count
                if done % 100 == 0 or done == total:
                    print(f"进度: {done}/{total} | 成功: {success_count} | 跳过: {skip_count} | 失败: {fail_count}")
                    
            except Exception as e:
                fail_count += 1
    
    print("-" * 60)
    print(f"下载完成！")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  失败: {fail_count}")
    
    return success_count, skip_count, fail_count

# ==============================================================================
# 保存/加载进度
# ==============================================================================

def save_progress(downloaded_codes):
    """保存下载进度"""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'downloaded': list(downloaded_codes),
            'timestamp': datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)

def load_progress():
    """加载下载进度"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get('downloaded', []))
    return set()

# ==============================================================================
# 主函数
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='股票数据下载脚本')
    parser.add_argument('--update', action='store_true', help='增量更新模式')
    parser.add_argument('--market', choices=['sh', 'sz', 'bj', 'all'], default='all', help='指定市场')
    parser.add_argument('--limit', type=int, default=0, help='限制下载数量')
    parser.add_argument('--resume', action='store_true', help='断点续传模式')
    args = parser.parse_args()
    
    print("=" * 60)
    print("股票数据下载脚本 v3.0")
    print("=" * 60)
    
    # 检查依赖
    check_dependencies()
    
    # 创建数据目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for market in ['sh', 'sz', 'bj']:
        (DATA_DIR / market).mkdir(exist_ok=True)
    
    # 获取股票列表
    stocks = get_all_stock_codes()
    if not stocks:
        print("错误: 无法获取股票列表")
        return
    
    # 过滤市场
    if args.market != 'all':
        stocks = [s for s in stocks if s['market'] == args.market]
        print(f"筛选 {args.market} 市场: {len(stocks)} 只股票")
    
    # 断点续传
    if args.resume:
        downloaded = load_progress()
        stocks = [s for s in stocks if s['code'] not in downloaded]
        print(f"断点续传: 剩余 {len(stocks)} 只股票待下载")
    
    # 限制数量
    if args.limit > 0:
        stocks = stocks[:args.limit]
        print(f"限制下载: {len(stocks)} 只股票")
    
    # 开始下载
    success, skip, fail = download_all_stocks(stocks, update_mode=args.update)
    
    # 统计信息
    print("\n" + "=" * 60)
    print("数据统计:")
    for market in ['sh', 'sz', 'bj']:
        market_dir = DATA_DIR / market
        if market_dir.exists():
            count = len(list(market_dir.glob('*.csv')))
            print(f"  {market}: {count} 只股票")
    print("=" * 60)

if __name__ == "__main__":
    main()
