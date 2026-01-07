#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
全A股日线数据下载脚本 (download_all_a_stocks.py)
================================================================================

功能说明:
    下载全部A股股票的历史日线数据，从上市日期到指定截止日期（默认2026年1月6日）
    
使用方法:
    python download_all_a_stocks.py                    # 下载全部A股
    python download_all_a_stocks.py --end 20260106    # 指定截止日期
    python download_all_a_stocks.py --limit 100       # 只下载前100只
    python download_all_a_stocks.py --resume          # 断点续传模式
    
数据来源:
    - 东方财富 API (通过 akshare)
    
存储路径:
    - 日线数据: data/day/{stock_code}.csv

作者: TradeGuide System
版本: 2.0.0
更新日期: 2026-01-07
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import requests
from pathlib import Path
import argparse

# 数据目录
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "day"
LOG_FILE = PROJECT_DIR / "download_progress.json"

# 默认配置
DEFAULT_END_DATE = "20260106"
DEFAULT_START_DATE = "19900101"  # 从1990年开始，覆盖所有历史数据
REQUEST_DELAY = 0.3  # 请求间隔（秒）
MAX_RETRIES = 3  # 最大重试次数


def ensure_data_dir():
    """确保数据目录存在"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def get_all_a_stock_list():
    """
    获取全部A股股票列表
    
    返回:
        list: 股票代码列表，格式如 ['000001', '000002', ...]
    """
    print("正在获取全部A股股票列表...")
    
    try:
        import akshare as ak
        # 获取A股实时行情，从中提取股票代码
        df = ak.stock_zh_a_spot_em()
        
        if df.empty:
            print("警告: 无法获取股票列表")
            return []
        
        # 提取股票代码
        stock_codes = df['代码'].tolist()
        
        # 过滤掉非股票代码（如指数、基金等）
        valid_codes = []
        for code in stock_codes:
            # A股股票代码规则:
            # 上海: 600xxx, 601xxx, 603xxx, 605xxx (主板), 688xxx (科创板)
            # 深圳: 000xxx, 001xxx (主板), 002xxx (中小板), 300xxx, 301xxx (创业板)
            if code.startswith(('000', '001', '002', '003', '300', '301', 
                               '600', '601', '603', '605', '688', '689')):
                valid_codes.append(code)
        
        print(f"共获取到 {len(valid_codes)} 只A股股票")
        return sorted(valid_codes)
        
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return []


def download_stock_data_eastmoney(stock_code, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """
    从东方财富下载股票数据
    
    参数:
        stock_code: 股票代码
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        
    返回:
        pd.DataFrame: 日线数据
    """
    # 确定市场代码
    if stock_code.startswith(('6', '9')):
        secid = f"1.{stock_code}"  # 上海
    else:
        secid = f"0.{stock_code}"  # 深圳
    
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",  # 日线
        "fqt": "1",    # 前复权
        "secid": secid,
        "beg": start_date,
        "end": end_date
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://quote.eastmoney.com/"
    }
    
    for retry in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            data = response.json()
            
            if data.get("data") is None or data["data"].get("klines") is None:
                return pd.DataFrame()
            
            klines = data["data"]["klines"]
            if not klines:
                return pd.DataFrame()
            
            # 解析数据
            records = []
            for line in klines:
                parts = line.split(",")
                if len(parts) >= 11:
                    records.append({
                        "date": parts[0],
                        "股票代码": stock_code,
                        "open": float(parts[1]),
                        "close": float(parts[2]),
                        "high": float(parts[3]),
                        "low": float(parts[4]),
                        "volume": int(float(parts[5])),
                        "amount": float(parts[6]),
                        "amplitude": float(parts[7]) if parts[7] != '-' else 0,
                        "pct_change": float(parts[8]) if parts[8] != '-' else 0,
                        "change": float(parts[9]) if parts[9] != '-' else 0,
                        "turnover": float(parts[10]) if parts[10] != '-' else 0,
                        "code": stock_code
                    })
            
            df = pd.DataFrame(records)
            return df
            
        except requests.exceptions.RequestException as e:
            if retry < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            else:
                raise e
        except Exception as e:
            raise e
    
    return pd.DataFrame()


def download_stock_data_akshare(stock_code, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """
    使用akshare下载股票数据（备用方法）
    
    参数:
        stock_code: 股票代码
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        
    返回:
        pd.DataFrame: 日线数据
    """
    try:
        import akshare as ak
        
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        
        if df.empty:
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
        
        df['code'] = stock_code
        df['股票代码'] = stock_code
        
        return df
        
    except Exception as e:
        raise e


def save_progress(downloaded_codes, failed_codes):
    """保存下载进度"""
    progress = {
        "downloaded": downloaded_codes,
        "failed": failed_codes,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def load_progress():
    """加载下载进度"""
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"downloaded": [], "failed": []}


def download_all_stocks(
    end_date=DEFAULT_END_DATE,
    start_date=DEFAULT_START_DATE,
    limit=None,
    resume=False
):
    """
    下载全部A股股票数据
    
    参数:
        end_date: 截止日期
        start_date: 开始日期
        limit: 限制下载数量（用于测试）
        resume: 是否断点续传
        
    返回:
        dict: 下载统计
    """
    ensure_data_dir()
    
    # 获取股票列表
    all_stocks = get_all_a_stock_list()
    
    if not all_stocks:
        print("错误: 无法获取股票列表")
        return {"success": 0, "failed": 0, "skipped": 0}
    
    # 断点续传
    progress = {"downloaded": [], "failed": []}
    if resume:
        progress = load_progress()
        print(f"断点续传模式: 已下载 {len(progress['downloaded'])} 只，失败 {len(progress['failed'])} 只")
    
    # 过滤已下载的股票
    if resume:
        all_stocks = [s for s in all_stocks if s not in progress['downloaded']]
    
    # 限制数量
    if limit:
        all_stocks = all_stocks[:limit]
    
    print(f"\n{'='*60}")
    print(f"开始下载 {len(all_stocks)} 只股票的历史数据")
    print(f"数据范围: {start_date} 至 {end_date}")
    print(f"{'='*60}\n")
    
    stats = {
        "success": len(progress['downloaded']) if resume else 0,
        "failed": 0,
        "skipped": 0,
        "total_rows": 0
    }
    
    downloaded_codes = progress['downloaded'].copy() if resume else []
    failed_codes = []
    
    start_time = datetime.now()
    
    for i, stock_code in enumerate(all_stocks, 1):
        try:
            # 检查是否已存在
            file_path = DATA_DIR / f"{stock_code}.csv"
            
            # 尝试下载
            print(f"[{i}/{len(all_stocks)}] 下载 {stock_code}...", end=" ")
            
            # 优先使用东方财富API
            try:
                df = download_stock_data_eastmoney(stock_code, start_date, end_date)
            except:
                # 备用方案
                df = download_stock_data_akshare(stock_code, start_date, end_date)
            
            if df.empty:
                print("无数据")
                stats['skipped'] += 1
                continue
            
            # 保存数据
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            stats['success'] += 1
            stats['total_rows'] += len(df)
            downloaded_codes.append(stock_code)
            
            print(f"成功 ({len(df)} 条记录)")
            
            # 定期保存进度
            if i % 50 == 0:
                save_progress(downloaded_codes, failed_codes)
                elapsed = (datetime.now() - start_time).total_seconds()
                speed = i / elapsed if elapsed > 0 else 0
                remaining = (len(all_stocks) - i) / speed if speed > 0 else 0
                print(f"\n--- 进度: {i}/{len(all_stocks)} ({i/len(all_stocks)*100:.1f}%), "
                      f"预计剩余: {remaining/60:.1f} 分钟 ---\n")
            
            # 请求间隔
            time.sleep(REQUEST_DELAY)
            
        except Exception as e:
            print(f"失败: {str(e)[:50]}")
            stats['failed'] += 1
            failed_codes.append(stock_code)
            time.sleep(1)  # 失败后多等一会
    
    # 保存最终进度
    save_progress(downloaded_codes, failed_codes)
    
    # 打印统计
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*60}")
    print("下载完成!")
    print(f"{'='*60}")
    print(f"成功: {stats['success']}")
    print(f"失败: {stats['failed']}")
    print(f"跳过: {stats['skipped']}")
    print(f"总数据行数: {stats['total_rows']}")
    print(f"耗时: {duration/60:.1f} 分钟")
    print(f"{'='*60}")
    
    # 生成数据报告
    generate_data_report()
    
    return stats


def generate_data_report():
    """生成数据下载报告"""
    print("\n生成数据下载报告...")
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        print("没有找到数据文件")
        return
    
    report = {
        "total_stocks": len(csv_files),
        "data_range": {},
        "stocks": []
    }
    
    min_date = None
    max_date = None
    
    for csv_file in csv_files[:10]:  # 只检查前10个文件
        try:
            df = pd.read_csv(csv_file)
            if 'date' in df.columns:
                dates = pd.to_datetime(df['date'])
                file_min = dates.min()
                file_max = dates.max()
                
                if min_date is None or file_min < min_date:
                    min_date = file_min
                if max_date is None or file_max > max_date:
                    max_date = file_max
        except:
            pass
    
    if min_date and max_date:
        report["data_range"] = {
            "start": min_date.strftime("%Y-%m-%d"),
            "end": max_date.strftime("%Y-%m-%d")
        }
    
    # 保存报告
    report_file = PROJECT_DIR / "data" / "download_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据下载报告:")
    print(f"  - 股票数量: {report['total_stocks']}")
    if report["data_range"]:
        print(f"  - 数据范围: {report['data_range']['start']} 至 {report['data_range']['end']}")
    print(f"  - 报告已保存: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='全A股日线数据下载工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python download_all_a_stocks.py                    # 下载全部A股
    python download_all_a_stocks.py --end 20260106    # 指定截止日期
    python download_all_a_stocks.py --limit 100       # 只下载前100只（测试用）
    python download_all_a_stocks.py --resume          # 断点续传模式
        """
    )
    
    parser.add_argument('--end', type=str, default=DEFAULT_END_DATE,
                        help=f'截止日期 (YYYYMMDD)，默认 {DEFAULT_END_DATE}')
    parser.add_argument('--start', type=str, default=DEFAULT_START_DATE,
                        help=f'开始日期 (YYYYMMDD)，默认 {DEFAULT_START_DATE}')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制下载数量（用于测试）')
    parser.add_argument('--resume', action='store_true',
                        help='断点续传模式')
    
    args = parser.parse_args()
    
    download_all_stocks(
        end_date=args.end,
        start_date=args.start,
        limit=args.limit,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
