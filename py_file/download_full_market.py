import akshare as ak
import yfinance as yf
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time
import random
import requests
import warnings

# 忽略 SSL 警告
warnings.filterwarnings("ignore")

# 配置
DATA_DIR = "../data/day"
MAX_WORKERS = 4
MAX_RETRIES = 5

def get_all_stock_codes():
    """获取所有A股代码"""
    print("正在获取A股股票列表...")
    
    # 1. 尝试 AkShare
    for i in range(3):
        try:
            print(f"尝试使用 AkShare 获取股票列表 (第 {i+1} 次)...")
            df = ak.stock_zh_a_spot_em()
            codes = df[['代码', '名称']].values.tolist()
            print(f"AkShare: 成功获取 {len(codes)} 只股票")
            return codes
        except Exception as e:
            print(f"AkShare 失败: {e}")
            time.sleep(2)

    return []

def download_from_akshare(code, start_date, end_date):
    """从 AkShare 下载"""
    try:
        return ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    except Exception as e:
        # print(f"AkShare download error for {code}: {e}")
        return None

def download_from_yfinance(code, start_date, end_date):
    """从 yfinance 下载"""
    try:
        # 转换代码格式：600000 -> 600000.SS
        yf_code = f"{code}.SS" if code.startswith('6') else f"{code}.SZ"
        # 转换日期格式
        yf_start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        
        ticker = yf.Ticker(yf_code)
        df = ticker.history(start=yf_start, end=datetime.now().strftime("%Y-%m-%d"))
        
        if df.empty:
            return None
            
        # 重置索引并将 Date 列转为字符串
        df = df.reset_index()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # 统一列名
        df = df.rename(columns={
            'Date': '日期', 'Open': '开盘', 'High': '最高', 'Low': '最低', 
            'Close': '收盘', 'Volume': '成交量'
        })
        # yfinance 没有成交额和涨跌幅，需要自己计算或留空
        df['成交额'] = 0
        df['涨跌幅'] = df['收盘'].pct_change() * 100
        
        return df
    except Exception:
        return None

def get_market_dir(code):
    """根据股票代码判断市场并返回对应目录"""
    if code.startswith('6') or code.startswith('sh'):
        return os.path.join(DATA_DIR, 'sh')
    elif code.startswith('0') or code.startswith('3') or code.startswith('sz'):
        return os.path.join(DATA_DIR, 'sz')
    elif code.startswith('4') or code.startswith('8') or code.startswith('bj'):
        return os.path.join(DATA_DIR, 'bj')
    else:
        return os.path.join(DATA_DIR, 'other')

def download_stock_history(stock_info):
    """下载单只股票历史数据"""
    code, name = stock_info
    market_dir = get_market_dir(code)
    if not os.path.exists(market_dir):
        os.makedirs(market_dir, exist_ok=True)
        
    file_path = os.path.join(market_dir, f"{code}.csv")
    
    if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:
        return f"[{code}] {name}: Skipped"

    start_date = "19900101"
    end_date = datetime.now().strftime("%Y%m%d")
    
    # 尝试顺序：AkShare -> yfinance
    # AkShare 失败率高但数据全，yfinance 稳定但数据可能不全
    
    # 1. AkShare 重试 3 次
    for i in range(3):
        df = download_from_akshare(code, start_date, end_date)
        if df is not None and not df.empty:
            try:
                if '日期' in df.columns:
                    df.to_csv(file_path, index=False)
                    return f"[{code}] {name}: Downloaded via AkShare"
            except:
                pass
        time.sleep(random.uniform(0.5, 1.5))

    # 2. yfinance 重试 2 次
    for i in range(2):
        df = download_from_yfinance(code, start_date, end_date)
        if df is not None and not df.empty:
            try:
                df.to_csv(file_path, index=False)
                return f"[{code}] {name}: Downloaded via yfinance"
            except:
                pass
        time.sleep(1)
            
    return f"[{code}] {name}: Failed all sources"

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    all_stocks = get_all_stock_codes()
    
    if not all_stocks:
        print("无法获取股票列表。")
        return

    print(f"准备下载 {len(all_stocks)} 只股票的数据...")
    
    total = len(all_stocks)
    completed = 0
    success = 0
    skipped = 0
    failed = 0
    
    # 减少并发数以提高稳定性
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_stock_history, stock): stock for stock in all_stocks}
        
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            
            if "Downloaded" in result:
                success += 1
            elif "Skipped" in result:
                skipped += 1
            else:
                failed += 1
                
            if completed % 10 == 0 or completed == total:
                print(f"进度: {completed}/{total} | 成功: {success} | 跳过: {skipped} | 失败: {failed}")
                print(f"最新状态: {result}")

    print("\n所有下载任务完成")

if __name__ == "__main__":
    main()
