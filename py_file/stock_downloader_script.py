import efinance as ef
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
股票日线数据下载脚本
使用方法：
1. 确保已安装 efinance 和 pandas 库：pip install efinance pandas
2. 运行此脚本即可下载沪、深、北三大市场所有股票的日线数据
3. 数据将保存在当前目录下的 'stock_data' 文件夹中
"""

def download_single_stock(stock_code, stock_name, output_dir):
    file_path = os.path.join(output_dir, f"{stock_code}.csv")
    if os.path.exists(file_path):
        return True, stock_code

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 获取日线数据 (klt=101)
            df = ef.stock.get_quote_history(stock_code, klt=101)
            if df is None or df.empty:
                return False, stock_code
            
            # 格式化数据
            df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量']]
            
            # 保存到 CSV
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            return True, stock_code
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                return False, stock_code

def main():
    output_dir = 'stock_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("正在获取最新股票列表...")
    try:
        stocks_df = ef.stock.get_realtime_quotes()
        stocks_df = stocks_df[['股票代码', '股票名称']]
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return
        
    total_stocks = len(stocks_df)
    print(f"开始下载股票数据，总计 {total_stocks} 只股票...")
    
    success_count = 0
    fail_count = 0
    
    # 使用线程池加速下载 (建议根据网络情况调整 max_workers)
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_stock = {
            executor.submit(download_single_stock, str(row['股票代码']), row['股票名称'], output_dir): row['股票代码'] 
            for _, row in stocks_df.iterrows()
        }
        
        for future in as_completed(future_to_stock):
            success, stock_code = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
            
            if (success_count + fail_count) % 100 == 0:
                print(f"进度: {success_count + fail_count}/{total_stocks} (成功: {success_count}, 失败: {fail_count})")

    print(f"\n下载任务结束。")
    print(f"成功下载: {success_count} 只股票")
    print(f"下载失败: {fail_count} 只股票")
    print(f"数据已保存至: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
