#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强制更新股票数据脚本
直接从efinance获取最新数据并覆盖本地文件，确保补全1月8日和9日的数据
"""

import os
import efinance as ef
import pandas as pd
from pathlib import Path
import time

# 自动定位数据目录
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'day'

def get_market_from_code(code):
    """根据股票代码判断市场"""
    if code.startswith('6'):
        return 'sh'
    elif code.startswith('0') or code.startswith('3'):
        return 'sz'
    elif code.startswith('4') or code.startswith('8') or code.startswith('9'):
        return 'bj'
    return None

def download_stock(code, name=None):
    """下载单只股票数据并保存为指定格式"""
    try:
        # 获取历史行情数据
        df = ef.stock.get_quote_history(code)
        if df is None or len(df) == 0:
            return False, "没有获取到数据"
        
        # 提取股票名称
        if name is None:
            name = df['股票名称'].iloc[0] if '股票名称' in df.columns else code
        
        # 判断所属市场目录
        market = get_market_from_code(code)
        if market is None:
            return False, f"未知市场代码: {code}"
        
        market_dir = DATA_DIR / market
        market_dir.mkdir(parents=True, exist_ok=True)
        
        # 按照项目要求的CSV格式重新组织数据
        # 格式：名称,日期,开盘,收盘,最高,最低,成交量
        result_df = pd.DataFrame({
            '名称': name,
            '日期': df['日期'],
            '开盘': df['开盘'],
            '收盘': df['收盘'],
            '最高': df['最高'],
            '最低': df['最低'],
            '成交量': df['成交量']
        })
        
        # 保存为带BOM的UTF-8格式，确保Excel和中文显示正常
        file_path = market_dir / f"{code}.csv"
        result_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        return True, f"成功保存 {len(result_df)} 条记录"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("🚀 正在启动强制数据更新程序...")
    print(f"📂 数据存储路径: {DATA_DIR}")
    print("=" * 60)
    
    # 1. 获取最新股票列表
    print("🔍 正在获取全市场股票列表...")
    try:
        stock_list = ef.stock.get_realtime_quotes()
        if stock_list is None or len(stock_list) == 0:
            print("❌ 错误: 无法获取股票列表，请检查网络连接")
            return
        total = len(stock_list)
        print(f"✅ 成功获取 {total} 只股票信息")
    except Exception as e:
        print(f"❌ 获取股票列表失败: {e}")
        return
    
    # 2. 循环下载
    success_count = 0
    fail_count = 0
    
    print(f"\n📥 开始下载数据 (预计耗时较长，请耐心等待)...")
    print("-" * 60)
    
    for i, row in stock_list.iterrows():
        code = row['股票代码']
        name = row['股票名称']
        
        success, msg = download_stock(code, name)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            print(f"⚠️  [{code}] {name} 下载失败: {msg}")
        
        # 每100只打印一次进度
        if (i + 1) % 100 == 0:
            print(f"📊 进度: {i+1}/{total} | 成功: {success_count} | 失败: {fail_count}")
        
        # 控制频率，避免被封IP
        time.sleep(0.05)
    
    print("-" * 60)
    print(f"✨ 下载任务完成！")
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {fail_count}")
    print("=" * 60)

if __name__ == '__main__':
    # 确保安装了依赖库: pip install efinance pandas
    main()
