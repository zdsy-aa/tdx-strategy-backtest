#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取股票名称并更新到数据文件
"""

import json
import os
import sys

try:
    import akshare as ak
    print("✓ akshare 已安装")
except ImportError:
    print("✗ akshare 未安装，正在安装...")
    os.system("pip3 install akshare -q")
    import akshare as ak

def fetch_stock_names():
    """从 akshare 获取所有股票的名称"""
    stock_name_map = {}
    
    try:
        print("\n正在获取股票列表...")
        
        # 方法1: 使用 stock_zh_a_spot_em 获取实时数据
        print("- 获取沪深A股实时数据...")
        stock_info = ak.stock_zh_a_spot_em()
        for _, row in stock_info.iterrows():
            code = row['代码']
            name = row['名称']
            stock_name_map[code] = name
        
        print(f"✓ 获取到 {len(stock_name_map)} 只股票名称")
        
    except Exception as e:
        print(f"✗ 获取股票名称失败: {e}")
        print("尝试使用备用方法...")
        
        try:
            # 方法2: 使用 stock_info_a_code_name
            stock_info = ak.stock_info_a_code_name()
            for _, row in stock_info.iterrows():
                code = row['code']
                name = row['name']
                stock_name_map[code] = name
            print(f"✓ 备用方法成功，获取到 {len(stock_name_map)} 只股票名称")
        except Exception as e2:
            print(f"✗ 备用方法也失败: {e2}")
            return None
    
    return stock_name_map

def update_stock_reports(stock_name_map):
    """更新 stock_reports.json 中的股票名称"""
    reports_file = "../web/client/src/data/stock_reports.json"
    
    if not os.path.exists(reports_file):
        print(f"✗ 文件不存在: {reports_file}")
        return False
    
    try:
        # 读取现有数据
        with open(reports_file, 'r', encoding='utf-8') as f:
            reports = json.load(f)
        
        print(f"\n正在更新 {len(reports)} 条股票记录...")
        
        updated_count = 0
        for report in reports:
            code = report['code']
            
            # 尝试从映射中获取名称
            if code in stock_name_map:
                old_name = report['name']
                new_name = stock_name_map[code]
                if old_name != new_name:
                    report['name'] = new_name
                    updated_count += 1
        
        # 保存更新后的数据
        with open(reports_file, 'w', encoding='utf-8') as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 成功更新 {updated_count} 条股票名称")
        print(f"✓ 文件已保存: {reports_file}")
        return True
        
    except Exception as e:
        print(f"✗ 更新失败: {e}")
        return False

def create_stock_name_mapping():
    """创建股票代码-名称映射文件"""
    mapping_file = "../web/client/src/data/stock_names.json"
    
    stock_name_map = fetch_stock_names()
    if not stock_name_map:
        return False
    
    try:
        os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
        
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(stock_name_map, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 股票名称映射已保存: {mapping_file}")
        return True
        
    except Exception as e:
        print(f"✗ 保存映射文件失败: {e}")
        return False

def main():
    print("=" * 60)
    print("股票名称获取工具")
    print("=" * 60)
    
    # 获取股票名称
    stock_name_map = fetch_stock_names()
    if not stock_name_map:
        print("\n✗ 获取股票名称失败，程序退出")
        sys.exit(1)
    
    # 创建映射文件
    if not create_stock_name_mapping():
        print("\n⚠ 映射文件创建失败")
    
    # 更新 stock_reports.json
    if update_stock_reports(stock_name_map):
        print("\n✓ 所有操作完成")
    else:
        print("\n✗ 更新 stock_reports.json 失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
