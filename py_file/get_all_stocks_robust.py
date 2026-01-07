#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取所有A股股票列表 (健壮版)
"""

import akshare as ak
import pandas as pd
import os
import time

def get_all_stock_codes():
    """
    尝试多种方法获取所有A股股票代码列表
    """
    methods = [
        ("ak.stock_zh_a_spot_em", lambda: ak.stock_zh_a_spot_em()['代码'].tolist()),
        ("ak.stock_info_a_code_name", lambda: ak.stock_info_a_code_name()['code'].tolist()),
        ("ak.stock_sh_a_spot_em", lambda: ak.stock_sh_a_spot_em()['代码'].tolist() + ak.stock_sz_a_spot_em()['代码'].tolist())
    ]
    
    for name, method in methods:
        try:
            print(f"尝试方法: {name}...")
            codes = method()
            if codes:
                # 去重并排序
                codes = sorted(list(set(codes)))
                print(f"成功获取 {len(codes)} 只股票")
                return codes
        except Exception as e:
            print(f"方法 {name} 失败: {str(e)}")
            time.sleep(1)
            
    return []

if __name__ == "__main__":
    codes = get_all_stock_codes()
    if codes:
        output_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_list.txt')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for code in codes:
                f.write(f"{code}\n")
        print(f"股票列表已保存到: {output_file}")
        print(f"\n总计: {len(codes)} 只股票")
    else:
        print("所有方法均失败，无法获取股票列表")
