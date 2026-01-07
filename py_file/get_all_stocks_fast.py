#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取所有A股股票列表
"""

import akshare as ak
import pandas as pd
import os

def get_all_stock_codes_fast():
    """
    快速获取所有A股股票代码列表

    返回:
        list: 股票代码列表
    """
    try:
        print("正在快速获取所有A股股票列表...")
        # 使用 stock_info_a_code_name 函数，这通常更快
        stock_info = ak.stock_info_a_code_name()

        if stock_info.empty:
            print("警告: 未获取到股票数据")
            return []

        # 提取股票代码
        stock_codes = stock_info['code'].tolist()

        print(f"成功获取 {len(stock_codes)} 只股票")

        # 保存股票列表到文件
        output_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_list.txt')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for code in stock_codes:
                f.write(f"{code}\n")

        print(f"股票列表已保存到: {output_file}")

        return stock_codes

    except Exception as e:
        print(f"错误: 获取股票列表失败 - {str(e)}")
        return []

if __name__ == "__main__":
    codes = get_all_stock_codes_fast()
    if codes:
        print(f"\n总计: {len(codes)} 只股票")
