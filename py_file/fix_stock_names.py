#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 stock_reports.json 中的股票名称

功能：
从 CSV 数据文件中读取真实的股票名称，更新到 stock_reports.json 中
"""

import json
import pandas as pd
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "day"
WEB_DATA_DIR = PROJECT_ROOT / "web" / "client" / "src" / "data"
STOCK_REPORTS_FILE = WEB_DATA_DIR / "stock_reports.json"


def get_stock_name_from_csv(code: str, market: str) -> str:
    """从CSV文件中读取股票名称"""
    csv_path = DATA_DIR / market / f"{code}.csv"
    
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig', nrows=1)
        if '名称' in df.columns:
            name = df['名称'].iloc[0]
            if pd.notna(name) and str(name).strip():
                return str(name).strip()
    except Exception as e:
        print(f"读取 {csv_path} 失败: {e}")
    
    return None


def fix_stock_names():
    """修复股票名称"""
    print("=" * 60)
    print("开始修复股票名称")
    print("=" * 60)
    
    # 读取现有的 stock_reports.json
    if not STOCK_REPORTS_FILE.exists():
        print(f"错误: 文件不存在 {STOCK_REPORTS_FILE}")
        return
    
    with open(STOCK_REPORTS_FILE, 'r', encoding='utf-8') as f:
        reports = json.load(f)
    
    print(f"共有 {len(reports)} 条记录")
    
    # 统计
    updated = 0
    failed = 0
    
    # 遍历并修复
    for report in reports:
        code = report.get('code', '')
        market = report.get('market', '')
        old_name = report.get('name', '')
        
        # 获取真实名称
        real_name = get_stock_name_from_csv(code, market)
        
        if real_name:
            if old_name != real_name:
                report['name'] = real_name
                updated += 1
        else:
            failed += 1
    
    # 保存更新后的文件
    with open(STOCK_REPORTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    
    print(f"\n修复完成:")
    print(f"  更新: {updated} 条")
    print(f"  失败: {failed} 条")
    print(f"  总计: {len(reports)} 条")
    print(f"\n已保存到: {STOCK_REPORTS_FILE}")


if __name__ == "__main__":
    fix_stock_names()
