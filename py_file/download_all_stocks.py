#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
下载所有A股股票历史数据 (download_all_stocks.py)
================================================================================

功能说明:
    本脚本用于下载所有A股股票的完整历史日线数据，包括：
    1. 获取所有A股股票列表
    2. 下载每只股票从上市至今的全部日线数据
    3. 生成下载报告，包含数据时间范围统计

数据来源:
    - AKShare: 开源财经数据接口库 (https://akshare.xyz/)

存储路径:
    - 日线数据: /data/day/{stock_code}.csv

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
from typing import List, Dict, Tuple, Optional
import time
import json

# 尝试导入 akshare
try:
    import akshare as ak
except ImportError:
    print("错误: 请先安装 akshare 库")
    print("安装命令: pip install akshare")
    sys.exit(1)


# ==============================================================================
# 配置常量
# ==============================================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据存储目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')

# 报告目录
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')

# 下载延迟 (秒)，避免被限流
DOWNLOAD_DELAY = 0.3

# 最早开始日期 (尽可能早，让API返回所有可用数据)
EARLIEST_DATE = '19900101'


# ==============================================================================
# 核心函数
# ==============================================================================

def ensure_dirs():
    """确保必要的目录存在"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def get_all_stock_codes() -> List[str]:
    """
    获取所有A股股票代码列表
    
    返回:
        List[str]: 股票代码列表
    """
    print("正在获取所有A股股票列表...")
    
    try:
        # 尝试使用 stock_zh_a_spot_em 获取实时行情中的股票列表
        df = ak.stock_zh_a_spot_em()
        if not df.empty:
            codes = df['代码'].tolist()
            print(f"成功获取 {len(codes)} 只股票")
            return codes
    except Exception as e:
        print(f"方法1失败: {str(e)}")
    
    try:
        # 备选方法：使用 stock_info_a_code_name
        df = ak.stock_info_a_code_name()
        if not df.empty:
            codes = df['code'].tolist()
            print(f"成功获取 {len(codes)} 只股票")
            return codes
    except Exception as e:
        print(f"方法2失败: {str(e)}")
    
    print("警告: 无法获取完整股票列表，使用默认列表")
    return get_default_stock_list()


def get_default_stock_list() -> List[str]:
    """
    获取默认股票列表（沪深300成分股 + 主要指数）
    """
    # 沪深300部分成分股
    stocks = [
        '000001', '000002', '000063', '000066', '000100',
        '000157', '000333', '000338', '000425', '000538',
        '000568', '000596', '000625', '000651', '000661',
        '000703', '000725', '000768', '000776', '000783',
        '000858', '000876', '000895', '000938', '000963',
        '002001', '002007', '002008', '002024', '002027',
        '002032', '002049', '002050', '002120', '002129',
        '002142', '002179', '002202', '002230', '002236',
        '002241', '002252', '002271', '002304', '002311',
        '002352', '002371', '002415', '002460', '002466',
        '002475', '002493', '002555', '002594', '002601',
        '002602', '002607', '002624', '002714', '002736',
        '002841', '002916', '002938', '002939', '002945',
        '300003', '300014', '300015', '300033', '300059',
        '300122', '300124', '300142', '300144', '300347',
        '300408', '300413', '300433', '300498', '300601',
        '300628', '300750', '300760', '300782', '300896',
        '600000', '600009', '600010', '600011', '600015',
        '600016', '600018', '600019', '600025', '600028',
        '600029', '600030', '600031', '600036', '600048',
        '600050', '600061', '600085', '600089', '600104',
        '600109', '600111', '600115', '600118', '600132',
        '600143', '600150', '600161', '600176', '600183',
        '600196', '600276', '600309', '600332', '600346',
        '600352', '600362', '600383', '600406', '600436',
        '600438', '600460', '600489', '600519', '600547',
        '600570', '600585', '600588', '600600', '600606',
        '600655', '600660', '600690', '600703', '600741',
        '600745', '600760', '600795', '600809', '600837',
        '600845', '600848', '600867', '600886', '600887',
        '600893', '600900', '600905', '600919', '600926',
        '600958', '600989', '601006', '601009', '601012',
        '601021', '601066', '601088', '601100', '601108',
        '601111', '601117', '601138', '601155', '601166',
        '601169', '601186', '601211', '601225', '601229',
        '601231', '601236', '601238', '601288', '601318',
        '601319', '601328', '601336', '601360', '601377',
        '601390', '601398', '601600', '601601', '601607',
        '601618', '601628', '601633', '601668', '601669',
        '601688', '601698', '601766', '601788', '601800',
        '601808', '601816', '601818', '601838', '601857',
        '601858', '601865', '601877', '601878', '601881',
        '601888', '601898', '601899', '601901', '601916',
        '601919', '601933', '601939', '601985', '601988',
        '601989', '601995', '601998', '603019', '603156',
        '603160', '603259', '603260', '603288', '603290',
        '603369', '603501', '603517', '603799', '603833',
        '603899', '603986', '603993', '688005', '688008',
        '688009', '688012', '688036', '688111', '688126',
        '688169', '688180', '688187', '688188', '688223',
        '688256', '688303', '688363', '688396', '688536',
        '688561', '688599', '688690', '688728', '688981'
    ]
    return stocks


def download_stock_data(
    stock_code: str,
    start_date: str = EARLIEST_DATE,
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    下载单只股票的历史数据
    
    参数:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    返回:
        Tuple[pd.DataFrame, Dict]: (数据DataFrame, 统计信息)
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    stats = {
        'code': stock_code,
        'success': False,
        'rows': 0,
        'start_date': None,
        'end_date': None,
        'error': None
    }
    
    try:
        # 使用 akshare 获取日线数据
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # 前复权
        )
        
        if df.empty:
            stats['error'] = '无数据'
            return pd.DataFrame(), stats
        
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
        
        # 确保日期列是 datetime 类型
        df['date'] = pd.to_datetime(df['date'])
        
        # 添加股票代码列
        df['code'] = stock_code
        
        # 更新统计信息
        stats['success'] = True
        stats['rows'] = len(df)
        stats['start_date'] = df['date'].min().strftime('%Y-%m-%d')
        stats['end_date'] = df['date'].max().strftime('%Y-%m-%d')
        
        return df, stats
        
    except Exception as e:
        stats['error'] = str(e)
        return pd.DataFrame(), stats


def save_stock_data(df: pd.DataFrame, stock_code: str) -> bool:
    """
    保存股票数据到CSV文件
    """
    if df.empty:
        return False
    
    filepath = os.path.join(DATA_DIR, f"{stock_code}.csv")
    try:
        df.to_csv(filepath, index=False, encoding='utf-8')
        return True
    except Exception as e:
        print(f"保存失败 {stock_code}: {str(e)}")
        return False


def download_all_stocks(stock_list: Optional[List[str]] = None) -> Dict:
    """
    批量下载所有股票数据
    
    参数:
        stock_list: 股票代码列表，默认获取所有A股
        
    返回:
        Dict: 下载统计报告
    """
    ensure_dirs()
    
    if stock_list is None:
        stock_list = get_all_stock_codes()
    
    total = len(stock_list)
    
    report = {
        'download_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_stocks': total,
        'success_count': 0,
        'failed_count': 0,
        'total_rows': 0,
        'earliest_date': None,
        'latest_date': None,
        'stock_details': []
    }
    
    print("=" * 60)
    print(f"开始下载 {total} 只股票的历史数据")
    print("=" * 60)
    
    earliest = None
    latest = None
    
    for i, code in enumerate(stock_list, 1):
        print(f"\r[{i}/{total}] 下载 {code}...", end='', flush=True)
        
        df, stats = download_stock_data(code)
        
        if stats['success']:
            save_stock_data(df, code)
            report['success_count'] += 1
            report['total_rows'] += stats['rows']
            
            # 更新日期范围
            if stats['start_date']:
                if earliest is None or stats['start_date'] < earliest:
                    earliest = stats['start_date']
            if stats['end_date']:
                if latest is None or stats['end_date'] > latest:
                    latest = stats['end_date']
        else:
            report['failed_count'] += 1
            print(f" 失败: {stats['error']}")
        
        report['stock_details'].append(stats)
        
        # 延迟避免被限流
        time.sleep(DOWNLOAD_DELAY)
    
    report['earliest_date'] = earliest
    report['latest_date'] = latest
    
    print("\n" + "=" * 60)
    print("下载完成!")
    print(f"成功: {report['success_count']}, 失败: {report['failed_count']}")
    print(f"总数据行数: {report['total_rows']}")
    print(f"数据时间范围: {earliest} 至 {latest}")
    print("=" * 60)
    
    # 保存下载报告
    save_download_report(report)
    
    return report


def download_indices() -> Dict:
    """
    下载主要指数数据
    """
    indices = [
        ('sh000001', '上证指数'),
        ('sh000300', '沪深300'),
        ('sz399001', '深证成指'),
        ('sz399006', '创业板指'),
        ('sh000016', '上证50'),
        ('sh000905', '中证500'),
    ]
    
    print("\n" + "=" * 60)
    print("开始下载指数数据")
    print("=" * 60)
    
    results = []
    
    for code, name in indices:
        print(f"下载 {name} ({code})...")
        try:
            df = ak.stock_zh_index_daily(symbol=code)
            if not df.empty:
                # 标准化列名
                df['code'] = code
                df['date'] = pd.to_datetime(df['date'])
                
                filepath = os.path.join(DATA_DIR, f"{code}.csv")
                df.to_csv(filepath, index=False, encoding='utf-8')
                
                results.append({
                    'code': code,
                    'name': name,
                    'success': True,
                    'rows': len(df),
                    'start_date': df['date'].min().strftime('%Y-%m-%d'),
                    'end_date': df['date'].max().strftime('%Y-%m-%d')
                })
                print(f"  成功: {len(df)} 条数据")
            else:
                results.append({'code': code, 'name': name, 'success': False, 'error': '无数据'})
        except Exception as e:
            results.append({'code': code, 'name': name, 'success': False, 'error': str(e)})
            print(f"  失败: {str(e)}")
        
        time.sleep(DOWNLOAD_DELAY)
    
    return results


def save_download_report(report: Dict):
    """
    保存下载报告
    """
    # 保存JSON格式的完整报告
    json_path = os.path.join(REPORT_DIR, 'download_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 保存Markdown格式的摘要报告
    md_path = os.path.join(REPORT_DIR, 'download_summary.md')
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 股票数据下载报告\n\n")
        f.write(f"**下载时间**: {report['download_time']}\n\n")
        f.write("## 下载统计\n\n")
        f.write(f"| 项目 | 数值 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| 总股票数 | {report['total_stocks']} |\n")
        f.write(f"| 成功数量 | {report['success_count']} |\n")
        f.write(f"| 失败数量 | {report['failed_count']} |\n")
        f.write(f"| 总数据行数 | {report['total_rows']:,} |\n")
        f.write(f"| 最早日期 | {report['earliest_date']} |\n")
        f.write(f"| 最新日期 | {report['latest_date']} |\n")
        f.write("\n")
        
        # 计算数据时间跨度
        if report['earliest_date'] and report['latest_date']:
            earliest = datetime.strptime(report['earliest_date'], '%Y-%m-%d')
            latest = datetime.strptime(report['latest_date'], '%Y-%m-%d')
            days = (latest - earliest).days
            years = days / 365.25
            f.write(f"## 数据时间跨度\n\n")
            f.write(f"从 **{report['earliest_date']}** 到 **{report['latest_date']}**，")
            f.write(f"共计 **{days}** 天（约 **{years:.1f}** 年）。\n\n")
        
        # 失败列表
        failed = [s for s in report['stock_details'] if not s['success']]
        if failed:
            f.write("## 下载失败的股票\n\n")
            f.write("| 股票代码 | 错误信息 |\n")
            f.write("|---------|----------|\n")
            for s in failed[:50]:  # 只显示前50个
                f.write(f"| {s['code']} | {s.get('error', '未知错误')} |\n")
            if len(failed) > 50:
                f.write(f"\n... 还有 {len(failed) - 50} 只股票下载失败\n")
    
    print(f"\n报告已保存:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='下载所有A股股票历史数据')
    parser.add_argument('--default', action='store_true', help='只下载默认股票列表（沪深300成分股）')
    parser.add_argument('--indices', action='store_true', help='只下载指数数据')
    parser.add_argument('--all', action='store_true', help='下载所有A股股票数据')
    
    args = parser.parse_args()
    
    if args.indices:
        download_indices()
    elif args.default:
        stock_list = get_default_stock_list()
        download_all_stocks(stock_list)
        download_indices()
    else:
        # 默认下载所有股票
        download_all_stocks()
        download_indices()


if __name__ == "__main__":
    main()
