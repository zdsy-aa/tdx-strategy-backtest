#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
定时数据更新脚本 (scheduled_data_update.py)
================================================================================

功能说明:
    本脚本用于每日定时更新股票日线数据，设计为在每个交易日
    下午3:30收盘后自动运行。

执行时间:
    - 每个工作日 15:30 (北京时间)
    - 通过系统 cron 或 Windows 任务计划程序调度

主要功能:
    1. 检查是否为交易日
    2. 下载当天的日线数据
    3. 更新本地数据文件
    4. 生成更新日志

使用方法:
    直接运行: python scheduled_data_update.py
    
    Linux cron 设置 (每天15:30执行):
    30 15 * * 1-5 cd /path/to/project && python py_file/scheduled_data_update.py
    
    Windows 任务计划程序:
    创建基本任务，设置触发器为每天15:30，操作为运行此脚本

作者: TradeGuide System
版本: 2.0.0
更新日期: 2026-01-07
================================================================================
"""

import os
import sys
from datetime import datetime, timedelta
import logging

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_fetcher import download_today_data, DEFAULT_STOCKS, DEFAULT_INDICES


# ==============================================================================
# 配置
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, f'data_update_{datetime.now().strftime("%Y%m%d")}.log'),
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 交易日判断
# ==============================================================================

def is_trading_day(date: datetime = None) -> bool:
    """
    判断是否为交易日
    
    简化版本：周一至周五视为交易日
    实际应用中应该对接交易日历API
    
    参数:
        date: 要判断的日期，默认为今天
        
    返回:
        bool: 是否为交易日
    """
    if date is None:
        date = datetime.now()
    
    # 周末不是交易日
    if date.weekday() >= 5:
        return False
    
    # TODO: 对接交易日历API，排除节假日
    # 这里简化处理，只判断周末
    
    return True


def is_market_closed() -> bool:
    """
    判断市场是否已收盘
    
    A股收盘时间: 15:00
    数据更新时间: 15:30 (给数据源一些处理时间)
    
    返回:
        bool: 市场是否已收盘
    """
    now = datetime.now()
    market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)
    
    return now >= market_close


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    """
    主程序入口
    
    执行流程:
    1. 检查是否为交易日
    2. 检查市场是否已收盘
    3. 下载当天数据
    4. 记录日志
    """
    logger.info("=" * 60)
    logger.info("开始执行定时数据更新任务")
    logger.info("=" * 60)
    
    today = datetime.now()
    logger.info(f"当前时间: {today.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 检查是否为交易日
    if not is_trading_day(today):
        logger.info("今天不是交易日，跳过更新")
        return
    
    # 2. 检查市场是否已收盘
    if not is_market_closed():
        logger.warning("市场尚未收盘，建议在15:30后执行")
        # 继续执行，但给出警告
    
    # 3. 下载数据
    logger.info("开始下载当天日线数据...")
    
    try:
        # 合并股票和指数列表
        all_codes = DEFAULT_STOCKS + DEFAULT_INDICES
        results = download_today_data(all_codes)
        
        logger.info(f"更新完成: 更新 {results['updated']}, 跳过 {results['skipped']}, 失败 {results['failed']}")
        
        # 4. 记录结果
        if results['failed'] > 0:
            logger.warning(f"有 {results['failed']} 个标的更新失败，请检查")
        else:
            logger.info("所有数据更新成功")
            
    except Exception as e:
        logger.error(f"数据更新过程中发生错误: {str(e)}")
        raise
    
    logger.info("=" * 60)
    logger.info("定时数据更新任务完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
