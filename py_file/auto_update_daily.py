#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化每日数据更新和回测脚本

功能：
1. 下载最新的股票数据
2. 执行完整回测
3. 生成最新的 JSON 数据文件
4. 记录执行日志

使用方式：
- 手动运行: python3 auto_update_daily.py
- 定时任务: 配合 crontab 或 systemd timer
"""

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# 设置日志
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"auto_update_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# 脚本目录
SCRIPT_DIR = Path(__file__).parent


def run_command(script_name, description):
    """运行 Python 脚本"""
    script_path = SCRIPT_DIR / script_name
    
    if not script_path.exists():
        logger.error(f"脚本不存在: {script_path}")
        return False
    
    logger.info(f"{'='*60}")
    logger.info(f"开始执行: {description}")
    logger.info(f"脚本路径: {script_path}")
    logger.info(f"{'='*60}")
    
    try:
        start_time = datetime.now()
        
        # 执行脚本
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 输出执行结果
        if result.stdout:
            logger.info(f"标准输出:\n{result.stdout}")
        
        if result.stderr:
            logger.warning(f"错误输出:\n{result.stderr}")
        
        if result.returncode == 0:
            logger.info(f"✅ {description} 执行成功 (耗时: {duration:.1f}秒)")
            return True
        else:
            logger.error(f"❌ {description} 执行失败 (返回码: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} 执行超时 (超过2小时)")
        return False
    except Exception as e:
        logger.error(f"❌ {description} 执行异常: {str(e)}")
        return False


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("股票数据自动更新任务开始")
    logger.info(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    total_start = datetime.now()
    
    # 步骤1: 下载股票数据
    success_download = run_command(
        'stock_downloader_script.py',
        '下载股票数据'
    )
    
    if not success_download:
        logger.error("数据下载失败，终止任务")
        return 1
    
    # 步骤2: 执行完整回测
    success_backtest = run_command(
        'full_backtest.py',
        '执行完整回测'
    )
    
    if not success_backtest:
        logger.error("回测执行失败")
        return 1
    
    # 任务完成
    total_end = datetime.now()
    total_duration = (total_end - total_start).total_seconds()
    
    logger.info("="*80)
    logger.info("✅ 所有任务执行完成")
    logger.info(f"总耗时: {total_duration/60:.1f} 分钟")
    logger.info(f"日志文件: {log_file}")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\n任务被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"任务执行异常: {str(e)}", exc_info=True)
        sys.exit(1)
