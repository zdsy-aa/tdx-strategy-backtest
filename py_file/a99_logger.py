#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
通用日志与内存监控模块 (Logger & Memory Monitor)
================================================================================

功能描述:
    1. 提供带时间戳的日志打印功能。
    2. 提供内存监控功能，防止内存占用过高导致系统崩溃。

作者: TradeGuide System
版本: 1.0.0
更新日期: 2026-01-17
================================================================================
"""

import sys
import time
import psutil
import os
from datetime import datetime

def log(message, level="INFO"):
    """
    打印带时间戳的日志
    格式: [YYYY-MM-DD HH:MM:SS] [LEVEL] MESSAGE
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def check_memory(threshold=0.90):
    """
    检查当前系统内存占用情况
    如果超过阈值（默认90%），则打印警告并退出程序
    """
    memory = psutil.virtual_memory()
    usage = memory.percent / 100.0
    
    if usage > threshold:
        log(f"内存占用过高: {memory.percent}% (阈值: {threshold*100}%)", level="CRITICAL")
        log("为了防止系统崩溃，程序将自动退出。", level="CRITICAL")
        # 尝试获取当前进程信息
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        log(f"当前进程内存占用: {mem_info.rss / 1024 / 1024:.2f} MB", level="CRITICAL")
        sys.exit(1)
    return usage

def init_monitoring(interval=60):
    """
    初始化监控（可选，用于在后台线程或循环中调用）
    """
    log("系统监控已初始化 (内存阈值: 90%)")
    check_memory()

# 替换内置 print (可选，但建议显式调用 log)
# print = log 
