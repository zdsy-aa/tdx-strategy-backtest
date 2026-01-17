#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
通用日志模块 (Logger)
================================================================================

功能描述:
    1. 提供带时间戳的日志打印功能。

作者: TradeGuide System
版本: 1.1.0
更新日期: 2026-01-17
================================================================================
"""

from datetime import datetime

def log(message, level="INFO"):
    """
    打印带时间戳的日志
    格式: [YYYY-MM-DD HH:MM:SS] [LEVEL] MESSAGE
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
