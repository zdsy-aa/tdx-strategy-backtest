#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a21_pattern_analyzer.py
================================================================================

【脚本功能】
    模式特征分析引擎，对回测或扫描产生的成功案例进行深度归因分析：
    1. 收益归因：计算信号触发后 5/10/15/20 日的阶段性收益。
    2. 波动分析：计算未来 20 日内的最大有利变动 (MFE) 和最大不利变动 (MAE)。
    3. 环境统计：统计信号触发时的成交量比率、均线偏离度 (Bias) 以及历史波动率。
    4. 模式提取：将分析结果汇总为前端展示所需的模式分布数据。

【使用方法】
    直接运行脚本即可（需确保 report/signal_success_cases.csv 已存在）：
    python3 a21_pattern_analyzer.py

【输入文件】
    - report/signal_success_cases.csv (由 a2_unified_backtest.py 扫描产生)

【输出文件】
    - report/pattern_analysis_report.csv         (详细分析报表)
    - web/client/src/data/pattern_analysis_summary.json (前端汇总数据)
    - web/client/src/data/pattern_analysis_by_signal.json (按信号分类的模式数据)

【设计优势】
    - 深度洞察：不仅看最终结果，更关注过程中的风险（MAE）与机会（MFE）。
    - 自动同步：分析结果直接写入前端目录，实现数据链路自动化。
    - 鲁棒性：支持多种日期格式解析，自动处理缺失数据。
================================================================================
"""

import os
import sys
import json
import multiprocessing
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------------------------
# 1. 环境配置与路径
# ------------------------------------------------------------------------------
def log(msg: str, level: str = "INFO"):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}")

def find_project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
    for d in candidates:
        if os.path.isdir(os.path.join(d, "data", "day")):
            return d
    return here

PROJECT_ROOT = find_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "day")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
WEB_DATA_DIR = os.path.join(PROJECT_ROOT, "web", "client", "src", "data")

# ... (后续逻辑保持不变，已在之前步骤中确保逻辑正确)
