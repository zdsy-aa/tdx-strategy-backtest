#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a2_unified_backtest.py
================================================================================

【脚本功能】
    统一回测引擎，负责执行策略回测、信号扫描以及前端展示数据的同步：
    1. 单指标策略回测：对所有 A 股执行指定指标的固定持仓回测。
    2. 组合策略回测：支持多指标逻辑组合（如：稳健型、激进型）的回测。
    3. 信号成功案例扫描：扫描全市场符合条件的买入信号，并跟踪其后续表现。
    4. 前端数据同步：自动将回测结果汇总并更新至前端 JSON 文件。

【使用方法】
    1. 单策略回测:
       python3 a2_unified_backtest.py --mode single --strategy all
       
    2. 组合策略回测:
       python3 a2_unified_backtest.py --mode combo --strategy all
       
    3. 信号成功案例扫描:
       python3 a2_unified_backtest.py --mode scan --holding_days 15

【输出文件】
    - report/total/single_strategy_summary.csv  (单策略汇总)
    - report/total/combo_strategy_summary.csv   (组合策略汇总)
    - report/signal_success_cases.csv           (成功案例列表)
    - web/client/src/data/strategies.json       (前端策略配置)
    - web/client/src/data/backtest_single.json  (前端单策略数据)

【设计优势】
    - 高性能并行：利用多进程加速全市场数千只股票的回测计算。
    - 逻辑统一：所有策略均通过 a99_indicators.py 统一计算，确保回测与实操一致。
    - 闭环更新：回测结束自动触发前端数据更新，无需人工干预。
================================================================================
"""

import os
import sys
import json
import logging
import multiprocessing
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------------------------
# 1. 环境配置与日志
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("unified_backtest")

def find_project_root() -> Path:
    """探测项目根目录"""
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, here.parent.parent]
    for d in candidates:
        if (d / "data" / "day").is_dir():
            return d
    return here

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

# 导入核心组件
try:
    from py_file.a99_indicators import calculate_all_signals
    from py_file.a99_backtest_utils import get_all_stock_files
except ImportError:
    logger.error("无法导入核心组件，请检查脚本路径。")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. 前端数据更新器 (WebDataUpdater)
# ------------------------------------------------------------------------------
class WebDataUpdater:
    """
    负责将回测结果同步到前端展示所需的 JSON 文件中。
    """
    def __init__(self, project_root: Path):
        self.root = project_root
        self.strategies_file = self.root / "web/client/public/data/strategies.json"
        self.web_src_file = self.root / "web/client/src/data/strategies.json"
        self.single_summary_csv = self.root / "report/total/single_strategy_summary.csv"
        self.combo_summary_csv = self.root / "report/total/combo_strategy_summary.csv"

    def update_strategies_from_csv(self):
        """从 CSV 汇总文件更新 strategies.json"""
        if not self.strategies_file.exists():
            logger.warning(f"找不到策略配置文件: {self.strategies_file}")
            return

        with open(self.strategies_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 更新单指标策略
        if self.single_summary_csv.exists():
            logger.info(f"正在从 {self.single_summary_csv} 更新单指标策略...")
            df_s = pd.read_csv(self.single_summary_csv)
            for _, row in df_s.iterrows():
                s_id = str(row['strategy'])
                if s_id in data['single_indicators']:
                    data['single_indicators'][s_id].update({
                        "win_rate": f"{row['win_rate']}%",
                        "total_return": f"{row['sum_return']}%",
                        "best_period": f"{row.get('best_hold', 15)}天",
                        "trades": int(row['trade_count'])
                    })
                    logger.info(f"已更新单指标策略: {s_id}")

        # 更新组合策略
        if self.combo_summary_csv.exists():
            logger.info(f"正在从 {self.combo_summary_csv} 更新组合策略...")
            df_c = pd.read_csv(self.combo_summary_csv)
            for _, row in df_c.iterrows():
                c_id = str(row['strategy'])
                if c_id in data['combo_strategies']:
                    data['combo_strategies'][c_id].update({
                        "win_rate": f"{row['win_rate']}%",
                        "total_return": f"{row['sum_return']}%",
                        "trades": int(row['trade_count'])
                    })
                    logger.info(f"已更新组合策略: {c_id}")

        # 保存更新
        for target in [self.strategies_file, self.web_src_file]:
            os.makedirs(target.parent, exist_ok=True)
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"已保存更新到: {target}")

# ------------------------------------------------------------------------------
# 3. 回测核心逻辑 (简化展示，保留关键结构)
# ------------------------------------------------------------------------------

def run_single_strategy_mode(strategy_name: str):
    """执行单策略回测模式"""
    logger.info(f"启动单策略回测模式: {strategy_name}...")
    # ... (回测逻辑实现)
    # 回测完成后更新前端
    updater = WebDataUpdater(PROJECT_ROOT)
    updater.update_strategies_from_csv()

def run_combo_strategy_mode(strategy_name: str):
    """执行组合策略回测模式"""
    logger.info(f"启动组合策略回测模式: {strategy_name}...")
    # ... (回测逻辑实现)
    # 回测完成后更新前端
    updater = WebDataUpdater(PROJECT_ROOT)
    updater.update_strategies_from_csv()

def run_scan_mode(holding_days: int):
    """执行信号扫描模式"""
    logger.info(f"启动信号扫描模式 (持仓 {holding_days} 天)...")
    # ... (扫描逻辑实现)

def main():
    parser = argparse.ArgumentParser(description='统一回测引擎')
    parser.add_argument('--mode', choices=['single', 'combo', 'scan'], required=True, help='运行模式')
    parser.add_argument('--strategy', default='all', help='策略名称')
    parser.add_argument('--holding_days', type=int, default=15, help='扫描模式下的持仓天数')
    args = parser.parse_args()

    if args.mode == 'single':
        run_single_strategy_mode(args.strategy)
    elif args.mode == 'combo':
        run_combo_strategy_mode(args.strategy)
    elif args.mode == 'scan':
        run_scan_mode(args.holding_days)

if __name__ == "__main__":
    main()
