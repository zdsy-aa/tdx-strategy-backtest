#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a0_auto_update_daily.py
================================================================================

【脚本功能】
    全自动更新主控脚本，负责协调并按顺序调用项目中的各个核心模块：
    1. 数据抓取 (a1_data_fetcher_mootdx.py)
    2. 统一回测与前端数据同步 (a2_unified_backtest.py)
    3. 模式特征分析 (a21_pattern_analyzer.py)
    4. 统一分析报表与预测 (a5_unified_analysis.py)

【使用方法】
    在终端或命令行中运行：
    
    1. 增量更新模式 (默认，推荐每日收盘后运行):
       python3 a0_auto_update_daily.py
       
    2. 全量更新模式 (首次运行或需要重刷全部数据时使用):
       python3 a0_auto_update_daily.py --full
       
    3. 指定日期范围的增量更新:
       python3 a0_auto_update_daily.py 20240101 20240131

【设计优势】
    - 流程自动化：一键完成从数据下载到前端展示所需的所有计算。
    - 错误容忍：单个脚本失败不会导致主控脚本崩溃，并会记录错误日志。
    - 逻辑解耦：各模块独立运行，主控脚本仅负责调度。
================================================================================
"""

import os
import subprocess
import datetime
import argparse
from pathlib import Path

try:
    from a99_logger import log
except ImportError:
    def log(msg, level="INFO"): 
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {msg}")

def run_script(script_name, description, args=None):
    """
    统一的脚本调用函数
    """
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        log(f"错误: 找不到脚本 {script_name}", level="ERROR")
        return False
    
    log("=" * 50)
    log(f"开始调用脚本: {script_name}")
    log(f"功能说明: {description}")
    log("=" * 50)
    
    cmd = ["python3", str(script_path)]
    if args:
        cmd.extend(args)
    
    log(f"正在执行命令: {' '.join(cmd)}")
    
    try:
        # 使用 subprocess.run 执行脚本并等待完成
        subprocess.run(cmd, check=True)
        log(f"脚本执行成功: {script_name}")
        return True
    except subprocess.CalledProcessError as e:
        log(f"执行失败: {script_name}, 错误代码: {e.returncode}", level="ERROR")
        return False
    except Exception as e:
        log(f"执行过程中发生异常: {script_name}, 错误: {str(e)}", level="ERROR")
        return False

def main():
    parser = argparse.ArgumentParser(description='每日自动更新主控脚本（整合优化版）')
    parser.add_argument('--full', action='store_true', help='全量更新模式')
    parser.add_argument('--incremental', action='store_true', default=True, help='增量更新模式 (默认)')
    parser.add_argument('dates', nargs='*', help='增量模式下的日期范围 (YYYYMMDD YYYYMMDD)')
    args_cmd = parser.parse_args()
    
    # 确定数据抓取脚本名称（优先使用 mootdx 版本）
    fetcher_script = "a1_data_fetcher_mootdx.py"
    
    if args_cmd.full:
        log("=== 全量更新模式 ===")
        # 1. 全量更新基础数据
        run_script(fetcher_script, "全量抓取最新基础数据", args=["--all"])
        
        # 2. 统一回测（整合了 a2/a3/a4，包含自动前端更新）
        run_script("a2_unified_backtest.py", "单策略回测（含前端更新）", args=["--mode", "single", "--strategy", "all"])
        run_script("a2_unified_backtest.py", "组合策略回测（含前端更新）", args=["--mode", "combo", "--strategy", "all"])
        run_script("a2_unified_backtest.py", "信号成功案例扫描", args=["--mode", "scan", "--holding_days", "15"])
        
        # 3. 模式分析
        run_script("a21_pattern_analyzer.py", "分析成功案例的共性模式特征")
        
        # 4. 统一分析（整合了 a5/a6/a7）
        run_script("a5_unified_analysis.py", "股票报表+仪表盘+预测分析", args=["--mode", "all"])
        
    else:
        log("=== 增量更新模式 ===")
        # 根据提供的日期范围过滤增量数据
        date_args = args_cmd.dates if args_cmd.dates else []
        
        # 1. 增量更新基础数据
        fetch_args = ["--all", "--start"]
        if date_args:
            fetch_args = ["--all", "--start", date_args[0]]
            if len(date_args) > 1:
                fetch_args.extend(["--end", date_args[1]])
        
        run_script(fetcher_script, "增量抓取基础数据", args=fetch_args)
        
        # 2. 统一回测（整合了 a2/a3/a4，包含自动前端更新）
        run_script("a2_unified_backtest.py", "单策略回测（含前端更新）", args=["--mode", "single", "--strategy", "all"])
        run_script("a2_unified_backtest.py", "组合策略回测（含前端更新）", args=["--mode", "combo", "--strategy", "all"])
        run_script("a2_unified_backtest.py", "信号成功案例扫描", args=["--mode", "scan", "--holding_days", "15"])
        
        # 3. 模式分析（始终全量分析全部成功案例）
        run_script("a21_pattern_analyzer.py", "分析成功案例的共性模式特征")
        
        # 4. 统一分析（整合了 a5/a6/a7）
        run_script("a5_unified_analysis.py", "股票报表+仪表盘+预测分析", args=["--mode", "all"])

    log("=" * 50)
    log("所有任务执行完毕")
    log("=" * 50)

if __name__ == "__main__":
    main()
