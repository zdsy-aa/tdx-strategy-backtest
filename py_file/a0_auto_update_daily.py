import os
import subprocess
import datetime
import argparse
from pathlib import Path
try:
    from a99_logger import log
except ImportError:
    def log(msg, level="INFO"): print(f"[{level}] {msg}")

def run_script(script_name, description, args=None):
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        log(f"错误: 找不到脚本 {script_name}", level="ERROR")
        return False
    
    # 添加开始调用的日志说明
    log("=" * 50)
    log(f"开始调用脚本: {script_name}")
    log(f"功能说明: {description}")
    log("=" * 50)
    
    cmd = ["python3", str(script_path)]
    if args:
        cmd.extend(args)
    
    log(f"正在执行命令: {' '.join(cmd)}")
    
    try:
        # 使用 subprocess.run 执行脚本
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
    
    if args_cmd.full:
        log("=== 全量更新模式 ===")
        # 1. 全量更新基础数据
        run_script("a1_data_fetcher.py", "全量抓取最新基础数据", args=["--full"])
        
        # 2. 统一回测（整合了 a2/a3/a4，包含自动前端更新）
        run_script("a2_unified_backtest.py", "单策略回测（含前端更新）", args=["--mode", "single", "--strategy", "all"])
        run_script("a2_unified_backtest.py", "组合策略回测（含前端更新）", args=["--mode", "combo", "--strategy", "all"])
        run_script("a2_unified_backtest.py", "信号成功案例扫描", args=["--mode", "scan", "--holding_days", "15"])
        
        # 3. 模式分析
        run_script("a21_pattern_analyzer.py", "分析成功案例的共性模式特征")
        
        # 4. 统一分析（整合了 a5/a6/a7）
        run_script("a5_unified_analysis.py", "股票报表+仪表盘+预测分析", args=["--mode", "all"])

        # ✅ 前端数据更新已整合到 a2_unified_backtest.py 中，无需单独调用
        
    else:
        log("=== 增量更新模式 ===")
        # 根据提供的日期范围过滤增量数据
        date_args = args_cmd.dates if args_cmd.dates else []
        
        # 1. 增量更新基础数据（指定日期范围）
        run_script("a1_data_fetcher.py", "增量抓取基础数据", args=date_args)
        
        # 2. 统一回测（整合了 a2/a3/a4，包含自动前端更新）
        run_script("a2_unified_backtest.py", "单策略回测（含前端更新）", args=["--mode", "single", "--strategy", "all"])
        run_script("a2_unified_backtest.py", "组合策略回测（含前端更新）", args=["--mode", "combo", "--strategy", "all"])
        run_script("a2_unified_backtest.py", "信号成功案例扫描", args=["--mode", "scan", "--holding_days", "15"])
        
        # 3. 模式分析（始终全量分析全部成功案例）
        run_script("a21_pattern_analyzer.py", "分析成功案例的共性模式特征")
        
        # 4. 统一分析（整合了 a5/a6/a7）
        run_script("a5_unified_analysis.py", "股票报表+仪表盘+预测分析", args=["--mode", "all"])

        # ✅ 前端数据更新已整合到 a2_unified_backtest.py 中，无需单独调用

    log("=" * 50)
    log("所有任务执行完毕")
    log("=" * 50)

if __name__ == "__main__":
    main()

print("a0_auto_update_daily.py 脚本执行完毕")
