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
    parser = argparse.ArgumentParser(description='每日自动更新主控脚本')
    parser.add_argument('--full', action='store_true', help='全量更新模式')
    parser.add_argument('--incremental', action='store_true', default=True, help='增量更新模式 (默认)')
    parser.add_argument('dates', nargs='*', help='增量模式下的日期范围 (YYYYMMDD YYYYMMDD)')
    args_cmd = parser.parse_args()
    
    if args_cmd.full:
        args_cmd.incremental = False
        
    mode_str = "全量" if args_cmd.full else "增量"
    log(f"=== 每日自动更新开始 (模式: {mode_str}) ===")
    
    # 1. 下载数据
    fetcher_args = ["--today"]
    if args_cmd.full:
        fetcher_args = ["--full"]
    elif args_cmd.incremental and len(args_cmd.dates) == 2:
        fetcher_args = ["--date", args_cmd.dates[0], args_cmd.dates[1]]
        log(f"指定日期范围: {args_cmd.dates[0]} - {args_cmd.dates[1]}")
        
    if not run_script("a1_data_fetcher.py", "下载并更新股票行情数据", fetcher_args):
        log("数据下载失败，停止后续任务", level="ERROR")
        return

    # 2. 运行回测
    backtest_args = ["--incremental"] if args_cmd.incremental else ["--full"]
    run_script("a2_single_strategy_backtest.py", "执行单指标策略回测", backtest_args)
    run_script("a3_combo_strategy_backtest.py", "执行组合指标策略回测", backtest_args)
    
    # 3. 运行分析
    run_script("a4_signal_success_scanner.py", "扫描并分析信号成功率")
    run_script("a21_pattern_analyzer.py", "执行形态模式分析")
    
    # 4. 生成个股报告
    run_script("a5_generate_stock_reports.py", "生成个股详细回测报告")
    
    # 5. 更新Web元数据
    run_script("a99_update_web_data.py", "更新Web前端所需的元数据文件")
    
    log("=" * 50)
    log("=== 每日自动更新完成 ===")
    log("=" * 50)

if __name__ == "__main__":
    main()
