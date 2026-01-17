import os
import subprocess
import datetime
import argparse
from pathlib import Path
try:
    from a99_logger import log
except ImportError:
    def log(msg, level="INFO"): print(f"[{level}] {msg}")

def run_script(script_name, args=None):
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print(f"错误: 找不到脚本 {script_name}")
        return False
    
    cmd = ["python3", str(script_path)]
    if args:
        cmd.extend(args)
    
    log(f"正在执行: {' '.join(cmd)}")
    # 执行前检查内存
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        log(f"执行失败: {script_name}, 错误: {e}", level="ERROR")
        return False

def main():
    parser = argparse.ArgumentParser(description='每日自动更新主控脚本')
    parser.add_argument('--full', action='store_true', help='全量更新模式')
    parser.add_argument('--incremental', action='store_true', default=True, help='增量更新模式 (默认)')
    args_cmd = parser.parse_args()
    
    if args_cmd.full:
        args_cmd.incremental = False
        
    mode_str = "全量" if args_cmd.full else "增量"
    log(f"=== 每日自动更新开始 (模式: {mode_str}) ===")
    
    # 1. 下载数据
    fetcher_args = ["--today"]
    if args_cmd.full:
        fetcher_args = ["--full"]
        
    if not run_script("a1_data_fetcher.py", fetcher_args):
        log("数据下载失败，停止后续任务", level="ERROR")
        return

    # 2. 运行回测
    backtest_args = ["--incremental"] if args_cmd.incremental else ["--full"]
    run_script("a2_single_strategy_backtest.py", backtest_args)
    run_script("a3_combo_strategy_backtest.py", backtest_args)
    
    # 3. 运行分析
    run_script("a4_signal_success_scanner.py")
    run_script("a21_pattern_analyzer.py")
    
    # 4. 生成个股报告
    run_script("a5_generate_stock_reports.py")
    
    # 5. 更新Web元数据
    run_script("a99_update_web_data.py")
    
    log("=== 每日自动更新完成 ===")

if __name__ == "__main__":
    main()
