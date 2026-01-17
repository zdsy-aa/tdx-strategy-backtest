import os
import subprocess
import datetime
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
    log("=== 每日自动更新开始 ===")
    
    # 1. 下载数据
    if not run_script("a1_data_fetcher.py", ["--today"]):
        log("数据下载失败，停止后续任务", level="ERROR")
        return

    # 2. 运行回测
    run_script("a2_single_strategy_backtest.py")
    run_script("a3_combo_strategy_backtest.py")
    
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
