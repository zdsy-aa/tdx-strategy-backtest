import os
import subprocess
import datetime
from pathlib import Path

def run_script(script_name, args=None):
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print(f"错误: 找不到脚本 {script_name}")
        return False
    
    cmd = ["python3", str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"正在执行: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"执行失败: {script_name}, 错误: {e}")
        return False

def main():
    print(f"=== 每日自动更新开始: {datetime.datetime.now()} ===")
    
    # 1. 下载数据
    if not run_script("data_fetcher_1.py", ["--today"]):
        print("数据下载失败，停止后续任务")
        return

    # 2. 运行回测
    run_script("2_single_strategy_backtest.py")
    run_script("2_combo_strategy_backtest.py")
    
    # 3. 运行分析
    run_script("2_signal_success_scanner.py")
    run_script("2_1_pattern_analyzer.py")
    
    # 4. 生成个股报告
    run_script("2_generate_stock_reports.py")
    
    # 5. 更新Web元数据
    run_script("99_update_web_data.py")
    
    print(f"=== 每日自动更新完成: {datetime.datetime.now()} ===")

if __name__ == "__main__":
    main()
