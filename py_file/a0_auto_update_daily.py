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
        log("=== 全量更新模式 ===")
        # 1. 全量更新基础数据
        run_script("a1_data_fetcher.py", "全量抓取最新基础数据", args=["--full"])
        # 2. 扫描所有信号成功案例
        run_script("a4_signal_success_scanner.py", "扫描全部股票的买入信号成功案例", args=["--full"])
        # 3. 分析成功案例共性模式
        run_script("a21_pattern_analyzer.py", "分析成功案例的共性模式特征")
        # 4. 回测所有单指标策略
        run_script("a2_single_strategy_backtest.py", "全市场单指标策略回测", args=["--strategy", "all"])
        # 5. 回测所有组合策略
        run_script("a3_combo_strategy_backtest.py", "全市场组合策略回测", args=["--strategy", "all"])
        # 6. 生成股票报告明细
        run_script("a5_generate_stock_reports.py", "生成所有股票的策略表现报告")
        # 7. 运行 AI 模型评分系统
        run_script("a6_models.py", "运行 AI 模型评分系统，生成仪表盘数据", args=["--include-series"])
        # 8. 运行高级预测分析
        run_script("a7_advanced_forecast.py", "运行高级预测分析，生成预测数据")
        # 9. 更新前端展示所需的数据文件 (功能已合并到 a2/a3 脚本中，此处移除)
        # run_script("a99_update_web_data.py", "更新前端网页数据")
    else:
        log("=== 增量更新模式 ===")
        # 根据提供的日期范围过滤增量数据
        date_args = args_cmd.dates if args_cmd.dates else []
        # 1. 增量更新基础数据（指定日期范围）
        run_script("a1_data_fetcher.py", "增量抓取基础数据", args=date_args)
        # 2. 扫描指定日期范围内的新成功案例
        run_script("a4_signal_success_scanner.py", "扫描新增成功案例信号", args=date_args)
        # 3. 模式分析（始终全量分析全部成功案例）
        run_script("a21_pattern_analyzer.py", "分析成功案例的共性模式特征")
        # 4. 单指标策略回测（全市场，每次全量回测以保持结果最新）
        run_script("a2_single_strategy_backtest.py", "全市场单指标策略回测", args=["--strategy", "all"])
        # 5. 组合策略回测（全市场，每次全量回测以保持结果最新）
        run_script("a3_combo_strategy_backtest.py", "全市场组合策略回测", args=["--strategy", "all"])
        # 6. 更新股票报告（全市场）
        run_script("a5_generate_stock_reports.py", "生成所有股票的策略表现报告")
        # 7. 运行 AI 模型评分系统
        run_script("a6_models.py", "运行 AI 模型评分系统，生成仪表盘数据", args=["--include-series"])
        # 8. 运行高级预测分析
        run_script("a7_advanced_forecast.py", "运行高级预测分析，生成预测数据")
        # 9. 更新前端数据文件 (功能已合并到 a2/a3 脚本中，此处移除)
        # run_script("a99_update_web_data.py", "更新前端网页数据")

    log("=" * 50)
    log("所有任务执行完毕")
    log("=" * 50)

if __name__ == "__main__":
    main()

print("a0_auto_update_daily.py 脚本执行完毕")
