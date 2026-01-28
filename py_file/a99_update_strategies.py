#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a99_update_strategies.py
功能描述: 根据回测结果汇总生成并更新 strategies.json
================================================================================
"""
import os
import json
import pandas as pd
from datetime import datetime

# ------------------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------------------
def find_project_root() -> str:
    # 尝试从当前工作目录或脚本所在目录向上查找
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for start_dir in [cwd, script_dir]:
        curr = start_dir
        while curr != os.path.dirname(curr):
            if os.path.isdir(os.path.join(curr, "web", "client", "public", "data")):
                return curr
            curr = os.path.dirname(curr)
    
    # 默认回退
    return "/home/ubuntu/tdx-strategy-backtest"

PROJECT_ROOT = find_project_root()
REPORT_DIR = os.path.join(PROJECT_ROOT, "report", "total")
WEB_DATA_DIR = os.path.join(PROJECT_ROOT, "web", "client", "src", "data")
PUBLIC_DATA_DIR = os.path.join(PROJECT_ROOT, "web", "client", "public", "data")
STRATEGIES_FILE = os.path.join(PUBLIC_DATA_DIR, "strategies.json")
SINGLE_CSV = os.path.join(REPORT_DIR, "single_strategy_summary.csv")
COMBO_CSV = os.path.join(REPORT_DIR, "combo_strategy_summary.csv")

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def update_strategies():
    log(f"项目根目录: {PROJECT_ROOT}")
    if not os.path.exists(STRATEGIES_FILE):
        log(f"错误: 找不到模板文件 {STRATEGIES_FILE}")
        return

    with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
        web_data = json.load(f)

    # 1. 更新单指标策略 (singleIndicatorStrategies)
    if os.path.exists(SINGLE_CSV):
        log(f"正在从 {SINGLE_CSV} 更新单指标策略...")
        df_single = pd.read_csv(SINGLE_CSV)
        
        # ID 映射 (前端 ID -> 脚本策略名)
        id_map = {
            "six_veins_6red": "four_red_plus", # 脚本中是 four_red_plus
            "buy_point_1": "buy1",
            "buy_point_2": "buy2",
            "chan_buy1": "chan_buy1",
            "macd_red": "macd_red",
            "kdj_red": "kdj_red",
            "rsi_red": "rsi_red",
            "lwr_red": "lwr_red",
            "bbi_red": "bbi_red",
            "mtm_red": "mtm_red"
        }

        for strategy in web_data.get('singleIndicatorStrategies', []):
            s_id = strategy['id']
            script_name = id_map.get(s_id, s_id)
            
            df_strat = df_single[df_single['strategy'] == script_name]
            if df_strat.empty:
                continue
            
            # 计算各周期的汇总
            summary_by_period = {}
            for period, df_p in df_strat.groupby('hold_period'):
                trades = int(df_p['trade_count'].sum())
                wins = int(df_p['win_count'].sum())
                sum_ret = float(df_p['sum_return'].sum())
                
                win_rate = round(wins / trades * 100, 2) if trades > 0 else 0.0
                avg_ret = round(sum_ret / trades, 2) if trades > 0 else 0.0
                
                summary_by_period[str(int(period))] = {
                    "winRate": f"{win_rate}%",
                    "avgReturn": f"{avg_ret}%",
                    "trades": trades
                }
            
            if not summary_by_period:
                continue
                
            # 寻找最优周期 (以胜率为准)
            best_period = max(summary_by_period.keys(), key=lambda k: float(summary_by_period[k]['winRate'].replace('%','')))
            best_stats = summary_by_period[best_period]
            
            strategy['stats']['total'] = {
                "winRate": best_stats['winRate'],
                "avgReturn": best_stats['avgReturn'],
                "optimalPeriod": f"{best_period}天",
                "trades": best_stats['trades']
            }
            log(f"已更新单指标策略: {s_id} (最优周期: {best_period}天, 胜率: {best_stats['winRate']})")

    # 2. 更新组合策略 (strategies)
    if os.path.exists(COMBO_CSV):
        log(f"正在从 {COMBO_CSV} 更新组合策略...")
        df_combo = pd.read_csv(COMBO_CSV)
        
        for strategy in web_data.get('strategies', []):
            s_id = strategy['id']
            df_strat = df_combo[df_combo['strategy'] == s_id]
            if df_strat.empty:
                continue
                
            summary_by_period = {}
            for period, df_p in df_strat.groupby('hold_period'):
                trades = int(df_p['trade_count'].sum())
                wins = int(df_p['win_count'].sum())
                sum_ret = float(df_p['sum_return'].sum())
                
                win_rate = round(wins / trades * 100, 2) if trades > 0 else 0.0
                avg_ret = round(sum_ret / trades, 2) if trades > 0 else 0.0
                
                summary_by_period[str(int(period))] = {
                    "winRate": f"{win_rate}%",
                    "avgReturn": f"{avg_ret}%",
                    "trades": trades
                }
            
            if not summary_by_period:
                continue
                
            best_period = max(summary_by_period.keys(), key=lambda k: float(summary_by_period[k]['winRate'].replace('%','')))
            best_stats = summary_by_period[best_period]
            
            strategy['stats']['total'] = {
                "winRate": best_stats['winRate'],
                "avgReturn": best_stats['avgReturn'],
                "optimalPeriod": f"{best_period}天",
                "trades": best_stats['trades']
            }
            log(f"已更新组合策略: {s_id} (最优周期: {best_period}天, 胜率: {best_stats['winRate']})")

    # 3. 保存到两个位置
    os.makedirs(WEB_DATA_DIR, exist_ok=True)
    target_files = [STRATEGIES_FILE, os.path.join(WEB_DATA_DIR, "strategies.json")]
    
    for target in target_files:
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(web_data, f, ensure_ascii=False, indent=2)
        log(f"已保存更新到: {target}")

if __name__ == "__main__":
    update_strategies()
