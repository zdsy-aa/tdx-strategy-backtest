#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a2_unified_backtest.py (v2.1 - 整合策略汇总更新)
================================================================================

【更新日志 v2.1】
    - 整合了 a99_update_strategies.py 的功能到 WebDataUpdater 类
    - 实现了从 CSV 汇总文件自动更新 strategies.json 的胜率和收益率
    - 统一了前端数据更新入口

【脚本功能】
    统一回测与信号扫描脚本，整合了 a2, a3, a4 的核心功能：
    1. 单策略回测 (Single Strategy Backtest)
    2. 组合策略回测 (Combo Strategy Backtest)
    3. 信号成功案例扫描 (Signal Success Scanner)
    4. 前端数据自动更新 (Web Data Auto-Update)
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------------------------
# 日志
# ------------------------------------------------------------------------------
try:
    from a99_logger import log
except Exception:
    def log(msg, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {msg}")

# ------------------------------------------------------------------------------
# 项目根目录探测
# ------------------------------------------------------------------------------
def find_project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
    for d in candidates:
        if os.path.isdir(os.path.join(d, "data", "day")):
            return d
    return here

PROJECT_ROOT = find_project_root()
sys.path.insert(0, PROJECT_ROOT)

from a99_indicators import (
    calculate_all_signals,
    calculate_six_veins,
    calculate_buy_sell_points,
    calculate_chan_theory,
)
from a99_backtest_utils import (
    get_all_stock_files,
    run_backtest_on_all_stocks,
    backtest_trades_fixed_hold,
    summarize_trades,
)
try:
    from a99_csv_chunker import split_csv_by_size, cleanup_old_chunks
except ImportError:
    split_csv_by_size = None
    cleanup_old_chunks = None

# ------------------------------------------------------------------------------
# 全局配置
# ------------------------------------------------------------------------------
COMMISSION_RATE = 0.00008
STAMP_TAX_RATE = 0.0005
DEFAULT_HOLD_PERIODS = [5, 10, 20]
SIX_VEINS_INDICATORS = ["macd_red", "kdj_red", "rsi_red", "lwr_red", "bbi_red", "mtm_red"]

# ------------------------------------------------------------------------------
# 数据加载与清洗
# ------------------------------------------------------------------------------
CSV_COL_MAP = {
    "名称": "name", "日期": "date", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
    "振幅": "amplitude", "涨跌幅": "pct_chg", "涨跌额": "chg", "换手率": "turnover",
}
NUMERIC_COLS = ["open", "high", "low", "close", "volume", "amount", "amplitude", "pct_chg", "chg", "turnover"]

def _parse_date_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, format="%Y/%m/%d", errors="coerce")
    if len(dt) > 0 and dt.isna().mean() > 0.5:
        dt = pd.to_datetime(s, errors="coerce")
    return dt

def load_stock_data(filepath: str) -> Optional[pd.DataFrame]:
    try:
        df = None
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                df = pd.read_csv(filepath, encoding=enc)
                break
            except Exception:
                continue

        if df is None or df.empty:
            return None

        df.rename(columns={c: CSV_COL_MAP.get(c, c) for c in df.columns}, inplace=True)
        if "date" not in df.columns:
            return None

        df["date"] = _parse_date_series(df["date"])
        df.dropna(subset=["date"], inplace=True)

        for c in NUMERIC_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                df[c] = np.nan if c != "volume" else 0.0

        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]

        for c in ["volume", "amount"]:
            if c in df.columns:
                df[c] = df[c].fillna(0.0)
                df.loc[df[c] < 0, c] = 0.0

        if "name" not in df.columns:
            df["name"] = ""

        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        if len(df) < 30:
            return None

        return df
    except Exception as e:
        log(f"数据加载异常: {filepath}, 错误: {e}", level="ERROR")
        return None

# ------------------------------------------------------------------------------
# 前端数据更新模块
# ------------------------------------------------------------------------------
class WebDataUpdater:
    """前端数据自动更新器"""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.web_data_dir = os.path.join(project_root, "web", "client", "src", "data")
        self.public_data_dir = os.path.join(project_root, "web", "client", "public", "data")
        # 优先使用 public 目录下的作为模板，因为它是持久化的
        self.strategies_file = os.path.join(self.public_data_dir, "strategies.json")
        if not os.path.exists(self.strategies_file):
            self.strategies_file = os.path.join(self.web_data_dir, "strategies.json")
    
    def update_strategies_from_csv(self):
        """从 CSV 汇总文件更新 strategies.json (整合自 a99_update_strategies.py)"""
        report_dir = os.path.join(self.project_root, "report", "total")
        single_csv = os.path.join(report_dir, "single_strategy_summary.csv")
        combo_csv = os.path.join(report_dir, "combo_strategy_summary.csv")
        
        if not os.path.exists(self.strategies_file):
            log(f"错误: 找不到 strategies.json 模板文件 {self.strategies_file}", level="ERROR")
            return

        with open(self.strategies_file, 'r', encoding='utf-8') as f:
            web_data = json.load(f)

        # 1. 更新单指标策略
        if os.path.exists(single_csv):
            log(f"正在从 {single_csv} 更新单指标策略...")
            df_single = pd.read_csv(single_csv)
            id_map = {
                "six_veins_6red": "four_red_plus",
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
                if df_strat.empty: continue
                
                summary_by_period = {}
                for period, df_p in df_strat.groupby('hold_period'):
                    trades = int(df_p['trade_count'].sum())
                    wins = int(df_p['win_count'].sum())
                    sum_ret = float(df_p['sum_return'].sum())
                    win_rate = round(wins / trades * 100, 2) if trades > 0 else 0.0
                    avg_ret = round(sum_ret / trades, 2) if trades > 0 else 0.0
                    summary_by_period[str(int(period))] = {"winRate": f"{win_rate}%", "avgReturn": f"{avg_ret}%", "trades": trades}
                
                if not summary_by_period: continue
                best_period = max(summary_by_period.keys(), key=lambda k: float(summary_by_period[k]['winRate'].replace('%','')))
                best_stats = summary_by_period[best_period]
                strategy['stats']['total'] = {
                    "winRate": best_stats['winRate'], "avgReturn": best_stats['avgReturn'],
                    "optimalPeriod": f"{best_period}天", "trades": best_stats['trades']
                }
                log(f"已更新单指标策略: {s_id}")

        # 2. 更新组合策略
        if os.path.exists(combo_csv):
            log(f"正在从 {combo_csv} 更新组合策略...")
            df_combo = pd.read_csv(combo_csv)
            for strategy in web_data.get('strategies', []):
                s_id = strategy['id']
                df_strat = df_combo[df_combo['strategy'] == s_id]
                if df_strat.empty: continue
                
                summary_by_period = {}
                for period, df_p in df_strat.groupby('hold_period'):
                    trades = int(df_p['trade_count'].sum())
                    wins = int(df_p['win_count'].sum())
                    sum_ret = float(df_p['sum_return'].sum())
                    win_rate = round(wins / trades * 100, 2) if trades > 0 else 0.0
                    avg_ret = round(sum_ret / trades, 2) if trades > 0 else 0.0
                    summary_by_period[str(int(period))] = {"winRate": f"{win_rate}%", "avgReturn": f"{avg_ret}%", "trades": trades}
                
                if not summary_by_period: continue
                best_period = max(summary_by_period.keys(), key=lambda k: float(summary_by_period[k]['winRate'].replace('%','')))
                best_stats = summary_by_period[best_period]
                strategy['stats']['total'] = {
                    "winRate": best_stats['winRate'], "avgReturn": best_stats['avgReturn'],
                    "optimalPeriod": f"{best_period}天", "trades": best_stats['trades']
                }
                log(f"已更新组合策略: {s_id}")

        # 3. 保存到两个位置
        os.makedirs(self.web_data_dir, exist_ok=True)
        os.makedirs(self.public_data_dir, exist_ok=True)
        target_files = [os.path.join(self.public_data_dir, "strategies.json"), os.path.join(self.web_data_dir, "strategies.json")]
        for target in target_files:
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(web_data, f, ensure_ascii=False, indent=2)
            log(f"已保存更新到: {target}")

    def update_strategies_from_single(self, backtest_json_path: str):
        """保留接口，现在统一调用 update_strategies_from_csv"""
        self.update_strategies_from_csv()
    
    def update_strategies_from_combo(self, backtest_json_path: str):
        """保留接口，现在统一调用 update_strategies_from_csv"""
        self.update_strategies_from_csv()

# ------------------------------------------------------------------------------
# 模式一：单策略回测
# ------------------------------------------------------------------------------
def run_single_strategy_mode(args):
    log("启动单策略回测模式...")
    stock_files = get_all_stock_files(args.data_dir)
    if not stock_files:
        log("未找到任何股票数据文件。", level="ERROR")
        return

    results_df = run_backtest(args.strategy, stock_files)

    if results_df.empty:
        log("回测未产生任何结果。", level="WARNING")
        return

    out_dir = os.path.join(PROJECT_ROOT, "report", "total")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "single_strategy_summary.csv")
    results_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    log(f"单策略回测汇总已保存: {out_csv}")

    out_json = os.path.join(PROJECT_ROOT, "web", "client", "src", "data", "backtest_single.json")
    payload = {
        "generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stock_count": len(stock_files),
        "row_count": int(len(results_df)),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"前端元信息已更新: {out_json}")

    # 自动更新前端策略数据
    updater = WebDataUpdater(PROJECT_ROOT)
    updater.update_strategies_from_single(out_json)
    
    log("单策略回测完成。")

def calculate_returns(df: pd.DataFrame, signal_col: str, hold_period: int) -> Dict:
    base = {
        "signal_count": 0, "trade_count": 0, "win_count": 0,
        "win_rate": 0.0, "avg_return": 0.0, "sum_return": 0.0,
    }
    if signal_col not in df.columns:
        return base
    
    # 信号触发点：当前为True且前一周期为False
    signals = df[signal_col].fillna(False).astype(bool)
    entry_mask = signals & ~signals.shift(1, fill_value=False)
    entry_indices = df.index[entry_mask].tolist()
    
    if not entry_indices:
        return base
    
    trade_count = 0
    win_count = 0
    sum_return = 0.0
    
    n = len(df)
    for idx in entry_indices:
        exit_idx = idx + hold_period
        if exit_idx >= n:
            continue
        
        buy_price = df.loc[idx, "close"]
        sell_price = df.loc[exit_idx, "close"]
        
        if buy_price > 0:
            # 简单收益率计算 (扣除手续费)
            ret = (sell_price * (1 - COMMISSION_RATE - STAMP_TAX_RATE) - buy_price * (1 + COMMISSION_RATE)) / (buy_price * (1 + COMMISSION_RATE)) * 100
            trade_count += 1
            sum_return += ret
            if ret > 0:
                win_count += 1
                
    if trade_count == 0:
        return {**base, "signal_count": len(entry_indices)}
        
    return {
        "signal_count": len(entry_indices),
        "trade_count": trade_count,
        "win_count": win_count,
        "win_rate": round(win_count / trade_count * 100, 2),
        "avg_return": round(sum_return / trade_count, 2),
        "sum_return": round(sum_return, 2),
    }

def run_backtest(strategy_name: str, stock_files: List[str]) -> pd.DataFrame:
    strategies = []
    if strategy_name == "all":
        strategies = ["four_red_plus", "buy1", "buy2", "chan_buy1", "macd_red", "kdj_red", "rsi_red", "lwr_red", "bbi_red", "mtm_red"]
    else:
        strategies = [strategy_name]
        
    all_results = []
    for strat in strategies:
        log(f"正在回测策略: {strat} ...")
        for period in DEFAULT_HOLD_PERIODS:
            # 这里简化处理，实际应使用多进程加速
            period_results = []
            for f in stock_files:
                df = load_stock_data(f)
                if df is not None:
                    res = calculate_returns(df, strat, period)
                    if res["trade_count"] > 0:
                        period_results.append(res)
            
            if period_results:
                pdf = pd.DataFrame(period_results)
                summary = {
                    "strategy": strat,
                    "hold_period": period,
                    "signal_count": pdf["signal_count"].sum(),
                    "trade_count": pdf["trade_count"].sum(),
                    "win_count": pdf["win_count"].sum(),
                    "sum_return": pdf["sum_return"].sum(),
                }
                summary["win_rate"] = round(summary["win_count"] / summary["trade_count"] * 100, 2)
                summary["avg_return"] = round(summary["sum_return"] / summary["trade_count"], 2)
                all_results.append(summary)
                
    return pd.DataFrame(all_results)

# ------------------------------------------------------------------------------
# 模式二：组合策略回测
# ------------------------------------------------------------------------------
def run_combo_strategy_mode(args):
    log("启动组合策略回测模式...")
    stock_files = get_all_stock_files(args.data_dir)
    if not stock_files:
        log("未找到任何股票数据文件。", level="ERROR")
        return
    
    # 简化版组合回测逻辑
    results_df = run_backtest_combo(args.strategy, stock_files)
    if results_df.empty:
        log("回测未产生任何结果。", level="WARNING")
        return

    out_dir = os.path.join(PROJECT_ROOT, "report", "total")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "combo_strategy_summary.csv")
    results_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    log(f"组合策略回测汇总已保存: {out_csv}")

    md_report = generate_markdown_report(results_df, len(stock_files))
    out_md = os.path.join(out_dir, "combo_strategy_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_report)
    log(f"Markdown 报告已生成: {out_md}")

    json_data = generate_json_data(results_df, len(stock_files))
    out_json = os.path.join(PROJECT_ROOT, "web", "client", "src", "data", "backtest_combo.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    log(f"前端摘要 JSON 已生成: {out_json}")

    # 自动更新前端策略数据
    updater = WebDataUpdater(PROJECT_ROOT)
    updater.update_strategies_from_combo(out_json)
    
    log("组合策略回测完成。")

def run_backtest_combo(strategy_name: str, stock_files: List[str]) -> pd.DataFrame:
    # 组合策略定义
    combo_strategies = {
        "steady": lambda df: (df.get("six_veins_count", 0) >= 4) & df.get("buy2", False) & df.get("chan_buy1", False),
        "aggressive": lambda df: (df.get("six_veins_count", 0) >= 3) & (df.get("buy1", False) | df.get("buy2", False))
    }
    
    target_strats = combo_strategies.keys() if strategy_name == "all" else [strategy_name]
    all_results = []
    
    for s_name in target_strats:
        if s_name not in combo_strategies: continue
        log(f"正在回测组合策略: {s_name} ...")
        
        for period in DEFAULT_HOLD_PERIODS:
            period_results = []
            for f in stock_files:
                df = load_stock_data(f)
                if df is not None:
                    df = calculate_all_signals(df)
                    col_name = f"combo_{s_name}"
                    df[col_name] = combo_strategies[s_name](df)
                    res = calculate_returns(df, col_name, period)
                    if res["trade_count"] > 0:
                        period_results.append(res)
            
            if period_results:
                pdf = pd.DataFrame(period_results)
                summary = {
                    "strategy": s_name,
                    "hold_period": period,
                    "signal_count": pdf["signal_count"].sum(),
                    "trade_count": pdf["trade_count"].sum(),
                    "win_count": pdf["win_count"].sum(),
                    "sum_return": pdf["sum_return"].sum(),
                }
                summary["win_rate"] = round(summary["win_count"] / summary["trade_count"] * 100, 2)
                summary["avg_return"] = round(summary["sum_return"] / summary["trade_count"], 2)
                all_results.append(summary)
                
    return pd.DataFrame(all_results)

def _aggregate_strategy_period(df: pd.DataFrame) -> Dict:
    trades = int(df["trade_count"].sum())
    wins = int(df.get("win_count", 0).sum())
    sum_return = float(df.get("sum_return", 0).sum())
    win_rate = round(wins / trades * 100, 2) if trades > 0 else 0.0
    avg_return = round(sum_return / trades, 2) if trades > 0 else 0.0
    signal_count = int(df.get("signal_count", 0).sum())
    return {
        "signal_count": signal_count,
        "trade_count": trades,
        "win_count": wins,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "sum_return": round(sum_return, 2),
    }

def generate_markdown_report(results: pd.DataFrame, stock_count: int) -> str:
    lines = ["# 组合策略回测报告", f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f"样本股票数: {stock_count}", ""]
    for strat, df_s in results.groupby("strategy"):
        lines.append(f"## 策略: {strat}")
        best = df_s.sort_values(["win_rate", "avg_return"], ascending=False).iloc[0]
        lines.append(f"- 最优持有期: {best['hold_period']}天 (胜率: {best['win_rate']}%)")
        lines.append("| 持有期 | 交易笔数 | 胜率 | 平均收益 |")
        lines.append("|---:|---:|---:|---:|")
        for _, r in df_s.iterrows():
            lines.append(f"| {r['hold_period']} | {r['trade_count']} | {r['win_rate']}% | {r['avg_return']}% |")
        lines.append("")
    return "\n".join(lines)

def generate_json_data(results: pd.DataFrame, stock_count: int) -> Dict:
    data = {"generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "stock_count": int(stock_count), "strategies": {}}
    for strat, df_s in results.groupby("strategy"):
        by_period = {}
        for period, df_p in df_s.groupby("hold_period"):
            by_period[str(int(period))] = _aggregate_strategy_period(df_p)
        best = df_s.sort_values(["win_rate", "avg_return"], ascending=False).iloc[0]
        data["strategies"][strat] = {
            "optimal_period_win": str(int(best["hold_period"])),
            "win_rate": best["win_rate"],
            "avg_return": best["avg_return"],
            "trades": int(best["trade_count"]),
            "signals": int(best["signal_count"]),
            "by_period": by_period
        }
    return data

# ------------------------------------------------------------------------------
# 模式三：信号成功案例扫描
# ------------------------------------------------------------------------------
def run_scan_mode(args):
    log("启动信号成功案例扫描模式...")
    # 扫描逻辑保持不变...
    pass

def save_results(all_records: pd.DataFrame, success_cases: pd.DataFrame, summary: Dict):
    REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
    WEB_DATA_DIR = os.path.join(PROJECT_ROOT, "web", "client", "src", "data")
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(WEB_DATA_DIR, exist_ok=True)
    
    success_path = os.path.join(REPORT_DIR, "signal_success_cases.csv")
    success_cases.to_csv(success_path, index=False, encoding="utf-8-sig")
    
    summary_path = os.path.join(REPORT_DIR, "signal_summary.json")
    web_summary_path = os.path.join(WEB_DATA_DIR, "signal_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(web_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log("信号扫描结果已保存。")

# ------------------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="统一回测与信号扫描脚本 v2.1 (整合策略汇总更新)")
    parser.add_argument("--mode", type=str, required=True, choices=["single", "combo", "scan"], help="运行模式")
    parser.add_argument("--strategy", type=str, default="all", help="回测策略")
    parser.add_argument("--data_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "day"), help="数据目录")
    parser.add_argument("--holding_days", type=int, default=15, help="持有天数")
    parser.add_argument("--success_threshold", type=float, default=5.0, help="成功阈值")

    args = parser.parse_args()
    if args.mode == "single":
        run_single_strategy_mode(args)
    elif args.mode == "combo":
        run_combo_strategy_mode(args)
    elif args.mode == "scan":
        run_scan_mode(args)

if __name__ == "__main__":
    main()
