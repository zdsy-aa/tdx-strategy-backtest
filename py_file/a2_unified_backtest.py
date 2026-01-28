#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a2_unified_backtest.py (完整整合版)
================================================================================

【脚本功能】
    统一回测与信号扫描引擎，整合了以下三个模块的完整功能：
    1. 单策略回测 (a2_single_strategy_backtest.py)：验证单个技术指标的胜率
    2. 组合策略回测 (a3_combo_strategy_backtest.py)：验证多指标共振的表现
    3. 信号成功案例扫描 (a4_signal_success_scanner.py)：扫描全市场当前符合信号的股票

【生成的前端JSON文件】
    - strategies.json          (策略配置与统计，自动更新)
    - backtest_single.json     (单指标回测元信息)
    - backtest_combo.json      (组合策略回测元信息)
    - signal_summary.json      (信号扫描汇总统计)

【使用方法】
    # 运行所有模式 (默认)
    python3 a2_unified_backtest.py
    
    # 单独运行某个模式
    python3 a2_unified_backtest.py --mode single
    python3 a2_unified_backtest.py --mode combo
    python3 a2_unified_backtest.py --mode scan
    
    # 自定义参数
    python3 a2_unified_backtest.py --mode scan --holding_days 15 --success_threshold 5.0

【输出文件】
    CSV 汇总:
    - report/total/single_strategy_summary.csv  (单指标汇总)
    - report/total/combo_strategy_summary.csv   (组合策略汇总)
    - report/total/combo_strategy_report.md     (组合策略Markdown报告)
    - report/all_signal_records*.csv            (所有信号记录，可能被切片)
    - report/signal_success_cases.csv           (成功案例记录)
    
    JSON 数据:
    - web/client/src/data/strategies.json       (前端策略数据)
    - web/client/src/data/backtest_single.json
    - web/client/src/data/backtest_combo.json
    - web/client/src/data/signal_summary.json

【设计优势】
    - 高性能：利用多进程并行计算
    - 完整整合：三大模块的所有核心功能
    - 自动化：回测完成后自动更新前端数据
================================================================================
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

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

# 路径配置
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')
REPORT_TOTAL_DIR = os.path.join(REPORT_DIR, 'total')
WEB_DATA_DIR = os.path.join(PROJECT_ROOT, 'web', 'client', 'src', 'data')
PUBLIC_DATA_DIR = os.path.join(PROJECT_ROOT, 'web', 'client', 'public', 'data')

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
    """加载并标准化单只股票CSV数据"""
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

def _extract_stock_meta(filepath: str, df: pd.DataFrame) -> Dict:
    """提取股票代码和名称"""
    stock_code = os.path.basename(filepath).replace('.csv', '')
    name = ''
    try:
        if 'name' in df.columns and len(df) > 0:
            name = str(df['name'].iloc[-1])
    except Exception:
        name = ''
    return {'stock_code': stock_code, 'name': name}

# ------------------------------------------------------------------------------
# 回测统计
# ------------------------------------------------------------------------------
def calculate_returns(df: pd.DataFrame, signal_col: str, hold_period: int) -> Dict:
    """固定持有回测统计"""
    base = {
        'signal_count': 0,
        'trade_count': 0,
        'win_count': 0,
        'win_rate': 0.0,
        'avg_return': 0.0,
        'sum_return': 0.0,
    }

    if df is None or df.empty or signal_col not in df.columns:
        return base

    sig = df[signal_col].fillna(False).astype(bool)
    base['signal_count'] = int(sig.sum())

    trades = backtest_trades_fixed_hold(
        df=df,
        signal_col=signal_col,
        hold_period=hold_period,
        commission_rate=COMMISSION_RATE,
        stamp_tax_rate=STAMP_TAX_RATE,
    )
    stats = summarize_trades(trades)

    for k, v in stats.items():
        base[k] = v
    return base

# ==============================================================================
# 模块 1: 单指标回测 (来自 a2_single_strategy_backtest.py)
# ==============================================================================

def backtest_six_veins_single(filepath: str) -> Optional[pd.DataFrame]:
    """六脉神剑策略回测：单项红柱转红触发 + >=4红共振触发"""
    df = load_stock_data(filepath)
    if df is None:
        return None

    meta = _extract_stock_meta(filepath, df)
    df = calculate_six_veins(df)

    results = []

    # 1) 单个红柱"转红"触发
    for indicator in SIX_VEINS_INDICATORS:
        if indicator not in df.columns:
            continue
        sig_col = f'{indicator}_sig'
        cur = df[indicator].fillna(False).astype(bool)
        prev = cur.shift(1, fill_value=False)
        df[sig_col] = cur & (~prev)

        for period in DEFAULT_HOLD_PERIODS:
            stats = calculate_returns(df, sig_col, period)
            stats.update(meta)
            stats.update({'strategy': indicator, 'hold_period': period, 'strategy_type': 'six_veins'})
            results.append(stats)

    # 2) >=4红共振跨越触发
    cur_cnt = pd.to_numeric(df.get('six_veins_count', 0), errors='coerce').fillna(0).astype(int)
    df['four_red_sig'] = (cur_cnt >= 4) & (cur_cnt.shift(1, fill_value=0) < 4)

    for period in DEFAULT_HOLD_PERIODS:
        stats = calculate_returns(df, 'four_red_sig', period)
        stats.update(meta)
        stats.update({'strategy': 'four_red_plus', 'hold_period': period, 'strategy_type': 'six_veins'})
        results.append(stats)

    return pd.DataFrame(results) if results else None

def backtest_buy_sell_single(filepath: str) -> Optional[pd.DataFrame]:
    """买卖点策略回测：buy1 / buy2"""
    df = load_stock_data(filepath)
    if df is None:
        return None

    meta = _extract_stock_meta(filepath, df)
    df = calculate_buy_sell_points(df)

    results = []
    for signal in ['buy1', 'buy2']:
        if signal not in df.columns:
            continue

        sig_col = f'{signal}_sig'
        cur = df[signal].fillna(False).astype(bool)
        prev = cur.shift(1, fill_value=False)
        df[sig_col] = cur & (~prev)

        for period in DEFAULT_HOLD_PERIODS:
            stats = calculate_returns(df, sig_col, period)
            stats.update(meta)
            stats.update({'strategy': signal, 'hold_period': period, 'strategy_type': 'buy_sell'})
            results.append(stats)

    return pd.DataFrame(results) if results else None

def backtest_chan_single(filepath: str) -> Optional[pd.DataFrame]:
    """缠论策略回测"""
    df = load_stock_data(filepath)
    if df is None:
        return None

    meta = _extract_stock_meta(filepath, df)
    df = calculate_chan_theory(df)

    results = []
    for signal in ['chan_buy1']:
        if signal not in df.columns:
            continue

        sig_col = f'{signal}_sig'
        cur = df[signal].fillna(False).astype(bool)
        prev = cur.shift(1, fill_value=False)
        df[sig_col] = cur & (~prev)

        for period in DEFAULT_HOLD_PERIODS:
            stats = calculate_returns(df, sig_col, period)
            stats.update(meta)
            stats.update({'strategy': signal, 'hold_period': period, 'strategy_type': 'chan'})
            results.append(stats)

    return pd.DataFrame(results) if results else None

def run_single_strategy_backtest(stock_files: List[str]) -> pd.DataFrame:
    """运行单指标回测"""
    funcs = {
        'six_veins': backtest_six_veins_single,
        'buy_sell': backtest_buy_sell_single,
        'chan': backtest_chan_single,
    }

    all_results = []
    for s_name, func in funcs.items():
        log(f"开始回测单指标策略: {s_name}")
        res_list = run_backtest_on_all_stocks(stock_files, func)
        if res_list:
            df_res = pd.concat(res_list, ignore_index=True)
            all_results.append(df_res)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# ==============================================================================
# 模块 2: 组合策略回测 (来自 a3_combo_strategy_backtest.py)
# ==============================================================================

def backtest_steady_single(filepath: str) -> Optional[pd.DataFrame]:
    """稳健组合: 六脉>=4红 + 买点2 + 缠论买点"""
    df = load_stock_data(filepath)
    if df is None:
        return None
    meta = _extract_stock_meta(filepath, df)

    df = calculate_all_signals(df)
    if df is None or df.empty:
        return None

    chan_any_buy = df.get('chan_buy1', False)
    df['combo_steady'] = (df.get('six_veins_count', 0) >= 4) & df.get('buy2', False) & chan_any_buy

    # 触发点：False -> True
    df['steady_sig'] = df['combo_steady'] & ~df['combo_steady'].shift(1, fill_value=False)
    if int(df['steady_sig'].sum()) == 0:
        return None

    rows = []
    for period in DEFAULT_HOLD_PERIODS:
        stats = calculate_returns(df, 'steady_sig', period)
        stats.update(meta)
        stats.update({'strategy': 'steady', 'hold_period': period})
        rows.append(stats)
    return pd.DataFrame(rows)

def backtest_aggressive_single(filepath: str) -> Optional[pd.DataFrame]:
    """激进组合: aggr1/aggr2/aggr3 的任意满足"""
    df = load_stock_data(filepath)
    if df is None:
        return None
    meta = _extract_stock_meta(filepath, df)

    df = calculate_all_signals(df)
    if df is None or df.empty:
        return None

    chan_any_buy = df.get('chan_buy1', False)

    df['aggr1'] = (df.get('six_veins_count', 0) >= 5) & df.get('buy2', False)
    df['aggr2'] = (df.get('six_veins_count', 0) == 6) & df.get('money_tree_signal', False)
    df['aggr3'] = (df.get('six_veins_count', 0) == 6) & chan_any_buy

    comb = (df['aggr1'] | df['aggr2'] | df['aggr3']).fillna(False)
    df['aggressive_sig'] = comb & ~comb.shift(1, fill_value=False)

    if int(df['aggressive_sig'].sum()) == 0:
        return None

    rows = []
    for period in DEFAULT_HOLD_PERIODS:
        stats = calculate_returns(df, 'aggressive_sig', period)
        stats.update(meta)
        stats.update({'strategy': 'aggressive', 'hold_period': period})
        rows.append(stats)
    return pd.DataFrame(rows)

def run_combo_strategy_backtest(stock_files: List[str]) -> pd.DataFrame:
    """运行组合策略回测"""
    funcs = {'steady': backtest_steady_single, 'aggressive': backtest_aggressive_single}

    all_dfs = []
    for strat, func in funcs.items():
        log(f"开始回测组合策略: {strat}")
        res_list = run_backtest_on_all_stocks(stock_files, func)
        if res_list:
            all_dfs.append(pd.concat(res_list, ignore_index=True))
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def _aggregate_strategy_period(df: pd.DataFrame) -> Dict:
    """按 trade_count 聚合统计"""
    trades = int(df['trade_count'].sum())
    wins = int(df.get('win_count', 0).sum())
    sum_return = float(df.get('sum_return', 0).sum())
    win_rate = round(wins / trades * 100, 2) if trades > 0 else 0.0
    avg_return = round(sum_return / trades, 2) if trades > 0 else 0.0

    signal_count = int(df.get('signal_count', 0).sum())
    return {
        'signal_count': signal_count,
        'trade_count': trades,
        'win_count': wins,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'sum_return': round(sum_return, 2),
    }

def generate_combo_markdown_report(results: pd.DataFrame, stock_count: int) -> str:
    """生成组合策略 Markdown 报告"""
    lines: List[str] = []
    lines.append("# 组合策略回测报告")
    lines.append("")
    lines.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 扫描股票数: {stock_count}")
    lines.append(f"- 结果行数: {len(results)}")
    lines.append("")

    for strat, df_s in results.groupby('strategy'):
        lines.append(f"## 策略: {strat}")
        lines.append("")
        period_rows = []
        for period, df_p in df_s.groupby('hold_period'):
            agg = _aggregate_strategy_period(df_p)
            agg['hold_period'] = int(period)
            period_rows.append(agg)

        period_rows.sort(key=lambda x: (x['win_rate'], x['avg_return']), reverse=True)
        if not period_rows:
            lines.append("- 无有效交易。")
            lines.append("")
            continue

        best = period_rows[0]
        lines.append(f"- 最优持有期(按胜率优先): **{best['hold_period']} 天**")
        lines.append(f"- 该持有期胜率: **{best['win_rate']:.2f}%**")
        lines.append(f"- 该持有期平均单笔收益: **{best['avg_return']:.2f}%**")
        lines.append(f"- 该持有期交易笔数: {best['trade_count']}")
        lines.append("")

        lines.append("| 持有期 | 信号次数 | 交易笔数 | 盈利笔数 | 胜率(%) | 平均单笔收益(%) | 累计收益(%) |")
        lines.append("|--------|----------|----------|----------|---------|----------------|------------|")
        for row in period_rows:
            lines.append(f"| {row['hold_period']} | {row['signal_count']} | {row['trade_count']} | "
                        f"{row['win_count']} | {row['win_rate']} | {row['avg_return']} | {row['sum_return']} |")
        lines.append("")

    return "\n".join(lines)

# ==============================================================================
# 模块 3: 信号扫描 (来自 a4_signal_success_scanner.py)
# ==============================================================================

def _calc_future_return(buy_price: float, sell_price: float) -> Optional[float]:
    """计算未来收益率（含手续费）"""
    if not np.isfinite(buy_price) or not np.isfinite(sell_price):
        return None
    if buy_price <= 0:
        return None

    buy_cost = buy_price * (1.0 + COMMISSION_RATE)
    sell_net = sell_price * (1.0 - COMMISSION_RATE - STAMP_TAX_RATE)
    if buy_cost <= 0:
        return None
    return (sell_net - buy_cost) / buy_cost * 100.0

def scan_single_stock(file_path: str, holding_days: int, min_red: int, success_threshold: float) -> pd.DataFrame:
    """扫描单只股票的信号记录"""
    df = load_stock_data(file_path)
    if df is None or df.empty:
        return pd.DataFrame()

    stock_code = os.path.basename(file_path).replace('.csv', '')
    stock_name = str(df['name'].iloc[-1]) if 'name' in df.columns and len(df) > 0 else ''

    df = calculate_all_signals(df)
    if df is None or df.empty:
        return pd.DataFrame()

    # 触发点定义
    six_cnt = pd.to_numeric(df.get('six_veins_count', 0), errors='coerce').fillna(0).astype(int)
    sig_six = (six_cnt >= int(min_red)) & (six_cnt.shift(1, fill_value=0) < int(min_red))

    chan = df.get('chan_buy1', False)
    sig_chan = chan.fillna(False).astype(bool) & ~chan.fillna(False).astype(bool).shift(1, fill_value=False)

    buy1 = df.get('buy1', False)
    buy2 = df.get('buy2', False)
    sig_buy1 = buy1.fillna(False).astype(bool) & ~buy1.fillna(False).astype(bool).shift(1, fill_value=False)
    sig_buy2 = buy2.fillna(False).astype(bool) & ~buy2.fillna(False).astype(bool).shift(1, fill_value=False)

    records: List[Dict] = []
    n = len(df)
    for i in range(n):
        j = i + int(holding_days)
        if j >= n:
            continue

        has_six = bool(sig_six.iat[i])
        has_chan = bool(sig_chan.iat[i])
        has_b1 = bool(sig_buy1.iat[i])
        has_b2 = bool(sig_buy2.iat[i])

        if not (has_six or has_chan or has_b1 or has_b2):
            continue

        buy_price = float(df.at[i, 'close'])
        sell_price = float(df.at[j, 'close'])
        fut_ret = _calc_future_return(buy_price, sell_price)
        if fut_ret is None:
            continue

        signal_types = []
        if has_six: signal_types.append('six_veins')
        if has_chan: signal_types.append('chan_buy')
        if has_b1 or has_b2: signal_types.append('buy_sell')

        rec = {
            'date': pd.Timestamp(df.at[i, 'date']).strftime('%Y/%m/%d'),
            'stock_code': stock_code,
            'name': stock_name,
            'holding_days': int(holding_days),
            'six_veins_count': int(six_cnt.iat[i]),
            'buy1': int(has_b1),
            'buy2': int(has_b2),
            'chan_any_buy': int(has_chan),
            'future_return': round(float(fut_ret), 4),
            'signal_type': '+'.join(signal_types),
            'is_success': int(fut_ret > float(success_threshold)),
        }

        # 六脉的6个红色指标列
        for col in ['macd_red', 'kdj_red', 'rsi_red', 'lwr_red', 'bbi_red', 'mtm_red']:
            rec[col] = int(bool(df.at[i, col])) if col in df.columns else 0

        records.append(rec)

    return pd.DataFrame(records)

def run_signal_scan(stock_files: List[str], holding_days: int = 15, min_red: int = 4, 
                    success_threshold: float = 5.0) -> pd.DataFrame:
    """运行信号扫描（多进程）"""
    log(f"开始信号扫描: 持有{holding_days}天, 六脉>={min_red}红, 成功阈值>{success_threshold}%")
    
    from functools import partial
    scan_func = partial(scan_single_stock, 
                       holding_days=holding_days, 
                       min_red=min_red, 
                       success_threshold=success_threshold)
    
    all_records = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {executor.submit(scan_func, f): f for f in stock_files}
        for future in as_completed(futures):
            try:
                result = future.result()
                if not result.empty:
                    all_records.append(result)
            except Exception as e:
                log(f"扫描失败: {e}", level="ERROR")
    
    if not all_records:
        return pd.DataFrame()
    
    return pd.concat(all_records, ignore_index=True)

def aggregate_scan_results(all_records: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, Dict):
    """聚合扫描结果"""
    if all_records is None or all_records.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_signals': 0,
            'success_signals': 0,
            'success_rate': 0.0,
            'signal_type_counts': {},
            'six_veins_success_by_count': {}
        }

    all_records['_date_dt'] = pd.to_datetime(all_records['date'], format='%Y/%m/%d', errors='coerce')
    all_records.sort_values(['_date_dt', 'stock_code'], inplace=True)
    all_records.drop(columns=['_date_dt'], inplace=True)
    all_records.reset_index(drop=True, inplace=True)

    success_cases = all_records[all_records['is_success'] == 1].copy()
    success_cases.reset_index(drop=True, inplace=True)

    summary = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_signals': int(len(all_records)),
        'success_signals': int(all_records['is_success'].sum()),
        'success_rate': round(float(all_records['is_success'].mean() * 100), 2) if len(all_records) > 0 else 0.0,
        'signal_type_counts': all_records['signal_type'].value_counts().to_dict(),
        'six_veins_success_by_count': success_cases['six_veins_count'].value_counts().to_dict() if not success_cases.empty else {},
    }
    return all_records, success_cases, summary

# ==============================================================================
# 前端数据更新模块
# ==============================================================================

class WebDataUpdater:
    """前端数据自动更新器"""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.web_data_dir = WEB_DATA_DIR
        self.public_data_dir = PUBLIC_DATA_DIR
        
        # 优先使用 public 目录
        self.strategies_file = os.path.join(self.public_data_dir, "strategies.json")
        if not os.path.exists(self.strategies_file):
            self.strategies_file = os.path.join(self.web_data_dir, "strategies.json")
    
    def update_strategies_from_csv(self):
        """从 CSV 汇总文件更新 strategies.json"""
        single_csv = os.path.join(REPORT_TOTAL_DIR, "single_strategy_summary.csv")
        combo_csv = os.path.join(REPORT_TOTAL_DIR, "combo_strategy_summary.csv")
        
        if not os.path.exists(self.strategies_file):
            log(f"错误: 找不到 strategies.json 模板文件 {self.strategies_file}", level="ERROR")
            return

        with open(self.strategies_file, 'r', encoding='utf-8') as f:
            web_data = json.load(f)

        # 1. 更新单指标策略
        if os.path.exists(single_csv):
            log(f"正在从 {single_csv} 更新单指标策略...")
            df_single = pd.read_csv(single_csv)
            
            # ID映射
            id_map = {
                "macd_red": "macd_red", "kdj_red": "kdj_red", "rsi_red": "rsi_red",
                "lwr_red": "lwr_red", "bbi_red": "bbi_red", "mtm_red": "mtm_red",
                "four_red_plus": "four_red_plus",
                "buy1": "buy1", "buy2": "buy2",
                "chan_buy1": "chan_buy1",
            }
            
            for strategy in web_data.get('singleIndicatorStrategies', []):
                s_id = strategy['id']
                matched_name = id_map.get(s_id, s_id)
                
                df_strat = df_single[df_single['strategy'] == matched_name]
                if df_strat.empty:
                    continue
                
                # 按持有期汇总
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
                    
                best_period = max(summary_by_period.keys(), 
                                 key=lambda k: float(summary_by_period[k]['winRate'].replace('%', '')))
                best_stats = summary_by_period[best_period]
                strategy['stats']['total'] = {
                    "winRate": best_stats['winRate'],
                    "avgReturn": best_stats['avgReturn'],
                    "optimalPeriod": f"{best_period}天",
                    "trades": best_stats['trades']
                }
                log(f"已更新单指标策略: {s_id}")

        # 2. 更新组合策略
        if os.path.exists(combo_csv):
            log(f"正在从 {combo_csv} 更新组合策略...")
            df_combo = pd.read_csv(combo_csv)
            
            combo_id_map = {"steady": "steady", "aggressive": "aggressive"}
            
            for strategy in web_data.get('comboStrategies', []):
                s_id = strategy['id']
                matched_name = combo_id_map.get(s_id, s_id)
                
                df_strat = df_combo[df_combo['strategy'] == matched_name]
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
                    
                best_period = max(summary_by_period.keys(),
                                 key=lambda k: float(summary_by_period[k]['winRate'].replace('%', '')))
                best_stats = summary_by_period[best_period]
                strategy['stats']['total'] = {
                    "winRate": best_stats['winRate'],
                    "avgReturn": best_stats['avgReturn'],
                    "optimalPeriod": f"{best_period}天",
                    "trades": best_stats['trades']
                }
                log(f"已更新组合策略: {s_id}")

        # 3. 保存到两个位置
        os.makedirs(self.web_data_dir, exist_ok=True)
        os.makedirs(self.public_data_dir, exist_ok=True)
        
        target_files = [
            os.path.join(self.public_data_dir, "strategies.json"),
            os.path.join(self.web_data_dir, "strategies.json")
        ]
        
        for target in target_files:
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(web_data, f, ensure_ascii=False, indent=2)
            log(f"已保存更新到: {target}")

# ==============================================================================
# 主流程控制
# ==============================================================================

def run_backtest_main(mode: str = "all", holding_days: int = 15, 
                      success_threshold: float = 5.0, min_red: int = 4):
    """主运行函数"""
    log(f"=" * 80)
    log(f"开始运行统一回测引擎，模式: {mode}")
    log(f"=" * 80)
    
    # 初始化
    updater = WebDataUpdater(PROJECT_ROOT)
    stock_files = get_all_stock_files(DATA_DIR)
    
    if not stock_files:
        log("错误: 未找到股票数据文件", level="ERROR")
        return
    
    log(f"找到 {len(stock_files)} 只股票数据")
    
    # 创建输出目录
    os.makedirs(REPORT_TOTAL_DIR, exist_ok=True)
    os.makedirs(WEB_DATA_DIR, exist_ok=True)
    
    # 模式 1: 单指标回测
    if mode in ["single", "all"]:
        log("\n" + "=" * 80)
        log("开始单指标回测...")
        log("=" * 80)
        
        results_df = run_single_strategy_backtest(stock_files)
        
        if not results_df.empty:
            # 保存CSV
            out_csv = os.path.join(REPORT_TOTAL_DIR, 'single_strategy_summary.csv')
            results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
            log(f"单指标回测汇总已保存: {out_csv}")
            
            # 保存前端JSON
            out_json = os.path.join(WEB_DATA_DIR, 'backtest_single.json')
            payload = {
                'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'stock_count': len(stock_files),
                'row_count': int(len(results_df)),
            }
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            log(f"前端元信息已更新: {out_json}")
        else:
            log("单指标回测无有效结果", level="WARNING")
    
    # 模式 2: 组合策略回测
    if mode in ["combo", "all"]:
        log("\n" + "=" * 80)
        log("开始组合策略回测...")
        log("=" * 80)
        
        results_df = run_combo_strategy_backtest(stock_files)
        
        if not results_df.empty:
            # 保存CSV
            out_csv = os.path.join(REPORT_TOTAL_DIR, 'combo_strategy_summary.csv')
            results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
            log(f"组合策略回测汇总已保存: {out_csv}")
            
            # 生成Markdown报告
            md_report = generate_combo_markdown_report(results_df, len(stock_files))
            out_md = os.path.join(REPORT_TOTAL_DIR, 'combo_strategy_report.md')
            with open(out_md, 'w', encoding='utf-8') as f:
                f.write(md_report)
            log(f"组合策略Markdown报告已保存: {out_md}")
            
            # 保存前端JSON
            out_json = os.path.join(WEB_DATA_DIR, 'backtest_combo.json')
            payload = {
                'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'stock_count': len(stock_files),
                'row_count': int(len(results_df)),
            }
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            log(f"前端元信息已更新: {out_json}")
        else:
            log("组合策略回测无有效结果", level="WARNING")
    
    # 模式 3: 信号扫描
    if mode in ["scan", "all"]:
        log("\n" + "=" * 80)
        log("开始信号扫描...")
        log("=" * 80)
        
        all_records = run_signal_scan(stock_files, holding_days, min_red, success_threshold)
        
        if not all_records.empty:
            all_records, success_cases, summary = aggregate_scan_results(all_records)
            
            # 保存CSV（带切片功能）
            if split_csv_by_size and cleanup_old_chunks:
                cleanup_old_chunks(REPORT_DIR, 'all_signal_records')
                temp_path = os.path.join(REPORT_DIR, 'all_signal_records_temp.csv')
                all_records.to_csv(temp_path, index=False, encoding='utf-8-sig')
                chunk_files = split_csv_by_size(temp_path, REPORT_DIR, 'all_signal_records', max_size_mb=40)
                os.remove(temp_path)
                log(f"all_signal_records 已切片为 {len(chunk_files)} 个文件")
            else:
                all_records_path = os.path.join(REPORT_DIR, 'all_signal_records.csv')
                all_records.to_csv(all_records_path, index=False, encoding='utf-8-sig')
                log(f"所有信号记录已保存: {all_records_path}")
            
            # 保存成功案例
            success_path = os.path.join(REPORT_DIR, 'signal_success_cases.csv')
            success_cases.to_csv(success_path, index=False, encoding='utf-8-sig')
            log(f"成功案例列表已保存: {success_path}")
            
            # 保存JSON汇总
            summary_path = os.path.join(REPORT_DIR, 'signal_summary.json')
            web_summary_path = os.path.join(WEB_DATA_DIR, 'signal_summary.json')
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            with open(web_summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            log(f"统计摘要已保存: {summary_path}")
            log(f"前端统计摘要已更新: {web_summary_path}")
            
            log(f"\n信号扫描汇总:")
            log(f"  总信号数: {summary['total_signals']}")
            log(f"  成功信号数: {summary['success_signals']}")
            log(f"  成功率: {summary['success_rate']}%")
        else:
            log("信号扫描无有效结果", level="WARNING")
    
    # 统一更新 strategies.json
    if mode in ["single", "combo", "all"]:
        log("\n" + "=" * 80)
        log("更新前端 strategies.json...")
        log("=" * 80)
        updater.update_strategies_from_csv()
    
    log("\n" + "=" * 80)
    log("统一回测引擎运行完成！")
    log("=" * 80)

# ==============================================================================
# 命令行入口
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一回测与信号扫描引擎")
    parser.add_argument("--mode", type=str, default="all", 
                       choices=["single", "combo", "scan", "all"],
                       help="运行模式: single(单指标)/combo(组合)/scan(扫描)/all(全部)")
    parser.add_argument("--holding_days", type=int, default=15,
                       help="信号扫描：固定持有天数（默认15）")
    parser.add_argument("--min_red", type=int, default=4,
                       help="信号扫描：六脉红色数量阈值（默认4）")
    parser.add_argument("--success_threshold", type=float, default=5.0,
                       help="信号扫描：成功阈值（净收益率%，默认5.0）")
    
    args = parser.parse_args()
    
    run_backtest_main(
        mode=args.mode,
        holding_days=args.holding_days,
        success_threshold=args.success_threshold,
        min_red=args.min_red
    )