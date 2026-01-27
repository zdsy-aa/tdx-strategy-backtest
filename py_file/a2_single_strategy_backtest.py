\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a2_single_strategy_backtest.py
================================================================================

【脚本功能】
    单指标策略回测系统：扫描 data/day 目录下的所有股票 CSV 文件，计算买入信号，
    并按“固定持有 N 天”口径进行回测统计，最终输出汇总 CSV 与前端元信息 JSON。

【数据输入要求 - 与你的CSV字段严格匹配】
    CSV 字段（至少包含以下列；无数据的列允许为空）：
        名称、日期(yyyy/mm/dd)、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌幅、涨跌额、换手率

    说明：
    - 指标/回测模块需要英文列：date/open/high/low/close/volume（本脚本会自动映射并清洗）
    - 日期列严格优先按 yyyy/mm/dd 解析；若存在少量脏数据会自动降级通用解析

【输出文件】
    - report/total/single_strategy_summary.csv    单策略回测汇总（UTF-8-SIG）
    - web/client/src/data/backtest_single.json    前端元信息（生成时间、扫描股票数等）

【使用方法】
    进入项目根目录后执行：
        python3 a2_single_strategy_backtest.py --strategy all
    指定数据目录：
        python3 a2_single_strategy_backtest.py --strategy six_veins --data_dir data/day

【重点边界/异常处理（本次修正项）】
    - 路径探测：自动识别项目根目录（避免脚本在根目录时 parent 误判）
    - 字段映射：严格按你给定的CSV字段映射为英文列
    - 日期解析：优先 yyyy/mm/dd，失败再降级
    - 异常值：价格<=0 或缺失会被剔除；成交量/成交额负值置 0
    - 除零：买入价为0/NaN会跳过，避免收益计算异常
================================================================================
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# 日志：优先使用项目内 logger；否则降级打印
# ------------------------------------------------------------------------------
try:
    from a99_logger import log
except Exception:
    def log(msg, level="INFO"):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}")

# ------------------------------------------------------------------------------
# 项目根目录探测（避免脚本位于根目录时 os.path.dirname(current_dir) 误判）
# ------------------------------------------------------------------------------
def find_project_root() -> str:
    """
    从当前脚本路径向上探测项目根目录。
    判据：目录中存在 data/day 目录。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
    for d in candidates:
        if os.path.isdir(os.path.join(d, "data", "day")):
            return d
    return here

PROJECT_ROOT = find_project_root()
sys.path.insert(0, PROJECT_ROOT)

from a99_indicators import (
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

# ------------------------------------------------------------------------------
# 回测参数（与历史口径保持一致）
# ------------------------------------------------------------------------------
COMMISSION_RATE = 0.00008  # 佣金费率（万0.8）
STAMP_TAX_RATE = 0.0005    # 印花税（千0.5）
DEFAULT_HOLD_PERIODS = [5, 10, 20]

# 六脉神剑指标列（由 calculate_six_veins 产出）
SIX_VEINS_INDICATORS = ['macd_red', 'kdj_red', 'rsi_red', 'lwr_red', 'bbi_red', 'mtm_red']

# ------------------------------------------------------------------------------
# CSV 读取与标准化
# ------------------------------------------------------------------------------
CSV_COL_MAP = {
    '名称': 'name',
    '日期': 'date',
    '开盘': 'open',
    '收盘': 'close',
    '最高': 'high',
    '最低': 'low',
    '成交量': 'volume',
    '成交额': 'amount',
    '振幅': 'amplitude',
    '涨跌幅': 'pct_chg',
    '涨跌额': 'chg',
    '换手率': 'turnover',

    # 少量兼容字段（若目录里混入旧文件，也尽量不崩）
    '成交金额': 'amount',
    '涨跌幅(%)': 'pct_chg',
    '涨跌额(元)': 'chg',
}

NUMERIC_COLS = ['open', 'high', 'low', 'close', 'volume', 'amount', 'amplitude', 'pct_chg', 'chg', 'turnover']

def _parse_date_series(s: pd.Series) -> pd.Series:
    """
    统一解析日期列：
    - 优先按 yyyy/mm/dd 解析（与你的 CSV 约定一致）
    - 若失败比例过高，再降级到通用解析
    """
    dt = pd.to_datetime(s, format='%Y/%m/%d', errors='coerce')
    if len(dt) > 0 and dt.isna().mean() > 0.5:
        dt = pd.to_datetime(s, errors='coerce')
    return dt

def load_stock_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    加载单只股票 CSV，并转换为指标模块需要的标准列：
        date/open/high/low/close/volume（必需）

    清洗规则：
    - 支持 utf-8-sig / utf-8 / gbk
    - 日期按 yyyy/mm/dd 优先解析
    - 价格列无法解析或 <=0 的行剔除（避免除零/无意义指标）
    - 成交量/成交额负值置 0
    """
    try:
        df = None
        for enc in ('utf-8-sig', 'utf-8', 'gbk'):
            try:
                df = pd.read_csv(filepath, encoding=enc)
                break
            except Exception:
                continue

        if df is None or df.empty:
            return None

        # 列名映射：中文 -> 英文
        df.rename(columns={c: CSV_COL_MAP.get(c, c) for c in df.columns}, inplace=True)

        if 'date' not in df.columns:
            log(f"缺少日期列，跳过: {filepath}", level="WARNING")
            return None

        # 日期解析 + 排序
        df['date'] = _parse_date_series(df['date'])
        df.dropna(subset=['date'], inplace=True)

        # 数值列统一转 numeric
        for c in NUMERIC_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # 缺失列补齐（volume 可为0；价格为空后面会被 drop）
        for c in ['open', 'high', 'low', 'close', 'volume']:
            if c not in df.columns:
                df[c] = np.nan if c != 'volume' else 0.0

        # 剔除关键价格为空/<=0
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

        # 成交量/成交额异常保护
        for c in ['volume', 'amount']:
            if c in df.columns:
                df[c] = df[c].fillna(0.0)
                df.loc[df[c] < 0, c] = 0.0

        # 股票名称缺失填空
        if 'name' not in df.columns:
            df['name'] = ''

        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 数据量检查：过短样本不回测
        if len(df) < 30:
            return None

        return df
    except Exception as e:
        log(f"数据加载异常: {filepath}, 错误: {e}", level="ERROR")
        return None

# ------------------------------------------------------------------------------
# 回测统计：固定持有 N 天
# ------------------------------------------------------------------------------
def calculate_returns(df: pd.DataFrame, signal_col: str, hold_period: int) -> Dict:
    """
    固定持有回测口径：
    - 在 signal_col 为 True 的K线处“买入”
    - 持有 hold_period 天后“卖出”
    - backtest_trades_fixed_hold 内部会考虑佣金/印花税
    """
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

    # 合并并保证字段齐全
    for k, v in stats.items():
        base[k] = v
    return base

def _extract_stock_meta(filepath: str, df: pd.DataFrame) -> Dict:
    """从文件路径/数据中提取 stock_code、name，统一写入结果。"""
    stock_code = os.path.basename(filepath).replace('.csv', '')
    name = ''
    try:
        if 'name' in df.columns and len(df) > 0:
            # 取最后一行更稳：有些数据会重复写入 name
            name = str(df['name'].iloc[-1])
    except Exception:
        name = ''
    return {'stock_code': stock_code, 'name': name}

# ------------------------------------------------------------------------------
# 策略：单只股票回测
# ------------------------------------------------------------------------------
def backtest_six_veins_single(filepath: str) -> Optional[pd.DataFrame]:
    """六脉神剑策略回测：单项红柱转红触发 + >=4红共振触发。"""
    df = load_stock_data(filepath)
    if df is None:
        return None

    meta = _extract_stock_meta(filepath, df)
    df = calculate_six_veins(df)

    results = []

    # 1) 单个红柱“转红”触发：今日为红 && 昨日不为红
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

    # 2) >=4红共振跨越触发：今日>=4 && 昨日<4
    cur_cnt = pd.to_numeric(df.get('six_veins_count', 0), errors='coerce').fillna(0).astype(int)
    df['four_red_sig'] = (cur_cnt >= 4) & (cur_cnt.shift(1, fill_value=0) < 4)

    for period in DEFAULT_HOLD_PERIODS:
        stats = calculate_returns(df, 'four_red_sig', period)
        stats.update(meta)
        stats.update({'strategy': 'four_red_plus', 'hold_period': period, 'strategy_type': 'six_veins'})
        results.append(stats)

    return pd.DataFrame(results) if results else None

def backtest_buy_sell_single(filepath: str) -> Optional[pd.DataFrame]:
    """买卖点策略回测：buy1 / buy2（触发点：False->True）。"""
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
    """
    缠论策略回测（当前版本仅使用 chan_buy1）。
    说明：a99_indicators.calculate_chan_theory 目前稳定产出 chan_buy1；
         若你未来补齐 chan_buy2/3，可在此扩展信号列表。
    """
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

# ------------------------------------------------------------------------------
# 批量回测控制器
# ------------------------------------------------------------------------------
def run_backtest(strategy: str, stock_files: List[str]) -> pd.DataFrame:
    """
    回测控制器：
    - 对每只股票执行对应 backtest_*_single
    - run_backtest_on_all_stocks 内部可多进程并行
    """
    funcs = {
        'six_veins': backtest_six_veins_single,
        'buy_sell': backtest_buy_sell_single,
        'chan': backtest_chan_single,
    }

    all_results = []
    strats_to_run = list(funcs.keys()) if strategy == 'all' else [strategy]

    for s_name in strats_to_run:
        if s_name not in funcs:
            log(f"无效的策略参数: {s_name}", level="ERROR")
            continue

        log(f"开始回测策略: {s_name}")
        res_list = run_backtest_on_all_stocks(stock_files, funcs[s_name])
        if res_list:
            # run_backtest_on_all_stocks 返回 DataFrame 列表
            df_res = pd.concat(res_list, ignore_index=True)
            all_results.append(df_res)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# ------------------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='单指标策略回测系统')
    parser.add_argument('--strategy', type=str, default='all', help="策略类型: six_veins/buy_sell/chan/all")
    parser.add_argument('--data_dir', type=str, default=os.path.join(PROJECT_ROOT, 'data', 'day'), help='数据目录（默认: data/day）')
    args = parser.parse_args()

    stock_files = get_all_stock_files(args.data_dir)
    if not stock_files:
        log(f"未找到数据文件: {args.data_dir}", level="ERROR")
        return

    results_df = run_backtest(args.strategy, stock_files)
    if results_df.empty:
        log("回测无有效结果。", level="WARNING")
        return

    # 保存结果
    out_dir = os.path.join(PROJECT_ROOT, 'report', 'total')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'single_strategy_summary.csv')
    results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    log(f"单策略回测汇总已保存: {out_csv}")

    # 保存前端元信息（当前仅元信息；如需前端展示详细结果，可扩展写入 results_df）
    json_dir = os.path.join(PROJECT_ROOT, 'web', 'client', 'src', 'data')
    os.makedirs(json_dir, exist_ok=True)
    out_json = os.path.join(json_dir, 'backtest_single.json')
    payload = {
        'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock_count': len(stock_files),
        'row_count': int(len(results_df)),
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"前端元信息已更新: {out_json}")

    log("回测任务完成。")

if __name__ == "__main__":
    main()
