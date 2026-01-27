\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a3_combo_strategy_backtest.py
================================================================================

【脚本功能】
    组合策略回测系统：在单指标信号基础上构造“稳健组合/激进组合”买入条件，
    并按固定持有 N 天口径回测，输出汇总 CSV、Markdown 报告、以及前端 JSON 摘要。

【组合策略定义（保持原脚本语义，不改变“信号口径”，仅修复统计与健壮性）】
    1) 稳健组合 (steady)
        六脉神剑 >=4 红 + 买点2 + 缠论买点(当前以 chan_buy1 代替“任意缠论买点”)

    2) 激进组合 (aggressive)
        aggr1: 六脉 >=5 红 + 买点2
        aggr2: 六脉 ==6 红 + 摇钱树信号 money_tree_signal
        aggr3: 六脉 ==6 红 + 缠论买点(以 chan_buy1 表示)

        综合信号：aggr1 OR aggr2 OR aggr3 的“触发点”(False->True)

【数据输入要求】
    CSV 字段（与你的真实数据一致）：
        名称、日期(yyyy/mm/dd)、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌幅、涨跌额、换手率
    脚本会自动映射为 date/open/high/low/close/volume 等英文列，并进行异常清洗。

【输出文件】
    - report/total/combo_strategy_summary.csv      组合策略回测汇总（UTF-8-SIG）
    - report/total/combo_strategy_report.md       回测分析报告（Markdown）
    - web/client/src/data/backtest_combo.json     前端摘要数据（JSON）

【使用方法】
    python3 a3_combo_strategy_backtest.py --strategy all
    python3 a3_combo_strategy_backtest.py --strategy steady
    python3 a3_combo_strategy_backtest.py --strategy aggressive

【本次重点修正项（结构性/逻辑性）】
    - 项目根目录探测：避免 PROJECT_ROOT 误判导致找不到 data/day 与输出目录
    - CSV字段映射与日期解析：严格按 yyyy/mm/dd，异常降级
    - 统计口径修复：报告/JSON 不再对每只股票简单均值，而是按真实 trade_count 聚合
    - 结果结构增强：每行包含 stock_code/name/hold_period 等，便于追溯
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
# 日志
# ------------------------------------------------------------------------------
try:
    from a99_logger import log
except Exception:
    def log(msg, level="INFO"):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}")

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

from a99_indicators import calculate_all_signals
from a99_backtest_utils import (
    get_all_stock_files,
    run_backtest_on_all_stocks,
    backtest_trades_fixed_hold,
    summarize_trades,
)

# ------------------------------------------------------------------------------
# 回测参数
# ------------------------------------------------------------------------------
COMMISSION_RATE = 0.00008
STAMP_TAX_RATE = 0.0005
DEFAULT_HOLD_PERIODS = [5, 10, 20]

# ------------------------------------------------------------------------------
# CSV读取与标准化（与 a2 保持同口径）
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
}
NUMERIC_COLS = ['open', 'high', 'low', 'close', 'volume', 'amount', 'amplitude', 'pct_chg', 'chg', 'turnover']

def _parse_date_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, format='%Y/%m/%d', errors='coerce')
    if len(dt) > 0 and dt.isna().mean() > 0.5:
        dt = pd.to_datetime(s, errors='coerce')
    return dt

def load_stock_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    读取并标准化单只股票数据。
    关键保护：
    - 价格<=0 的行剔除（避免除0、指标异常）
    - volume/amount 负值置 0
    """
    df = None
    for enc in ('utf-8-sig', 'utf-8', 'gbk'):
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except Exception:
            continue
    if df is None or df.empty:
        return None

    df.rename(columns={c: CSV_COL_MAP.get(c, c) for c in df.columns}, inplace=True)
    if 'date' not in df.columns:
        return None

    df['date'] = _parse_date_series(df['date'])
    df.dropna(subset=['date'], inplace=True)

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c not in df.columns:
            df[c] = np.nan if c != 'volume' else 0.0

    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

    for c in ['volume', 'amount']:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
            df.loc[df[c] < 0, c] = 0.0

    if 'name' not in df.columns:
        df['name'] = ''

    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    if len(df) < 30:
        return None
    return df

def _extract_stock_meta(filepath: str, df: pd.DataFrame) -> Dict:
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
    """
    固定持有回测：
    - 返回 summarize_trades 的统计字段
    - 附带 signal_count（触发次数）
    """
    base = {'signal_count': 0, 'trade_count': 0, 'win_count': 0, 'win_rate': 0.0, 'avg_return': 0.0, 'sum_return': 0.0}
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

# ------------------------------------------------------------------------------
# 单只股票：稳健组合 / 激进组合
# ------------------------------------------------------------------------------
def backtest_steady_single(filepath: str) -> Optional[pd.DataFrame]:
    df = load_stock_data(filepath)
    if df is None:
        return None
    meta = _extract_stock_meta(filepath, df)

    df = calculate_all_signals(df)
    if df is None or df.empty:
        return None

    # 缠论“任意买点”：当前版本以 chan_buy1 表示
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

# ------------------------------------------------------------------------------
# 全市场回测
# ------------------------------------------------------------------------------
def run_backtest(strategy: str, stock_files: List[str]) -> pd.DataFrame:
    funcs = {'steady': backtest_steady_single, 'aggressive': backtest_aggressive_single}

    if strategy == 'all':
        all_dfs = []
        for strat, func in funcs.items():
            log(f"开始回测组合策略: {strat}")
            res_list = run_backtest_on_all_stocks(stock_files, func)
            if res_list:
                all_dfs.append(pd.concat(res_list, ignore_index=True))
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if strategy not in funcs:
        log(f"无效的策略参数: {strategy}", level="ERROR")
        return pd.DataFrame()

    log(f"开始回测组合策略: {strategy}")
    res_list = run_backtest_on_all_stocks(stock_files, funcs[strategy])
    return pd.concat(res_list, ignore_index=True) if res_list else pd.DataFrame()

# ------------------------------------------------------------------------------
# 汇总统计（按策略/持有期聚合）
# ------------------------------------------------------------------------------
def _aggregate_strategy_period(df: pd.DataFrame) -> Dict:
    """
    按 trade_count 聚合，避免“逐股票均值”导致统计偏差。
    输入 df 为某个 strategy + hold_period 的多行（每行=某股票统计）。
    """
    trades = int(df['trade_count'].sum())
    wins = int(df.get('win_count', 0).sum())
    sum_return = float(df.get('sum_return', 0).sum())  # 每行 sum_return 已是“该行所有交易收益率%之和”，可直接累加
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

def generate_markdown_report(results: pd.DataFrame, stock_count: int) -> str:
    """
    生成 Markdown 报告：
    - 按 strategy -> hold_period 汇总
    - 给出最佳持有期（以 win_rate 优先）
    """
    lines: List[str] = []
    lines.append("# 组合策略回测报告")
    lines.append("")
    lines.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 扫描股票数: {stock_count}")
    lines.append(f"- 结果行数: {len(results)}（每行=单股票×持有期×策略汇总）")
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
        lines.append(f"- 交易笔数: **{best['trade_count']}**")
        lines.append("")

        lines.append("| 持有期 | 信号次数 | 交易笔数 | 胜率 | 平均单笔收益 | 总收益(求和) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for r in period_rows:
            lines.append(f"| {r['hold_period']} | {r['signal_count']} | {r['trade_count']} | {r['win_rate']:.2f}% | {r['avg_return']:.2f}% | {r['sum_return']:.2f}% |")
        lines.append("")

    return "\n".join(lines)

def generate_json_data(results: pd.DataFrame, stock_count: int) -> Dict:
    """
    前端摘要 JSON：
    - strategies[strategy]：最优持有期 + 该持有期汇总指标
    - by_period：每个持有期的汇总指标（可选，不破坏旧字段）
    """
    data = {
        'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock_count': int(stock_count),
        'strategies': {}
    }

    for strat, df_s in results.groupby('strategy'):
        by_period = {}
        for period, df_p in df_s.groupby('hold_period'):
            by_period[str(int(period))] = _aggregate_strategy_period(df_p)

        # 选择最优持有期：胜率优先，其次平均收益
        items = [{'hold_period': int(p), **v} for p, v in by_period.items()]
        items.sort(key=lambda x: (x['win_rate'], x['avg_return']), reverse=True)

        if not items:
            continue

        best = items[0]
        data['strategies'][strat] = {
            'optimal_period_win': str(best['hold_period']),
            'win_rate': best['win_rate'],
            'avg_return': best['avg_return'],
            'trades': best['trade_count'],
            'signals': best['signal_count'],
            'by_period': by_period
        }

    return data

# ------------------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='组合策略回测系统')
    parser.add_argument('--strategy', type=str, default='all', choices=['all', 'steady', 'aggressive'], help='要运行的组合策略类型')
    parser.add_argument('--data_dir', type=str, default=os.path.join(PROJECT_ROOT, 'data', 'day'), help='数据目录（默认: data/day）')
    args = parser.parse_args()

    stock_files = get_all_stock_files(args.data_dir)
    if not stock_files:
        log(f"未找到股票数据文件: {args.data_dir}", level="ERROR")
        return

    results_df = run_backtest(args.strategy, stock_files)
    if results_df.empty:
        log("回测没有产生有效结果。", level="WARNING")
        return

    # 输出目录
    out_dir = os.path.join(PROJECT_ROOT, 'report', 'total')
    os.makedirs(out_dir, exist_ok=True)

    # 保存CSV
    summary_csv = os.path.join(out_dir, 'combo_strategy_summary.csv')
    results_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    log(f"组合策略回测结果汇总已保存: {summary_csv}")

    # 保存Markdown
    report_md_path = os.path.join(out_dir, 'combo_strategy_report.md')
    md_report = generate_markdown_report(results_df, stock_count=len(stock_files))
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    log(f"组合策略Markdown报告已保存: {report_md_path}")

    # 保存JSON
    json_path = os.path.join(PROJECT_ROOT, 'web', 'client', 'src', 'data', 'backtest_combo.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    json_data = generate_json_data(results_df, stock_count=len(stock_files))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    log(f"Web前端组合数据已保存: {json_path}")

if __name__ == "__main__":
    main()
