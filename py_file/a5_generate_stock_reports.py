\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a5_generate_stock_reports.py
================================================================================

【脚本功能】
    生成“全市场股票信号与收益概览报表”，并输出为前端可直接读取的 JSON 文件：
        web/client/src/data/stock_reports.json

    每只股票计算：
    - 总体信号交易的累计收益/胜率（按固定持有期）
    - 年内/当月的累计收益/胜率
    - 最近5个交易日内的最新信号摘要（用于快速筛选）

【数据输入要求】
    CSV 字段（与你的真实数据一致）：
        名称、日期(yyyy/mm/dd)、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌幅、涨跌额、换手率
    脚本会自动映射并清洗为英文列（date/open/high/low/close/volume 等）。

【输出文件】
    - web/client/src/data/stock_reports.json

【使用方法】
    python3 a5_generate_stock_reports.py
    可选参数：
        --end_date 2026-01-27         # 统计截止日（默认当天）
        --hold_days 14                # 固定持有天数（默认14）
        --limit 500                   # 仅处理前 N 个文件（用于调试）

【口径说明（避免误解）】
    - 本脚本的“累计收益”= 各笔交易收益率(%)简单求和（用于排行榜/对比）
      并非复利回撤曲线，不代表真实资金曲线。
    - 如需严格交易成本口径，请使用 a2/a3/a4/a99_backtest_utils 的回测口径。

【本次修复重点】
    1) 项目根目录识别（避免 parent.parent 误判）
    2) 股票名称读取修复：使用 name 列（原脚本读取 “名称” 列会失效）
    3) 输出日期统一：lastSignalDate 使用 yyyy/mm/dd
    4) 结构性修复：移除脚本末尾“无条件 print/执行”副作用，保证只在 __main__ 下运行
================================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from functools import partial
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

try:
    from a99_logger import log
except Exception:
    def log(msg, level="INFO"):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}")

# ------------------------------------------------------------------------------
# 项目根目录探测
# ------------------------------------------------------------------------------
def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, here.parent.parent]
    for d in candidates:
        if (d / "data" / "day").is_dir():
            return d
    return here

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from a99_indicators import calculate_all_signals

DATA_DIR = PROJECT_ROOT / "data" / "day"
WEB_DATA_DIR = PROJECT_ROOT / "web" / "client" / "src" / "data"
STOCK_REPORTS_FILE = WEB_DATA_DIR / "stock_reports.json"

# ------------------------------------------------------------------------------
# CSV读取与标准化
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

def load_stock_data(csv_path: Path) -> Optional[pd.DataFrame]:
    df = None
    for enc in ('utf-8-sig', 'utf-8', 'gbk'):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
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

# ------------------------------------------------------------------------------
# 信号识别与收益计算
# ------------------------------------------------------------------------------
def find_signals(df: pd.DataFrame, signal_type: str) -> List[Dict]:
    """
    返回指定信号类型的触发列表，每项包含 date(时间戳) 与 price(收盘价)。
    触发口径：False -> True（避免连续多日重复计数）。
    """
    signals: List[Dict] = []

    if signal_type == 'six_veins_6red':
        cond = (df.get('six_veins_count', 0) == 6)
    elif signal_type == 'six_veins_5red':
        cond = (df.get('six_veins_count', 0) >= 5)
    elif signal_type == 'six_veins_4red':
        cond = (df.get('six_veins_count', 0) >= 4)
    elif signal_type == 'buy_point_1':
        cond = df.get('buy1', False).fillna(False).astype(bool)
    elif signal_type == 'buy_point_2':
        cond = df.get('buy2', False).fillna(False).astype(bool)
    else:
        # 其他类型（例如 chan_buy1/chan_buy2 等）
        cond = df.get(signal_type, False)
        if isinstance(cond, pd.Series):
            cond = cond.fillna(False).astype(bool)
        else:
            cond = pd.Series([False] * len(df))

    cond = cond.fillna(False).astype(bool)
    trigger = cond & ~cond.shift(1, fill_value=False)

    idxs = list(df.index[trigger])
    for i in idxs:
        signals.append({'date': df.at[i, 'date'], 'price': float(df.at[i, 'close'])})
    return signals

def calculate_trade_result(df: pd.DataFrame, signal: Dict, hold_days: int) -> Optional[Dict]:
    """
    固定持有 hold_days 天后的收益率（%）。
    保护：避免买价为0/NaN、越界、卖价NaN 等情况。
    """
    buy_dt = signal['date']
    # 定位买入行
    buy_idx_list = df.index[df['date'] == buy_dt].tolist()
    if not buy_idx_list:
        return None
    buy_idx = buy_idx_list[0]
    sell_idx = buy_idx + int(hold_days)
    if sell_idx >= len(df):
        return None

    buy_price = float(signal.get('price', np.nan))
    sell_price = float(df.at[sell_idx, 'close'])

    if not np.isfinite(buy_price) or buy_price <= 0:
        return None
    if not np.isfinite(sell_price) or sell_price <= 0:
        return None

    ret = (sell_price - buy_price) / buy_price * 100.0
    return {'buy_date': buy_dt, 'return': float(ret), 'win': bool(ret > 0)}

def get_market_info(file_path: Path):
    """从路径判断市场信息（用于前端展示）。"""
    path_str = str(file_path).replace("\\", "/")
    if "/sh/" in path_str: return 'sh', '沪市'
    if "/sz/" in path_str: return 'sz', '深市'
    if "/bj/" in path_str: return 'bj', '北交所'
    return 'unknown', '未知'

def process_single_stock(stock_file: Path, end_dt: pd.Timestamp, year_start: pd.Timestamp, month_start: pd.Timestamp, hold_days: int) -> Optional[Dict]:
    """
    单只股票处理（用于并行）。
    发生异常返回 None，避免一个文件拖垮全局。
    """
    try:
        stock_code = stock_file.stem
        market, market_name = get_market_info(stock_file)

        df = load_stock_data(stock_file)
        if df is None or len(df) < 30:
            return None

        stock_name = str(df['name'].iloc[-1]) if 'name' in df.columns and len(df) > 0 else stock_code

        # 指标/信号计算
        df = calculate_all_signals(df)
        if df is None or df.empty:
            return None

        # 统计各策略信号（包含缠论买点：当前稳定为 chan_buy1；若未来扩展也可继续加）
        base_types = ['six_veins_6red', 'six_veins_5red', 'six_veins_4red', 'buy_point_1', 'buy_point_2']
        chan_types = ['chan_buy1', 'chan_buy2', 'chan_buy3', 'chan_strong_buy2', 'chan_like_buy2']

        all_trades: List[Dict] = []
        year_trades: List[Dict] = []
        month_trades: List[Dict] = []

        # 汇总所有信号交易
        for stype in base_types + chan_types:
            sigs = find_signals(df, stype)
            for sig in sigs:
                res = calculate_trade_result(df, sig, hold_days)
                if res is None:
                    continue
                res['signal_type'] = stype
                all_trades.append(res)

        if not all_trades:
            return None

        # 年/月过滤
        year_trades = [t for t in all_trades if t['buy_date'] >= year_start]
        month_trades = [t for t in all_trades if t['buy_date'] >= month_start]

        # 统计函数
        def _sum_ret(trades: List[Dict]) -> float:
            return float(np.sum([t['return'] for t in trades])) if trades else 0.0

        def _win_rate(trades: List[Dict]) -> float:
            if not trades:
                return 0.0
            wins = sum(1 for t in trades if t['win'])
            return wins / len(trades) * 100.0

        # 最新信号（最近 5 日，优先6红，其次>=5红）
        last_signal = "无"
        last_date = "-"
        df_recent = df.tail(5).copy()
        for idx in df_recent.index[::-1]:
            cnt = int(df_recent.at[idx, 'six_veins_count']) if 'six_veins_count' in df_recent.columns else 0
            if cnt == 6:
                last_signal = "六脉6红"
                last_date = pd.Timestamp(df_recent.at[idx, 'date']).strftime('%Y/%m/%d')
                break
            elif cnt >= 5:
                last_signal = "六脉5红"
                last_date = pd.Timestamp(df_recent.at[idx, 'date']).strftime('%Y/%m/%d')
                break

        return {
            'code': stock_code,
            'name': stock_name,
            'market': market,
            'marketName': market_name,

            'totalReturn': f"{_sum_ret(all_trades):.1f}%",
            'yearReturn': f"{_sum_ret(year_trades):.1f}%",
            'monthReturn': f"{_sum_ret(month_trades):.1f}%",

            'totalWinRate': f"{_win_rate(all_trades):.1f}%",
            'yearWinRate': f"{_win_rate(year_trades):.1f}%",
            'monthWinRate': f"{_win_rate(month_trades):.1f}%",

            'totalTrades': len(all_trades),
            'yearTrades': len(year_trades),
            'monthTrades': len(month_trades),

            'lastSignal': last_signal,
            'lastSignalDate': last_date,

            # 额外信息：便于前端/调试
            'holdDays': int(hold_days),
            'endDate': end_dt.strftime('%Y-%m-%d'),
        }
    except Exception as e:
        return None

# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def generate_reports(end_date: Optional[str], hold_days: int, limit: Optional[int]):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    end_dt = pd.to_datetime(end_date)
    year_start = pd.to_datetime(f"{end_dt.year}-01-01")
    month_start = pd.to_datetime(f"{end_dt.year}-{end_dt.month:02d}-01")

    log(f"开始生成报告，截止日期: {end_date}, 固定持有: {hold_days} 天")
    stock_files = list(DATA_DIR.rglob("*.csv"))
    if limit:
        stock_files = stock_files[:int(limit)]
    log(f"找到 {len(stock_files)} 只股票数据文件")

    if not stock_files:
        log("未找到数据文件，退出。", level="ERROR")
        return

    # 并行处理
    num_cores = max(1, cpu_count() - 1)
    log(f"使用 {num_cores} 个核心进行并行计算...")

    worker = partial(process_single_stock, end_dt=end_dt, year_start=year_start, month_start=month_start, hold_days=hold_days)

    with Pool(num_cores) as pool:
        results = pool.map(worker, stock_files)

    final_reports = [r for r in results if r is not None]

    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(STOCK_REPORTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_reports, f, ensure_ascii=False, indent=2)

    log(f"报告生成完成，共计 {len(final_reports)} 只股票，已保存至 {STOCK_REPORTS_FILE}")

def main():
    parser = argparse.ArgumentParser(description='生成股票信号收益概览报表')
    parser.add_argument('--end_date', type=str, default=None, help='统计截止日期（YYYY-MM-DD，默认当天）')
    parser.add_argument('--hold_days', type=int, default=14, help='固定持有天数（默认14）')
    parser.add_argument('--limit', type=int, default=None, help='仅处理前N只股票（调试用）')
    args = parser.parse_args()

    import time
    t0 = time.time()
    generate_reports(args.end_date, args.hold_days, args.limit)
    log(f"总耗时: {time.time() - t0:.2f} 秒")

if __name__ == "__main__":
    main()
