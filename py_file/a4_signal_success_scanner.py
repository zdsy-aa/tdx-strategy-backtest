\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a4_signal_success_scanner.py
================================================================================

【脚本功能】
    “信号成功案例扫描器”：
    扫描 data/day 目录下所有股票日线 CSV，识别买入信号，并计算“未来固定持有 N 天”的
    净收益率（包含手续费/印花税），标记是否为成功案例，输出信号明细与成功案例列表。

【为什么要做这个脚本】
    a21_pattern_analyzer.py 需要输入一份“成功案例列表”，用于做模式归因/特征统计。
    该脚本负责生成：
        report/all_signal_records.csv      所有信号记录（成功+失败）
        report/signal_success_cases.csv    仅成功案例

【数据输入要求】
    CSV 字段（与你的数据一致）：
        名称、日期(yyyy/mm/dd)、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌幅、涨跌额、换手率
    本脚本会自动映射/清洗为英文列，并处理异常值（避免除0/NaN）。

【识别的信号类型（不改变原语义，只修复实现Bug与健壮性）】
    - six_veins: 六脉神剑 >= MIN_RED_COUNT 的“跨越触发点”（昨日<阈值，今日>=阈值）
    - chan_buy : 缠论买点（当前版本仅 chan_buy1；触发点 False->True）
    - buy_sell : 买卖点（buy1/buy2；触发点 False->True）

    注：单日可能同时满足多个类型，signal_type 会以 “+” 连接。

【成功定义】
    - 未来 HOLDING_DAYS 天后卖出（使用收盘价）
    - 计算净收益率 future_return（含佣金+印花税）
    - is_success = future_return > SUCCESS_THRESHOLD

【使用方法】
    python3 a4_signal_success_scanner.py
    可选参数：
        --holding_days 15
        --min_red 4
        --success_threshold 5

【本次修复重点】
    1) 去重Bug：原脚本对同一 record append 两次导致重复记录（已修复）
    2) stock_code 字段统一：输出 stock_code（兼容 a21 读取）
    3) 日期格式：输出 date 为 yyyy/mm/dd 字符串，便于后续一致解析
    4) 末端样本：未来不足 HOLDING_DAYS 的信号不纳入（避免 NaN）
================================================================================
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# ------------------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')
WEB_DATA_DIR = os.path.join(PROJECT_ROOT, 'web', 'client', 'src', 'data')

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

def load_stock_data(filepath: str) -> Optional[pd.DataFrame]:
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

# ------------------------------------------------------------------------------
# 交易成本（沿用回测口径）
# ------------------------------------------------------------------------------
COMMISSION_RATE = 0.00008
STAMP_TAX_RATE = 0.0005

def _calc_future_return(buy_price: float, sell_price: float) -> Optional[float]:
    """
    计算净收益率（%）：
        buy_cost  = buy_price  * (1 + commission)
        sell_net  = sell_price * (1 - commission - stamp_tax)
        future_return = (sell_net - buy_cost) / buy_cost * 100

    异常保护：价格<=0 或 NaN 返回 None
    """
    if buy_price is None or sell_price is None:
        return None
    if not np.isfinite(buy_price) or not np.isfinite(sell_price):
        return None
    if buy_price <= 0:
        return None

    buy_cost = buy_price * (1.0 + COMMISSION_RATE)
    sell_net = sell_price * (1.0 - COMMISSION_RATE - STAMP_TAX_RATE)
    if buy_cost <= 0:
        return None
    return (sell_net - buy_cost) / buy_cost * 100.0

# ------------------------------------------------------------------------------
# 单只股票扫描
# ------------------------------------------------------------------------------
def scan_single_stock(file_path: str, holding_days: int, min_red: int, success_threshold: float) -> pd.DataFrame:
    """
    返回单只股票的信号记录 DataFrame；无信号返回空 DataFrame。
    """
    df = load_stock_data(file_path)
    if df is None or df.empty:
        return pd.DataFrame()

    stock_code = os.path.basename(file_path).replace('.csv', '')
    stock_name = str(df['name'].iloc[-1]) if 'name' in df.columns and len(df) > 0 else ''

    # 计算全量信号（六脉/买卖点/摇钱树/缠论）
    df = calculate_all_signals(df)
    if df is None or df.empty:
        return pd.DataFrame()

    # 触发点定义：避免连续多日重复计数
    six_cnt = pd.to_numeric(df.get('six_veins_count', 0), errors='coerce').fillna(0).astype(int)
    sig_six = (six_cnt >= int(min_red)) & (six_cnt.shift(1, fill_value=0) < int(min_red))

    chan = df.get('chan_buy1', False)
    sig_chan = chan.fillna(False).astype(bool) & ~chan.fillna(False).astype(bool).shift(1, fill_value=False)

    buy1 = df.get('buy1', False)
    buy2 = df.get('buy2', False)
    sig_buy1 = buy1.fillna(False).astype(bool) & ~buy1.fillna(False).astype(bool).shift(1, fill_value=False)
    sig_buy2 = buy2.fillna(False).astype(bool) & ~buy2.fillna(False).astype(bool).shift(1, fill_value=False)

    # 逐日扫描（只在有触发的日子生成记录）
    records: List[Dict] = []
    n = len(df)
    for i in range(n):
        # 未来不足 holding_days 的样本不统计（避免 NaN）
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

        # 六脉的6个红色指标列（若不存在则默认0）
        for col in ['macd_red', 'kdj_red', 'rsi_red', 'lwr_red', 'bbi_red', 'mtm_red']:
            rec[col] = int(bool(df.at[i, col])) if col in df.columns else 0

        records.append(rec)

    return pd.DataFrame(records)

# ------------------------------------------------------------------------------
# 汇总与落盘
# ------------------------------------------------------------------------------
def aggregate_results(all_records: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, Dict):
    if all_records is None or all_records.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            'total_signals': 0, 'success_signals': 0, 'success_rate': 0.0,
            'signal_type_counts': {}, 'six_veins_success_by_count': {}
        }

    # 排序：用 date 字符串也可，但这里转成 datetime 再排序更稳
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

def save_results(all_records: pd.DataFrame, success_cases: pd.DataFrame, summary: Dict):
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(WEB_DATA_DIR, exist_ok=True)

    all_records_path = os.path.join(REPORT_DIR, 'all_signal_records.csv')
    success_path = os.path.join(REPORT_DIR, 'signal_success_cases.csv')

    all_records.to_csv(all_records_path, index=False, encoding='utf-8-sig')
    success_cases.to_csv(success_path, index=False, encoding='utf-8-sig')

    summary_path = os.path.join(REPORT_DIR, 'signal_summary.json')
    web_summary_path = os.path.join(WEB_DATA_DIR, 'signal_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(web_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"所有信号记录已保存: {all_records_path}")
    log(f"成功案例列表已保存: {success_path}")
    log(f"统计摘要已保存: {summary_path}")
    log(f"前端统计摘要已更新: {web_summary_path}")

# ------------------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='信号成功案例扫描器')
    parser.add_argument('--holding_days', type=int, default=15, help='固定持有天数')
    parser.add_argument('--min_red', type=int, default=4, help='六脉红色数量阈值')
    parser.add_argument('--success_threshold', type=float, default=5.0, help='成功阈值(净收益率%, 大于该值为成功)')
    args = parser.parse_args()

    # 收集数据文件
    stock_files: List[str] = []
    if not os.path.isdir(DATA_DIR):
        log(f"数据目录不存在: {DATA_DIR}", level="ERROR")
        return

    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.csv'):
                stock_files.append(os.path.join(root, f))

    if not stock_files:
        log("未找到任何股票CSV文件，请检查 data/day 目录。", level="ERROR")
        return

    log(f"开始扫描 {len(stock_files)} 只股票的买入信号...")
    results: List[pd.DataFrame] = []

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(scan_single_stock, fp, args.holding_days, args.min_red, args.success_threshold): fp
            for fp in stock_files
        }
        for future in as_completed(futures):
            fp = futures[future]
            try:
                df_res = future.result()
                if df_res is not None and not df_res.empty:
                    results.append(df_res)
            except Exception as e:
                log(f"股票扫描出错: {fp}, 错误: {e}", level="ERROR")

    if not results:
        log("未检测到任何买入信号。", level="WARNING")
        return

    all_records_df = pd.concat(results, ignore_index=True)
    all_records_df, success_cases_df, summary = aggregate_results(all_records_df)
    save_results(all_records_df, success_cases_df, summary)

    log("扫描完成。")

if __name__ == "__main__":
    main()
