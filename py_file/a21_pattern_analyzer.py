\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a21_pattern_analyzer.py
================================================================================

【脚本功能】
    对 a4_signal_success_scanner.py 输出的信号案例（默认成功案例列表）做
    “回看归因/模式特征统计”，输出分析报表 CSV + 摘要 JSON。

【输入】
    默认读取：
        report/signal_success_cases.csv
    如果不存在，则尝试读取：
        report/all_signal_records.csv 并筛选 is_success==1

    输入文件要求至少包含：
        stock_code, date(yyyy/mm/dd)
    其他列（如 future_return/signal_type/six_veins_count 等）会被保留并写入报表。

【输出】
    1) report/pattern_analysis_report.csv
    2) report/pattern_analysis_summary.json
    3) web/client/src/data/pattern_analysis_summary.json

【本次修复点（你点名的三条）】
    1) 输入列兼容：允许 stock 或 stock_code；统一转 stock_code
    2) 日期格式严格支持 yyyy/mm/dd，并降级通用解析
    3) idx 生成方式修复：全局唯一 idx（不再按每只股票从0开始）

【分析内容（基础但稳定，适合16GB环境）】
    针对每个信号点，计算：
    - buy_close：信号当日收盘价
    - forward_ret_5/10/15/20：未来N日收益率(%)（不含交易成本）
    - mfe_20 / mae_20：未来20日最大有利/最大不利波动(%)（基于最高/最低）
    - vol_20：信号前20日收益率波动率（std）
    - vol_ratio_20：信号当日成交量 / 过去20日平均成交量
    - ma_bias_20：信号当日收盘 vs 20日均线偏离

    注：future_return（含交易成本）以输入文件的 future_return 为准，本脚本不覆盖，只做补充计算。
================================================================================
"""

import os
import sys
import json
import multiprocessing
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------------------------
# 日志
# ------------------------------------------------------------------------------
def log(msg: str, level: str = "INFO"):
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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "day")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
WEB_DATA_DIR = os.path.join(PROJECT_ROOT, "web", "client", "src", "data")

SIGNAL_SUCCESS_FILE = os.path.join(REPORT_DIR, "signal_success_cases.csv")
SIGNAL_ALL_FILE = os.path.join(REPORT_DIR, "all_signal_records.csv")

REPORT_OUTPUT_FILE = os.path.join(REPORT_DIR, "pattern_analysis_report.csv")
SUMMARY_OUTPUT_FILE = os.path.join(REPORT_DIR, "pattern_analysis_summary.json")
BY_SIGNAL_OUTPUT_FILE = os.path.join(REPORT_DIR, "pattern_analysis_by_signal.json")
WEB_BY_SIGNAL_FILE = os.path.join(WEB_DATA_DIR, "pattern_analysis_by_signal.json")
WEB_SUMMARY_FILE = os.path.join(WEB_DATA_DIR, "pattern_analysis_summary.json")

# ------------------------------------------------------------------------------
# CSV读取与标准化（中文列 -> 英文列）
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

def load_daily_csv(filepath: str) -> Optional[pd.DataFrame]:
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
    return df if len(df) > 0 else None

def find_stock_csv_path(stock_code: str) -> Optional[str]:
    code = str(stock_code).strip()
    if code.upper().startswith(("SH.", "SZ.", "BJ.")):
        code = code.split(".", 1)[1]

    candidates = [
        os.path.join(DATA_DIR, 'sh', f'{code}.csv'),
        os.path.join(DATA_DIR, 'sz', f'{code}.csv'),
        os.path.join(DATA_DIR, 'bj', f'{code}.csv'),
        os.path.join(DATA_DIR, f'{code}.csv'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # 兜底：遍历查找
    for root, _, files in os.walk(DATA_DIR):
        if f'{code}.csv' in files:
            return os.path.join(root, f'{code}.csv')
    return None

# ------------------------------------------------------------------------------
# 案例分析（单条）
# ------------------------------------------------------------------------------
HORIZON_DAYS = 20
FORWARD_WINDOWS = [5, 10, 15, 20]

def _find_index_for_date(dates: np.ndarray, signal_dt: np.datetime64) -> Optional[int]:
    """
    尽量精确匹配 signal_dt；若无精确匹配，则取最后一个 <= signal_dt 的索引。
    """
    # 精确匹配
    idxs = np.where(dates == signal_dt)[0]
    if len(idxs) > 0:
        return int(idxs[0])

    # <= 的最近日期
    # dates 已排序
    pos = np.searchsorted(dates, signal_dt, side='right') - 1
    if pos >= 0 and pos < len(dates):
        return int(pos)
    return None

def analyze_single_case(idx: int, stock_code: str, signal_date: pd.Timestamp, price_df: pd.DataFrame, input_row: Dict[str, Any]) -> Dict[str, Any]:
    dates = price_df['date'].values.astype('datetime64[D]')
    close = price_df['close'].values.astype(float)
    high = price_df['high'].values.astype(float)
    low = price_df['low'].values.astype(float)
    vol = price_df['volume'].values.astype(float)

    sd = np.datetime64(pd.Timestamp(signal_date).date(), 'D')
    i = _find_index_for_date(dates, sd)
    if i is None:
        raise ValueError("signal date not found in price series")

    buy_close = float(close[i])
    if not np.isfinite(buy_close) or buy_close <= 0:
        raise ValueError("invalid buy_close")

    out: Dict[str, Any] = {}
    out['idx'] = int(idx)
    out['stock_code'] = str(stock_code)
    out['date'] = pd.Timestamp(signal_date).strftime('%Y/%m/%d')
    out['name'] = str(price_df['name'].iloc[-1]) if 'name' in price_df.columns and len(price_df) > 0 else ''

    # 把输入行字段原样带过来（future_return / signal_type / six_veins_count 等）
    for k, v in input_row.items():
        if k in ('date', 'stock_code'):  # 避免覆盖
            continue
        out[k] = v

    out['buy_close'] = round(buy_close, 4)

    # forward returns
    n = len(price_df)
    for w in FORWARD_WINDOWS:
        j = i + int(w)
        if j < n:
            out[f'forward_ret_{w}'] = round((float(close[j]) - buy_close) / buy_close * 100.0, 4)
        else:
            out[f'forward_ret_{w}'] = np.nan

    # MFE/MAE over 20 days
    j_end = min(n - 1, i + HORIZON_DAYS)
    if j_end > i:
        max_high = float(np.nanmax(high[i+1:j_end+1]))
        min_low = float(np.nanmin(low[i+1:j_end+1]))
        out['mfe_20'] = round((max_high - buy_close) / buy_close * 100.0, 4) if np.isfinite(max_high) else np.nan
        out['mae_20'] = round((min_low - buy_close) / buy_close * 100.0, 4) if np.isfinite(min_low) else np.nan
    else:
        out['mfe_20'] = np.nan
        out['mae_20'] = np.nan

    # pre-20 volatility
    lookback = 20
    if i >= lookback + 1:
        rets = pd.Series(close[i-lookback:i+1]).pct_change().dropna().to_numpy(dtype=float)
        out['vol_20'] = round(float(np.nanstd(rets)), 6) if len(rets) > 0 else np.nan
    else:
        out['vol_20'] = np.nan

    # volume ratio
    if i >= lookback and np.nanmean(vol[i-lookback:i]) > 0:
        out['vol_ratio_20'] = round(float(vol[i] / np.nanmean(vol[i-lookback:i])), 4)
    else:
        out['vol_ratio_20'] = np.nan

    # MA bias 20
    if i >= lookback:
        ma20 = float(np.nanmean(close[i-lookback+1:i+1]))
        out['ma_bias_20'] = round((buy_close - ma20) / ma20, 6) if ma20 > 0 else np.nan
    else:
        out['ma_bias_20'] = np.nan

    return out

# ------------------------------------------------------------------------------
# 分组处理（单只股票）
# ------------------------------------------------------------------------------
def analyze_stock_group(item: Tuple[str, List[Tuple[int, pd.Timestamp]], List[Dict[str, Any]]]) -> List[Tuple[int, Dict[str, Any]]]:
    """
    输入：
        (stock_code, [(idx, date), ...], [input_row_dict, ...])
    返回：
        [(idx, analysis_dict), ...]
    """
    stock_code, idx_dates, rows = item

    path = find_stock_csv_path(stock_code)
    if not path:
        return []

    df = load_daily_csv(path)
    if df is None or len(df) < 30:
        return []

    out: List[Tuple[int, Dict[str, Any]]] = []
    for (idx, dt), row in zip(idx_dates, rows):
        try:
            analysis = analyze_single_case(idx, stock_code, dt, df, row)
            out.append((idx, analysis))
        except Exception:
            # 单条失败不影响整只股票
            continue
    return out

# ------------------------------------------------------------------------------
# 模式分析（按信号类型）
# ------------------------------------------------------------------------------
def generate_by_signal_analysis(report_df: pd.DataFrame) -> Dict:
    """按信号类型生成模式分析"""
    by_signal = {}
    
    # 确保 signal_type 存在且非空
    if 'signal_type' not in report_df.columns or report_df['signal_type'].empty:
        return {}
    
    # 确保六脉神剑指标存在
    indicator_cols = ['macd_red', 'kdj_red', 'rsi_red', 'lwr_red', 'bbi_red', 'mtm_red']
    
    for signal_type, signal_data in report_df.groupby('signal_type'):
        
        # 计算关键模式
        key_patterns = {}
        total_cases = len(signal_data)
        
        for col in indicator_cols:
            if col in signal_data.columns:
                # 统计红柱（True）的数量
                true_count = signal_data[col].sum()
                key_patterns[col.upper()] = {
                    'true_count': int(true_count),
                    'true_rate': round(true_count / total_cases * 100, 2) if total_cases > 0 else 0
                }
        
        by_signal[signal_type] = {
            'total_cases': int(total_cases),
            'analyzed_cases': int(total_cases),
            'key_patterns': key_patterns
        }
    
    return by_signal

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------
def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(WEB_DATA_DIR, exist_ok=True)

    input_path = SIGNAL_SUCCESS_FILE if os.path.exists(SIGNAL_SUCCESS_FILE) else SIGNAL_ALL_FILE
    if not os.path.exists(input_path):
        log("未找到输入文件：report/signal_success_cases.csv 或 report/all_signal_records.csv", level="ERROR")
        return

    df = pd.read_csv(input_path, encoding='utf-8-sig')

    # 列兼容：stock 或 stock_code
    if 'stock_code' not in df.columns and 'stock' in df.columns:
        df.rename(columns={'stock': 'stock_code'}, inplace=True)

    if 'stock_code' not in df.columns or 'date' not in df.columns:
        raise ValueError("输入文件必须包含 stock_code 与 date 列")

    # 日期解析：优先 yyyy/mm/dd
    df['date'] = _parse_date_series(df['date'])
    df.dropna(subset=['date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 若是 all_signal_records.csv，则默认只取成功案例
    if os.path.basename(input_path) == os.path.basename(SIGNAL_ALL_FILE) and 'is_success' in df.columns:
        df = df[df['is_success'] == 1].copy()
        df.reset_index(drop=True, inplace=True)

    if df.empty:
        log("输入案例为空，退出。", level="WARNING")
        return

    # 全局唯一 idx：按输入顺序
    df['idx'] = np.arange(len(df), dtype=int)

    # 分组任务：每只股票一个任务，携带 (idx,date) + 对应行dict（用于保留输入字段）
    groups: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        code = str(r['stock_code'])
        groups.setdefault(code, {'idx_dates': [], 'rows': []})
        groups[code]['idx_dates'].append((int(r['idx']), pd.Timestamp(r['date'])))
        groups[code]['rows'].append(r.to_dict())

    tasks = [(code, v['idx_dates'], v['rows']) for code, v in groups.items()]
    log(f"开始分析案例: 共 {len(df)} 条，涉及 {len(tasks)} 只股票，输入文件: {input_path}")

    results: List[Tuple[int, Dict[str, Any]]] = []
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(analyze_stock_group, t) for t in tasks]
        for fu in as_completed(futs):
            try:
                results.extend(fu.result())
            except Exception:
                continue

    if not results:
        log("未得到任何分析结果（可能价格数据找不到对应日期）。", level="WARNING")
        return

    # 按 idx 排序，输出报表
    results.sort(key=lambda x: x[0])
    report_rows = [x[1] for x in results]
    report_df = pd.DataFrame(report_rows)

    report_df.to_csv(REPORT_OUTPUT_FILE, index=False, encoding='utf-8-sig')
    log(f"分析报表已保存: {REPORT_OUTPUT_FILE}")

    # 摘要统计
    summary: Dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": os.path.basename(input_path),
        "total_cases": int(len(report_df)),
    }

    # 输入的 future_return（含成本）统计
    if 'future_return' in report_df.columns:
        fr = pd.to_numeric(report_df['future_return'], errors='coerce')
        summary['future_return_mean'] = round(float(fr.mean()), 4) if fr.notna().any() else None
        summary['future_return_median'] = round(float(fr.median()), 4) if fr.notna().any() else None

    # forward_ret_15（不含成本）统计
    if 'forward_ret_15' in report_df.columns:
        f15 = pd.to_numeric(report_df['forward_ret_15'], errors='coerce')
        summary['forward_ret_15_mean'] = round(float(f15.mean()), 4) if f15.notna().any() else None
        summary['forward_ret_15_median'] = round(float(f15.median()), 4) if f15.notna().any() else None

    # 信号类型分布
    if 'signal_type' in report_df.columns:
        summary['signal_type_counts'] = report_df['signal_type'].value_counts().to_dict()

    # 六脉数量分布
    if 'six_veins_count' in report_df.columns:
        svc = pd.to_numeric(report_df['six_veins_count'], errors='coerce')
        summary['six_veins_count_counts'] = svc.value_counts().sort_index().to_dict()

    with open(SUMMARY_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(WEB_SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"摘要已保存: {SUMMARY_OUTPUT_FILE}")
    log(f"前端摘要已更新: {WEB_SUMMARY_FILE}")

    # 生成按信号类型的分析 (pattern_analysis_by_signal.json)
    by_signal_data = generate_by_signal_analysis(report_df)
    
    with open(BY_SIGNAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(by_signal_data, f, ensure_ascii=False, indent=2)
    with open(WEB_BY_SIGNAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(by_signal_data, f, ensure_ascii=False, indent=2)
        
    log(f"按信号类型分析已保存: {BY_SIGNAL_OUTPUT_FILE}")
    log(f"前端按信号类型分析已更新: {WEB_BY_SIGNAL_FILE}")

if __name__ == "__main__":
    main()
