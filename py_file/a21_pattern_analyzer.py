#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a21_pattern_analyzer.py
================================================================================

【脚本功能】
    模式特征分析引擎，对回测或扫描产生的成功案例进行深度归因分析：
    1. 收益归归因：计算信号触发后 5/10/15/20 日的阶段性收益。
    2. 波动分析：计算未来 20 日内的最大有利变动 (MFE) 和最大不利变动 (MAE)。
    3. 环境统计：统计信号触发时的成交量比率、均线偏离度 (Bias) 以及历史波动率。
    4. 模式提取：将分析结果汇总为前端展示所需的模式分布数据。

【使用方法】
    直接运行脚本即可（需确保 report/signal_success_cases.csv 已存在）：
    python3 a21_pattern_analyzer.py

【输入文件】
    - report/signal_success_cases.csv (由 a2_unified_backtest.py 扫描产生)

【输出文件】
    - report/pattern_analysis_report.csv         (详细分析报表)
    - web/client/src/data/pattern_analysis_summary.json (前端汇总数据)
    - web/client/src/data/pattern_analysis_by_signal.json (按信号分类的模式数据)

【设计优势】
    - 深度洞察：不仅看最终结果，更关注过程中的风险（MAE）与机会（MFE）。
    - 自动同步：分析结果直接写入前端目录，实现数据链路自动化。
    - 鲁棒性：支持多种日期格式解析，自动处理缺失数据。
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
# 1. 环境配置与路径
# ------------------------------------------------------------------------------
def log(msg: str, level: str = "INFO"):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}")

def find_project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
    for d in candidates:
        if os.path.isdir(os.path.join(d, "data", "day")):
            return d
    return here

PROJECT_ROOT = find_project_root()
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
# 2. 数据处理辅助函数
# ------------------------------------------------------------------------------
CSV_COL_MAP = {
    '名称': 'name', '日期': 'date', '开盘': 'open', '收盘': 'close',
    '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount',
    '振幅': 'amplitude', '涨跌幅': 'pct_chg', '涨跌额': 'chg', '换手率': 'turnover',
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

    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

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
        if os.path.exists(p): return p
    return None

# ------------------------------------------------------------------------------
# 3. 核心分析逻辑
# ------------------------------------------------------------------------------
def analyze_single_case(idx: int, stock_code: str, signal_date: pd.Timestamp, price_df: pd.DataFrame, input_row: Dict[str, Any]) -> Dict[str, Any]:
    dates = price_df['date'].values.astype('datetime64[D]')
    close = price_df['close'].values.astype(float)
    high = price_df['high'].values.astype(float)
    low = price_df['low'].values.astype(float)
    vol = price_df['volume'].values.astype(float)

    sd = np.datetime64(pd.Timestamp(signal_date).date(), 'D')
    # 查找日期索引
    idxs = np.where(dates == sd)[0]
    if len(idxs) == 0:
        pos = np.searchsorted(dates, sd, side='right') - 1
        if pos < 0: raise ValueError("date not found")
        i = int(pos)
    else:
        i = int(idxs[0])

    buy_close = float(close[i])
    out = {'idx': idx, 'stock_code': stock_code, 'date': pd.Timestamp(signal_date).strftime('%Y/%m/%d')}
    for k, v in input_row.items():
        if k not in ('date', 'stock_code'): out[k] = v

    # 未来收益
    n = len(price_df)
    for w in [5, 10, 15, 20]:
        j = i + w
        out[f'forward_ret_{w}'] = round((float(close[j]) - buy_close) / buy_close * 100.0, 4) if j < n else np.nan

    # MFE/MAE
    j_end = min(n - 1, i + 20)
    if j_end > i:
        out['mfe_20'] = round((float(np.nanmax(high[i+1:j_end+1])) - buy_close) / buy_close * 100.0, 4)
        out['mae_20'] = round((float(np.nanmin(low[i+1:j_end+1])) - buy_close) / buy_close * 100.0, 4)
    
    # 环境指标
    if i >= 20:
        rets = pd.Series(close[i-20:i+1]).pct_change().dropna()
        out['vol_20'] = round(float(rets.std()), 6)
        avg_vol = np.nanmean(vol[i-20:i])
        out['vol_ratio_20'] = round(float(vol[i] / avg_vol), 4) if avg_vol > 0 else np.nan
        ma20 = np.nanmean(close[i-19:i+1])
        out['ma_bias_20'] = round((buy_close - ma20) / ma20, 6) if ma20 > 0 else np.nan

    return out

def analyze_stock_group(item: Tuple[str, List[Tuple[int, pd.Timestamp]], List[Dict[str, Any]]]) -> List[Tuple[int, Dict[str, Any]]]:
    stock_code, idx_dates, rows = item
    path = find_stock_csv_path(stock_code)
    if not path: return []
    df = load_daily_csv(path)
    if df is None or len(df) < 30: return []
    
    out = []
    for (idx, dt), row in zip(idx_dates, rows):
        try:
            res = analyze_single_case(idx, stock_code, dt, df, row)
            out.append((idx, res))
        except: continue
    return out

def generate_by_signal_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """生成按信号分类的模式分析数据"""
    if 'signal_type' not in df.columns: return {}
    
    results = {}
    for stype in df['signal_type'].unique():
        sdf = df[df['signal_type'] == stype]
        results[stype] = {
            "total_cases": len(sdf),
            "analyzed_cases": len(sdf),
            "key_patterns": {
                "avg_mfe_20": round(float(sdf['mfe_20'].mean()), 2) if 'mfe_20' in sdf.columns else 0,
                "avg_mae_20": round(float(sdf['mae_20'].mean()), 2) if 'mae_20' in sdf.columns else 0,
                "avg_vol_ratio": round(float(sdf['vol_ratio_20'].mean()), 2) if 'vol_ratio_20' in sdf.columns else 0,
                "avg_bias": round(float(sdf['ma_bias_20'].mean() * 100), 2) if 'ma_bias_20' in sdf.columns else 0
            }
        }
    return results

def main():
    if os.path.exists(SIGNAL_SUCCESS_FILE):
        df = pd.read_csv(SIGNAL_SUCCESS_FILE, encoding='utf-8-sig')
        input_path = SIGNAL_SUCCESS_FILE
    elif os.path.exists(SIGNAL_ALL_FILE):
        df = pd.read_csv(SIGNAL_ALL_FILE, encoding='utf-8-sig')
        if 'is_success' in df.columns: df = df[df['is_success'] == 1]
        input_path = SIGNAL_ALL_FILE
    else:
        log("未找到输入文件", level="ERROR")
        return

    df['date'] = _parse_date_series(df['date'])
    df.dropna(subset=['date'], inplace=True)
    if 'stock_code' not in df.columns and 'stock' in df.columns:
        df.rename(columns={'stock': 'stock_code'}, inplace=True)
    
    df['idx'] = np.arange(len(df))
    groups = {}
    for _, r in df.iterrows():
        code = str(r['stock_code'])
        groups.setdefault(code, {'idx_dates': [], 'rows': []})
        groups[code]['idx_dates'].append((int(r['idx']), pd.Timestamp(r['date'])))
        groups[code]['rows'].append(r.to_dict())
    
    tasks = [(c, v['idx_dates'], v['rows']) for c, v in groups.items()]
    log(f"开始分析 {len(df)} 条案例...")
    
    results = []
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(analyze_stock_group, t) for t in tasks]
        for fu in as_completed(futs):
            results.extend(fu.result())
    
    if not results: return
    results.sort(key=lambda x: x[0])
    report_df = pd.DataFrame([x[1] for x in results])
    report_df.to_csv(REPORT_OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    summary = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "total_cases": len(df),
        "analyzed_cases": len(report_df),
        "indicators": {}, "theories": {}
    }
    
    for target in [SUMMARY_OUTPUT_FILE, WEB_SUMMARY_FILE]:
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
    by_signal = generate_by_signal_analysis(report_df)
    for target in [BY_SIGNAL_OUTPUT_FILE, WEB_BY_SIGNAL_FILE]:
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(by_signal, f, ensure_ascii=False, indent=2)
    
    log("分析完成。")

if __name__ == "__main__":
    main()
