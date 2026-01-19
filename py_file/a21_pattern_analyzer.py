#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
成功案例模式分析器 (a21_pattern_analyzer.py)
================================================================================

功能说明:
    本脚本用于分析成功买入信号案例的共性特征：
    1. 计算成功案例在信号触发时的多种技术指标状态
    2. 应用道氏理论、威科夫理论的相关判据
    3. 按买点类型分类统计共性特征
    4. 生成详细的分析报告

分析维度:
    - 常用指标：MACD, KDJ, BOLL, RSI, DMI, DMA, SAR, BBI, OBV, WR, CCI等
    - 市场理论：道氏理论、威科夫理论的相关判据
    - 均线系统：多周期均线与金叉状态

输入:
    - report/signal_success_cases.csv: 成功案例数据
    - data/day/ 目录下的股票CSV文件

输出:
    - report/pattern_analysis_report.csv: 详细分析报告
    - report/pattern_summary.json: 模式统计摘要
    - report/pattern_analysis_by_signal.json: 按信号类型分类的统计

性能优化原则（不改变原逻辑/口径）:
    - 同一只股票：只读CSV一次，只做标准化一次
    - 同一只股票多条信号：指标序列“全量预计算一次”，每条信号只做O(1)索引取值
    - 信号日期定位：使用numpy二分查找
    - 并行粒度：按股票分组后多进程

作者: TradeGuide System
版本: 1.0.2 (进一步性能优化版，不改变原逻辑)
创建日期: 2026-01-15
================================================================================
"""

try:
    from a99_logger import log
except ImportError:
    def log(msg, level="INFO"):
        print(f"[{level}] {msg}")

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入基础函数（保持原逻辑依赖）
from a99_indicators import (
    REF, MA, EMA, SMA, HHV, LLV, CROSS, COUNT, ABS, MAX, IF
)

# ==============================================================================
# 配置常量
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')
WEB_DATA_DIR = os.path.join(PROJECT_ROOT, 'web', 'client', 'src', 'data')
SUCCESS_CASES_FILE = os.path.join(REPORT_DIR, 'signal_success_cases.csv')

# ==============================================================================
# 指标序列预计算（不改变公式逻辑，只改变“何时计算/复用”）
# ==============================================================================

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float('nan')


def _safe_bool(x) -> bool:
    try:
        if pd.isna(x):
            return False
        return bool(x)
    except Exception:
        return False


def _precompute_feature_cache(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    预计算整只股票全量序列的指标中间量。
    说明：这些序列在任意 idx 处的值，等价于“截断到 idx 后再计算”的结果（递推/滚动均仅依赖过去）。
    """
    C = df['close']
    H = df['high']
    L = df['low']
    O = df['open']
    V = df['volume']

    cache: Dict[str, pd.Series] = {}

    # ---------------- MACD ----------------
    DIF = EMA(C, 12) - EMA(C, 26)
    DEA = EMA(DIF, 9)
    MACD = (DIF - DEA) * 2
    cache['macd_dif'] = DIF
    cache['macd_dea'] = DEA
    cache['macd_macd'] = MACD
    cache['macd_gc'] = CROSS(DIF, DEA)
    cache['macd_dc'] = CROSS(DEA, DIF)

    # ---------------- KDJ ----------------
    LOW9 = LLV(L, 9)
    HIGH9 = HHV(H, 9)
    RSV = (C - LOW9) / (HIGH9 - LOW9 + 0.0001) * 100
    K = SMA(RSV, 3, 1)
    D = SMA(K, 3, 1)
    J = 3 * K - 2 * D
    cache['kdj_k'] = K
    cache['kdj_d'] = D
    cache['kdj_j'] = J
    cache['kdj_gc'] = CROSS(K, D)
    cache['kdj_dc'] = CROSS(D, K)

    # ---------------- BOLL ----------------
    MID = MA(C, 20)
    STD = C.rolling(window=20).std()
    UPPER = MID + 2 * STD
    LOWER = MID - 2 * STD
    cache['boll_mid'] = MID
    cache['boll_upper'] = UPPER
    cache['boll_lower'] = LOWER

    # ---------------- RSI ----------------
    LC = REF(C, 1)

    def _rsi(n):
        return SMA(MAX(C - LC, 0), n, 1) / (SMA(ABS(C - LC), n, 1) + 0.0001) * 100

    RSI6 = _rsi(6)
    RSI12 = _rsi(12)
    RSI24 = _rsi(24)
    cache['rsi_6'] = RSI6
    cache['rsi_12'] = RSI12
    cache['rsi_24'] = RSI24

    # ---------------- DMI ----------------
    N = 14
    M = 6
    TR = pd.concat([H - L, (H - LC).abs(), (L - LC).abs()], axis=1).max(axis=1)

    HD = H - REF(H, 1)
    LD = REF(L, 1) - L

    DMP = IF((HD > 0) & (HD > LD), HD, pd.Series(0, index=df.index))
    DMM = IF((LD > 0) & (LD > HD), LD, pd.Series(0, index=df.index))

    TR_SUM = SMA(TR, N, 1)
    DMP_SUM = SMA(DMP, N, 1)
    DMM_SUM = SMA(DMM, N, 1)

    PDI = DMP_SUM / (TR_SUM + 0.0001) * 100
    MDI = DMM_SUM / (TR_SUM + 0.0001) * 100

    DX = ABS(PDI - MDI) / (PDI + MDI + 0.0001) * 100
    ADX = SMA(DX, M, 1)
    ADXR = (ADX + REF(ADX, M)) / 2

    cache['dmi_pdi'] = PDI
    cache['dmi_mdi'] = MDI
    cache['dmi_adx'] = ADX
    cache['dmi_adxr'] = ADXR
    cache['dmi_gc'] = CROSS(PDI, MDI)

    # ---------------- DMA ----------------
    DMA_DIF = MA(C, 10) - MA(C, 50)
    DMA_DIFMA = MA(DMA_DIF, 10)
    cache['dma_dif'] = DMA_DIF
    cache['dma_difma'] = DMA_DIFMA
    cache['dma_gc'] = CROSS(DMA_DIF, DMA_DIFMA)

    # ---------------- SAR (简化版，保持原判断逻辑) ----------------
    sar_period = 10
    recent_high = HHV(H, sar_period)
    recent_low = LLV(L, sar_period)
    mid_price = (recent_high + recent_low) / 2
    bullish = C > mid_price
    sar_val = pd.Series(np.where(bullish.values, recent_low.values, recent_high.values), index=df.index)
    cache['sar_val'] = sar_val
    cache['sar_bullish'] = bullish  # 多头状态标记
    cache['sar_signal'] = CROSS(C, sar_val)  # C 上穿/下穿 SAR 线

    # ---------------- BBI ----------------
    BBI = (MA(C, 3) + MA(C, 6) + MA(C, 12) + MA(C, 24)) / 4
    cache['bbi'] = BBI

    # ---------------- OBV ----------------
    OBV = (np.sign(C.diff()) * V).fillna(0).cumsum()
    cache['obv'] = OBV

    # ---------------- WR ----------------
    WR10 = (HHV(H, 10) - C) / (HHV(H, 10) - LLV(L, 10) + 0.0001) * 100
    WR6 = (HHV(H, 6) - C) / (HHV(H, 6) - LLV(L, 6) + 0.0001) * 100
    cache['wr_10'] = WR10
    cache['wr_6'] = WR6

    # ---------------- CCI ----------------
    cci_n = 14
    TP = (H + L + C) / 3
    MA_TP = MA(TP, cci_n)
    # 平均绝对偏差
    def _mean_abs_dev_raw(x):
        m = x.mean()
        return np.mean(np.abs(x - m))

    MD = TP.rolling(window=cci_n).apply(_mean_abs_dev_raw, raw=True)
    CCI = (TP - MA_TP) / (MD * 0.015 + 0.0001)
    cache['cci'] = CCI

    # 预计算完成
    return cache

# ==============================================================================
# 个案分析逻辑
# ==============================================================================

def _analyze_single_case(idx: int, stock_code: str, signal_date: pd.Timestamp, price_df: pd.DataFrame, feature_cache: Dict[str, pd.Series]) -> Tuple[int, Dict]:
    """
    分析单个成功案例，返回分析结果字典。
    """
    try:
        # 定位日期索引（用numpy searchsorted提高性能）
        dates = price_df['date'].values
        # 信号日期向下取整（防止时间戳不精确导致匹配失败）
        signal_dt = np.datetime64(signal_date)
        i = np.searchsorted(dates, signal_dt)
        if i < len(dates) and dates[i] == signal_dt:
            idx_in_df = i
        else:
            # 日期不在数据中
            return idx, {}

        # 读取指标缓存中的值（O(1)取值）
        result = {}
        # 价格类
        result['close'] = _safe_float(price_df.at[idx_in_df, 'close'])
        result['open'] = _safe_float(price_df.at[idx_in_df, 'open'])
        result['high'] = _safe_float(price_df.at[idx_in_df, 'high'])
        result['low'] = _safe_float(price_df.at[idx_in_df, 'low'])
        result['volume'] = _safe_float(price_df.at[idx_in_df, 'volume'])

        # 常用指标状态（True/False或数值）
        result['macd.dif_above_zero'] = _safe_bool(feature_cache['macd_dif'].iat[idx_in_df] > 0)
        result['macd.dea_above_zero'] = _safe_bool(feature_cache['macd_dea'].iat[idx_in_df] > 0)
        result['macd.macd_positive'] = _safe_bool(feature_cache['macd_macd'].iat[idx_in_df] > 0)
        result['kdj.k_above_d'] = _safe_bool(feature_cache['kdj_k'].iat[idx_in_df] > feature_cache['kdj_d'].iat[idx_in_df])
        result['kdj.j_above_100'] = _safe_bool(feature_cache['kdj_j'].iat[idx_in_df] > 100)
        result['boll.break_upper'] = _safe_bool(price_df.at[idx_in_df, 'close'] > feature_cache['boll_upper'].iat[idx_in_df])
        result['boll.break_lower'] = _safe_bool(price_df.at[idx_in_df, 'close'] < feature_cache['boll_lower'].iat[idx_in_df])
        result['rsi.6'] = _safe_float(feature_cache['rsi_6'].iat[idx_in_df])
        result['rsi.12'] = _safe_float(feature_cache['rsi_12'].iat[idx_in_df])
        result['rsi.24'] = _safe_float(feature_cache['rsi_24'].iat[idx_in_df])
        result['dmi.pdi_above_mdi'] = _safe_bool(feature_cache['dmi_pdi'].iat[idx_in_df] > feature_cache['dmi_mdi'].iat[idx_in_df])
        result['dmi.adx'] = _safe_float(feature_cache['dmi_adx'].iat[idx_in_df])
        result['dmi.adxr'] = _safe_float(feature_cache['dmi_adxr'].iat[idx_in_df])
        result['dma.dif_above_difma'] = _safe_bool(feature_cache['dma_dif'].iat[idx_in_df] > feature_cache['dma_difma'].iat[idx_in_df])
        result['sar.trend_up'] = _safe_bool(feature_cache['sar_bullish'].iat[idx_in_df])
        result['sar.break'] = _safe_bool(feature_cache['sar_signal'].iat[idx_in_df])
        result['bbi.position'] = _safe_bool(feature_cache['bbi'].iat[idx_in_df] < price_df.at[idx_in_df, 'close'])
        result['wr.10'] = _safe_float(feature_cache['wr_10'].iat[idx_in_df])
        result['wr.6'] = _safe_float(feature_cache['wr_6'].iat[idx_in_df])
        result['cci.value'] = _safe_float(feature_cache['cci'].iat[idx_in_df])

        # 市场理论判据
        # 道氏理论 - 顶/底分型（判断信号当日是否存在某种分型）
        result['theory.top_fractal'] = _safe_bool(price_df.at[idx_in_df, 'top_fractal']) if 'top_fractal' in price_df.columns else False
        result['theory.bottom_fractal'] = _safe_bool(price_df.at[idx_in_df, 'bottom_fractal']) if 'bottom_fractal' in price_df.columns else False
        # 威科夫理论 - Spring/Upthrust（以相对10日极值作为辅助判定）
        highest_10 = price_df['high'].rolling(window=10, min_periods=1).max().iat[idx_in_df]
        lowest_10 = price_df['low'].rolling(window=10, min_periods=1).min().iat[idx_in_df]
        prev_close = price_df.at[idx_in_df - 1, 'close'] if idx_in_df - 1 >= 0 else price_df.at[idx_in_df, 'open']
        result['theory.spring'] = _safe_bool(prev_close < lowest_10 * 1.02 and price_df.at[idx_in_df, 'close'] > lowest_10 * 1.02)
        result['theory.upthrust'] = _safe_bool(prev_close > highest_10 * 0.98 and price_df.at[idx_in_df, 'close'] < highest_10 * 0.98)

        # 均线系统判据
        # 5 日、10 日、20 日均线多头排列
        ma5 = price_df['close'].rolling(window=5, min_periods=1).mean().iat[idx_in_df]
        ma10 = price_df['close'].rolling(window=10, min_periods=1).mean().iat[idx_in_df]
        ma20 = price_df['close'].rolling(window=20, min_periods=1).mean().iat[idx_in_df]
        result['ma.goldencross_5_10_20'] = _safe_bool(ma5 > ma10 and ma10 > ma20)

        # 返回结果字典
        return idx, result
    except Exception as e:
        log(f"案例分析异常 - 股票: {stock_code}, 日期: {signal_date}, 错误: {e}", level="ERROR")
        return idx, {}

def analyze_stock_cases(item: Tuple[str, List[pd.Timestamp]]) -> List[Tuple[int, Dict]]:
    """
    分析单只股票的全部案例（供多进程调用）。
    返回值为该股票所有案例结果的列表，每项为 (原始列表索引, 结果dict)。
    """
    stock_code, dates = item
    try:
        filepath = None
        # 数据文件名有可能带市场前缀，如 SH600000.csv 或 SZ000001.csv
        # 支持两种文件命名格式
        for market in ['', 'SH', 'SZ']:
            trial_name = f"{market}{stock_code}.csv" if market else f"{stock_code}.csv"
            trial_path = os.path.join(DATA_DIR, trial_name)
            if os.path.exists(trial_path):
                filepath = trial_path
                break
        if filepath is None:
            log(f"数据文件缺失，跳过股票: {stock_code}", level="ERROR")
            return []

        # 读取股票数据
        try:
            df = pd.read_csv(
                filepath,
                encoding='utf-8-sig',
                usecols=lambda c: c in ['日期', 'open', 'high', 'low', 'close', 'volume'],
                dtype={'open': float, 'high': float, 'low': float, 'close': float, 'volume': float},
                parse_dates=['日期'],
            )
        except Exception:
            # 回退：兼容列不全/编码/解析异常
            df = pd.read_csv(filepath, encoding='gbk')
            df.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'}, inplace=True)
        else:
            df.rename(columns={'日期': 'date'}, inplace=True)
        if df.empty or len(df) < 10:
            log(f"股票 {stock_code} 数据不足，跳过", level="WARNING")
            return []

        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 标准化数据（确保列全，不影响性能）
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = 0.0

        # 预计算指标序列缓存
        cache = _precompute_feature_cache(df)
        df = df.assign(**{
            # 顶/底分型
            'top_fractal': (REF(df['high'], 1) > REF(df['high'], 2)) & (REF(df['high'], 1) > df['high']),
            'bottom_fractal': (REF(df['low'], 1) < REF(df['low'], 2)) & (REF(df['low'], 1) < df['low']),
        })
        df['top_fractal'] = df['top_fractal'].fillna(False)
        df['bottom_fractal'] = df['bottom_fractal'].fillna(False)

        # 顺序分析每个案例
        results = []
        for dt in dates:
            idx, res = _analyze_single_case(len(results), stock_code, dt, df, cache)
            results.append((idx, res))
        return results
    except Exception as e:
        log(f"股票案例分析失败 - 股票: {stock_code}, 错误: {e}", level="ERROR")
        return []

# ==============================================================================
# 主函数
# ==============================================================================

def main():
    # 读取成功案例列表
    try:
        df = pd.read_csv(SUCCESS_CASES_FILE, encoding='utf-8-sig', parse_dates=['date'])
    except Exception as e:
        log(f"无法读取成功案例文件: {e}", level="ERROR")
        return
    if df.empty:
        log("没有成功案例数据可供分析。")
        return

    # 按股票代码分组任务
    tasks = list(zip(df['stock_code'].tolist(), df['date'].tolist()))
    total_tasks = len(tasks)

    log(f"\n开始分析 {total_tasks} 个成功案例...")
    log("-" * 60)

    # 将任务按股票归组，以便单个股票的数据只加载一次
    stock_to_items: Dict[str, List[pd.Timestamp]] = {}
    for code, date in tasks:
        stock_to_items.setdefault(code, []).append(date)
    grouped_tasks = list(stock_to_items.items())

    # 多进程并行分析
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    sorted_results: List[Dict] = [{} for _ in range(total_tasks)]

    def analyze_case_wrapper(item):
        stock_code, dates = item
        results = analyze_stock_cases((stock_code, dates))
        return results

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        chunksize = max(1, len(grouped_tasks) // (10 * max_workers))
        for batch_results in executor.map(analyze_case_wrapper, grouped_tasks, chunksize=chunksize):
            # batch_results: [(idx, result), ...]
            for idx, result in batch_results:
                sorted_results[idx] = result if isinstance(result, dict) else {}

    # 构建 DataFrame 并保存报告
    report_df = pd.DataFrame(sorted_results)
    report_path = os.path.join(REPORT_DIR, 'pattern_analysis_report.csv')
    report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    log(f"\n分析报告已保存: {report_path}")

    # 统计模式摘要
    analysis_results = sorted_results  # 别名，更清晰
    summary = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_cases': len(df),
        'analyzed_cases': sum(1 for r in analysis_results if r),
        'indicators': {},
        'theories': {},
    }

    # 指标类特征统计
    indicator_fields = {
        'MACD': [
            ('macd.dif_above_zero', 'DIF在零轴上方'),
            ('macd.dea_above_zero', 'DEA在零轴上方'),
            ('macd.macd_positive', 'MACD柱线上升'),
        ],
        'KDJ': [
            ('kdj.k_above_d', 'K线上穿D线'),
            ('kdj.j_above_100', 'J值大于100'),
        ],
        'BOLL': [
            ('boll.break_upper', '向上突破BOLL上轨'),
            ('boll.break_lower', '跌破BOLL下轨'),
        ],
        'RSI': [
            ('rsi.6', 'RSI6数值'),
            ('rsi.12', 'RSI12数值'),
            ('rsi.24', 'RSI24数值'),
        ],
        'DMI': [
            ('dmi.pdi_above_mdi', 'PDI在MDI上方'),
            ('dmi.adx', 'ADX数值'),
            ('dmi.adxr', 'ADXR数值'),
        ],
        'DMA': [
            ('dma.dif_above_difma', 'DMA DIF上穿DIFMA'),
        ],
        'SAR': [
            ('sar.trend_up', '处于多头趋势'),
            ('sar.break', '价格穿越SAR线'),
        ],
        'BBI': [
            ('bbi.position', '收盘价在BBI上方'),
        ],
        'WR': [
            ('wr.10', 'WR(10)数值'),
            ('wr.6', 'WR(6)数值'),
        ],
        'CCI': [
            ('cci.value', 'CCI数值'),
        ],
    }
    for indicator, fields in indicator_fields.items():
        stats = {}
        for field_key, field_desc in fields:
            values = [r.get(field_key) for r in analysis_results if r]
            # 根据字段类型统计：布尔求真率，数值求均值/中位数
            if values and isinstance(values[0], bool):
                true_rate = round(sum(1 for v in values if v) / len(values) * 100, 1)
                stats[field_desc] = {'true_rate': true_rate}
            elif values and isinstance(values[0], (int, float)):
                # 排除nan计算均值/中位数
                numeric_vals = [v for v in values if v == v]
                if numeric_vals:
                    mean_val = round(float(pd.Series(numeric_vals).mean()), 2)
                    median_val = round(float(pd.Series(numeric_vals).median()), 2)
                    stats[field_desc] = {'mean': mean_val, 'median': median_val}
        if stats:
            summary['indicators'][indicator] = stats

    # 市场理论统计
    theory_fields = {
        'DowTheory': [
            ('theory.top_fractal', '出现顶分型'),
            ('theory.bottom_fractal', '出现底分型'),
        ],
        'WyckoffTheory': [
            ('theory.spring', '出现Spring'),
            ('theory.upthrust', '出现Upthrust'),
        ],
        'MA System': [
            ('ma.goldencross_5_10_20', '5/10/20日均线多头排列'),
        ],
    }
    for theory, fields in theory_fields.items():
        stats = {}
        for field_key, field_desc in fields:
            values = [r.get(field_key) for r in analysis_results if r]
            if values and isinstance(values[0], bool):
                true_rate = round(sum(1 for v in values if v) / len(values) * 100, 1)
                stats[field_desc] = {'true_rate': true_rate}
        if stats:
            summary['theories'][theory] = stats

    # 保存模式统计摘要 JSON
    summary_path = os.path.join(REPORT_DIR, 'pattern_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"模式统计摘要已保存: {summary_path}")

    # 按信号类型分别统计（成功案例按买点类型聚类）
    signal_types = sorted(df['signal_type'].unique()) if 'signal_type' in df.columns else []
    analysis_by_signal = {}
    for stype in signal_types:
        stype_results = [r for r, t in zip(analysis_results, df['signal_type']) if r and t == stype]
        if not stype_results:
            continue
        # 统计该信号类型下各特征
        stype_stats = {}
        for indicator, fields in indicator_fields.items():
            for field_key, field_desc in fields:
                values = [r.get(field_key) for r in stype_results if r]
                if values and isinstance(values[0], bool):
                    true_rate = round(sum(1 for v in values if v) / len(values) * 100, 1)
                    stype_stats.setdefault(indicator, {})[field_desc] = {'true_rate': true_rate}
                elif values and isinstance(values[0], (int, float)):
                    numeric_vals = [v for v in values if v == v]
                    if numeric_vals:
                        mean_val = round(float(pd.Series(numeric_vals).mean()), 2)
                        median_val = round(float(pd.Series(numeric_vals).median()), 2)
                        stype_stats.setdefault(indicator, {})[field_desc] = {'mean': mean_val, 'median': median_val}
        for theory, fields in theory_fields.items():
            for field_key, field_desc in fields:
                values = [r.get(field_key) for r in stype_results if r]
                if values and isinstance(values[0], bool):
                    true_rate = round(sum(1 for v in values if v) / len(values) * 100, 1)
                    stype_stats.setdefault(theory, {})[field_desc] = {'true_rate': true_rate}
        analysis_by_signal[stype] = stype_stats

    analysis_by_signal_path = os.path.join(REPORT_DIR, 'pattern_analysis_by_signal.json')
    with open(analysis_by_signal_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_by_signal, f, ensure_ascii=False, indent=2)
    log(f"分类统计已保存: {analysis_by_signal_path}")

    log("\n" + "=" * 60)
    log("分析完成!")
    log("=" * 60)

if __name__ == "__main__":
    main()
