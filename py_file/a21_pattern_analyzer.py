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
    RSV = (C - LLV(L, 9)) / (HHV(H, 9) - LLV(L, 9) + 0.0001) * 100
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
    TR = pd.concat([
        H - L,
        (H - REF(C, 1)).abs(),
        (L - REF(C, 1)).abs()
    ], axis=1).max(axis=1)

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
    cache['sar_bullish'] = bullish

    # ---------------- BBI ----------------
    BBI = (MA(C, 3) + MA(C, 6) + MA(C, 12) + MA(C, 24)) / 4
    cache['bbi'] = BBI

    # ---------------- OBV ----------------
    direction = np.sign((C - REF(C, 1)).values)
    if len(direction) > 0:
        direction[0] = 0
    OBV = pd.Series((direction * V.values).cumsum(), index=df.index)
    OBV_MA = MA(OBV, 20)
    cache['obv'] = OBV
    cache['obv_ma'] = OBV_MA

    # ---------------- WR ----------------
    WR10 = (HHV(H, 10) - C) / (HHV(H, 10) - LLV(L, 10) + 0.0001) * 100
    WR6 = (HHV(H, 6) - C) / (HHV(H, 6) - LLV(L, 6) + 0.0001) * 100
    cache['wr10'] = WR10
    cache['wr6'] = WR6

    # ---------------- CCI ----------------
    TP = (H + L + C) / 3
    cci_n = 14
    MA_TP = MA(TP, cci_n)

    def _mean_abs_dev_raw(x):
        m = x.mean()
        return np.mean(np.abs(x - m))

    MD = TP.rolling(window=cci_n).apply(_mean_abs_dev_raw, raw=True)
    CCI = (TP - MA_TP) / (MD * 0.015 + 0.0001)
    cache['cci'] = CCI

    # ---------------- MA System ----------------
    cache['ma5'] = MA(C, 5)
    cache['ma10'] = MA(C, 10)
    cache['ma20'] = MA(C, 20)
    cache['ma30'] = MA(C, 30)
    cache['ma60'] = MA(C, 60)
    cache['ma120'] = MA(C, 120)
    cache['ma_gc_5_10'] = CROSS(cache['ma5'], cache['ma10'])
    cache['ma_gc_10_20'] = CROSS(cache['ma10'], cache['ma20'])

    # ---------------- Dow Theory 辅助量 ----------------
    cache['dow_recent_high_20'] = HHV(H, 20)
    cache['dow_recent_low_20'] = LLV(L, 20)
    # “20-40日前窗口”的高低点：用rolling(20)后shift(20)等价于原逻辑切片 max/min
    cache['dow_prev_high_20_40'] = H.rolling(window=20).max().shift(20)
    cache['dow_prev_low_20_40'] = L.rolling(window=20).min().shift(20)

    # ---------------- Wyckoff 辅助量 ----------------
    cache['wyck_avg_vol_20'] = MA(V, 20)
    cache['wy_price_change'] = C - C.shift(1)
    cache['wy_daily_range'] = H - L
    cache['wy_avg_range_20'] = (H - L).rolling(20).mean()
    cache['wy_price_pos_60'] = (C - LLV(L, 60)) / (HHV(H, 60) - LLV(L, 60) + 0.0001)
    cache['wy_vol_ma5'] = MA(V, 5)
    cache['wy_vol_ma20'] = MA(V, 20)

    # 原始列也放入 cache 便于取值
    cache['close'] = C
    cache['open'] = O
    cache['high'] = H
    cache['low'] = L
    cache['volume'] = V

    return cache


def _extract_analysis_at_idx(cache: Dict[str, pd.Series], idx: int) -> Dict:
    """
    从预计算 cache 中在指定 idx 处抽取与原函数等价的结果字典结构。
    注意：保持原字段名、布尔判断、阈值与round精度不变。
    """
    # 等价于原：df_for_analysis = df.iloc[:idx+1]，len(df_for_analysis)<30则返回{}
    if idx + 1 < 30:
        return {}

    C = cache['close']
    H = cache['high']
    L = cache['low']
    V = cache['volume']

    # ========== MACD ==========
    dif = cache['macd_dif'].iloc[idx]
    dea = cache['macd_dea'].iloc[idx]
    macd = cache['macd_macd'].iloc[idx]
    macd_dict = {
        'dif': round(_safe_float(dif), 4),
        'dea': round(_safe_float(dea), 4),
        'macd': round(_safe_float(macd), 4),
        'golden_cross': _safe_bool(cache['macd_gc'].iloc[idx]) if idx > 0 else False,
        'death_cross': _safe_bool(cache['macd_dc'].iloc[idx]) if idx > 0 else False,
        'dif_above_zero': _safe_float(dif) > 0,
        'dea_above_zero': _safe_float(dea) > 0,
        'macd_positive': _safe_float(macd) > 0,
        'dif_rising': (_safe_float(dif) > _safe_float(cache['macd_dif'].iloc[idx - 1])) if idx > 0 else False,
    }

    # ========== KDJ ==========
    k = _safe_float(cache['kdj_k'].iloc[idx])
    d = _safe_float(cache['kdj_d'].iloc[idx])
    j = _safe_float(cache['kdj_j'].iloc[idx])
    kdj_dict = {
        'k': round(k, 2),
        'd': round(d, 2),
        'j': round(j, 2),
        'golden_cross': _safe_bool(cache['kdj_gc'].iloc[idx]) if idx > 0 else False,
        'death_cross': _safe_bool(cache['kdj_dc'].iloc[idx]) if idx > 0 else False,
        'oversold': k < 20 and d < 20,
        'overbought': k > 80 and d > 80,
        'j_oversold': j < 0,
        'j_overbought': j > 100,
        'k_above_d': k > d,
    }

    # ========== BOLL ==========
    mid = _safe_float(cache['boll_mid'].iloc[idx])
    upper = _safe_float(cache['boll_upper'].iloc[idx])
    lower = _safe_float(cache['boll_lower'].iloc[idx])
    current_close = _safe_float(C.iloc[idx])

    bandwidth = (upper - lower) / mid * 100 if mid and not np.isnan(mid) and mid > 0 else 0
    position = (current_close - lower) / (upper - lower) * 100 if (upper - lower) and not np.isnan(upper - lower) and (upper - lower) > 0 else 50

    boll_dict = {
        'mid': round(mid, 2) if not np.isnan(mid) else 0.0,
        'upper': round(upper, 2) if not np.isnan(upper) else 0.0,
        'lower': round(lower, 2) if not np.isnan(lower) else 0.0,
        'bandwidth': round(bandwidth, 2),
        'position': round(position, 2),
        'above_mid': current_close > mid if not np.isnan(mid) else False,
        'near_upper': position > 80,
        'near_lower': position < 20,
        'squeeze': bandwidth < 10,
        'expansion': bandwidth > 20,
    }

    # ========== RSI ==========
    rsi6 = _safe_float(cache['rsi_6'].iloc[idx])
    rsi12 = _safe_float(cache['rsi_12'].iloc[idx])
    rsi24 = _safe_float(cache['rsi_24'].iloc[idx])
    rsi_dict = {
        'rsi6': round(rsi6, 2),
        'rsi12': round(rsi12, 2),
        'rsi24': round(rsi24, 2),
        'oversold': rsi6 < 20,
        'overbought': rsi6 > 80,
        'rsi6_above_rsi12': rsi6 > rsi12,
        'rsi12_above_rsi24': rsi12 > rsi24,
        'bullish_alignment': (rsi6 > rsi12 > rsi24),
    }

    # ========== DMI ==========
    pdi = _safe_float(cache['dmi_pdi'].iloc[idx])
    mdi = _safe_float(cache['dmi_mdi'].iloc[idx])
    adx = _safe_float(cache['dmi_adx'].iloc[idx])
    dmi_dict = {
        'pdi': round(pdi, 2),
        'mdi': round(mdi, 2),
        'adx': round(adx, 2),
        'adxr': round(_safe_float(cache['dmi_adxr'].iloc[idx]), 2),
        'pdi_above_mdi': pdi > mdi,
        'strong_trend': adx > 25,
        'weak_trend': adx < 20,
        'golden_cross': _safe_bool(cache['dmi_gc'].iloc[idx]) if idx > 0 else False,
    }

    # ========== DMA ==========
    dma_dif = _safe_float(cache['dma_dif'].iloc[idx])
    dma_difma = _safe_float(cache['dma_difma'].iloc[idx])
    dma_dict = {
        'dif': round(dma_dif, 4),
        'difma': round(dma_difma, 4),
        'dif_above_zero': dma_dif > 0,
        'dif_above_difma': dma_dif > dma_difma,
        'golden_cross': _safe_bool(cache['dma_gc'].iloc[idx]) if idx > 0 else False,
        'rising': (dma_dif > _safe_float(cache['dma_dif'].iloc[idx - 1])) if idx > 0 else False,
    }

    # ========== SAR ==========
    if idx < 5:
        sar_dict = {'sar': 0, 'bullish': False, 'reversal': False}
    else:
        sar_val = _safe_float(cache['sar_val'].iloc[idx])
        sar_bullish = _safe_bool(cache['sar_bullish'].iloc[idx])
        sar_dict = {
            'sar': round(sar_val, 2),
            'bullish': sar_bullish,
            'price_above_sar': current_close > sar_val if sar_val and not np.isnan(sar_val) else False,
            'distance_pct': round((current_close - sar_val) / sar_val * 100, 2) if sar_val and not np.isnan(sar_val) and sar_val > 0 else 0,
        }

    # ========== BBI ==========
    bbi = _safe_float(cache['bbi'].iloc[idx])
    bbi_dict = {
        'bbi': round(bbi, 2),
        'price_above_bbi': current_close > bbi if not np.isnan(bbi) else False,
        'distance_pct': round((current_close - bbi) / bbi * 100, 2) if bbi and not np.isnan(bbi) and bbi > 0 else 0,
        'bbi_rising': (_safe_float(cache['bbi'].iloc[idx]) > _safe_float(cache['bbi'].iloc[idx - 1])) if idx > 0 else False,
    }

    # ========== OBV ==========
    obv = _safe_float(cache['obv'].iloc[idx])
    obv_ma = _safe_float(cache['obv_ma'].iloc[idx])
    if idx > 5:
        obv_trend = 'rising' if _safe_float(cache['obv'].iloc[idx]) > _safe_float(cache['obv'].iloc[idx - 5]) else 'falling'
        volume_price_sync = ((obv_trend == 'rising') == (_safe_float(C.iloc[idx]) > _safe_float(C.iloc[idx - 5])))
    else:
        obv_trend = 'falling'
        volume_price_sync = True
    obv_dict = {
        'obv': round(obv, 0),
        'obv_ma': round(obv_ma, 0),
        'obv_above_ma': obv > obv_ma,
        'obv_trend': obv_trend,
        'volume_price_sync': bool(volume_price_sync),
    }

    # ========== WR ==========
    wr10 = _safe_float(cache['wr10'].iloc[idx])
    wr6 = _safe_float(cache['wr6'].iloc[idx])
    wr_dict = {
        'wr10': round(wr10, 2),
        'wr6': round(wr6, 2),
        'oversold': wr10 > 80,
        'overbought': wr10 < 20,
        'wr6_above_wr10': wr6 > wr10,
    }

    # ========== CCI ==========
    cci = _safe_float(cache['cci'].iloc[idx])
    cci_dict = {
        'cci': round(cci, 2),
        'overbought': cci > 100,
        'oversold': cci < -100,
        'strong_bullish': cci > 200,
        'strong_bearish': cci < -200,
        'rising': (cci > _safe_float(cache['cci'].iloc[idx - 1])) if idx > 0 else False,
    }

    # ========== Dow Theory ==========
    # 原逻辑：主要趋势 MA60 / 次级 MA20 / 短期 MA5；高低点结构最近20日 vs 20-40日前窗口
    ma60 = cache['ma60']
    ma20 = cache['ma20']
    ma5 = cache['ma5']

    primary_trend = 'bullish' if _safe_float(C.iloc[idx]) > _safe_float(ma60.iloc[idx]) else 'bearish'
    secondary_trend = 'bullish' if _safe_float(C.iloc[idx]) > _safe_float(ma20.iloc[idx]) else 'bearish'
    short_trend = 'bullish' if _safe_float(C.iloc[idx]) > _safe_float(ma5.iloc[idx]) else 'bearish'

    recent_high = _safe_float(cache['dow_recent_high_20'].iloc[idx])
    recent_low = _safe_float(cache['dow_recent_low_20'].iloc[idx])

    if idx >= 40:
        prev_high = _safe_float(cache['dow_prev_high_20_40'].iloc[idx])
        prev_low = _safe_float(cache['dow_prev_low_20_40'].iloc[idx])
    else:
        prev_high = recent_high
        prev_low = recent_low

    higher_high = recent_high > prev_high
    higher_low = recent_low > prev_low
    lower_high = recent_high < prev_high
    lower_low = recent_low < prev_low

    uptrend_confirmed = higher_high and higher_low
    downtrend_confirmed = lower_high and lower_low

    dow_dict = {
        'primary_trend': primary_trend,
        'secondary_trend': secondary_trend,
        'short_trend': short_trend,
        'trend_alignment': (primary_trend == secondary_trend == short_trend),
        'higher_high': higher_high,
        'higher_low': higher_low,
        'lower_high': lower_high,
        'lower_low': lower_low,
        'uptrend_confirmed': uptrend_confirmed,
        'downtrend_confirmed': downtrend_confirmed,
        'recent_high': round(recent_high, 2),
        'recent_low': round(recent_low, 2),
    }

    # ========== Wyckoff Theory ==========
    if idx < 20:
        wy_dict = {'phase': 'unknown', 'volume_analysis': {}, 'price_action': {}}
    else:
        avg_vol_20 = _safe_float(cache['wyck_avg_vol_20'].iloc[idx])
        cur_vol = _safe_float(V.iloc[idx])
        volume_ratio = cur_vol / avg_vol_20 if avg_vol_20 and not np.isnan(avg_vol_20) and avg_vol_20 > 0 else 1

        price_change = _safe_float(cache['wy_price_change'].iloc[idx])
        prev_close = _safe_float(C.iloc[idx - 1]) if idx > 0 else float('nan')
        price_change_pct = (price_change / prev_close * 100) if prev_close and not np.isnan(prev_close) and prev_close > 0 else 0

        daily_range = _safe_float(cache['wy_daily_range'].iloc[idx])
        avg_range = _safe_float(cache['wy_avg_range_20'].iloc[idx])
        range_ratio = daily_range / avg_range if avg_range and not np.isnan(avg_range) and avg_range > 0 else 1

        cur_low = _safe_float(L.iloc[idx])
        close_position = ((current_close - cur_low) / daily_range) if daily_range and not np.isnan(daily_range) and daily_range > 0 else 0.5

        if price_change > 0 and volume_ratio > 1.2:
            supply_demand = 'strong_demand'
        elif price_change > 0 and volume_ratio < 0.8:
            supply_demand = 'weak_demand'
        elif price_change < 0 and volume_ratio > 1.2:
            supply_demand = 'strong_supply'
        elif price_change < 0 and volume_ratio < 0.8:
            supply_demand = 'weak_supply'
        else:
            supply_demand = 'balanced'

        price_position = _safe_float(cache['wy_price_pos_60'].iloc[idx])
        vol_ma5 = _safe_float(cache['wy_vol_ma5'].iloc[idx])
        vol_ma20 = _safe_float(cache['wy_vol_ma20'].iloc[idx])
        volume_expanding = vol_ma5 > vol_ma20

        if price_position < 0.3 and (not volume_expanding):
            phase = 'accumulation'
        elif price_position < 0.3 and volume_expanding:
            phase = 'markup_start'
        elif 0.3 <= price_position <= 0.7:
            phase = 'markup' if price_change > 0 else 'markdown'
        elif price_position > 0.7 and volume_expanding:
            phase = 'distribution'
        else:
            phase = 'markdown_start'

        wy_dict = {
            'phase': phase,
            'supply_demand': supply_demand,
            'volume_ratio': round(volume_ratio, 2),
            'volume_expanding': bool(volume_expanding),
            'price_position': round(price_position * 100, 2),
            'close_position': round(close_position * 100, 2),
            'range_ratio': round(range_ratio, 2),
            'price_change_pct': round(price_change_pct, 2),
        }

    # ========== MA System ==========
    ma_values = {
        'ma5': _safe_float(cache['ma5'].iloc[idx]),
        'ma10': _safe_float(cache['ma10'].iloc[idx]),
        'ma20': _safe_float(cache['ma20'].iloc[idx]),
        'ma30': _safe_float(cache['ma30'].iloc[idx]),
        'ma60': _safe_float(cache['ma60'].iloc[idx]) if idx >= 60 else np.nan,
        'ma120': _safe_float(cache['ma120'].iloc[idx]) if idx >= 120 else np.nan,
    }

    bullish_alignment = (
        (not np.isnan(ma_values['ma5'])) and (not np.isnan(ma_values['ma10'])) and
        (not np.isnan(ma_values['ma20'])) and (not np.isnan(ma_values['ma30'])) and
        (ma_values['ma5'] > ma_values['ma10'] > ma_values['ma20'] > ma_values['ma30'])
    )
    bearish_alignment = (
        (not np.isnan(ma_values['ma5'])) and (not np.isnan(ma_values['ma10'])) and
        (not np.isnan(ma_values['ma20'])) and (not np.isnan(ma_values['ma30'])) and
        (ma_values['ma5'] < ma_values['ma10'] < ma_values['ma20'] < ma_values['ma30'])
    )

    above_ma5 = current_close > ma_values['ma5']
    above_ma10 = current_close > ma_values['ma10']
    above_ma20 = current_close > ma_values['ma20']
    above_ma60 = current_close > ma_values['ma60'] if not np.isnan(ma_values['ma60']) else False

    ma_support_count = sum([above_ma5, above_ma10, above_ma20, above_ma60])

    ma_dict = {
        'ma_values': {k: (round(v, 2) if not np.isnan(v) else None) for k, v in ma_values.items()},
        'bullish_alignment': bool(bullish_alignment),
        'bearish_alignment': bool(bearish_alignment),
        'above_ma5': bool(above_ma5),
        'above_ma10': bool(above_ma10),
        'above_ma20': bool(above_ma20),
        'above_ma60': bool(above_ma60),
        'ma_support_count': int(ma_support_count),
        'golden_cross_5_10': _safe_bool(cache['ma_gc_5_10'].iloc[idx]) if idx > 0 else False,
        'golden_cross_10_20': _safe_bool(cache['ma_gc_10_20'].iloc[idx]) if idx > 0 else False,
    }

    return {
        'macd': macd_dict,
        'kdj': kdj_dict,
        'boll': boll_dict,
        'rsi': rsi_dict,
        'dmi': dmi_dict,
        'dma': dma_dict,
        'sar': sar_dict,
        'bbi': bbi_dict,
        'obv': obv_dict,
        'wr': wr_dict,
        'cci': cci_dict,
        'dow_theory': dow_dict,
        'wyckoff': wy_dict,
        'ma_system': ma_dict,
    }


# ==============================================================================
# 数据加载与信号定位（保持原口径）
# ==============================================================================

def _load_stock_dataframe(stock_code: str) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """加载并预处理单只股票的日线数据。

    返回:
        (df, dates_array)
        - df: 标准化后的DataFrame（已按date升序，reset_index）
        - dates_array: df['date'] 的 numpy datetime64[ns] 数组（用于二分定位）
    """
    parts = stock_code.split('_')
    if len(parts) != 2:
        return None, None

    market, code = parts
    filepath = os.path.join(DATA_DIR, market, f"{code}.csv")
    if not os.path.exists(filepath):
        return None, None

    required_cols_cn = {'名称', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率'}
    dtype_map = {
        '开盘': 'float32',
        '收盘': 'float32',
        '最高': 'float32',
        '最低': 'float32',
        '成交量': 'float64',   # 部分数据可能超int32，使用float64更稳妥（不改变逻辑，避免溢出）
        '成交额': 'float64',
        '振幅': 'float32',
        '涨跌幅': 'float32',
        '涨跌额': 'float32',
        '换手率': 'float32',
    }

    try:
        df = pd.read_csv(
            filepath,
            encoding='utf-8-sig',
            usecols=lambda c: c in required_cols_cn,
            dtype={k: v for k, v in dtype_map.items() if k in required_cols_cn},
            parse_dates=['日期'],
        )
    except Exception:
        # 回退：兼容列不全/编码/解析异常
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
        except Exception:
            return None, None

    column_mapping = {
        '名称': 'name',
        '日期': 'date',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'amount',
        '振幅': 'amplitude',
        '涨跌幅': 'pct_change',
        '涨跌额': 'change',
        '换手率': 'turnover'
    }
    df = df.rename(columns=column_mapping)

    if 'date' not in df.columns:
        return None, None

    # 确保日期类型
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # 关键列检查（分析必须字段）
    must_cols = ['open', 'close', 'high', 'low', 'volume']
    for c in must_cols:
        if c not in df.columns:
            return None, None

    # 排序与重建索引
    df = df.sort_values('date').reset_index(drop=True)
    if df.empty:
        return None, None

    dates = df['date'].values  # datetime64[ns]
    return df, dates


def _locate_signal_index(dates: np.ndarray, signal_date: str) -> Optional[int]:
    """用二分查找定位信号日期索引（等价于原先全表扫描，但更快）。"""
    try:
        signal_dt = np.datetime64(pd.to_datetime(signal_date))
    except Exception:
        return None

    idx = int(np.searchsorted(dates, signal_dt))
    if idx < len(dates) and dates[idx] == signal_dt:
        return idx
    return None


# ==============================================================================
# 对外分析接口（保持不变）
# ==============================================================================

def analyze_single_case(stock_code: str, signal_date: str) -> Dict:
    """分析单个成功案例（保留原接口）。"""
    try:
        df, dates = _load_stock_dataframe(stock_code)
        if df is None or dates is None:
            return {}

        signal_idx = _locate_signal_index(dates, signal_date)
        if signal_idx is None:
            return {}

        cache = _precompute_feature_cache(df)
        return _extract_analysis_at_idx(cache, signal_idx)

    except Exception as e:
        log(f"分析失败 {stock_code} {signal_date}: {str(e)}")
        return {}


def analyze_stock_cases(stock_code: str, indexed_dates: List[Tuple[int, str]]) -> List[Tuple[int, Dict]]:
    """同一只股票的多条信号批量分析。

    目的：避免对同一股票重复读CSV与重复预处理；并进一步避免对同一股票每条信号重复计算指标序列。
    """
    try:
        df, dates = _load_stock_dataframe(stock_code)
        if df is None or dates is None:
            return [(i, {}) for i, _ in indexed_dates]

        # 预计算整只股票指标序列一次
        cache = _precompute_feature_cache(df)

        out: List[Tuple[int, Dict]] = []
        for i, d in indexed_dates:
            signal_idx = _locate_signal_index(dates, d)
            if signal_idx is None:
                out.append((i, {}))
                continue
            out.append((i, _extract_analysis_at_idx(cache, signal_idx)))
        return out

    except Exception as e:
        log(f"批量分析失败 {stock_code}: {str(e)}")
        return [(i, {}) for i, _ in indexed_dates]


def analyze_case_wrapper(args):
    """多进程包装函数（按股票粒度批量处理）。"""
    stock_code, indexed_dates = args
    return analyze_stock_cases(stock_code, indexed_dates)


# ==============================================================================
# 统计分析函数（保持原逻辑）
# ==============================================================================

def calculate_statistics(analysis_results: List[Dict], field_path: str) -> Dict:
    values = []
    for result in analysis_results:
        if not result:
            continue
        parts = field_path.split('.')
        value = result
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = None
                break
        if value is not None:
            values.append(value)

    if not values:
        return {'count': 0}

    if isinstance(values[0], bool):
        true_count = sum(values)
        return {
            'count': len(values),
            'true_count': true_count,
            'true_rate': round(true_count / len(values) * 100, 2),
        }
    elif isinstance(values[0], (int, float)):
        return {
            'count': len(values),
            'mean': round(np.mean(values), 4),
            'median': round(np.median(values), 4),
            'std': round(np.std(values), 4),
            'min': round(min(values), 4),
            'max': round(max(values), 4),
            'q25': round(np.percentile(values, 25), 4),
            'q75': round(np.percentile(values, 75), 4),
        }
    elif isinstance(values[0], str):
        from collections import Counter
        counter = Counter(values)
        return {
            'count': len(values),
            'distribution': dict(counter.most_common()),
        }

    return {'count': len(values)}


def generate_pattern_summary(df: pd.DataFrame, analysis_results: List[Dict]) -> Dict:
    summary = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_cases': len(df),
        'analyzed_cases': len([r for r in analysis_results if r]),
        'indicators': {},
        'theories': {},
    }

    indicator_fields = {
        'MACD': [
            ('macd.dif_above_zero', 'DIF在零轴上方'),
            ('macd.dea_above_zero', 'DEA在零轴上方'),
            ('macd.macd_positive', 'MACD柱为正'),
            ('macd.golden_cross', '金叉'),
            ('macd.dif_rising', 'DIF上升'),
        ],
        'KDJ': [
            ('kdj.k_above_d', 'K线在D线上方'),
            ('kdj.oversold', '超卖区'),
            ('kdj.overbought', '超买区'),
            ('kdj.golden_cross', '金叉'),
            ('kdj.k', 'K值'),
            ('kdj.d', 'D值'),
        ],
        'BOLL': [
            ('boll.above_mid', '价格在中轨上方'),
            ('boll.near_lower', '接近下轨'),
            ('boll.near_upper', '接近上轨'),
            ('boll.squeeze', '布林带收窄'),
            ('boll.position', '布林带位置'),
        ],
        'RSI': [
            ('rsi.oversold', 'RSI超卖'),
            ('rsi.overbought', 'RSI超买'),
            ('rsi.bullish_alignment', '多头排列'),
            ('rsi.rsi6', 'RSI6值'),
        ],
        'DMI': [
            ('dmi.pdi_above_mdi', 'PDI>MDI'),
            ('dmi.strong_trend', '强趋势'),
            ('dmi.weak_trend', '弱趋势'),
            ('dmi.adx', 'ADX值'),
        ],
        'DMA': [
            ('dma.dif_above_zero', 'DIF>0'),
            ('dma.dif_above_difma', 'DIF>DIFMA'),
            ('dma.golden_cross', '金叉'),
        ],
        'SAR': [
            ('sar.bullish', '多头'),
            ('sar.price_above_sar', '价格在SAR上方'),
        ],
        'BBI': [
            ('bbi.price_above_bbi', '价格在BBI上方'),
            ('bbi.bbi_rising', 'BBI上升'),
        ],
        'OBV': [
            ('obv.obv_above_ma', 'OBV在均线上方'),
            ('obv.obv_trend', 'OBV趋势'),
            ('obv.volume_price_sync', '量价同步'),
        ],
        'WR': [
            ('wr.oversold', 'WR超卖'),
            ('wr.overbought', 'WR超买'),
        ],
        'CCI': [
            ('cci.oversold', 'CCI超卖'),
            ('cci.overbought', 'CCI超买'),
            ('cci.rising', 'CCI上升'),
            ('cci.cci', 'CCI值'),
        ],
    }

    for indicator_name, fields in indicator_fields.items():
        summary['indicators'][indicator_name] = {}
        for field_path, field_name in fields:
            stats = calculate_statistics(analysis_results, field_path)
            summary['indicators'][indicator_name][field_name] = stats

    theory_fields = {
        '道氏理论': [
            ('dow_theory.primary_trend', '主要趋势'),
            ('dow_theory.secondary_trend', '次级趋势'),
            ('dow_theory.trend_alignment', '趋势一致'),
            ('dow_theory.uptrend_confirmed', '上升趋势确认'),
            ('dow_theory.higher_high', '更高的高点'),
            ('dow_theory.higher_low', '更高的低点'),
        ],
        '威科夫理论': [
            ('wyckoff.phase', '市场阶段'),
            ('wyckoff.supply_demand', '供需关系'),
            ('wyckoff.volume_expanding', '成交量扩张'),
            ('wyckoff.volume_ratio', '成交量比率'),
            ('wyckoff.price_position', '价格位置'),
        ],
        '均线系统': [
            ('ma_system.bullish_alignment', '多头排列'),
            ('ma_system.bearish_alignment', '空头排列'),
            ('ma_system.above_ma20', '价格在MA20上方'),
            ('ma_system.above_ma60', '价格在MA60上方'),
            ('ma_system.ma_support_count', '均线支撑数量'),
            ('ma_system.golden_cross_5_10', 'MA5上穿MA10'),
        ],
    }

    for theory_name, fields in theory_fields.items():
        summary['theories'][theory_name] = {}
        for field_path, field_name in fields:
            stats = calculate_statistics(analysis_results, field_path)
            summary['theories'][theory_name][field_name] = stats

    return summary


def generate_signal_type_summary(df: pd.DataFrame, analysis_results: List[Dict]) -> Dict:
    df_with_analysis = df.copy()
    df_with_analysis['analysis'] = analysis_results

    signal_summary = {}
    for signal_type in df['signal_type'].unique():
        type_df = df_with_analysis[df_with_analysis['signal_type'] == signal_type]
        type_analysis = [r for r in type_df['analysis'].tolist() if r]
        if not type_analysis:
            continue

        signal_summary[signal_type] = {
            'total_cases': len(type_df),
            'analyzed_cases': len(type_analysis),
            'key_patterns': {},
        }

        key_fields = [
            ('macd.dif_above_zero', 'MACD_DIF>0'),
            ('macd.golden_cross', 'MACD金叉'),
            ('kdj.k_above_d', 'KDJ_K>D'),
            ('kdj.oversold', 'KDJ超卖'),
            ('boll.above_mid', 'BOLL中轨上方'),
            ('boll.near_lower', 'BOLL下轨附近'),
            ('rsi.oversold', 'RSI超卖'),
            ('rsi.bullish_alignment', 'RSI多头排列'),
            ('dmi.pdi_above_mdi', 'DMI多头'),
            ('dmi.strong_trend', 'DMI强趋势'),
            ('obv.volume_price_sync', 'OBV量价同步'),
            ('dow_theory.uptrend_confirmed', '道氏上升趋势'),
            ('dow_theory.trend_alignment', '道氏趋势一致'),
            ('wyckoff.phase', '威科夫阶段'),
            ('wyckoff.supply_demand', '威科夫供需'),
            ('ma_system.bullish_alignment', '均线多头排列'),
            ('ma_system.above_ma20', 'MA20上方'),
        ]

        for field_path, field_name in key_fields:
            stats = calculate_statistics(type_analysis, field_path)
            signal_summary[signal_type]['key_patterns'][field_name] = stats

    return signal_summary


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    log("=" * 60)
    log("成功案例模式分析器")
    log("=" * 60)
    log(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    if not os.path.exists(SUCCESS_CASES_FILE):
        log(f"错误: 成功案例文件不存在: {SUCCESS_CASES_FILE}")
        log("请先运行 signal_success_scanner.py 生成成功案例数据")
        return

    df = pd.read_csv(SUCCESS_CASES_FILE, encoding='utf-8-sig')
    log(f"加载成功案例: {len(df)} 条")
    if df.empty:
        log("没有成功案例可分析")
        return

    tasks = list(zip(df['stock_code'].tolist(), df['date'].tolist()))
    total_tasks = len(tasks)

    log(f"\n开始分析 {total_tasks} 个成功案例...")
    log("-" * 60)

    # 按股票分组：同一只股票只加载一次CSV；并在该进程内预计算指标序列一次
    stock_to_items: Dict[str, List[Tuple[int, str]]] = {}
    for i, (stock_code, signal_date) in enumerate(tasks):
        stock_to_items.setdefault(stock_code, []).append((i, signal_date))
    grouped_tasks = list(stock_to_items.items())

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    sorted_results: List[Dict] = [{} for _ in range(total_tasks)]

    # ProcessPoolExecutor.map 支持 chunksize，可降低调度开销（不改变逻辑）
    chunksize = 8 if len(grouped_tasks) >= 200 else 1

    completed_cases = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch_results in executor.map(analyze_case_wrapper, grouped_tasks, chunksize=chunksize):
            # batch_results: [(idx, result), ...]
            for idx, result in batch_results:
                sorted_results[idx] = result if isinstance(result, dict) else {}
            completed_cases += len(batch_results)

            if completed_cases % 100 == 0 or completed_cases >= total_tasks:
                log(f"进度: {completed_cases}/{total_tasks} ({completed_cases/total_tasks*100:.1f}%)")

    analysis_results = sorted_results
    log(f"\n分析完成，成功分析 {len([r for r in analysis_results if r])} 个案例")

    log("\n生成分析报告...")

    flat_results = []
    # 保持原回填顺序
    for i, (_, row) in enumerate(df.iterrows()):
        result = analysis_results[i] if i < len(analysis_results) else {}

        flat_record = {
            'stock_code': row['stock_code'],
            'stock_name': row['stock_name'],
            'date': row['date'],
            'signal_type': row['signal_type'],
            'signal_detail': row.get('signal_detail', ''),
            'max_return_pct': row['max_return_pct'],
            'final_return_pct': row['final_return_pct'],
        }

        if result:
            for indicator, values in result.items():
                if isinstance(values, dict):
                    for key, val in values.items():
                        if isinstance(val, dict):
                            for k2, v2 in val.items():
                                flat_record[f'{indicator}_{key}_{k2}'] = v2
                        else:
                            flat_record[f'{indicator}_{key}'] = val

        flat_results.append(flat_record)

    report_df = pd.DataFrame(flat_results)
    report_path = os.path.join(REPORT_DIR, 'pattern_analysis_report.csv')
    report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    log(f"详细分析报告已保存: {report_path}")

    os.makedirs(WEB_DATA_DIR, exist_ok=True)

    summary = generate_pattern_summary(df, analysis_results)

    summary_path = os.path.join(REPORT_DIR, 'pattern_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"模式统计摘要已保存: {summary_path}")

    web_summary_path = os.path.join(WEB_DATA_DIR, 'pattern_summary.json')
    with open(web_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"模式统计摘要已保存到Web目录: {web_summary_path}")

    signal_summary = generate_signal_type_summary(df, analysis_results)

    signal_summary_path = os.path.join(REPORT_DIR, 'pattern_analysis_by_signal.json')
    with open(signal_summary_path, 'w', encoding='utf-8') as f:
        json.dump(signal_summary, f, ensure_ascii=False, indent=2)
    log(f"按信号类型统计已保存: {signal_summary_path}")

    web_signal_summary_path = os.path.join(WEB_DATA_DIR, 'pattern_analysis_by_signal.json')
    with open(web_signal_summary_path, 'w', encoding='utf-8') as f:
        json.dump(signal_summary, f, ensure_ascii=False, indent=2)
    log(f"按信号类型统计已保存到Web目录: {web_signal_summary_path}")

    # 打印关键发现（保持原输出口径）
    log("\n" + "=" * 60)
    log("关键发现摘要")
    log("=" * 60)

    log(f"\n总分析案例: {summary['analyzed_cases']}")

    log("\n【技术指标共性特征】")
    log("-" * 60)

    for indicator, stats in summary['indicators'].items():
        log(f"\n{indicator}:")
        for field_name, field_stats in stats.items():
            if 'true_rate' in field_stats:
                log(f"  {field_name}: {field_stats['true_rate']}% 的案例满足")
            elif 'mean' in field_stats:
                log(f"  {field_name}: 均值={field_stats['mean']}, 中位数={field_stats['median']}")
            elif 'distribution' in field_stats:
                log(f"  {field_name}: {field_stats['distribution']}")

    log("\n【市场理论分析】")
    log("-" * 60)

    for theory, stats in summary['theories'].items():
        log(f"\n{theory}:")
        for field_name, field_stats in stats.items():
            if 'true_rate' in field_stats:
                log(f"  {field_name}: {field_stats['true_rate']}% 的案例满足")
            elif 'distribution' in field_stats:
                log(f"  {field_name}: {field_stats['distribution']}")
            elif 'mean' in field_stats:
                log(f"  {field_name}: 均值={field_stats['mean']}")

    log("\n" + "=" * 60)
    log("分析完成!")
    log("=" * 60)


if __name__ == "__main__":
    main()
