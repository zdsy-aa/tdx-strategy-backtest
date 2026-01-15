#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
成功案例模式分析器 (pattern_analyzer.py)
================================================================================

功能说明:
    本脚本用于分析成功买入信号案例的共性特征：
    1. 计算成功案例在信号触发时的多种技术指标状态
    2. 应用道氏理论、威科夫理论的相关判据
    3. 按买点类型分类统计共性特征
    4. 生成详细的分析报告

分析维度:
    - 常用指标：MACD, KDJ, BOLL, RSI, DMI, DMA, SAR, BBI, OBV等
    - 市场理论：道氏理论、威科夫理论的相关判据

输入:
    - report/signal_success_cases.csv: 成功案例数据
    - data/day/ 目录下的股票CSV文件

输出:
    - report/pattern_analysis_report.csv: 详细分析报告
    - report/pattern_summary.json: 模式统计摘要
    - report/pattern_analysis_by_signal.json: 按信号类型分类的统计

作者: TradeGuide System
版本: 1.0.0
创建日期: 2026-01-15
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入基础函数
from indicators import (
    REF, MA, EMA, SMA, HHV, LLV, CROSS, COUNT, ABS, MAX, IF
)


# ==============================================================================
# 配置常量
# ==============================================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')

# 报告目录
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')

# 成功案例文件
SUCCESS_CASES_FILE = os.path.join(REPORT_DIR, 'signal_success_cases.csv')


# ==============================================================================
# 技术指标计算函数
# ==============================================================================

def calculate_macd(df: pd.DataFrame) -> Dict:
    """
    计算MACD指标状态
    
    返回:
        Dict: MACD指标状态
            - dif: DIF值
            - dea: DEA值
            - macd: MACD柱值
            - golden_cross: 是否金叉
            - death_cross: 是否死叉
            - above_zero: DIF是否在零轴上方
            - trend: 趋势方向 (上升/下降/震荡)
    """
    C = df['close']
    
    # 标准MACD参数 (12, 26, 9)
    DIF = EMA(C, 12) - EMA(C, 26)
    DEA = EMA(DIF, 9)
    MACD = (DIF - DEA) * 2
    
    current_idx = len(df) - 1
    
    return {
        'dif': round(float(DIF.iloc[current_idx]), 4),
        'dea': round(float(DEA.iloc[current_idx]), 4),
        'macd': round(float(MACD.iloc[current_idx]), 4),
        'golden_cross': bool(CROSS(DIF, DEA).iloc[current_idx]) if current_idx > 0 else False,
        'death_cross': bool(CROSS(DEA, DIF).iloc[current_idx]) if current_idx > 0 else False,
        'dif_above_zero': bool(DIF.iloc[current_idx] > 0),
        'dea_above_zero': bool(DEA.iloc[current_idx] > 0),
        'macd_positive': bool(MACD.iloc[current_idx] > 0),
        'dif_rising': bool(DIF.iloc[current_idx] > DIF.iloc[current_idx-1]) if current_idx > 0 else False,
    }


def calculate_kdj(df: pd.DataFrame) -> Dict:
    """
    计算KDJ指标状态
    
    返回:
        Dict: KDJ指标状态
    """
    C = df['close']
    H = df['high']
    L = df['low']
    
    # 标准KDJ参数 (9, 3, 3)
    RSV = (C - LLV(L, 9)) / (HHV(H, 9) - LLV(L, 9) + 0.0001) * 100
    K = SMA(RSV, 3, 1)
    D = SMA(K, 3, 1)
    J = 3 * K - 2 * D
    
    current_idx = len(df) - 1
    
    k_val = float(K.iloc[current_idx])
    d_val = float(D.iloc[current_idx])
    j_val = float(J.iloc[current_idx])
    
    return {
        'k': round(k_val, 2),
        'd': round(d_val, 2),
        'j': round(j_val, 2),
        'golden_cross': bool(CROSS(K, D).iloc[current_idx]) if current_idx > 0 else False,
        'death_cross': bool(CROSS(D, K).iloc[current_idx]) if current_idx > 0 else False,
        'oversold': k_val < 20 and d_val < 20,  # 超卖区
        'overbought': k_val > 80 and d_val > 80,  # 超买区
        'j_oversold': j_val < 0,  # J值超卖
        'j_overbought': j_val > 100,  # J值超买
        'k_above_d': k_val > d_val,
    }


def calculate_boll(df: pd.DataFrame) -> Dict:
    """
    计算布林带指标状态
    
    返回:
        Dict: BOLL指标状态
    """
    C = df['close']
    
    # 标准布林带参数 (20, 2)
    MID = MA(C, 20)
    STD = C.rolling(window=20).std()
    UPPER = MID + 2 * STD
    LOWER = MID - 2 * STD
    
    current_idx = len(df) - 1
    current_close = float(C.iloc[current_idx])
    mid_val = float(MID.iloc[current_idx])
    upper_val = float(UPPER.iloc[current_idx])
    lower_val = float(LOWER.iloc[current_idx])
    
    # 计算带宽
    bandwidth = (upper_val - lower_val) / mid_val * 100 if mid_val > 0 else 0
    
    # 计算价格在布林带中的位置 (0-100)
    position = (current_close - lower_val) / (upper_val - lower_val) * 100 if (upper_val - lower_val) > 0 else 50
    
    return {
        'mid': round(mid_val, 2),
        'upper': round(upper_val, 2),
        'lower': round(lower_val, 2),
        'bandwidth': round(bandwidth, 2),
        'position': round(position, 2),
        'above_mid': current_close > mid_val,
        'near_upper': position > 80,
        'near_lower': position < 20,
        'squeeze': bandwidth < 10,  # 布林带收窄
        'expansion': bandwidth > 20,  # 布林带扩张
    }


def calculate_rsi(df: pd.DataFrame) -> Dict:
    """
    计算RSI指标状态
    
    返回:
        Dict: RSI指标状态
    """
    C = df['close']
    LC = REF(C, 1)
    
    # RSI6, RSI12, RSI24
    def calc_rsi(n):
        return SMA(MAX(C - LC, 0), n, 1) / (SMA(ABS(C - LC), n, 1) + 0.0001) * 100
    
    RSI6 = calc_rsi(6)
    RSI12 = calc_rsi(12)
    RSI24 = calc_rsi(24)
    
    current_idx = len(df) - 1
    
    rsi6_val = float(RSI6.iloc[current_idx])
    rsi12_val = float(RSI12.iloc[current_idx])
    rsi24_val = float(RSI24.iloc[current_idx])
    
    return {
        'rsi6': round(rsi6_val, 2),
        'rsi12': round(rsi12_val, 2),
        'rsi24': round(rsi24_val, 2),
        'oversold': rsi6_val < 20,
        'overbought': rsi6_val > 80,
        'rsi6_above_rsi12': rsi6_val > rsi12_val,
        'rsi12_above_rsi24': rsi12_val > rsi24_val,
        'bullish_alignment': rsi6_val > rsi12_val > rsi24_val,  # 多头排列
    }


def calculate_dmi(df: pd.DataFrame) -> Dict:
    """
    计算DMI指标状态 (趋向指标)
    
    返回:
        Dict: DMI指标状态
    """
    C = df['close']
    H = df['high']
    L = df['low']
    
    N = 14
    M = 6
    
    # 计算TR (真实波幅)
    TR = pd.concat([
        H - L,
        (H - REF(C, 1)).abs(),
        (L - REF(C, 1)).abs()
    ], axis=1).max(axis=1)
    
    # 计算方向变动
    HD = H - REF(H, 1)
    LD = REF(L, 1) - L
    
    DMP = IF((HD > 0) & (HD > LD), HD, pd.Series(0, index=df.index))
    DMM = IF((LD > 0) & (LD > HD), LD, pd.Series(0, index=df.index))
    
    # 平滑计算
    TR_SUM = SMA(TR, N, 1)
    DMP_SUM = SMA(DMP, N, 1)
    DMM_SUM = SMA(DMM, N, 1)
    
    PDI = DMP_SUM / (TR_SUM + 0.0001) * 100
    MDI = DMM_SUM / (TR_SUM + 0.0001) * 100
    
    # ADX计算
    DX = ABS(PDI - MDI) / (PDI + MDI + 0.0001) * 100
    ADX = SMA(DX, M, 1)
    ADXR = (ADX + REF(ADX, M)) / 2
    
    current_idx = len(df) - 1
    
    pdi_val = float(PDI.iloc[current_idx])
    mdi_val = float(MDI.iloc[current_idx])
    adx_val = float(ADX.iloc[current_idx])
    
    return {
        'pdi': round(pdi_val, 2),
        'mdi': round(mdi_val, 2),
        'adx': round(adx_val, 2),
        'adxr': round(float(ADXR.iloc[current_idx]), 2),
        'pdi_above_mdi': pdi_val > mdi_val,  # 多头趋势
        'strong_trend': adx_val > 25,  # 强趋势
        'weak_trend': adx_val < 20,  # 弱趋势/震荡
        'golden_cross': bool(CROSS(PDI, MDI).iloc[current_idx]) if current_idx > 0 else False,
    }


def calculate_dma(df: pd.DataFrame) -> Dict:
    """
    计算DMA指标状态 (平均线差)
    
    返回:
        Dict: DMA指标状态
    """
    C = df['close']
    
    # DMA参数
    DIF = MA(C, 10) - MA(C, 50)
    DIFMA = MA(DIF, 10)
    
    current_idx = len(df) - 1
    
    dif_val = float(DIF.iloc[current_idx])
    difma_val = float(DIFMA.iloc[current_idx])
    
    return {
        'dif': round(dif_val, 4),
        'difma': round(difma_val, 4),
        'dif_above_zero': dif_val > 0,
        'dif_above_difma': dif_val > difma_val,
        'golden_cross': bool(CROSS(DIF, DIFMA).iloc[current_idx]) if current_idx > 0 else False,
        'rising': bool(DIF.iloc[current_idx] > DIF.iloc[current_idx-1]) if current_idx > 0 else False,
    }


def calculate_sar(df: pd.DataFrame) -> Dict:
    """
    计算SAR指标状态 (抛物线转向)
    
    返回:
        Dict: SAR指标状态
    """
    H = df['high']
    L = df['low']
    C = df['close']
    
    # 简化版SAR计算
    n = len(df)
    sar = pd.Series(index=df.index, dtype=float)
    
    # 初始化
    af = 0.02
    af_max = 0.2
    af_step = 0.02
    
    # 判断初始趋势
    if n < 5:
        return {'sar': 0, 'bullish': False, 'reversal': False}
    
    # 使用简化逻辑：SAR基于最近N日的极值
    sar_period = 10
    
    # 上升趋势SAR = 最近N日最低价
    # 下降趋势SAR = 最近N日最高价
    
    current_idx = n - 1
    recent_high = float(HHV(H, sar_period).iloc[current_idx])
    recent_low = float(LLV(L, sar_period).iloc[current_idx])
    current_close = float(C.iloc[current_idx])
    
    # 判断趋势
    mid_price = (recent_high + recent_low) / 2
    bullish = current_close > mid_price
    
    sar_val = recent_low if bullish else recent_high
    
    return {
        'sar': round(sar_val, 2),
        'bullish': bullish,
        'price_above_sar': current_close > sar_val,
        'distance_pct': round((current_close - sar_val) / sar_val * 100, 2) if sar_val > 0 else 0,
    }


def calculate_bbi(df: pd.DataFrame) -> Dict:
    """
    计算BBI指标状态 (多空指数)
    
    返回:
        Dict: BBI指标状态
    """
    C = df['close']
    
    # BBI = (MA3 + MA6 + MA12 + MA24) / 4
    BBI = (MA(C, 3) + MA(C, 6) + MA(C, 12) + MA(C, 24)) / 4
    
    current_idx = len(df) - 1
    bbi_val = float(BBI.iloc[current_idx])
    current_close = float(C.iloc[current_idx])
    
    return {
        'bbi': round(bbi_val, 2),
        'price_above_bbi': current_close > bbi_val,
        'distance_pct': round((current_close - bbi_val) / bbi_val * 100, 2) if bbi_val > 0 else 0,
        'bbi_rising': bool(BBI.iloc[current_idx] > BBI.iloc[current_idx-1]) if current_idx > 0 else False,
    }


def calculate_obv(df: pd.DataFrame) -> Dict:
    """
    计算OBV指标状态 (能量潮)
    
    返回:
        Dict: OBV指标状态
    """
    C = df['close']
    V = df['volume']
    
    # OBV计算
    direction = np.sign(C - REF(C, 1))
    direction.iloc[0] = 0
    
    OBV = (direction * V).cumsum()
    OBV_MA = MA(OBV, 20)
    
    current_idx = len(df) - 1
    obv_val = float(OBV.iloc[current_idx])
    obv_ma_val = float(OBV_MA.iloc[current_idx])
    
    # OBV趋势
    obv_trend = 'rising' if current_idx > 5 and OBV.iloc[current_idx] > OBV.iloc[current_idx-5] else 'falling'
    
    return {
        'obv': round(obv_val, 0),
        'obv_ma': round(obv_ma_val, 0),
        'obv_above_ma': obv_val > obv_ma_val,
        'obv_trend': obv_trend,
        'volume_price_sync': (obv_trend == 'rising') == (C.iloc[current_idx] > C.iloc[current_idx-5]) if current_idx > 5 else True,
    }


def calculate_wr(df: pd.DataFrame) -> Dict:
    """
    计算威廉指标状态
    
    返回:
        Dict: WR指标状态
    """
    C = df['close']
    H = df['high']
    L = df['low']
    
    # WR10, WR6
    WR10 = (HHV(H, 10) - C) / (HHV(H, 10) - LLV(L, 10) + 0.0001) * 100
    WR6 = (HHV(H, 6) - C) / (HHV(H, 6) - LLV(L, 6) + 0.0001) * 100
    
    current_idx = len(df) - 1
    
    wr10_val = float(WR10.iloc[current_idx])
    wr6_val = float(WR6.iloc[current_idx])
    
    return {
        'wr10': round(wr10_val, 2),
        'wr6': round(wr6_val, 2),
        'oversold': wr10_val > 80,  # WR超卖
        'overbought': wr10_val < 20,  # WR超买
        'wr6_above_wr10': wr6_val > wr10_val,
    }


def calculate_cci(df: pd.DataFrame) -> Dict:
    """
    计算CCI指标状态 (顺势指标)
    
    返回:
        Dict: CCI指标状态
    """
    C = df['close']
    H = df['high']
    L = df['low']
    
    # 典型价格
    TP = (H + L + C) / 3
    
    # CCI计算
    N = 14
    MA_TP = MA(TP, N)
    MD = TP.rolling(window=N).apply(lambda x: np.abs(x - x.mean()).mean())
    
    CCI = (TP - MA_TP) / (MD * 0.015 + 0.0001)
    
    current_idx = len(df) - 1
    cci_val = float(CCI.iloc[current_idx])
    
    return {
        'cci': round(cci_val, 2),
        'overbought': cci_val > 100,
        'oversold': cci_val < -100,
        'strong_bullish': cci_val > 200,
        'strong_bearish': cci_val < -200,
        'rising': bool(CCI.iloc[current_idx] > CCI.iloc[current_idx-1]) if current_idx > 0 else False,
    }


# ==============================================================================
# 市场理论分析函数
# ==============================================================================

def analyze_dow_theory(df: pd.DataFrame) -> Dict:
    """
    道氏理论分析
    
    分析维度:
    - 主要趋势判断
    - 次级趋势判断
    - 高低点结构
    
    返回:
        Dict: 道氏理论分析结果
    """
    C = df['close']
    H = df['high']
    L = df['low']
    
    current_idx = len(df) - 1
    
    # 使用不同周期判断趋势
    # 主要趋势 (60日)
    ma60 = MA(C, 60)
    primary_trend = 'bullish' if C.iloc[current_idx] > ma60.iloc[current_idx] else 'bearish'
    
    # 次级趋势 (20日)
    ma20 = MA(C, 20)
    secondary_trend = 'bullish' if C.iloc[current_idx] > ma20.iloc[current_idx] else 'bearish'
    
    # 短期趋势 (5日)
    ma5 = MA(C, 5)
    short_trend = 'bullish' if C.iloc[current_idx] > ma5.iloc[current_idx] else 'bearish'
    
    # 高低点分析 (最近20日)
    recent_high = float(HHV(H, 20).iloc[current_idx])
    recent_low = float(LLV(L, 20).iloc[current_idx])
    
    # 前期高低点 (20-40日前)
    if current_idx >= 40:
        prev_high = float(H.iloc[current_idx-40:current_idx-20].max())
        prev_low = float(L.iloc[current_idx-40:current_idx-20].min())
    else:
        prev_high = recent_high
        prev_low = recent_low
    
    # 高低点结构判断
    higher_high = recent_high > prev_high
    higher_low = recent_low > prev_low
    lower_high = recent_high < prev_high
    lower_low = recent_low < prev_low
    
    # 趋势确认
    uptrend_confirmed = higher_high and higher_low
    downtrend_confirmed = lower_high and lower_low
    
    return {
        'primary_trend': primary_trend,
        'secondary_trend': secondary_trend,
        'short_trend': short_trend,
        'trend_alignment': primary_trend == secondary_trend == short_trend,
        'higher_high': higher_high,
        'higher_low': higher_low,
        'lower_high': lower_high,
        'lower_low': lower_low,
        'uptrend_confirmed': uptrend_confirmed,
        'downtrend_confirmed': downtrend_confirmed,
        'recent_high': round(recent_high, 2),
        'recent_low': round(recent_low, 2),
    }


def analyze_wyckoff_theory(df: pd.DataFrame) -> Dict:
    """
    威科夫理论分析
    
    分析维度:
    - 供需关系
    - 成交量特征
    - 价格行为
    - 市场阶段判断
    
    返回:
        Dict: 威科夫理论分析结果
    """
    C = df['close']
    H = df['high']
    L = df['low']
    V = df['volume']
    O = df['open']
    
    current_idx = len(df) - 1
    
    if current_idx < 20:
        return {'phase': 'unknown', 'volume_analysis': {}, 'price_action': {}}
    
    # 成交量分析
    avg_volume_20 = float(MA(V, 20).iloc[current_idx])
    current_volume = float(V.iloc[current_idx])
    volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
    
    # 价格变动
    price_change = float(C.iloc[current_idx] - C.iloc[current_idx-1])
    price_change_pct = price_change / float(C.iloc[current_idx-1]) * 100 if C.iloc[current_idx-1] > 0 else 0
    
    # 价格波动范围
    daily_range = float(H.iloc[current_idx] - L.iloc[current_idx])
    avg_range = float((H - L).rolling(20).mean().iloc[current_idx])
    range_ratio = daily_range / avg_range if avg_range > 0 else 1
    
    # 收盘位置
    close_position = (float(C.iloc[current_idx]) - float(L.iloc[current_idx])) / daily_range if daily_range > 0 else 0.5
    
    # 威科夫供需分析
    # 上涨放量 = 需求增加
    # 下跌放量 = 供应增加
    # 上涨缩量 = 需求减少
    # 下跌缩量 = 供应减少
    
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
    
    # 市场阶段判断 (简化版)
    # 吸筹阶段特征：低位横盘，成交量萎缩后放大
    # 上涨阶段特征：价格突破，成交量放大
    # 派发阶段特征：高位横盘，成交量放大
    # 下跌阶段特征：价格跌破，成交量放大
    
    # 计算近期价格位置
    price_position = (float(C.iloc[current_idx]) - float(LLV(L, 60).iloc[current_idx])) / \
                     (float(HHV(H, 60).iloc[current_idx]) - float(LLV(L, 60).iloc[current_idx]) + 0.0001)
    
    # 成交量趋势
    vol_ma5 = float(MA(V, 5).iloc[current_idx])
    vol_ma20 = float(MA(V, 20).iloc[current_idx])
    volume_expanding = vol_ma5 > vol_ma20
    
    # 阶段判断
    if price_position < 0.3 and not volume_expanding:
        phase = 'accumulation'  # 吸筹阶段
    elif price_position < 0.3 and volume_expanding:
        phase = 'markup_start'  # 上涨启动
    elif 0.3 <= price_position <= 0.7:
        phase = 'markup' if price_change > 0 else 'markdown'  # 上涨/下跌阶段
    elif price_position > 0.7 and volume_expanding:
        phase = 'distribution'  # 派发阶段
    else:
        phase = 'markdown_start'  # 下跌启动
    
    return {
        'phase': phase,
        'supply_demand': supply_demand,
        'volume_ratio': round(volume_ratio, 2),
        'volume_expanding': volume_expanding,
        'price_position': round(price_position * 100, 2),
        'close_position': round(close_position * 100, 2),
        'range_ratio': round(range_ratio, 2),
        'price_change_pct': round(price_change_pct, 2),
    }


# ==============================================================================
# 均线系统分析
# ==============================================================================

def analyze_ma_system(df: pd.DataFrame) -> Dict:
    """
    均线系统分析
    
    返回:
        Dict: 均线系统分析结果
    """
    C = df['close']
    
    # 计算各周期均线
    ma5 = MA(C, 5)
    ma10 = MA(C, 10)
    ma20 = MA(C, 20)
    ma30 = MA(C, 30)
    ma60 = MA(C, 60)
    ma120 = MA(C, 120)
    
    current_idx = len(df) - 1
    current_close = float(C.iloc[current_idx])
    
    ma_values = {
        'ma5': float(ma5.iloc[current_idx]),
        'ma10': float(ma10.iloc[current_idx]),
        'ma20': float(ma20.iloc[current_idx]),
        'ma30': float(ma30.iloc[current_idx]),
        'ma60': float(ma60.iloc[current_idx]) if current_idx >= 60 else np.nan,
        'ma120': float(ma120.iloc[current_idx]) if current_idx >= 120 else np.nan,
    }
    
    # 多头排列判断
    bullish_alignment = (
        ma_values['ma5'] > ma_values['ma10'] > ma_values['ma20'] > ma_values['ma30']
    )
    
    # 空头排列判断
    bearish_alignment = (
        ma_values['ma5'] < ma_values['ma10'] < ma_values['ma20'] < ma_values['ma30']
    )
    
    # 价格与均线关系
    above_ma5 = current_close > ma_values['ma5']
    above_ma10 = current_close > ma_values['ma10']
    above_ma20 = current_close > ma_values['ma20']
    above_ma60 = current_close > ma_values['ma60'] if not np.isnan(ma_values['ma60']) else False
    
    # 均线支撑/压力
    ma_support_count = sum([above_ma5, above_ma10, above_ma20, above_ma60])
    
    return {
        'ma_values': {k: round(v, 2) if not np.isnan(v) else None for k, v in ma_values.items()},
        'bullish_alignment': bullish_alignment,
        'bearish_alignment': bearish_alignment,
        'above_ma5': above_ma5,
        'above_ma10': above_ma10,
        'above_ma20': above_ma20,
        'above_ma60': above_ma60,
        'ma_support_count': ma_support_count,
        'golden_cross_5_10': bool(CROSS(ma5, ma10).iloc[current_idx]) if current_idx > 0 else False,
        'golden_cross_10_20': bool(CROSS(ma10, ma20).iloc[current_idx]) if current_idx > 0 else False,
    }


# ==============================================================================
# 综合分析函数
# ==============================================================================

def analyze_single_case(stock_code: str, signal_date: str) -> Dict:
    """
    分析单个成功案例
    
    参数:
        stock_code: 股票代码 (格式: market_code, 如 sz_000001)
        signal_date: 信号日期
        
    返回:
        Dict: 分析结果
    """
    # 解析股票代码
    parts = stock_code.split('_')
    if len(parts) != 2:
        return {}
    
    market, code = parts
    filepath = os.path.join(DATA_DIR, market, f"{code}.csv")
    
    if not os.path.exists(filepath):
        return {}
    
    try:
        # 加载数据
        df = pd.read_csv(filepath, encoding='utf-8')
        
        # 标准化列名
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
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # 找到信号日期的位置
        signal_dt = pd.to_datetime(signal_date)
        signal_idx = df[df['date'] == signal_dt].index
        
        if len(signal_idx) == 0:
            return {}
        
        signal_idx = signal_idx[0]
        
        # 截取信号日期及之前的数据用于计算指标
        df_for_analysis = df.iloc[:signal_idx + 1].copy()
        
        if len(df_for_analysis) < 30:
            return {}
        
        # 计算各项指标
        result = {
            'macd': calculate_macd(df_for_analysis),
            'kdj': calculate_kdj(df_for_analysis),
            'boll': calculate_boll(df_for_analysis),
            'rsi': calculate_rsi(df_for_analysis),
            'dmi': calculate_dmi(df_for_analysis),
            'dma': calculate_dma(df_for_analysis),
            'sar': calculate_sar(df_for_analysis),
            'bbi': calculate_bbi(df_for_analysis),
            'obv': calculate_obv(df_for_analysis),
            'wr': calculate_wr(df_for_analysis),
            'cci': calculate_cci(df_for_analysis),
            'dow_theory': analyze_dow_theory(df_for_analysis),
            'wyckoff': analyze_wyckoff_theory(df_for_analysis),
            'ma_system': analyze_ma_system(df_for_analysis),
        }
        
        return result
        
    except Exception as e:
        print(f"分析失败 {stock_code} {signal_date}: {str(e)}")
        return {}


def analyze_case_wrapper(args):
    """多进程包装函数"""
    stock_code, signal_date = args
    return analyze_single_case(stock_code, signal_date)


# ==============================================================================
# 统计分析函数
# ==============================================================================

def calculate_statistics(analysis_results: List[Dict], field_path: str) -> Dict:
    """
    计算指定字段的统计信息
    
    参数:
        analysis_results: 分析结果列表
        field_path: 字段路径 (如 'macd.dif_above_zero')
        
    返回:
        Dict: 统计信息
    """
    values = []
    
    for result in analysis_results:
        if not result:
            continue
        
        # 解析字段路径
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
    
    # 根据值类型计算统计
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
    """
    生成模式统计摘要
    
    参数:
        df: 成功案例DataFrame
        analysis_results: 分析结果列表
        
    返回:
        Dict: 模式统计摘要
    """
    summary = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_cases': len(df),
        'analyzed_cases': len([r for r in analysis_results if r]),
        'indicators': {},
        'theories': {},
    }
    
    # 定义要统计的指标字段
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
    
    # 计算各指标统计
    for indicator_name, fields in indicator_fields.items():
        summary['indicators'][indicator_name] = {}
        for field_path, field_name in fields:
            stats = calculate_statistics(analysis_results, field_path)
            summary['indicators'][indicator_name][field_name] = stats
    
    # 市场理论统计
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
    """
    按信号类型生成统计摘要
    
    参数:
        df: 成功案例DataFrame
        analysis_results: 分析结果列表
        
    返回:
        Dict: 按信号类型分类的统计
    """
    # 将分析结果与原始数据关联
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
        
        # 计算关键模式统计
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
    """
    主程序入口
    """
    print("=" * 60)
    print("成功案例模式分析器")
    print("=" * 60)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 检查成功案例文件
    if not os.path.exists(SUCCESS_CASES_FILE):
        print(f"错误: 成功案例文件不存在: {SUCCESS_CASES_FILE}")
        print("请先运行 signal_success_scanner.py 生成成功案例数据")
        return
    
    # 加载成功案例
    df = pd.read_csv(SUCCESS_CASES_FILE, encoding='utf-8-sig')
    print(f"加载成功案例: {len(df)} 条")
    
    if df.empty:
        print("没有成功案例可分析")
        return
    
    # 准备分析任务
    tasks = list(zip(df['stock_code'].tolist(), df['date'].tolist()))
    total_tasks = len(tasks)
    
    print(f"\n开始分析 {total_tasks} 个成功案例...")
    print("-" * 60)
    
    # 多进程分析
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    analysis_results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_case_wrapper, task): i for i, task in enumerate(tasks)}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
                analysis_results.append(result)
                
                if completed % 100 == 0:
                    print(f"进度: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")
                    
            except Exception as e:
                analysis_results.append({})
                print(f"分析失败: {str(e)}")
    
    # 按原始顺序排列结果
    sorted_results = [None] * len(tasks)
    for future, idx in futures.items():
        try:
            sorted_results[idx] = future.result()
        except:
            sorted_results[idx] = {}
    
    analysis_results = sorted_results
    
    print(f"\n分析完成，成功分析 {len([r for r in analysis_results if r])} 个案例")
    
    # 生成详细分析报告
    print("\n生成分析报告...")
    
    # 将分析结果展平并保存
    flat_results = []
    for i, (idx, row) in enumerate(df.iterrows()):
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
        
        # 展平指标数据
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
    
    # 保存详细报告
    report_df = pd.DataFrame(flat_results)
    report_path = os.path.join(REPORT_DIR, 'pattern_analysis_report.csv')
    report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"详细分析报告已保存: {report_path}")
    
    # 生成模式统计摘要
    summary = generate_pattern_summary(df, analysis_results)
    summary_path = os.path.join(REPORT_DIR, 'pattern_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"模式统计摘要已保存: {summary_path}")
    
    # 生成按信号类型分类的统计
    signal_summary = generate_signal_type_summary(df, analysis_results)
    signal_summary_path = os.path.join(REPORT_DIR, 'pattern_analysis_by_signal.json')
    with open(signal_summary_path, 'w', encoding='utf-8') as f:
        json.dump(signal_summary, f, ensure_ascii=False, indent=2)
    print(f"按信号类型统计已保存: {signal_summary_path}")
    
    # 打印关键发现
    print("\n" + "=" * 60)
    print("关键发现摘要")
    print("=" * 60)
    
    print(f"\n总分析案例: {summary['analyzed_cases']}")
    
    # 打印各指标的关键统计
    print("\n【技术指标共性特征】")
    print("-" * 60)
    
    for indicator, stats in summary['indicators'].items():
        print(f"\n{indicator}:")
        for field_name, field_stats in stats.items():
            if 'true_rate' in field_stats:
                print(f"  {field_name}: {field_stats['true_rate']}% 的案例满足")
            elif 'mean' in field_stats:
                print(f"  {field_name}: 均值={field_stats['mean']}, 中位数={field_stats['median']}")
            elif 'distribution' in field_stats:
                print(f"  {field_name}: {field_stats['distribution']}")
    
    print("\n【市场理论分析】")
    print("-" * 60)
    
    for theory, stats in summary['theories'].items():
        print(f"\n{theory}:")
        for field_name, field_stats in stats.items():
            if 'true_rate' in field_stats:
                print(f"  {field_name}: {field_stats['true_rate']}% 的案例满足")
            elif 'distribution' in field_stats:
                print(f"  {field_name}: {field_stats['distribution']}")
            elif 'mean' in field_stats:
                print(f"  {field_name}: 均值={field_stats['mean']}")
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
