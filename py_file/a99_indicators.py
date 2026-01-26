#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a99_indicators.py
功能描述: 通达信核心技术指标计算模块
使用方法: 被其他回测脚本(a2, a6, a7等)调用，不直接运行。
依赖库: pandas, numpy
安装命令: pip install pandas numpy
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union

# ==============================================================================
# 内部辅助工具
# ==============================================================================

def _ensure_series(x, reference_index=None) -> pd.Series:
    """内部辅助函数：确保输入是 Series，如果不是则根据参考索引转换。"""
    if isinstance(x, pd.Series):
        return x
    if reference_index is not None:
        return pd.Series(x, index=reference_index)
    return pd.Series(x)

# ==============================================================================
# 第一部分: 基础函数 (通达信公式对应函数)
# ==============================================================================

def REF(series: pd.Series, n: int = 1) -> pd.Series:
    """REF(X, N): 返回X在N周期前的值"""
    series = _ensure_series(series)
    return series.shift(n)

def MA(series: pd.Series, n: int) -> pd.Series:
    """MA(X, N): 简单移动平均"""
    series = _ensure_series(series)
    return series.rolling(window=n, min_periods=1).mean()

def EMA(series: pd.Series, n: int) -> pd.Series:
    """EMA(X, N): 指数移动平均"""
    series = _ensure_series(series)
    return series.ewm(span=n, adjust=False).mean()

def SMA(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """SMA(X, N, M): 移动平均 (通达信特有)"""
    series = _ensure_series(series)
    arr = series.to_numpy(dtype=float, na_value=np.nan)
    if arr.size == 0:
        return pd.Series(dtype=float, index=series.index)
    result_arr = np.empty_like(arr, dtype=float)
    result_arr[0] = arr[0]
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            result_arr[i] = result_arr[i-1]
        else:
            result_arr[i] = (m * arr[i] + (n - m) * result_arr[i-1]) / n
    return pd.Series(result_arr, index=series.index)

def HHV(series: pd.Series, n: int) -> pd.Series:
    """HHV(X, N): 返回最近N周期内X的最高值"""
    series = _ensure_series(series)
    return series.rolling(window=n, min_periods=1).max()

def LLV(series: pd.Series, n: int) -> pd.Series:
    """LLV(X, N): 返回最近N周期内X的最低值"""
    series = _ensure_series(series)
    return series.rolling(window=n, min_periods=1).min()

def CROSS(a, b) -> pd.Series:
    """CROSS(A, B): 判断序列A是否上穿序列B"""
    a = _ensure_series(a)
    b = _ensure_series(b, reference_index=a.index)
    
    a_prev = a.shift(1)
    b_prev = b.shift(1)
    
    # 填充第一个 NaN 避免逻辑错误
    if len(a) > 0:
        a_prev.iloc[0] = a.iloc[0]
        b_prev.iloc[0] = b.iloc[0]
        
    return (a_prev < b_prev) & (a >= b)

def COUNT(condition, n: int) -> pd.Series:
    """COUNT(COND, N): 统计最近N周期内COND为True的次数"""
    condition = _ensure_series(condition).astype(int)
    return condition.rolling(window=n, min_periods=1).sum()

def ABS(series: pd.Series) -> pd.Series:
    """ABS(X): 取绝对值"""
    series = _ensure_series(series)
    return series.abs()

def MAX(a, b) -> pd.Series:
    """MAX(A, B): 取大值"""
    a = _ensure_series(a)
    b = _ensure_series(b, reference_index=a.index)
    return pd.Series(np.maximum(a.values, b.values), index=a.index)

def MIN(a, b) -> pd.Series:
    """MIN(A, B): 取小值"""
    a = _ensure_series(a)
    b = _ensure_series(b, reference_index=a.index)
    return pd.Series(np.minimum(a.values, b.values), index=a.index)

def IF(condition, true_val, false_val) -> pd.Series:
    """IF(COND, A, B): 条件选择函数"""
    condition = _ensure_series(condition)
    true_val = _ensure_series(true_val, reference_index=condition.index)
    false_val = _ensure_series(false_val, reference_index=condition.index)
    return pd.Series(np.where(condition, true_val, false_val), index=condition.index)

# ==============================================================================
# 第二部分: 六脉神剑指标 (Six Veins Sword)
# ==============================================================================

def calculate_six_veins(df: pd.DataFrame) -> pd.DataFrame:
    """计算六脉神剑指标"""
    df = df.copy()
    C = df['close']
    
    # 1. MACD
    dif = EMA(C, 12) - EMA(C, 26)
    dea = EMA(dif, 9)
    df['macd_red'] = dif > dea
    
    # 2. KDJ
    low_list = LLV(df['low'], 9)
    high_list = HHV(df['high'], 9)
    rsv = (C - low_list) / (high_list - low_list + 0.0001) * 100
    k = SMA(rsv, 3, 1)
    d = SMA(k, 3, 1)
    df['kdj_red'] = k > d
    
    # 3. RSI
    def rsi_logic(n):
        lc = REF(C, 1)
        diff = MAX(C - lc, 0)
        abs_diff = ABS(C - lc)
        return SMA(diff, n, 1) / SMA(abs_diff, n, 1) * 100
    df['rsi_red'] = rsi_logic(6) > rsi_logic(12)
    
    # 4. LWR (威廉指标变种)
    lwr_low = LLV(df['low'], 9)
    lwr_high = HHV(df['high'], 9)
    lwr_rsv = (lwr_high - C) / (lwr_high - lwr_low + 0.0001) * 100
    lwr1 = SMA(lwr_rsv, 3, 1)
    lwr2 = SMA(lwr1, 3, 1)
    df['lwr_red'] = lwr1 < lwr2  # LWR是反向指标
    
    # 5. BBI
    bbi = (MA(C, 3) + MA(C, 6) + MA(C, 12) + MA(C, 24)) / 4
    df['bbi_red'] = C > bbi
    
    # 6. MTM
    mtm = C - REF(C, 6)
    mamatm = MA(mtm, 6)
    df['mtm_red'] = mtm > mamatm
    
    # 综合信号
    df['six_veins_count'] = (
        df['macd_red'].astype(int) + df['kdj_red'].astype(int) + 
        df['rsi_red'].astype(int) + df['lwr_red'].astype(int) + 
        df['bbi_red'].astype(int) + df['mtm_red'].astype(int)
    )
    df['six_veins_signal'] = df['six_veins_count'] == 6
    return df

# ==============================================================================
# 第三部分: 买卖点指标 (Buy/Sell Points)
# ==============================================================================

def calculate_buy_sell_points(df: pd.DataFrame, n=9, m=14) -> pd.DataFrame:
    """计算庄家买卖点指标"""
    df = df.copy()
    C = df['close']; H = df['high']; L = df['low']
    
    # 散户线
    hhm = HHV(H, m); llm = LLV(L, m)
    df['retail'] = 100 * (hhm - C) / (hhm - llm + 0.0001)
    
    # 庄家线
    hhn = HHV(H, n); lln = LLV(L, n)
    rsv = (C - lln) / (hhn - lln + 0.0001) * 100
    k = SMA(rsv, 3, 1); d = SMA(k, 3, 1); j = 3 * k - 2 * d
    df['banker'] = EMA(j, 6)
    
    # 吸筹指标
    var1 = (C - LLV(L, 30)) / (HHV(H, 30) - LLV(L, 30) + 0.0001) * 100
    var2 = SMA(var1, 3, 1)
    acc_signal = IF(L <= LLV(L, 30), var2, 0)
    df['accumulate'] = EMA(acc_signal, 3) / 10
    
    # 信号
    df['buy1'] = CROSS(df['accumulate'], 14)
    df['buy2'] = CROSS(df['banker'], df['retail']) & (df['banker'] < 50)
    df['sell1'] = CROSS(88, df['banker'])
    df['sell2'] = CROSS(df['retail'], df['banker'])
    return df

# ==============================================================================
# 第四部分: 黄金摇钱树指标 (Money Tree)
# ==============================================================================

def calculate_money_tree(df: pd.DataFrame) -> pd.DataFrame:
    """计算黄金摇钱树指标"""
    df = df.copy()
    C = df['close']; H = df['high']; L = df['low']
    
    # 1. 底部信号 (简化)
    xg55 = COUNT(C < MA(C, 20) * 0.9, 5) > 0
    
    # 2. 动量交叉
    ma5 = MA(C, 5)
    pred_line = MA(C, 10)
    xg66 = CROSS(ma5, pred_line) & (REF(C, 1) / REF(C, 2) > 1.025)
    
    df['money_tree_signal'] = xg55 & xg66
    return df

# ==============================================================================
# 第五部分: 缠论买点 (Chan Theory)
# ==============================================================================

def calculate_chan_theory(df: pd.DataFrame) -> pd.DataFrame:
    """简化版缠论笔结构买点"""
    df = df.copy()
    C = df['close']; L = df['low']; H = df['high']
    
    # 寻找局部低点
    is_llv = L == LLV(L, 10)
    # 寻找局部高点
    is_hhv = H == HHV(H, 10)
    
    df['chan_buy1'] = is_llv & (REF(L, 1) > L) & (C > REF(C, 1))
    df['chan_sell1'] = is_hhv & (REF(H, 1) < H) & (C < REF(C, 1))
    return df

# ==============================================================================
# 第六部分: 汇总函数
# ==============================================================================

def calculate_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有指标信号的汇总函数"""
    df = calculate_six_veins(df)
    df = calculate_buy_sell_points(df)
    df = calculate_money_tree(df)
    df = calculate_chan_theory(df)
    return df
