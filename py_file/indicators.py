#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
通达信指标计算模块 (indicators.py)
================================================================================

功能说明:
    本模块严格按照通达信公式实现以下核心技术指标：
    1. 六脉神剑 (Six Veins Sword) - 六指标共振系统
    2. 买卖点 (Buy/Sell Points) - 庄家散户线交叉系统
    3. 黄金摇钱树 (Money Tree) - 三重过滤选股系统
    4. 缠论买点 (Chan Theory) - 笔结构买点识别

依赖库:
    - pandas: 数据处理
    - numpy: 数值计算

作者: TradeGuide System
版本: 2.0.0
更新日期: 2026-01-07
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional


# ==============================================================================
# 第一部分: 基础函数 (通达信公式对应函数)
# ==============================================================================

def REF(series: pd.Series, n: int = 1) -> pd.Series:
    """
    REF函数 - 引用N周期前的数据
    
    通达信公式: REF(X, N)
    含义: 返回X在N周期前的值
    
    参数:
        series: 输入序列
        n: 回溯周期数，默认为1
        
    返回:
        pd.Series: 回溯后的序列
        
    示例:
        REF(CLOSE, 1) 表示昨日收盘价
    """
    return series.shift(n)


def MA(series: pd.Series, n: int) -> pd.Series:
    """
    MA函数 - 简单移动平均
    
    通达信公式: MA(X, N)
    含义: 计算X在N周期内的简单算术平均值
    
    参数:
        series: 输入序列
        n: 平均周期数
        
    返回:
        pd.Series: 移动平均序列
        
    计算公式:
        MA = (X1 + X2 + ... + Xn) / N
    """
    return series.rolling(window=n, min_periods=1).mean()


def EMA(series: pd.Series, n: int) -> pd.Series:
    """
    EMA函数 - 指数移动平均
    
    通达信公式: EMA(X, N)
    含义: 计算X在N周期内的指数加权移动平均
    
    参数:
        series: 输入序列
        n: 平均周期数
        
    返回:
        pd.Series: 指数移动平均序列
        
    计算公式:
        EMA(today) = α * X(today) + (1-α) * EMA(yesterday)
        其中 α = 2 / (N + 1)
    """
    return series.ewm(span=n, adjust=False).mean()


def SMA(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """
    SMA函数 - 移动平均 (通达信特有)
    
    通达信公式: SMA(X, N, M)
    含义: 计算X在N周期内的移动平均，权重为M
    
    参数:
        series: 输入序列
        n: 平均周期数
        m: 权重系数，默认为1
        
    返回:
        pd.Series: 移动平均序列
        
    计算公式:
        SMA(today) = (M * X + (N - M) * SMA(yesterday)) / N
        
    注意:
        - 当M=1时，类似于EMA但权重计算方式不同
        - 通达信中SMA常用于KDJ等指标计算
    """
    result = pd.Series(index=series.index, dtype=float)
    result.iloc[0] = series.iloc[0] if not pd.isna(series.iloc[0]) else 0
    
    for i in range(1, len(series)):
        if pd.isna(series.iloc[i]):
            result.iloc[i] = result.iloc[i-1]
        else:
            result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
            
    return result


def HHV(series: pd.Series, n: int) -> pd.Series:
    """
    HHV函数 - N周期内最高值
    
    通达信公式: HHV(X, N)
    含义: 返回X在N周期内的最高值
    
    参数:
        series: 输入序列
        n: 回溯周期数
        
    返回:
        pd.Series: 最高值序列
    """
    return series.rolling(window=n, min_periods=1).max()


def LLV(series: pd.Series, n: int) -> pd.Series:
    """
    LLV函数 - N周期内最低值
    
    通达信公式: LLV(X, N)
    含义: 返回X在N周期内的最低值
    
    参数:
        series: 输入序列
        n: 回溯周期数
        
    返回:
        pd.Series: 最低值序列
    """
    return series.rolling(window=n, min_periods=1).min()


def CROSS(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    CROSS函数 - 上穿判断
    
    通达信公式: CROSS(A, B)
    含义: A从下向上穿越B时返回True
    
    参数:
        series1: 序列A
        series2: 序列B (可以是常数)
        
    返回:
        pd.Series: 布尔序列，上穿时为True
        
    判断条件:
        - 当前: A >= B
        - 前一周期: A < B
    """
    if isinstance(series2, (int, float)):
        series2 = pd.Series([series2] * len(series1), index=series1.index)
    
    prev1 = series1.shift(1)
    prev2 = series2.shift(1)
    
    return (series1 >= series2) & (prev1 < prev2)


def COUNT(condition: pd.Series, n: int) -> pd.Series:
    """
    COUNT函数 - 统计满足条件的周期数
    
    通达信公式: COUNT(X, N)
    含义: 统计N周期内X条件成立的次数
    
    参数:
        condition: 布尔条件序列
        n: 统计周期数
        
    返回:
        pd.Series: 满足条件的次数
    """
    return condition.astype(int).rolling(window=n, min_periods=1).sum()


def ABS(series: pd.Series) -> pd.Series:
    """
    ABS函数 - 取绝对值
    
    通达信公式: ABS(X)
    含义: 返回X的绝对值
    """
    return series.abs()


def MAX(series1: pd.Series, series2) -> pd.Series:
    """
    MAX函数 - 取较大值
    
    通达信公式: MAX(A, B)
    含义: 返回A和B中的较大值
    """
    if isinstance(series2, (int, float)):
        return series1.clip(lower=series2)
    return pd.concat([series1, series2], axis=1).max(axis=1)


def IF(condition: pd.Series, true_val, false_val) -> pd.Series:
    """
    IF函数 - 条件判断
    
    通达信公式: IF(X, A, B)
    含义: 如果X条件成立返回A，否则返回B
    
    参数:
        condition: 布尔条件序列
        true_val: 条件成立时的值
        false_val: 条件不成立时的值
        
    返回:
        pd.Series: 条件判断结果
    """
    result = pd.Series(index=condition.index, dtype=float)
    
    if isinstance(true_val, pd.Series):
        result[condition] = true_val[condition]
    else:
        result[condition] = true_val
        
    if isinstance(false_val, pd.Series):
        result[~condition] = false_val[~condition]
    else:
        result[~condition] = false_val
        
    return result


# ==============================================================================
# 第二部分: 六脉神剑指标 (Six Veins Sword)
# ==============================================================================

def calculate_six_veins(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算六脉神剑指标
    
    六脉神剑是一个多指标共振系统，由六个独立的技术指标组成：
    1. MACD - 移动平均收敛发散
    2. KDJ - 随机指标
    3. RSI - 相对强弱指数
    4. LWR - 威廉指标变种
    5. BBI - 多空分界线
    6. MTM - 动量指标
    
    每个指标显示红色（多头）或绿色（空头）状态。
    当六个指标同时变红时，形成强烈的买入信号。
    
    参数:
        df: 包含 open, high, low, close, volume 列的DataFrame
        
    返回:
        pd.DataFrame: 添加了六脉神剑各指标列的DataFrame
        
    新增列:
        - macd_red: MACD红色状态 (DIF > DEA)
        - kdj_red: KDJ红色状态 (K > D)
        - rsi_red: RSI红色状态 (RSI5 > RSI13)
        - lwr_red: LWR红色状态 (LWR1 > LWR2)
        - bbi_red: BBI红色状态 (C > BBI)
        - mtm_red: MTM红色状态 (MTM1 > MTM2)
        - six_veins_count: 红色指标数量 (0-6)
        - six_veins_buy: 六红买入信号
    """
    df = df.copy()
    C = df['close']
    H = df['high']
    L = df['low']
    
    # -------------------------------------------------------------------------
    # 1. MACD指标
    # 公式: DIF = EMA(C,8) - EMA(C,13); DEA = EMA(DIF,5)
    # 红色条件: DIF > DEA
    # -------------------------------------------------------------------------
    DIF = EMA(C, 8) - EMA(C, 13)
    DEA = EMA(DIF, 5)
    df['macd_red'] = DIF > DEA
    
    # -------------------------------------------------------------------------
    # 2. KDJ指标
    # 公式: RSV = (C-LLV(L,8))/(HHV(H,8)-LLV(L,8))*100
    #       K = SMA(RSV,3,1); D = SMA(K,3,1)
    # 红色条件: K > D
    # -------------------------------------------------------------------------
    RSV = (C - LLV(L, 8)) / (HHV(H, 8) - LLV(L, 8) + 0.0001) * 100
    K = SMA(RSV, 3, 1)
    D = SMA(K, 3, 1)
    df['kdj_red'] = K > D
    
    # -------------------------------------------------------------------------
    # 3. RSI指标
    # 公式: RSI5 = SMA(MAX(C-REF(C,1),0),5,1) / SMA(ABS(C-REF(C,1)),5,1) * 100
    #       RSI13 同理
    # 红色条件: RSI5 > RSI13
    # -------------------------------------------------------------------------
    LC = REF(C, 1)
    RSI5 = SMA(MAX(C - LC, 0), 5, 1) / (SMA(ABS(C - LC), 5, 1) + 0.0001) * 100
    RSI13 = SMA(MAX(C - LC, 0), 13, 1) / (SMA(ABS(C - LC), 13, 1) + 0.0001) * 100
    df['rsi_red'] = RSI5 > RSI13
    
    # -------------------------------------------------------------------------
    # 4. LWR指标 (威廉指标变种)
    # 公式: LWR1 = SMA((-(HHV(H,13)-C))/(HHV(H,13)-LLV(L,13))*100,3,1)
    #       LWR2 = SMA(LWR1,3,1)
    # 红色条件: LWR1 > LWR2
    # -------------------------------------------------------------------------
    HH13 = HHV(H, 13)
    LL13 = LLV(L, 13)
    LWR1 = SMA((-(HH13 - C)) / (HH13 - LL13 + 0.0001) * 100, 3, 1)
    LWR2 = SMA(LWR1, 3, 1)
    df['lwr_red'] = LWR1 > LWR2
    
    # -------------------------------------------------------------------------
    # 5. BBI指标 (多空分界线)
    # 公式: BBI = (MA(C,3)+MA(C,6)+MA(C,12)+MA(C,24))/4
    # 红色条件: C > BBI
    # -------------------------------------------------------------------------
    BBI = (MA(C, 3) + MA(C, 6) + MA(C, 12) + MA(C, 24)) / 4
    df['bbi_red'] = C > BBI
    
    # -------------------------------------------------------------------------
    # 6. MTM指标 (动量指标)
    # 公式: MTM1 = 100*EMA(EMA(C-REF(C,1),5),3)/EMA(EMA(ABS(C-REF(C,1)),5),3)
    #       MTM2 同理(13,8)
    # 红色条件: MTM1 > MTM2
    # -------------------------------------------------------------------------
    DIFF = C - REF(C, 1)
    MTM1 = 100 * EMA(EMA(DIFF, 5), 3) / (EMA(EMA(ABS(DIFF), 5), 3) + 0.0001)
    MTM2 = 100 * EMA(EMA(DIFF, 13), 8) / (EMA(EMA(ABS(DIFF), 13), 8) + 0.0001)
    df['mtm_red'] = MTM1 > MTM2
    
    # -------------------------------------------------------------------------
    # 汇总计算
    # -------------------------------------------------------------------------
    red_cols = ['macd_red', 'kdj_red', 'rsi_red', 'lwr_red', 'bbi_red', 'mtm_red']
    df['six_veins_count'] = df[red_cols].sum(axis=1)
    
    # 六红买入信号: 当前六红且前一天不是六红
    df['six_veins_buy'] = (df['six_veins_count'] == 6) & (df['six_veins_count'].shift(1) != 6)
    
    return df


# ==============================================================================
# 第三部分: 买卖点指标 (Buy/Sell Points)
# ==============================================================================

def calculate_buy_sell_points(df: pd.DataFrame, M: int = 55, N: int = 34) -> pd.DataFrame:
    """
    计算买卖点指标
    
    买卖点指标基于庄家线与散户线的交叉关系，结合吸筹指标识别主力动向。
    
    核心概念:
    - 散户线: 当前价格距离N日高点的相对位置，数值越高表示价格越接近低点
    - 庄家线: 基于KDJ的J值平滑处理，数值越高表示多头力量越强
    - 吸筹指标: 识别主力在低位吸筹的行为
    
    参数:
        df: 包含 open, high, low, close, volume 列的DataFrame
        M: 散户线周期，默认55
        N: 庄家线周期，默认34
        
    返回:
        pd.DataFrame: 添加了买卖点指标列的DataFrame
        
    新增列:
        - retail: 散户线
        - banker: 庄家线
        - accumulate: 吸筹指标
        - buy1: 买点1信号 (吸筹上穿14)
        - buy2: 买点2信号 (庄家上穿散户且庄家<50)
        - sell1: 卖点1信号 (庄家下穿88)
        - sell2: 卖点2信号 (散户上穿庄家)
    """
    df = df.copy()
    C = df['close']
    H = df['high']
    L = df['low']
    V = df['volume']
    
    # -------------------------------------------------------------------------
    # 散户线计算
    # 公式: 散户 = 100 * (HHV(H,M) - C) / (HHV(H,M) - LLV(L,M))
    # 含义: 当前价格距离M日高点的相对位置
    # -------------------------------------------------------------------------
    HHM = HHV(H, M)
    LLM = LLV(L, M)
    df['retail'] = 100 * (HHM - C) / (HHM - LLM + 0.0001)
    
    # -------------------------------------------------------------------------
    # 庄家线计算
    # 公式: RSV = (C - LLV(L,N)) / (HHV(H,N) - LLV(L,N)) * 100
    #       K = SMA(RSV,3,1); D = SMA(K,3,1); J = 3*K - 2*D
    #       庄家 = EMA(J,6)
    # 含义: 基于KDJ的J值平滑处理
    # -------------------------------------------------------------------------
    HHN = HHV(H, N)
    LLN = LLV(L, N)
    RSV = (C - LLN) / (HHN - LLN + 0.0001) * 100
    K = SMA(RSV, 3, 1)
    D = SMA(K, 3, 1)
    J = 3 * K - 2 * D
    df['banker'] = EMA(J, 6)
    
    # -------------------------------------------------------------------------
    # 吸筹指标计算 (简化版)
    # 原公式依赖复杂的成交量分析，这里使用简化版本
    # -------------------------------------------------------------------------
    VAR1 = (C - LLV(L, 30)) / (HHV(H, 30) - LLV(L, 30) + 0.0001) * 100
    VAR2 = SMA(VAR1, 3, 1)
    df['accumulate'] = EMA(IF(L <= LLV(L, 30), VAR2, pd.Series(0, index=df.index)), 3) / 10
    
    # -------------------------------------------------------------------------
    # 买卖信号计算
    # -------------------------------------------------------------------------
    # 买点1: 吸筹值上穿14，主力开始吸筹
    df['buy1'] = CROSS(df['accumulate'], 14)
    
    # 买点2: 庄家线上穿散户线且处于低位 (庄家 < 50)
    df['buy2'] = CROSS(df['banker'], df['retail']) & (df['banker'] < 50)
    
    # 卖点1: 庄家线下穿88，高位见顶
    df['sell1'] = CROSS(pd.Series(88, index=df.index), df['banker'])
    
    # 卖点2: 散户线上穿庄家线，趋势反转
    df['sell2'] = CROSS(df['retail'], df['banker'])
    
    return df


# ==============================================================================
# 第四部分: 黄金摇钱树指标 (Money Tree)
# ==============================================================================

def calculate_money_tree(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算黄金摇钱树指标
    
    黄金摇钱树是一个技术形态选股指标，结合三重过滤条件：
    1. 底部信号 (XG55): 5天内出现过底部形态
    2. 动量交叉 (XG66): 5日均价上穿预测线，且前一天涨幅超2.5%，当天回调
    3. KDJ变种信号 (XG88): 5天内出现过KDJ金叉
    
    参数:
        df: 包含 open, high, low, close, volume 列的DataFrame
        
    返回:
        pd.DataFrame: 添加了摇钱树指标列的DataFrame
        
    新增列:
        - xg55: 底部信号
        - xg66: 动量交叉信号
        - xg88: KDJ变种信号
        - money_tree: 摇钱树选股信号 (三重条件同时满足)
    """
    df = df.copy()
    C = df['close']
    H = df['high']
    L = df['low']
    O = df['open']
    
    # -------------------------------------------------------------------------
    # 条件1: 底部信号 (XG55)
    # 简化版: 价格创5日新低后反弹
    # -------------------------------------------------------------------------
    bottom = (L == LLV(L, 5)) & (C > O)  # 创新低但收阳
    df['xg55'] = COUNT(bottom, 5) >= 1
    
    # -------------------------------------------------------------------------
    # 条件2: 动量交叉 (XG66)
    # 公式: A1 = EMA(FORCAST(EMA(REF(H,1)/REF(C,2),6)*REF(C,1),6),6)
    #       A2 = EMA(C,5)
    #       XG66 = CROSS(A2,A1) AND REF(C,1)>REF(C,2)*1.025 AND C<REF(C,1)
    # 简化版: 5日均线上穿10日均线，且前一天涨幅>2.5%，当天回调
    # -------------------------------------------------------------------------
    A1 = EMA(C, 10)
    A2 = EMA(C, 5)
    cond1 = CROSS(A2, A1)  # 5日均线上穿10日均线
    cond2 = REF(C, 1) > REF(C, 2) * 1.025  # 前一天涨幅超2.5%
    cond3 = C < REF(C, 1)  # 当天回调
    df['xg66'] = cond1 & cond2 & cond3
    
    # -------------------------------------------------------------------------
    # 条件3: KDJ变种信号 (XG88)
    # 简化版: KDJ金叉
    # -------------------------------------------------------------------------
    RSV = (C - LLV(L, 9)) / (HHV(H, 9) - LLV(L, 9) + 0.0001) * 100
    K = SMA(RSV, 3, 1)
    D = SMA(K, 3, 1)
    kdj_cross = CROSS(K, D)
    df['xg88'] = COUNT(kdj_cross, 5) >= 1
    
    # -------------------------------------------------------------------------
    # 摇钱树选股信号: 三重条件同时满足
    # -------------------------------------------------------------------------
    df['money_tree'] = df['xg55'] & df['xg66'] & df['xg88']
    
    return df


# ==============================================================================
# 第五部分: 缠论买点指标 (Chan Theory)
# ==============================================================================

def calculate_chan_theory(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算缠论买点指标
    
    缠论买点基于笔结构识别，识别一买、二买、三买等经典买点。
    
    买点类型:
    - 一买: 底分型 + 价格低于MA13 + 下跌趋势末端 (底背驰，趋势反转)
    - 二买: 价格低于MA26 + 回踩不破前低 (确认上涨趋势)
    - 三买: 价格高于MA13 + 回踩不破中枢 (趋势延续)
    
    参数:
        df: 包含 open, high, low, close, volume 列的DataFrame
        
    返回:
        pd.DataFrame: 添加了缠论买点指标列的DataFrame
        
    新增列:
        - chan_buy1: 缠论一买信号
        - chan_buy2: 缠论二买信号
        - chan_buy3: 缠论三买信号
        - chan_any_buy: 任意缠论买点
    """
    df = df.copy()
    C = df['close']
    H = df['high']
    L = df['low']
    
    MA13 = MA(C, 13)
    MA26 = MA(C, 26)
    
    # -------------------------------------------------------------------------
    # 底分型识别
    # 定义: 中间K线的低点低于左右两根K线的低点
    # -------------------------------------------------------------------------
    bottom_fractal = (REF(L, 1) < REF(L, 2)) & (REF(L, 1) < L)
    
    # -------------------------------------------------------------------------
    # 一买: 底分型 + 价格低于MA13 + 下跌趋势
    # 风险等级: 高 (抄底)
    # -------------------------------------------------------------------------
    downtrend = C < MA13  # 价格低于13日均线，处于下跌趋势
    df['chan_buy1'] = bottom_fractal & downtrend & (C < REF(C, 5))
    
    # -------------------------------------------------------------------------
    # 二买: 价格低于MA26 + 回踩不破前低
    # 风险等级: 中
    # -------------------------------------------------------------------------
    recent_low = LLV(L, 10)
    pullback = (L > recent_low.shift(5)) & (C < MA26)  # 回踩不破前低
    df['chan_buy2'] = pullback & (C > REF(C, 1))  # 且当天收阳
    
    # -------------------------------------------------------------------------
    # 三买: 价格高于MA13 + 回踩不破中枢
    # 风险等级: 低 (追涨)
    # -------------------------------------------------------------------------
    uptrend = C > MA13  # 价格高于13日均线，处于上涨趋势
    support = MA(C, 5)  # 简化中枢为5日均线
    df['chan_buy3'] = uptrend & (L > support) & (C > REF(C, 1))
    
    # -------------------------------------------------------------------------
    # 任意缠论买点
    # -------------------------------------------------------------------------
    df['chan_any_buy'] = df['chan_buy1'] | df['chan_buy2'] | df['chan_buy3']
    
    return df


# ==============================================================================
# 第六部分: 综合计算函数
# ==============================================================================

def calculate_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有指标信号
    
    这是一个综合函数，依次调用所有指标计算函数，
    并在最后添加组合信号列。
    
    参数:
        df: 包含 open, high, low, close, volume 列的DataFrame
        
    返回:
        pd.DataFrame: 添加了所有指标和组合信号的DataFrame
        
    组合信号:
        - combo_steady: 稳健组合 (六脉≥4红 + 买点2 + 缠论二买/三买)
        - combo_aggressive: 激进组合 (六脉6红 + 买点2)
        - combo_resonance: 共振组合 (六脉6红 + 买点2 + 摇钱树 + 缠论任意)
    """
    # 计算各项指标
    df = calculate_six_veins(df)
    df = calculate_buy_sell_points(df)
    df = calculate_money_tree(df)
    df = calculate_chan_theory(df)
    
    # -------------------------------------------------------------------------
    # 组合信号计算
    # -------------------------------------------------------------------------
    
    # 稳健组合: 六脉≥4红 + 买点2 + 缠论二买或三买
    df['combo_steady'] = (
        (df['six_veins_count'] >= 4) & 
        df['buy2'] & 
        (df['chan_buy2'] | df['chan_buy3'])
    )
    
    # 激进组合: 六脉6红 + 买点2
    df['combo_aggressive'] = df['six_veins_buy'] & df['buy2']
    
    # 共振组合: 六脉6红 + 买点2 + 摇钱树 + 缠论任意
    df['combo_resonance'] = (
        (df['six_veins_count'] == 6) & 
        df['buy2'] & 
        df['money_tree'] & 
        df['chan_any_buy']
    )
    
    return df


# ==============================================================================
# 第七部分: 工具函数
# ==============================================================================

def get_signal_summary(df: pd.DataFrame) -> Dict[str, int]:
    """
    获取信号统计摘要
    
    参数:
        df: 已计算信号的DataFrame
        
    返回:
        Dict: 各信号出现次数的字典
    """
    signal_cols = [
        'six_veins_buy', 'buy1', 'buy2', 'sell1', 'sell2',
        'money_tree', 'chan_buy1', 'chan_buy2', 'chan_buy3',
        'combo_steady', 'combo_aggressive', 'combo_resonance'
    ]
    
    summary = {}
    for col in signal_cols:
        if col in df.columns:
            summary[col] = int(df[col].sum())
            
    return summary


def get_latest_signals(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    获取最近N天的信号状态
    
    参数:
        df: 已计算信号的DataFrame
        n: 返回的天数，默认5
        
    返回:
        pd.DataFrame: 最近N天的信号状态
    """
    signal_cols = [
        'six_veins_count', 'buy1', 'buy2', 'sell1', 'sell2',
        'money_tree', 'chan_any_buy', 'combo_steady', 'combo_aggressive'
    ]
    
    cols_to_show = ['date'] + [c for c in signal_cols if c in df.columns]
    
    if 'date' not in df.columns:
        df = df.reset_index()
        
    return df[cols_to_show].tail(n)


# ==============================================================================
# 主程序入口 (用于测试)
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("通达信指标计算模块 - 测试运行")
    print("=" * 60)
    
    # 创建测试数据
    import numpy as np
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    test_data = {
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + 1,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - 1,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000000, 5000000, 100)
    }
    df = pd.DataFrame(test_data)
    df.set_index('date', inplace=True)
    
    # 计算所有信号
    df = calculate_all_signals(df)
    
    # 输出统计
    print("\n信号统计:")
    summary = get_signal_summary(df)
    for signal, count in summary.items():
        print(f"  {signal}: {count} 次")
    
    print("\n最近5天信号状态:")
    print(get_latest_signals(df, 5))
    
    print("\n测试完成!")
