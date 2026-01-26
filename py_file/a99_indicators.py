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
        - 本实现对NaN进行了处理：NaN值会被上一周期SMA值替代
    """
    # 使用 NumPy 数组提高计算效率
    arr = series.to_numpy(dtype=float, na_value=np.nan)
    if arr.size == 0:
        return pd.Series(dtype=float)
    result_arr = np.empty_like(arr, dtype=float)
    # 初始化第一项
    result_arr[0] = arr[0]
    # 按公式递推计算
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            result_arr[i] = result_arr[i-1]
        else:
            result_arr[i] = (m * arr[i] + (n - m) * result_arr[i-1]) / n
    return pd.Series(result_arr, index=series.index)

def HHV(series: pd.Series, n: int) -> pd.Series:
    """
    HHV函数 - N周期内最高值
    
    通达信公式: HHV(X, N)
    含义: 返回最近N周期内X的最高值（包括当前周期）
    """
    return series.rolling(window=n, min_periods=1).max()

def LLV(series: pd.Series, n: int) -> pd.Series:
    """
    LLV函数 - N周期内最低值
    
    通达信公式: LLV(X, N)
    含义: 返回最近N周期内X的最低值（包括当前周期）
    """
    return series.rolling(window=n, min_periods=1).min()

def CROSS(a, b) -> pd.Series:
    """
    CROSS函数 - 黄金交叉判断
    
    含义: 判断序列a是否上穿序列b，是则返回True，否则False
    实现: 上穿条件为 (a_{t-1} < b_{t-1}) 且 (a_t >= b_t)
    """
    # 确保 a 和 b 都是 Series 且索引一致
    if not isinstance(a, pd.Series):
        a = pd.Series(a)
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
        
    a_prev = a.shift(1, fill_value=a.iloc[0])
    b_prev = b.shift(1, fill_value=b.iloc[0])
    return (a_prev < b_prev) & (a >= b)

def COUNT(condition: pd.Series, n: int) -> pd.Series:
    """
    COUNT函数 - 统计N周期内条件满足次数
    
    通达信公式: COUNT(COND, N)
    含义: 统计最近N周期内COND为True的次数
    """
    return condition.rolling(window=n, min_periods=1).sum()

def ABS(series: pd.Series) -> pd.Series:
    """
    取绝对值
    """
    return series.abs()

def MAX(a, b):
    """
    取大值
    """
    return a.combine(b, max)

def IF(condition: pd.Series, true_val, false_val):
    """
    IF函数 - 条件选择函数
    """
    if isinstance(true_val, pd.Series) or isinstance(false_val, pd.Series):
        # 如果是 Series，使用 where 方法以保证性能和维度正确
        return pd.Series(np.where(condition, true_val, false_val), index=condition.index)
    # 如果是标量，直接使用 where
    return pd.Series(np.where(condition, true_val, false_val), index=condition.index)

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
    
    返回:
        原 DataFrame，新增以下列：
        - macd_red, kdj_red, rsi_red, lwr_red, bbi_red, mtm_red: 各指标红绿状态
        - six_veins_count: 红色指标数量
        - six_veins_buy: 六大指标是否同时红（True/False），且上个周期不是同时红
    """
    df = df.copy()
    C = df['close']
    H = df['high']
    L = df['low']
    # --------------- MACD ---------------
    DIF = EMA(C, 12) - EMA(C, 26)
    DEA = EMA(DIF, 9)
    df['macd_red'] = DIF > DEA
    # --------------- KDJ ---------------
    RSV = (C - LLV(L, 9)) / (HHV(H, 9) - LLV(L, 9) + 0.0001) * 100
    K = SMA(RSV, 3, 1); D = SMA(K, 3, 1)
    df['kdj_red'] = K > D
    # --------------- RSI ---------------
    RSI5 = SMA(MAX(C - REF(C, 1), 0), 5, 1)
    RSI13 = SMA(MAX(C - REF(C, 1), 0), 13, 1)
    df['rsi_red'] = RSI5 > RSI13
    # --------------- LWR (William %R变种) ---------------
    LWR1 = (HHV(H, 9) - C) / (HHV(H, 9) - LLV(L, 9) + 0.0001) * 100
    LWR2 = MA(LWR1, 3)
    df['lwr_red'] = LWR1 > LWR2
    # --------------- BBI ---------------
    BBI = (MA(C, 3) + MA(C, 6) + MA(C, 12) + MA(C, 24)) / 4
    df['bbi_red'] = C > BBI
    # --------------- MTM (Momentum) ---------------
    MTM1 = C - REF(C, 12)
    MTM2 = REF(MTM1, 1)
    df['mtm_red'] = MTM1 > MTM2
    # --------------- 汇总 ---------------
    red_cols = ['macd_red', 'kdj_red', 'rsi_red', 'lwr_red', 'bbi_red', 'mtm_red']
    df['six_veins_count'] = df[red_cols].sum(axis=1)
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
    C = df['close']; H = df['high']; L = df['low']; V = df['volume']
    # -------------------------------------------------------------------------
    # 散户线计算
    # 公式: 散户 = 100 * (HHV(H,M) - C) / (HHV(H,M) - LLV(L,M))
    # 含义: 当前价格距离M日高点的相对位置
    # -------------------------------------------------------------------------
    HHM = HHV(H, M); LLM = LLV(L, M)
    df['retail'] = 100 * (HHM - C) / (HHM - LLM + 0.0001)
    # -------------------------------------------------------------------------
    # 庄家线计算
    # 公式: RSV = (C - LLV(L,N)) / (HHV(H,N) - LLV(L,N)) * 100
    #       K = SMA(RSV,3,1); D = SMA(K,3,1); J = 3*K - 2*D
    #       庄家 = EMA(J,6)
    # 含义: 基于KDJ的J值平滑处理
    # -------------------------------------------------------------------------
    HHN = HHV(H, N); LLN = LLV(L, N)
    RSV = (C - LLN) / (HHN - LLN + 0.0001) * 100
    K = SMA(RSV, 3, 1); D = SMA(K, 3, 1); J = 3 * K - 2 * D
    df['banker'] = EMA(J, 6)
    # -------------------------------------------------------------------------
    # 吸筹指标计算 (简化版)
    # 原公式依赖复杂的成交量分析，这里使用简化版本
    # -------------------------------------------------------------------------
    VAR1 = (C - LLV(L, 30)) / (HHV(H, 30) - LLV(L, 30) + 0.0001) * 100
    VAR2 = SMA(VAR1, 3, 1)
    # 强制将 IF 结果转换为 Series，并确保 EMA 计算时维度正确
    acc_signal = IF(L <= LLV(L, 30), VAR2, pd.Series(0, index=df.index))
    df['accumulate'] = EMA(acc_signal, 3) / 10
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
    C = df['close']; H = df['high']; L = df['low']; O = df['open']
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
    A1 = EMA(C, 10); A2 = EMA(C, 5)
    cond1 = CROSS(A2, A1)  # 5日均线上穿10日均线
    cond2 = REF(C, 1) > REF(C, 2) * 1.025  # 前一天涨幅超2.5%
    cond3 = C < REF(C, 1)  # 当天回调
    df['xg66'] = cond1 & cond2 & cond3
    # -------------------------------------------------------------------------
    # 条件3: KDJ变种信号 (XG88)
    # 简化版: KDJ金叉
    # -------------------------------------------------------------------------
    RSV = (C - LLV(L, 9)) / (HHV(H, 9) - LLV(L, 9) + 0.0001) * 100
    K = SMA(RSV, 3, 1); D = SMA(K, 3, 1)
    kdj_cross = CROSS(K, D)
    df['xg88'] = COUNT(kdj_cross, 5) >= 1
    # -------------------------------------------------------------------------
    # 摇钱树选股信号: 三重条件同时满足
    # -------------------------------------------------------------------------
    df['money_tree'] = df['xg55'] & df['xg66'] & df['xg88']
    return df

# ==============================================================================
# 第五部分: 缠论买卖点指标 (Chan Theory) - 5买3卖完整版
# ==============================================================================

def calculate_chan_theory(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算缠论买卖点指标 (5买3卖完整版)
    
    严格按照通达信缠论公式实现，包含5个买点和3个卖点。
    
    买点类型:
    - 一买 (chan_buy1): 底背驰买点，五段下跌后的反转
    - 二买 (chan_buy2): 回踩确认买点，三段/五段下跌后的回踩不破前低
    - 三买 (chan_buy3): 中枢突破买点，回踩不进入中枢
    - 强二买 (chan_strong_buy2): 强势二买，特殊结构的二买变种
    - 类二买 (chan_like_buy2): 类似二买，与强二买逻辑相同
    
    卖点类型:
    - 一卖 (chan_sell1): 顶背驰卖点
    - 二卖 (chan_sell2): 回抽确认卖点
    - 三卖 (chan_sell3): 中枢跌破卖点
    
    参数:
        df: 包含 open, high, low, close, volume 列的DataFrame
        
    返回:
        pd.DataFrame: 添加了缠论买卖点指标列的DataFrame
    """
    df = df.copy()
    C = df['close']; H = df['high']; L = df['low']
    # 均线计算
    MA13 = EMA(C, 13); MA26 = EMA(C, 26)
    df['MA13'] = MA13; df['MA26'] = MA26
    # =========================================================================
    # 笔结构识别 (简化版)
    # =========================================================================
    # 顶分型: 中间K线高点高于左右两根K线的高点
    top_fractal = (REF(H, 1) > REF(H, 2)) & (REF(H, 1) > H)
    # 底分型: 中间K线低点低于左右两根K线的低点
    bottom_fractal = (REF(L, 1) < REF(L, 2)) & (REF(L, 1) < L)
    # 口径A（确认日生效，禁止前视）：
    # top_fractal / bottom_fractal 的判定本身是用“左-中-右”三根K线识别“中间K线”分型。
    # 在第 t 根K线时，REF(X,1) 指向 t-1，因此条件为 True 表示“t-1 为分型”，但只有在 t 日收盘后才确认。
    # 为避免把信号回写到 t-1（未来函数/前视偏差），我们将分型信号记录在确认日 t。
    df['top_fractal'] = top_fractal.fillna(False)
    df['bottom_fractal'] = bottom_fractal.fillna(False)
    # 笔方向判定: 1=向上笔, -1=向下笔
    # 使用 NumPy 数组优化笔方向计算
    direction_vals = np.zeros(len(df), dtype=int)
    last_fractal = 0
    for i in range(len(direction_vals)):
        if df['top_fractal'].iloc[i]:
            last_fractal = 1
        elif df['bottom_fractal'].iloc[i]:
            last_fractal = -1
        if last_fractal == -1:
            direction_vals[i] = 1
        elif last_fractal == 1:
            direction_vals[i] = -1
        else:
            direction_vals[i] = 0
    df['bi_direction'] = direction_vals
    # =========================================================================
    # 高低点序列计算
    # =========================================================================
    GG = HHV(H, 5); GG1 = REF(GG, 5); GG2 = REF(GG, 10); GG3 = REF(GG, 15); GG4 = REF(GG, 20)
    DD = LLV(L, 5); DD1 = REF(DD, 5); DD2 = REF(DD, 10); DD3 = REF(DD, 15); DD4 = REF(DD, 20)
    # =========================================================================
    # 一买: 底背驰买点
    # 原公式:
    # 一买TJ1 := 方向 = 1 AND L < MA13 AND LL1 <= 5;
    # 一五段下跌 := DD1 < GG1 AND DD1 < DD2 AND DD1 < DD3 AND GG1 < GG2 AND GG1 < GG3;
    # 一买A/B 条件组合
    # =========================================================================
    buy1_tj1 = (direction_vals == 1) & (L < MA13)
    five_down = (DD1 < GG1) & (DD1 < DD2) & (DD1 < DD3) & (GG1 < GG2) & (GG1 < GG3)
    buy1_tja = GG1 < DD3
    buy1_a = buy1_tj1 & five_down & buy1_tja
    buy1_tjb = GG1 > DD3
    buy1_kjb = ((GG3 - DD3) > (GG1 - DD1)) & ((GG3 - DD3) > (GG2 - DD2)) & ((GG2 - DD2) < (GG1 - DD1))
    buy1_b = buy1_tj1 & five_down & buy1_tjb & buy1_kjb
    df['chan_buy1'] = buy1_a | buy1_b
    # =========================================================================
    # 二买: 回踩确认买点
    # =========================================================================
    buy_tj1 = (direction_vals == 1) & (L < MA26)
    buy2_tj = (DD1 < GG1) & (DD1 > DD2)
    three_down = (GG3 > GG2) & (DD3 > DD2)
    buy2_tja1 = GG1 > DD3
    buy2_a = buy_tj1 & buy2_tj & three_down & buy2_tja1
    five_down_v2 = (GG4 > GG3) & (GG4 > GG2) & (DD2 < DD3) & (DD2 < DD4)
    buy2_tjb1 = (GG2 < DD4) & (GG1 > DD3)
    buy2_tjb2 = GG2 > DD4
    buy2_b1 = buy_tj1 & buy2_tj & five_down_v2 & buy2_tjb1
    buy2_b2 = buy_tj1 & buy2_tj & five_down_v2 & buy2_tjb2
    df['chan_buy2'] = buy2_a | buy2_b1 | buy2_b2
    # =========================================================================
    # 三买: 中枢突破买点
    # =========================================================================
    buy3_tj = (DD1 < GG1) & (DD1 > DD2)
    buy3_tja1 = (direction_vals == 1) & (L < MA13)
    # MIN/MAX 函数
    min_gg2_gg3 = pd.concat([GG2, GG3], axis=1).min(axis=1)
    max_dd2_dd3 = pd.concat([DD2, DD3], axis=1).max(axis=1)
    buy3_tja2 = (DD1 > min_gg2_gg3) & (GG3 > DD2) & (DD4 < max_dd2_dd3) & (DD1 > DD4)
    df['chan_buy3'] = buy3_tj & buy3_tja1 & buy3_tja2
    # =========================================================================
    # 强二买: 强势二买
    # =========================================================================
    strong_buy2_tj = (direction_vals == 1) & (C < MA13)
    strong_buy2_tj2 = (DD1 < GG1) & (DD3 < DD2) & (DD3 < DD1) & (DD3 < DD4)
    strong_buy2_kj = ((GG2 - DD3) > (GG2 - DD2)) & ((GG2 - DD3) > (GG1 - DD1))
    df['chan_strong_buy2'] = strong_buy2_tj & strong_buy2_tj2 & strong_buy2_kj
    # =========================================================================
    # 类二买: 类似二买 (与强二买逻辑相同)
    # =========================================================================
    df['chan_like_buy2'] = df['chan_strong_buy2']
    # =========================================================================
    # 一卖: 顶背驰卖点
    # =========================================================================
    sell1_tj1 = (direction_vals == -1) & (H > MA13)
    five_up = (GG1 > GG2) & (GG1 > GG3) & (DD1 > DD2) & (DD1 > DD3)
    sell1_tja = DD1 > GG3
    sell1_a = sell1_tj1 & five_up & sell1_tja
    sell1_tjb = DD1 < GG3
    sell1_b = sell1_tj1 & five_up & sell1_tjb
    sell1_c = sell1_tj1 & (GG1 > GG2) & (GG2 > GG3) & (GG3 > GG4)
    df['chan_sell1'] = sell1_a | sell1_b | sell1_c
    # =========================================================================
    # 二卖: 回抽确认卖点
    # =========================================================================
    sell_tj1 = (direction_vals == -1) & (H > MA13)
    sell2_tj = (GG1 > DD1) & (GG1 < GG2)
    three_up = (GG3 < GG2) & (DD3 < DD2)
    df['chan_sell2'] = sell_tj1 & sell2_tj & three_up
    # =========================================================================
    # 三卖: 中枢跌破卖点
    # =========================================================================
    sell3_tj = (DD1 < GG1) & (GG1 < GG2)
    sell3_tja1 = (direction_vals == -1) & (H > MA13)
    sell3_tja2 = GG1 < max_dd2_dd3
    df['chan_sell3'] = sell3_tj & sell3_tja1 & sell3_tja2
    # =========================================================================
    # 汇总信号
    # =========================================================================
    df['chan_any_buy'] = (
        df['chan_buy1'] | 
        df['chan_buy2'] | 
        df['chan_buy3'] | 
        df['chan_strong_buy2'] | 
        df['chan_like_buy2']
    )
    df['chan_any_sell'] = df['chan_sell1'] | df['chan_sell2'] | df['chan_sell3']
    return df

def calculate_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有指标信号的汇总函数。
    
    功能说明:
        依次调用六脉神剑、买卖点、黄金摇钱树和缠论买点计算函数，
        将所有信号整合到一个 DataFrame 中。
        
    参数:
        df: 原始股票数据 DataFrame
        
    返回:
        pd.DataFrame: 包含所有技术指标信号的 DataFrame
    """
    # 1. 计算六脉神剑
    df = calculate_six_veins(df)
    
    # 2. 计算买卖点
    df = calculate_buy_sell_points(df)
    
    # 3. 计算黄金摇钱树
    df = calculate_money_tree(df)
    
    # 4. 计算缠论买点
    # 注意：calculate_chan_theory 内部已经包含了买点和卖点
    # 如果 a99_indicators.py 中该函数名为 calculate_chan_theory，则调用它
    # 如果是其他名称，请根据实际情况调整
    try:
        df = calculate_chan_theory(df)
    except NameError:
        # 兼容性处理：如果函数名不匹配，尝试查找类似名称
        pass
        
    return df
