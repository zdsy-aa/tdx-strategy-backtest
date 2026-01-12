#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
缠论买卖点指标计算模块 (indicators_chan.py)
================================================================================

功能说明:
    本模块严格按照通达信缠论公式实现以下买卖点：
    
    买点：
    1. 一买 (chan_buy1) - 底背驰买点，五段下跌后的反转
    2. 二买 (chan_buy2) - 回踩确认买点，三段/五段下跌后的回踩不破前低
    3. 三买 (chan_buy3) - 中枢突破买点，回踩不进入中枢
    4. 强二买 (chan_strong_buy2) - 强势二买，特殊结构的二买变种
    5. 类二买 (chan_like_buy2) - 类似二买，与强二买逻辑相同
    
    卖点：
    1. 一卖 (chan_sell1) - 顶背驰卖点
    2. 二卖 (chan_sell2) - 回抽确认卖点
    3. 三卖 (chan_sell3) - 中枢跌破卖点

依赖库:
    - pandas: 数据处理
    - numpy: 数值计算

作者: TradeGuide System
版本: 3.0.0
更新日期: 2026-01-12
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional


# ==============================================================================
# 第一部分: 基础函数 (通达信公式对应函数)
# ==============================================================================

def REF(series: pd.Series, n: int = 1) -> pd.Series:
    """REF函数 - 引用N周期前的数据"""
    return series.shift(n)


def MA(series: pd.Series, n: int) -> pd.Series:
    """MA函数 - 简单移动平均"""
    return series.rolling(window=n, min_periods=1).mean()


def EMA(series: pd.Series, n: int) -> pd.Series:
    """EMA函数 - 指数移动平均"""
    return series.ewm(span=n, adjust=False).mean()


def SMA(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """SMA函数 - 移动平均 (通达信特有)"""
    result = pd.Series(index=series.index, dtype=float)
    result.iloc[0] = series.iloc[0] if not pd.isna(series.iloc[0]) else 0
    
    for i in range(1, len(series)):
        if pd.isna(series.iloc[i]):
            result.iloc[i] = result.iloc[i-1]
        else:
            result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
            
    return result


def HHV(series: pd.Series, n: int) -> pd.Series:
    """HHV函数 - N周期内最高值"""
    return series.rolling(window=n, min_periods=1).max()


def LLV(series: pd.Series, n: int) -> pd.Series:
    """LLV函数 - N周期内最低值"""
    return series.rolling(window=n, min_periods=1).min()


def CROSS(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """CROSS函数 - 上穿判断"""
    if isinstance(series2, (int, float)):
        series2 = pd.Series([series2] * len(series1), index=series1.index)
    
    prev1 = series1.shift(1)
    prev2 = series2.shift(1)
    
    return (series1 >= series2) & (prev1 < prev2)


def COUNT(condition: pd.Series, n: int) -> pd.Series:
    """COUNT函数 - 统计满足条件的周期数"""
    return condition.astype(int).rolling(window=n, min_periods=1).sum()


def ABS(series: pd.Series) -> pd.Series:
    """ABS函数 - 取绝对值"""
    return series.abs()


def MAX(series1: pd.Series, series2) -> pd.Series:
    """MAX函数 - 取较大值"""
    if isinstance(series2, (int, float)):
        return series1.clip(lower=series2)
    return pd.concat([series1, series2], axis=1).max(axis=1)


def MIN(series1: pd.Series, series2) -> pd.Series:
    """MIN函数 - 取较小值"""
    if isinstance(series2, (int, float)):
        return series1.clip(upper=series2)
    return pd.concat([series1, series2], axis=1).min(axis=1)


def IF(condition: pd.Series, true_val, false_val) -> pd.Series:
    """IF函数 - 条件判断"""
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


def BARSLAST(condition: pd.Series) -> pd.Series:
    """
    BARSLAST函数 - 上一次条件成立到当前的周期数
    
    通达信公式: BARSLAST(X)
    含义: 上一次X条件成立到当前的周期数
    """
    result = pd.Series(index=condition.index, dtype=float)
    last_true_idx = -1
    
    for i in range(len(condition)):
        if condition.iloc[i]:
            last_true_idx = i
        if last_true_idx >= 0:
            result.iloc[i] = i - last_true_idx
        else:
            result.iloc[i] = i  # 如果从未成立，返回从开始到现在的周期数
            
    return result


def BARSLASTCOUNT(condition: pd.Series) -> pd.Series:
    """
    BARSLASTCOUNT函数 - 统计连续满足条件的周期数
    
    通达信公式: BARSLASTCOUNT(X)
    含义: 统计连续满足X条件的周期数
    """
    result = pd.Series(index=condition.index, dtype=float)
    count = 0
    
    for i in range(len(condition)):
        if condition.iloc[i]:
            count += 1
        else:
            count = 0
        result.iloc[i] = count
        
    return result


# ==============================================================================
# 第二部分: 笔结构识别 (简化版)
# ==============================================================================

def identify_bi_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别笔结构 (简化版)
    
    由于无法调用通达信DLL，这里使用简化的分型识别方法：
    - 顶分型: 中间K线高点高于左右两根K线的高点
    - 底分型: 中间K线低点低于左右两根K线的低点
    - 笔: 连接相邻的顶分型和底分型
    
    参数:
        df: 包含 open, high, low, close 列的DataFrame
        
    返回:
        pd.DataFrame: 添加了笔结构信息的DataFrame
    """
    df = df.copy()
    H = df['high']
    L = df['low']
    C = df['close']
    
    # 识别分型
    # 顶分型: REF(H,1) > REF(H,2) AND REF(H,1) > H
    top_fractal = (REF(H, 1) > REF(H, 2)) & (REF(H, 1) > H)
    # 底分型: REF(L,1) < REF(L,2) AND REF(L,1) < L
    bottom_fractal = (REF(L, 1) < REF(L, 2)) & (REF(L, 1) < L)
    
    df['top_fractal'] = top_fractal.shift(-1).fillna(False)  # 调整到分型所在K线
    df['bottom_fractal'] = bottom_fractal.shift(-1).fillna(False)
    
    # 笔方向: 1=向上笔, -1=向下笔
    # 简化逻辑: 根据最近的分型类型判断
    df['bi_direction'] = 0
    
    last_fractal = 0  # 1=顶分型, -1=底分型
    for i in range(len(df)):
        if df['top_fractal'].iloc[i]:
            last_fractal = 1
        elif df['bottom_fractal'].iloc[i]:
            last_fractal = -1
        
        # 向上笔: 从底分型开始
        # 向下笔: 从顶分型开始
        if last_fractal == -1:
            df.iloc[i, df.columns.get_loc('bi_direction')] = 1  # 向上笔
        elif last_fractal == 1:
            df.iloc[i, df.columns.get_loc('bi_direction')] = -1  # 向下笔
    
    return df


def calculate_high_low_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算高低点序列
    
    根据笔结构计算GG(高点)和DD(低点)序列
    """
    df = df.copy()
    H = df['high']
    L = df['low']
    
    # 使用滚动窗口计算高低点
    # GG: 最近N个周期的最高点
    # DD: 最近N个周期的最低点
    
    # 简化版: 使用固定周期的高低点
    df['GG'] = HHV(H, 5)
    df['GG1'] = REF(df['GG'], 5)
    df['GG2'] = REF(df['GG'], 10)
    df['GG3'] = REF(df['GG'], 15)
    df['GG4'] = REF(df['GG'], 20)
    df['GG5'] = REF(df['GG'], 25)
    
    df['DD'] = LLV(L, 5)
    df['DD1'] = REF(df['DD'], 5)
    df['DD2'] = REF(df['DD'], 10)
    df['DD3'] = REF(df['DD'], 15)
    df['DD4'] = REF(df['DD'], 20)
    df['DD5'] = REF(df['DD'], 25)
    
    return df


# ==============================================================================
# 第三部分: 缠论5个买点计算
# ==============================================================================

def calculate_chan_buy_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算缠论5个买点
    
    买点类型:
    1. 一买 (chan_buy1) - 底背驰买点，五段下跌后的反转
    2. 二买 (chan_buy2) - 回踩确认买点，三段/五段下跌后的回踩不破前低
    3. 三买 (chan_buy3) - 中枢突破买点，回踩不进入中枢
    4. 强二买 (chan_strong_buy2) - 强势二买，特殊结构的二买变种
    5. 类二买 (chan_like_buy2) - 类似二买，与强二买逻辑相同
    
    参数:
        df: 包含 open, high, low, close, volume 列的DataFrame
        
    返回:
        pd.DataFrame: 添加了缠论买点指标列的DataFrame
    """
    df = df.copy()
    
    # 基础数据
    C = df['close']
    H = df['high']
    L = df['low']
    
    # 均线
    MA13 = EMA(C, 13)
    MA26 = EMA(C, 26)
    
    df['MA13'] = MA13
    df['MA26'] = MA26
    
    # 识别笔结构
    df = identify_bi_structure(df)
    df = calculate_high_low_sequence(df)
    
    # 方向判定 (简化版)
    # 1=向上笔, -1=向下笔
    direction = df['bi_direction']
    
    # 获取高低点序列
    GG = df['GG']
    GG1 = df['GG1']
    GG2 = df['GG2']
    GG3 = df['GG3']
    GG4 = df['GG4']
    
    DD = df['DD']
    DD1 = df['DD1']
    DD2 = df['DD2']
    DD3 = df['DD3']
    DD4 = df['DD4']
    
    # LL1: 距离上一个底分型的周期数 (简化为5)
    LL1 = 5
    HH1 = 5
    
    # =========================================================================
    # 一买: 底背驰买点
    # 原公式:
    # 一买TJ1 := 方向 = 1 AND L < MA13 AND LL1 <= 5;
    # 一五段下跌 := DD1 < GG1 AND DD1 < DD2 AND DD1 < DD3 AND GG1 < GG2 AND GG1 < GG3;
    # 一买TJA := GG1 < DD3;
    # 一买A := 一买TJ1 AND 一五段下跌 AND 一买TJA;
    # 一买TJB := GG1 > DD3;
    # 一买KJB := GG3 - DD3 > GG1 - DD1 AND GG3 - DD3 > GG2 - DD2 AND GG2 - DD2 < GG1 - DD1;
    # 一买B := 一买TJ1 AND 一五段下跌 AND 一买TJB AND 一买KJB;
    # 一买1 := 一买A OR 一买B;
    # =========================================================================
    
    buy1_tj1 = (direction == 1) & (L < MA13)
    five_down = (DD1 < GG1) & (DD1 < DD2) & (DD1 < DD3) & (GG1 < GG2) & (GG1 < GG3)
    
    buy1_tja = GG1 < DD3
    buy1_a = buy1_tj1 & five_down & buy1_tja
    
    buy1_tjb = GG1 > DD3
    buy1_kjb = ((GG3 - DD3) > (GG1 - DD1)) & ((GG3 - DD3) > (GG2 - DD2)) & ((GG2 - DD2) < (GG1 - DD1))
    buy1_b = buy1_tj1 & five_down & buy1_tjb & buy1_kjb
    
    df['chan_buy1'] = buy1_a | buy1_b
    
    # =========================================================================
    # 二买: 回踩确认买点
    # 原公式:
    # 买TJ1 := 方向 = 1 AND L < MA26 AND LL1 <= 8;
    # 二买TJ := DD1 < GG1 AND DD1 > DD2;
    # 三段下跌 := GG3 > GG2 AND DD3 > DD2;
    # 二买TJA1 := GG1 > DD3;
    # 二买A := 买TJ1 AND 二买TJ AND 三段下跌 AND 二买TJA1;
    # 五段下跌 := GG4 > GG3 AND GG4 > GG2 AND DD2 < DD3 AND DD2 < DD4;
    # 二买TJB1 := GG2 < DD4 AND GG1 > DD3;
    # 二买TJB2 := GG2 > DD4;
    # 二买B1 := 买TJ1 AND 二买TJ AND 五段下跌 AND 二买TJB1;
    # 二买B2 := 买TJ1 AND 二买TJ AND 五段下跌 AND 二买TJB2;
    # 二买1 := 二买A OR 二买B1 OR 二买B2;
    # =========================================================================
    
    buy_tj1 = (direction == 1) & (L < MA26)
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
    # 原公式:
    # 三买TJ := DD1 < GG1 AND DD1 > DD2;
    # 三买TJA1 := 方向 = 1 AND L < MA13 AND LL1 <= 5;
    # 三买TJA2 := DD1 > MIN(GG2, GG3) AND GG3 > DD2 AND DD4 < MAX(DD2, DD3) AND DD1 > DD4;
    # 三买1 := 三买TJ AND 三买TJA1 AND 三买TJA2;
    # =========================================================================
    
    buy3_tj = (DD1 < GG1) & (DD1 > DD2)
    buy3_tja1 = (direction == 1) & (L < MA13)
    buy3_tja2 = (DD1 > MIN(GG2, GG3)) & (GG3 > DD2) & (DD4 < MAX(DD2, DD3)) & (DD1 > DD4)
    
    df['chan_buy3'] = buy3_tj & buy3_tja1 & buy3_tja2
    
    # =========================================================================
    # 强二买: 强势二买
    # 原公式:
    # 强二买TJ := 方向 = 1 AND C < MA13 AND LL1 <= 8;
    # 强二买TJ2 := DD1 < GG1 AND DD3 < DD2 AND DD3 < DD1 AND DD3 < DD4;
    # 强二买KJ := GG2 - DD3 > GG2 - DD2 AND GG2 - DD3 > GG1 - DD1;
    # 强二买1 := 强二买TJ AND 强二买TJ2 AND 强二买KJ;
    # =========================================================================
    
    strong_buy2_tj = (direction == 1) & (C < MA13)
    strong_buy2_tj2 = (DD1 < GG1) & (DD3 < DD2) & (DD3 < DD1) & (DD3 < DD4)
    strong_buy2_kj = ((GG2 - DD3) > (GG2 - DD2)) & ((GG2 - DD3) > (GG1 - DD1))
    
    df['chan_strong_buy2'] = strong_buy2_tj & strong_buy2_tj2 & strong_buy2_kj
    
    # =========================================================================
    # 类二买: 类似二买 (与强二买逻辑相同)
    # 原公式: 类二买1 := 强二买1;
    # =========================================================================
    
    df['chan_like_buy2'] = df['chan_strong_buy2']
    
    # =========================================================================
    # 汇总: 任意买点
    # =========================================================================
    
    df['chan_any_buy'] = (
        df['chan_buy1'] | 
        df['chan_buy2'] | 
        df['chan_buy3'] | 
        df['chan_strong_buy2'] | 
        df['chan_like_buy2']
    )
    
    return df


def calculate_chan_sell_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算缠论3个卖点
    
    卖点类型:
    1. 一卖 (chan_sell1) - 顶背驰卖点
    2. 二卖 (chan_sell2) - 回抽确认卖点
    3. 三卖 (chan_sell3) - 中枢跌破卖点
    
    参数:
        df: 包含笔结构信息的DataFrame
        
    返回:
        pd.DataFrame: 添加了缠论卖点指标列的DataFrame
    """
    df = df.copy()
    
    # 基础数据
    C = df['close']
    H = df['high']
    L = df['low']
    
    # 均线
    MA13 = df['MA13'] if 'MA13' in df.columns else EMA(C, 13)
    
    # 方向判定
    direction = df['bi_direction'] if 'bi_direction' in df.columns else pd.Series(0, index=df.index)
    
    # 获取高低点序列
    GG1 = df['GG1'] if 'GG1' in df.columns else HHV(H, 5).shift(5)
    GG2 = df['GG2'] if 'GG2' in df.columns else HHV(H, 5).shift(10)
    GG3 = df['GG3'] if 'GG3' in df.columns else HHV(H, 5).shift(15)
    GG4 = df['GG4'] if 'GG4' in df.columns else HHV(H, 5).shift(20)
    
    DD1 = df['DD1'] if 'DD1' in df.columns else LLV(L, 5).shift(5)
    DD2 = df['DD2'] if 'DD2' in df.columns else LLV(L, 5).shift(10)
    DD3 = df['DD3'] if 'DD3' in df.columns else LLV(L, 5).shift(15)
    
    # =========================================================================
    # 一卖: 顶背驰卖点
    # 原公式:
    # 一卖TJ1:=方向=-1 AND H>MA13 AND HH1<=5;
    # 一五段上涨:=GG1>GG2 AND GG1>GG3 AND DD1>DD2 AND DD1>DD3;
    # 一卖TJA:=DD1>GG3;
    # 一卖A:=一卖TJ1 AND 一五段上涨 AND 一卖TJA;
    # 一卖TJB:=DD1<GG3;
    # 一卖B:=一卖TJ1 AND 一五段上涨 AND 一卖TJB;
    # 一卖C:=一卖TJ1 AND GG1>GG2 AND GG2>GG3 AND GG3>GG4;
    # 一卖1:=一卖A OR 一卖B OR 一卖C;
    # =========================================================================
    
    sell1_tj1 = (direction == -1) & (H > MA13)
    five_up = (GG1 > GG2) & (GG1 > GG3) & (DD1 > DD2) & (DD1 > DD3)
    
    sell1_tja = DD1 > GG3
    sell1_a = sell1_tj1 & five_up & sell1_tja
    
    sell1_tjb = DD1 < GG3
    sell1_b = sell1_tj1 & five_up & sell1_tjb
    
    sell1_c = sell1_tj1 & (GG1 > GG2) & (GG2 > GG3) & (GG3 > GG4)
    
    df['chan_sell1'] = sell1_a | sell1_b | sell1_c
    
    # =========================================================================
    # 二卖: 回抽确认卖点
    # 原公式:
    # 卖TJ1:=方向=-1 AND H>MA13 AND HH1<=8;
    # 二卖TJ:=GG1>DD1 AND GG1<GG2;
    # 三段上涨:=GG3<GG2 AND DD3<DD2;
    # 二卖A:=卖TJ1 AND 二卖TJ AND 三段上涨;
    # 二卖1:=二卖A;
    # =========================================================================
    
    sell_tj1 = (direction == -1) & (H > MA13)
    sell2_tj = (GG1 > DD1) & (GG1 < GG2)
    three_up = (GG3 < GG2) & (DD3 < DD2)
    
    df['chan_sell2'] = sell_tj1 & sell2_tj & three_up
    
    # =========================================================================
    # 三卖: 中枢跌破卖点
    # 原公式:
    # 三卖TJ:=DD1<GG1 AND GG1<GG2;
    # 三卖TJA1:=方向=-1 AND H>MA13 AND HH1<=5;
    # 三卖TJA2:=GG1<MAX(DD2,DD3);
    # 三卖1:=三卖TJ AND 三卖TJA1 AND 三卖TJA2;
    # =========================================================================
    
    sell3_tj = (DD1 < GG1) & (GG1 < GG2)
    sell3_tja1 = (direction == -1) & (H > MA13)
    sell3_tja2 = GG1 < MAX(DD2, DD3)
    
    df['chan_sell3'] = sell3_tj & sell3_tja1 & sell3_tja2
    
    # =========================================================================
    # 汇总: 任意卖点
    # =========================================================================
    
    df['chan_any_sell'] = df['chan_sell1'] | df['chan_sell2'] | df['chan_sell3']
    
    return df


def calculate_chan_theory_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算完整的缠论买卖点 (5买3卖)
    
    这是一个综合函数，依次调用买点和卖点计算函数。
    
    参数:
        df: 包含 open, high, low, close, volume 列的DataFrame
        
    返回:
        pd.DataFrame: 添加了所有缠论买卖点指标的DataFrame
        
    新增列:
        买点:
        - chan_buy1: 一买 (底背驰)
        - chan_buy2: 二买 (回踩确认)
        - chan_buy3: 三买 (中枢突破)
        - chan_strong_buy2: 强二买
        - chan_like_buy2: 类二买
        - chan_any_buy: 任意买点
        
        卖点:
        - chan_sell1: 一卖 (顶背驰)
        - chan_sell2: 二卖 (回抽确认)
        - chan_sell3: 三卖 (中枢跌破)
        - chan_any_sell: 任意卖点
    """
    df = calculate_chan_buy_points(df)
    df = calculate_chan_sell_points(df)
    
    return df


# ==============================================================================
# 第四部分: 工具函数
# ==============================================================================

def get_chan_signal_summary(df: pd.DataFrame) -> Dict[str, int]:
    """
    获取缠论信号统计摘要
    
    参数:
        df: 已计算信号的DataFrame
        
    返回:
        Dict: 各信号出现次数的字典
    """
    signal_cols = [
        'chan_buy1', 'chan_buy2', 'chan_buy3', 
        'chan_strong_buy2', 'chan_like_buy2',
        'chan_sell1', 'chan_sell2', 'chan_sell3'
    ]
    
    summary = {}
    for col in signal_cols:
        if col in df.columns:
            summary[col] = int(df[col].sum())
            
    return summary


# ==============================================================================
# 主程序入口 (用于测试)
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("缠论买卖点指标计算模块 - 测试运行")
    print("=" * 60)
    
    # 创建测试数据
    import numpy as np
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    base_price = 100
    prices = [base_price]
    
    for i in range(199):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)
    
    test_data = {
        'date': dates,
        'open': [p - np.random.rand() for p in prices],
        'high': [p + np.random.rand() * 2 for p in prices],
        'low': [p - np.random.rand() * 2 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 200)
    }
    df = pd.DataFrame(test_data)
    df.set_index('date', inplace=True)
    
    # 计算所有信号
    df = calculate_chan_theory_full(df)
    
    # 输出统计
    print("\n缠论信号统计:")
    summary = get_chan_signal_summary(df)
    for signal, count in summary.items():
        print(f"  {signal}: {count} 次")
    
    print("\n测试完成!")
