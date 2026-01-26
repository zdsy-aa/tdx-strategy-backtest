#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
纯离线 CSV -> 信号/策略整合 -> 输出 JSON（给网页展示）
抛弃 data_fetcher / ml_models，不依赖任何项目内其他模块。

输入目录（默认）：
  \\data\\day\\bj
  \\data\\day\\sz
  \\data\\day\\sh

输出目录（默认）：
  \\web\\client\\src\\data
  - dashboard.json
  - errors.json
  - symbols/<code>.json

CSV 列（中文）：
  名称,日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率

你要整合的"三个脚本核心信号/策略"：
  填入下方 Strategy Slot A / B / C 的函数 strategy_core_A / strategy_core_B / strategy_core_C
  不要改它们的函数签名与返回结构，这样网页展示层稳定。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("offline_signal_json_pipeline")

# =========================
# 1) 基础配置与字段规范化
# =========================

CSV_COLUMNS_CN = [
    "名称", "日期", "开盘", "收盘", "最高", "最低",
    "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率",
]

CN_TO_EN = {
    "名称": "name",
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}

REQUIRED_EN = ["date", "open", "close", "high", "low", "volume"]

NUMERIC_EN = [
    "open", "close", "high", "low",
    "volume", "amount", "amplitude",
    "pct_change", "change", "turnover",
]


def normalize_path(p: str) -> Path:
    """
    标准化路径，支持 Windows 和 Linux 跨平台兼容。
    
    功能：
    1. 将 Windows 反斜杠转换为正斜杠
    2. 处理相对路径（相对于项目根目录）
    3. 正确处理 .. 目录导航
    4. 返回绝对路径
    
    参数：
        p: 输入路径字符串（可以是 Windows 或 Unix 格式）
    
    返回：
        Path: 解析后的绝对路径对象
    """
    # 将 Windows 路径分隔符转换为 Unix 风格
    p_str = p.replace("\\", "/")
    
    # 处理绝对路径
    if os.path.isabs(p_str):
        return Path(p_str).resolve()
    
    # 处理相对路径
    # 获取当前脚本所在目录的父目录作为项目根目录
    project_root = Path(__file__).resolve().parent.parent
    
    # 使用 / 作为分隔符，正确处理 .. 和 .
    # Path 对象会自动处理 .. 和 . 的导航
    path = project_root / p_str
    
    return path.resolve()


def _to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return str(x)


def safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def read_day_csv(csv_path: Path, code: str, market: str) -> pd.DataFrame:
    """
    读取 CSV -> 标准化列名（中文->英文）-> 类型转换 -> 排序去重
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    # 映射到英文列
    df = df.rename(columns={c: CN_TO_EN.get(c, c) for c in df.columns})

    # 必须列检查
    missing = [c for c in REQUIRED_EN if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required cols={missing} file={csv_path}")

    # date 解析
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # 数值列
    for c in NUMERIC_EN:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 元信息
    df["stock_code"] = code
    df["market"] = market

    # 排序去重
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return df


def discover_csv_files(data_root: Path) -> List[Tuple[str, str, Path]]:
    """
    返回 (market, code, csv_path)
    """
    items: List[Tuple[str, str, Path]] = []
    for market in ("bj", "sz", "sh"):
        d = data_root / market
        if not d.exists():
            logger.warning("Market dir missing: %s", d)
            continue
        for p in sorted(d.glob("*.csv")):
            code = p.stem.strip()
            if code:
                items.append((market, code, p))
    return items


# =========================
# 2) 统一的指标工具（可选）
#    你两份策略如果需要常用指标，可直接复用这些小工具
# =========================

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(n, min_periods=n).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(n, min_periods=n).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ef = ema(close, fast)
    es = ema(close, slow)
    diff = ef - es
    dea = ema(diff, signal)
    hist = diff - dea
    return diff, dea, hist

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def volume_ratio(volume: pd.Series, n: int = 5) -> pd.Series:
    vma = volume.rolling(n, min_periods=n).mean()
    return volume / (vma + 1e-12)

# --- a99_indicators.py 基础函数 ---

def REF(series: pd.Series, n: int = 1) -> pd.Series:
    """REF函数 - 引用N周期前的数据"""
    return series.shift(n)

def MA(series: pd.Series, n: int) -> pd.Series:
    """MA函数 - 简单移动平均"""
    return series.rolling(window=n, min_periods=1).mean()

def EMA_A99(series: pd.Series, n: int) -> pd.Series:
    """EMA函数 - 指数移动平均 (使用 A99 的命名避免冲突)"""
    return series.ewm(span=n, adjust=False).mean()

def SMA_A99(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """SMA函数 - 移动平均 (通达信特有) (使用 A99 的命名避免冲突)"""
    arr = series.to_numpy(dtype=float, na_value=np.nan)
    if arr.size == 0:
        return pd.Series(dtype=float)
    result_arr = np.empty_like(arr, dtype=float)
    result_arr[0] = arr[0]
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            result_arr[i] = result_arr[i-1]
        else:
            result_arr[i] = (m * arr[i] + (n - m) * result_arr[i-1]) / n
    return pd.Series(result_arr, index=series.index)

def HHV(series: pd.Series, n: int) -> pd.Series:
    """HHV函数 - N周期内最高值"""
    return series.rolling(window=n, min_periods=1).max()

def LLV(series: pd.Series, n: int) -> pd.Series:
    """LLV函数 - N周期内最低值"""
    return series.rolling(window=n, min_periods=1).min()

def CROSS(a: pd.Series, b: pd.Series) -> pd.Series:
    """CROSS函数 - 黄金交叉判断"""
    a_prev = a.shift(1, fill_value=a.iloc[0])
    b_prev = b.shift(1, fill_value=b.iloc[0])
    return (a_prev < b_prev) & (a >= b)

def COUNT(condition: pd.Series, n: int) -> pd.Series:
    """COUNT函数 - 统计N周期内条件满足次数"""
    return condition.rolling(window=n, min_periods=1).sum()

def ABS(series: pd.Series) -> pd.Series:
    """取绝对值"""
    return series.abs()

def MAX(a, b):
    """取大值"""
    return a.combine(b, max)

def IF(condition: pd.Series, true_val, false_val):
    """IF函数 - 条件选择函数"""
    return condition.apply(lambda x: true_val if x else false_val)

# --- a99_indicators.py 四大高级指标计算函数 ---

def calculate_six_veins(df: pd.DataFrame) -> pd.DataFrame:
    """计算六脉神剑指标"""
    df = df.copy()
    C = df['close']; H = df['high']; L = df['low']
    # --------------- MACD ---------------
    DIF = EMA_A99(C, 12) - EMA_A99(C, 26)
    DEA = EMA_A99(DIF, 9)
    df['macd_red'] = DIF > DEA
    # --------------- KDJ ---------------
    RSV = (C - LLV(L, 9)) / (HHV(H, 9) - LLV(L, 9) + 0.0001) * 100
    K = SMA_A99(RSV, 3, 1); D = SMA_A99(K, 3, 1)
    df['kdj_red'] = K > D
    # --------------- RSI ---------------
    RSI5 = SMA_A99(MAX(C - REF(C, 1), pd.Series(0, index=df.index)), 5, 1)
    RSI13 = SMA_A99(MAX(C - REF(C, 1), pd.Series(0, index=df.index)), 13, 1)
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

def calculate_buy_sell_points(df: pd.DataFrame, M: int = 55, N: int = 34) -> pd.DataFrame:
    """计算买卖点指标"""
    df = df.copy()
    C = df['close']; H = df['high']; L = df['low']
    # 散户线计算
    HHM = HHV(H, M); LLM = LLV(L, M)
    df['retail'] = 100 * (HHM - C) / (HHM - LLM + 0.0001)
    # 庄家线计算
    HHN = HHV(H, N); LLN = LLV(L, N)
    RSV = (C - LLN) / (HHN - LLN + 0.0001) * 100
    K = SMA_A99(RSV, 3, 1); D = SMA_A99(K, 3, 1); J = 3 * K - 2 * D
    df['banker'] = EMA_A99(J, 6)
    # 吸筹指标计算 (简化版)
    VAR1 = (C - LLV(L, 30)) / (HHV(H, 30) - LLV(L, 30) + 0.0001) * 100
    VAR2 = SMA_A99(VAR1, 3, 1)
    # 修复：确保 IF 函数返回的是 Series，且 EMA_A99 接收 Series
    # IF(L <= LLV(L, 30), VAR2, 0)
    cond = L <= LLV(L, 30)
    var2_series = pd.Series(VAR2, index=df.index)
    result_series = pd.Series(np.where(cond, var2_series, 0), index=df.index)
    df['accumulate'] = EMA_A99(result_series, 3) / 10
    # 买卖信号计算
    df['buy2'] = (df['banker'] > df['retail']) & (df['banker'].shift(1) <= df['retail'].shift(1))
    df['sell2'] = (df['banker'] < df['retail']) & (df['banker'].shift(1) >= df['retail'].shift(1))
    return df

def calculate_chan_theory(df: pd.DataFrame) -> pd.DataFrame:
    """计算缠论指标"""
    df = df.copy()
    C = df['close']; H = df['high']; L = df['low']
    MA5 = MA(C, 5); MA13 = MA(C, 13); MA34 = MA(C, 34)
    
    # 笔方向判断
    direction_vals = np.zeros(len(df))
    for i in range(1, len(df)):
        if H.iloc[i] > H.iloc[i-1]:
            direction_vals[i] = 1
        elif L.iloc[i] < L.iloc[i-1]:
            direction_vals[i] = -1
        else:
            direction_vals[i] = direction_vals[i-1]
    df['bi_direction'] = direction_vals
    
    # 计算高低点
    GG1 = HHV(H, 5); GG2 = HHV(H, 10); GG3 = HHV(H, 20); GG4 = HHV(H, 30)
    DD1 = LLV(L, 5); DD2 = LLV(L, 10); DD3 = LLV(L, 20); DD4 = LLV(L, 30)
    
    # 一买: 向上笔完成后第一次回调后反弹
    buy1_tj = (df['bi_direction'] == 1) & (C < MA13)
    buy1_tja = (DD1 > DD2) & (DD2 > DD3)
    buy1_tjb = GG1 > GG2
    df['chan_buy1'] = buy1_tj & buy1_tja & buy1_tjb
    
    # 二买: 中枢震荡后向上突破
    buy_tj1 = (df['bi_direction'] == 1) & (C < MA13)
    buy2_tj = (GG1 < GG2) & (DD1 > DD2)
    three_down = (DD2 > DD3) & (GG2 < GG3)
    buy2_tja1 = GG1 > DD3
    buy2_a = buy_tj1 & buy2_tj & three_down & buy2_tja1
    five_down_v2 = (GG4 > GG3) & (GG4 > GG2) & (DD2 < DD3) & (DD2 < DD4)
    buy2_tjb1 = (GG2 < DD4) & (GG1 > DD3)
    buy2_tjb2 = GG2 > DD4
    buy2_b1 = buy_tj1 & buy2_tj & five_down_v2 & buy2_tjb1
    buy2_b2 = buy_tj1 & buy2_tj & five_down_v2 & buy2_tjb2
    df['chan_buy2'] = buy2_a | buy2_b1 | buy2_b2
    # 三买: 中枢突破买点
    buy3_tj = (DD1 < GG1) & (DD1 > DD2)
    buy3_tja1 = (df['bi_direction'] == 1) & (L < MA13)
    min_gg2_gg3 = pd.concat([GG2, GG3], axis=1).min(axis=1)
    max_dd2_dd3 = pd.concat([DD2, DD3], axis=1).max(axis=1)
    buy3_tja2 = (DD1 > min_gg2_gg3) & (GG3 > DD2) & (DD4 < max_dd2_dd3) & (DD1 > DD4)
    df['chan_buy3'] = buy3_tj & buy3_tja1 & buy3_tja2
    # 强二买: 强势二买
    strong_buy2_tj = (df['bi_direction'] == 1) & (C < MA13)
    strong_buy2_tj2 = (DD1 < GG1) & (DD3 < DD2) & (DD3 < DD1) & (DD3 < DD4)
    strong_buy2_kj = ((GG2 - DD3) > (GG2 - DD2)) & ((GG2 - DD3) > (GG1 - DD1))
    df['chan_strong_buy2'] = strong_buy2_tj & strong_buy2_tj2 & strong_buy2_kj
    # 类二买: 类似二买 (与强二买逻辑相同)
    df['chan_like_buy2'] = df['chan_strong_buy2']
    # 卖点逻辑（仅保留汇总，不作为评分项）
    sell1_tj1 = (df['bi_direction'] == -1) & (H > MA13)
    five_up = (GG1 > GG2) & (GG1 > GG3) & (DD1 > DD2) & (DD1 > DD3)
    sell1_tja = DD1 > GG3
    sell1_a = sell1_tj1 & five_up & sell1_tja
    sell1_tjb = DD1 < GG3
    sell1_b = sell1_tj1 & five_up & sell1_tjb
    sell1_c = sell1_tj1 & (GG1 > GG2) & (GG2 > GG3) & (GG3 > GG4)
    df['chan_sell1'] = sell1_a | sell1_b | sell1_c
    sell_tj1 = (direction_vals == -1) & (H > MA13)
    sell2_tj = (GG1 > DD1) & (GG1 < GG2)
    three_up = (GG3 < GG2) & (DD3 < DD2)
    df['chan_sell2'] = sell_tj1 & sell2_tj & three_up
    sell3_tj = (DD1 < GG1) & (GG1 < GG2)
    sell3_tja1 = (df['bi_direction'] == -1) & (H > MA13)
    sell3_tja2 = GG1 < max_dd2_dd3
    df['chan_sell3'] = sell3_tj & sell3_tja1 & sell3_tja2
    # 汇总信号
    df['chan_any_buy'] = (
        df['chan_buy1'] | 
        df['chan_buy2'] | 
        df['chan_buy3'] | 
        df['chan_strong_buy2'] | 
        df['chan_like_buy2']
    )
    df['chan_any_sell'] = df['chan_sell1'] | df['chan_sell2'] | df['chan_sell3']
    return df


# =========================
# 3) 信号输出的统一格式（网页稳定）
# =========================

def make_signal(
    date: pd.Timestamp,
    sig_type: str,
    strength: int,
    score: float,
    message: str,
    extras: Optional[Dict[str, Any]] = None,
    source: str = "A",
) -> Dict[str, Any]:
    """
    signal 标准结构：
      date: 时间
      type: 信号类型（字符串）
      strength: 1~5 强度（你自己定标，但保持一致）
      score: 0~100 用于排序/展示
      message: 简短说明
      source: A/B/C（来自哪个策略槽）
      extras: 任意扩展字段（网页可折叠展示）
    """
    obj = {
        "date": date,
        "type": sig_type,
        "strength": int(strength),
        "score": float(score),
        "message": str(message),
        "source": str(source),
    }
    if extras:
        obj["extras"] = extras
    return obj


# =========================
# 4) 三个"核心策略槽"：把你那三个脚本的核心逻辑搬到这里
# =========================

def strategy_core_A(df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Strategy Slot A：趋势跟踪策略 (MA交叉)
    """
    if len(df) < 60:
        return 0.0, []

    close = df["close"]
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)

    # 金叉信号
    cross_up = (ma5.shift(1) < ma10.shift(1)) & (ma5 >= ma10)
    # 死叉信号
    cross_down = (ma5.shift(1) > ma10.shift(1)) & (ma5 < ma10)

    signals = []
    score = 0.0

    if cross_up.iloc[-1]:
        score = max(score, 60.0)
        signals.append(make_signal(
            date=df["date"].iloc[-1],
            sig_type="A_MA_CROSS_UP",
            strength=3,
            score=60.0,
            message="A策略：MA5 上穿 MA10（金叉）",
            extras={"ma5": float(ma5.iloc[-1]), "ma10": float(ma10.iloc[-1])},
            source="A",
        ))

    if cross_down.iloc[-1]:
        score = max(score, 30.0)
        signals.append(make_signal(
            date=df["date"].iloc[-1],
            sig_type="A_MA_CROSS_DOWN",
            strength=2,
            score=30.0,
            message="A策略：MA5 下穿 MA10（死叉）",
            extras={"ma5": float(ma5.iloc[-1]), "ma10": float(ma10.iloc[-1])},
            source="A",
        ))

    return score, signals


def strategy_core_B(df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Strategy Slot B：缠论买点
    """
    if len(df) < 60:
        return 0.0, []

    df = calculate_chan_theory(df)

    signals = []
    score = 0.0
    latest = df.iloc[-1]

    # 一买
    if latest['chan_buy1']:
        score = max(score, 50.0)
        signals.append(make_signal(
            date=df["date"].iloc[-1],
            sig_type="B_CHAN_BUY1",
            strength=2,
            score=50.0,
            message="B策略：缠论一买",
            source="B",
        ))

    # 二买
    if latest['chan_buy2']:
        score = max(score, 60.0)
        signals.append(make_signal(
            date=df["date"].iloc[-1],
            sig_type="B_CHAN_BUY2",
            strength=3,
            score=60.0,
            message="B策略：缠论二买",
            source="B",
        ))

    # 三买
    if latest['chan_buy3']:
        score = max(score, 55.0)
        signals.append(make_signal(
            date=df["date"].iloc[-1],
            sig_type="B_CHAN_BUY3",
            strength=3,
            score=55.0,
            message="B策略：缠论三买",
            source="B",
        ))

    # 强二买
    if latest['chan_strong_buy2']:
        score = max(score, 80.0)
        signals.append(make_signal(
            date=df["date"].iloc[-1],
            sig_type="B_CHAN_STRONG_BUY2",
            strength=4,
            score=80.0,
            message="B策略：缠论强二买",
            source="B",
        ))

    return score, signals


def strategy_core_C(df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Strategy Slot C：六脉神剑 + 买卖点
    """
    if len(df) < 60:
        return 0.0, []

    df = calculate_six_veins(df)
    df = calculate_buy_sell_points(df)

    signals = []
    score = 0.0
    latest = df.iloc[-1]

    # 六脉神剑共振买入
    if latest['six_veins_buy']:
        score = max(score, 70.0)
        signals.append(make_signal(
            date=df["date"].iloc[-1],
            sig_type="C_SIX_VEINS",
            strength=4,
            score=70.0,
            message="C策略：六脉神剑共振买入",
            extras={"red_count": int(latest['six_veins_count'])},
            source="C",
        ))

    # 黄金摇钱树 (中优先级，分值 60)
    if latest['money_tree']:
        score = max(score, 60.0)
        signals.append(make_signal(
            date=df["date"].iloc[-1],
            sig_type="C_MONEY_TREE",
            strength=3,
            score=60.0,
            message="C策略：黄金摇钱树选股信号",
            extras={"xg55": bool(latest['xg55']), "xg66": bool(latest['xg66']), "xg88": bool(latest['xg88'])},
            source="C",
        ))

    # 买卖点 (庄家上穿散户，分值 50)
    if latest['buy2']:
        score = max(score, 50.0)
        signals.append(make_signal(
            date=df["date"].iloc[-1],
            sig_type="C_BANKER_CROSS",
            strength=2,
            score=50.0,
            message="C策略：庄家线上穿散户线",
            extras={"banker": float(latest['banker']), "retail": float(latest['retail'])},
            source="C",
        ))

    return score, signals


# =========================
# 5) 策略融合与 JSON 产出
# =========================

def build_symbol_payload(
    market: str,
    code: str,
    csv_path: Path,
    recent_bars: int,
    include_series: bool,
    series_bars: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    返回 (symbol_payload, error_payload)
    """
    try:
        df = read_day_csv(csv_path, code=code, market=market)

        if recent_bars > 0 and len(df) > recent_bars:
            df = df.iloc[-recent_bars:].copy()

        if len(df) < 60: # 提高最低数据要求以满足高级指标计算
            # 静默跳过数据不足 60 天的股票
            return None, None

        name = None
        if "name" in df.columns and df["name"].notna().any():
            name = df["name"].dropna().iloc[-1]

        # 三策略分别跑
        score_a, sig_a = strategy_core_A(df)
        score_b, sig_b = strategy_core_B(df)
        score_c, sig_c = strategy_core_C(df) # 新增策略 C

        # 合并信号：按 date desc、score desc
        all_signals = sorted(
            sig_a + sig_b + sig_c,
            key=lambda x: (x["date"], x["score"]),
            reverse=True,
        )

        # 综合评分：简单相加
        final_score = score_a + score_b + score_c

        payload = {
            "meta": {
                "market": market,
                "code": code,
                "name": name,
                "source_file": str(csv_path),
                "last_date": df["date"].iloc[-1],
            },
            "score": {
                "final_score": final_score,
                "score_A": score_a,
                "score_B": score_b,
                "score_C": score_c, # 新增 score_C
            },
            "signals": all_signals,
        }

        if include_series:
            # 仅保留最近 series_bars 根 K 线数据
            series_df = df.iloc[-series_bars:].copy()
            
            # 仅保留核心列，避免 JSON 过大
            series_df = series_df[["date", "open", "close", "high", "low", "volume", "amount"]]
            
            # 转换为 JSON 友好的格式
            payload["series"] = series_df.to_dict(orient="records")

        return payload, None

    except Exception as e:
        error_payload = {
            "market": market,
            "code": code,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        return None, error_payload


# ... (main 函数保持不变) ...
def main():
    parser = argparse.ArgumentParser(
        description="纯离线 CSV -> 信号/策略整合 -> 输出 JSON (给网页展示)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/day",
        help="CSV 数据根目录 (e.g., ../data/day)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="web/client/src/data",
        help="JSON 输出目录 (e.g., web/client/src/data)",
    )
    parser.add_argument(
        "--recent-bars",
        type=int,
        default=0,
        help="处理时仅保留最近 N 根 K 线数据 (0=全部)",
    )
    parser.add_argument(
        "--series-bars",
        type=int,
        default=120,
        help="个股 JSON 中包含的 K 线数据根数",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="最大处理股票数量 (0=全部)",
    )
    parser.add_argument(
        "--dashboard-days",
        type=int,
        default=3,
        help="仪表盘统计的最近天数 (0=全部)",
    )
    
    args = parser.parse_args()

    data_root = normalize_path(args.data_root)
    out_dir = normalize_path(args.out_dir)
    
    if not data_root.is_dir():
        logger.error("Data root directory not found: %s", data_root)
        return

    # 发现所有 CSV 文件
    all_symbols = discover_csv_files(data_root)
    
    if args.max_symbols > 0:
        all_symbols = all_symbols[:args.max_symbols]

    logger.info("Found %d symbols to process.", len(all_symbols))

    dashboard_data: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "csv_columns_cn": CSV_COLUMNS_CN,
        "markets": {"bj": {"total": 0, "ok": 0, "fail": 0}, "sz": {"total": 0, "ok": 0, "fail": 0}, "sh": {"total": 0, "ok": 0, "fail": 0}},
        "counts": {"symbols_total": len(all_symbols), "symbols_ok": 0, "symbols_fail": 0},
        "top": [],
        "dashboard_days": args.dashboard_days,
    }
    
    errors: List[Dict[str, Any]] = []
    
    # 确保 symbols 目录存在
    symbols_out_dir = out_dir / "symbols"
    symbols_out_dir.mkdir(parents=True, exist_ok=True)

    for market, code, csv_path in all_symbols:
        dashboard_data["markets"][market]["total"] += 1
        
        payload, error_payload = build_symbol_payload(
            market=market,
            code=code,
            csv_path=csv_path,
            recent_bars=args.recent_bars,
            include_series=True,
            series_bars=args.series_bars,
        )

        if payload:
            # 写入个股 JSON
            symbol_json_path = symbols_out_dir / f"{code}.json"
            safe_write_json(symbol_json_path, payload)
            
            # 更新 dashboard 数据
            dashboard_data["counts"]["symbols_ok"] += 1
            dashboard_data["markets"][market]["ok"] += 1
            
            # 仅将有信号的股票加入 top 列表
            if payload["score"]["final_score"] > 0:
                # 检查信号是否在最近 N 天内
                last_date = pd.to_datetime(payload["meta"]["last_date"])
                cutoff_date = None
                if args.dashboard_days > 0:
                    cutoff_date = datetime.utcnow() - pd.Timedelta(days=args.dashboard_days)
                    cutoff_date = pd.to_datetime(cutoff_date)
                
                # 如果启用了天数限制，检查最后日期是否在范围内
                if cutoff_date is None or last_date >= cutoff_date:
                    top_item = {
                        "market": market,
                        "code": code,
                        "name": payload["meta"]["name"],
                        "last_date": payload["meta"]["last_date"],
                        "final_score": payload["score"]["final_score"],
                        "score_A": payload["score"]["score_A"],
                        "score_B": payload["score"]["score_B"],
                        "score_C": payload["score"]["score_C"], # 新增 score_C
                        "signals_count": len(payload["signals"]),
                    }
                    dashboard_data["top"].append(top_item)
            
            logger.info("Processed %s (%s): Score=%.2f, Signals=%d", code, payload["meta"]["name"], payload["score"]["final_score"], len(payload["signals"]))

        elif error_payload:
            dashboard_data["counts"]["symbols_fail"] += 1
            dashboard_data["markets"][market]["fail"] += 1
            errors.append(error_payload)
            logger.error("Failed to process %s: %s", code, error_payload["error"])

    # 排序 top 列表
    dashboard_data["top"].sort(key=lambda x: x["final_score"], reverse=True)

    # 写入 dashboard.json
    safe_write_json(out_dir / "dashboard.json", dashboard_data)
    
    # 写入 errors.json
    safe_write_json(out_dir / "errors.json", errors)

    logger.info("Pipeline finished. Success: %d, Fail: %d", dashboard_data["counts"]["symbols_ok"], dashboard_data["counts"]["symbols_fail"])


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
