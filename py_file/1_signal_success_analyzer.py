#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
脚本 1（内存安全版）：
    1_signal_success_analyzer_partitioned.py

【核心职责】
    - 从 data/day/*.csv 中提取「信号级样本」
    - 计算未来 N 日收益 + 成功标签
    - 输出为 Parquet，并按 year 分区（为 Walk-forward 服务）

【重要原则】
    ❌ 不训练模型
    ❌ 不做阈值判断
    ✅ 只生成：样本 + 特征 + 标签

【输出结构】
    output/signal_samples_parquet/
        ├── year=2016/part-*.parquet
        ├── year=2017/part-*.parquet
        └── ...

【为什么要 year 分区】
    👉 脚本 2 只需读取：某一年 valid + 前 N 年 train
    👉 避免一次性加载千万行数据
===============================================================================
"""

import os
import warnings
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from indicators import calculate_all_signals

warnings.simplefilter("ignore", category=FutureWarning)

# =============================================================================
# 日志工具
# =============================================================================

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# =============================================================================
# 路径配置
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "day")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "signal_samples_parquet")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 参数区（工程可调）
# =============================================================================

MIN_BARS = 80            # 最少历史K线
FUTURE_DAYS = 20         # 未来观察窗口
GAIN_THRESHOLD = 5.0     # 成功涨幅阈值（%）
COOLDOWN = 5             # 信号冷却期（交易日）

NUM_WORKERS = 12          # 并行进程数（16G 推荐 4~8）
FLUSH_EVERY = 300        # 单进程累计多少条样本后落盘

# =============================================================================
# 中文列名映射
# =============================================================================

CN_COL_MAP = {
    "名称": "name",
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_chg",
    "涨跌额": "chg",
    "换手率": "turnover",
}

# =============================================================================
# 特征列（⚠ 必须与脚本 2 完全一致）
# =============================================================================

FEATURE_COLS = [
    "close", "volume", "MA13", "MA26",
    "six_veins_count", "banker", "retail", "accumulate",
    "pct_chg", "amplitude", "turnover",
    "macd_red", "kdj_red", "rsi_red", "bbi_red",
]

# =============================================================================
# 工具函数
# =============================================================================

def apply_cooldown(idxs, cooldown):
    """对信号索引应用冷却期，只保留间隔 >= cooldown 的第一个"""
    keep, last = [], -9999
    for i in idxs:
        if i - last >= cooldown:
            keep.append(i)
            last = i
    return keep


def calc_future_gain(df: pd.DataFrame) -> pd.Series:
    """same-day 买入，未来 FUTURE_DAYS 内最大涨幅"""
    future_high = (
        df["high"]
        .shift(-1)
        .rolling(FUTURE_DAYS, min_periods=FUTURE_DAYS)
        .max()
        .shift(-(FUTURE_DAYS - 1))
    )
    return (future_high - df["close"]) / df["close"] * 100

# =============================================================================
# 单股票处理（子进程）
# =============================================================================

def process_one(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={k: v for k, v in CN_COL_MAP.items() if k in df.columns})

        if "date" not in df.columns:
            return None

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        if len(df) < MIN_BARS:
            return None

        # 计算所有指标 & 信号
        df = calculate_all_signals(df)

        # 自动识别买点类信号
        signal_map = {}
        for col in df.columns:
            cl = col.lower()
            if "sell" in cl:
                continue

            if df[col].dtype == bool and df[col].any():
                signal_map[col] = df.index[df[col]].tolist()
            elif any(k in cl for k in ["buy", "six_veins", "combo"]):
                try:
                    idxs = df.index[df[col] > 0].tolist()
                    if idxs:
                        signal_map[col] = idxs
                except:
                    pass

        if not signal_map:
            return None

        df["future_gain_20d"] = calc_future_gain(df)
        df["success_20d"] = (df["future_gain_20d"] >= GAIN_THRESHOLD).astype("int8")

        stock = os.path.splitext(os.path.basename(csv_path))[0]
        name = df["name"].iloc[-1] if "name" in df.columns else ""

        records = []

        for sig, idxs in signal_map.items():
            idxs = apply_cooldown(idxs, COOLDOWN)
            for i in idxs:
                if pd.isna(df.at[i, "future_gain_20d"]):
                    continue

                rec = {
                    "stock": stock,
                    "name": name,
                    "signal": sig,
                    "date": df.at[i, "date"],
                    "year": int(df.at[i, "date"].year),
                    "entry_price": df.at[i, "close"],
                    "future_gain_20d": df.at[i, "future_gain_20d"],
                    "success_20d": df.at[i, "success_20d"],
                }

                for f in FEATURE_COLS:
                    rec[f] = df.at[i, f] if f in df.columns else np.nan

                records.append(rec)

        return pd.DataFrame(records) if records else None

    except Exception:
        return None

# =============================================================================
# 主程序
# =============================================================================

def main():
    log("脚本1启动：生成信号样本（year 分区）")

    csvs = [
        os.path.join(r, f)
        for r, _, fs in os.walk(DATA_DIR)
        for f in fs if f.endswith(".csv")
    ]
    log(f"发现 CSV 数量: {len(csvs)}")
    log(f"并行进程数: {NUM_WORKERS}")

    buffer = []

    with Pool(NUM_WORKERS) as pool:
        for res in tqdm(
            pool.imap_unordered(process_one, csvs),
            total=len(csvs),
            desc="生成信号样本"
        ):
            if res is not None and not res.empty:
                buffer.append(res)

            if len(buffer) >= FLUSH_EVERY:
                flush_buffer(buffer)
                buffer.clear()

    if buffer:
        flush_buffer(buffer)

    log("脚本1结束")


def flush_buffer(buffer):
    """将 buffer 中的数据按 year 分区写入 parquet"""
    df = pd.concat(buffer, ignore_index=True)

    # 降内存（非常关键）
    for c in FEATURE_COLS:
        df[c] = df[c].astype("float32")

    df["success_20d"] = df["success_20d"].astype("int8")

    df.to_parquet(
        OUTPUT_DIR,
        partition_cols=["year"],
        index=False
    )

if __name__ == "__main__":
    main()
