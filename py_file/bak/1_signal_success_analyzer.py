#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
脚本名：1_signal_success_analyzer.py

【脚本功能】
    全市场「信号 + 特征 + 标签」样本生成器（same-day 买入）

【核心作用】
    1. 遍历 data/day 下所有股票日线 CSV
    2. 调用 indicators.calculate_all_signals 生成：
        - 所有技术指标
        - 所有买卖信号
    3. 自动识别“买点类信号”
    4. same-day 买入
    5. 计算未来 N 日最大涨幅，生成成功标签
    6. 输出【可直接用于机器学习】的样本 CSV

【本脚本输出的 CSV = 第二阶段唯一输入】
===============================================================================
"""

import os
import warnings
from multiprocessing import Pool

import pandas as pd
import numpy as np
from tqdm import tqdm

from indicators import calculate_all_signals

warnings.simplefilter("ignore", category=FutureWarning)

# =============================================================================
# 1) 路径配置
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "day")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "signal_samples_same_day.csv")

# =============================================================================
# 2) 可调研究参数（作用说明一定要看）
# =============================================================================

MIN_BARS = 80
"""
最少历史K线数：
- 少于该数量的股票不参与研究
- 防止新股 / 数据残缺
"""

FUTURE_DAYS = 20
"""
未来观察窗口（交易日）：
- same-day 买入
- 向后看 FUTURE_DAYS 内的最高价
"""

GAIN_THRESHOLD = 5.0
"""
成功阈值（百分比）：
- future_gain >= GAIN_THRESHOLD → success_20d = 1
"""

COOLDOWN = 5
"""
信号冷却期（交易日）：
- 同一信号短期内多次触发，只保留第一个
"""

NUM_WORKERS = 12
CHUNK_SIZE = 20
FLUSH_EVERY = 100

# =============================================================================
# 3) 中文 → 英文列名映射
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
# 4) 机器学习特征列（★ 脚本2会直接使用 ★）
# =============================================================================

FEATURE_COLS = [
    "close",
    "volume",
    "MA13",
    "MA26",
    "six_veins_count",
    "banker",
    "retail",
    "accumulate",
]

# =============================================================================
# 5) 工具函数
# =============================================================================

def apply_cooldown(idxs, cooldown):
    keep, last = [], -9999
    for i in idxs:
        if i - last >= cooldown:
            keep.append(i)
            last = i
    return keep


def calc_future_gain(df):
    future_high = (
        df["high"]
        .shift(-1)
        .rolling(FUTURE_DAYS, min_periods=FUTURE_DAYS)
        .max()
        .shift(-(FUTURE_DAYS - 1))
    )
    return (future_high - df["close"]) / df["close"] * 100


# =============================================================================
# 6) 单股票处理逻辑
# =============================================================================

def process_one(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={k: v for k, v in CN_COL_MAP.items() if k in df.columns})

        if "date" not in df.columns:
            return None

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        if len(df) < MIN_BARS:
            return None

        df = calculate_all_signals(df)

        # 自动识别买点信号
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
        df["success_20d"] = (df["future_gain_20d"] >= GAIN_THRESHOLD).astype(int)

        records = []
        stock = os.path.splitext(os.path.basename(csv_path))[0]
        name = df["name"].iloc[-1] if "name" in df.columns else ""

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
                    "entry_price": df.at[i, "close"],
                    "future_gain_20d": df.at[i, "future_gain_20d"],
                    "success_20d": df.at[i, "success_20d"],
                }

                # ★ 写入所有 ML 特征
                for f in FEATURE_COLS:
                    rec[f] = df.at[i, f] if f in df.columns else np.nan

                records.append(rec)

        return pd.DataFrame(records) if records else None

    except Exception:
        return None


# =============================================================================
# 7) 主程序
# =============================================================================

def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    csvs = []
    for r, _, fs in os.walk(DATA_DIR):
        for f in fs:
            if f.endswith(".csv"):
                csvs.append(os.path.join(r, f))

    buffer, header = [], True

    with Pool(NUM_WORKERS) as pool:
        for res in tqdm(
            pool.imap_unordered(process_one, csvs, chunksize=CHUNK_SIZE),
            total=len(csvs),
            desc="生成信号样本"
        ):
            if res is not None and not res.empty:
                buffer.append(res)

            if len(buffer) >= FLUSH_EVERY:
                pd.concat(buffer).to_csv(
                    OUTPUT_FILE, mode="a", header=header, index=False, encoding="utf-8-sig"
                )
                header = False
                buffer.clear()

    if buffer:
        pd.concat(buffer).to_csv(
            OUTPUT_FILE, mode="a", header=header, index=False, encoding="utf-8-sig"
        )

    print(f"[OK] 输出完成：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
