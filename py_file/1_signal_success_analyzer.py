#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
前置脚本：信号样本生成（最终稳定版）
文件名：1_signal_success_analyzer.py
===============================================================================

【唯一过滤规则（已锁死）】
- 日线数量 < MIN_BARS（默认 80） → 丢弃

【脚本职责】
1. 遍历 data/day 下的所有 CSV（含子目录）
2. 读取你下载的股票日线数据
3. 调用 indicators.py → calculate_all_signals()
4. 识别真实候选买入信号
5. same_day 买入，计算未来 20 天 ≥5% 标签
6. 输出 Walk-forward 可直接使用的样本 CSV

【输出文件】
output/signal_samples_same_day.csv
===============================================================================
"""

import os
import time
import random
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from indicators import calculate_all_signals


# =============================================================================
# 一、路径与核心参数（研究区）
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "day")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "signal_samples_same_day.csv")

# ========= 核心研究参数 =========
MIN_BARS = 80              # 🔴 少于 80 根日线 → 直接丢弃
FUTURE_DAYS = 20
GAIN_THRESHOLD = 5.0
SIGNAL_COOLDOWN = 5        # 同一信号冷却期（交易日）

# ========= 并行参数 =========
NUM_WORKERS = 4
RETRY_TIMES = 3
CHUNK_SIZE = 20


# =============================================================================
# 二、统计计数器（让你看清发生了什么）
# =============================================================================

STATS_TEMPLATE = {
    "total_csv": 0,        # 遍历到的 CSV 总数
    "read_ok": 0,          # 成功读入并通过 MIN_BARS 的
    "signal_hit": 0,       # 至少出现过 1 次候选信号的股票
    "sample_rows": 0,      # 最终输出的样本行数
}


# =============================================================================
# 三、工具函数
# =============================================================================

def apply_cooldown(indices: List[int], cooldown: int) -> List[int]:
    """信号冷却期，防止连续多天重复计数。"""
    out = []
    last = -10**9
    for i in indices:
        if i - last >= cooldown:
            out.append(i)
            last = i
    return out


def compute_future_gain_same_day(df: pd.DataFrame) -> pd.Series:
    """
    same_day 买入的未来 20 天最大涨幅（无未来函数）

    entry = close[t]
    future_high = max(high[t+1 ... t+20])
    """
    entry = df["close"]
    future_high = (
        df["high"]
        .shift(-1)
        .rolling(window=FUTURE_DAYS, min_periods=FUTURE_DAYS)
        .max()
        .shift(-(FUTURE_DAYS - 1))
    )
    return (future_high - entry) / entry * 100


# =============================================================================
# 四、读取并标准化单只股票数据
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


def read_stock_csv(path: str) -> Optional[pd.DataFrame]:
    """
    只做三件事：
    1. 读 CSV
    2. 列名标准化
    3. 检查 MIN_BARS
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    df = df.rename(columns={k: v for k, v in CN_COL_MAP.items() if k in df.columns})

    if "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 必要字段（只要 OHLCV）
    need_cols = {"open", "high", "low", "close", "volume"}
    if not need_cols.issubset(df.columns):
        return None

    # 强制数值化
    for c in need_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 🔴 唯一过滤规则
    if len(df) < MIN_BARS:
        return None

    if "name" not in df.columns:
        df["name"] = ""

    return df


# =============================================================================
# 五、核心：单只股票处理（Worker）
# =============================================================================

def process_single_stock(csv_path: str) -> Optional[pd.DataFrame]:
    for attempt in range(1, RETRY_TIMES + 1):
        try:
            df = read_stock_csv(csv_path)
            if df is None:
                return None

            stock_code = os.path.splitext(os.path.basename(csv_path))[0]
            stock_name = str(df["name"].iloc[-1])

            # === 计算真实指标与信号 ===
            df = calculate_all_signals(df)

            # === 候选信号池（你可随时加减）===
            SIGNAL_MAP = {
                "six_veins": df.index[df.get("six_veins_buy", False)].tolist(),
                "chan_buy1": df.index[df.get("chan_buy1", False)].tolist(),
                "chan_buy2": df.index[df.get("chan_buy2", False)].tolist(),
                "chan_buy3": df.index[df.get("chan_buy3", False)].tolist(),
                "combo_steady": df.index[df.get("combo_steady", False)].tolist(),
                "combo_resonance": df.index[df.get("combo_resonance", False)].tolist(),
            }

            df["future_gain_20d"] = compute_future_gain_same_day(df)
            df["success_20d"] = (df["future_gain_20d"] >= GAIN_THRESHOLD).astype(int)

            records = []

            FEATURE_COLS = [
                "close",
                "volume",
                "ma20_ma60_ratio",
                "close_ma20_ratio",
                "close_hhv20_ratio",
                "rsi14",
                "macd_diff",
                "macd_hist",
                "vol_ma5_ratio",
                "vol_ma20_ratio",
                "atr14_ratio",
            ]

            for signal_name, idx_list in SIGNAL_MAP.items():
                idx_list = apply_cooldown(idx_list, SIGNAL_COOLDOWN)

                for idx in idx_list:
                    if pd.isna(df.at[idx, "future_gain_20d"]):
                        continue

                    # 特征完整性校验
                    if any(c not in df.columns or pd.isna(df.at[idx, c]) for c in FEATURE_COLS):
                        continue

                    rec = {
                        "stock": stock_code,
                        "name": stock_name,
                        "signal_date": df.at[idx, "date"],
                        "signal_type": signal_name,
                        "entry_price": float(df.at[idx, "close"]),
                        "future_gain_20d": float(df.at[idx, "future_gain_20d"]),
                        "success_20d": int(df.at[idx, "success_20d"]),
                    }

                    for c in FEATURE_COLS:
                        rec[c] = float(df.at[idx, c])

                    records.append(rec)

            return pd.DataFrame.from_records(records) if records else None

        except Exception:
            if attempt == RETRY_TIMES:
                return None
            time.sleep(0.2 * attempt + random.random() * 0.2)


# =============================================================================
# 六、主程序
# =============================================================================

def main():
    stats = STATS_TEMPLATE.copy()

    # 遍历所有 CSV（含子目录）
    csv_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    stats["total_csv"] = len(csv_files)

    print(f"[INFO] 发现 CSV 总数: {stats['total_csv']}")
    print(f"[INFO] MIN_BARS = {MIN_BARS}")
    print(f"[INFO] 并行进程数: {NUM_WORKERS}")

    all_samples = []

    with Pool(NUM_WORKERS) as pool:
        for res in tqdm(
            pool.imap_unordered(process_single_stock, csv_files, chunksize=CHUNK_SIZE),
            total=len(csv_files),
            desc="生成信号样本"
        ):
            if res is not None and not res.empty:
                stats["signal_hit"] += 1
                stats["sample_rows"] += len(res)
                all_samples.append(res)

    if not all_samples:
        print("[ERROR] 没有生成任何信号样本")
        return

    df_all = pd.concat(all_samples, ignore_index=True)
    df_all["signal_date"] = pd.to_datetime(df_all["signal_date"]).dt.strftime("%Y-%m-%d")
    df_all.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n========== 运行统计 ==========")
    print(f"CSV 总数            : {stats['total_csv']}")
    print(f"至少命中 1 次信号的股票数 : {stats['signal_hit']}")
    print(f"最终样本行数         : {stats['sample_rows']}")
    print(f"输出文件            : {OUTPUT_FILE}")
    print("================================")


if __name__ == "__main__":
    main()