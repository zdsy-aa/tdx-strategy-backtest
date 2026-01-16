#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
脚本 2（内存安全版）：
    2_jichutest_walkforward_partitioned.py

【功能】
    - Walk-forward 年度训练 / 验证
    - 每次只加载：N 年 train + 1 年 valid
    - 输出：每年预测结果 + 汇总

【内存策略】
    ✔ Parquet year 分区
    ✔ 只读必要年份
    ✔ float32 特征
===============================================================================
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 日志
# =============================================================================

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# =============================================================================
# 路径
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "output", "signal_samples_parquet")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "walk_forward")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 参数
# =============================================================================

TRAIN_WINDOW_YEARS = 5
LABEL_COL = "success_20d"

FEATURE_COLS = [
    "close", "volume", "MA13", "MA26",
    "six_veins_count", "banker", "retail", "accumulate",
    "pct_chg", "amplitude", "turnover",
    "macd_red", "kdj_red", "rsi_red", "bbi_red",
]

# =============================================================================
# 工具函数
# =============================================================================

def read_years(years):
    """只读取指定年份的数据"""
    df = pd.read_parquet(
        DATA_DIR,
        filters=[("year", "in", years)]
    )

    # 保证特征完整
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])

    df[FEATURE_COLS] = df[FEATURE_COLS].astype("float32")
    df[LABEL_COL] = df[LABEL_COL].astype("int8")

    return df

# =============================================================================
# 主流程
# =============================================================================

def main():
    log("脚本2启动：Walk-forward（year 分区）")

    # 读取所有可用年份（轻量）
    years = sorted([
        int(p.split("=")[1])
        for p in os.listdir(DATA_DIR)
        if p.startswith("year=")
    ])

    log(f"可用年份: {years}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000))
    ])

    summaries = []

    for y in years[TRAIN_WINDOW_YEARS:]:
        train_years = list(range(y - TRAIN_WINDOW_YEARS, y))

        log(f"YEAR={y} | train={train_years} valid={[y]}")

        train = read_years(train_years)
        valid = read_years([y])

        if len(train) < 300 or len(valid) < 100:
            log("样本不足，跳过")
            continue

        model.fit(train[FEATURE_COLS], train[LABEL_COL])

        probs = model.predict_proba(valid[FEATURE_COLS])[:, 1]
        valid = valid.copy()
        valid["prob"] = probs

        out_path = os.path.join(OUTPUT_DIR, f"pred_{y}.csv")
        valid.to_csv(out_path, index=False, encoding="utf-8-sig")

        summaries.append({
            "year": y,
            "train_samples": len(train),
            "valid_samples": len(valid),
            "avg_prob": float(np.mean(probs)),
            "success_rate": float(valid[LABEL_COL].mean())
        })

    summary_path = os.path.join(OUTPUT_DIR, "walk_forward_summary.csv")
    pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")

    log(f"汇总完成: {summary_path}")
    log("脚本2结束")

if __name__ == "__main__":
    main()
