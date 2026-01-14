#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
脚本名：2_jichutest.py

【脚本功能】
    Walk-forward 买点确认模型（Logistic Regression）

【输入】
    output/signal_samples_same_day.csv
    （由 1_signal_success_analyzer.py 生成，已包含特征）

【输出】
    1) walk_forward_summary_train_5y.csv
    2) walk_forward_predictions_train_5y.csv
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =============================================================================
# 日志工具
# =============================================================================

def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

# =============================================================================
# 参数区（作用说明）
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "output", "signal_samples_same_day.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "walk_forward")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_COL = "success_20d"

TRAIN_WINDOW_YEARS = 5
"""
训练窗口长度（年）：
- 使用过去 N 年训练
"""

PROB_THRESHOLD = 0.65
"""
最终买入确认阈值：
- prob >= 该值 → buy_decision = 1
"""

NUM_WORKERS = 12

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
# 单年份任务
# =============================================================================

def process_year(args):
    year, df = args
    train = df[df["year"] < year].tail(50000)
    valid = df[df["year"] == year]

    if len(train) < 300 or len(valid) < 100:
        return None

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000))
    ])

    model.fit(train[FEATURE_COLS], train[LABEL_COL])

    valid = valid.copy()
    valid["prob"] = model.predict_proba(valid[FEATURE_COLS])[:, 1]
    valid["buy_decision"] = (valid["prob"] >= PROB_THRESHOLD).astype(int)

    return valid


# =============================================================================
# 主程序
# =============================================================================

def main():
    log("读取样本数据")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    years = sorted(df["year"].unique())
    log(f"年份范围：{years[0]} ~ {years[-1]}")

    tasks = [(y, df) for y in years[1:]]

    results = []
    with Pool(NUM_WORKERS) as pool:
        for res in pool.imap_unordered(process_year, tasks):
            if res is not None:
                results.append(res)

    final_df = pd.concat(results)
    out_file = os.path.join(
        OUTPUT_DIR, f"walk_forward_predictions_train_{TRAIN_WINDOW_YEARS}y.csv"
    )
    final_df.to_csv(out_file, index=False, encoding="utf-8-sig")

    log(f"输出完成：{out_file}")


if __name__ == "__main__":
    main()
