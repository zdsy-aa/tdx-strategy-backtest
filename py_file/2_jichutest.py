#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
脚本名：2_jichutest.py

【脚本功能】
    Walk-forward 买点确认脚本（并行版，带完整步骤日志）

【脚本职责】
    1. 读取 1_signal_success_analyzer.py 生成的样本 CSV
       （已包含：信号、标签、机器学习特征）
    2. 按“年份”做 Walk-forward：
       - 用历史年份训练 Logistic Regression
       - 在当年做验证
    3. 输出：
       - 每一条样本的预测概率 prob
       - 是否通过确认 buy_decision
    4. 全过程并行执行，且每一步都有日志

【设计原则】
    - 不再回读原始行情
    - 不再计算指标
    - 只做模型与验证（职责单一、稳定）
===============================================================================
"""

import os
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =============================================================================
# 0) 日志工具（强制 flush，保证 nohup / 重定向下实时可见）
# =============================================================================

def log(msg: str):
    """
    统一日志输出：
    - 带时间戳
    - 强制 flush
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# =============================================================================
# 1) 路径配置
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 输入：脚本 1 的唯一输出
DATA_PATH = os.path.join(PROJECT_ROOT, "output", "signal_samples_same_day.csv")

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "walk_forward")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 2) 可调参数（含作用说明）
# =============================================================================

LABEL_COL = "success_20d"
"""
标签列：
- 1 = 未来 N 日达到阈值
- 0 = 未达到
"""

TRAIN_WINDOW_YEARS = 5
"""
训练窗口长度（年）：
- 每个验证年份，使用之前 TRAIN_WINDOW_YEARS 年作为训练集
"""

MIN_TRAIN_SAMPLES = 300
"""
最小训练样本数：
- 少于该数量，跳过该年份（防止模型不稳定）
"""

MIN_VALID_SAMPLES = 100
"""
最小验证样本数：
- 少于该数量，跳过该年份
"""

PROB_THRESHOLD = 0.65
"""
买入确认阈值：
- prob >= PROB_THRESHOLD → buy_decision = 1
"""

# 并行进程数（不建议超过 CPU 核心数）
NUM_WORKERS = min(cpu_count(), 8)

# =============================================================================
# 3) 机器学习特征列（必须来自脚本 1 的输出）
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
# 4) 单年份 Walk-forward 任务（并行 worker）
# =============================================================================

def process_one_year(args):
    """
    单个验证年份的完整处理流程（并行执行）

    参数：
        args = (valid_year, df)

    返回：
        DataFrame（该年份验证集的预测结果）
        或 None（样本不足）
    """
    valid_year, df = args
    t0 = time.time()

    log(f"[YEAR {valid_year}] 开始处理")

    # ================= 切分训练 / 验证 =================
    train_start = valid_year - TRAIN_WINDOW_YEARS
    train_end = valid_year

    train_df = df[(df["year"] >= train_start) & (df["year"] < train_end)]
    valid_df = df[df["year"] == valid_year]

    log(
        f"[YEAR {valid_year}] "
        f"训练样本={len(train_df)}, 验证样本={len(valid_df)}"
    )

    # ================= 样本量检查 =================
    if len(train_df) < MIN_TRAIN_SAMPLES or len(valid_df) < MIN_VALID_SAMPLES:
        log(f"[YEAR {valid_year}] 样本不足，跳过")
        return None

    # ================= 构建模型 =================
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            n_jobs=1   # ⚠️ 并行在年份层面，这里必须是 1
        ))
    ])

    # ================= 训练 =================
    log(f"[YEAR {valid_year}] 开始训练模型")
    model.fit(train_df[FEATURE_COLS], train_df[LABEL_COL])

    # ================= 预测 =================
    log(f"[YEAR {valid_year}] 开始预测验证集")
    valid_df = valid_df.copy()
    valid_df["prob"] = model.predict_proba(valid_df[FEATURE_COLS])[:, 1]
    valid_df["buy_decision"] = (valid_df["prob"] >= PROB_THRESHOLD).astype(int)

    cost = time.time() - t0
    log(f"[YEAR {valid_year}] 完成，用时 {cost:.1f}s")

    return valid_df

# =============================================================================
# 5) 主程序
# =============================================================================

def main():
    log("Walk-forward 脚本启动")

    # ================= 输入检查 =================
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"未找到输入文件: {DATA_PATH}")

    log("读取 signal_samples_same_day.csv")
    df = pd.read_csv(DATA_PATH)

    # ================= 基础清洗 =================
    log("基础字段处理")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # 清理 NaN / inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])

    years = sorted(df["year"].unique().tolist())

    log(f"样本总数: {len(df)}")
    log(f"年份范围: {years[0]} ~ {years[-1]}")
    log(f"Walk-forward 年份数: {len(years)}")
    log(f"并行进程数: {NUM_WORKERS}")

    # ================= 构造并行任务 =================
    tasks = [(y, df) for y in years]

    results = []
    done = 0

    log("开始并行 Walk-forward")

    with Pool(NUM_WORKERS) as pool:
        for res in pool.imap_unordered(process_one_year, tasks):
            done += 1
            log(f"[PROGRESS] 完成 {done}/{len(tasks)} 个年份")

            if res is not None:
                results.append(res)

    if not results:
        raise RuntimeError("所有年份均未产生有效结果，请检查参数")

    # ================= 汇总输出 =================
    final_df = pd.concat(results, ignore_index=True)

    out_file = os.path.join(
        OUTPUT_DIR,
        f"walk_forward_predictions_train_{TRAIN_WINDOW_YEARS}y.csv"
    )

    final_df.to_csv(out_file, index=False, encoding="utf-8-sig")

    log(f"输出完成: {out_file}")
    log("Walk-forward 脚本结束")

# =============================================================================
# 6) 入口
# =============================================================================

if __name__ == "__main__":
    main()
