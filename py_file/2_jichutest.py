#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
脚本名：2_jichutest.py  （长期稳定版：并行 + 详细日志 + 抗 BrokenPipe）

【脚本功能】
    Walk-forward 买点确认模型（Logistic Regression）
    - 输入：1_signal_success_analyzer.py 生成的 output/signal_samples_same_day.csv
    - 输出：
        1) output/walk_forward/walk_forward_predictions_train_{N}y.csv
           （逐样本：prob + buy_decision）
        2) output/walk_forward/walk_forward_year_summary_train_{N}y.csv
           （按年份汇总：样本数、通过数、通过率、通过样本胜率）

【为什么这是“长期稳定版”】
    - worker 不再通过 multiprocessing pipe 回传大 DataFrame（彻底解决 BrokenPipe）
    - worker 只写临时 CSV + 返回小结果（year / rows / tmp_path）
    - 主进程以“文件拼接”方式生成最终输出，不把结果读回内存

【并行策略】
    - 并行粒度：valid_year（最合理/最安全）
    - 共享只读数据：Pool(initializer=...) + Linux fork copy-on-write
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
# 0) 日志工具（强制 flush，保证 nohup/重定向实时可见）
# =============================================================================

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# =============================================================================
# 1) 路径配置
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 输入：脚本1的输出
DATA_PATH = os.path.join(PROJECT_ROOT, "output", "signal_samples_same_day.csv")

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "walk_forward")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 临时目录：按年份产出 tmp 文件（主进程会拼接后删除）
TMP_DIR = os.path.join(OUTPUT_DIR, "_tmp_year_parts")
os.makedirs(TMP_DIR, exist_ok=True)


# =============================================================================
# 2) 可调参数（含作用说明）
# =============================================================================

LABEL_COL = "success_20d"
"""
标签列：
- 1 = 未来 N 日达到收益阈值
- 0 = 未达到
"""

TRAIN_WINDOW_YEARS = 5
"""
训练窗口长度（年）：
- 对每个验证年份 y：
  训练集使用 [y-TRAIN_WINDOW_YEARS, y) 的数据
"""

MIN_TRAIN_SAMPLES = 300
"""
最小训练样本数：
- 训练样本过少时模型不稳定，直接跳过该年份
"""

MIN_VALID_SAMPLES = 100
"""
最小验证样本数：
- 验证样本过少时统计意义弱，直接跳过该年份
"""

PROB_THRESHOLD = 0.65
"""
买入确认阈值：
- prob >= PROB_THRESHOLD → buy_decision = 1
"""

LOGIT_MAX_ITER = 2000
"""
LogisticRegression 最大迭代次数：
- 样本量大时建议提高，避免不收敛
"""

NUM_WORKERS = 2
"""
并行进程数：
- 建议 <= CPU 核心数
- 8 通常是一个稳健上限（避免内存/调度压力）
"""


# =============================================================================
# 3) 特征列（必须与脚本1输出一致）
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

# 预测输出的基础列（脚本1应提供）
BASE_OUT_COLS = [
    "stock",
    "name",
    "signal",
    "date",
    "entry_price",
    "future_gain_20d",
    LABEL_COL,
]


# =============================================================================
# 4) multiprocessing 共享只读全局（关键：不传大DF给worker）
# =============================================================================

GLOBAL_DF = None
GLOBAL_YEAR_IDX = None

def init_worker(df: pd.DataFrame, year_idx: dict):
    """
    Pool initializer：
    - 每个 worker 启动时执行一次
    - Linux 默认 fork：父进程内存 copy-on-write 共享（只读安全）
    """
    global GLOBAL_DF, GLOBAL_YEAR_IDX
    GLOBAL_DF = df
    GLOBAL_YEAR_IDX = year_idx


# =============================================================================
# 5) 文件拼接工具：把 tmp CSV 追加到最终 CSV（不读入内存）
# =============================================================================

def append_csv_noheader(src_path: str, dst_path: str, dst_has_header: bool) -> bool:
    """
    把 src_path 追加写入 dst_path。
    - 如果 dst_has_header=False：复制整个文件（包含 header）
    - 如果 dst_has_header=True ：跳过 src 第一行 header 再追加
    返回：更新后的 dst_has_header=True
    """
    # 二进制复制，避免逐行 Python 解释开销
    with open(src_path, "rb") as fsrc:
        if dst_has_header:
            # 跳过第一行（header）
            fsrc.readline()

        with open(dst_path, "ab") as fdst:
            while True:
                chunk = fsrc.read(1024 * 1024 * 8)  # 8MB
                if not chunk:
                    break
                fdst.write(chunk)

    return True


# =============================================================================
# 6) 单年份任务（worker）：训练->预测->写临时文件->返回小结果
# =============================================================================

def process_one_year(valid_year: int):
    """
    单年份 Walk-forward：
    - 切分训练/验证
    - 训练 Logistic
    - 预测 prob + buy_decision
    - 写 tmp CSV（该年份完整预测结果）
    - 返回：year / rows / tmp_path / 统计信息（小对象）
    """
    global GLOBAL_DF, GLOBAL_YEAR_IDX
    df = GLOBAL_DF
    year_idx = GLOBAL_YEAR_IDX

    t0 = time.time()

    # ====== 构造训练索引 ======
    train_years = list(range(valid_year - TRAIN_WINDOW_YEARS, valid_year))
    train_parts = []
    for y in train_years:
        idx = year_idx.get(y)
        if idx is not None and len(idx) > 0:
            train_parts.append(idx)

    valid_idx = year_idx.get(valid_year)

    if not train_parts or valid_idx is None or len(valid_idx) == 0:
        # 训练或验证缺失
        return {
            "year": valid_year,
            "skipped": True,
            "reason": "no_train_or_valid",
        }

    train_idx = np.concatenate(train_parts)
    # 选择需要的列，减少复制成本
    need_cols = [c for c in (BASE_OUT_COLS + FEATURE_COLS + ["year"]) if c in df.columns]

    train_df = df.iloc[train_idx][need_cols]
    valid_df = df.iloc[valid_idx][need_cols]

    n_train, n_valid = len(train_df), len(valid_df)

    if n_train < MIN_TRAIN_SAMPLES or n_valid < MIN_VALID_SAMPLES:
        return {
            "year": valid_year,
            "skipped": True,
            "reason": f"small_samples train={n_train} valid={n_valid}",
        }

    # ====== 训练模型 ======
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=LOGIT_MAX_ITER,
            solver="lbfgs",
            n_jobs=1  # 并行在年份层面，这里必须=1
        ))
    ])

    # 注意：训练/预测只使用特征列
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[LABEL_COL].astype(int)
    X_valid = valid_df[FEATURE_COLS]

    model.fit(X_train, y_train)

    # ====== 预测并生成决策 ======
    valid_df = valid_df.copy()
    valid_df["prob"] = model.predict_proba(X_valid)[:, 1]
    valid_df["buy_decision"] = (valid_df["prob"] >= PROB_THRESHOLD).astype(int)

    # ====== 写临时文件（该年份结果）=====
    tmp_path = os.path.join(TMP_DIR, f"pred_year_{valid_year}.csv")
    out_cols = [c for c in (BASE_OUT_COLS + FEATURE_COLS + ["prob", "buy_decision", "year"]) if c in valid_df.columns]
    valid_df[out_cols].to_csv(tmp_path, index=False, encoding="utf-8-sig")

    # ====== 年份级统计（小结果）=====
    passed = int(valid_df["buy_decision"].sum())
    pass_rate = float(passed / n_valid) if n_valid else 0.0
    pass_success_rate = float(valid_df.loc[valid_df["buy_decision"] == 1, LABEL_COL].mean()) if passed > 0 else 0.0

    cost = time.time() - t0

    return {
        "year": valid_year,
        "skipped": False,
        "tmp_path": tmp_path,
        "train_n": int(n_train),
        "valid_n": int(n_valid),
        "passed_n": int(passed),
        "pass_rate": pass_rate,
        "pass_success_rate": pass_success_rate,
        "seconds": float(cost),
    }


# =============================================================================
# 7) 主程序：读取->清洗->构建year索引->并行->拼接输出->汇总
# =============================================================================

def main():
    log("Walk-forward 脚本启动（稳定版）")

    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"未找到输入文件: {DATA_PATH}")

    # -----------------------------
    # 7.1 读取样本（可能较慢）
    # -----------------------------
    log("读取 signal_samples_same_day.csv（大文件，可能需要数分钟）")
    t_read = time.time()
    df = pd.read_csv(DATA_PATH)
    log(f"读取完成，用时 {time.time() - t_read:.1f}s")

    # -----------------------------
    # 7.2 基础字段处理与降内存
    # -----------------------------
    log("基础字段处理（date/year/NaN/inf 清洗）")
    t_clean = time.time()

    # date -> datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # year
    df["year"] = df["date"].dt.year

    # label
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # 过滤 inf/NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])

    # 只保留必要列，降低内存（非常重要）
    keep_cols = list(dict.fromkeys([c for c in (BASE_OUT_COLS + FEATURE_COLS + ["year"]) if c in df.columns]))
    df = df[keep_cols].copy()

    log(f"清洗完成，用时 {time.time() - t_clean:.1f}s")
    log(f"清洗后样本数: {len(df)}")

    # -----------------------------
    # 7.3 年份索引（一次构建，避免重复扫描）
    # -----------------------------
    log("构建 year->row_indices 索引（一次性）")
    t_idx = time.time()
    year_idx = df.groupby("year", sort=True).indices  # dict: year -> ndarray indices
    years = sorted(year_idx.keys())
    log(f"索引构建完成，用时 {time.time() - t_idx:.1f}s")
    log(f"年份范围: {years[0]} ~ {years[-1]}")
    log(f"年份数: {len(years)}")
    log(f"并行进程数: {NUM_WORKERS}")

    # -----------------------------
    # 7.4 输出文件准备
    # -----------------------------
    pred_out = os.path.join(OUTPUT_DIR, f"walk_forward_predictions_train_{TRAIN_WINDOW_YEARS}y.csv")
    summary_out = os.path.join(OUTPUT_DIR, f"walk_forward_year_summary_train_{TRAIN_WINDOW_YEARS}y.csv")

    # 覆盖旧结果
    if os.path.exists(pred_out):
        os.remove(pred_out)
    if os.path.exists(summary_out):
        os.remove(summary_out)

    dst_has_header = False
    summary_rows = []

    # -----------------------------
    # 7.5 并行执行（worker 写 tmp，主进程拼接）
    # -----------------------------
    log("开始并行 Walk-forward（worker 写 tmp，主进程拼接最终文件）")
    t_all = time.time()

    done = 0
    ok = 0
    skipped = 0
    failed = 0

    with Pool(
        processes=NUM_WORKERS,
        initializer=init_worker,
        initargs=(df, year_idx),
    ) as pool:
        for result in pool.imap_unordered(process_one_year, years):
            done += 1

            # 统一处理返回结果（小对象）
            if result.get("skipped", False):
                skipped += 1
                log(f"[PROGRESS] {done}/{len(years)} YEAR={result['year']} SKIPPED reason={result.get('reason')}")
                summary_rows.append({
                    "year": result["year"],
                    "skipped": 1,
                    "reason": result.get("reason", ""),
                    "train_n": 0,
                    "valid_n": 0,
                    "passed_n": 0,
                    "pass_rate": 0.0,
                    "pass_success_rate": 0.0,
                    "seconds": 0.0,
                })
                continue

            tmp_path = result.get("tmp_path")
            if not tmp_path or not os.path.exists(tmp_path):
                failed += 1
                log(f"[PROGRESS] {done}/{len(years)} YEAR={result.get('year')} FAILED (tmp missing)")
                continue

            # 追加拼接到最终 predictions（不读入内存）
            try:
                dst_has_header = append_csv_noheader(tmp_path, pred_out, dst_has_header)
                ok += 1

                # 写完就删临时文件，避免磁盘堆积
                os.remove(tmp_path)

                log(
                    f"[PROGRESS] {done}/{len(years)} "
                    f"YEAR={result['year']} OK "
                    f"train={result['train_n']} valid={result['valid_n']} "
                    f"passed={result['passed_n']} pass_rate={result['pass_rate']:.4f} "
                    f"pass_success={result['pass_success_rate']:.4f} "
                    f"time={result['seconds']:.1f}s"
                )

                summary_rows.append({
                    "year": result["year"],
                    "skipped": 0,
                    "reason": "",
                    "train_n": result["train_n"],
                    "valid_n": result["valid_n"],
                    "passed_n": result["passed_n"],
                    "pass_rate": result["pass_rate"],
                    "pass_success_rate": result["pass_success_rate"],
                    "seconds": result["seconds"],
                })

            except Exception as e:
                failed += 1
                log(f"[PROGRESS] {done}/{len(years)} YEAR={result.get('year')} FAILED append error={repr(e)}")

    # -----------------------------
    # 7.6 输出汇总
    # -----------------------------
    log("并行阶段结束，写出年份汇总 summary")
    summary_df = pd.DataFrame(summary_rows).sort_values("year")
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

    log(f"最终 predictions 输出: {pred_out}")
    log(f"最终 summary 输出: {summary_out}")
    log(f"统计：done={done} ok={ok} skipped={skipped} failed={failed}")
    log(f"总用时：{time.time() - t_all:.1f}s")
    log("Walk-forward 脚本结束（稳定版）")


if __name__ == "__main__":
    main()
