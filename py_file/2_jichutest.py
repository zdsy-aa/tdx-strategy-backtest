# -*- coding: utf-8 -*-

"""
================================================================================
Walk-forward 买点确认系统（same_day 后置脚本）: jichutest.py
================================================================================

【脚本定位】
本脚本读取前置脚本输出的信号样本文件：
    output/signal_samples_same_day.csv

对每条信号样本（信号当天）做：
- Walk-forward（滚动前推）训练/验证
- Logistic Regression 输出概率：P(未来20天内 ≥5% 上冲)
- 将概率分箱统计每年的成功率（稳定性检验）
- 输出逐样本预测概率，以及是否满足“买入阈值”的决策结果

【注意】
- 本脚本不再读取 data/day（原始下载日线目录）
- 本脚本只使用“信号样本 CSV”作为输入
- 若你看到 PermissionError 或 read_csv 目录错误，说明 DATA_PATH 指向了目录

【输出文件】
1) output/walk_forward/walk_forward_summary_train_{TRAIN_WINDOW_YEARS}y.csv
   - 每个验证年份、每个概率分箱的样本数与成功率

2) output/walk_forward/walk_forward_predictions_train_{TRAIN_WINDOW_YEARS}y.csv
   - 验证集每个样本的预测概率 prob
   - 以及 buy_decision（prob >= PROB_THRESHOLD）

================================================================================
"""

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =============================================================================
# 1) 路径与参数（研究配置区）
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 输入：前置脚本输出（注意：必须是 CSV 文件路径）
DATA_PATH = os.path.join(PROJECT_ROOT, "output", "signal_samples_same_day.csv")

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "walk_forward")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 标签定义（与前置脚本一致）
FUTURE_DAYS = 20
LABEL_COL = "success_20d"   # 前置脚本已经写死为 success_20d

# Walk-forward 参数（训练窗口可调）
TRAIN_WINDOW_YEARS = 5      # 你要求可调：3/5/7...
VALID_WINDOW_YEARS = 1      # 通常固定 1 年验证窗口

# 概率分箱：用于观察“高概率段是否稳定更好”
PROB_BINS = [0.0, 0.4, 0.6, 0.65, 1.0]

# 买入阈值（最终决策）：prob >= 阈值 才认为“确认通过，允许买入”
PROB_THRESHOLD = 0.65

# 模型参数（Logistic 回归）
LOGIT_MAX_ITER = 2000


# =============================================================================
# 2) 特征列（必须与前置脚本输出一致）
# =============================================================================

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


# =============================================================================
# 3) 读取数据与防呆校验
# =============================================================================

def require_columns(df: pd.DataFrame, cols: list, where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{where}] 缺少必要列: {missing}")


def main():
    # 防呆：必须是文件而不是目录
    if os.path.isdir(DATA_PATH):
        raise ValueError(f"DATA_PATH 指向的是目录而不是文件，请改为 CSV 文件路径: {DATA_PATH}")

    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"未找到输入文件: {DATA_PATH}。请先运行前置脚本生成该文件。")

    df = pd.read_csv(DATA_PATH)

    # 必要列校验
    require_columns(df, ["signal_date", LABEL_COL], "输入样本")
    require_columns(df, FEATURE_COLS, "输入样本")

    # 日期处理：用于按年份 Walk-forward
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    df = df.dropna(subset=["signal_date"]).copy()
    df["year"] = df["signal_date"].dt.year

    # 标签强制 int
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # 去掉特征中的 NaN/inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + [LABEL_COL])

    years = sorted(df["year"].unique().tolist())
    if len(years) < 2:
        raise ValueError("年份跨度过小，无法做 Walk-forward。请确认样本覆盖多个年份。")

    print(f"[INFO] 输入样本: {DATA_PATH}")
    print(f"[INFO] 样本数: {len(df)}")
    print(f"[INFO] 年份范围: {years[0]} ~ {years[-1]}")
    print(f"[INFO] TRAIN_WINDOW_YEARS={TRAIN_WINDOW_YEARS}, VALID_WINDOW_YEARS={VALID_WINDOW_YEARS}")
    print(f"[INFO] PROB_THRESHOLD={PROB_THRESHOLD}")

    # =============================================================================
    # 4) Walk-forward：逐年滚动训练/验证
    # =============================================================================

    summary_rows = []
    pred_rows = []  # 保存验证期逐样本预测

    for valid_year in years:
        train_start = valid_year - TRAIN_WINDOW_YEARS
        train_end = valid_year
        valid_end = valid_year + VALID_WINDOW_YEARS

        train_df = df[(df["year"] >= train_start) & (df["year"] < train_end)].copy()
        valid_df = df[(df["year"] >= valid_year) & (df["year"] < valid_end)].copy()

        # 样本不足直接跳过（你可以按需调整阈值）
        if len(train_df) < 300 or len(valid_df) < 100:
            continue

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[LABEL_COL]

        X_valid = valid_df[FEATURE_COLS]
        y_valid = valid_df[LABEL_COL]

        # 模型：标准化 + Logistic
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=LOGIT_MAX_ITER
            ))
        ])

        model.fit(X_train, y_train)

        # 验证期预测概率
        valid_df["prob"] = model.predict_proba(X_valid)[:, 1]
        valid_df["buy_decision"] = (valid_df["prob"] >= PROB_THRESHOLD).astype(int)

        # 保存逐样本预测（用于你后续做回测/分析/筛选）
        keep_cols = ["stock", "name", "signal_date", "signal_type", "entry_price", "future_gain_20d", LABEL_COL]
        keep_cols = [c for c in keep_cols if c in valid_df.columns]  # 兼容缺失列
        keep_cols += FEATURE_COLS + ["prob", "buy_decision", "year"]
        pred_rows.append(valid_df[keep_cols])

        # 概率分箱统计（稳定性验证）
        for i in range(len(PROB_BINS) - 1):
            low, high = PROB_BINS[i], PROB_BINS[i + 1]
            bin_df = valid_df[(valid_df["prob"] >= low) & (valid_df["prob"] < high)]
            if len(bin_df) == 0:
                continue

            summary_rows.append({
                "valid_year": valid_year,
                "train_window_years": TRAIN_WINDOW_YEARS,
                "prob_bin": f"{low:.2f}-{high:.2f}",
                "sample_count": int(len(bin_df)),
                "success_rate": float(bin_df[LABEL_COL].mean()),
                "avg_prob": float(bin_df["prob"].mean()),
            })

    if not summary_rows:
        raise RuntimeError("Walk-forward 没有产生任何有效年度窗口（可能样本不足或年份跨度不足）。")

    # =============================================================================
    # 5) 输出结果文件
    # =============================================================================

    summary_df = pd.DataFrame(summary_rows)
    summary_out = os.path.join(OUTPUT_DIR, f"walk_forward_summary_train_{TRAIN_WINDOW_YEARS}y.csv")
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

    pred_df = pd.concat(pred_rows, ignore_index=True)
    pred_out = os.path.join(OUTPUT_DIR, f"walk_forward_predictions_train_{TRAIN_WINDOW_YEARS}y.csv")
    pred_df.to_csv(pred_out, index=False, encoding="utf-8-sig")

    # 控制台输出一个“全局分箱平均成功率”，便于你快速判断模型是否有效
    print(f"[OK] 输出汇总: {summary_out}")
    print(f"[OK] 输出预测: {pred_out}")
    print("[OK] 各概率分箱平均成功率（跨年度平均）:")
    print(summary_df.groupby("prob_bin")["success_rate"].mean().sort_index())


if __name__ == "__main__":
    main()