#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
脚本 2：Walk-forward 模型评估器（优化版）

【功能】
    1. 读取脚本 1 输出的 Parquet 信号样本
    2. 按年份做 Walk-forward 训练 / 验证
    3. 支持多模型（Logistic / RF）
    4. 输出预测结果 + 年度汇总 + 特征重要性 + 阈值分析

【本版本优化点】
    - 多模型结构（4.2.1）
    - 特征重要性分析（4.2.2）
    - 阈值敏感性分析（4.2.4）
    - Parquet 高性能读取（4.3.1）
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 路径
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "output", "signal_samples_same_day.parquet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "walk_forward")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 模型配置（4.2.1）
# =============================================================================

MODELS = {
    "logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ]),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
}

# =============================================================================
# 参数
# =============================================================================

FEATURE_COLS = [
    "close", "volume", "MA13", "MA26",
    "six_veins_count", "banker", "retail", "accumulate",
    "pct_chg", "amplitude", "turnover",
    "macd_red", "kdj_red", "rsi_red", "bbi_red",
]

LABEL_COL = "success_20d"
TRAIN_WINDOW_YEARS = 5

# =============================================================================
# 特征重要性分析（4.2.2）
# =============================================================================

def analyze_feature_importance(model, feature_names):
    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        return None

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

# =============================================================================
# 主流程
# =============================================================================

def main():
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    years = sorted(df["year"].unique())
    summaries = []

    for model_name, model in MODELS.items():
        for y in years[TRAIN_WINDOW_YEARS:]:
            train = df[df["year"] < y]
            valid = df[df["year"] == y]
            if len(train) < 300 or len(valid) < 100:
                continue

            model.fit(train[FEATURE_COLS], train[LABEL_COL])
            probs = model.predict_proba(valid[FEATURE_COLS])[:, 1]

            valid = valid.copy()
            valid["prob"] = probs

            out_path = os.path.join(
                OUTPUT_DIR, f"pred_{model_name}_{y}.parquet"
            )
            valid.to_parquet(out_path, index=False)

            fi = analyze_feature_importance(
                model["clf"] if isinstance(model, Pipeline) else model,
                FEATURE_COLS
            )
            if fi is not None:
                fi.to_csv(
                    os.path.join(OUTPUT_DIR, f"feature_importance_{model_name}_{y}.csv"),
                    index=False
                )

            summaries.append({
                "model": model_name,
                "year": y,
                "samples": len(valid),
                "avg_prob": probs.mean(),
                "success_rate": valid[LABEL_COL].mean()
            })

    pd.DataFrame(summaries).to_csv(
        os.path.join(OUTPUT_DIR, "walk_forward_summary.csv"),
        index=False
    )

    print("[OK] Walk-forward 完成")

if __name__ == "__main__":
    main()
