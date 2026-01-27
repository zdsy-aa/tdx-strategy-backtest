#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a6_models.py
功能描述: 模拟仪表盘数据生成模块
使用方法: python3 a6_models.py --dashboard-days 3
依赖库: pandas, numpy, a99_indicators
安装命令: pip install pandas numpy
================================================================================
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加项目路径，以便导入内部模块
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

# 导入统一的指标计算模块
from a99_indicators import (
    calculate_six_veins,
    calculate_buy_sell_points,
    calculate_money_tree,
    calculate_chan_theory
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("a6_models")

# =========================
# 1) 配置与路径处理
# =========================

CN_TO_EN = {
    "名称": "name", "日期": "date", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
    "振幅": "amplitude", "涨跌幅": "pct_change", "涨跌额": "change", "换手率": "turnover",
}

def normalize_path(p: str) -> Path:
    """标准化路径，支持 Windows 和 Linux。"""
    p_str = p.replace("\\", "/")
    if os.path.isabs(p_str):
        return Path(p_str).resolve()
    project_root = Path(__file__).resolve().parent.parent
    return (project_root / p_str).resolve()

def _to_jsonable(x: Any) -> Any:
    """将数据转换为 JSON 可序列化格式。"""
    if x is None: return None
    if isinstance(x, (str, int, float, bool)): return x
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.ndarray,)): return x.tolist()
    if isinstance(x, (pd.Timestamp, datetime)): return x.isoformat()
    if isinstance(x, dict): return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_to_jsonable(v) for v in x]
    return str(x)

# =========================
# 2) 数据处理逻辑
# =========================

def read_day_csv(csv_path: Path, code: str, market: str) -> Optional[pd.DataFrame]:
    """读取并标准化 CSV 数据。"""
    try:
        # 尝试多种编码
        df = None
        for enc in ['utf-8', 'gbk', 'utf-8-sig']:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                break
            except: continue
        
        if df is None: return None
        
        # 映射列名
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns={c: CN_TO_EN.get(c, c) for c in df.columns})
        
        # 解析日期
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        
        # 元信息
        df["stock_code"] = code
        df["market"] = market
        return df
    except Exception as e:
        logger.error(f"读取失败 {csv_path}: {e}")
        return None

def build_symbol_payload(df: pd.DataFrame, dashboard_days: int) -> Dict[str, Any]:
    """计算指标并生成单只股票的详细数据。"""
    # 1. 计算所有指标
    df = calculate_six_veins(df)
    df = calculate_buy_sell_points(df)
    df = calculate_money_tree(df)
    df = calculate_chan_theory(df)
    
    # 2. 提取最新数据
    latest = df.iloc[-1]
    
    # 3. 判断最近 N 天内是否有信号出现
    recent_df = df.tail(dashboard_days)
    
    # 策略 A: 六脉神剑 (6红)
    has_six_veins = recent_df['six_veins_signal'].any()
    # 策略 B: 庄家买点 (buy1 或 buy2)
    has_buy_sell = recent_df['buy1'].any() or recent_df['buy2'].any()
    # 策略 C: 摇钱树或缠论
    has_special = recent_df['money_tree_signal'].any() or recent_df['chan_buy1'].any()
    
    # 4. 构建返回结构
    return {
        "info": {
            "code": latest.get("stock_code", ""),
            "name": latest.get("name", ""),
            "market": latest.get("market", ""),
            "last_date": latest["date"].strftime("%Y-%m-%d"),
            "price": float(latest["close"]),
            "pct_change": float(latest.get("pct_change", 0)) if pd.notna(latest.get("pct_change", 0)) else None,
        },
        "signals": {
            "slot_a": bool(has_six_veins),
            "slot_b": bool(has_buy_sell),
            "slot_c": bool(has_special),
            "score": int(recent_df['six_veins_count'].max()),
            "strength": "High" if recent_df['six_veins_count'].max() >= 5 else "Normal"
        }
    }

def main():
    parser = argparse.ArgumentParser(description='模拟仪表盘数据生成')
    parser.add_argument('--dashboard-days', type=int, default=3, help='统计最近几天内的信号')
    parser.add_argument('--limit', type=int, default=0, help='限制处理数量')
    args = parser.parse_args()

    data_root = normalize_path("data/day")
    out_root = normalize_path("web/client/src/data")
    
    all_symbols = []
    count = 0
    
    for market in ["sh", "sz", "bj"]:
        m_dir = data_root / market
        if not m_dir.exists(): continue
        
        for csv_file in m_dir.glob("*.csv"):
            if args.limit > 0 and count >= args.limit: break
            
            df = read_day_csv(csv_file, csv_file.stem, market)
            if df is not None and len(df) >= 30:
                payload = build_symbol_payload(df, args.dashboard_days)
                all_symbols.append(payload)
                count += 1
                if count % 100 == 0: logger.info(f"已处理 {count} 只股票")
    
    # 保存汇总数据
    dashboard_data = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": len(all_symbols),
        "symbols": all_symbols
    }
    
    os.makedirs(out_root, exist_ok=True)
    with open(out_root / "dashboard.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(dashboard_data), f, ensure_ascii=False, indent=2)
        
    logger.info(f"生成完毕，共 {len(all_symbols)} 只股票数据。")

if __name__ == "__main__":
    main()
