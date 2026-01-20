#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
纯离线 CSV -> 信号/策略整合 -> 输出 JSON（给网页展示）
抛弃 data_fetcher / ml_models，不依赖任何项目内其他模块。

输入目录（默认）：
  \\data\\day\\bj
  \\data\\day\\sz
  \\data\\day\\sh

输出目录（默认）：
  \\web\\client\\src\\data
  - dashboard.json
  - errors.json
  - symbols/<code>.json

CSV 列（中文）：
  名称,日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率

你要整合的“两个脚本核心信号/策略”：
  填入下方 Strategy Slot A / B 的函数 strategy_core_A / strategy_core_B
  不要改它们的函数签名与返回结构，这样网页展示层稳定。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("offline_signal_json_pipeline")

# =========================
# 1) 基础配置与字段规范化
# =========================

CSV_COLUMNS_CN = [
    "名称", "日期", "开盘", "收盘", "最高", "最低",
    "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率",
]

CN_TO_EN = {
    "名称": "name",
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}

REQUIRED_EN = ["date", "open", "close", "high", "low", "volume"]

NUMERIC_EN = [
    "open", "close", "high", "low",
    "volume", "amount", "amplitude",
    "pct_change", "change", "turnover",
]


def normalize_path(p: str) -> Path:
    """
    标准化路径：
    1. 兼容 Windows 反斜杠。
    2. 如果是相对路径，则相对于项目根目录（py_file 的父目录）。
    """
    p_str = p.replace("\\", "/")
    path = Path(p_str)
    
    if not path.is_absolute():
        # 获取当前脚本所在目录的父目录作为项目根目录
        project_root = Path(__file__).resolve().parent.parent
        # 如果路径以 / 开头（在 Windows 下可能被误认为绝对路径），去掉它
        if p_str.startswith("/"):
            p_str = p_str.lstrip("/")
        path = project_root / p_str
        
    return path


def _to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return str(x)


def safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def read_day_csv(csv_path: Path, code: str, market: str) -> pd.DataFrame:
    """
    读取 CSV -> 标准化列名（中文->英文）-> 类型转换 -> 排序去重
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    # 映射到英文列
    df = df.rename(columns={c: CN_TO_EN.get(c, c) for c in df.columns})

    # 必须列检查
    missing = [c for c in REQUIRED_EN if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required cols={missing} file={csv_path}")

    # date 解析
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # 数值列
    for c in NUMERIC_EN:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 元信息
    df["stock_code"] = code
    df["market"] = market

    # 排序去重
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return df


def discover_csv_files(data_root: Path) -> List[Tuple[str, str, Path]]:
    """
    返回 (market, code, csv_path)
    """
    items: List[Tuple[str, str, Path]] = []
    for market in ("bj", "sz", "sh"):
        d = data_root / market
        if not d.exists():
            logger.warning("Market dir missing: %s", d)
            continue
        for p in sorted(d.glob("*.csv")):
            code = p.stem.strip()
            if code:
                items.append((market, code, p))
    return items


# =========================
# 2) 统一的指标工具（可选）
#    你两份策略如果需要常用指标，可直接复用这些小工具
# =========================

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(n, min_periods=n).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(n, min_periods=n).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ef = ema(close, fast)
    es = ema(close, slow)
    diff = ef - es
    dea = ema(diff, signal)
    hist = diff - dea
    return diff, dea, hist

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def volume_ratio(volume: pd.Series, n: int = 5) -> pd.Series:
    vma = volume.rolling(n, min_periods=n).mean()
    return volume / (vma + 1e-12)


# =========================
# 3) 信号输出的统一格式（网页稳定）
# =========================

def make_signal(
    date: pd.Timestamp,
    sig_type: str,
    strength: int,
    score: float,
    message: str,
    extras: Optional[Dict[str, Any]] = None,
    source: str = "A",
) -> Dict[str, Any]:
    """
    signal 标准结构：
      date: 时间
      type: 信号类型（字符串）
      strength: 1~5 强度（你自己定标，但保持一致）
      score: 0~100 用于排序/展示
      message: 简短说明
      source: A/B（来自哪个策略槽）
      extras: 任意扩展字段（网页可折叠展示）
    """
    obj = {
        "date": date,
        "type": sig_type,
        "strength": int(strength),
        "score": float(score),
        "message": str(message),
        "source": str(source),
    }
    if extras:
        obj["extras"] = extras
    return obj


# =========================
# 4) 两个“核心策略槽”：把你那两个脚本的核心逻辑搬到这里
# =========================

def strategy_core_A(df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Strategy Slot A：
      输入 df：已标准化、按日期升序、包含 open/high/low/close/volume 等列
      返回：
        - score_A: 0~100（该策略对该票的综合评分，用于 dashboard 排序）
        - signals_A: List[signal]（可以是多条信号）
    你需要把“脚本1”的核心信号/策略逻辑复制到这里，保持返回结构不变。
    """
    # ---- 示例（占位）：你应替换为脚本1核心逻辑 ----
    # 这里给一个非常保守的示例：MA5 上穿 MA20
    if len(df) < 60:
        return 0.0, []

    close = df["close"]
    ma5 = sma(close, 5)
    ma20 = sma(close, 20)

    signals: List[Dict[str, Any]] = []
    score = 0.0

    if pd.notna(ma5.iloc[-2]) and pd.notna(ma20.iloc[-2]) and pd.notna(ma5.iloc[-1]) and pd.notna(ma20.iloc[-1]):
        if ma5.iloc[-2] <= ma20.iloc[-2] and ma5.iloc[-1] > ma20.iloc[-1]:
            score = 60.0
            signals.append(make_signal(
                date=df["date"].iloc[-1],
                sig_type="A_MA_CROSS_UP",
                strength=3,
                score=60.0,
                message="A策略：MA5上穿MA20",
                extras={"ma5": float(ma5.iloc[-1]), "ma20": float(ma20.iloc[-1])},
                source="A",
            ))
    return score, signals


def strategy_core_B(df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Strategy Slot B：
      你需要把“脚本2”的核心信号/策略逻辑复制到这里。
    """
    # ---- 示例（占位）：你应替换为脚本2核心逻辑 ----
    # 示例：RSI 超卖 + 放量
    if len(df) < 30:
        return 0.0, []

    signals: List[Dict[str, Any]] = []
    score = 0.0

    r = rsi(df["close"], 14)
    vr = volume_ratio(df["volume"], 5)

    if pd.notna(r.iloc[-1]) and r.iloc[-1] < 30:
        score += 40
        signals.append(make_signal(
            date=df["date"].iloc[-1],
            sig_type="B_RSI_OVERSOLD",
            strength=2,
            score=40.0,
            message=f"B策略：RSI超卖({float(r.iloc[-1]):.2f})",
            extras={"rsi14": float(r.iloc[-1])},
            source="B",
        ))

    if pd.notna(vr.iloc[-1]) and vr.iloc[-1] >= 1.5:
        score += 30
        signals.append(make_signal(
            date=df["date"].iloc[-1],
            sig_type="B_VOLUME_SPIKE",
            strength=1,
            score=30.0,
            message=f"B策略：放量({float(vr.iloc[-1]):.2f})",
            extras={"volume_ratio_5": float(vr.iloc[-1])},
            source="B",
        ))

    score = float(max(0.0, min(100.0, score)))
    return score, signals


# =========================
# 5) 策略融合与 JSON 产出
# =========================

def build_symbol_payload(
    market: str,
    code: str,
    csv_path: Path,
    recent_bars: int,
    include_series: bool,
    series_bars: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    返回 (symbol_payload, error_payload)
    """
    try:
        df = read_day_csv(csv_path, code=code, market=market)

        if recent_bars > 0 and len(df) > recent_bars:
            df = df.iloc[-recent_bars:].copy()

        if len(df) < 20:
            # 静默跳过数据不足 20 天的股票，不抛出异常
            return None, None

        name = None
        if "name" in df.columns and df["name"].notna().any():
            name = df["name"].dropna().iloc[-1]

        # 两策略分别跑
        score_a, sig_a = strategy_core_A(df)
        score_b, sig_b = strategy_core_B(df)

        # 合并信号：按 date desc、score desc
        signals = (sig_a or []) + (sig_b or [])
        signals.sort(key=lambda x: (str(x.get("date", "")), float(x.get("score", 0.0))), reverse=True)

        # 汇总分：给 dashboard 用（你可改成更符合你的原逻辑的融合方式）
        # 默认：max(score_a, score_b) + 0.2*min(...)（上限100）
        s_hi = max(score_a, score_b)
        s_lo = min(score_a, score_b)
        final_score = min(100.0, float(s_hi + 0.2 * s_lo))

        # series（用于前端K线/指标展示；可关闭避免 JSON 太大）
        series_records: Optional[List[Dict[str, Any]]] = None
        if include_series:
            df_tail = df.iloc[-series_bars:].copy() if series_bars > 0 else df.copy()

            # 可选：附带少量基础指标（不依赖外部脚本）
            df_tail["ma5"] = sma(df_tail["close"], 5)
            df_tail["ma20"] = sma(df_tail["close"], 20)
            df_tail["rsi14"] = rsi(df_tail["close"], 14)
            diff, dea, hist = macd(df_tail["close"])
            df_tail["macd_diff"] = diff
            df_tail["macd_dea"] = dea
            df_tail["macd_hist"] = hist
            df_tail["atr14"] = atr(df_tail, 14)
            df_tail["vol_ratio_5"] = volume_ratio(df_tail["volume"], 5)

            keep = [c for c in [
                "date", "open", "high", "low", "close", "volume",
                "amount", "amplitude", "pct_change", "change", "turnover",
                "ma5", "ma20", "rsi14", "macd_diff", "macd_dea", "macd_hist",
                "atr14", "vol_ratio_5",
            ] if c in df_tail.columns]

            series_records = [{k: row[k] for k in keep} for _, row in df_tail.iterrows()]

        last_date = df["date"].iloc[-1]

        symbol_payload: Dict[str, Any] = {
            "meta": {
                "market": market,
                "code": code,
                "name": name,
                "source_file": str(csv_path),
                "last_date": last_date,
            },
            "score": {
                "final_score": float(final_score),
                "score_A": float(score_a),
                "score_B": float(score_b),
            },
            "signals": signals,
        }
        if include_series and series_records is not None:
            symbol_payload["series"] = series_records

        return symbol_payload, None

    except Exception as e:
        err = {
            "market": market,
            "code": code,
            "file": str(csv_path),
            "error": str(e),
            "trace": traceback.format_exc(limit=8),
        }
        return None, err


@dataclass
class RunConfig:
    data_root: Path
    out_dir: Path
    recent_bars: int
    include_series: bool
    series_bars: int
    top_n: int


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline CSV -> Signals(A+B) -> JSON for web")
    parser.add_argument("--data-root", type=str, default="data/day", help="Root dir containing bj/sz/sh (relative to project root)")
    parser.add_argument("--out-dir", type=str, default="web/client/src/data", help="JSON output dir for web (relative to project root)")
    parser.add_argument("--recent-bars", type=int, default=300, help="Per symbol read last N rows (0=all)")
    parser.add_argument("--include-series", action="store_true", help="Include series[] for chart display")
    parser.add_argument("--series-bars", type=int, default=240, help="Series last N bars (0=all)")
    parser.add_argument("--top-n", type=int, default=200, help="Top N in dashboard.json")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    cfg = RunConfig(
        data_root=normalize_path(args.data_root),
        out_dir=normalize_path(args.out_dir),
        recent_bars=int(args.recent_bars),
        include_series=bool(args.include_series),
        series_bars=int(args.series_bars),
        top_n=int(args.top_n),
    )

    items = discover_csv_files(cfg.data_root)
    if not items:
        logger.error("No CSV found under %s (expect bj/sz/sh)", cfg.data_root)
        return 2

    symbols_dir = cfg.out_dir / "symbols"
    symbols_dir.mkdir(parents=True, exist_ok=True)

    errors: List[Dict[str, Any]] = []
    dashboard_rows: List[Dict[str, Any]] = []

    market_stats = {m: {"total": 0, "ok": 0, "fail": 0} for m in ("bj", "sz", "sh")}

    for market, code, csv_path in items:
        market_stats[market]["total"] += 1

        payload, err = build_symbol_payload(
            market=market,
            code=code,
            csv_path=csv_path,
            recent_bars=cfg.recent_bars,
            include_series=cfg.include_series,
            series_bars=cfg.series_bars,
        )
        
        # 如果 payload 和 err 都为 None，说明是静默跳过（如数据不足 20 天）
        if payload is None and err is None:
            continue

        if err is not None:
            market_stats[market]["fail"] += 1
            errors.append(err)
            continue

        market_stats[market]["ok"] += 1
        safe_write_json(symbols_dir / f"{code}.json", payload)

        final_score = float(payload["score"]["final_score"])
        dashboard_rows.append({
            "market": market,
            "code": code,
            "name": payload["meta"].get("name"),
            "last_date": payload["meta"].get("last_date"),
            "final_score": final_score,
            "score_A": float(payload["score"]["score_A"]),
            "score_B": float(payload["score"]["score_B"]),
            "signals_count": len(payload.get("signals", [])),
        })

    dashboard_rows.sort(key=lambda x: (x["final_score"], x["signals_count"]), reverse=True)

    dashboard = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_root": str(cfg.data_root),
        "out_dir": str(cfg.out_dir),
        "csv_columns_cn": CSV_COLUMNS_CN,
        "markets": market_stats,
        "counts": {
            "symbols_total": len(items),
            "symbols_ok": sum(m["ok"] for m in market_stats.values()),
            "symbols_fail": sum(m["fail"] for m in market_stats.values()),
        },
        "top": dashboard_rows[: cfg.top_n],
    }

    safe_write_json(cfg.out_dir / "dashboard.json", dashboard)
    safe_write_json(cfg.out_dir / "errors.json", {"errors": errors})

    logger.info("Done. dashboard=%s symbols=%s errors=%s",
                cfg.out_dir / "dashboard.json",
                symbols_dir,
                cfg.out_dir / "errors.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
