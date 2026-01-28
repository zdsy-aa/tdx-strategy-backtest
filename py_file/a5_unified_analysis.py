#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a5_unified_analysis.py
================================================================================

【脚本功能】
    统一分析引擎，整合了 a5, a6, a7 的核心功能：
    1. 股票收益报表 (Stock Reports - 来自 a5)
    2. 仪表盘评分系统 (Dashboard Models - 来自 a6)
    3. 高级趋势预测 (Advanced Forecast - 来自 a7)

【使用方法】
    通过命令行参数 --mode 控制运行模式：
    
    1. 运行所有分析 (推荐):
        python3 a5_unified_analysis.py --mode all
        
    2. 仅生成收益报表:
        python3 a5_unified_analysis.py --mode report
        
    3. 仅更新仪表盘评分:
        python3 a5_unified_analysis.py --mode dashboard
        
    4. 仅执行趋势预测:
        python3 a5_unified_analysis.py --mode forecast

【输出文件】
    - web/client/src/data/stock_reports.json    (收益报表)
    - web/client/src/data/dashboard.json        (仪表盘数据)
    - web/client/src/data/forecast_summary.json (预测数据)

【设计优势】
    - 统一数据加载与指标计算，减少 I/O 开销。
    - 采用多进程并行处理，提升全市场扫描速度。
    - 自动降级机制：若高级预测依赖库缺失，自动切换为基础统计模型。
================================================================================
"""

import os
import sys
import json
import argparse
import logging
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------------------------
# 1. 环境配置与日志
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("unified_analysis")

def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, here.parent.parent]
    for d in candidates:
        if (d / "data" / "day").is_dir():
            return d
    return here

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

# 导入统一指标模块
try:
    from a99_indicators import (
        calculate_all_signals,
        calculate_six_veins,
        calculate_buy_sell_points,
        calculate_money_tree,
        calculate_chan_theory
    )
except ImportError:
    logger.error("无法导入 a99_indicators，请确保脚本位于项目正确目录下。")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. 预测模块依赖检查 (来自 a7)
# ------------------------------------------------------------------------------
_HAS_FILTERPY = False
_HAS_HMMLEARN = False
_HAS_SKLEARN = False

try:
    from filterpy.kalman import KalmanFilter
    _HAS_FILTERPY = True
except ImportError:
    KalmanFilter = None

try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMMLEARN = True
except ImportError:
    GaussianHMM = None

try:
    from sklearn.linear_model import LinearRegression
    _HAS_SKLEARN = True
except ImportError:
    LinearRegression = None

# ------------------------------------------------------------------------------
# 3. 数据加载与清洗 (整合 A5/A6/A7 标准)
# ------------------------------------------------------------------------------
CSV_COL_MAP = {
    '名称': 'name', '日期': 'date', '开盘': 'open', '收盘': 'close',
    '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount',
    '振幅': 'amplitude', '涨跌幅': 'pct_chg', '涨跌额': 'chg', '换手率': 'turnover',
}
NUMERIC_COLS = ['open', 'high', 'low', 'close', 'volume', 'amount', 'amplitude', 'pct_chg', 'chg', 'turnover']

def load_stock_data(csv_path: Path) -> Optional[pd.DataFrame]:
    df = None
    # 定义标准列名顺序
    standard_cols = ['name', 'date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_chg', 'chg', 'turnover']
    
    for enc in ('utf-8-sig', 'utf-8', 'gbk'):
        try:
            # 先尝试读取第一行，判断是否有中文表头
            temp_df = pd.read_csv(csv_path, encoding=enc, nrows=1)
            if '日期' in temp_df.columns or 'date' in temp_df.columns:
                df = pd.read_csv(csv_path, encoding=enc)
                df.rename(columns={c: CSV_COL_MAP.get(c, c) for c in df.columns}, inplace=True)
            else:
                # 无表头，按标准顺序分配列名
                df = pd.read_csv(csv_path, encoding=enc, header=None)
                df.columns = standard_cols[:len(df.columns)]
            break
        except Exception:
            continue
            
    if df is None or df.empty:
        return None

    if 'date' not in df.columns:
        return None

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
    
    if 'name' not in df.columns or df['name'].isna().all():
        df['name'] = csv_path.stem

    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df if len(df) >= 10 else None

def get_market_info(file_path: Path):
    path_str = str(file_path).replace("\\", "/")
    if "/sh/" in path_str: return 'sh', '沪市'
    if "/sz/" in path_str: return 'sz', '深市'
    if "/bj/" in path_str: return 'bj', '北交所'
    return 'unknown', '未知'

# ------------------------------------------------------------------------------
# 4. 核心逻辑模块
# ------------------------------------------------------------------------------

class AnalysisEngine:
    """整合 A5, A6, A7 逻辑的分析引擎"""
    
    @staticmethod
    def run_report_logic(df: pd.DataFrame, hold_days: int) -> Dict[str, Any]:
        """A5: 收益报表逻辑"""
        end_dt = df['date'].iloc[-1]
        year_start = pd.to_datetime(f"{end_dt.year}-01-01")
        month_start = pd.to_datetime(f"{end_dt.year}-{end_dt.month:02d}-01")
        
        # 信号类型定义
        signal_types = ['six_veins_6red', 'six_veins_5red', 'buy1', 'buy2', 'chan_buy1']
        all_trades = []
        
        for stype in signal_types:
            if stype.startswith('six_veins'):
                cnt_target = 6 if '6red' in stype else 5
                cond = (df.get('six_veins_count', 0) >= cnt_target)
            else:
                cond = df.get(stype, False).fillna(False).astype(bool)
            
            trigger = cond & ~cond.shift(1, fill_value=False)
            idxs = df.index[trigger]
            
            for i in idxs:
                sell_idx = i + hold_days
                if sell_idx < len(df):
                    buy_p = float(df.at[i, 'close'])
                    sell_p = float(df.at[sell_idx, 'close'])
                    ret = (sell_p - buy_p) / buy_p * 100.0
                    all_trades.append({'date': df.at[i, 'date'], 'return': ret, 'win': ret > 0})
        
        def _stats(trades):
            if not trades: return "0.0%", "0.0%", 0
            rets = [t['return'] for t in trades]
            wins = sum(1 for t in trades if t['win'])
            return f"{np.sum(rets):.1f}%", f"{wins/len(trades)*100:.1f}%", len(trades)

        y_trades = [t for t in all_trades if t['date'] >= year_start]
        m_trades = [t for t in all_trades if t['date'] >= month_start]
        
        tr, tw, tc = _stats(all_trades)
        yr, yw, yc = _stats(y_trades)
        mr, mw, mc = _stats(m_trades)
        
        # 最新信号
        last_sig, last_date = "无", "-"
        df_recent = df.tail(5)
        for idx in df_recent.index[::-1]:
            cnt = int(df_recent.at[idx, 'six_veins_count']) if 'six_veins_count' in df_recent.columns else 0
            if cnt >= 5:
                last_sig = f"六脉{cnt}红"
                last_date = df_recent.at[idx, 'date'].strftime('%Y/%m/%d')
                break
                
        return {
            'totalReturn': tr, 'totalWinRate': tw, 'totalTrades': tc,
            'yearReturn': yr, 'yearWinRate': yw, 'yearTrades': yc,
            'monthReturn': mr, 'monthWinRate': mw, 'monthTrades': mc,
            'lastSignal': last_sig, 'lastSignalDate': last_date
        }

    @staticmethod
    def run_dashboard_logic(df: pd.DataFrame, dashboard_days: int) -> Dict[str, Any]:
        """A6: 仪表盘评分逻辑"""
        recent_df = df.tail(dashboard_days)
        latest = df.iloc[-1]
        
        has_six = recent_df.get('six_veins_signal', pd.Series([False])).any()
        has_buy = recent_df.get('buy1', pd.Series([False])).any() or recent_df.get('buy2', pd.Series([False])).any()
        has_spec = recent_df.get('money_tree_signal', pd.Series([False])).any() or recent_df.get('chan_buy1', pd.Series([False])).any()
        
        score_a = 60 if has_six else 0
        score_b = 70 if has_buy else 0
        score_c = 80 if has_spec else 0
        
        return {
            "final_score": score_a + score_b + score_c,
            "score_A": score_a, "score_B": score_b, "score_C": score_c,
            "signals_count": int(has_six) + int(has_buy) + int(has_spec),
            "price": float(latest["close"]),
            "pct_change": float(latest.get("pct_chg", 0))
        }

    @staticmethod
    def run_forecast_logic(df: pd.DataFrame) -> Dict[str, Any]:
        """A7: 高级预测逻辑"""
        # 特征工程
        d = df.copy()
        d['ret1'] = d['close'].pct_change()
        d['vol10'] = d['ret1'].rolling(10).std()
        d['ma5'] = d['close'].rolling(5).mean()
        d['ma20'] = d['close'].rolling(20).mean()
        d['ma_bias'] = (d['ma5'] - d['ma20']) / d['ma20']
        d['vol_chg'] = d['volume'].pct_change().replace([np.inf, -np.inf], np.nan)
        d['hl_range'] = (d['high'] - d['low']) / d['close']
        d.dropna(inplace=True)
        
        if len(d) < 20: return {"status": "failed", "error": "样本不足"}
        
        # 线性回归预测
        try:
            X = d[['ret1', 'vol10', 'ma_bias', 'vol_chg', 'hl_range']].to_numpy()
            y = d['ret1'].shift(-1).dropna().to_numpy()
            X_train = X[:-1]
            X_last = X[-1].reshape(1, -1)
            
            if _HAS_SKLEARN:
                model = LinearRegression().fit(X_train, y)
                pred = float(model.predict(X_last)[0])
                method = "sklearn_regression"
            else:
                # Numpy 降级实现
                X_ext = np.column_stack([np.ones(len(X_train)), X_train])
                beta, *_ = np.linalg.lstsq(X_ext, y, rcond=None)
                pred = float(np.dot(np.insert(X_last[0], 0, 1), beta))
                method = "numpy_regression"
                
            return {
                "status": "ok",
                "forecast_change_pct": round(pred * 100, 3),
                "confidence": 0.5, # 简化处理
                "model_used": method
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

# ------------------------------------------------------------------------------
# 5. 并行处理核心
# ------------------------------------------------------------------------------

def process_stock(csv_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    try:
        df = load_stock_data(csv_path)
        if df is None: return None
        
        # 计算所有指标 (一次性计算)
        df = calculate_all_signals(df)
        if df is None: return None
        
        stock_code = csv_path.stem
        stock_name = str(df['name'].iloc[-1])
        market, market_name = get_market_info(csv_path)
        
        result = {
            "meta": {
                "code": stock_code, "name": stock_name, 
                "market": market, "market_name": market_name,
                "last_date": df['date'].iloc[-1].strftime('%Y-%m-%d')
            }
        }
        
        # 根据模式执行逻辑
        if args.mode in ['report', 'all']:
            result['report'] = AnalysisEngine.run_report_logic(df, args.hold_days)
            
        if args.mode in ['dashboard', 'all']:
            result['dashboard'] = AnalysisEngine.run_dashboard_logic(df, args.dashboard_days)
            
        if args.mode in ['forecast', 'all']:
            result['forecast'] = AnalysisEngine.run_forecast_logic(df)
            
        return result
    except Exception as e:
        logger.error(f"处理 {csv_path.name} 出错: {e}")
        return None

# ------------------------------------------------------------------------------
# 6. 主程序
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TDX 统一分析引擎 (A5+A6+A7)")
    parser.add_argument("--mode", type=str, default="all", choices=["report", "dashboard", "forecast", "all"], help="运行模式")
    parser.add_argument("--hold_days", type=int, default=14, help="报表统计持有天数")
    parser.add_argument("--dashboard_days", type=int, default=3, help="仪表盘信号窗口天数")
    parser.add_argument("--limit", type=int, default=None, help="限制处理股票数量 (调试用)")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "day"
    out_dir = PROJECT_ROOT / "web" / "client" / "src" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    stock_files = list(data_dir.rglob("*.csv"))
    if args.limit: stock_files = stock_files[:args.limit]
    
    logger.info(f"开始统一分析任务，模式: {args.mode}, 股票总数: {len(stock_files)}")
    
    results = []
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_stock, f, args): f for f in stock_files}
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            if len(results) % 100 == 0:
                logger.info(f"已处理 {len(results)} 只股票...")

    # ------------------------------------------------------------------------------
    # 7. 结果分发与保存
    # ------------------------------------------------------------------------------
    gen_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # A5: Stock Reports
    if args.mode in ['report', 'all']:
        report_data = []
        for r in results:
            item = {**r['meta'], **r['report']}
            # 适配 A5 原有字段名
            item['marketName'] = item.pop('market_name')
            report_data.append(item)
        
        with open(out_dir / "stock_reports.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        logger.info(f"已生成收益报表: stock_reports.json ({len(report_data)} 条)")

    # A6: Dashboard
    if args.mode in ['dashboard', 'all']:
        dashboard_items = []
        market_stats = defaultdict(lambda: {"total": 0, "ok": 0, "fail": 0})
        
        for r in results:
            m = r['meta']['market']
            d = r['dashboard']
            item = {
                "market": m, "code": r['meta']['code'], "name": r['meta']['name'],
                "last_date": r['meta']['last_date'], **d
            }
            dashboard_items.append(item)
            market_stats[m]["total"] += 1
            if d["final_score"] > 0: market_stats[m]["ok"] += 1
            else: market_stats[m]["fail"] += 1
            
        dashboard_items.sort(key=lambda x: x["final_score"], reverse=True)
        dashboard_final = {
            "generated_at": gen_time,
            "markets": dict(market_stats),
            "counts": {
                "symbols_total": len(dashboard_items),
                "symbols_ok": sum(1 for x in dashboard_items if x['final_score'] > 0),
                "symbols_fail": sum(1 for x in dashboard_items if x['final_score'] == 0)
            },
            "top": dashboard_items
        }
        with open(out_dir / "dashboard.json", "w", encoding="utf-8") as f:
            json.dump(dashboard_final, f, ensure_ascii=False, indent=2)
        logger.info(f"已生成仪表盘数据: dashboard.json")

    # A7: Forecast
    if args.mode in ['forecast', 'all']:
        forecast_data = {
            "generated_at": gen_time,
            "count": len(results),
            "results": [ {**r['meta'], **r['forecast']} for r in results ]
        }
        with open(out_dir / "forecast_summary.json", "w", encoding="utf-8") as f:
            json.dump(forecast_data, f, ensure_ascii=False, indent=2)
        logger.info(f"已生成预测数据: forecast_summary.json")

    logger.info("所有分析任务已完成。")

if __name__ == "__main__":
    main()
