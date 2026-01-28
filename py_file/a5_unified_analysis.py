#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a5_unified_analysis.py (完整整合版)
================================================================================

【脚本功能】
    统一分析报表引擎，整合了以下三个模块的完整功能：
    1. 股票收益报表 (a5_generate_stock_reports.py)：统计收益表现
    2. 仪表盘评分系统 (a6_models.py)：基于技术指标的综合评分
    3. 趋势预测 (a7_advanced_forecast.py)：高级预测算法

【生成的前端JSON文件】
    - stock_reports.json       (收益报表数据)
    - dashboard.json           (仪表盘评分数据)
    - forecast_summary.json    (趋势预测数据)

【使用方法】
    # 运行所有分析 (推荐)
    python3 a5_unified_analysis.py --mode all
    
    # 单独运行某个模式
    python3 a5_unified_analysis.py --mode report
    python3 a5_unified_analysis.py --mode dashboard
    python3 a5_unified_analysis.py --mode forecast
    
    # 自定义参数
    python3 a5_unified_analysis.py --mode all --hold_days 14 --limit 0

【设计优势】
    - 资源复用：一次性加载数据，并行执行多项分析
    - 智能降级：若高级库缺失，自动切换至基础实现
    - 格式统一：输出标准JSON数据，直接对接前端
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

# 可选高级依赖
try:
    from filterpy.kalman import KalmanFilter
    _HAS_FILTERPY = True
except Exception:
    KalmanFilter = None
    _HAS_FILTERPY = False

try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMMLEARN = True
except Exception:
    GaussianHMM = None
    _HAS_HMMLEARN = False

try:
    from sklearn.linear_model import LinearRegression
    _HAS_SKLEARN = True
except Exception:
    LinearRegression = None
    _HAS_SKLEARN = False

# ------------------------------------------------------------------------------
# 日志配置
# ------------------------------------------------------------------------------
def log(msg: str, level: str = "INFO"):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}")

# ------------------------------------------------------------------------------
# 项目根目录探测
# ------------------------------------------------------------------------------
def find_project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
    for d in candidates:
        if os.path.isdir(os.path.join(d, "data", "day")):
            return d
    return here

PROJECT_ROOT = find_project_root()
sys.path.insert(0, PROJECT_ROOT)

from a99_indicators import calculate_all_signals, calculate_six_veins, calculate_buy_sell_points, calculate_chan_theory

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "day")
WEB_DATA_DIR = os.path.join(PROJECT_ROOT, "web", "client", "src", "data")

# ------------------------------------------------------------------------------
# 数据加载与清洗
# ------------------------------------------------------------------------------
CSV_COL_MAP = {
    '名称': 'name', '日期': 'date', '开盘': 'open', '收盘': 'close',
    '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount',
    '振幅': 'amplitude', '涨跌幅': 'pct_chg', '涨跌额': 'chg', '换手率': 'turnover',
}
NUMERIC_COLS = ['open', 'high', 'low', 'close', 'volume', 'amount', 'amplitude', 'pct_chg', 'chg', 'turnover']

def _parse_date_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, format='%Y/%m/%d', errors='coerce')
    if len(dt) > 0 and dt.isna().mean() > 0.5:
        dt = pd.to_datetime(s, errors='coerce')
    return dt

def load_stock_data(filepath: str) -> Optional[pd.DataFrame]:
    """加载并标准化单只股票CSV数据"""
    df = None
    for enc in ('utf-8-sig', 'utf-8', 'gbk'):
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except Exception:
            continue
    if df is None or df.empty:
        return None

    df.rename(columns={c: CSV_COL_MAP.get(c, c) for c in df.columns}, inplace=True)
    if 'date' not in df.columns:
        return None

    df['date'] = _parse_date_series(df['date'])
    df.dropna(subset=['date'], inplace=True)

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c not in df.columns:
            df[c] = np.nan if c != 'volume' else 0.0

    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

    for c in ['volume', 'amount']:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
            df.loc[df[c] < 0, c] = 0.0

    if 'name' not in df.columns:
        df['name'] = ''

    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df if len(df) > 0 else None

def _to_jsonable(x: Any) -> Any:
    """将数据转换为JSON可序列化格式，处理NaN和Inf"""
    if x is None:
        return None
    if isinstance(x, (str, bool)):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if np.isnan(x) or np.isinf(x):
            return None
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

def get_market_info(file_path: str):
    """从路径判断市场信息"""
    path_str = str(file_path).replace("\\", "/")
    if "/sh/" in path_str: return 'sh', '沪市'
    if "/sz/" in path_str: return 'sz', '深市'
    if "/bj/" in path_str: return 'bj', '北交所'
    return 'unknown', '未知'

# ==============================================================================
# 模块 1: 股票收益报表 (来自 a5_generate_stock_reports.py)
# ==============================================================================

def find_signals(df: pd.DataFrame, signal_type: str) -> List[Dict]:
    """返回指定信号类型的触发列表"""
    signals: List[Dict] = []

    if signal_type == 'six_veins_6red':
        cond = (df.get('six_veins_count', 0) == 6)
    elif signal_type == 'six_veins_5red':
        cond = (df.get('six_veins_count', 0) >= 5)
    elif signal_type == 'six_veins_4red':
        cond = (df.get('six_veins_count', 0) >= 4)
    elif signal_type == 'buy_point_1':
        cond = df.get('buy1', False).fillna(False).astype(bool)
    elif signal_type == 'buy_point_2':
        cond = df.get('buy2', False).fillna(False).astype(bool)
    else:
        cond = df.get(signal_type, False)
        if isinstance(cond, pd.Series):
            cond = cond.fillna(False).astype(bool)
        else:
            cond = pd.Series([False] * len(df))

    cond = cond.fillna(False).astype(bool)
    trigger = cond & ~cond.shift(1, fill_value=False)

    idxs = list(df.index[trigger])
    for i in idxs:
        signals.append({'date': df.at[i, 'date'], 'price': float(df.at[i, 'close'])})
    return signals

def calculate_trade_result(df: pd.DataFrame, signal: Dict, hold_days: int) -> Optional[Dict]:
    """固定持有hold_days天后的收益率"""
    buy_dt = signal['date']
    buy_idx_list = df.index[df['date'] == buy_dt].tolist()
    if not buy_idx_list:
        return None
    buy_idx = buy_idx_list[0]
    sell_idx = buy_idx + int(hold_days)
    if sell_idx >= len(df):
        return None

    buy_price = float(signal.get('price', np.nan))
    sell_price = float(df.at[sell_idx, 'close'])

    if not np.isfinite(buy_price) or buy_price <= 0:
        return None
    if not np.isfinite(sell_price) or sell_price <= 0:
        return None

    ret = (sell_price - buy_price) / buy_price * 100.0
    return {'buy_date': buy_dt, 'return': float(ret), 'win': bool(ret > 0)}

def generate_stock_report(stock_file: str, end_date: str, hold_days: int) -> Optional[Dict]:
    """生成单只股票的收益报表"""
    try:
        stock_code = Path(stock_file).stem
        market, market_name = get_market_info(stock_file)

        df = load_stock_data(stock_file)
        if df is None or len(df) < 30:
            return None

        stock_name = str(df['name'].iloc[-1]) if 'name' in df.columns and len(df) > 0 else stock_code

        # 计算指标
        df = calculate_all_signals(df)
        if df is None or df.empty:
            return None

        # 时间范围
        end_dt = pd.to_datetime(end_date)
        year_start = pd.Timestamp(f"{end_dt.year}-01-01")
        month_start = pd.Timestamp(f"{end_dt.year}-{end_dt.month:02d}-01")

        # 统计各策略信号
        base_types = ['six_veins_6red', 'six_veins_5red', 'six_veins_4red', 'buy_point_1', 'buy_point_2']
        chan_types = ['chan_buy1', 'chan_buy2', 'chan_buy3', 'chan_strong_buy2', 'chan_like_buy2']

        all_trades: List[Dict] = []
        year_trades: List[Dict] = []
        month_trades: List[Dict] = []

        for stype in base_types + chan_types:
            sigs = find_signals(df, stype)
            for sig in sigs:
                res = calculate_trade_result(df, sig, hold_days)
                if res is None:
                    continue
                res['signal_type'] = stype
                all_trades.append(res)

        if not all_trades:
            return None

        year_trades = [t for t in all_trades if t['buy_date'] >= year_start]
        month_trades = [t for t in all_trades if t['buy_date'] >= month_start]

        def _sum_ret(trades: List[Dict]) -> float:
            return float(np.sum([t['return'] for t in trades])) if trades else 0.0

        def _win_rate(trades: List[Dict]) -> float:
            if not trades:
                return 0.0
            wins = sum(1 for t in trades if t['win'])
            return wins / len(trades) * 100.0

        # 最新信号（最近5日）
        last_signal = "无"
        last_date = "-"
        recent_df = df.tail(5)
        if any(recent_df.get('six_veins_count', 0) == 6):
            last_signal = "六脉6红"
            idx = recent_df[recent_df.get('six_veins_count', 0) == 6].index[-1]
            last_date = df.at[idx, 'date'].strftime('%Y/%m/%d')
        elif any(recent_df.get('six_veins_count', 0) >= 5):
            last_signal = "六脉5红+"
            idx = recent_df[recent_df.get('six_veins_count', 0) >= 5].index[-1]
            last_date = df.at[idx, 'date'].strftime('%Y/%m/%d')

        return {
            'code': stock_code,
            'name': stock_name,
            'market': market,
            'market_name': market_name,
            'cumulative_return': round(_sum_ret(all_trades), 2),
            'cumulative_win_rate': round(_win_rate(all_trades), 2),
            'cumulative_trades': len(all_trades),
            'yearly_return': round(_sum_ret(year_trades), 2),
            'yearly_win_rate': round(_win_rate(year_trades), 2),
            'yearly_trades': len(year_trades),
            'monthly_return': round(_sum_ret(month_trades), 2),
            'monthly_win_rate': round(_win_rate(month_trades), 2),
            'monthly_trades': len(month_trades),
            'last_signal': last_signal,
            'last_signal_date': last_date,
        }
    except Exception as e:
        log(f"处理股票报表失败 {stock_file}: {e}", level="ERROR")
        return None

# ==============================================================================
# 模块 2: 仪表盘评分 (来自 a6_models.py)
# ==============================================================================

def calculate_dashboard_score(stock_file: str, dashboard_days: int = 3) -> Optional[Dict]:
    """计算仪表盘评分"""
    try:
        stock_code = Path(stock_file).stem
        market, market_name = get_market_info(stock_file)

        df = load_stock_data(stock_file)
        if df is None or len(df) < 30:
            return None

        stock_name = str(df['name'].iloc[-1]) if 'name' in df.columns and len(df) > 0 else stock_code

        # 计算所有指标
        df = calculate_six_veins(df)
        df = calculate_buy_sell_points(df)
        df = calculate_chan_theory(df)

        # 提取最新数据
        latest = df.iloc[-1]

        # 判断最近N天内是否有信号出现
        recent_df = df.tail(dashboard_days)

        # 策略A: 六脉神剑 (6红)
        has_six_veins = recent_df.get('six_veins_signal', pd.Series([False] * len(recent_df))).any()
        # 策略B: 庄家买点 (buy1 或 buy2)
        has_buy_sell = recent_df.get('buy1', pd.Series([False] * len(recent_df))).any() or \
                      recent_df.get('buy2', pd.Series([False] * len(recent_df))).any()
        # 策略C: 缠论买点
        has_chan = recent_df.get('chan_buy1', pd.Series([False] * len(recent_df))).any()

        # 计算各策略得分
        score_A = 60 if has_six_veins else 0
        score_B = 70 if has_buy_sell else 0
        score_C = 80 if has_chan else 0
        final_score = score_A + score_B + score_C

        # 统计信号数量
        signals_count = int(has_six_veins) + int(has_buy_sell) + int(has_chan)

        return {
            'market': market,
            'code': stock_code,
            'name': stock_name,
            'last_date': latest['date'].strftime('%Y-%m-%d'),
            'final_score': final_score,
            'score_A': score_A,
            'score_B': score_B,
            'score_C': score_C,
            'signals_count': signals_count,
            'price': float(latest['close']) if pd.notna(latest['close']) else None,
            'pct_change': float(latest.get('pct_chg', 0)) if pd.notna(latest.get('pct_chg', 0)) else None,
        }
    except Exception as e:
        log(f"处理仪表盘评分失败 {stock_file}: {e}", level="ERROR")
        return None

# ==============================================================================
# 模块 3: 高级趋势预测 (来自 a7_advanced_forecast.py)
# ==============================================================================

class AdvancedForecaster:
    """高级预测器（支持依赖自动降级）"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.code = self.csv_path.stem
        self.df = load_stock_data(self.csv_path)
        self.name = ''
        if self.df is not None and 'name' in self.df.columns and len(self.df) > 0:
            self.name = str(self.df['name'].iloc[-1])

    def _basic_features(self) -> Optional[pd.DataFrame]:
        """构建基础特征"""
        if self.df is None or len(self.df) < 60:
            return None
        df = self.df.copy()
        df['ret1'] = df['close'].pct_change()
        df['vol10'] = df['ret1'].rolling(10).std()
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma_bias'] = (df['ma5'] - df['ma20']) / df['ma20']
        df['vol_chg'] = df['volume'].pct_change().replace([np.inf, -np.inf], np.nan)
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df.dropna(inplace=True)
        if len(df) < 30:
            return None
        return df

    def _kalman_smooth(self, series: np.ndarray) -> np.ndarray:
        """使用Kalman滤波平滑收盘价"""
        if not _HAS_FILTERPY or series is None or len(series) < 5:
            return series
        try:
            kf = KalmanFilter(dim_x=2, dim_z=1)
            kf.x = np.array([[series[0]], [0.]])
            kf.F = np.array([[1., 1.], [0., 1.]])
            kf.H = np.array([[1., 0.]])
            kf.P *= 1000.
            kf.R = 0.01
            kf.Q = np.array([[1e-5, 0.], [0., 1e-5]])

            smoothed = []
            for z in series:
                kf.predict()
                kf.update(np.array([[z]]))
                smoothed.append(float(kf.x[0, 0]))
            return np.array(smoothed, dtype=float)
        except Exception:
            return series

    def _hmm_regime(self, returns: np.ndarray) -> Optional[int]:
        """简单HMM市场状态识别"""
        if not _HAS_HMMLEARN or returns is None or len(returns) < 50:
            return None
        try:
            X = returns.reshape(-1, 1)
            model = GaussianHMM(n_components=2, covariance_type="full", n_iter=50, random_state=7)
            model.fit(X)
            states = model.predict(X)
            return int(states[-1])
        except Exception:
            return None

    def _regression_forecast(self, feat_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """回归预测下一日收益率"""
        df = feat_df.copy()
        df['y'] = df['ret1'].shift(-1)
        df.dropna(inplace=True)
        if len(df) < 30:
            return None

        X = df[['ret1', 'vol10', 'ma_bias', 'vol_chg', 'hl_range']].to_numpy(dtype=float)
        y = df['y'].to_numpy(dtype=float)
        x_last = df[['ret1', 'vol10', 'ma_bias', 'vol_chg', 'hl_range']].iloc[-1].to_numpy(dtype=float).reshape(1, -1)

        if _HAS_SKLEARN:
            try:
                lr = LinearRegression()
                lr.fit(X, y)
                pred = float(lr.predict(x_last)[0])
                model_used = "sklearn_linear_regression"
                y_hat = lr.predict(X)
                resid = y - y_hat
                sigma = float(np.nanstd(resid))
                confidence = max(0.0, 1.0 - min(1.0, sigma * 10))
            except Exception:
                pred = float(np.nan)
                model_used = "sklearn_failed"
                confidence = 0.3
        else:
            # numpy 线性回归降级
            try:
                X1 = np.column_stack([np.ones(len(X)), X])
                beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
                pred = float(np.dot(np.r_[1.0, x_last.flatten()], beta))
                model_used = "numpy_linear_regression"
                y_hat = X1 @ beta
                resid = y - y_hat
                sigma = float(np.nanstd(resid))
                confidence = max(0.0, 1.0 - min(1.0, sigma * 10))
            except Exception:
                pred = float(np.nan)
                model_used = "numpy_failed"
                confidence = 0.3

        if not np.isfinite(pred):
            return None

        return {
            'forecast_change_pct': round(pred * 100, 3),
            'confidence': round(float(confidence), 3),
            'model_used': model_used
        }

    def run(self) -> Dict[str, Any]:
        """执行预测"""
        if self.df is None or len(self.df) < 60:
            return {
                'code': self.code,
                'name': self.name or self.code,
                'status': 'failed',
                'error': '数据不足(<60) 或读取失败'
            }

        feat_df = self._basic_features()
        if feat_df is None:
            return {
                'code': self.code,
                'name': self.name or self.code,
                'status': 'failed',
                'error': '特征构建失败或样本太少'
            }

        # Kalman平滑（可选）
        close = feat_df['close'].to_numpy(dtype=float)
        smooth_close = self._kalman_smooth(close)
        if smooth_close is not None and len(smooth_close) == len(feat_df):
            feat_df['smooth_close'] = smooth_close

        # HMM市场状态（可选）
        ret = feat_df['ret1'].to_numpy(dtype=float)
        regime = self._hmm_regime(ret)

        # 回归预测
        reg_result = self._regression_forecast(feat_df)
        if reg_result is None:
            return {
                'code': self.code,
                'name': self.name or self.code,
                'status': 'failed',
                'error': '回归预测失败'
            }

        return {
            'code': self.code,
            'name': self.name or self.code,
            'status': 'success',
            'forecast_change_pct': reg_result['forecast_change_pct'],
            'confidence': reg_result['confidence'],
            'model_used': reg_result['model_used'],
            'regime': regime if regime is not None else -1,
            'has_kalman': _HAS_FILTERPY,
            'has_hmm': _HAS_HMMLEARN,
        }

# ==============================================================================
# 主流程控制
# ==============================================================================

def run_analysis_main(mode: str = "all", hold_days: int = 14, 
                     dashboard_days: int = 3, limit: int = 0):
    """主运行函数"""
    log("=" * 80)
    log(f"开始运行统一分析引擎，模式: {mode}")
    log(f"参数: hold_days={hold_days}, dashboard_days={dashboard_days}, limit={limit}")
    log("=" * 80)
    
    # 获取所有股票文件
    all_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".csv"):
                all_files.append(os.path.join(root, f))
    
    if limit > 0:
        all_files = all_files[:limit]
    
    log(f"找到 {len(all_files)} 只股票数据文件")
    
    os.makedirs(WEB_DATA_DIR, exist_ok=True)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 模式 1: 股票收益报表
    if mode in ["report", "all"]:
        log("\n" + "=" * 80)
        log("开始生成股票收益报表...")
        log("=" * 80)
        
        report_func = partial(generate_stock_report, end_date=end_date, hold_days=hold_days)
        reports = []
        
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = {executor.submit(report_func, f): f for f in all_files}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        reports.append(result)
                except Exception as e:
                    log(f"报表生成失败: {e}", level="ERROR")
        
        if reports:
            # 排序：按累计收益降序
            reports.sort(key=lambda x: x.get('cumulative_return', 0), reverse=True)
            
            output_path = os.path.join(WEB_DATA_DIR, "stock_reports.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(reports), f, ensure_ascii=False, indent=2)
            log(f"股票收益报表已保存: {output_path} (共{len(reports)}只)")
        else:
            log("收益报表无有效结果", level="WARNING")
    
    # 模式 2: 仪表盘评分
    if mode in ["dashboard", "all"]:
        log("\n" + "=" * 80)
        log("开始生成仪表盘评分...")
        log("=" * 80)
        
        dashboard_func = partial(calculate_dashboard_score, dashboard_days=dashboard_days)
        dashboards = []
        market_stats = defaultdict(lambda: {"total": 0, "ok": 0, "fail": 0})
        
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = {executor.submit(dashboard_func, f): f for f in all_files}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        dashboards.append(result)
                        market = result.get('market', 'unknown')
                        market_stats[market]["total"] += 1
                        if result['final_score'] > 0:
                            market_stats[market]["ok"] += 1
                        else:
                            market_stats[market]["fail"] += 1
                except Exception as e:
                    log(f"仪表盘评分失败: {e}", level="ERROR")
        
        if dashboards:
            # 排序：按评分降序
            dashboards.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            
            total_ok = sum(s["final_score"] > 0 for s in dashboards)
            total_fail = len(dashboards) - total_ok
            
            dashboard_data = {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "markets": dict(market_stats),
                "counts": {
                    "symbols_total": len(dashboards),
                    "symbols_ok": total_ok,
                    "symbols_fail": total_fail
                },
                "top": dashboards
            }
            
            output_path = os.path.join(WEB_DATA_DIR, "dashboard.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(dashboard_data), f, ensure_ascii=False, indent=2)
            log(f"仪表盘评分已保存: {output_path} (共{len(dashboards)}只)")
            log(f"市场统计: {dict(market_stats)}")
            log(f"信号统计: OK={total_ok}, FAIL={total_fail}")
        else:
            log("仪表盘评分无有效结果", level="WARNING")
    
    # 模式 3: 趋势预测
    if mode in ["forecast", "all"]:
        log("\n" + "=" * 80)
        log("开始执行趋势预测...")
        log(f"可用库: Kalman={_HAS_FILTERPY}, HMM={_HAS_HMMLEARN}, Sklearn={_HAS_SKLEARN}")
        log("=" * 80)
        
        forecasts = []
        success_count = 0
        fail_count = 0
        
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = {executor.submit(AdvancedForecaster(f).run): f for f in all_files}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        forecasts.append(result)
                        if result.get('status') == 'success':
                            success_count += 1
                        else:
                            fail_count += 1
                except Exception as e:
                    fail_count += 1
                    log(f"趋势预测失败: {e}", level="ERROR")
        
        if forecasts:
            # 按置信度排序
            forecasts_success = [f for f in forecasts if f.get('status') == 'success']
            forecasts_success.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            forecast_data = {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total": len(forecasts),
                "success": success_count,
                "failed": fail_count,
                "has_filterpy": _HAS_FILTERPY,
                "has_hmmlearn": _HAS_HMMLEARN,
                "has_sklearn": _HAS_SKLEARN,
                "forecasts": forecasts_success[:500]  # 只保存前500个成功结果
            }
            
            output_path = os.path.join(WEB_DATA_DIR, "forecast_summary.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(forecast_data), f, ensure_ascii=False, indent=2)
            log(f"趋势预测已保存: {output_path}")
            log(f"预测统计: 成功={success_count}, 失败={fail_count}")
        else:
            log("趋势预测无有效结果", level="WARNING")
    
    log("\n" + "=" * 80)
    log("统一分析引擎运行完成！")
    log("=" * 80)

# ==============================================================================
# 命令行入口
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一分析报表引擎")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["report", "dashboard", "forecast", "all"],
                       help="运行模式: report(收益报表)/dashboard(仪表盘)/forecast(趋势预测)/all(全部)")
    parser.add_argument("--hold_days", type=int, default=14,
                       help="收益报表：固定持有天数（默认14）")
    parser.add_argument("--dashboard_days", type=int, default=3,
                       help="仪表盘：统计最近几天内的信号（默认3）")
    parser.add_argument("--limit", type=int, default=0,
                       help="限制处理数量（0=不限制，用于调试）")
    
    args = parser.parse_args()
    
    run_analysis_main(
        mode=args.mode,
        hold_days=args.hold_days,
        dashboard_days=args.dashboard_days,
        limit=args.limit
    )