\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a7_advanced_forecast.py
================================================================================

【脚本功能】
    对 data/day 目录下的股票日线数据做“高级预测”汇总，输出前端可用 JSON：
        web/client/src/data/forecast_summary.json

    本脚本尽量使用更复杂的统计/机器学习方法（如 Kalman / HMM / 回归），
    但在依赖库缺失时会自动降级为“稳健的基础预测”，保证脚本可跑、不崩。

【数据输入要求】
    CSV 字段（与你的数据一致）：
        名称、日期(yyyy/mm/dd)、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌幅、涨跌额、换手率
    脚本会自动映射为英文列 date/open/high/low/close/volume 并清洗异常。

【输出文件】
    - web/client/src/data/forecast_summary.json

【使用方法】
    python3 a7_advanced_forecast.py
    可选参数：
        --limit 300          # 仅预测前N只股票（调试）

【异常/边界处理（重点）】
    - 价格<=0/NaN 行会剔除；避免除0与指标崩溃
    - 数据长度过短（<60）直接返回 failed
    - 依赖库缺失自动降级：filterpy/hmmlearn/sklearn 都是可选依赖
================================================================================
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

# 可选依赖：缺失则降级
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
# 日志
# ------------------------------------------------------------------------------
def log(msg: str, level: str = "INFO"):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}")

# ------------------------------------------------------------------------------
# 项目根目录探测
# ------------------------------------------------------------------------------
def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, here.parent.parent]
    for d in candidates:
        if (d / "data" / "day").is_dir():
            return d
    return here

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------------------
# CSV读取与标准化
# ------------------------------------------------------------------------------
CSV_COL_MAP = {
    '名称': 'name',
    '日期': 'date',
    '开盘': 'open',
    '收盘': 'close',
    '最高': 'high',
    '最低': 'low',
    '成交量': 'volume',
    '成交额': 'amount',
    '振幅': 'amplitude',
    '涨跌幅': 'pct_chg',
    '涨跌额': 'chg',
    '换手率': 'turnover',
}
NUMERIC_COLS = ['open', 'high', 'low', 'close', 'volume', 'amount', 'amplitude', 'pct_chg', 'chg', 'turnover']

def _parse_date_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, format='%Y/%m/%d', errors='coerce')
    if len(dt) > 0 and dt.isna().mean() > 0.5:
        dt = pd.to_datetime(s, errors='coerce')
    return dt

def load_stock_data(csv_path: Path) -> Optional[pd.DataFrame]:
    df = None
    for enc in ('utf-8-sig', 'utf-8', 'gbk'):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
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

# ------------------------------------------------------------------------------
# 预测器
# ------------------------------------------------------------------------------
class AdvancedForecaster:
    """
    高级预测器（可选依赖自动降级）：
    - 基础特征：收益率、波动率、均线偏离、量能变化
    - 预测目标：下一交易日收盘变化百分比（forecast_change_pct）
    - 输出：code/name/status/model_used/confidence 等字段
    """
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.code = self.csv_path.stem
        self.df = load_stock_data(self.csv_path)
        self.name = ''
        if self.df is not None and 'name' in self.df.columns and len(self.df) > 0:
            self.name = str(self.df['name'].iloc[-1])

    def _basic_features(self) -> Optional[pd.DataFrame]:
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
        """
        使用Kalman滤波平滑收盘价（若 filterpy 不可用，则直接返回原序列）。
        """
        if not _HAS_FILTERPY or series is None or len(series) < 5:
            return series
        try:
            kf = KalmanFilter(dim_x=2, dim_z=1)
            # 状态：价格 + 速度
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
        """
        简单HMM市场状态识别：
        返回最近一天的 regime id（若 hmmlearn 不可用则返回 None）。
        """
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
        """
        回归预测下一日收益率（%）：
        - 若 sklearn 不可用，使用 numpy 最小二乘（线性回归）降级实现
        """
        # 目标：下一日 ret1
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
            except Exception:
                pred = float(np.nan)
                model_used = "sklearn_linear_regression_failed"
        else:
            # numpy 线性回归：加一列常数项
            try:
                X1 = np.column_stack([np.ones(len(X)), X])
                beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
                pred = float(np.dot(np.r_[1.0, x_last.flatten()], beta))
                model_used = "numpy_linear_regression"
            except Exception:
                pred = float(np.nan)
                model_used = "numpy_linear_regression_failed"

        if not np.isfinite(pred):
            return None

        # 简单置信度：用历史残差的标准差估计（越小越有信心）
        try:
            y_hat = (X1 @ beta) if not _HAS_SKLEARN else lr.predict(X)
            resid = y - y_hat
            sigma = float(np.nanstd(resid))
            confidence = max(0.0, 1.0 - min(1.0, sigma * 10))  # 经验缩放
        except Exception:
            confidence = 0.3

        return {
            'forecast_change_pct': round(pred * 100, 3),
            'confidence': round(float(confidence), 3),
            'model_used': model_used
        }

    def run(self) -> Dict[str, Any]:
        if self.df is None or len(self.df) < 60:
            return {'code': self.code, 'name': self.name or self.code, 'status': 'failed', 'error': '数据不足(<60) 或读取失败'}

        feat_df = self._basic_features()
        if feat_df is None:
            return {'code': self.code, 'name': self.name or self.code, 'status': 'failed', 'error': '特征构建失败或样本太少'}

        # Kalman 平滑（用于辅助特征/信号，不强依赖）
        close = feat_df['close'].to_numpy(dtype=float)
        smooth_close = self._kalman_smooth(close)
        if smooth_close is not None and len(smooth_close) == len(feat_df):
            feat_df['smooth_close'] = smooth_close
            feat_df['smooth_ret'] = pd.Series(smooth_close).pct_change().to_numpy()
            feat_df['smooth_ret'].fillna(0.0, inplace=True)

        # HMM 状态（可选）
        regime = self._hmm_regime(feat_df['ret1'].to_numpy(dtype=float))

        # 回归预测
        pred = self._regression_forecast(feat_df)
        if pred is None:
            # 最终降级：用最近5日收益均值
            recent = feat_df['ret1'].tail(5).to_numpy(dtype=float)
            recent = recent[np.isfinite(recent)]
            if len(recent) == 0:
                return {'code': self.code, 'name': self.name or self.code, 'status': 'failed', 'error': '预测失败'}
            forecast_pct = float(np.mean(recent) * 100)
            pred = {'forecast_change_pct': round(forecast_pct, 3), 'confidence': 0.2, 'model_used': 'fallback_mean_return'}

        out = {
            'code': self.code,
            'name': self.name or self.code,
            'status': 'ok',
            **pred
        }
        if regime is not None:
            out['regime'] = regime
        return out

# ------------------------------------------------------------------------------
# 并行处理
# ------------------------------------------------------------------------------
def process_task(csv_path: str) -> Dict[str, Any]:
    try:
        return AdvancedForecaster(csv_path).run()
    except Exception as e:
        code = Path(csv_path).stem
        return {"code": code, "name": code, "status": "failed", "error": str(e)}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='高级股票预测分析')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / 'data' / 'day'
    stock_files = [str(p) for p in data_dir.rglob("*.csv")]
    if args.limit:
        stock_files = stock_files[:int(args.limit)]

    log(f"开始预测 {len(stock_files)} 只股票...")
    if not stock_files:
        log("未找到任何CSV文件。", level="ERROR")
        return

    with Pool(max(1, cpu_count() - 1)) as pool:
        results = list(pool.imap_unordered(process_task, stock_files))

    successful = [r for r in results if r.get('status') == 'ok']
    successful.sort(key=lambda x: x.get('forecast_change_pct', 0.0), reverse=True)

    out_path = PROJECT_ROOT / 'web' / 'client' / 'src' / 'data' / 'forecast_summary.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dependencies": {
            "filterpy": _HAS_FILTERPY,
            "hmmlearn": _HAS_HMMLEARN,
            "sklearn": _HAS_SKLEARN
        },
        "predictions": successful
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log(f"预测完成，结果已保存: {out_path}")

if __name__ == "__main__":
    main()
