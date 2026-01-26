#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
高级股票预测分析模块 (a7_advanced_forecast.py)
================================================================================

功能说明:
    本脚本利用多种高级算法（卡尔曼滤波、粒子滤波、隐马尔可夫模型及随机森林集成）
    对股票的下一交易日收盘价进行预测。

主要功能:
    1. 卡尔曼滤波 (Kalman Filter)：对价格曲线进行平滑处理。
    2. 粒子滤波 (Particle Filter)：模拟价格走势并预测下一时刻。
    3. 隐马尔可夫模型 (HMM)：捕捉市场状态（上涨、下跌、震荡）。
    4. 随机森林集成 (Random Forest)：综合上述特征进行最终的价格预测。
    5. 严格计算下一交易日：自动跳过周末，确保预测日期准确。
    6. 生成精简版 JSON：供前端 ForecastDashboard 页面展示。

使用方法:
    python a7_advanced_forecast.py [options]
    参数:
      --limit:  限制处理的股票数量 (测试用)
      --market: 只处理指定市场 (sh/sz/bj)

依赖库:
    pandas, numpy, filterpy, hmmlearn, scikit-learn

安装命令:
    pip install pandas numpy filterpy hmmlearn scikit-learn

================================================================================
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from multiprocessing import Pool, cpu_count
from filterpy.kalman import KalmanFilter
from filterpy.monte_carlo import systematic_resample
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# 忽略警告信息
warnings.filterwarnings('ignore')


class AdvancedForecaster:
    """高级预测器类"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.code = Path(csv_path).stem
        self.df = self._load_data()
        self.scaler = StandardScaler()
        
    def _load_data(self):
        try:
            cols = ['名称', '日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
            df = pd.read_csv(self.csv_path, names=cols, header=None, encoding='utf-8-sig')
            
            if df.iloc[0]['日期'] == '日期':
                df = df.iloc[1:].reset_index(drop=True)
            
            df['日期'] = pd.to_datetime(df['日期'])
            for col in ['开盘', '最高', '最低', '收盘', '成交量']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(subset=['收盘'], inplace=True)
            
            for col in ['开盘', '最高', '最低', '收盘']:
                df[col] = df[col].apply(lambda x: x if x > 0.01 else np.nan)
            
            df.dropna(subset=['收盘'], inplace=True)
            df = df.ffill().bfill()
            df = df.sort_values('日期')
            
            if len(df) < 10:
                raise ValueError(f"数据不足，需要至少10条记录")
            
            return df
        except Exception as e:
            raise Exception(f"加载数据失败: {str(e)}")

    def kalman_filter_smooth(self):
        prices = self.df['收盘'].astype(float).values
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([[prices[0]], [0.]])
        kf.F = np.array([[1., 1.], [0., 1.]])
        kf.H = np.array([[1., 0.]])
        kf.P *= 1000.
        kf.R = 5
        kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])

        smoothed_prices = []
        for z in prices:
            kf.predict()
            kf.update(z)
            smoothed_prices.append(float(kf.x[0, 0]))
        
        self.df['kalman_price'] = smoothed_prices
        return smoothed_prices

    def particle_filter_predict(self, n_particles=500):
        prices = self.df['收盘'].astype(float).values
        particles = np.random.normal(prices[0], 1.0, size=(n_particles, 1))
        weights = np.ones(n_particles) / n_particles
        
        def fx(p, dt, std):
            return p + np.random.normal(0, std, size=p.shape)

        predicted_val = []
        for z in prices:
            particles = fx(particles, 1, 0.5)
            distance = np.abs(particles[:, 0] - z)
            weights *= 1. / (distance + 1.e-7)
            weights += 1.e-300
            weights /= sum(weights)
            
            if 1. / sum(np.square(weights)) < n_particles / 2:
                indexes = systematic_resample(weights)
                particles = particles[indexes]
                weights.fill(1.0 / n_particles)
            
            predicted_val.append(float(np.average(particles.flatten(), weights=weights)))
            
        self.df['particle_price'] = predicted_val
        return predicted_val

    def hmm_state_analysis(self, n_states=3):
        returns = self.df['收盘'].pct_change().replace([np.inf, -np.inf], np.nan).dropna().values.reshape(-1, 1)
        
        if len(returns) < 20:
            self.df['market_state'] = 2
            return self.df['market_state'].values
            
        try:
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)
            model.fit(returns)
            states = model.predict(returns)
            full_states = np.insert(states, 0, states[0])
            self.df['market_state'] = full_states
            return full_states
        except Exception:
            self.df['market_state'] = 2
            return self.df['market_state'].values

    def ensemble_predict(self):
        self.df['ma5'] = self.df['收盘'].rolling(5).mean()
        self.df['vol_ma5'] = self.df['成交量'].rolling(5).mean()
        
        df_clean = self.df.dropna().copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df_clean) < 10:
            return self.df['收盘'].iloc[-1]
        
        features = ['开盘', '最高', '最低', '收盘', '成交量', 'kalman_price', 'particle_price', 'market_state', 'ma5', 'vol_ma5']
        X = df_clean[features]
        y = df_clean['收盘'].shift(-1).fillna(df_clean['收盘'])
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X[:-1], y[:-1])
        
        last_features = X.iloc[[-1]]
        next_day_pred = rf.predict(last_features)[0]
        
        return float(next_day_pred)

    def run_all(self):
        try:
            self.kalman_filter_smooth()
            self.particle_filter_predict()
            self.hmm_state_analysis()
            next_day_pred = self.ensemble_predict()
            
            last_row = self.df.iloc[-1]
            current_close = float(last_row['收盘'])
            
            if current_close <= 0.01:
                raise ValueError(f"当前收盘价异常: {current_close}")
                
            forecast_change_pct = ((next_day_pred - current_close) / current_close * 100)
            forecast_change_pct = max(min(forecast_change_pct, 20.0), -20.0)
            confidence = self._calculate_confidence(next_day_pred, current_close)
            
            last_date = last_row['日期']
            
            # 严格计算下一交易日
            next_date = last_date + timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)

            result = {
                "code": self.code,
                "name": str(last_row['名称']),
                "latest_close": current_close,
                "forecast_price": round(float(next_day_pred), 2),
                "forecast_change_pct": round(float(forecast_change_pct), 2),
                "confidence": float(confidence),
                "analysis_date": last_date.strftime('%Y-%m-%d'),
                "forecast_date": next_date.strftime('%Y-%m-%d')
            }
            
            return result
        except Exception as e:
            return {
                "code": self.code,
                "error": str(e),
                "status": "failed"
            }
    
    def _calculate_confidence(self, forecast, current):
        if current == 0: return 0.5
        change_pct = abs((forecast - current) / current)
        if change_pct < 0.01:
            confidence = 0.95
        elif change_pct < 0.05:
            confidence = 0.85
        elif change_pct < 0.1:
            confidence = 0.75
        else:
            confidence = 0.65
        return round(confidence, 2)


def process_single_stock(csv_path):
    try:
        forecaster = AdvancedForecaster(csv_path)
        return forecaster.run_all()
    except Exception as e:
        code = Path(csv_path).stem
        return {"code": code, "error": str(e), "status": "failed"}


def get_project_root():
    current_file = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(script_dir)
    return project_root


def get_all_stock_files():
    stock_files = []
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data', 'day')
    
    if not os.path.exists(data_dir):
        return []
    
    for market in ['sh', 'sz', 'bj']:
        market_dir = os.path.join(data_dir, market)
        if os.path.exists(market_dir) and os.path.isdir(market_dir):
            csv_files = [f for f in os.listdir(market_dir) if f.endswith('.csv')]
            for csv_file in csv_files:
                stock_files.append(os.path.join(market_dir, csv_file))
    
    return stock_files


def main():
    import argparse
    parser = argparse.ArgumentParser(description='高级股票预测分析')
    parser.add_argument('--limit', type=int, default=None, help='限制处理的股票数量')
    parser.add_argument('--market', type=str, default=None, help='只处理指定市场 (sh/sz/bj)')
    args = parser.parse_args()
    
    stock_files = get_all_stock_files()
    
    if not stock_files:
        print("错误: 未找到任何股票数据文件!")
        sys.exit(1)
    
    if args.market:
        stock_files = [f for f in stock_files if os.path.join(args.market, '') in f.replace('\\', '/')]
    
    if args.limit:
        stock_files = stock_files[:args.limit]
    
    print(f"开始处理 {len(stock_files)} 只股票的预测分析...")
    
    num_workers = max(1, cpu_count() - 1)
    
    results = []
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_stock, stock_files), 1):
            results.append(result)
            if i % 100 == 0:
                print(f"已处理 {i}/{len(stock_files)} 只股票")
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"\n预测完成！成功: {len(successful)}, 失败: {len(failed)}")
    
    # 按预测涨幅排序
    successful_sorted = sorted(successful, key=lambda x: x.get('forecast_change_pct', 0), reverse=True)
    
    # 精简版汇总：仅包含预测数据，不包含市场概览和排行
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predictions": successful_sorted
    }
    
    web_data_dir = os.path.join(get_project_root(), 'web', 'client', 'src', 'data')
    os.makedirs(web_data_dir, exist_ok=True)
    
    summary_path = os.path.join(web_data_dir, "forecast_summary.json")
    with open(summary_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    details_dir = os.path.join(web_data_dir, "forecast_details")
    os.makedirs(details_dir, exist_ok=True)
    
    for result in successful:
        code = result.get('code')
        if code:
            detail_path = os.path.join(details_dir, f"{code}.json")
            with open(detail_path, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"预测结果已保存至: {web_data_dir}")


if __name__ == "__main__":
    main()
