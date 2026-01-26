#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a7_advanced_forecast.py
功能描述: 高级股票预测分析模块
使用方法: python3 a7_advanced_forecast.py --limit 10
依赖库: pandas, numpy, filterpy, hmmlearn, scikit-learn
安装命令: pip install pandas numpy filterpy hmmlearn scikit-learn
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
from sklearn.ensemble import RandomForestRegressor

# 忽略警告信息
warnings.filterwarnings('ignore')

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

class AdvancedForecaster:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.code = Path(csv_path).stem
        self.df = self._load_data()
        
    def _load_data(self):
        try:
            # 统一使用 gbk/utf-8 尝试读取
            df = None
            for enc in ['utf-8', 'gbk', 'utf-8-sig']:
                try:
                    df = pd.read_csv(self.csv_path, encoding=enc)
                    break
                except: continue
            
            if df is None: raise ValueError("无法读取CSV文件")
            
            # 统一列名
            column_map = {'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'}
            df.rename(columns={c: column_map.get(c, c) for c in df.columns}, inplace=True)
            
            # 解析日期
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date', 'close'], inplace=True)
            df = df.sort_values('date').reset_index(drop=True)
            
            if len(df) < 20: raise ValueError("数据不足20条")
            return df
        except Exception as e:
            raise Exception(f"加载失败: {e}")

    def kalman_filter_smooth(self):
        prices = self.df['close'].astype(float).values
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([[prices[0]], [0.]]); kf.F = np.array([[1., 1.], [0., 1.]])
        kf.H = np.array([[1., 0.]]); kf.P *= 1000.; kf.R = 5; kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])
        smoothed = []
        for z in prices:
            kf.predict(); kf.update(z)
            smoothed.append(float(kf.x[0, 0]))
        self.df['kalman_price'] = smoothed

    def particle_filter_predict(self, n_particles=200):
        prices = self.df['close'].astype(float).values
        particles = np.random.normal(prices[0], 1.0, size=(n_particles, 1))
        weights = np.ones(n_particles) / n_particles
        predicted = []
        for z in prices:
            particles = particles + np.random.normal(0, 0.5, size=particles.shape)
            distance = np.abs(particles[:, 0] - z)
            weights *= 1. / (distance + 1.e-7); weights += 1.e-300; weights /= sum(weights)
            if 1. / sum(np.square(weights)) < n_particles / 2:
                indexes = systematic_resample(weights)
                particles = particles[indexes]; weights.fill(1.0 / n_particles)
            predicted.append(float(np.average(particles.flatten(), weights=weights)))
        self.df['particle_price'] = predicted

    def hmm_state_analysis(self):
        returns = self.df['close'].pct_change().dropna().values.reshape(-1, 1)
        try:
            model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
            model.fit(returns)
            states = model.predict(returns)
            self.df['market_state'] = np.insert(states, 0, states[0])
        except:
            self.df['market_state'] = 0

    def ensemble_predict(self):
        df_clean = self.df.dropna().copy()
        if len(df_clean) < 15: return self.df['close'].iloc[-1]
        features = ['open', 'high', 'low', 'close', 'volume', 'kalman_price', 'particle_price', 'market_state']
        X = df_clean[features]; y = df_clean['close'].shift(-1).fillna(df_clean['close'])
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        rf.fit(X[:-1], y[:-1])
        return float(rf.predict(X.iloc[[-1]])[0])

    def run(self):
        try:
            self.kalman_filter_smooth()
            self.particle_filter_predict()
            self.hmm_state_analysis()
            pred = self.ensemble_predict()
            latest = self.df.iloc[-1]
            curr = float(latest['close'])
            change = ((pred - curr) / curr * 100)
            
            # 计算预测日期
            next_date = latest['date'] + timedelta(days=1)
            while next_date.weekday() >= 5: next_date += timedelta(days=1)
            
            return {
                "code": self.code, "name": str(latest.get('name', 'Unknown')),
                "latest_close": round(curr, 2), "forecast_price": round(pred, 2),
                "forecast_change_pct": round(change, 2),
                "forecast_date": next_date.strftime('%Y-%m-%d')
            }
        except Exception as e:
            return {"code": self.code, "error": str(e), "status": "failed"}

def process_task(csv_path):
    try:
        return AdvancedForecaster(csv_path).run()
    except Exception as e:
        code = Path(csv_path).stem
        return {"code": code, "error": str(e), "status": "failed"}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='高级股票预测分析')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'day')
    stock_files = []
    for m in ['sh', 'sz', 'bj']:
        p = os.path.join(data_dir, m)
        if os.path.exists(p):
            stock_files.extend([os.path.join(p, f) for f in os.listdir(p) if f.endswith('.csv')])
    
    if args.limit: stock_files = stock_files[:args.limit]
    
    print(f"开始预测 {len(stock_files)} 只股票...")
    with Pool(max(1, cpu_count()-1)) as pool:
        results = list(pool.imap_unordered(process_task, stock_files))
    
    successful = [r for r in results if 'error' not in r]
    successful.sort(key=lambda x: x.get('forecast_change_pct', 0), reverse=True)
    
    out_path = os.path.join(PROJECT_ROOT, 'web', 'client', 'src', 'data', 'forecast_summary.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "predictions": successful}, f, ensure_ascii=False, indent=2)
    print(f"预测完成，结果已保存。")

if __name__ == "__main__":
    main()
