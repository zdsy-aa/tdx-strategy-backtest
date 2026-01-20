"""
高级股票预测脚本 (Advanced Stock Forecast)
===========================================

功能说明：
    使用多种高级机器学习和信号处理技术对股票进行预测分析：
    1. 卡尔曼滤波 (Kalman Filter) - 平滑价格曲线，减少噪声
    2. 粒子滤波 (Particle Filter) - 处理非高斯分布，预测下一时刻价格
    3. 隐马尔可夫模型 (HMM) - 捕捉市场状态（牛市、熊市、震荡）
    4. 随机森林集成 (Random Forest Ensemble) - 结合多种特征进行回归预测

生成的数据文件：
    - forecast_summary.json: 预测汇总数据（所有股票的最新预测）
    - forecast_details/<code>.json: 每只股票的详细预测数据

使用方法：
    python3 a7_advanced_forecast.py [--limit N] [--market MARKET]
    
参数说明：
    --limit N: 限制处理的股票数量（默认处理所有）
    --market MARKET: 只处理指定市场（sh/sz/bj，默认处理所有）

输出数据结构：
    forecast_summary.json:
    {
        "generated_at": "2026-01-19 10:30:00",
        "total_stocks": 100,
        "successful": 95,
        "failed": 5,
        "top_predictions": [
            {
                "code": "000001",
                "name": "平安银行",
                "market": "sz",
                "latest_close": 10.50,
                "kalman_price": 10.48,
                "particle_price": 10.52,
                "market_state": 1,
                "ensemble_forecast": 10.65,
                "forecast_change_pct": 1.43,
                "confidence": 0.85
            }
        ]
    }
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
from filterpy.kalman import KalmanFilter
from filterpy.monte_carlo import systematic_resample
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')


class AdvancedForecaster:
    """高级预测器类"""
    
    def __init__(self, csv_path):
        """
        初始化预测器
        
        参数：
            csv_path (str): 股票数据 CSV 文件路径
        """
        self.csv_path = csv_path
        self.code = Path(csv_path).stem  # 从文件名提取股票代码
        self.df = self._load_data()
        self.scaler = StandardScaler()
        
    def _load_data(self):
        """
        加载股票数据
        
        返回：
            pd.DataFrame: 加载并清理后的数据
        """
        try:
            # 根据 backtest 数据格式读取：名称,日期,开盘,最高,最低,收盘,成交量,成交额...
            cols = ['name', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'extra1', 'extra2', 'extra3', 'extra4']
            df = pd.read_csv(self.csv_path, names=cols, header=None, encoding='utf-8-sig')
            
            # 检查第一行是否是表头
            if df.iloc[0]['date'] == '日期':
                df = df.iloc[1:].reset_index(drop=True)
            
            # 转换数据类型
            df['date'] = pd.to_datetime(df['date'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 清理缺失值
            df.dropna(subset=['close'], inplace=True)
            df = df.sort_values('date')
            
            if len(df) < 10:
                raise ValueError(f"数据不足，需要至少10条记录，当前仅有{len(df)}条")
            
            return df
        except Exception as e:
            raise Exception(f"加载数据失败: {str(e)}")

    def kalman_filter_smooth(self):
        """
        卡尔曼滤波：平滑价格曲线，减少噪声
        
        返回：
            list: 平滑后的价格列表
        """
        prices = self.df['close'].astype(float).values
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([[prices[0]], [0.]])       # 初始状态 [价格, 速度]
        kf.F = np.array([[1., 1.], [0., 1.]])      # 状态转移矩阵
        kf.H = np.array([[1., 0.]])                # 观测矩阵
        kf.P *= 1000.                              # 初始协方差
        kf.R = 5                                   # 测量噪声
        kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])  # 过程噪声

        smoothed_prices = []
        for z in prices:
            kf.predict()
            kf.update(z)
            smoothed_prices.append(float(kf.x[0, 0]))
        
        self.df['kalman_price'] = smoothed_prices
        return smoothed_prices

    def particle_filter_predict(self, n_particles=500):
        """
        粒子滤波：处理非高斯分布，预测下一时刻价格
        
        参数：
            n_particles (int): 粒子数量
            
        返回：
            list: 预测的价格列表
        """
        prices = self.df['close'].astype(float).values
        particles = np.random.normal(prices[0], 1.0, size=(n_particles, 1))
        weights = np.ones(n_particles) / n_particles
        
        def fx(p, dt, std):
            return p + np.random.normal(0, std, size=p.shape)

        predicted_val = []
        for z in prices:
            # 预测
            particles = fx(particles, 1, 0.5)
            # 更新权重
            distance = np.abs(particles[:, 0] - z)
            weights *= 1. / (distance + 1.e-7)
            weights += 1.e-300
            weights /= sum(weights)
            
            # 重采样
            if 1. / sum(np.square(weights)) < n_particles / 2:
                indexes = systematic_resample(weights)
                particles = particles[indexes]
                weights.fill(1.0 / n_particles)
            
            predicted_val.append(float(np.average(particles.flatten(), weights=weights)))
            
        self.df['particle_price'] = predicted_val
        return predicted_val

    def hmm_state_analysis(self, n_states=3):
        """
        隐马尔可夫模型：捕捉市场状态（牛市、熊市、震荡）
        
        参数：
            n_states (int): 市场状态数量
            
        返回：
            np.ndarray: 市场状态数组
        """
        returns = self.df['close'].pct_change().dropna().values.reshape(-1, 1)
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)
        model.fit(returns)
        states = model.predict(returns)
        
        # 补齐第一行
        full_states = np.insert(states, 0, states[0])
        self.df['market_state'] = full_states
        return full_states

    def ensemble_predict(self):
        """
        集成方法：结合多种特征进行回归预测
        
        返回：
            float: 下一日预测收盘价
        """
        # 准备特征
        self.df['ma5'] = self.df['close'].rolling(5).mean()
        self.df['vol_ma5'] = self.df['volume'].rolling(5).mean()
        df_clean = self.df.dropna()
        
        if len(df_clean) < 10:
            return self.df['close'].iloc[-1]
        
        features = ['open', 'high', 'low', 'close', 'volume', 'kalman_price', 'particle_price', 'market_state', 'ma5', 'vol_ma5']
        X = df_clean[features]
        y = df_clean['close'].shift(-1).fillna(df_clean['close'])
        
        # 训练随机森林作为集成模型
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X[:-1], y[:-1])
        
        self.df['ensemble_forecast'] = np.nan
        self.df.loc[df_clean.index, 'ensemble_forecast'] = rf.predict(X)
        
        return float(self.df['ensemble_forecast'].iloc[-1])

    def run_all(self):
        """
        运行所有预测分析
        
        返回：
            dict: 预测结果汇总
        """
        try:
            self.kalman_filter_smooth()
            self.particle_filter_predict()
            self.hmm_state_analysis()
            last_pred = self.ensemble_predict()
            
            # 提取最后一行数据
            last_row = self.df.iloc[-1]
            current_close = float(last_row['close'])
            forecast_change_pct = ((last_pred - current_close) / current_close * 100) if current_close != 0 else 0
            
            # 计算置信度（基于多个指标的一致性）
            confidence = self._calculate_confidence(last_pred, current_close)
            
            result = {
                "code": self.code,
                "name": str(last_row['name']),
                "latest_close": current_close,
                "kalman_price": float(last_row['kalman_price']),
                "particle_price": float(last_row['particle_price']),
                "market_state": int(last_row['market_state']),
                "ensemble_forecast": last_pred,
                "forecast_change_pct": round(forecast_change_pct, 2),
                "confidence": confidence,
                "analysis_date": str(last_row['date'].date())
            }
            
            return result
        except Exception as e:
            return {
                "code": self.code,
                "error": str(e),
                "status": "failed"
            }

    def _calculate_confidence(self, forecast, current):
        """
        计算预测置信度
        
        参数：
            forecast (float): 预测价格
            current (float): 当前价格
            
        返回：
            float: 置信度（0-1）
        """
        # 简单的置信度计算：基于预测与当前价格的偏离程度
        change_pct = abs((forecast - current) / current) if current != 0 else 0
        
        # 偏离度越小，置信度越高
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
    """
    处理单只股票的预测
    
    参数：
        csv_path (str): 股票数据文件路径
        
    返回：
        dict: 预测结果
    """
    try:
        forecaster = AdvancedForecaster(csv_path)
        return forecaster.run_all()
    except Exception as e:
        code = Path(csv_path).stem
        return {
            "code": code,
            "error": str(e),
            "status": "failed"
        }


def get_all_stock_files(data_dir="data/day"):
    """
    获取所有股票数据文件
    
    参数：
        data_dir (str): 数据目录路径
        
    返回：
        list: 股票文件路径列表
    """
    stock_files = []
    for market in ['sh', 'sz', 'bj']:
        market_dir = os.path.join(data_dir, market)
        if os.path.exists(market_dir):
            for file in os.listdir(market_dir):
                if file.endswith('.csv'):
                    stock_files.append(os.path.join(market_dir, file))
    return stock_files


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='高级股票预测分析')
    parser.add_argument('--limit', type=int, default=None, help='限制处理的股票数量')
    parser.add_argument('--market', type=str, default=None, help='只处理指定市场 (sh/sz/bj)')
    args = parser.parse_args()
    
    # 获取所有股票文件
    stock_files = get_all_stock_files()
    
    # 按市场过滤
    if args.market:
        stock_files = [f for f in stock_files if f"/{args.market}/" in f]
    
    # 限制数量
    if args.limit:
        stock_files = stock_files[:args.limit]
    
    print(f"开始处理 {len(stock_files)} 只股票的预测分析...")
    
    # 并行处理
    num_workers = max(1, cpu_count() - 1)
    results = []
    
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_stock, stock_files), 1):
            results.append(result)
            if i % 100 == 0:
                print(f"已处理 {i}/{len(stock_files)} 只股票")
    
    # 统计结果
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"\n预测完成！成功: {len(successful)}, 失败: {len(failed)}")
    
    # 按 ensemble_forecast 排序，获取前 100 只
    successful_sorted = sorted(successful, key=lambda x: x.get('ensemble_forecast', 0), reverse=True)[:100]
    
    # 生成汇总数据
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_stocks": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "top_predictions": successful_sorted
    }
    
    # 保存到 Web 数据目录
    web_data_dir = "web/client/src/data"
    os.makedirs(web_data_dir, exist_ok=True)
    
    # 保存汇总数据
    summary_path = os.path.join(web_data_dir, "forecast_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"汇总数据已保存至: {summary_path}")
    
    # 保存详细数据
    details_dir = os.path.join(web_data_dir, "forecast_details")
    os.makedirs(details_dir, exist_ok=True)
    
    for result in successful:
        code = result.get('code')
        detail_path = os.path.join(details_dir, f"{code}.json")
        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"详细数据已保存至: {details_dir}")


if __name__ == "__main__":
    main()

print("a7_advanced_forecast.py 脚本执行完毕")
