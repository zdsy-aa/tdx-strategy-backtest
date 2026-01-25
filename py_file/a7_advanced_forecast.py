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
    """高级预测器类
    用于进行股票预测分析，使用多种高级模型，如卡尔曼滤波、粒子滤波、隐马尔可夫模型和随机森林集成方法进行预测。
    """
    
    def __init__(self, csv_path):
        """
        初始化预测器类
        
        参数：
            csv_path (str): 股票数据 CSV 文件路径
        """
        self.csv_path = csv_path
        self.code = Path(csv_path).stem  # 从文件名提取股票代码
        self.df = self._load_data()  # 加载并清洗数据
        self.scaler = StandardScaler()  # 初始化标准化工具
        
    def _load_data(self):
        """
        加载股票数据
        
        返回：
            pd.DataFrame: 加载并清理后的数据
        """
        try:
            # 根据提供的中文列名读取数据文件
            cols = ['名称', '日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
            df = pd.read_csv(self.csv_path, names=cols, header=None, encoding='utf-8-sig')
            
            # 检查是否需要跳过第一行表头
            if df.iloc[0]['日期'] == '日期':
                df = df.iloc[1:].reset_index(drop=True)
            
            # 转换数据类型
            df['日期'] = pd.to_datetime(df['日期'])
            for col in ['开盘', '最高', '最低', '收盘', '成交量']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 清理缺失值，特别是缺少收盘价的数据
            df.dropna(subset=['收盘'], inplace=True)
            
            # 关键修复：处理非正数值（0 或 负数），这些数值会导致 pct_change 产生 Inf 或异常值
            # 股票价格理论上应为正数，这里将小于等于 0.01 的价格视为异常并处理
            for col in ['开盘', '最高', '最低', '收盘']:
                df[col] = df[col].apply(lambda x: x if x > 0.01 else np.nan)
            
            # 再次清理因处理异常值产生的 NaN
            df.dropna(subset=['收盘'], inplace=True)
            
            # 填充其他列的 NaN（如果有）
            df = df.ffill().bfill()
            
            df = df.sort_values('日期')  # 按日期排序
            
            # 如果数据量少于10条记录，抛出异常
            if len(df) < 10:
                raise ValueError(f"数据不足，需要至少10条记录，当前仅有{len(df)}条")
            
            return df
        except Exception as e:
            raise Exception(f"加载数据失败: {str(e)}")

    def kalman_filter_smooth(self):
        """
        使用卡尔曼滤波器对价格曲线进行平滑，减少噪声
        """
        # print("应用卡尔曼滤波器平滑价格曲线...")
        prices = self.df['收盘'].astype(float).values
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([[prices[0]], [0.]])  # 初始状态：价格和速度
        kf.F = np.array([[1., 1.], [0., 1.]])  # 状态转移矩阵
        kf.H = np.array([[1., 0.]])  # 观测矩阵
        kf.P *= 1000.  # 初始协方差
        kf.R = 5  # 测量噪声
        kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])  # 过程噪声

        smoothed_prices = []
        for z in prices:
            kf.predict()  # 预测
            kf.update(z)  # 更新
            smoothed_prices.append(float(kf.x[0, 0]))
        
        self.df['kalman_price'] = smoothed_prices
        return smoothed_prices

    def particle_filter_predict(self, n_particles=500):
        """
        使用粒子滤波器处理非高斯分布，预测下一时刻价格
        """
        # print("应用粒子滤波器进行价格预测...")
        prices = self.df['收盘'].astype(float).values
        particles = np.random.normal(prices[0], 1.0, size=(n_particles, 1))  # 初始化粒子
        weights = np.ones(n_particles) / n_particles  # 初始化权重
        
        def fx(p, dt, std):
            return p + np.random.normal(0, std, size=p.shape)

        predicted_val = []
        for z in prices:
            particles = fx(particles, 1, 0.5)  # 预测
            distance = np.abs(particles[:, 0] - z)  # 计算误差
            weights *= 1. / (distance + 1.e-7)  # 更新权重
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
        使用隐马尔可夫模型捕捉市场状态（牛市、熊市、震荡）
        """
        # print("应用隐马尔可夫模型分析市场状态...")
        # 关键修复：确保 pct_change 不会产生 Inf 或 NaN
        returns = self.df['收盘'].pct_change().replace([np.inf, -np.inf], np.nan).dropna().values.reshape(-1, 1)
        
        if len(returns) < 20:
            self.df['market_state'] = 2  # 默认震荡状态
            return self.df['market_state'].values
            
        try:
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)
            model.fit(returns)
            states = model.predict(returns)
            
            # 补齐第一行状态
            full_states = np.insert(states, 0, states[0])
            self.df['market_state'] = full_states
            return full_states
        except Exception:
            self.df['market_state'] = 2
            return self.df['market_state'].values

    def ensemble_predict(self):
        """
        使用集成方法（随机森林）预测次日收盘价
        """
        # print("使用随机森林进行集成预测...")
        self.df['ma5'] = self.df['收盘'].rolling(5).mean()  # 5日均线
        self.df['vol_ma5'] = self.df['成交量'].rolling(5).mean()  # 5日成交量均线
        
        # 关键修复：处理特征中的异常值
        df_clean = self.df.dropna().copy()
        
        # 确保没有 Inf 或过大的数值
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df_clean) < 10:
            return self.df['收盘'].iloc[-1]
        
        features = ['开盘', '最高', '最低', '收盘', '成交量', 'kalman_price', 'particle_price', 'market_state', 'ma5', 'vol_ma5']
        X = df_clean[features]
        y = df_clean['收盘'].shift(-1).fillna(df_clean['收盘'])  # 预测次日收盘
        
        # 训练随机森林回归模型
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X[:-1], y[:-1])
        
        # 预测最后一行（即次日预测）
        last_features = X.iloc[[-1]]
        next_day_pred = rf.predict(last_features)[0]
        
        return float(next_day_pred)

    def run_all(self):
        """
        运行所有预测分析，结合卡尔曼滤波、粒子滤波、HMM 和 随机森林的结果
        """
        # print(f"开始对股票 {self.code} 进行预测分析...")
        try:
            # 应用卡尔曼滤波、粒子滤波、HMM分析和集成预测
            self.kalman_filter_smooth()
            self.particle_filter_predict()
            self.hmm_state_analysis()
            next_day_pred = self.ensemble_predict()
            
            # 获取最新收盘价和预测涨幅
            last_row = self.df.iloc[-1]
            current_close = float(last_row['收盘'])
            
            # 关键修复：防止除零错误
            if current_close <= 0.01:
                raise ValueError(f"当前收盘价异常: {current_close}")
                
            forecast_change_pct = ((next_day_pred - current_close) / current_close * 100)
            
            # 限制涨幅在合理范围内（例如 -20% 到 20%），防止数值爆炸
            forecast_change_pct = max(min(forecast_change_pct, 20.0), -20.0)
            
            # 计算预测置信度
            confidence = self._calculate_confidence(next_day_pred, current_close)
            
            # 计算预测日期（次日）
            last_date = last_row['日期']
            next_date = last_date + timedelta(days=1)
            if next_date.weekday() >= 5:  # 如果是周末，跳过
                next_date += timedelta(days=2)

            result = {
                "code": self.code,
                "name": str(last_row['名称']),
                "latest_close": current_close,
                "kalman_price": float(last_row['kalman_price']),
                "particle_price": float(last_row['particle_price']),
                "market_state": int(last_row['market_state']),
                "ensemble_forecast": float(next_day_pred),
                "forecast_change_pct": round(float(forecast_change_pct), 2),
                "confidence": float(confidence),
                "analysis_date": str(last_date.date()),
                "forecast_date": str(next_date.date())
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
        根据预测和当前价格计算预测置信度
        """
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
    """
    处理单个股票的预测分析
    """
    try:
        # print(f"处理股票文件: {csv_path}")
        forecaster = AdvancedForecaster(csv_path)
        return forecaster.run_all()
    except Exception as e:
        code = Path(csv_path).stem
        return {"code": code, "error": str(e), "status": "failed"}


def get_project_root():
    """
    获取项目根目录路径
    """
    current_file = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(script_dir)
    return project_root


def get_all_stock_files():
    """
    获取所有市场的股票数据文件，兼容 Windows 和 Linux 系统
    """
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
    """
    主函数，执行股票预测分析任务
    """
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
    print(f"使用 {num_workers} 个进程进行处理")
    
    results = []
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_stock, stock_files), 1):
            results.append(result)
            if i % 100 == 0:
                print(f"已处理 {i}/{len(stock_files)} 只股票")
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"\n预测完成！成功: {len(successful)}, 失败: {len(failed)}")
    
    if failed:
        print("失败的股票示例:")
        for f in failed[:10]:
            print(f"  {f.get('code', '未知')}: {f.get('error', '未知错误')}")
    
    # 按预测涨幅排序
    successful_sorted = sorted(successful, key=lambda x: x.get('forecast_change_pct', 0), reverse=True)
    
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_stocks": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "all_predictions": successful_sorted
    }
    
    web_data_dir = os.path.join(get_project_root(), 'web', 'client', 'src', 'data')
    os.makedirs(web_data_dir, exist_ok=True)
    
    summary_path = os.path.join(web_data_dir, "forecast_summary.json")
    # 明确指定 newline='\n' 以确保生成 Unix (LF) 格式的 JSON 文件
    with open(summary_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    details_dir = os.path.join(web_data_dir, "forecast_details")
    os.makedirs(details_dir, exist_ok=True)
    
    for result in successful:
        code = result.get('code')
        if code:
            detail_path = os.path.join(details_dir, f"{code}.json")
            # 明确指定 newline='\n' 以确保生成 Unix (LF) 格式的 JSON 文件
            with open(detail_path, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"预测结果已保存至: {web_data_dir}")


if __name__ == "__main__":
    main()
