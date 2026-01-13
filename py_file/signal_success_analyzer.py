#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
信号成功率分析脚本 (signal_success_analyzer.py)
================================================================================

功能说明:
    分析当六脉神剑4红+、缠论5个买点、买卖点1/2买信号出现后，
    哪些技术指标或规则能够预测后续涨幅超过5%的情况。

分析维度:
    1. 信号触发后的涨幅统计
    2. 成功案例（涨幅>5%）的共同特征挖掘
    3. 多维度指标分析（MACD, KDJ, BOLL, RSI, DMI, DMA, SAR, BBI, OBV等）
    4. 道式理论、威克夫理论相关特征

输出:
    - 控制台: 分析进度和关键统计
    - 文件: report/total/signal_success_analysis_report.md

作者: Manus AI
版本: 1.0.0
更新日期: 2026-01-13
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import (
    calculate_six_veins, calculate_buy_sell_points, calculate_chan_theory,
    REF, MA, EMA, SMA, HHV, LLV, CROSS, COUNT, ABS, MAX, IF
)

# ==============================================================================
# 配置常量
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'day')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report', 'total')

# 分析参数
TARGET_GAIN = 5.0  # 目标涨幅 5%
MAX_HOLD_DAYS = 20  # 最大持有天数
MIN_DATA_DAYS = 100  # 最少数据天数

# ==============================================================================
# 扩展指标计算函数
# ==============================================================================

def calculate_extended_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算扩展技术指标，用于特征分析
    
    包含: MACD, KDJ, BOLL, RSI, DMI, DMA, SAR, BBI, OBV, 
          道式理论特征, 威克夫理论特征
    """
    df = df.copy()
    C = df['close']
    H = df['high']
    L = df['low']
    O = df['open']
    V = df['volume']
    
    # =========================================================================
    # 1. MACD 指标 (已在六脉神剑中计算，这里提取详细值)
    # =========================================================================
    DIF = EMA(C, 12) - EMA(C, 26)
    DEA = EMA(DIF, 9)
    MACD = (DIF - DEA) * 2
    df['macd_dif'] = DIF
    df['macd_dea'] = DEA
    df['macd_hist'] = MACD
    df['macd_golden_cross'] = CROSS(DIF, DEA)  # MACD金叉
    df['macd_above_zero'] = DIF > 0  # DIF在零轴上方
    df['macd_hist_positive'] = MACD > 0  # MACD柱状图为正
    
    # =========================================================================
    # 2. KDJ 指标
    # =========================================================================
    RSV = (C - LLV(L, 9)) / (HHV(H, 9) - LLV(L, 9) + 0.0001) * 100
    K = SMA(RSV, 3, 1)
    D = SMA(K, 3, 1)
    J = 3 * K - 2 * D
    df['kdj_k'] = K
    df['kdj_d'] = D
    df['kdj_j'] = J
    df['kdj_golden_cross'] = CROSS(K, D)  # KDJ金叉
    df['kdj_oversold'] = J < 20  # 超卖区
    df['kdj_overbought'] = J > 80  # 超买区
    
    # =========================================================================
    # 3. BOLL 布林带
    # =========================================================================
    BOLL_MID = MA(C, 20)
    BOLL_STD = C.rolling(window=20).std()
    BOLL_UP = BOLL_MID + 2 * BOLL_STD
    BOLL_DN = BOLL_MID - 2 * BOLL_STD
    df['boll_mid'] = BOLL_MID
    df['boll_up'] = BOLL_UP
    df['boll_dn'] = BOLL_DN
    df['boll_width'] = (BOLL_UP - BOLL_DN) / BOLL_MID * 100  # 布林带宽度
    df['boll_position'] = (C - BOLL_DN) / (BOLL_UP - BOLL_DN + 0.0001)  # 价格在布林带中的位置
    df['boll_squeeze'] = df['boll_width'] < df['boll_width'].rolling(20).quantile(0.2)  # 布林带收窄
    
    # =========================================================================
    # 4. RSI 相对强弱指数
    # =========================================================================
    LC = REF(C, 1)
    RSI6 = SMA(MAX(C - LC, 0), 6, 1) / (SMA(ABS(C - LC), 6, 1) + 0.0001) * 100
    RSI12 = SMA(MAX(C - LC, 0), 12, 1) / (SMA(ABS(C - LC), 12, 1) + 0.0001) * 100
    RSI24 = SMA(MAX(C - LC, 0), 24, 1) / (SMA(ABS(C - LC), 24, 1) + 0.0001) * 100
    df['rsi6'] = RSI6
    df['rsi12'] = RSI12
    df['rsi24'] = RSI24
    df['rsi_oversold'] = RSI6 < 30  # RSI超卖
    df['rsi_overbought'] = RSI6 > 70  # RSI超买
    df['rsi_golden_cross'] = CROSS(RSI6, RSI12)  # RSI金叉
    
    # =========================================================================
    # 5. DMI 趋向指标
    # =========================================================================
    TR = pd.concat([H - L, ABS(H - REF(C, 1)), ABS(L - REF(C, 1))], axis=1).max(axis=1)
    HD = H - REF(H, 1)
    LD = REF(L, 1) - L
    DMP = IF((HD > 0) & (HD > LD), HD, pd.Series(0, index=df.index))
    DMM = IF((LD > 0) & (LD > HD), LD, pd.Series(0, index=df.index))
    
    PDI = SMA(DMP, 14, 1) / (SMA(TR, 14, 1) + 0.0001) * 100
    MDI = SMA(DMM, 14, 1) / (SMA(TR, 14, 1) + 0.0001) * 100
    ADX = SMA(ABS(PDI - MDI) / (PDI + MDI + 0.0001) * 100, 6, 1)
    
    df['dmi_pdi'] = PDI
    df['dmi_mdi'] = MDI
    df['dmi_adx'] = ADX
    df['dmi_trend_up'] = PDI > MDI  # 上升趋势
    df['dmi_strong_trend'] = ADX > 25  # 强趋势
    
    # =========================================================================
    # 6. DMA 平均差指标
    # =========================================================================
    DMA_DIF = MA(C, 10) - MA(C, 50)
    DMA_AMA = MA(DMA_DIF, 10)
    df['dma_dif'] = DMA_DIF
    df['dma_ama'] = DMA_AMA
    df['dma_golden_cross'] = CROSS(DMA_DIF, DMA_AMA)
    df['dma_above_zero'] = DMA_DIF > 0
    
    # =========================================================================
    # 7. SAR 抛物线指标 (简化版)
    # =========================================================================
    # 使用简化的SAR计算
    sar = pd.Series(index=df.index, dtype=float)
    af = 0.02
    af_max = 0.2
    ep = L.iloc[0]
    sar.iloc[0] = H.iloc[0]
    trend = -1  # -1下降, 1上升
    
    for i in range(1, len(df)):
        if trend == 1:
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            if L.iloc[i] < sar.iloc[i]:
                trend = -1
                sar.iloc[i] = ep
                ep = L.iloc[i]
                af = 0.02
            else:
                if H.iloc[i] > ep:
                    ep = H.iloc[i]
                    af = min(af + 0.02, af_max)
        else:
            sar.iloc[i] = sar.iloc[i-1] - af * (sar.iloc[i-1] - ep)
            if H.iloc[i] > sar.iloc[i]:
                trend = 1
                sar.iloc[i] = ep
                ep = H.iloc[i]
                af = 0.02
            else:
                if L.iloc[i] < ep:
                    ep = L.iloc[i]
                    af = min(af + 0.02, af_max)
    
    df['sar'] = sar
    df['sar_bullish'] = C > sar  # SAR看涨
    
    # =========================================================================
    # 8. BBI 多空分界线
    # =========================================================================
    BBI = (MA(C, 3) + MA(C, 6) + MA(C, 12) + MA(C, 24)) / 4
    df['bbi'] = BBI
    df['bbi_bullish'] = C > BBI  # 价格在BBI上方
    df['bbi_cross_up'] = CROSS(C, BBI)  # 价格上穿BBI
    
    # =========================================================================
    # 9. OBV 能量潮
    # =========================================================================
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = V.iloc[0]
    for i in range(1, len(df)):
        if C.iloc[i] > C.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + V.iloc[i]
        elif C.iloc[i] < C.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - V.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    df['obv'] = obv
    df['obv_ma5'] = MA(obv, 5)
    df['obv_ma10'] = MA(obv, 10)
    df['obv_rising'] = obv > REF(obv, 1)  # OBV上升
    df['obv_golden_cross'] = CROSS(df['obv_ma5'], df['obv_ma10'])  # OBV金叉
    
    # =========================================================================
    # 10. 道式理论特征
    # =========================================================================
    # 趋势判断: 高点和低点都在抬高
    HH5 = HHV(H, 5)
    LL5 = LLV(L, 5)
    df['dow_higher_high'] = HH5 > REF(HH5, 5)  # 更高的高点
    df['dow_higher_low'] = LL5 > REF(LL5, 5)  # 更高的低点
    df['dow_uptrend'] = df['dow_higher_high'] & df['dow_higher_low']  # 上升趋势
    
    # 均线多头排列
    MA5 = MA(C, 5)
    MA10 = MA(C, 10)
    MA20 = MA(C, 20)
    MA60 = MA(C, 60)
    df['ma_bullish_align'] = (MA5 > MA10) & (MA10 > MA20) & (MA20 > MA60)  # 均线多头排列
    df['price_above_ma20'] = C > MA20  # 价格在20日均线上方
    df['price_above_ma60'] = C > MA60  # 价格在60日均线上方
    
    # =========================================================================
    # 11. 威克夫理论特征
    # =========================================================================
    # 成交量分析
    VOL_MA5 = MA(V, 5)
    VOL_MA20 = MA(V, 20)
    df['vol_above_avg'] = V > VOL_MA20  # 成交量高于平均
    df['vol_surge'] = V > VOL_MA5 * 1.5  # 成交量激增
    df['vol_shrink'] = V < VOL_MA5 * 0.7  # 成交量萎缩
    
    # 价量配合
    df['price_vol_up'] = (C > REF(C, 1)) & (V > REF(V, 1))  # 价涨量增
    df['price_up_vol_down'] = (C > REF(C, 1)) & (V < REF(V, 1))  # 价涨量缩
    
    # 吸筹特征: 低位放量
    price_position = (C - LLV(L, 60)) / (HHV(H, 60) - LLV(L, 60) + 0.0001)
    df['wyckoff_accumulation'] = (price_position < 0.3) & df['vol_above_avg']  # 低位放量
    
    # =========================================================================
    # 12. 其他辅助特征
    # =========================================================================
    # 涨跌幅
    df['pct_change'] = (C - REF(C, 1)) / REF(C, 1) * 100
    df['pct_change_3d'] = (C - REF(C, 3)) / REF(C, 3) * 100
    df['pct_change_5d'] = (C - REF(C, 5)) / REF(C, 5) * 100
    
    # 振幅
    df['amplitude'] = (H - L) / REF(C, 1) * 100
    
    # 实体比例
    df['body_ratio'] = ABS(C - O) / (H - L + 0.0001)
    
    # 上下影线
    df['upper_shadow'] = (H - pd.concat([C, O], axis=1).max(axis=1)) / (H - L + 0.0001)
    df['lower_shadow'] = (pd.concat([C, O], axis=1).min(axis=1) - L) / (H - L + 0.0001)
    
    return df


# ==============================================================================
# 信号识别函数
# ==============================================================================

def identify_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别目标信号：六脉神剑4红+、缠论5买点、买卖点1/2买
    """
    # 计算基础指标
    df = calculate_six_veins(df)
    df = calculate_buy_sell_points(df)
    df = calculate_chan_theory(df)
    
    # 定义目标信号
    # 六脉神剑4红以上
    df['signal_six_veins_4plus'] = df['six_veins_count'] >= 4
    
    # 缠论5个买点
    df['signal_chan_buy1'] = df['chan_buy1']
    df['signal_chan_buy2'] = df['chan_buy2']
    df['signal_chan_buy3'] = df['chan_buy3']
    df['signal_chan_strong_buy2'] = df['chan_strong_buy2']
    df['signal_chan_like_buy2'] = df['chan_like_buy2']
    
    # 买卖点的1买和2买
    df['signal_buy1'] = df['buy1']
    df['signal_buy2'] = df['buy2']
    
    # 任意信号
    df['any_signal'] = (
        df['signal_six_veins_4plus'] |
        df['signal_chan_buy1'] |
        df['signal_chan_buy2'] |
        df['signal_chan_buy3'] |
        df['signal_chan_strong_buy2'] |
        df['signal_chan_like_buy2'] |
        df['signal_buy1'] |
        df['signal_buy2']
    )
    
    return df


# ==============================================================================
# 涨幅计算函数
# ==============================================================================

def calculate_future_returns(df: pd.DataFrame, max_days: int = MAX_HOLD_DAYS) -> pd.DataFrame:
    """
    计算信号出现后的未来涨幅
    """
    C = df['close']
    
    # 计算未来N天的最高涨幅
    for days in [5, 10, 15, 20]:
        future_high = HHV(C.shift(-days), days)
        df[f'future_max_gain_{days}d'] = (future_high - C) / C * 100
    
    # 计算是否达到目标涨幅
    df['reached_target'] = df['future_max_gain_20d'] >= TARGET_GAIN
    
    return df


# ==============================================================================
# 单只股票处理函数
# ==============================================================================

def process_single_stock(filepath: str) -> Optional[Dict]:
    """
    处理单只股票，提取信号和特征
    """
    try:
        # 读取数据
        df = pd.read_csv(filepath)
        if len(df) < MIN_DATA_DAYS:
            return None
        
        # 标准化列名
        name_map = {
            '日期': 'date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume',
            '成交额': 'amount', '振幅': 'amplitude_pct',
            '涨跌幅': 'pct_chg', '涨跌额': 'change',
            '换手率': 'turnover', '名称': 'name'
        }
        df = df.rename(columns=name_map)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        # 确保必要列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None
        
        # 计算指标
        df = identify_signals(df)
        df = calculate_extended_indicators(df)
        df = calculate_future_returns(df)
        
        # 提取信号点
        signal_rows = df[df['any_signal'] == True].copy()
        if len(signal_rows) == 0:
            return None
        
        # 收集结果
        results = {
            'total_signals': len(signal_rows),
            'success_signals': int(signal_rows['reached_target'].sum()),
            'signal_details': []
        }
        
        # 提取每个信号点的特征
        feature_cols = [
            # MACD
            'macd_golden_cross', 'macd_above_zero', 'macd_hist_positive',
            # KDJ
            'kdj_golden_cross', 'kdj_oversold', 'kdj_overbought',
            # BOLL
            'boll_squeeze', 'boll_position',
            # RSI
            'rsi_oversold', 'rsi_overbought', 'rsi_golden_cross',
            # DMI
            'dmi_trend_up', 'dmi_strong_trend',
            # DMA
            'dma_golden_cross', 'dma_above_zero',
            # SAR
            'sar_bullish',
            # BBI
            'bbi_bullish', 'bbi_cross_up',
            # OBV
            'obv_rising', 'obv_golden_cross',
            # 道式理论
            'dow_uptrend', 'ma_bullish_align', 'price_above_ma20', 'price_above_ma60',
            # 威克夫理论
            'vol_above_avg', 'vol_surge', 'vol_shrink',
            'price_vol_up', 'price_up_vol_down', 'wyckoff_accumulation',
            # 其他
            'pct_change', 'amplitude', 'body_ratio', 'upper_shadow', 'lower_shadow'
        ]
        
        for idx, row in signal_rows.iterrows():
            detail = {
                'reached_target': bool(row['reached_target']),
                'future_max_gain_20d': float(row['future_max_gain_20d']) if not pd.isna(row['future_max_gain_20d']) else 0,
                'features': {}
            }
            for col in feature_cols:
                if col in row:
                    val = row[col]
                    if isinstance(val, (bool, np.bool_)):
                        detail['features'][col] = bool(val)
                    elif isinstance(val, (int, float, np.integer, np.floating)):
                        detail['features'][col] = float(val) if not pd.isna(val) else 0
            
            results['signal_details'].append(detail)
        
        return results
        
    except Exception as e:
        return None


# ==============================================================================
# 特征分析函数
# ==============================================================================

def analyze_features(all_results: List[Dict]) -> Dict:
    """
    分析成功案例的共同特征
    """
    # 收集所有信号的特征
    success_features = defaultdict(list)
    failure_features = defaultdict(list)
    
    for result in all_results:
        if result is None:
            continue
        for detail in result['signal_details']:
            target = success_features if detail['reached_target'] else failure_features
            for feat_name, feat_val in detail['features'].items():
                target[feat_name].append(feat_val)
    
    # 计算特征统计
    analysis = {}
    
    for feat_name in success_features.keys():
        success_vals = success_features[feat_name]
        failure_vals = failure_features.get(feat_name, [])
        
        if len(success_vals) == 0:
            continue
        
        # 布尔型特征
        if all(isinstance(v, bool) for v in success_vals):
            success_rate = sum(success_vals) / len(success_vals) * 100
            failure_rate = sum(failure_vals) / len(failure_vals) * 100 if failure_vals else 0
            diff = success_rate - failure_rate
            
            analysis[feat_name] = {
                'type': 'boolean',
                'success_rate': success_rate,
                'failure_rate': failure_rate,
                'difference': diff,
                'predictive_power': abs(diff)
            }
        # 数值型特征
        else:
            success_mean = np.mean([v for v in success_vals if not np.isnan(v)])
            failure_mean = np.mean([v for v in failure_vals if not np.isnan(v)]) if failure_vals else 0
            
            analysis[feat_name] = {
                'type': 'numeric',
                'success_mean': success_mean,
                'failure_mean': failure_mean,
                'difference': success_mean - failure_mean
            }
    
    return analysis


# ==============================================================================
# 报告生成函数
# ==============================================================================

def generate_report(all_results: List[Dict], feature_analysis: Dict) -> str:
    """
    生成分析报告
    """
    # 统计总体数据
    total_signals = sum(r['total_signals'] for r in all_results if r)
    success_signals = sum(r['success_signals'] for r in all_results if r)
    success_rate = success_signals / total_signals * 100 if total_signals > 0 else 0
    
    report = f"""# 信号成功率分析报告

> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> **目标涨幅**: {TARGET_GAIN}%
> **最大持有天数**: {MAX_HOLD_DAYS}天

---

## 1. 总体统计

| 指标 | 数值 |
| :--- | :--- |
| 分析股票数 | {len([r for r in all_results if r])} |
| 总信号数 | {total_signals} |
| 成功信号数（涨幅≥{TARGET_GAIN}%） | {success_signals} |
| **总体成功率** | **{success_rate:.2f}%** |

---

## 2. 成功案例的共同特征分析

以下是成功案例（信号出现后涨幅超过{TARGET_GAIN}%）与失败案例在各项技术指标上的差异。
**差异值越大，说明该特征对预测成功的参考价值越高。**

### 2.1 布尔型特征（出现率对比）

| 特征名称 | 成功案例出现率 | 失败案例出现率 | 差异 | 预测价值 |
| :--- | :---: | :---: | :---: | :---: |
"""
    
    # 布尔型特征排序
    bool_features = [(k, v) for k, v in feature_analysis.items() if v['type'] == 'boolean']
    bool_features.sort(key=lambda x: x[1]['predictive_power'], reverse=True)
    
    for feat_name, feat_data in bool_features[:20]:
        report += f"| {feat_name} | {feat_data['success_rate']:.1f}% | {feat_data['failure_rate']:.1f}% | {feat_data['difference']:+.1f}% | {'⭐⭐⭐' if feat_data['predictive_power'] > 15 else '⭐⭐' if feat_data['predictive_power'] > 10 else '⭐'} |\n"
    
    report += """
### 2.2 数值型特征（均值对比）

| 特征名称 | 成功案例均值 | 失败案例均值 | 差异 |
| :--- | :---: | :---: | :---: |
"""
    
    # 数值型特征排序
    num_features = [(k, v) for k, v in feature_analysis.items() if v['type'] == 'numeric']
    num_features.sort(key=lambda x: abs(x[1]['difference']), reverse=True)
    
    for feat_name, feat_data in num_features[:15]:
        report += f"| {feat_name} | {feat_data['success_mean']:.2f} | {feat_data['failure_mean']:.2f} | {feat_data['difference']:+.2f} |\n"
    
    report += """
---

## 3. 关键发现与预测规则

基于以上分析，以下是能够提高信号成功率的关键特征组合：

### 3.1 高预测价值特征（差异>10%）

"""
    
    high_value_features = [f for f, d in bool_features if d['predictive_power'] > 10]
    for i, feat in enumerate(high_value_features[:10], 1):
        report += f"{i}. **{feat}**\n"
    
    report += """
### 3.2 推荐的过滤规则

当出现六脉神剑4红+、缠论买点、或买卖点1/2买信号时，如果同时满足以下条件，成功率会显著提高：

"""
    
    # 生成推荐规则
    rules = []
    for feat_name, feat_data in bool_features[:5]:
        if feat_data['difference'] > 10:
            rules.append(f"- **{feat_name}** 为 True（成功率提升约 {feat_data['difference']:.1f}%）")
    
    if rules:
        report += "\n".join(rules)
    else:
        report += "- 暂无显著的单一特征能大幅提升成功率，建议组合多个特征使用。"
    
    report += """

---

## 4. 理论解读

### 4.1 道式理论视角

道式理论强调趋势的重要性。从分析结果来看，当信号出现时：
- **均线多头排列** (`ma_bullish_align`) 和 **价格在均线上方** (`price_above_ma20`, `price_above_ma60`) 是重要的趋势确认指标。
- **更高的高点和低点** (`dow_uptrend`) 表明上升趋势已经形成。

### 4.2 威克夫理论视角

威克夫理论关注价量关系和主力行为：
- **成交量激增** (`vol_surge`) 配合价格上涨，可能表明主力资金介入。
- **低位放量** (`wyckoff_accumulation`) 是主力吸筹的典型特征。
- **价涨量增** (`price_vol_up`) 是健康上涨的标志。

### 4.3 综合建议

1. 当信号出现时，优先关注趋势确认指标（均线排列、道式趋势）。
2. 结合成交量分析，确认是否有资金配合。
3. 使用 MACD、KDJ 等动量指标确认买入时机。
4. 避免在超买区域追高（RSI > 70, KDJ J值 > 80）。

---

*本报告由自动化分析脚本生成，仅供参考，不构成投资建议。*
"""
    
    return report


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    print("=" * 60)
    print("信号成功率分析脚本")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 获取所有股票文件
    stock_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.csv'):
                stock_files.append(os.path.join(root, file))
    
    print(f"\n找到 {len(stock_files)} 只股票数据")
    
    if len(stock_files) == 0:
        print("错误: 未找到股票数据文件")
        return
    
    # 多进程处理
    num_workers = max(1, cpu_count() - 1)
    print(f"使用 {num_workers} 个进程并行处理...")
    
    with Pool(num_workers) as pool:
        all_results = pool.map(process_single_stock, stock_files)
    
    # 过滤空结果
    all_results = [r for r in all_results if r is not None]
    print(f"成功处理 {len(all_results)} 只股票")
    
    # 特征分析
    print("\n正在分析特征...")
    feature_analysis = analyze_features(all_results)
    
    # 生成报告
    print("正在生成报告...")
    report = generate_report(all_results, feature_analysis)
    
    # 保存报告
    report_path = os.path.join(REPORT_DIR, 'signal_success_analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存到: {report_path}")
    print("\n分析完成!")


if __name__ == "__main__":
    main()
