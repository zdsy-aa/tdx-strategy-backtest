# Python 脚本目录 (py_file/)

> 包含所有指标计算、回测验证和数据处理的 Python 脚本。

## 📁 目录结构

```
py_file/
├── README.md                      # 本说明文件
├── indicators.py                  # 核心指标计算模块
├── data_fetcher.py                # 基础数据下载模块
├── download_full_market.py        # [新增] 全量数据下载脚本 (支持多源/断点续传)
├── full_backtest.py               # [新增] 完整回测脚本 (生成总/年/月报表)
├── scheduled_data_update.py       # 定时更新脚本
│
├── 101_six_veins_test.py          # 六脉神剑单指标测试
├── 102_buy_sell_points_test.py    # 买卖点单指标测试
├── 103_chan_lun_test.py           # 缠论买点测试（待实现）
├── 104_money_tree_test.py         # 摇钱树指标测试（待实现）
│
├── 201_steady_combo_test.py       # 稳健组合测试
├── 202_aggressive_combo_test.py   # 激进组合测试
└── 203_resonance_combo_test.py    # 共振组合测试（待实现）
```

## 🔢 脚本编号规则

### 核心工具脚本

| 脚本名称 | 功能描述 | 运行方式 |
| :--- | :--- | :--- |
| `download_full_market.py` | **全量数据下载**<br>下载所有 A 股历史数据，支持多源切换 (AkShare/yfinance) 和断点续传。数据将按市场分类保存到 `data/day/{sh,sz,bj}`。 | `python download_full_market.py` |
| `full_backtest.py` | **完整回测**<br>对所有已下载的股票数据进行全量回测，生成详细的胜率和收益统计报告。 | `python full_backtest.py` |

### 单指标测试（101-199）

| 编号 | 脚本名称 | 测试内容 | 状态 |
|------|---------|---------|------|
| **101** | `101_six_veins_test.py` | 六脉神剑指标测试 | ✅ 已完成 |
| **102** | `102_buy_sell_points_test.py` | 买卖点指标测试 | ✅ 已完成 |
| 103 | `103_chan_lun_test.py` | 缠论买点测试 | 🔄 待实现 |
| 104 | `104_money_tree_test.py` | 摇钱树指标测试 | 🔄 待实现 |

### 组合指标测试（201-299）

| 编号 | 脚本名称 | 测试内容 | 状态 |
|------|---------|---------|------|
| **201** | `201_steady_combo_test.py` | 稳健组合测试 | ✅ 已完成 |
| **202** | `202_aggressive_combo_test.py` | 激进组合测试 | ✅ 已完成 |
| 203 | `203_resonance_combo_test.py` | 共振组合测试 | 🔄 待实现 |

## 📝 核心模块说明

### indicators.py - 核心指标计算模块

包含所有技术指标的计算函数，严格按照通达信原始公式实现：

```python
# 主要函数
calculate_six_veins(df)      # 计算六脉神剑（MACD/KDJ/RSI/LWR/BBI/MTM）
calculate_buy_sell_points(df) # 计算买卖点（庄家线/散户线）
calculate_chan_lun(df)        # 计算缠论买点
calculate_money_tree(df)      # 计算摇钱树信号
calculate_all_signals(df)     # 计算所有信号
```

### download_full_market.py - 全量下载模块

支持从 AkShare/yfinance 下载 A 股历史数据，具备以下特性：
*   **多源切换**：自动尝试 AkShare 和 yfinance，确保数据获取成功率。
*   **断点续传**：自动跳过已下载且完整的文件。
*   **分市场存储**：自动将数据分类存放到 `data/day/sh`, `data/day/sz`, `data/day/bj`。

## 🚀 使用方法

### 1. 安装依赖
```bash
pip install -r ../requirements.txt
```

### 2. 下载全量数据
```bash
python download_full_market.py
```

### 3. 运行完整回测
```bash
python full_backtest.py
```

## 📊 输出结果

所有测试脚本会在 `../data/backtest_results/` 目录下生成对应的报告数据。
