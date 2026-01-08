# Python 脚本目录 (py_file/)

> 包含所有指标计算、回测验证和数据处理的 Python 脚本。

## 📁 目录结构

```
py_file/
├── README.md                      # 本说明文件
├── indicators.py                  # 核心指标计算模块
├── data_fetcher.py                # 基础数据下载模块
├── download_full_market.py        # 全量数据下载脚本
├── full_backtest.py               # 完整回测脚本 (生成总/年/月报表)
├── bottom_fishing_strategy.py     # [新增] 抄底方案回测脚本 (缠论一买 + 六脉神剑)
├── chan_buy_point_test.py         # [新增] 缠论买点单指标测试脚本
├── scheduled_data_update.py       # 定时更新脚本
│
├── 101_six_veins_test.py          # 六脉神剑单指标测试
├── 102_buy_sell_points_test.py    # 买卖点单指标测试
├── 103_chan_lun_test.py           # 缠论买点测试 (已由 chan_buy_point_test.py 实现)
├── 104_money_tree_test.py         # 摇钱树指标测试
│
├── 201_steady_combo_test.py       # 稳健组合测试
├── 202_aggressive_combo_test.py   # 激进组合测试
└── 203_resonance_combo_test.py    # 共振组合测试
```

## 🔢 脚本编号规则

### 核心工具脚本

| 脚本名称 | 功能描述 | 运行方式 |
| :--- | :--- | :--- |
| `download_full_market.py` | **全量数据下载**<br>下载所有 A 股历史数据，支持多源切换和断点续传。数据按市场分类保存到 `data/day/{sh,sz,bj}`。 | `python download_full_market.py` |
| `full_backtest.py` | **完整回测**<br>对所有已下载的股票数据进行全量回测，生成详细的胜率和收益统计报告。 | `python full_backtest.py` |
| `bottom_fishing_strategy.py` | **抄底方案回测**<br>组合方案：缠论一买 + 六脉神剑 ≥ 3红。卖出条件：二买减仓50%，三买清仓。 | `python bottom_fishing_strategy.py` |
| `chan_buy_point_test.py` | **缠论买点测试**<br>单指标方案：独立测试缠论一买、二买、三买的胜率和收益。 | `python chan_buy_point_test.py` |

### 单指标测试（101-199）

| 编号 | 脚本名称 | 测试内容 | 状态 |
|------|---------|---------|------|
| **101** | `101_six_veins_test.py` | 六脉神剑指标测试 | ✅ 已完成 |
| **102** | `102_buy_sell_points_test.py` | 买卖点指标测试 | ✅ 已完成 |
| **103** | `chan_buy_point_test.py` | 缠论买点测试 | ✅ 已完成 |
| 104 | `104_money_tree_test.py` | 摇钱树指标测试 | 🔄 待实现 |

### 组合指标测试（201-299）

| 编号 | 脚本名称 | 测试内容 | 状态 |
|------|---------|---------|------|
| **201** | `201_steady_combo_test.py` | 稳健组合测试 | ✅ 已完成 |
| **202** | `202_aggressive_combo_test.py` | 激进组合测试 | ✅ 已完成 |
| **203** | `bottom_fishing_strategy.py` | 抄底方案组合测试 | ✅ 已完成 |

## 🚀 脚本执行顺序建议

为了确保数据完整和回测准确，建议按照以下顺序执行脚本：

1.  **数据准备**：
    运行 `download_full_market.py` 下载并更新全量股票日线数据。
2.  **单指标验证**：
    运行 `101_six_veins_test.py`、`102_buy_sell_points_test.py` 和 `chan_buy_point_test.py` 验证基础指标的有效性。
3.  **组合方案回测**：
    运行 `201_steady_combo_test.py`、`202_aggressive_combo_test.py` 和 `bottom_fishing_strategy.py` 测试多指标共振方案。
4.  **全量综合回测**：
    运行 `full_backtest.py` 生成最终的综合回测报告。

## 📝 核心模块说明

### indicators.py - 核心指标计算模块

包含所有技术指标的计算函数，严格按照通达信原始公式实现：

```python
# 主要函数
calculate_six_veins(df)      # 计算六脉神剑
calculate_buy_sell_points(df) # 计算买卖点
calculate_chan_theory(df)     # 计算缠论买点
calculate_money_tree(df)      # 计算摇钱树信号
calculate_all_signals(df)     # 计算所有信号
```

## 📊 输出结果

所有测试脚本会在 `../data/backtest_results/` 目录下生成对应的报告数据。
