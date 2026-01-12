# 股票指标回测系统 (Stock Strategy Backtest System)

这是一个基于 Python 和 React 的全栈股票量化回测系统。它能够下载 A 股历史数据，运行多种技术指标策略（如六脉神剑、买卖点策略），并生成详细的可视化回测报告。

## 🚀 项目简介

本项目旨在帮助投资者验证技术指标策略在历史数据上的表现。系统分为两部分：
1.  **Python 后端**：负责数据下载、指标计算和策略回测，生成 JSON 格式的分析结果。
2.  **Web 前端**：读取 Python 生成的数据，以图表 and 表格形式展示策略的胜率、收益率和交易明细。

---

## 📊 网页与脚本对应关系

网页上的数据展示完全依赖于 Python 脚本的执行结果。以下是各板块数据来源的详细对照表：

| 网页板块 | 展示内容 | 数据来源文件 | 生成该数据的 Python 脚本 | 脚本作用 |
| :--- | :--- | :--- | :--- | :--- |
| **仪表盘 (Dashboard)** | 策略总览、最高收益/胜率策略 | `web/client/src/data/backtest_results.json` | `py_file/full_backtest.py` | 执行所有策略的全量回测，计算总/年/月统计数据 |
| **指标方案 (Strategies)** | 六脉神剑、买卖点等策略详情 | `web/client/src/data/backtest_results.json` | `py_file/full_backtest.py` | 同上，提供策略在不同时间周期的详细表现 |
| **卖出点分析 (Sell Points)** | 不同持仓天数的收益对比 | `web/client/src/data/backtest_results.json` (sell_points字段) | `py_file/104_test_sell_points.py` | 专门测试持仓 1-30 天对收益率和胜率的影响 |
| **股票明细 (Stock Reports)** | 单只股票的详细回测数据 | `web/client/src/data/stock_reports.json` | `py_file/full_backtest.py` | 生成每只股票的独立回测报告，包含最新信号 |

> **注意**：在运行网页之前，必须先运行上述 Python 脚本生成数据文件，否则网页将显示为空或报错。

---

## 📂 核心脚本说明 (`py_file/` 目录)

以下是 `py_file` 目录下各个脚本的详细功能说明：

### 1. 数据准备类
*   **`stock_downloader.py`** (推荐)
    *   **作用**：多功能股票数据下载工具。
    *   **功能**：
        *   `--full`: 全量下载所有股票历史数据。
        *   `--incremental`: 增量下载更新数据。
        *   `--stocks`: 下载指定股票（如 `000001,600000`）。
        *   **自动停止**: 若股票最近10个交易日无交易，将自动加入跳过列表，后续不再下载。
    *   **输出**：数据保存在 `data/day/` 目录下。

### 2. 策略回测类
*   **单指标测试 (10开头)**
    *   `101_six_veins_test.py`: 六脉神剑策略全量股票测试，计算全市场平均胜率。
    *   `102_buy_sell_points_test.py`: 基础买卖点策略全量股票测试。
    *   `103_chan_buy_point_test.py`: 缠论买点策略全量股票测试。
    *   `104_test_sell_points.py`: 卖出点优化全量股票测试。
*   **组合指标测试 (20开头)**
    *   `201_steady_combo_test.py`: 稳健型组合策略全量股票测试，计算全市场平均胜率。
    *   `202_aggressive_combo_test.py`: 激进型组合策略全量股票测试，计算全市场平均胜率。
*   **核心回测引擎**
    *   `full_backtest.py`: 执行所有策略的全量回测主脚本。
    *   `quick_backtest.py`: 多进程加速版回测。

### 3. 基础模块类
*   **`indicators.py`**：定义了各种技术指标（MACD, KDJ, RSI等）的计算公式。
*   **`data_fetcher.py`**：封装了数据读取和预处理的通用函数。

---

## 🛠️ 快速开始

详细的安装和运行步骤，请参考项目根目录下的 **[指标回测.md](./指标回测.md)** 文件。

---

## 📝 目录结构

```
tdx-strategy-backtest/
├── data/
│   └── day/              # 存放股票历史数据 (CSV格式)
├── py_file/              # Python 核心代码
│   ├── stock_downloader.py      # 数据下载脚本
│   ├── full_backtest.py         # 回测主脚本
│   └── ...
├── web/                  # 网页前端代码
│   ├── client/
│   │   └── src/
│   │       └── data/     # 存放 Python 生成的 JSON 数据
│   └── ...
├── 指标回测.md            # 详细操作手册
└── README.md             # 项目说明文档
```
