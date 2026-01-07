# 股票指标回测系统 (Stock Strategy Backtest System)

这是一个基于 Python 和 React 的全栈股票量化回测系统。它能够下载 A 股历史数据，运行多种技术指标策略（如六脉神剑、买卖点策略），并生成详细的可视化回测报告。

## 🚀 项目简介

本项目旨在帮助投资者验证技术指标策略在历史数据上的表现。系统分为两部分：
1.  **Python 后端**：负责数据下载、指标计算和策略回测，生成 JSON 格式的分析结果。
2.  **Web 前端**：读取 Python 生成的数据，以图表和表格形式展示策略的胜率、收益率和交易明细。

---

## 📊 网页与脚本对应关系

网页上的数据展示完全依赖于 Python 脚本的执行结果。以下是各板块数据来源的详细对照表：

| 网页板块 | 展示内容 | 数据来源文件 | 生成该数据的 Python 脚本 | 脚本作用 |
| :--- | :--- | :--- | :--- | :--- |
| **仪表盘 (Dashboard)** | 策略总览、最高收益/胜率策略 | `web/client/src/data/backtest_results.json` | `py_file/full_backtest.py` | 执行所有策略的全量回测，计算总/年/月统计数据 |
| **指标方案 (Strategies)** | 六脉神剑、买卖点等策略详情 | `web/client/src/data/backtest_results.json` | `py_file/full_backtest.py` | 同上，提供策略在不同时间周期的详细表现 |
| **卖出点分析 (Sell Points)** | 不同持仓天数的收益对比 | `web/client/src/data/backtest_results.json` (sell_points字段) | `py_file/test_sell_points.py` | 专门测试持仓 1-30 天对收益率和胜率的影响 |
| **股票明细 (Stock Reports)** | 单只股票的详细回测数据 | `web/client/src/data/stock_reports.json` | `py_file/full_backtest.py` | 生成每只股票的独立回测报告，包含最新信号 |

> **注意**：在运行网页之前，必须先运行上述 Python 脚本生成数据文件，否则网页将显示为空或报错。

---

## 📂 核心脚本说明 (`py_file/` 目录)

以下是 `py_file` 目录下各个脚本的详细功能说明，小白用户请按顺序阅读：

### 1. 数据准备类
*   **`download_full_market.py`** (推荐)
    *   **作用**：下载所有 A 股（5400+只）的历史日线数据。
    *   **输出**：数据保存在 `data/day/` 目录下，每只股票一个 CSV 文件。
    *   **何时运行**：项目初始化时，或需要更新最新行情数据时。

*   **`download_all_a_stocks.py`**
    *   **作用**：旧版下载脚本，功能同上，但可能包含一些旧的逻辑。

### 2. 策略回测类
*   **`full_backtest.py`** (核心)
    *   **作用**：读取 `data/day/` 下的所有数据，运行所有内置策略（六脉神剑、买卖点等）。
    *   **输出**：
        *   `web/client/src/data/backtest_results.json`：策略整体表现统计。
        *   `web/client/src/data/stock_reports.json`：单只股票明细。
    *   **何时运行**：数据下载完成后，或者修改了策略逻辑后。

*   **`test_sell_points.py`**
    *   **作用**：专注于测试"买入后持有 X 天卖出"的最佳天数。
    *   **输出**：更新 `backtest_results.json` 中的 `sell_points` 字段。
    *   **何时运行**：当你想要优化卖出策略时。

*   **`quick_backtest.py`**
    *   **作用**：`full_backtest.py` 的多进程加速版，适合电脑性能较好的用户。

### 3. 基础模块类 (不直接运行)
*   **`indicators.py`**：定义了各种技术指标（MACD, KDJ, RSI等）的计算公式。
*   **`data_fetcher.py`**：封装了数据读取和预处理的通用函数。

---

## 🛠️ 快速开始

详细的安装和运行步骤，请参考项目根目录下的 **[指标回测.md](./指标回测.md)** 文件。该文档包含：
*   Windows / Linux 环境搭建指南
*   常见问题解决方案
*   从零开始的操作手册

---

## 📝 目录结构

```
tdx-strategy-backtest/
├── data/
│   └── day/              # 存放股票历史数据 (CSV格式)
├── py_file/              # Python 核心代码
│   ├── download_full_market.py  # 数据下载脚本
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
