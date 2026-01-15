# 股票策略回测与分析系统 (Stock Strategy Backtest & Analysis System)

这是一个基于 Python 和 React 的全栈股票量化系统，旨在为A股投资者提供一个从数据获取、策略回测到模式分析的完整解决方案。系统不仅能够验证技术指标策略的历史表现，还能深入挖掘成功信号背后的共性特征，为策略优化提供数据支持。

---

## 🚀 系统核心功能

| 功能模块 | 描述 |
| :--- | :--- |
| **数据中心** | - **全量/增量下载**：通过 `data_fetcher.py` 获取A股所有股票的历史日线数据，并支持每日增量更新。<br>- **自动管理**：自动识别并跳过长期停牌或无交易的股票，提高更新效率。 |
| **策略回测** | - **单指标策略**：整合了六脉神剑、买卖点、缠论买点等多种基础策略，通过 `single_strategy_backtest.py` 一键回测。<br>- **组合策略**：提供稳健型和激进型两种组合策略，通过 `combo_strategy_backtest.py` 进行回测，寻找更优的信号组合。 |
| **信号分析** | - **成功案例扫描**：`signal_success_scanner.py` 自动扫描所有股票，识别指定买入信号，并验证其在未来15个交易日的表现。<br>- **模式识别**：`pattern_analyzer.py` 对涨幅超过5%的成功案例进行深度分析，挖掘其在MACD, KDJ, BOLL, RSI等11种常用指标以及道氏、威科夫理论上的共性特征。 |
| **Web可视化** | - **前端框架**：基于 React, TypeScript 和 TailwindCSS 构建的现代化Web界面。<br>- **数据展示**：将所有回测和分析结果以图表和表格形式清晰展示，包括策略对比、信号统计和模式分析报告。 |

---

## 📊 网页与脚本的自动化联动

新版系统实现了**完全自动化**的数据流转，Python脚本的输出会**直接生成**到Web前端所需的数据目录中，无需任何手动复制操作。运行回测或分析脚本后，只需刷新网页即可看到最新结果。

| 网页板块 | 展示内容 | 数据来源 (自动生成) | 生成脚本 | 脚本作用 |
| :--- | :--- | :--- | :--- | :--- |
| **单指标回测** | 六脉神剑、买卖点、缠论等策略表现 | `web/client/src/data/backtest_single.json` | `single_strategy_backtest.py` | 执行所有单指标策略的全量回测，并按胜率和收益排名。 |
| **组合策略回测** | 稳健型、激进型组合策略表现 | `web/client/src/data/backtest_combo.json` | `combo_strategy_backtest.py` | 执行两种核心组合策略的回测，寻找最佳策略配置。 |
| **分析报告** | 成功信号的统计与共性特征分析 | `web/client/src/data/signal_summary.json`<br>`web/client/src/data/pattern_summary.json`<br>`web/client/src/data/pattern_analysis_by_signal.json` | `signal_success_scanner.py`<br>`pattern_analyzer.py` | 扫描并分析成功信号，生成多维度统计报告。 |

> **重要提示**：在启动Web服务前，请务必至少运行一次相关Python脚本以生成初始数据文件，否则页面可能无法正常显示。

---

## 📂 核心脚本说明 (`py_file/` 目录)

经过重构，脚本结构更加清晰，功能更加集中。

### 1. 数据获取
*   **`data_fetcher.py`**
    *   **作用**：A股日线数据下载与管理模块。
    *   **核心命令**：
        *   `python data_fetcher.py --all`: 首次使用时，下载所有A股历史数据。
        *   `python data_fetcher.py --today`: 每日收盘后，增量更新当天数据。

### 2. 策略回测 (Backtesting)
*   **`single_strategy_backtest.py`**
    *   **作用**：整合了原101-105脚本，用于回测各类**单指标策略**。
    *   **核心命令**：`python single_strategy_backtest.py`
*   **`combo_strategy_backtest.py`**
    *   **作用**：整合了原201-202脚本，用于回测**组合策略**。
    *   **核心命令**：`python combo_strategy_backtest.py`

### 3. 信号分析 (Analysis)
*   **`signal_success_scanner.py`**
    *   **作用**：扫描全市场股票，寻找成功的买入信号案例。
    *   **核心命令**：`python signal_success_scanner.py`
*   **`pattern_analyzer.py`**
    *   **作用**：对成功案例进行深度分析，挖掘共性模式。
    *   **核心命令**：`python pattern_analyzer.py`

### 4. 基础模块
*   **`indicators.py`**：定义了所有技术指标（MACD, KDJ, RSI, 缠论等）的计算函数。
*   **`backtest_utils.py`**：提供多进程、结果汇总等回测通用工具函数。

---

## 🛠️ 快速开始

详细的安装、配置和运行步骤，请参考项目根目录下的 **[部署与使用手册.md](./部署与使用手册.md)**。

---

## 📝 目录结构

```
tdx-strategy-backtest/
├── data/                     # 存放股票历史数据 (CSV格式)
│   └── day/
├── py_file/                  # Python 核心代码
│   ├── data_fetcher.py             # 数据下载脚本
│   ├── single_strategy_backtest.py # 单指标回测
│   ├── combo_strategy_backtest.py  # 组合策略回测
│   ├── signal_success_scanner.py   # 信号成功案例扫描
│   ├── pattern_analyzer.py         # 成功案例模式分析
│   ├── indicators.py             # 技术指标库
│   └── backtest_utils.py           # 回测工具库
├── web/                      # Web 前端代码 (React + TS)
│   ├── client/
│   │   └── src/
│   │       └── data/         # 存放 Python 生成的 JSON 数据 (自动更新)
│   └── ...
├── report/                   # 存放详细的分析报告 (Markdown, CSV)
├── 部署与使用手册.md         # 详细的安装、配置和操作手册
├── 开发说明文档.md           # 系统设计、模块功能等开发者文档
└── README.md                 # 本文档
```
