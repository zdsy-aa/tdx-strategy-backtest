# py_file 脚本详细说明与执行指南

## 1. 核心思想：数据与展示分离

本项目的核心工作流是“**后端产出数据，前端消费数据**”。

- **Python 脚本 (`py_file/`)**：作为后端，负责从网络获取股票数据、执行策略回测、计算指标，并将最终结果以 `.json` 格式产出。
- **Web 前端 (`web/`)**：作为数据展示层，读取 Python 脚本生成的 `.json` 文件，并通过图表和表格将其可视化。

**关键点**：前端网页本身不产生或计算数据，它只是一个“看板”。要更新网页上的内容，必须重新运行 Python 脚本来生成新的 `.json` 数据文件。

---

## 2. 脚本功能详解

`py_file` 目录下的脚本可以分为三大类：数据下载、策略回测、工具模块。

### A. 数据下载类 (Data Acquisition)

这类脚本负责从互联网获取原始的股票历史数据，并保存为本地 CSV 文件。

| 脚本文件 | 主要作用 | 何时运行 |
| :--- | :--- | :--- |
| `download_full_market.py` | **（主下载脚本）** 下载**全部 A 股**（沪、深、北交所共 5400+ 只）的日线数据。支持断点续传，并将数据分类存储在 `data/day/sh`、`data/day/sz`、`data/day/bj` 目录下。 | 1. **首次运行项目时**，必须执行一次以获取基础数据。<br>2. **定期（如每周）**，运行以补充最新的交易日数据。 |
| `download_all_a_stocks.py` | （旧版脚本）功能与 `download_full_market.py` 类似，作为备用。 | `download_full_market.py` 失败时可以尝试。 |
| `data_fetcher.py` | （基础模块）提供了按股票代码下载单个文件的核心函数，被上述主下载脚本调用。不建议直接运行。 | - |
| `get_all_stocks*.py` | （辅助工具）用于获取全市场股票代码列表，是下载脚本的第一步。不建议直接运行。 | - |

### B. 策略回测与数据生成类 (Backtesting & Data Generation)

这类脚本是项目的核心，它们读取下载好的 CSV 数据，运行各种交易策略，并生成供前端使用的 JSON 文件。

| 脚本文件 | 主要作用 | 输出文件 | 何时运行 |
| :--- | :--- | :--- | :--- |
| `full_backtest.py` | **（核心回测脚本）** 遍历 `data/day/` 下的所有股票数据，执行所有已定义的策略（如六脉神剑、买卖点等），并汇总所有结果。 | 1. `web/client/src/data/backtest_results.json`<br>2. `web/client/src/data/stock_reports.json` | 1. **数据下载完成后**，必须运行此脚本才能生成网页所需的数据。<br>2. **策略逻辑修改后**，需要重新运行以查看新策略的效果。 |
| `quick_backtest.py` | （快速版）`full_backtest.py` 的多进程加速版本，适合在性能较好的机器上运行，可以显著缩短回测时间。 | 同上 | 需要快速验证少量数据或策略时。 |
| `test_sell_points.py` | **（专项测试）** 专门用于测试“买入后持有多久卖出”对策略收益的影响，分析最优持仓周期。 | 更新 `backtest_results.json` 中的 `sell_points` 字段。 | 当你想优化策略的卖出时机时。 |
| `update_web_data.py` | （辅助工具）一个简单的包装脚本，用于按顺序调用数据下载和回测脚本，实现一键更新。 | 同 `full_backtest.py` | 想要自动化“下载-回测”流程时。 |
| `simple_backtest.py` | （简化版）一个功能简化的回测脚本，用于教学或快速演示，不涉及复杂的统计。 | - | 用于理解回测基本原理时。 |
| `*test.py` (e.g., `101_six_veins_test.py`) | （单元测试）用于单独测试某一个具体指标或策略的脚本，方便调试和验证。 | 结果通常直接打印在控制台，或生成临时文件。 | 开发新策略或调试现有策略时。 |

### C. 核心模块与工具类 (Core Modules & Utilities)

这类脚本定义了项目所需的基础函数和计算逻辑，它们通常不直接运行，而是被其他脚本导入和调用。

| 脚本文件 | 主要作用 |
| :--- | :--- |
| `indicators.py` | **（最重要的模块）** 定义了所有技术指标（如 MACD, KDJ, RSI, BBI, 缠论买点等）的数学计算公式。所有策略的信号都源于此。 |
| `scheduled_data_update.py` | 一个用于设置定时任务的示例脚本，可以配置系统在每天固定时间自动更新数据和执行回测。 |

---

## 3. 标准执行顺序 (Standard Operating Procedure)

请严格按照以下顺序执行，以确保数据完整和结果准确。

**第 1 步：安装依赖**

在项目根目录下，首先为 Python 创建虚拟环境并安装依赖。

```bash
# 1. 创建并激活虚拟环境 (Linux/macOS)
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt
```

**第 2 步：下载股票历史数据**

这是所有分析的基础。进入 `py_file` 目录，运行主下载脚本。

```bash
cd py_file
python download_full_market.py
```
*此过程会下载数千个文件，耗时较长，请耐心等待。如果中断，再次运行即可断点续传。*

**第 3 步：执行完整回测**

数据下载完成后，运行核心回测脚本，生成网页所需的 JSON 数据。

```bash
python full_backtest.py
```
*此过程会遍历所有股票数据进行计算，同样需要较长时间。可使用 `quick_backtest.py` 加速。*

**第 4 步：启动网页查看结果**

回测完成后，回到项目根目录，进入 `web` 目录启动前端服务。

```bash
cd ../web
pnpm install  # 首次运行需要安装前端依赖
pnpm dev      # 启动开发服务器
```

现在，打开浏览器访问 `http://localhost:5173` 即可看到可视化的回测仪表盘。

---

## 4. 如何更新数据？

当需要更新到最新的交易数据时，只需重复 **第 2 步** 和 **第 3 步** 即可。

```bash
# 1. 激活虚拟环境
source venv/bin/activate

# 2. 更新数据
cd py_file
python download_full_market.py

# 3. 重新生成报告
python full_backtest.py
```

完成后，刷新网页即可看到最新的回测结果。
