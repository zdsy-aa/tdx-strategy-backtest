# py_file 脚本详细说明与执行指南

## 1. 核心思想：数据与展示分离

本项目的核心工作流是"**后端产出数据，前端消费数据**"。

- **Python 脚本 (`py_file/`)**：作为后端，负责从网络获取股票数据、执行策略回测、计算指标，并将最终结果以 `.json` 格式产出。
- **Web 前端 (`web/`)**：作为数据展示层，读取 Python 脚本生成的 `.json` 文件，并通过图表和表格将其可视化。

**关键点**：前端网页本身不产生或计算数据，它只是一个"看板"。要更新网页上的内容，必须重新运行 Python 脚本来生成新的 `.json` 数据文件。

---

## 2. 数据流向图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           数据流向                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [互联网数据源]                                                          │
│       │                                                                 │
│       ▼                                                                 │
│  stock_downloader_script.py  ──────────────────────────────────────┐    │
│       │                                                            │    │
│       ▼                                                            │    │
│  data/day/                                                         │    │
│  ├── sh/  (沪市 2400+ 只)                                          │    │
│  ├── sz/  (深市 3000+ 只)                                          │    │
│  └── bj/  (北交所 290+ 只)                                         │    │
│       │                                                            │    │
│       ▼                                                            │    │
│  full_backtest.py / generate_web_data.py                           │    │
│       │                                                            │    │
│       ▼                                                            │    │
│  web/client/src/data/                                              │    │
│  ├── backtest_results.json  ──► 回测数据页面                        │    │
│  ├── stock_reports.json     ──► 报告明细页面                        │    │
│  └── strategies.json        ──► 策略介绍页面（手动维护）             │    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 脚本功能详解

`py_file` 目录下的脚本可以分为三大类：数据下载、策略回测、工具模块。

### A. 数据下载类 (Data Acquisition)

这类脚本负责从互联网获取原始的股票历史数据，并保存为本地 CSV 文件。

| 脚本文件 | 主要作用 | 状态 |
| :--- | :--- | :--- |
| **`stock_downloader_script.py`** | **（主下载脚本）** 统一的股票数据下载入口，整合了 efinance 和 akshare 两个数据源，支持增量更新、断点续传、市场筛选等功能。数据分类存储在 `data/day/sh`、`data/day/sz`、`data/day/bj` 目录下。 | ✅ **推荐使用** |
| `download_full_market.py` | （旧版）功能与主下载脚本类似，使用 akshare + yfinance 双数据源。 | ⚠️ 备用 |
| `download_all_a_stocks.py` | （旧版）使用 akshare 下载全 A 股数据。 | ⚠️ 备用 |
| `download_all_stocks.py` | （旧版）早期版本的下载脚本。 | ⚠️ 备用 |
| `data_fetcher.py` | （基础模块）提供了按股票代码下载单个文件的核心函数。 | 🔧 模块 |
| `get_all_stocks*.py` | （辅助工具）用于获取全市场股票代码列表。 | 🔧 模块 |

**主下载脚本使用方法：**

```bash
# 下载全部 A 股（首次运行）
python stock_downloader_script.py

# 增量更新（只更新已有数据）
python stock_downloader_script.py --update

# 只下载沪市
python stock_downloader_script.py --market sh

# 只下载深市
python stock_downloader_script.py --market sz

# 只下载北交所
python stock_downloader_script.py --market bj

# 限制下载数量（测试用）
python stock_downloader_script.py --limit 100

# 断点续传
python stock_downloader_script.py --resume
```

### B. 策略回测与数据生成类 (Backtesting & Data Generation)

这类脚本是项目的核心，它们读取下载好的 CSV 数据，运行各种交易策略，并生成供前端使用的 JSON 文件。

| 脚本文件 | 主要作用 | 输出文件 | 状态 |
| :--- | :--- | :--- | :--- |
| **`full_backtest.py`** | **（完整回测脚本）** 遍历 `data/day/` 下的所有股票数据，执行所有已定义的策略（如六脉神剑、买卖点等），并汇总所有结果。 | `backtest_results.json`<br>`stock_reports.json` | ✅ **推荐使用** |
| **`generate_web_data.py`** | **（快速生成脚本）** 简化版的数据生成脚本，处理速度更快，适合快速验证。 | `stock_reports.json` | ✅ **快速验证** |
| `quick_backtest.py` | （快速版）`full_backtest.py` 的多进程加速版本。 | 同上 | ⚠️ 备用 |
| `test_sell_points.py` | **（专项测试）** 专门用于测试"买入后持有多久卖出"对策略收益的影响。 | 更新 `backtest_results.json` | 🔬 测试 |
| `update_web_data.py` | （辅助工具）一键更新脚本，按顺序调用下载和回测。 | 同 `full_backtest.py` | 🔧 工具 |
| `simple_backtest.py` | （简化版）用于教学或快速演示的简化回测脚本。 | - | 📚 教学 |
| `*test.py` | （单元测试）用于单独测试某一个具体指标或策略的脚本。 | 控制台输出 | 🔬 测试 |

### C. 核心模块与工具类 (Core Modules & Utilities)

这类脚本定义了项目所需的基础函数和计算逻辑，它们通常不直接运行，而是被其他脚本导入和调用。

| 脚本文件 | 主要作用 |
| :--- | :--- |
| **`indicators.py`** | **（最重要的模块）** 定义了所有技术指标（如 MACD, KDJ, RSI, BBI, 缠论买点等）的数学计算公式。所有策略的信号都源于此。 |
| `scheduled_data_update.py` | 一个用于设置定时任务的示例脚本，可以配置系统在每天固定时间自动更新数据和执行回测。 |

---

## 4. 网页数据文件说明

网页所需的数据文件位于 `web/client/src/data/` 目录下：

| 文件名 | 用途 | 生成脚本 | 更新频率 |
| :--- | :--- | :--- | :--- |
| `backtest_results.json` | 回测数据页面：各策略的总体统计、年度统计、月度统计 | `full_backtest.py` | 每次回测后 |
| `stock_reports.json` | 报告明细页面：每只股票的详细回测数据 | `full_backtest.py` 或 `generate_web_data.py` | 每次回测后 |
| `strategies.json` | 策略介绍页面：策略说明、指标解释 | **手动维护** | 策略变更时 |

---

## 5. 标准执行顺序 (Standard Operating Procedure)

请严格按照以下顺序执行，以确保数据完整和结果准确。

### 第 1 步：安装依赖

在项目根目录下，首先为 Python 创建虚拟环境并安装依赖。

```bash
# 1. 创建虚拟环境
python3 -m venv venv

# 2. 激活虚拟环境 (Linux/macOS)
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 第 2 步：下载股票历史数据

这是所有分析的基础。进入 `py_file` 目录，运行主下载脚本。

```bash
cd py_file
python stock_downloader_script.py
```

*此过程会下载数千个文件，耗时较长，请耐心等待。如果中断，使用 `--resume` 参数断点续传。*

### 第 3 步：执行完整回测

数据下载完成后，运行核心回测脚本，生成网页所需的 JSON 数据。

```bash
# 完整回测（推荐）
python full_backtest.py

# 或者使用快速生成脚本（速度更快）
python generate_web_data.py
```

*此过程会遍历所有股票数据进行计算，同样需要较长时间。*

### 第 4 步：启动网页查看结果

回测完成后，回到项目根目录，进入 `web` 目录启动前端服务。

```bash
cd ../web
pnpm install  # 首次运行需要安装前端依赖
pnpm dev      # 启动开发服务器
```

现在，打开浏览器访问 `http://localhost:5173` 即可看到可视化的回测仪表盘。

---

## 6. 如何更新数据？

当需要更新到最新的交易数据时，只需重复 **第 2 步** 和 **第 3 步** 即可。

```bash
# 1. 激活虚拟环境
source venv/bin/activate

# 2. 更新数据（增量模式）
cd py_file
python stock_downloader_script.py --update

# 3. 重新生成报告
python full_backtest.py
```

完成后，刷新网页即可看到最新的回测结果。

---

## 7. 常见问题 (FAQ)

### Q: 为什么网页上的数据没有变化？

A: 网页是静态部署的，它只显示构建时打包的 JSON 数据。要更新数据，需要：
1. 运行 Python 脚本生成新的 JSON 文件
2. 将新文件提交到 Git
3. 触发 Vercel 重新部署

### Q: 下载脚本报错怎么办？

A: 
1. 检查网络连接
2. 尝试使用 `--resume` 参数断点续传
3. 如果 efinance 失败，脚本会自动切换到 akshare
4. 可以尝试使用备用脚本 `download_full_market.py`

### Q: 回测太慢怎么办？

A: 
1. 使用 `generate_web_data.py` 快速生成脚本
2. 使用 `--limit` 参数限制处理的股票数量
3. 使用 `quick_backtest.py` 多进程版本

### Q: 如何添加新的策略？

A: 
1. 在 `indicators.py` 中添加新的指标计算函数
2. 在 `full_backtest.py` 的 `signal_types` 列表中添加新策略
3. 运行回测生成新数据
4. 更新 `strategies.json` 添加策略说明
