# py_file 脚本详细说明与执行指南

## 1. 核心思想：数据与展示分离

本项目的核心工作流是"**后端产出数据，前端消费数据**"。

- **Python 脚本 (`py_file/`)**：作为后端，负责从网络获取股票数据、执行策略回测、计算指标，并将最终结果以 `.json` 格式产出。
- **Web 前端 (`web/`)**：作为数据展示层，读取 Python 脚本生成的 `.json` 文件，并通过图表和表格将其可视化。

---

## 2. 脚本命名规范

为了方便管理和识别，脚本遵循以下命名规范：
- **10开头**：单指标测试脚本（如 `101_six_veins_test.py`）
- **20开头**：组合指标测试脚本（如 `201_steady_combo_test.py`）

---

## 3. 脚本功能详解

### A. 数据下载类 (Data Acquisition)

| 脚本文件 | 主要作用 | 状态 |
| :--- | :--- | :--- |
| **`stock_downloader.py`** | **（推荐下载脚本）** 支持全量、增量、指定股票下载。具备自动停止机制：若股票最近10个交易日无交易，则自动跳过。 | ✅ **推荐使用** |
| `data_fetcher.py` | （基础模块）提供了数据读取和预处理的通用函数。 | 🔧 模块 |
| `get_all_stocks.py` | （辅助工具）用于获取全市场股票代码列表。 | 🔧 模块 |

**推荐下载脚本使用方法：**

```bash
# 全量下载所有股票
python stock_downloader.py --full

# 增量更新（只下载新数据并合并）
python stock_downloader.py --incremental

# 下载指定股票
python stock_downloader.py --stocks 000001,600000

# 同时下载指数数据
python stock_downloader.py --incremental --indices
```

### B. 策略回测类 (Backtesting)

| 脚本文件 | 主要作用 | 类别 |
| :--- | :--- | :--- |
| **`full_backtest.py`** | **（核心回测）** 遍历所有股票，执行所有策略并生成 Web 所需数据。 | 核心 |
| `101_six_veins_test.py` | 六脉神剑策略专项测试。 | 单指标 |
| `102_buy_sell_points_test.py` | 基础买卖点策略专项测试。 | 单指标 |
| `103_chan_buy_point_test.py` | 缠论买点策略专项测试。 | 单指标 |
| `104_test_sell_points.py` | 卖出点优化专项测试。 | 单指标 |
| `201_steady_combo_test.py` | 稳健型组合策略测试。 | 组合指标 |
| `202_aggressive_combo_test.py` | 激进型组合策略测试。 | 组合指标 |
| `quick_backtest.py` | 多进程加速版回测。 | 工具 |

### C. 核心模块与工具类

| 脚本文件 | 主要作用 |
| :--- | :--- |
| **`indicators.py`** | **（核心指标库）** 定义了所有技术指标的计算公式。 |
| `auto_update_daily.py` | 每日自动更新脚本。 |
| `generate_web_data.py` | 为 Web 前端生成展示数据。 |

---

## 4. 标准执行顺序

1. **下载数据**：`python stock_downloader.py --incremental`
2. **执行回测**：`python full_backtest.py`
3. **启动 Web**：在 `web` 目录下执行 `pnpm dev`
