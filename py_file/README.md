# py_file 脚本详细说明与执行指南

## 1. 核心架构：数据驱动与自动化

本项目的核心工作流是"**数据获取 -> 策略回测 -> 模式分析 -> Web展示**"。所有脚本均位于 `py_file/` 目录下，并遵循统一的命名和日志规范。

- **自动化主控**：通过 `a0_auto_update_daily.py` 一键调度所有任务。
- **增量机制**：回测脚本具备智能增量功能，自动识别并仅处理更新过的股票数据，极大提升效率。
- **统一日志**：所有脚本输出均带有 `[YYYY-MM-DD HH:MM:SS]` 时间戳，方便追踪。

---

## 2. 脚本命名规范

脚本采用前缀命名法以区分功能阶段：
- **a0-**：主控脚本 (Master Controller)
- **a1-**：数据获取 (Data Acquisition)
- **a2/a3-**：策略回测 (Backtesting)
- **a4/a21-**：信号分析 (Analysis)
- **a5-**：报告生成 (Reporting)
- **a99-**：通用工具与模块 (Utilities)

---

## 3. 核心脚本详解

### A. 自动化与数据获取

| 脚本文件 | 主要作用 | 核心参数 |
| :--- | :--- | :--- |
| **`a0_auto_update_daily.py`** | **（主控脚本）** 一键执行下载、回测、分析全流程。 | `--incremental` (默认), `--full`, `[START_DATE END_DATE]` |
| **`a1_data_fetcher.py`** | **（数据下载）** 从 AKShare 获取 A 股日线数据。 | `--today`, `--full`, `--date START END` |

### B. 策略回测 (支持增量模式)

| 脚本文件 | 主要作用 | 核心参数 |
| :--- | :--- | :--- |
| **`a2_single_strategy_backtest.py`** | **（单指标回测）** 回测六脉神剑、买卖点、缠论等。 | `--incremental` (默认), `--full`, `--strategy` |
| **`a3_combo_strategy_backtest.py`** | **（组合策略回测）** 回测稳健型和激进型组合策略。 | `--incremental` (默认), `--full`, `--strategy` |

### C. 信号分析与报告

| 脚本文件 | 主要作用 | 核心参数 |
| :--- | :--- | :--- |
| **`a4_signal_success_scanner.py`** | 扫描全市场，识别并验证成功信号案例。 | `--limit` |
| **`a21_pattern_analyzer.py`** | 对成功案例进行多维度指标共性分析。 | `--limit` |
| **`a5_generate_stock_reports.py`** | 生成个股详细报告及买卖点可视化数据。 | `--stock`, `--limit` |

### D. 核心支撑模块

| 脚本文件 | 主要作用 |
| :--- | :--- |
| **`a99_indicators.py`** | **（指标库）** 定义了所有技术指标的计算公式。 |
| **`a99_backtest_utils.py`** | **（工具库）** 提供多进程、增量过滤、结果汇总等功能。 |
| **`a99_logger.py`** | **（日志库）** 提供统一的时间戳日志打印功能。 |

---

## 4. 常用执行示例

### 1. 每日收盘后自动更新
```bash
python3 a0_auto_update_daily.py
```

### 2. 修复/补充特定时间段数据并回测
```bash
python3 a0_auto_update_daily.py 20250101 20250404
```

### 3. 修改策略逻辑后进行全量重算
```bash
python3 a0_auto_update_daily.py --full
```

---

## 5. 注意事项
- **执行环境**：请确保在 `py_file/` 目录下执行脚本，或确保 `PYTHONPATH` 包含该目录。
- **依赖库**：运行前请确保已安装 `akshare`, `pandas`, `numpy`, `psutil` 等依赖。
- **数据存储**：数据默认存储在项目根目录的 `data/day/` 下，回测状态记录在 `status/` 下。
