# TDX Strategy Backtest - 通达信策略回测与分析系统

这是一个基于 Python 和 React 构建的专业级股票策略回测与可视化分析系统。系统旨在帮助投资者通过量化手段验证通达信指标策略的有效性，并提供深度的模式归因分析。

## 🚀 核心功能

- **全自动数据流**：集成 `mootdx` 接口，支持沪深京全市场 A 股数据的全量下载与增量更新。
- **统一回测引擎**：支持单指标、多指标组合回测，自动计算胜率、盈亏比、最大回撤等核心指标。
- **深度模式分析**：对成功案例进行“回看归因”，分析信号触发时的成交量、均线偏离、波动率等共性特征。
- **智能预测评分**：内置仪表盘评分系统与基于线性回归的短期趋势预测模型。
- **现代化可视化**：基于 React + TailwindCSS + ShadcnUI 构建的响应式前端界面，直观展示回测报表与分析结果。

## 📂 项目结构

```text
├── py_file/                # Python 核心脚本
│   ├── a0_auto_update_daily.py    # 每日自动更新主控脚本
│   ├── a1_data_fetcher_mootdx.py  # 数据抓取模块
│   ├── a2_unified_backtest.py     # 统一回测与数据同步引擎
│   ├── a21_pattern_analyzer.py    # 模式特征分析引擎
│   └── a5_unified_analysis.py     # 综合报表与预测引擎
├── data/                   # 股票基础数据 (CSV)
├── report/                 # 回测与分析生成的本地报表
├── web/                    # 前端可视化项目 (React)
└── docs/                   # 项目详细文档
```

## 🛠️ 快速开始

### 1. 环境准备
确保已安装 Python 3.10+ 和 Node.js 18+。

```bash
# 安装 Python 依赖
pip install pandas numpy mootdx scikit-learn

# 安装前端依赖
cd web/client
pnpm install
```

### 2. 一键更新数据
运行主控脚本，完成从数据下载到分析的全流程：

```bash
python3 py_file/a0_auto_update_daily.py --full  # 首次运行使用全量模式
```

### 3. 启动可视化界面
```bash
cd web/client
pnpm dev
```

## 📖 详细文档

- [使用手册](./使用手册.md)：如何配置策略与查看报表。
- [项目部署手册](./项目部署手册.md)：本地及云端部署指南。
- [开发说明文档](./开发说明文档.md)：系统架构与二次开发指南。

## ⚖️ 免责声明
本系统仅供量化研究与学习使用，不构成任何投资建议。股市有风险，入市需谨慎。
