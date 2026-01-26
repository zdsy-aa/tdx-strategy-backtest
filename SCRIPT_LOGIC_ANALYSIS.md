# 脚本逻辑分析与优化建议

**生成日期**：2026-01-26  
**版本**：v1.0

---

## 1. 脚本概览

本项目包含三个核心数据处理脚本，分别负责数据获取、策略信号生成和高级预测：

| 脚本名称 | 主要功能 | 输出路径 | 状态 |
| :--- | :--- | :--- | :--- |
| `a1_data_fetcher_mootdx.py` | 从通达信服务器获取日线数据 | `data/day/{market}/*.csv` | ✓ 已优化 |
| `a6_models.py` | 离线策略信号生成与仪表盘统计 | `web/client/src/data/` | ✓ 已优化 |
| `a7_advanced_forecast.py` | 高级机器学习预测分析 | `web/client/src/data/forecast_*` | ✓ 已优化 |

---

## 2. 脚本逻辑详解

### 2.1 a1_data_fetcher_mootdx.py（数据获取模块）

**核心逻辑**：
```
加载本地依赖 → 初始化 mootdx 客户端 → 按代码获取日线数据 → 保存为 CSV
```

**关键特性**：
- ✓ 自动加载 `external/` 目录下的源码依赖（httpx, tdxpy, mootdx）
- ✓ 支持 tenacity 重试机制（已集成）
- ✓ 自动识别市场（沪市 6/9 开头 → sh，其他 → sz）
- ✓ 统一列名处理

**依赖包状态**：
| 依赖 | 位置 | 状态 |
| :--- | :--- | :--- |
| httpx | `external/httpx/` | ✓ 已集成 |
| tdxpy | `external/tdxpy/` | ✓ 已集成 |
| tenacity | `external/tenacity/` | ✓ **新增** |
| mootdx | `external/mootdx/` | ✓ 已集成 |

**逻辑检查**：
- ✓ 模块加载路径正确
- ✓ 错误处理完善
- ✓ 无明显逻辑错误

**建议**：
1. 可考虑添加 `--retry-count` 参数控制重试次数
2. 建议添加数据验证（检查 NaN、异常值）

---

### 2.2 a6_models.py（策略信号生成）

**核心逻辑**：
```
读取 CSV → 标准化列名 → 计算三大策略信号 → 生成 JSON → 统计仪表盘
```

**关键特性**：
- ✓ 纯离线处理，无外部依赖
- ✓ 三策略槽（A/B/C）独立计算，最后融合评分
- ✓ 支持最近 N 天数据过滤（新增 `--dashboard-days` 参数）
- ✓ 输出路径正确指向 `web/client/src/data/`

**策略说明**：

#### Strategy A：趋势跟踪（MA交叉）
- 使用 5/10/20 日均线交叉判断趋势
- 最高分值：60 分
- 信号类型：`A_MA_CROSS`

#### Strategy B：缠论买点
- 一买、二买、三买、强二买等多层次信号
- 最高分值：80 分
- 信号类型：`B_CHAN_*`

#### Strategy C：六脉神剑与摇钱树
- 六脉神剑：MACD + KDJ + RSI + LWR + BBI + MTM 共振
- 黄金摇钱树：庄家线与散户线配合
- 最高分值：70 分
- 信号类型：`C_SIX_VEINS`, `C_MONEY_TREE`, `C_BANKER_CROSS`

**新增功能**：
```python
--dashboard-days N  # 仪表盘统计最近 N 天的数据（默认 3 天）
```

**逻辑检查**：
- ✓ 路径规范化处理完善
- ✓ JSON 序列化处理正确
- ✓ 三策略融合逻辑清晰
- ✓ 时间过滤逻辑正确

**潜在问题与改进**：

| 问题 | 当前状态 | 建议 |
| :--- | :--- | :--- |
| 最小数据要求 | 60 天 | ✓ 合理，足以计算高级指标 |
| 信号排序 | 按日期和分值 | ✓ 合理 |
| 仪表盘统计 | 支持天数限制 | ✓ 已实现 |
| 错误处理 | 静默跳过数据不足 | ⚠ 建议记录日志 |

**建议**：
1. ✓ 已实现：添加 `--dashboard-days` 参数（默认 3 天）
2. 考虑添加 `--min-bars` 参数控制最小数据要求
3. 建议添加信号强度权重调整参数

---

### 2.3 a7_advanced_forecast.py（高级预测模块）

**核心逻辑**：
```
加载 CSV → 数据清洗 → 卡尔曼滤波 → 粒子滤波 → HMM 状态分析 → 随机森林预测 → JSON 输出
```

**关键特性**：
- ✓ 多进程处理（加速）
- ✓ 四大高级模型：卡尔曼滤波、粒子滤波、HMM、随机森林
- ✓ 输出路径正确指向 `web/client/src/data/forecast_*`
- ✓ 异常值处理完善（处理 Inf、NaN）

**四大模型说明**：

| 模型 | 用途 | 输出字段 |
| :--- | :--- | :--- |
| 卡尔曼滤波 | 平滑价格曲线，减少噪声 | `kalman_price` |
| 粒子滤波 | 处理非高斯分布，预测下一时刻 | `particle_price` |
| HMM | 捕捉市场状态（牛/熊/震荡） | `market_state` |
| 随机森林 | 集成预测次日收盘价 | `forecast_change_pct` |

**逻辑检查**：
- ✓ 数据清洗逻辑完善
- ✓ 异常值处理正确（Inf/NaN 替换）
- ✓ 模型参数设置合理
- ✓ 多进程实现正确

**潜在问题与改进**：

| 问题 | 当前状态 | 建议 |
| :--- | :--- | :--- |
| 最小数据要求 | 10 条 | ✓ 合理 |
| 异常值处理 | 价格 > 0.01 | ✓ 合理 |
| HMM 状态数 | 3（牛/熊/震荡） | ✓ 合理 |
| 随机森林参数 | 默认 | ⚠ 建议参数化 |
| 错误处理 | 完善 | ✓ 已处理 |

**建议**：
1. 考虑添加 `--model-params` 参数支持自定义模型参数
2. 建议添加模型性能评估（R²、RMSE 等）
3. 考虑添加特征重要性分析

---

## 3. 数据流转与输出路径验证

### 3.1 输出路径确认

| 脚本 | 输出文件 | 路径 | 状态 |
| :--- | :--- | :--- | :--- |
| a6_models.py | dashboard.json | `web/client/src/data/` | ✓ 正确 |
| a6_models.py | symbols/*.json | `web/client/src/data/symbols/` | ✓ 正确 |
| a6_models.py | errors.json | `web/client/src/data/` | ✓ 正确 |
| a7_advanced_forecast.py | forecast_summary.json | `web/client/src/data/` | ✓ 正确 |
| a7_advanced_forecast.py | forecast_details/*.json | `web/client/src/data/forecast_details/` | ✓ 正确 |

### 3.2 路径规范化机制

**a6_models.py** 的 `normalize_path()` 函数：
```python
def normalize_path(p: str) -> Path:
    """
    1. 兼容 Windows 反斜杠
    2. 相对路径相对于项目根目录
    """
```
- ✓ 处理 Windows 路径
- ✓ 处理相对路径
- ✓ 返回绝对路径

**a7_advanced_forecast.py** 的 `get_project_root()` 函数：
```python
def get_project_root():
    """获取项目根目录"""
    current_file = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(script_dir)
    return project_root
```
- ✓ 正确获取项目根目录
- ✓ 兼容 Windows 和 Linux

---

## 4. 新增功能总结

### 4.1 仪表盘天数限制功能

**参数**：`--dashboard-days N`（默认 3）

**实现逻辑**：
```python
if args.dashboard_days > 0:
    cutoff_date = datetime.utcnow() - pd.Timedelta(days=args.dashboard_days)
    if last_date >= cutoff_date:
        # 将股票加入 top 列表
```

**使用示例**：
```bash
# 统计最近 3 天（默认）
python a6_models.py

# 统计最近 7 天
python a6_models.py --dashboard-days 7

# 统计所有数据
python a6_models.py --dashboard-days 0
```

---

## 5. 依赖包集成状态

### 5.1 已集成的依赖

| 包名 | 版本 | 位置 | 用途 |
| :--- | :--- | :--- | :--- |
| httpx | 最新 | `external/httpx/` | HTTP 客户端 |
| tdxpy | 最新 | `external/tdxpy/` | 通达信数据解析 |
| mootdx | 最新 | `external/mootdx/` | 行情接口 |
| tenacity | 最新 | `external/tenacity/` | **新增** 重试装饰器 |

### 5.2 tenacity 集成验证

```bash
$ cd /home/ubuntu/tdx-strategy-backtest
$ python3 -c "import sys; sys.path.insert(0, 'external'); from tenacity import retry, stop_after_attempt; print('✓ tenacity 集成成功')"
✓ tenacity 集成成功
```

---

## 6. 逻辑错误检查总结

### 6.1 a1_data_fetcher_mootdx.py

| 检查项 | 结果 | 说明 |
| :--- | :--- | :--- |
| 模块加载 | ✓ | 正确处理本地依赖 |
| 错误处理 | ✓ | 完善的异常捕获 |
| 数据保存 | ✓ | 正确的市场识别和路径处理 |
| 列名统一 | ✓ | 正确添加 `名称` 列 |

**无明显逻辑错误**

### 6.2 a6_models.py

| 检查项 | 结果 | 说明 |
| :--- | :--- | :--- |
| 路径规范化 | ✓ | 完善的跨平台处理 |
| 列名映射 | ✓ | 完整的中英文映射 |
| 策略融合 | ✓ | 清晰的三策略融合逻辑 |
| JSON 序列化 | ✓ | 完善的类型转换 |
| 仪表盘统计 | ✓ | 新增天数限制功能 |

**无明显逻辑错误**

### 6.3 a7_advanced_forecast.py

| 检查项 | 结果 | 说明 |
| :--- | :--- | :--- |
| 数据清洗 | ✓ | 完善的异常值处理 |
| 模型实现 | ✓ | 四大模型逻辑正确 |
| 多进程 | ✓ | 正确的进程池使用 |
| 输出路径 | ✓ | 正确的项目根目录获取 |

**无明显逻辑错误**

---

## 7. 建议与优化方向

### 7.1 短期优化（立即可实施）

1. ✓ **已完成**：集成 tenacity 重试库
2. ✓ **已完成**：添加仪表盘天数限制功能
3. 建议：为 a1_data_fetcher_mootdx.py 添加数据验证函数
4. 建议：为 a7_advanced_forecast.py 添加模型性能评估

### 7.2 中期优化（下一版本）

1. 参数化随机森林模型参数
2. 添加信号强度权重调整机制
3. 实现增量更新机制（避免重复处理）
4. 添加数据备份和恢复机制

### 7.3 长期优化（架构升级）

1. 考虑使用配置文件管理所有参数
2. 实现实时数据流处理（Kafka/Redis）
3. 添加模型版本管理和 A/B 测试框架
4. 建立完整的监控和告警系统

---

## 8. 文档更新清单

| 文档 | 更新内容 | 状态 |
| :--- | :--- | :--- |
| py_file/README.md | 添加 a6_models 和 a7_advanced_forecast 说明 | ⏳ 待更新 |
| 部署与使用手册.md | 添加依赖集成和参数说明 | ⏳ 待更新 |
| 开发说明文档.md | 添加新功能说明 | ⏳ 待更新 |

---

## 9. 总结

### 核心成果

1. ✓ **依赖集成**：成功集成 tenacity 到 external 目录
2. ✓ **功能增强**：为 a6_models.py 添加仪表盘天数限制功能
3. ✓ **路径验证**：确认所有脚本输出路径正确对应 web/client/src/data
4. ✓ **逻辑检查**：三个核心脚本均无明显逻辑错误

### 后续行动

1. 根据本文档更新相关 MD 文件
2. 在本地执行脚本进行验证
3. 提交更改到代码仓

---

**文档版本**：v1.0  
**最后更新**：2026-01-26  
**维护者**：AI Assistant
