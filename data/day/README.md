# 日线数据目录 (data/day/)

> 存储股票日线历史数据，用于回测分析和信号验证。

## 📁 目录结构

```
data/day/
├── README.md                    # 本说明文件
├── hs300_daily.csv              # 沪深300指数日线数据
├── sz50_daily.csv               # 上证50指数日线数据
├── cyb_daily.csv                # 创业板指数日线数据
└── stocks/                      # 个股数据（可选）
    ├── 600519_daily.csv         # 贵州茅台
    ├── 000001_daily.csv         # 平安银行
    └── ...
```

## 📊 数据格式

所有 CSV 文件采用统一格式：

| 字段名 | 类型 | 说明 |
|-------|------|------|
| date | string | 日期，格式：YYYY-MM-DD |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| volume | int | 成交量 |
| amount | float | 成交额（可选） |

### 示例数据
```csv
date,open,high,low,close,volume
2024-01-02,3500.12,3520.45,3480.33,3510.88,15234567890
2024-01-03,3512.00,3545.67,3505.12,3540.23,16789012345
```

## 🔄 数据更新

### 自动更新
系统配置了定时任务，每个交易日 15:30 自动下载当天数据：
```bash
# crontab 配置
30 15 * * 1-5 cd /path/to/py_file && python scheduled_data_update.py
```

### 手动更新
```bash
cd ../py_file
python data_fetcher.py
```

## 📥 数据来源

- **主要来源**：AKShare（免费开源）
- **备用来源**：Tushare、Yahoo Finance
- **数据范围**：2016-01-01 至今（约 10 年）

## ⚠️ 注意事项

1. **数据完整性**：确保数据无缺失，否则会影响指标计算
2. **复权处理**：默认使用前复权数据
3. **存储空间**：10 年日线数据约占用 50-100MB
4. **更新频率**：建议每日收盘后更新

## 🔧 相关脚本

| 脚本 | 功能 |
|------|------|
| `py_file/data_fetcher.py` | 批量下载历史数据 |
| `py_file/scheduled_data_update.py` | 定时增量更新 |
