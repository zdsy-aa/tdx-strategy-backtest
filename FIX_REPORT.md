# 修复报告 - tdx-strategy-backtest

**日期**: 2026-01-26  
**分支**: develop  
**提交**: a47a6c84

## 问题概述

在本次维护任务中，发现并修复了以下关键问题：

### 1. CROSS 函数索引对齐问题

**文件**: `py_file/a99_indicators.py`

**问题描述**:  
`CROSS(A, B)` 函数在处理标量与 Series 混合输入时，会出现 `ValueError: Can only compare identically-labeled Series objects` 或 `ValueError: Length of values does not match length of index` 错误。

**根本原因**:  
- `_ensure_series` 函数在将标量转换为 Series 时，没有正确处理参考索引的长度
- `CROSS` 函数使用 pandas 的 shift 和比较操作时，索引不一致导致比较失败

**修复方案**:
1. 修改 `_ensure_series` 函数，当输入是标量且有参考索引时，创建一个广播到整个索引长度的 Series
2. 修改 `CROSS` 函数，使用 numpy 数组进行比较操作，避免 pandas 索引对齐问题
3. 优先使用 Series 类型参数的索引作为参考索引

**修复代码**:
```python
def _ensure_series(x, reference_index=None) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if reference_index is not None:
        if np.isscalar(x):
            return pd.Series([x] * len(reference_index), index=reference_index)
        return pd.Series(x, index=reference_index)
    if np.isscalar(x):
        return pd.Series([x])
    return pd.Series(x)

def CROSS(a, b) -> pd.Series:
    # 确定参考索引：优先使用 Series 类型的参数索引
    a_is_series = isinstance(a, pd.Series)
    b_is_series = isinstance(b, pd.Series)
    
    if a_is_series:
        ref_index = a.index
    elif b_is_series:
        ref_index = b.index
    else:
        ref_index = pd.RangeIndex(1)
    
    a_series = _ensure_series(a, reference_index=ref_index)
    b_series = _ensure_series(b, reference_index=ref_index)
    
    # 使用 numpy 数组进行比较
    a_arr = a_series.values
    b_arr = b_series.values
    a_prev = np.roll(a_arr, 1)
    b_prev = np.roll(b_arr, 1)
    
    if len(a_arr) > 0:
        a_prev[0] = a_arr[0]
        b_prev[0] = b_arr[0]
        
    result = (a_prev < b_prev) & (a_arr >= b_arr)
    return pd.Series(result, index=ref_index)
```

### 2. a7 预测脚本错误处理不足

**文件**: `py_file/a7_advanced_forecast.py`

**问题描述**:  
当某个股票数据文件不足 20 条记录时，整个预测任务会失败。

**修复方案**:
1. 在 `process_task` 函数中增加 try-except 包装
2. 使用 `imap_unordered` 替代 `map`，提高容错性

**修复代码**:
```python
def process_task(csv_path):
    try:
        return AdvancedForecaster(csv_path).run()
    except Exception as e:
        code = Path(csv_path).stem
        return {"code": code, "error": str(e), "status": "failed"}
```

## 端到端验证结果

使用 5 只测试股票（100 天数据）进行验证：

| 脚本 | 状态 | 说明 |
|------|------|------|
| a1_data_fetcher_mootdx.py | ✅ 通过 | 成功下载 100 天数据 |
| a2_single_strategy_backtest.py | ✅ 通过 | 三个策略回测全部完成 |
| a6_models.py | ✅ 通过 | 生成 dashboard.json |
| a7_advanced_forecast.py | ✅ 通过 | 生成 forecast_summary.json |

## 测试股票列表

- sh600036 (招商银行)
- sh601318 (中国平安)
- sz000001 (平安银行)
- sz300750 (宁德时代)
- sz002594 (比亚迪)

## 后续建议

1. **数据完整性检查**: 建议在运行回测前检查数据文件的完整性
2. **依赖管理**: 建议将 `filterpy` 和 `hmmlearn` 添加到 requirements.txt
3. **单元测试**: 建议为核心指标函数添加单元测试

## 提交历史

```
commit a47a6c84
Author: Manus
Date:   2026-01-26

fix: 修复 CROSS 函数索引对齐问题和 a7 预测脚本错误处理

- a99_indicators.py: 修复 _ensure_series 函数，正确处理标量广播到 Series
- a99_indicators.py: 修复 CROSS 函数，使用 numpy 数组避免 pandas 索引不匹配错误
- a7_advanced_forecast.py: 增加 process_task 函数的异常处理，避免单个文件失败导致整体失败
- a7_advanced_forecast.py: 使用 imap_unordered 替代 map 提高容错性
```
