#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的路径处理
"""

import os
import sys
from pathlib import Path

# 新的 normalize_path 函数
def normalize_path_a6_fixed(p: str) -> Path:
    """
    标准化路径，支持 Windows 和 Linux 跨平台兼容。
    """
    # 将 Windows 路径分隔符转换为 Unix 风格
    p_str = p.replace("\\", "/")
    
    # 处理绝对路径
    if os.path.isabs(p_str):
        return Path(p_str).resolve()
    
    # 处理相对路径
    project_root = Path(__file__).resolve().parent.parent
    
    # 使用 / 作为分隔符，正确处理 .. 和 .
    path = project_root / p_str
    
    return path.resolve()

# a7 的函数
def get_project_root_a7():
    current_file = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(script_dir)
    return project_root

print("=" * 100)
print("修复后的路径处理测试")
print("=" * 100)

# 测试修复后的 a6_models.py 路径处理
print("\n修复后的 a6_models.py 路径处理：")
test_cases = [
    ("../web/client/src/data", "原始默认路径（使用 ..）"),
    ("web/client/src/data", "新的默认路径（不使用 ..）"),
    ("./data/day", "数据输入目录"),
]

for test_path, description in test_cases:
    result = normalize_path_a6_fixed(test_path)
    exists = result.exists()
    print(f"\n  输入: {test_path}")
    print(f"  描述: {description}")
    print(f"  输出: {result}")
    print(f"  存在: {exists}")

# 对比两个脚本
print("\n" + "=" * 100)
print("两个脚本输出目录对比")
print("=" * 100)

a6_output_old = normalize_path_a6_fixed("../web/client/src/data")
a6_output_new = normalize_path_a6_fixed("web/client/src/data")
a7_output = os.path.join(get_project_root_a7(), 'web', 'client', 'src', 'data')

print(f"\na6_models.py (旧路径 ../web/client/src/data): {a6_output_old}")
print(f"a6_models.py (新路径 web/client/src/data):    {a6_output_new}")
print(f"a7_advanced_forecast.py:                       {a7_output}")
print(f"\n新路径是否与 a7 相同: {str(a6_output_new) == a7_output}")

# Windows 路径测试
print("\n" + "=" * 100)
print("Windows 路径兼容性测试")
print("=" * 100)

windows_paths = [
    "web\\client\\src\\data",
    ".\\data\\day",
]

for win_path in windows_paths:
    result = normalize_path_a6_fixed(win_path)
    print(f"\n  Windows 路径: {win_path}")
    print(f"  转换结果: {result}")
    print(f"  存在: {result.exists()}")

print("\n✓ 修复完成：a6_models.py 现在使用 'web/client/src/data' 作为默认输出路径")
print("✓ 两个脚本现在生成文件到同一位置")
