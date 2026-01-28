#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: a99_csv_chunker.py
功能描述: CSV 文件切片与合并工具
================================================================================
用于将大型 CSV 文件切片为多个小文件，并在读取时自动合并。
主要用于 all_signal_records.csv 文件的切片处理。
"""

import os
import pandas as pd
from typing import List, Optional


def split_csv_by_size(
    input_path: str,
    output_dir: str,
    base_name: str,
    max_size_mb: int = 40
) -> List[str]:
    """
    将 CSV 文件按大小切片。
    
    Args:
        input_path: 输入 CSV 文件路径
        output_dir: 输出目录
        base_name: 输出文件基础名称（不含扩展名）
        max_size_mb: 每个切片的最大大小（MB）
    
    Returns:
        切片文件路径列表
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始文件
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    total_rows = len(df)
    
    if total_rows == 0:
        return []
    
    # 估算每个切片的行数
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    rows_per_chunk = int((max_size_mb / file_size_mb) * total_rows)
    
    if rows_per_chunk >= total_rows:
        # 文件本身小于限制，不需要切片
        output_path = os.path.join(output_dir, f"{base_name}.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        return [output_path]
    
    # 切片处理
    chunk_files = []
    chunk_idx = 0
    
    for start_idx in range(0, total_rows, rows_per_chunk):
        end_idx = min(start_idx + rows_per_chunk, total_rows)
        chunk_df = df.iloc[start_idx:end_idx]
        
        chunk_file = os.path.join(output_dir, f"{base_name}_part{chunk_idx:03d}.csv")
        chunk_df.to_csv(chunk_file, index=False, encoding='utf-8-sig')
        chunk_files.append(chunk_file)
        chunk_idx += 1
    
    return chunk_files


def merge_csv_chunks(
    chunk_dir: str,
    base_name: str
) -> Optional[pd.DataFrame]:
    """
    合并 CSV 切片文件。
    
    Args:
        chunk_dir: 切片文件所在目录
        base_name: 文件基础名称（不含扩展名）
    
    Returns:
        合并后的 DataFrame，如果没有找到文件则返回 None
    """
    if not os.path.exists(chunk_dir):
        return None
    
    # 查找所有切片文件
    chunk_files = []
    
    # 先尝试找单个文件（未切片的情况）
    single_file = os.path.join(chunk_dir, f"{base_name}.csv")
    if os.path.exists(single_file):
        return pd.read_csv(single_file, encoding='utf-8-sig')
    
    # 查找切片文件
    for f in sorted(os.listdir(chunk_dir)):
        if f.startswith(f"{base_name}_part") and f.endswith('.csv'):
            chunk_files.append(os.path.join(chunk_dir, f))
    
    if not chunk_files:
        return None
    
    # 合并所有切片
    dfs = []
    for chunk_file in chunk_files:
        df = pd.read_csv(chunk_file, encoding='utf-8-sig')
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def cleanup_old_chunks(chunk_dir: str, base_name: str):
    """
    清理旧的切片文件。
    
    Args:
        chunk_dir: 切片文件所在目录
        base_name: 文件基础名称（不含扩展名）
    """
    if not os.path.exists(chunk_dir):
        return
    
    # 删除单个文件
    single_file = os.path.join(chunk_dir, f"{base_name}.csv")
    if os.path.exists(single_file):
        os.remove(single_file)
    
    # 删除所有切片文件
    for f in os.listdir(chunk_dir):
        if f.startswith(f"{base_name}_part") and f.endswith('.csv'):
            os.remove(os.path.join(chunk_dir, f))
