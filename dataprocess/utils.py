import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from config import TIME_CONFIG
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def align_timestamps(dfs: List[pd.DataFrame], 
                    timestamp_col: str = 'timestamp',
                    method: str = 'intersection') -> List[pd.DataFrame]:
    """
    对齐多个DataFrame的时间戳
    
    Args:
        dfs: DataFrame列表
        timestamp_col: 时间戳列名
        method: 对齐方法，'intersection'或'union'
    
    Returns:
        对齐后的DataFrame列表
    """
    if not dfs:
        return []
    
    # 确保时间戳列为datetime类型
    for df in dfs:
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # 获取所有时间戳
    all_timestamps = []
    for df in dfs:
        if timestamp_col in df.columns:
            all_timestamps.extend(df[timestamp_col].tolist())
    
    # 根据方法确定目标时间戳
    if method == 'intersection':
        # 取交集
        target_timestamps = set(dfs[0][timestamp_col])
        for df in dfs[1:]:
            target_timestamps = target_timestamps.intersection(set(df[timestamp_col]))
        target_timestamps = sorted(list(target_timestamps))
    else:  # union
        # 取并集
        target_timestamps = sorted(list(set(all_timestamps)))
    
    # 对齐所有DataFrame
    aligned_dfs = []
    for df in dfs:
        if timestamp_col in df.columns:
            # 重新索引并填充缺失值
            df_aligned = df.set_index(timestamp_col).reindex(target_timestamps)
            df_aligned = df_aligned.reset_index()
            aligned_dfs.append(df_aligned)
        else:
            aligned_dfs.append(df)
    
    logger.info(f"时间戳对齐完成，目标时间点数量: {len(target_timestamps)}")
    return aligned_dfs

def resample_timeseries(df: pd.DataFrame, 
                       timestamp_col: str = 'timestamp',
                       interval: int = TIME_CONFIG['sampling_interval'],
                       agg_method: str = 'mean') -> pd.DataFrame:
    """
    重采样时间序列数据
    
    Args:
        df: 输入DataFrame
        timestamp_col: 时间戳列名
        interval: 采样间隔（秒）
        agg_method: 聚合方法
    
    Returns:
        重采样后的DataFrame
    """
    # 只保留数值型和时间戳列
    keep_cols = [timestamp_col] + list(df.select_dtypes(include=[np.number]).columns)
    keep_cols = list(dict.fromkeys(keep_cols))  # 去重
    df = df[keep_cols].copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    rule = f"{interval}s"
    df_resampled = df.resample(rule).mean()
    df_resampled = df_resampled.reset_index()
    return df_resampled

def z_score_normalize(data: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, Dict]:
    """
    Z-score标准化
    
    Args:
        data: 输入数据
        eps: 防止除零的小数
    
    Returns:
        标准化后的数据和统计信息
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std = np.where(std < eps, eps, std)  # 防止除零
    
    normalized_data = (data - mean) / std
    
    stats = {
        'mean': mean,
        'std': std,
        'method': 'z_score'
    }
    
    return normalized_data, stats

def create_time_windows(timestamps: List[datetime], 
                       window_size: int = 60) -> List[Tuple[datetime, datetime]]:
    """
    创建时间窗口
    
    Args:
        timestamps: 时间戳列表
        window_size: 窗口大小（秒）
    
    Returns:
        时间窗口列表
    """
    if not timestamps:
        return []
    
    start_time = min(timestamps)
    end_time = max(timestamps)
    
    windows = []
    current_time = start_time
    
    while current_time < end_time:
        window_end = current_time + timedelta(seconds=window_size)
        windows.append((current_time, window_end))
        current_time = window_end
    
    return windows

def fill_missing_values(df: pd.DataFrame, 
                       method: str = 'forward',
                       fill_value: Any = 0) -> pd.DataFrame:
    """
    填充缺失值
    
    Args:
        df: 输入DataFrame
        method: 填充方法
        fill_value: 填充值
    
    Returns:
        填充后的DataFrame
    """
    if method == 'forward':
        return df.fillna(method='ffill').fillna(fill_value)
    elif method == 'backward':
        return df.fillna(method='bfill').fillna(fill_value)
    elif method == 'interpolate':
        return df.interpolate().fillna(fill_value)
    else:
        return df.fillna(fill_value)

def detect_outliers(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    使用Z-score检测异常值
    
    Args:
        data: 输入数据
        threshold: 阈值
    
    Returns:
        异常值掩码
    """
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return z_scores > threshold

def save_processed_data(data: Dict[str, Any], 
                       filepath: str,
                       format: str = 'pickle') -> None:
    """
    保存处理后的数据
    
    Args:
        data: 数据字典
        filepath: 保存路径
        format: 保存格式
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == 'pickle':
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'json':
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif format == 'numpy':
        np.save(filepath, data)
    
    logger.info(f"数据已保存到: {filepath}")

def load_processed_data(filepath: str, format: str = 'pickle') -> Any:
    """
    加载处理后的数据
    
    Args:
        filepath: 文件路径
        format: 文件格式
    
    Returns:
        加载的数据
    """
    if format == 'pickle':
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format == 'json':
        import json
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == 'numpy':
        return np.load(filepath, allow_pickle=True)
    
    logger.info(f"数据已从 {filepath} 加载")

def collect_gaia_file_paths(root_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    扫描GAIA数据集目录，收集所有日期下的log、metric、trace、groundtruth文件路径

    Args:
        root_dir: 数据集根目录（如 /home/fuxian/DataSet/new_GAIA）

    Returns:
        {
            'log': {'2021-07-01': [log文件路径, ...], ...},
            'metric': {'2021-07-01': [metric文件路径, ...], ...},
            'trace': {'2021-07-01': [trace文件路径, ...], ...},
            'groundtruth': {'2021-07-01': [groundtruth文件路径, ...], ...}
        }
    """
    data = {'log': {}, 'metric': {}, 'trace': {}, 'groundtruth': {}}
    for date_dir in sorted(os.listdir(root_dir)):
        date_path = os.path.join(root_dir, date_dir)
        if not os.path.isdir(date_path):
            continue
        # log
        log_dir = os.path.join(date_path, 'log')
        if os.path.isdir(log_dir):
            log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.csv') or f.endswith('.json') or f.endswith('.log') or f.endswith('.txt')]
            if log_files:
                data['log'][date_dir] = log_files
        # metric
        metric_dir = os.path.join(date_path, 'metric')
        if os.path.isdir(metric_dir):
            metric_files = [os.path.join(metric_dir, f) for f in os.listdir(metric_dir) if f.endswith('.csv') or f.endswith('.json') or f.endswith('.parquet')]
            if metric_files:
                data['metric'][date_dir] = metric_files
        # trace
        trace_dir = os.path.join(date_path, 'trace')
        if os.path.isdir(trace_dir):
            trace_files = [os.path.join(trace_dir, f) for f in os.listdir(trace_dir) if f.endswith('.csv') or f.endswith('.json')]
            if trace_files:
                data['trace'][date_dir] = trace_files
        # groundtruth
        gt_file = os.path.join(date_path, 'groundtruth.csv')
        if os.path.isfile(gt_file):
            data['groundtruth'][date_dir] = [gt_file]
    return data