"""
指标数据处理器 - 处理时间序列指标数据
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from config import METRIC_CONFIG, TIME_CONFIG
from utils import align_timestamps, resample_timeseries, fill_missing_values, detect_outliers

logger = logging.getLogger(__name__)

class MetricProcessor:
    """指标数据处理器"""

    def __init__(self, config: Dict = None):
        self.config = config or METRIC_CONFIG
        self.metric_stats = {}
        self.instance_metrics = {}

    def load_metric_data(self, file_paths: List[str] or str, instance_id_col: str = 'instance_id') -> Dict[str, pd.DataFrame]:
        """
        加载指标数据，自动处理时间戳和实例ID
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        metric_data = {}

        for file_path in file_paths:
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path)
                elif file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                else:
                    logger.warning(f"不支持的文件格式: {file_path}")
                    continue

                # 时间戳自动识别
                if 'timestamp' in df.columns:
                    if np.issubdtype(df['timestamp'].dtype, np.number):
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    else:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                else:
                    logger.warning(f"缺少timestamp列: {file_path}")
                    continue

                # 实例ID自动识别
                if instance_id_col in df.columns:
                    for instance_id, group in df.groupby(instance_id_col):
                        if instance_id not in metric_data:
                            metric_data[instance_id] = []
                        metric_data[instance_id].append(group)
                else:
                    # 如果没有instance_id，假设每个文件代表一个实例
                    instance_id = file_path.split('/')[-1].split('.')[0]
                    metric_data[instance_id] = [df]

                logger.info(f"成功加载指标数据: {file_path}")

            except Exception as e:
                logger.error(f"加载指标数据失败 {file_path}: {str(e)}")

        # 合并同一实例的数据
        for instance_id in metric_data:
            metric_data[instance_id] = pd.concat(metric_data[instance_id], ignore_index=True)

        return metric_data

    def pivot_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将指标数据透视为宽格式
        """
        if 'metric_name' in df.columns and 'value' in df.columns:
            pivoted = df.pivot_table(
                index='timestamp',
                columns='metric_name',
                values='value',
                aggfunc='mean'
            ).reset_index()
            pivoted.columns.name = None
            return pivoted
        else:
            return df

    def align_metric_schemas(self, metric_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        对齐不同实例的指标模式
        """
        all_metrics = set()
        for df in metric_data.values():
            metric_cols = [col for col in df.columns if col != 'timestamp']
            all_metrics.update(metric_cols)
        all_metrics = sorted(list(all_metrics))

        if self.config['strategy'] == 'intersection':
            common_metrics = set(all_metrics)
            for df in metric_data.values():
                metric_cols = [col for col in df.columns if col != 'timestamp']
                common_metrics = common_metrics.intersection(set(metric_cols))
            target_metrics = ['timestamp'] + sorted(list(common_metrics))
            logger.info(f"使用指标交集策略，共 {len(common_metrics)} 个指标")
        else:
            target_metrics = ['timestamp'] + all_metrics
            logger.info(f"使用指标并集策略，共 {len(all_metrics)} 个指标")

        aligned_data = {}
        for instance_id, df in metric_data.items():
            aligned_df = df.copy()
            for metric in target_metrics:
                if metric not in aligned_df.columns:
                    aligned_df[metric] = self.config['fill_value']
            aligned_df = aligned_df[target_metrics]
            aligned_data[instance_id] = aligned_df

        return aligned_data

    def clean_metric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清理指标数据，包括异常值检测和缺失值填充
        """
        cleaned_df = df.copy()
        if 'timestamp' in cleaned_df.columns:
            cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
            cleaned_df = cleaned_df.sort_values('timestamp').reset_index(drop=True)

        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.config.get('outlier_threshold'):
                outlier_mask = detect_outliers(
                    cleaned_df[col].values,
                    self.config['outlier_threshold']
                )
                if outlier_mask.any():
                    logger.info(f"检测到 {col} 列中有 {outlier_mask.sum()} 个异常值")
                    median_val = cleaned_df[col].median()
                    cleaned_df.loc[outlier_mask, col] = median_val

        cleaned_df = fill_missing_values(cleaned_df, fill_value=self.config['fill_value'])
        return cleaned_df

    def resample_metrics(self, metric_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        重采样指标数据
        """
        resampled_data = {}
        for instance_id, df in metric_data.items():
            resampled_df = resample_timeseries(
                df,
                interval=TIME_CONFIG['sampling_interval'],
                agg_method='mean'
            )
            resampled_data[instance_id] = resampled_df
            logger.info(f"实例 {instance_id} 重采样完成")
        return resampled_data

    def process_metrics(self, file_paths: List[str] or str) -> Dict[str, pd.DataFrame]:
        """
        完整的指标数据处理流程
        """
        logger.info("开始处理指标数据...")

        metric_data = self.load_metric_data(file_paths)
        for instance_id in metric_data:
            metric_data[instance_id] = self.pivot_metrics(metric_data[instance_id])
        metric_data = self.align_metric_schemas(metric_data)
        for instance_id in metric_data:
            metric_data[instance_id] = self.clean_metric_data(metric_data[instance_id])
        metric_data = self.resample_metrics(metric_data)

        dfs = list(metric_data.values())
        aligned_dfs = align_timestamps(dfs, method='intersection')

        processed_data = {}
        instance_ids = list(metric_data.keys())
        for i, instance_id in enumerate(instance_ids):
            processed_data[instance_id] = aligned_dfs[i]

        self.instance_metrics = processed_data
        self._compute_statistics()
        logger.info(f"指标数据处理完成，共处理 {len(processed_data)} 个实例")
        return processed_data

    def _compute_statistics(self):
        """计算指标统计信息"""
        stats = {}
        for instance_id, df in self.instance_metrics.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            instance_stats = {}
            for col in numeric_cols:
                instance_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'count': df[col].count()
                }
            stats[instance_id] = instance_stats
        self.metric_stats = stats

    def get_feature_matrix(self, timestamp: str = None) -> Dict[str, np.ndarray]:
        """
        获取指定时间戳的特征矩阵
        """
        feature_matrices = {}
        for instance_id, df in self.instance_metrics.items():
            if timestamp:
                mask = df['timestamp'] == pd.to_datetime(timestamp)
                if mask.any():
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    features = df.loc[mask, numeric_cols].values.flatten()
                    feature_matrices[instance_id] = features
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                features = df[numeric_cols].values
                feature_matrices[instance_id] = features
        return feature_matrices
if __name__ == "__main__":
    from utils import collect_gaia_file_paths
    from config import DATA_PATHS
    import os
    import re
    import gc

    root_dir = DATA_PATHS['metric_data']
    output_dir = os.path.join(DATA_PATHS['output_data'], 'metric')
    os.makedirs(output_dir, exist_ok=True)

    file_dict = collect_gaia_file_paths(root_dir)
    all_metric_files = []
    for date, files in file_dict['metric'].items():
        all_metric_files.extend(files)

    print("将要处理的指标文件列表：")
    for f in all_metric_files:
        print(f)
    print(f"共计 {len(all_metric_files)} 个指标文件\n")

    # 分批处理，每次只处理一个文件，处理完就释放内存
    for metric_file in all_metric_files:
        print(f"\n正在处理: {metric_file}")
        processor = MetricProcessor()
        processed_data = processor.process_metrics([metric_file])
        print(f"本文件共处理实例数: {len(processed_data)}")

        # 从路径中提取日期（假设路径中有类似 2021-07-01 这样的日期）
        match = re.search(r'(\d{4}-\d{2}-\d{2})', metric_file)
        date_str = match.group(1) if match else "unknown_date"

        # 创建对应日期的文件夹
        date_output_dir = os.path.join(output_dir, date_str)
        os.makedirs(date_output_dir, exist_ok=True)

        for instance_id, df in processed_data.items():
            print(f"实例: {instance_id}, 特征shape: {df.shape}")
            out_path = os.path.join(
                date_output_dir,
                f"{instance_id}_metric_features.csv"
            )
            df.to_csv(out_path, index=False)
            print(f"已保存: {out_path}")
        # 主动释放内存
        del processed_data
        gc.collect()