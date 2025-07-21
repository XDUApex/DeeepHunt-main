"""
调用链数据处理器 - 处理分布式调用链数据
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from collections import defaultdict
from config import TRACE_CONFIG, TIME_CONFIG
from utils import resample_timeseries, create_time_windows

logger = logging.getLogger(__name__)

class TraceProcessor:
    """调用链数据处理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or TRACE_CONFIG
        self.trace_features = {}
        self.instance_trace_features = {}
        self.span_statistics = {}
    
    def load_trace_data(self, file_paths: List[str] or str) -> List[Dict]:
        """
        加载调用链数据
        
        Args:
            file_paths: 调用链数据文件路径列表或单个路径
        
        Returns:
            调用链span数据列表
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        all_spans = []
        
        for file_path in file_paths:
            try:
                if file_path.endswith('.json'):
                    import json
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # 处理不同的JSON格式
                    if isinstance(data, list):
                        spans = data
                    elif 'spans' in data:
                        spans = data['spans']
                    elif 'traces' in data:
                        # 从traces中提取spans
                        spans = []
                        for trace in data['traces']:
                            if 'spans' in trace:
                                spans.extend(trace['spans'])
                    else:
                        spans = [data]
                    
                    all_spans.extend(spans)
                    
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    spans = df.to_dict('records')
                    all_spans.extend(spans)
                
                logger.info(f"成功加载调用链数据: {file_path}，包含 {len(spans)} 个span")
                
            except Exception as e:
                logger.error(f"加载调用链数据失败 {file_path}: {str(e)}")
        
        logger.info(f"总计加载 {len(all_spans)} 个span")
        return all_spans
    
    def parse_span_data(self, spans: List[Dict]) -> pd.DataFrame:
        """
        解析span数据
        
        Args:
            spans: span数据列表
        
        Returns:
            解析后的DataFrame
        """
        parsed_spans = []
        
        for span in spans:
            try:
                parsed_span = self._extract_span_features(span)
                if parsed_span:
                    parsed_spans.append(parsed_span)
            except Exception as e:
                logger.warning(f"解析span失败: {str(e)}")
                continue
        
        if not parsed_spans:
            logger.warning("没有成功解析的span数据")
            return pd.DataFrame()
        
        df = pd.DataFrame(parsed_spans)
        logger.info(f"成功解析 {len(df)} 个span")
        return df
    
    def _extract_span_features(self, span: Dict) -> Optional[Dict]:
        """
        从单个span中提取特征
        
        Args:
            span: span数据
        
        Returns:
            提取的特征字典
        """
        # 标准化字段名（处理不同的命名约定）
        field_mappings = {
            'traceId': 'trace_id',
            'spanId': 'span_id',
            'parentSpanId': 'parent_span_id',
            'operationName': 'operation_name',
            'serviceName': 'service_name',
            'startTime': 'start_time',
            'endTime': 'end_time',
            'duration': 'duration',
            'statusCode': 'status_code',
            'tags': 'tags',
            'process': 'process'
        }
        
        normalized_span = {}
        for old_key, new_key in field_mappings.items():
            if old_key in span:
                normalized_span[new_key] = span[old_key]
            elif new_key in span:
                normalized_span[new_key] = span[new_key]
        
        # 提取基本信息
        try:
            # 时间信息
            start_time = self._parse_timestamp(normalized_span.get('start_time'))
            end_time = self._parse_timestamp(normalized_span.get('end_time'))
            
            if not start_time:
                return None
            
            # 计算持续时间（如果没有提供）
            duration = normalized_span.get('duration')
            if duration is None and end_time:
                duration = (end_time - start_time).total_seconds() * 1000  # 转换为毫秒
            
            # 提取服务信息
            service_name = self._extract_service_name(normalized_span)
            instance_id = self._extract_instance_id(normalized_span)
            
            # 提取状态信息
            status_code = self._extract_status_code(normalized_span)
            is_error = self._is_error_span(normalized_span)
            
            # 提取请求类型
            request_type = self._extract_request_type(normalized_span)
            
            return {
                'timestamp': start_time,
                'trace_id': normalized_span.get('trace_id'),
                'span_id': normalized_span.get('span_id'),
                'parent_span_id': normalized_span.get('parent_span_id'),
                'service_name': service_name,
                'instance_id': instance_id,
                'operation_name': normalized_span.get('operation_name', 'unknown'),
                'duration': duration or 0,
                'status_code': status_code,
                'is_error': is_error,
                'request_type': request_type
            }
            
        except Exception as e:
            logger.warning(f"提取span特征失败: {str(e)}")
            return None
    
    def _parse_timestamp(self, timestamp) -> Optional[datetime]:
        """解析时间戳"""
        if not timestamp:
            return None
        
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, (int, float)):
            # 假设是Unix时间戳
            if timestamp > 1e12:  # 毫秒级时间戳
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp)
        
        if isinstance(timestamp, str):
            # 尝试解析字符串时间戳
            try:
                return pd.to_datetime(timestamp)
            except:
                return None
        
        return None
    
    def _extract_service_name(self, span: Dict) -> str:
        """提取服务名"""
        # 尝试多个可能的字段
        for field in ['service_name', 'serviceName', 'service']:
            if field in span and span[field]:
                return str(span[field])
        
        # 从process信息中提取
        if 'process' in span and isinstance(span['process'], dict):
            process = span['process']
            if 'serviceName' in process:
                return str(process['serviceName'])
            if 'tags' in process:
                tags = process['tags']
                if isinstance(tags, dict) and 'service.name' in tags:
                    return str(tags['service.name'])
        
        # 从tags中提取
        if 'tags' in span and isinstance(span['tags'], dict):
            tags = span['tags']
            for key in ['service.name', 'service', 'component']:
                if key in tags:
                    return str(tags[key])
        
        return 'unknown'
    
    def _extract_instance_id(self, span: Dict) -> str:
        """提取实例ID"""
        # 尝试多个可能的字段
        for field in ['instance_id', 'instanceId', 'host', 'hostname']:
            if field in span and span[field]:
                return str(span[field])
        
        # 从process信息中提取
        if 'process' in span and isinstance(span['process'], dict):
            process = span['process']
            if 'tags' in process and isinstance(process['tags'], dict):
                tags = process['tags']
                for key in ['hostname', 'host.name', 'instance_id']:
                    if key in tags:
                        return str(tags[key])
        
        # 从tags中提取
        if 'tags' in span and isinstance(span['tags'], dict):
            tags = span['tags']
            for key in ['hostname', 'host.name', 'instance_id', 'pod.name']:
                if key in tags:
                    return str(tags[key])
        
        # 使用服务名作为fallback
        service_name = self._extract_service_name(span)
        return f"{service_name}_instance"
    
    def _extract_status_code(self, span: Dict) -> str:
        """提取状态码"""
        # 直接字段
        if 'status_code' in span:
            return str(span['status_code'])
        
        # 从tags中提取
        if 'tags' in span and isinstance(span['tags'], dict):
            tags = span['tags']
            for key in ['http.status_code', 'status_code', 'response.status']:
                if key in tags:
                    return str(tags[key])
        
        # 从状态对象中提取
        if 'status' in span and isinstance(span['status'], dict):
            if 'code' in span['status']:
                return str(span['status']['code'])
        
        return '200'  # 默认成功状态
    
    def _is_error_span(self, span: Dict) -> bool:
        """判断是否为错误span"""
        # 检查错误标志
        if 'error' in span:
            return bool(span['error'])
        
        # 检查状态码
        status_code = self._extract_status_code(span)
        if status_code.startswith('4') or status_code.startswith('5'):
            return True
        
        # 检查tags中的错误标志
        if 'tags' in span and isinstance(span['tags'], dict):
            tags = span['tags']
            if 'error' in tags and tags['error']:
                return True
        
        return False
    
    def _extract_request_type(self, span: Dict) -> str:
        """提取请求类型"""
        operation_name = span.get('operation_name', '').lower()
        
        # 从操作名推断
        if 'http' in operation_name or 'get' in operation_name or 'post' in operation_name:
            return 'http'
        elif 'rpc' in operation_name or 'grpc' in operation_name:
            return 'rpc'
        elif 'db' in operation_name or 'sql' in operation_name or 'database' in operation_name:
            return 'db'
        
        # 从tags推断
        if 'tags' in span and isinstance(span['tags'], dict):
            tags = span['tags']
            if 'component' in tags:
                component = tags['component'].lower()
                if 'http' in component:
                    return 'http'
                elif 'grpc' in component or 'rpc' in component:
                    return 'rpc'
                elif 'sql' in component or 'db' in component:
                    return 'db'
        
        return 'unknown'
    
    def compute_trace_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        计算调用链特征
        
        Args:
            df: 解析后的span数据
        
        Returns:
            按实例分组的特征数据
        """
        instance_features = {}
        
        # 按实例分组处理
        for instance_id, group in df.groupby('instance_id'):
            logger.info(f"计算实例 {instance_id} 的调用链特征...")
            
            # 创建时间窗口
            group['time_window'] = group['timestamp'].dt.floor(f'{TIME_CONFIG["sampling_interval"]}S')
            
            # 计算各种特征
            features_list = []
            
            for time_window, window_group in group.groupby('time_window'):
                features = self._compute_window_features(window_group)
                features['timestamp'] = time_window
                features_list.append(features)
            
            if features_list:
                features_df = pd.DataFrame(features_list)
                instance_features[instance_id] = features_df
        
        return instance_features
    
    def _compute_window_features(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算时间窗口内的特征
        
        Args:
            window_data: 时间窗口内的span数据
        
        Returns:
            特征字典
        """
        features = {}
        
        # 基本统计特征
        features['request_count'] = len(window_data)
        features['error_count'] = window_data['is_error'].sum()
        features['error_rate'] = features['error_count'] / features['request_count'] if features['request_count'] > 0 else 0
        
        # 延迟特征
        if 'duration' in window_data.columns and len(window_data) > 0:
            durations = window_data['duration'].dropna()
            if len(durations) > 0:
                features['avg_latency'] = durations.mean()
                features['max_latency'] = durations.max()
                features['min_latency'] = durations.min()
                
                # 百分位数
                for p in self.config['latency_percentiles']:
                    features[f'p{p}_latency'] = durations.quantile(p/100)
        
        # 状态码统计
        status_counts = window_data['status_code'].value_counts()
        for status_code in self.config['status_codes']:
            features[f'status_{status_code}_count'] = status_counts.get(status_code, 0)
        
        # 请求类型统计
        type_counts = window_data['request_type'].value_counts()
        for req_type in self.config['request_types']:
            features[f'{req_type}_request_count'] = type_counts.get(req_type, 0)
        
        # 唯一trace数量
        features['unique_trace_count'] = window_data['trace_id'].nunique()
        
        # 服务调用关系特征
        features['unique_service_count'] = window_data['service_name'].nunique()
        
        return features
    
    def process_traces(self, file_paths: List[str] or str) -> Dict[str, pd.DataFrame]:
        """
        完整的调用链数据处理流程
        
        Args:
            file_paths: 调用链文件路径
        
        Returns:
            处理后的调用链特征数据
        """
        logger.info("开始处理调用链数据...")
        
        # 1. 加载调用链数据
        spans = self.load_trace_data(file_paths)
        
        if not spans:
            logger.warning("没有找到调用链数据")
            return {}
        
        # 2. 解析span数据
        df = self.parse_span_data(spans)
        
        if df.empty:
            logger.warning("没有成功解析的span数据")
            return {}
        
        # 3. 计算特征
        trace_features = self.compute_trace_features(df)
        
        # 4. 时间对齐
        if trace_features:
            from utils import align_timestamps
            dfs = list(trace_features.values())
            aligned_dfs = align_timestamps(dfs, method='intersection')
            
            # 重新组织数据
            processed_data = {}
            instance_ids = list(trace_features.keys())
            for i, instance_id in enumerate(instance_ids):
                processed_data[instance_id] = aligned_dfs[i]
        else:
            processed_data = {}
        
        # 保存处理结果
        self.instance_trace_features = processed_data
        self._compute_trace_statistics()
        
        logger.info(f"调用链数据处理完成，共处理 {len(processed_data)} 个实例")
        return processed_data
    
    def _compute_trace_statistics(self):
        """计算调用链统计信息"""
        stats = {}
        
        for instance_id, df in self.instance_trace_features.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            instance_stats = {}
            
            for col in numeric_cols:
                instance_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'max': df[col].max(),
                    'min': df[col].min(),
                    'sum': df[col].sum()
                }
            
            stats[instance_id] = instance_stats
        
        self.span_statistics = stats
    
    def get_feature_matrix(self, timestamp: str = None) -> Dict[str, np.ndarray]:
        """
        获取指定时间戳的调用链特征矩阵
        
        Args:
            timestamp: 目标时间戳，如果为None则返回所有时间戳
        
        Returns:
            特征矩阵字典
        """
        feature_matrices = {}
        
        for instance_id, df in self.instance_trace_features.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if timestamp:
                # 获取特定时间戳的特征
                mask = df['timestamp'] == pd.to_datetime(timestamp)
                if mask.any():
                    features = df.loc[mask, numeric_cols].values.flatten()
                    feature_matrices[instance_id] = features
            else:
                # 获取所有时间戳的特征
                features = df[numeric_cols].values
                feature_matrices[instance_id] = features
        
        return feature_matrices