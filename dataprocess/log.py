"""
日志数据处理器 - 提取日志模板并统计频次
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
import logging
from datetime import datetime
from config import LOG_CONFIG, TIME_CONFIG
from utils import resample_timeseries, create_time_windows

logger = logging.getLogger(__name__)

class DrainLogParser:
    """基于Drain算法的日志解析器"""
    
    def __init__(self, sim_th=0.4, depth=4):
        self.sim_th = sim_th
        self.depth = depth
        self.log_clusters = []
        self.templates = {}
    
    def parse_logs(self, log_messages: List[str]) -> List[int]:
        """
        解析日志消息，返回模板ID
        
        Args:
            log_messages: 日志消息列表
        
        Returns:
            模板ID列表
        """
        template_ids = []
        
        for message in log_messages:
            template_id = self._match_template(message)
            template_ids.append(template_id)
        
        return template_ids
    
    def _match_template(self, message: str) -> int:
        """匹配或创建新的日志模板"""
        # 预处理消息
        tokens = self._preprocess_message(message)
        
        # 查找匹配的模板
        best_match_id = -1
        best_similarity = 0
        
        for template_id, template_tokens in self.templates.items():
            similarity = self._calculate_similarity(tokens, template_tokens)
            if similarity > best_similarity and similarity >= self.sim_th:
                best_similarity = similarity
                best_match_id = template_id
        
        if best_match_id != -1:
            return best_match_id
        else:
            # 创建新模板
            new_template_id = len(self.templates)
            self.templates[new_template_id] = tokens
            return new_template_id
    
    def _preprocess_message(self, message: str) -> List[str]:
        """预处理日志消息"""
        # 移除时间戳、IP地址、数字等变量部分
        message = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '<TIMESTAMP>', message)
        message = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', message)
        message = re.sub(r'\d+', '<NUM>', message)
        message = re.sub(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', '<UUID>', message)
        
        # 分词
        tokens = message.split()
        return tokens
    
    def _calculate_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """计算两个token序列的相似度"""
        if len(tokens1) != len(tokens2):
            return 0.0
        
        matches = sum(1 for t1, t2 in zip(tokens1, tokens2) if t1 == t2)
        return matches / len(tokens1) if tokens1 else 0.0
    
    def get_template_string(self, template_id: int) -> str:
        """获取模板字符串"""
        if template_id in self.templates:
            return ' '.join(self.templates[template_id])
        return f"TEMPLATE_{template_id}"

class LogProcessor:
    """日志数据处理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or LOG_CONFIG
        self.log_parser = DrainLogParser(
            sim_th=self.config['drain_config']['sim_th'],
            depth=self.config['drain_config']['depth']
        )
        self.template_frequencies = {}
        self.instance_log_features = {}
    
    def load_log_data(self, file_paths: List[str] or str) -> Dict[str, pd.DataFrame]:
        """
        加载日志数据，并按 service+ip 分组为实例
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        log_data = {}

        for file_path in file_paths:
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    # 修正：如果timestamp是float或int，按Unix时间戳解析
                    if 'timestamp' in df.columns and np.issubdtype(df['timestamp'].dtype, np.number):
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path)
                else:
                    # 假设是纯文本日志文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    parsed_logs = []
                    for line in lines:
                        parsed_log = self._parse_log_line(line.strip())
                        if parsed_log:
                            parsed_logs.append(parsed_log)
                    df = pd.DataFrame(parsed_logs)

                # 确保必要的列存在
                if 'timestamp' not in df.columns or 'message' not in df.columns:
                    logger.warning(f"日志文件缺少必要列: {file_path}")
                    continue

                # 从 message 字段中提取 IP
                df['ip'] = df['message'].str.extract(r'\|\s*([\d\.]+)\s*\|')
                # 组合 service 和 ip 作为实例ID
                df['instance_id'] = df['service'].astype(str)
                if df['ip'].notnull().any():
                    df['instance_id'] = df['service'].astype(str) + '_' + df['ip'].fillna('NA')

                # 按 instance_id 分组
                for instance_id, group in df.groupby('instance_id'):
                    if instance_id not in log_data:
                        log_data[instance_id] = []
                    log_data[instance_id].append(group)

                logger.info(f"成功加载日志数据: {file_path}")

            except Exception as e:
                logger.error(f"加载日志数据失败 {file_path}: {str(e)}")

        # 合并同一实例的数据
        for instance_id in log_data:
            if len(log_data[instance_id]) > 1:
                log_data[instance_id] = pd.concat(log_data[instance_id], ignore_index=True)
            else:
                log_data[instance_id] = log_data[instance_id][0]

        return log_data
    
    def _parse_log_line(self, line: str) -> Optional[Dict]:
        """
        解析单行日志
        
        Args:
            line: 日志行
        
        Returns:
            解析后的日志字典
        """
        if not line.strip():
            return None
        
        # 尝试提取时间戳 (支持多种格式)
        timestamp_patterns = [
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
            r'(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})',
            r'(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})',
            r'(\w{3} \d{1,2} \d{2}:\d{2}:\d{2})',  # 如 "Jan 15 10:30:45"
        ]
        
        timestamp = None
        message = line
        
        for pattern in timestamp_patterns:
            match = re.search(pattern, line)
            if match:
                timestamp_str = match.group(1)
                try:
                    if pattern == timestamp_patterns[0]:  # YYYY-MM-DD HH:MM:SS
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    elif pattern == timestamp_patterns[1]:  # YYYY/MM/DD HH:MM:SS
                        timestamp = datetime.strptime(timestamp_str, '%Y/%m/%d %H:%M:%S')
                    elif pattern == timestamp_patterns[2]:  # MM-DD-YYYY HH:MM:SS
                        timestamp = datetime.strptime(timestamp_str, '%m-%d-%Y %H:%M:%S')
                    # 对于其他格式，可以添加更多解析逻辑
                    
                    # 移除时间戳，获取消息部分
                    message = line.replace(timestamp_str, '').strip()
                    break
                except:
                    continue
        
        if not timestamp:
            # 如果无法提取时间戳，使用当前时间
            timestamp = datetime.now()
        
        return {
            'timestamp': timestamp,
            'message': message,
            'raw_log': line
        }
    
    def extract_log_templates(self, log_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        提取日志模板
        
        Args:
            log_data: 日志数据字典
        
        Returns:
            包含模板信息的数据字典
        """
        processed_data = {}
        
        for instance_id, df in log_data.items():
            logger.info(f"处理实例 {instance_id} 的日志模板...")
            
            # 解析日志获取模板ID
            messages = df['message'].tolist()
            template_ids = self.log_parser.parse_logs(messages)
            
            # 添加模板ID到数据框
            df_processed = df.copy()
            df_processed['template_id'] = template_ids
            
            processed_data[instance_id] = df_processed
            
            logger.info(f"实例 {instance_id} 识别出 {len(set(template_ids))} 个日志模板")
        
        return processed_data
    
    def compute_template_frequencies(self, log_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        计算日志模板频次
        
        Args:
            log_data: 包含模板ID的日志数据
        
        Returns:
            模板频次时间序列数据
        """
        frequency_data = {}
        
        for instance_id, df in log_data.items():
            logger.info(f"计算实例 {instance_id} 的模板频次...")
            
            # 确保时间戳为datetime类型
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 按时间窗口和模板ID统计频次
            df['time_window'] = df['timestamp'].dt.floor(f'{TIME_CONFIG["sampling_interval"]}S')
            
            # 统计每个时间窗口内每个模板的出现次数
            template_counts = df.groupby(['time_window', 'template_id']).size().reset_index(name='count')
            
            # 透视为宽格式
            template_freq_df = template_counts.pivot(
                index='time_window', 
                columns='template_id', 
                values='count'
            ).fillna(0).reset_index()
            
            # 重命名列
            template_freq_df.columns.name = None
            template_freq_df = template_freq_df.rename(columns={'time_window': 'timestamp'})
            
            # 为模板ID添加前缀
            template_cols = [col for col in template_freq_df.columns if col != 'timestamp']
            rename_dict = {col: f'template_{col}' for col in template_cols}
            template_freq_df = template_freq_df.rename(columns=rename_dict)
            
            frequency_data[instance_id] = template_freq_df
        
        return frequency_data
    
    def merge_rare_templates(self, frequency_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        合并低频模板
        
        Args:
            frequency_data: 模板频次数据
        
        Returns:
            合并低频模板后的数据
        """
        processed_data = {}
        
        for instance_id, df in frequency_data.items():
            logger.info(f"处理实例 {instance_id} 的低频模板...")
            
            df_processed = df.copy()
            
            # 计算每个模板的总频次
            template_cols = [col for col in df.columns if col.startswith('template_')]
            total_frequencies = df[template_cols].sum()
            
            # 识别低频模板
            rare_templates = total_frequencies[
                total_frequencies < self.config['min_template_frequency']
            ].index.tolist()
            
            if rare_templates:
                logger.info(f"发现 {len(rare_templates)} 个低频模板，将合并为一个特征")
                
                # 合并低频模板
                df_processed[self.config['rare_template_name']] = df_processed[rare_templates].sum(axis=1)
                
                # 删除原始低频模板列
                df_processed = df_processed.drop(columns=rare_templates)
            
            processed_data[instance_id] = df_processed
        
        return processed_data
    
    def process_logs(self, file_paths: List[str] or str) -> Dict[str, pd.DataFrame]:
        """
        完整的日志数据处理流程
        
        Args:
            file_paths: 日志文件路径
        
        Returns:
            处理后的日志特征数据
        """
        logger.info("开始处理日志数据...")
        
        # 1. 加载日志数据
        log_data = self.load_log_data(file_paths)
        
        # 2. 提取日志模板
        log_data = self.extract_log_templates(log_data)
        
        # 3. 计算模板频次
        frequency_data = self.compute_template_frequencies(log_data)
        
        # 4. 合并低频模板
        frequency_data = self.merge_rare_templates(frequency_data)
        
        # 5. 时间对齐
        from utils import align_timestamps
        dfs = list(frequency_data.values())
        aligned_dfs = align_timestamps(dfs, method='intersection')
        
        # 重新组织数据
        processed_data = {}
        instance_ids = list(frequency_data.keys())
        for i, instance_id in enumerate(instance_ids):
            processed_data[instance_id] = aligned_dfs[i]
        
        # 保存处理结果
        self.instance_log_features = processed_data
        self._compute_template_statistics()
        
        logger.info(f"日志数据处理完成，共处理 {len(processed_data)} 个实例")
        return processed_data
    
    def _compute_template_statistics(self):
        """计算模板统计信息"""
        stats = {}
        
        for instance_id, df in self.instance_log_features.items():
            template_cols = [col for col in df.columns if col.startswith('template_') or col == self.config['rare_template_name']]
            
            instance_stats = {}
            for col in template_cols:
                instance_stats[col] = {
                    'total_frequency': df[col].sum(),
                    'mean_frequency': df[col].mean(),
                    'max_frequency': df[col].max(),
                    'non_zero_count': (df[col] > 0).sum()
                }
            
            stats[instance_id] = instance_stats
        
        self.template_frequencies = stats
    
    def get_feature_matrix(self, timestamp: str = None) -> Dict[str, np.ndarray]:
        """
        获取指定时间戳的日志特征矩阵
        
        Args:
            timestamp: 目标时间戳，如果为None则返回所有时间戳
        
        Returns:
            特征矩阵字典
        """
        feature_matrices = {}
        
        for instance_id, df in self.instance_log_features.items():
            template_cols = [col for col in df.columns if col.startswith('template_') or col == self.config['rare_template_name']]
            
            if timestamp:
                # 获取特定时间戳的特征
                mask = df['timestamp'] == pd.to_datetime(timestamp)
                if mask.any():
                    features = df.loc[mask, template_cols].values.flatten()
                    feature_matrices[instance_id] = features
            else:
                # 获取所有时间戳的特征
                features = df[template_cols].values
                feature_matrices[instance_id] = features
        
        return feature_matrices
    
    def get_template_info(self) -> Dict[int, str]:
        """获取模板信息"""
        template_info = {}
        for template_id in self.log_parser.templates.keys():
            template_info[template_id] = self.log_parser.get_template_string(template_id)
        return template_info
    
if __name__ == "__main__":
    from utils import collect_gaia_file_paths
    from config import DATA_PATHS
    import os
    import gc
    import re

    root_dir = DATA_PATHS['log_data']
    output_dir = os.path.join(DATA_PATHS['output_data'], 'log')
    os.makedirs(output_dir, exist_ok=True)

    file_dict = collect_gaia_file_paths(root_dir)
    all_log_files = []
    for date, files in file_dict['log'].items():
        all_log_files.extend(files)

    print("将要处理的日志文件列表：")
    for f in all_log_files:
        print(f)
    print(f"共计 {len(all_log_files)} 个日志文件\n")

    # 分批处理，每次只处理一个文件，处理完就释放内存
    for log_file in all_log_files:
        print(f"\n正在处理: {log_file}")
        processor = LogProcessor()
        processed_data = processor.process_logs([log_file])
        print(f"本文件共处理实例数: {len(processed_data)}")

        # 从路径中提取日期（假设路径中有类似 2021-07-01 这样的日期）
        match = re.search(r'(\d{4}-\d{2}-\d{2})', log_file)
        date_str = match.group(1) if match else "unknown_date"

        # 创建对应日期的文件夹
        date_output_dir = os.path.join(output_dir, date_str)
        os.makedirs(date_output_dir, exist_ok=True)

        for instance_id, df in processed_data.items():
            print(f"实例: {instance_id}, 特征shape: {df.shape}")
            out_path = os.path.join(
                date_output_dir,
                f"{instance_id}_log_features.csv"
            )
            df.to_csv(out_path, index=False)
            print(f"已保存: {out_path}")
        # 主动释放内存
        del processed_data
        gc.collect()