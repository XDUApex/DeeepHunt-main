"""
配置文件 - 定义数据预处理的各种参数
"""

# 时间相关配置
TIME_CONFIG = {
    'sampling_interval': 60,  # 采样间隔，单位：秒
    'time_format': '%Y-%m-%d %H:%M:%S',  # 时间格式
    'timezone': 'UTC'  # 时区
}

# 指标数据配置
METRIC_CONFIG = {
    'strategy': 'intersection',  # 指标处理策略: 'intersection' 或 'union'
    'fill_value': 0,  # 缺失指标的填充值
    'outlier_threshold': 3,  # 异常值检测阈值（Z-score）
}

# 日志数据配置
LOG_CONFIG = {
    'min_template_frequency': 10,  # 模板最小出现频次阈值
    'rare_template_name': 'RARE_TEMPLATES',  # 低频模板合并后的名称
    'template_extraction_method': 'drain',  # 模板提取方法
    'drain_config': {
        'sim_th': 0.4,
        'depth': 4
    }
}

# 调用链数据配置
TRACE_CONFIG = {
    'status_codes': ['200', '404', '500', '503'],  # 关注的状态码
    'latency_percentiles': [50, 90, 95, 99],  # 时延百分位数
    'request_types': ['http', 'rpc', 'db'],  # 请求类型
}

# 标准化配置
NORMALIZATION_CONFIG = {
    'method': 'z_score',  # 标准化方法: 'z_score', 'min_max', 'robust'
    'eps': 1e-8  # 防止除零的小数
}

# 数据路径配置
DATA_PATHS = {
    'metric_data': '/home/fuxian/DataSet/new_GAIA',   # 根目录，后续代码可拼接/metric
    'log_data': '/home/fuxian/DataSet/new_GAIA',
    'trace_data': '/home/fuxian/DataSet/new_GAIA',
    'output_data': './data/processed/'
}

# 并行处理配置
PROCESSING_CONFIG = {
    'n_jobs': -1,  # 并行处理的进程数，-1表示使用所有CPU
    'chunk_size': 1000,  # 数据块大小
    'memory_limit': '8GB'  # 内存限制
}