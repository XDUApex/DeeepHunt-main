import os
import json
import numpy as np
from collections import defaultdict

def debug_data_structure(file_path, service_name, data_type):
    """
    调试函数：检查JSON文件的数据结构
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"DEBUG - {service_name} {data_type}数据结构:")
        print(f"  - 数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"  - 字典键: {list(data.keys())}")
            for key, value in data.items():
                print(f"  - 键 '{key}' 的值类型: {type(value)}")
                if isinstance(value, list):
                    print(f"    - 列表长度: {len(value)}")
                    if value and isinstance(value[0], list):
                        print(f"    - 第一个子列表长度: {len(value[0])}")
                break  # 只查看第一个键
        elif isinstance(data, list):
            print(f"  - 列表长度: {len(data)}")
            if data:
                print(f"  - 第一个元素类型: {type(data[0])}")
        
        return data
    
    except Exception as e:
        print(f"DEBUG - 读取 {service_name} {data_type}数据时出错: {e}")
        return None

def load_existing_timestamps(metric_file_path):
    """
    从现有的processed_metric.json文件中加载时间戳
    
    Args:
        metric_file_path: processed_metric.json文件路径
        
    Returns:
        sorted_timestamps: 排序后的时间戳列表（整数）
    """
    try:
        with open(metric_file_path, 'r', encoding='utf-8') as f:
            metric_data = json.load(f)
        
        # 提取所有时间戳并转换为整数，然后排序
        timestamps = [int(ts) for ts in metric_data.keys()]
        timestamps.sort()
        
        print(f"从 {metric_file_path} 加载了 {len(timestamps)} 个时间戳")
        print(f"时间戳范围: {timestamps[0]} - {timestamps[-1]}")
        
        return timestamps
        
    except Exception as e:
        print(f"读取metric文件时出错: {e}")
        print("将使用默认时间戳生成方式")
        return None

def reorganize_aiops_data(base_path, metric_file_path=None, output_path=None):
    """
    将AIOps服务实例数据从 服务实例-时间窗口 重组为 时间戳-服务实例向量
    模仿train-ticket数据集的格式，使用与metric文件一致的时间戳
    
    Args:
        base_path: 服务实例根目录路径 (/home/fuxian/DeepHunt-main/DataSet/aiops/service)
        metric_file_path: processed_metric.json文件路径
        output_path: 输出目录路径（可选）
    """
    
    if output_path is None:
        output_path = os.path.join(base_path, "processed_data")
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 加载现有的时间戳
    if metric_file_path and os.path.exists(metric_file_path):
        timestamps = load_existing_timestamps(metric_file_path)
    else:
        timestamps = None
        print("未提供metric文件路径或文件不存在，将使用默认时间戳")
    
    # 获取所有服务实例目录
    service_dirs = [d for d in os.listdir(base_path) 
                   if os.path.isdir(os.path.join(base_path, d)) and d != "processed_data"]
    
    service_dirs = sorted(service_dirs)  # 确保服务顺序一致
    print(f"发现 {len(service_dirs)} 个服务实例目录: {service_dirs}")
    
    # 收集所有服务的数据
    all_log_data = []  # 存储每个服务的log向量序列
    all_trace_data = []  # 存储每个服务的trace向量序列
    
    max_time_windows = 0  # 记录最大时间窗口数
    
    # 先调试第一个服务的数据结构
    if service_dirs:
        first_service = service_dirs[0]
        first_service_path = os.path.join(base_path, first_service)
        
        log_file = os.path.join(first_service_path, "aiops22_log_tmp.json")
        trace_file = os.path.join(first_service_path, "aiops22_trace_tmp.json")
        
        if os.path.exists(log_file):
            debug_data_structure(log_file, first_service, "log")
        if os.path.exists(trace_file):
            debug_data_structure(trace_file, first_service, "trace")
    
    # 遍历每个服务实例，收集数据
    for service_name in service_dirs:
        service_path = os.path.join(base_path, service_name)
        
        # 处理log数据
        log_file = os.path.join(service_path, "aiops22_log_tmp.json")
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                
                # 提取向量序列 - 专门提取 "X" 键的数据
                if isinstance(log_data, dict) and "X" in log_data:
                    log_vectors = log_data["X"]
                    print(f"  - {service_name} log数据从键 'X' 提取，共 {len(log_vectors)} 个时间窗口")
                elif isinstance(log_data, list):
                    # 如果直接就是列表（向后兼容）
                    log_vectors = log_data
                    print(f"  - {service_name} log数据为直接列表格式")
                else:
                    print(f"警告: {service_name} 的log数据中未找到 'X' 键，数据类型: {type(log_data)}")
                    if isinstance(log_data, dict):
                        print(f"  可用的键: {list(log_data.keys())}")
                    log_vectors = []
                
                all_log_data.append(log_vectors)
                max_time_windows = max(max_time_windows, len(log_vectors))
                print(f"加载 {service_name} 的log数据，共 {len(log_vectors)} 个时间窗口")
                
            except Exception as e:
                print(f"处理 {service_name} 的log数据时出错: {e}")
                all_log_data.append([])
        else:
            print(f"未找到 {service_name} 的log数据文件")
            all_log_data.append([])
        
        # 处理trace数据
        trace_file = os.path.join(service_path, "aiops22_trace_tmp.json")
        if os.path.exists(trace_file):
            try:
                with open(trace_file, 'r', encoding='utf-8') as f:
                    trace_data = json.load(f)
                
                # 提取向量序列 - 专门提取 "X" 键的数据
                if isinstance(trace_data, dict) and "X" in trace_data:
                    trace_vectors = trace_data["X"]
                    print(f"  - {service_name} trace数据从键 'X' 提取，共 {len(trace_vectors)} 个时间窗口")
                elif isinstance(trace_data, list):
                    # 如果直接就是列表（向后兼容）
                    trace_vectors = trace_data
                    print(f"  - {service_name} trace数据为直接列表格式")
                else:
                    print(f"警告: {service_name} 的trace数据中未找到 'X' 键，数据类型: {type(trace_data)}")
                    if isinstance(trace_data, dict):
                        print(f"  可用的键: {list(trace_data.keys())}")
                    trace_vectors = []
                
                all_trace_data.append(trace_vectors)
                print(f"加载 {service_name} 的trace数据，共 {len(trace_vectors)} 个时间窗口")
                
            except Exception as e:
                print(f"处理 {service_name} 的trace数据时出错: {e}")
                all_trace_data.append([])
        else:
            print(f"未找到 {service_name} 的trace数据文件")
            all_trace_data.append([])
    
    print(f"\n最大时间窗口数: {max_time_windows}")
    
    # 确定要使用的时间戳
    if timestamps is None:
        # 如果没有从metric文件加载时间戳，使用默认生成方式
        base_timestamp = 1651334400  # 基础时间戳
        timestamps = [base_timestamp + i * 300 for i in range(max_time_windows)]
        print(f"使用默认生成的时间戳，共 {len(timestamps)} 个")
    else:
        # 使用从metric文件加载的时间戳
        # 如果数据的时间窗口数超过了metric文件的时间戳数，截断数据
        # 如果数据的时间窗口数少于metric文件的时间戳数，使用现有数据数量
        actual_windows = min(max_time_windows, len(timestamps))
        timestamps = timestamps[:actual_windows]
        print(f"使用从metric文件加载的时间戳，实际使用 {len(timestamps)} 个")
    
    # 重组数据格式：{时间戳: [[服务1向量], [服务2向量], ...]}
    def reorganize_data(all_data, data_type):
        result = {}
        
        for time_idx in range(len(timestamps)):
            timestamp = str(timestamps[time_idx])
            time_window_data = []
            
            # 遍历每个服务，获取该时间窗口的向量
            for service_idx, service_vectors in enumerate(all_data):
                if time_idx < len(service_vectors) and service_vectors[time_idx]:
                    # 如果该服务在该时间窗口有数据
                    time_window_data.append(service_vectors[time_idx])
                else:
                    # 如果没有数据，添加零向量（长度根据第一个有效向量确定）
                    if any(all_data) and any(any(vectors) for vectors in all_data):
                        # 找到第一个非空向量来确定维度
                        for vectors in all_data:
                            if vectors and any(vectors):
                                default_length = len(vectors[0])
                                time_window_data.append([0.0] * default_length)
                                break
                    else:
                        time_window_data.append([])
            
            result[timestamp] = time_window_data
        
        return result
    
    # 重组log和trace数据
    log_result = reorganize_data(all_log_data, "log")
    trace_result = reorganize_data(all_trace_data, "trace")
    
    # 保存重组后的数据
    print("\n开始保存重组后的数据...")
    
    # 保存log数据
    log_output_file = os.path.join(output_path, "processed_log.json")
    with open(log_output_file, 'w', encoding='utf-8') as f:
        json.dump(log_result, f, indent=4, ensure_ascii=False)
    
    # 保存trace数据
    trace_output_file = os.path.join(output_path, "processed_trace.json")
    with open(trace_output_file, 'w', encoding='utf-8') as f:
        json.dump(trace_result, f, indent=4, ensure_ascii=False)
    
    # 保存服务实例名称映射
    service_mapping = {
        "service_order": service_dirs,
        "description": "每个时间戳下的向量列表按此顺序对应服务实例",
        "timestamps_info": {
            "total_timestamps": len(timestamps),
            "first_timestamp": timestamps[0] if timestamps else None,
            "last_timestamp": timestamps[-1] if timestamps else None,
            "source": "loaded from metric file" if metric_file_path else "generated"
        }
    }
    
    mapping_file = os.path.join(output_path, "service_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(service_mapping, f, indent=4, ensure_ascii=False)
    
    # 生成统计信息
    print("\n=== 数据重组完成 ===")
    print(f"处理了 {len(service_dirs)} 个服务实例")
    print(f"生成了 {len(log_result)} 个时间戳的log数据")
    print(f"生成了 {len(trace_result)} 个时间戳的trace数据")
    print(f"时间戳范围: {timestamps[0]} - {timestamps[-1]}")
    
    # 检查数据形状
    if log_result:
        first_timestamp = next(iter(log_result.keys()))
        log_shape = np.array(log_result[first_timestamp]).shape
        print(f"Log数据形状 (每个时间戳): {log_shape}")
    
    if trace_result:
        first_timestamp = next(iter(trace_result.keys()))
        trace_shape = np.array(trace_result[first_timestamp]).shape
        print(f"Trace数据形状 (每个时间戳): {trace_shape}")
    
    print(f"\n输出文件已保存到: {output_path}")
    print(f"- Log数据: {log_output_file}")
    print(f"- Trace数据: {trace_output_file}")
    print(f"- 服务映射: {mapping_file}")
    
    return log_result, trace_result, service_dirs, timestamps

def create_combined_format(log_result, trace_result, service_dirs, timestamps, output_path):
    """
    创建类似train-ticket格式的合并数据
    """
    print("\n创建合并格式数据...")
    
    # 使用传入的时间戳顺序
    
    # Log数据合并
    log_combined = []
    for ts in timestamps:
        ts_str = str(ts)
        if ts_str in log_result:
            log_combined.append(log_result[ts_str])
    
    # Trace数据合并
    trace_combined = []
    for ts in timestamps:
        ts_str = str(ts)
        if ts_str in trace_result:
            trace_combined.append(trace_result[ts_str])
    
    # 创建最终的合并数据结构
    combined_data = {
        "timestamps": timestamps,
        "service_instances": service_dirs,
        "log_data": log_combined,  # shape: (时间戳数, 服务数, 特征维度)
        "trace_data": trace_combined,  # shape: (时间戳数, 服务数, 特征维度)
        "description": {
            "log_shape": f"({len(log_combined)}, {len(service_dirs)}, vector_dim)",
            "trace_shape": f"({len(trace_combined)}, {len(service_dirs)}, vector_dim)",
            "format": "时间戳 -> 服务实例 -> 特征向量",
            "timestamp_source": "consistent with processed_metric.json"
        }
    }
    
    # 保存合并数据
    combined_file = os.path.join(output_path, "combined_data.json")
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)
    
    print(f"合并数据已保存到: {combined_file}")
    
    # 打印最终形状信息
    if log_combined:
        log_array = np.array(log_combined)
        print(f"最终Log数据形状: {log_array.shape}")
    
    if trace_combined:
        trace_array = np.array(trace_combined)
        print(f"最终Trace数据形状: {trace_array.shape}")

# 主函数
if __name__ == "__main__":
    # 设置路径
    base_path = "/home/fuxian/DeepHunt-main/DataSet/aiops/service"
    metric_file_path = "/home/fuxian/DeepHunt-main/data/D1/processed_metric.json"
    
    print("开始AIOps数据重组...")
    print("=" * 50)
    
    # 检查metric文件是否存在
    if os.path.exists(metric_file_path):
        print(f"找到metric文件: {metric_file_path}")
    else:
        print(f"警告: 未找到metric文件 {metric_file_path}")
        print("将使用默认时间戳生成方式")
    
    # 执行数据重组
    log_result, trace_result, service_dirs, timestamps = reorganize_aiops_data(
        base_path, 
        metric_file_path
    )
    
    # 创建合并格式
    output_path = os.path.join(base_path, "processed_data")
    create_combined_format(log_result, trace_result, service_dirs, timestamps, output_path)
    
    print("\n" + "=" * 50)
    print("所有数据处理完成！")
    
    # 显示示例数据格式
    if log_result:
        first_ts = next(iter(log_result.keys()))
        print(f"\n示例数据格式 (时间戳 {first_ts}):")
        print(f"Log数据长度: {len(log_result[first_ts])}")
        print(f"第一个服务的向量长度: {len(log_result[first_ts][0]) if log_result[first_ts] else 0}")
        
        # 显示前几个向量的前几个值作为示例
        if log_result[first_ts] and len(log_result[first_ts][0]) > 0:
            print(f"第一个服务前5个特征值: {log_result[first_ts][0][:5]}")
    
    # 验证时间戳一致性
    print(f"\n时间戳验证:")
    print(f"使用的时间戳数量: {len(timestamps)}")
    print(f"前5个时间戳: {timestamps[:5]}")
    print(f"后5个时间戳: {timestamps[-5:]}")






























# import os
# import json
# import numpy as np
# from datetime import datetime

# def get_aiops_timestamps(base_path="./DataSet/aiops"):
#     """
#     遍历 base_path 下的日期文件夹（如 2022-05-01），返回升序排列的时间戳列表
#     """
#     dates = []
#     for service in os.listdir(base_path):
#         service_path = os.path.join(base_path, service)
#         if not os.path.isdir(service_path):
#             continue
#         for date_folder in os.listdir(service_path):
#             if date_folder.startswith("2022-05-"):
#                 try:
#                     dt = datetime.strptime(date_folder, "%Y-%m-%d")
#                     timestamp = int(dt.timestamp())
#                     dates.append(timestamp)
#                 except Exception:
#                     continue
#     unique_dates = sorted(list(set(dates)))
#     return unique_dates

# def get_aiops_instances(base_path="./DataSet/aiops"):
#     """
#     返回base_path下所有服务名（即实例名）的列表
#     """
#     services = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
#     return sorted(services)

# def load_aiops_log(base_path, output_path):
#     """
#     加载AIOps log特征并转换格式
#     合并所有实例和所有日期的log特征，输出格式: {timestamp: {instance: features}}
#     """
#     timestamps = get_aiops_timestamps(base_path)
#     instances = get_aiops_instances(base_path)
#     result = {}

#     for instance in instances:
#         for date_folder in os.listdir(os.path.join(base_path, instance)):
#             if not date_folder.startswith("2022-05-"):
#                 continue
#             log_file = os.path.join(base_path, instance, date_folder, "log.json")
#             if not os.path.exists(log_file):
#                 continue
#             try:
#                 with open(log_file, "r") as f:
#                     data = json.load(f)
#                 log_data = np.array(data["X"]) if "X" in data else np.array(data)
#                 dt = datetime.strptime(date_folder, "%Y-%m-%d")
#                 ts = int(dt.timestamp())
#                 # 按需处理log_data的形状
#                 if ts not in result:
#                     result[ts] = {}
#                 result[ts][instance] = log_data.tolist()
#             except Exception as e:
#                 print(f"读取 {log_file} 失败: {e}")
#                 continue

#     # 补全所有timestamp和instance，缺失用空列表
#     for ts in timestamps:
#         if ts not in result:
#             result[ts] = {}
#         for instance in instances:
#             if instance not in result[ts]:
#                 result[ts][instance] = []

#     # 按时间排序
#     sorted_result = {str(ts): result[ts] for ts in sorted(result.keys())}

#     with open(output_path, "w") as f:
#         json.dump(sorted_result, f, indent=4)
#     print(f"Processed AIOps Log saved to: {output_path}")
#     return sorted_result

# def load_aiops_trace(base_path, output_path):
#     """
#     加载AIOps trace特征并转换格式
#     合并所有实例和所有日期的trace特征，输出格式: {timestamp: {instance: features}}
#     """
#     timestamps = get_aiops_timestamps(base_path)
#     instances = get_aiops_instances(base_path)
#     result = {}

#     for instance in instances:
#         for date_folder in os.listdir(os.path.join(base_path, instance)):
#             if not date_folder.startswith("2022-05-"):
#                 continue
#             trace_file = os.path.join(base_path, instance, date_folder, "trace.json")
#             if not os.path.exists(trace_file):
#                 continue
#             try:
#                 with open(trace_file, "r") as f:
#                     data = json.load(f)
#                 trace_data = np.array(data["X"]) if "X" in data else np.array(data)
#                 dt = datetime.strptime(date_folder, "%Y-%m-%d")
#                 ts = int(dt.timestamp())
#                 # 按需处理trace_data的形状
#                 if ts not in result:
#                     result[ts] = {}
#                 result[ts][instance] = trace_data.tolist()
#             except Exception as e:
#                 print(f"读取 {trace_file} 失败: {e}")
#                 continue

#     # 补全所有timestamp和instance，缺失用空列表
#     for ts in timestamps:
#         if ts not in result:
#             result[ts] = {}
#         for instance in instances:
#             if instance not in result[ts]:
#                 result[ts][instance] = []

#     # 按时间排序
#     sorted_result = {str(ts): result[ts] for ts in sorted(result.keys())}

#     with open(output_path, "w") as f:
#         json.dump(sorted_result, f, indent=4)
#     print(f"Processed AIOps Trace saved to: {output_path}")
#     return sorted_result
# def load_aiops_metric(base_path, output_path):
#     metric_file = os.path.join(base_path, "aiops22_metric_tmp.json")
#     if not os.path.exists(metric_file):
#         print(f"Metric文件不存在: {metric_file}")
#         return {}
#     try:
#         with open(metric_file, "r") as f:
#             data = json.load(f)
#         metric_data = np.array(data["X"]) if "X" in data else np.array(data)
#         print(f"原始 AIOps Metric JSON Shape: {metric_data.shape}")
#         timestamps = get_aiops_timestamps(base_path)
#         if len(metric_data.shape) == 4:
#             selected_metric = metric_data[:, :, -1, :]
#         elif len(metric_data.shape) == 3:
#             selected_metric = metric_data
#         elif len(metric_data.shape) == 2:
#             num_instances = len([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("2022-05-")])
#             if num_instances == 0:
#                 num_instances = 10
#             selected_metric = np.expand_dims(metric_data, axis=1)
#             selected_metric = np.repeat(selected_metric, num_instances, axis=1)
#         else:
#             print(f"未知的metric数据格式: {metric_data.shape}")
#             return {}
#         if len(timestamps) != selected_metric.shape[0]:
#             if len(timestamps) > selected_metric.shape[0]:
#                 timestamps = timestamps[:selected_metric.shape[0]]
#             else:
#                 while len(timestamps) < selected_metric.shape[0]:
#                     timestamps.append(timestamps[-1] + 86400 if timestamps else 1651680000)
#         print(f"Processed AIOps Metric Shape: {selected_metric.shape}")
#         result = {str(ts): selected_metric[i].tolist() for i, ts in enumerate(timestamps)}
#         with open(output_path, "w") as f:
#             json.dump(result, f, indent=4)
#         return result
#     except Exception as e:
#         print(f"读取metric文件失败: {metric_file}, 错误: {e}")
#         return {}
    
# if __name__ == "__main__":
#         base_path = "./DataSet/aiops"
#         # 输出路径
#         metric_output_path = "./data/D1/processed_metric.json"
#         log_output_path = "./data/D1/processed_log.json"
#         trace_output_path = "./data/D1/processed_trace.json"
        
#         # 确保输出目录存在
#         os.makedirs("./data/D1", exist_ok=True)
        
#         print("开始处理AIOps数据集...")
        
#         # 处理metric数据
#         print("\n处理Metric数据...")
#         metric_json = load_aiops_metric(base_path, metric_output_path)
        
#         # 处理log数据
#         print("\n处理Log数据...")
#         log_json = load_aiops_log(base_path, log_output_path)
        
#         # 处理trace数据
#         print("\n处理Trace数据...")
#         trace_json = load_aiops_trace(base_path, trace_output_path)
        
#         print("\nAIOps数据集处理完成！")
#         print(f"输出文件:")
#         print(f"- Metric: {metric_output_path}")
#         print(f"- Log: {log_output_path}")
#         print(f"- Trace: {trace_output_path}")