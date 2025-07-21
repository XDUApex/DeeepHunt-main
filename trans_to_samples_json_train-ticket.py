import pickle
import dgl
import config
import torch
import json
import numpy as np
import os
from config import train_ticket
import pandas as pd
from datetime import datetime
import glob


def load_json_metric(file_path, output_path):
    """加载 metric 特征，并转换格式后保存为 JSON
       原始 Metric JSON Shape:  Shape of X: (100, 26, 6)
       处理后 Metric JSON Shape: (71时间戳, 10实例, 183)
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    # 获取 metric 特征矩阵 "X"，shape = (故障, 实例, 20, 特征维度)
    metric_data = np.array(data["X"])  

    print("原始 Metric JSON Shape:", metric_data.shape)  # (71, 10, 20, 183)

    # 获取原始维度信息
    num_faults, num_instances,  feature_dim = metric_data.shape

    # 只取前245维（索引 9）
    selected_metric = metric_data[:, :24, :]  # (71, 10, 183)

    # === 获取真实时间戳 ===
    #     # 读取时间戳信息
    timestamps = []

    # 获取所有符合 "2025-03-03 18_57_36" 格式的文件夹
    base_path = "../DataSet/train-ticket-original"
    folders = sorted(glob.glob(os.path.join(base_path, "*-*-* *_*_*")))  # 匹配时间格式的文件夹

    for folder in folders:
        csv_path = os.path.join(folder, "groundtruth.csv")
        if os.path.exists(csv_path):
            print(f"处理文件: {csv_path}")
            df = pd.read_csv(csv_path)
            # 确保 'st_time' 列存在
            if "st_time" in df.columns:
                df["st_time"] = df["st_time"].apply(
                    lambda x: int(datetime.strptime(x.split(".")[0], "%Y-%m-%d %H:%M:%S").timestamp())
                )
                timestamps.extend(df["st_time"].tolist())  # 提取转换后的时间戳

    # 只保留前 71 个时间戳（与 feature 维度对齐）
    timestamps = timestamps[:selected_metric.shape[0]]

    # 如果时间戳不足 71 个，补零（保证时间戳与数据一一对应）
    while len(timestamps) < selected_metric.shape[0]:
        timestamps.append(0)

    # === 转换为 JSON 格式 ===
    result = {str(ts): selected_metric[i].tolist() for i, ts in enumerate(timestamps)}

    # 保存到 JSON 文件
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    # 打印最终形状
    processed_data = np.array(list(result.values()))
    print("Processed Metric JSON Shape:", processed_data.shape)  # (71, 10, 183)

    return result  # 返回 JSON 结构，方便检查


def load_json_log(base_path, instances, output_path):
    """加载 log 特征，并转换格式后保存为 JSON
    (71, 768)->(71时间戳,10,768)
    原始 Metric JSON Shape: Shape of X: (100, 768)
    Processed Log JSON Shape: (71, 10, 768)
    """
    instance_features = []

    for instance in instances:
        file_path = os.path.join(base_path, instance, "gaia_log_tmp.json")

        # 读取 JSON 数据
        with open(file_path, "r") as f:
            data = json.load(f)

        log_data = np.array(data["X"])

        print("原始 Metric JSON Shape:", np.array(log_data.shape))

        # 存入列表，代表一个实例的数据
        instance_features.append(log_data)

    # 转置 shape：(时间戳, 实例数, 特征向量)
    final_data = np.stack(instance_features, axis=1)  # (时间戳数, 10, 特征维度)

    # === 获取真实时间戳 ===
    #     # 读取时间戳信息
    timestamps = []

    # 获取所有符合 "2025-03-03 18_57_36" 格式的文件夹
    base_path = "../DataSet/train-ticket-original"
    folders = sorted(glob.glob(os.path.join(base_path, "*-*-* *_*_*")))  # 匹配时间格式的文件夹

    for folder in folders:
        csv_path = os.path.join(folder, "groundtruth.csv")
        if os.path.exists(csv_path):
            print(f"处理文件: {csv_path}")
            df = pd.read_csv(csv_path)
            # 确保 'st_time' 列存在
            if "st_time" in df.columns:
                df["st_time"] = df["st_time"].apply(
                    lambda x: int(datetime.strptime(x.split(".")[0], "%Y-%m-%d %H:%M:%S").timestamp())
                )
                timestamps.extend(df["st_time"].tolist())  # 提取转换后的时间戳


    # 只保留前 71 个时间戳（与 feature 维度对齐）
    timestamps = timestamps[:final_data.shape[0]]

    # 如果时间戳不足 71 个，补零（保证时间戳与数据一一对应）
    while len(timestamps) < final_data.shape[0]:
        timestamps.append(0)

    # 转换为 JSON 格式
    result = {str(ts): final_data[i].tolist() for i, ts in enumerate(timestamps)}

    # 保存到新的 JSON 文件
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    return result  # 返回 JSON 结构，方便检查

def load_json_trace(base_path, instances, output_path):
    """加载 trace 特征，并转换格式后保存为 JSON
    原始 trace JSON Shape: Shape of X: (100, 10)
    Processed Log JSON Shape: (71, 10, 10)  
    """
    instance_features = []

    for instance in instances:
        file_path = os.path.join(base_path, instance, "gaia_trace_tmp.json")

        # 读取 JSON 数据
        with open(file_path, "r") as f:
            data = json.load(f)

        log_data = np.array(data["X"])

        print("原始 Trace JSON Shape:", np.array(log_data.shape))

        # 存入列表，代表一个实例的数据
        instance_features.append(log_data)

    # 转置 shape：(时间戳, 实例数, 特征向量)
    final_data = np.stack(instance_features, axis=1)  # (时间戳数, 10, 特征维度)

    # === 获取真实时间戳 ===
    #     # 读取时间戳信息
    timestamps = []

    # 获取所有符合 "2025-03-03 18_57_36" 格式的文件夹
    base_path = "../DataSet/train-ticket-original"
    folders = sorted(glob.glob(os.path.join(base_path, "*-*-* *_*_*")))  # 匹配时间格式的文件夹

    for folder in folders:
        csv_path = os.path.join(folder, "groundtruth.csv")
        if os.path.exists(csv_path):
            print(f"处理文件: {csv_path}")
            df = pd.read_csv(csv_path)
            # 确保 'st_time' 列存在
            if "st_time" in df.columns:
                df["st_time"] = df["st_time"].apply(
                    lambda x: int(datetime.strptime(x.split(".")[0], "%Y-%m-%d %H:%M:%S").timestamp())
                )
                timestamps.extend(df["st_time"].tolist())  # 提取转换后的时间戳


    # 只保留前 71 个时间戳（与 feature 维度对齐）
    timestamps = timestamps[:final_data.shape[0]]

    # 如果时间戳不足 71 个，补零（保证时间戳与数据一一对应）
    while len(timestamps) < final_data.shape[0]:
        timestamps.append(0)

    # 转换为 JSON 格式
    result = {str(ts): final_data[i].tolist() for i, ts in enumerate(timestamps)}


    # 保存到新的 JSON 文件
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    return result  # 返回 JSON 结构，方便检查


# 示例调用
metric_file_path = "../DataSet/train-ticket-original/gaia_metric_tmp.json"
instances = train_ticket["instances"].split()
base_path = "../DataSet/train-ticket-service"

metric_output_path = "./data/D3/processed_metric.json"
log_output_path = "./data/D3/processed_log.json"
trace_output_path = "./data/D3/processed_trace.json"

metric_json = load_json_metric(metric_file_path, metric_output_path)
log_json = load_json_log(base_path, instances, log_output_path)
trace_json = load_json_trace(base_path, instances, trace_output_path)