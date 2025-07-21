import json
import pickle
import torch
import os
import pandas as pd
import dgl

def load_json_file(file_path):
    """加载 JSON 并按时间戳排序"""
    with open(file_path, "r") as f:
        data = json.load(f)
    return {int(k): torch.tensor(v) for k, v in sorted(data.items())}

def find_all_trace_csvs(data_dir):
    """
    递归寻找所有 trace.csv 文件的路径（真实结构: aiops/<service_name>/<date>/trace/trace.csv）
    """
    trace_csv_paths = []
    for service_name in sorted(os.listdir(data_dir)):
        service_path = os.path.join(data_dir, service_name)
        if not os.path.isdir(service_path):
            continue
        for date_folder in sorted(os.listdir(service_path)):
            trace_file = os.path.join(service_path, date_folder, "trace", "trace.csv")
            if os.path.exists(trace_file):
                trace_csv_paths.append(trace_file)
    return trace_csv_paths

def read_trace_get_edges(edges_path, node_hash_path, data_dir):
    """
    读取所有 trace.csv，提取跨服务边，生成 edges.json
    """
    # 读取 service_name -> node_id 的映射
    with open(node_hash_path, "r") as f:
        node_map = json.load(f)

    edges_list = []

    # 遍历所有 trace.csv
    trace_csv_paths = find_all_trace_csvs(data_dir)
    for trace_path in trace_csv_paths:
        print(f"正在处理 {trace_path} ...")

        # 用 dtype 降低内存
        df = pd.read_csv(trace_path, usecols=["service_name", "span_id", "parent_id"], dtype=str)
        span_to_service = df.set_index("span_id")["service_name"].to_dict()
        df["parent_service"] = df["parent_id"].map(span_to_service)
        df = df.dropna(subset=["parent_service"])
        df["source"] = df["parent_service"].map(node_map)
        df["target"] = df["service_name"].map(node_map)
        df = df.dropna(subset=["source", "target"])
        edges_list.extend(zip(df["source"].astype(int), df["target"].astype(int)))

    # 去重
    edges_df = pd.DataFrame(edges_list, columns=["source", "target"]).drop_duplicates()
    edges_json = [edges_df["source"].tolist(), edges_df["target"].tolist()]

    # 保存
    with open(edges_path, "w") as f:
        json.dump(edges_json, f, indent=4)
    print(f"✅ 处理完成！共保存 {len(edges_json[0])} 条去重后调用关系到 {edges_path}")

def save_to_pkl(metric_path, log_path, trace_path, output_path, node_hash_path, data_dir):
    """
    加载三个 JSON 文件并合并，最终保存为 .pkl
    """
    # 读取 JSON 数据并转成 PyTorch Tensor
    metric_data = load_json_file(metric_path)
    log_data = load_json_file(log_path)
    trace_data = load_json_file(trace_path)
    timestamps = list(metric_data.keys())
    processed_data = []

    # 生成边文件
    edges_dir = os.path.dirname(metric_path)
    edges_path = os.path.join(edges_dir, "edges.json")
    read_trace_get_edges(edges_path, node_hash_path, data_dir)

    # 读取边
    with open(edges_path, "r") as f:
        edges = json.load(f)
    src_nodes, dst_nodes = edges[0], edges[1]

    for ts in timestamps:
        metric_tensor = metric_data[ts]
        log_tensor = log_data[ts]
        trace_tensor = trace_data[ts]

        # # 如需恢复真实特征，这两行可注释掉
        # log_tensor = torch.zeros_like(log_tensor)
        # metric_tensor = torch.zeros_like(metric_tensor)
        # 拼接特征向量
        combined_tensor = torch.cat([metric_tensor, log_tensor, trace_tensor], dim=-1)

        # 假设 num_nodes=10，可根据实际调整
        num_nodes = combined_tensor.shape[0]
        graph = dgl.graph((list(src_nodes), list(dst_nodes)), num_nodes=num_nodes)
        graph = dgl.add_self_loop(graph)
        processed_data.append([ts, graph, combined_tensor])

    # 输出目录
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for name in ["samples.pkl", "train_samples.pkl", "test_samples.pkl"]:
        final_path = os.path.join(output_dir, name)
        with open(final_path, "wb") as f:
            pickle.dump(processed_data, f)
        print(f"✅ 数据已成功保存至 {final_path}")

if __name__ == "__main__":
    # 路径可根据实际情况调整
    save_to_pkl(
        "./data/D1/processed_metric.json",
        "./data/D1/processed_log.json",
        "./data/D1/processed_trace.json",
        "./data/D1/test_samples",
        "./data/D1/graphs_info/node_hash.json",
        "/home/fuxian/DataSet/NewDataset_ByService/aiops"
    )