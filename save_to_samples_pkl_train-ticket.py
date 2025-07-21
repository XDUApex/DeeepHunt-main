import json
import pickle
import torch
import os
import pandas as pd
import dgl
import copy

def load_json_file(file_path):
    """加载 JSON 并按时间戳排序"""
    with open(file_path, "r") as f:
        data = json.load(f)
    return {int(k): torch.tensor(v) for k, v in sorted(data.items())}  # 确保时间戳是整数并排序

def read_trace_get_edges(edges_path):
    # 读取服务名对应的编号
    with open("./data/D3/graphs_info/node.json", "r") as f:
        node_map = json.load(f)

    data_dir = "../DataSet/train-ticket-original"
    edges_list = []  # 存储 (起点, 终点) 关系

    # 遍历所有 trace.csv 文件
    for date_folder in sorted(os.listdir(data_dir)):  # 按日期顺序遍历
        trace_path = os.path.join(data_dir, date_folder, "trace", "trace.csv")
        if not os.path.exists(trace_path):
            continue

        print(f"正在处理 {trace_path} ...")

        # **优化1：用 dtype 降低内存占用**
        df = pd.read_csv(trace_path, usecols=["service_name", "span_id", "parent_id"], dtype=str)

        # **优化2：创建 `span_id -> service_name` 的快速查找表**
        span_to_service = df.set_index("span_id")["service_name"].to_dict()

        # **优化3：用 Pandas 向量化计算起点终点**
        df["parent_service"] = df["parent_id"].map(span_to_service)
        
        # 过滤掉找不到父节点的行
        df = df.dropna(subset=["parent_service"])

        # 映射 service_name 到 node_map
        df["source"] = df["parent_service"].map(node_map)
        df["target"] = df["service_name"].map(node_map)

        # 过滤掉无效映射（可能 `service_name` 不在 `node.json` 里）
        df = df.dropna(subset=["source", "target"])

        # **优化4：批量存储，避免 Python for 循环**
        edges_list.extend(zip(df["source"].astype(int), df["target"].astype(int)))

    # **优化5：去重并保持顺序**
    edges_df = pd.DataFrame(edges_list, columns=["source", "target"]).drop_duplicates()

    # **转换为最终格式**
    edges_json = [edges_df["source"].tolist(), edges_df["target"].tolist()]

    # **保存 JSON**
    edges_path = "./data/D3/edges.json"
    with open(edges_path, "w") as f:
        json.dump(edges_json, f, indent=4)

    print(f"✅ 处理完成！共保存 {len(edges_json[0])} 条去重后调用关系到 {edges_path}")

    
    

def save_to_pkl(metric_path, log_path, trace_path, output_path):
    """加载三个 JSON 文件并合并，最终保存为 .pkl"""
    
    # 读取 JSON 数据并转成 PyTorch Tensor
    metric_data = load_json_file(metric_path)  # {timestamp: tensor(71, 10, 183)}
    log_data = load_json_file(log_path)        # {timestamp: tensor(71, 10, log_dim)}
    trace_data = load_json_file(trace_path)    # {timestamp: tensor(71, 10, trace_dim)}

    # 取得所有的时间戳（确保时间对齐）
    timestamps = list(metric_data.keys())

    processed_data = []  # 存放最终的数据

    # 读取trace中的调用关系，生成边文件edges.json
    edges_dir = os.path.dirname(metric_path)
    edges_path = os.path.join(edges_dir, "edges.json")
    read_trace_get_edges(edges_path)

    for ts in timestamps:
        metric_tensor = metric_data[ts]  # (10, 183)
        log_tensor = log_data[ts]        # (10, log_dim)
        trace_tensor = trace_data[ts]    # (10, trace_dim)

        # 拼接特征向量 (10, 183+log_dim+trace_dim)
        combined_tensor = torch.cat([metric_tensor, log_tensor, trace_tensor], dim=-1)

        with open(edges_path, "r") as f:
            edges = json.load(f)  # 解析 JSON 文件

        # 拆分起点和终点
        src_nodes = edges[0]  # 第一行是所有起点
        dst_nodes = edges[1]  # 第二行是所有终点
  
        graph = dgl.graph((list(src_nodes), list(dst_nodes)), num_nodes=24)
        graph = dgl.add_self_loop(graph)

        # 3️⃣ 组合成最终数据格式
        processed_data.append([ts, graph, combined_tensor])

    # 复制 processed_data 几份
    copied_data  = []
    for _ in range(5):  # 假设你想复制3次
        copied_data.extend(copy.deepcopy(processed_data))  # 深拷贝每个元素

    # 获取输出路径的目录部分
    output_dir = os.path.dirname(output_path)
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 保存为 .pkl
    output_path_1 = os.path.join(output_dir, "samples.pkl")
    with open(output_path_1, "wb") as f:
        pickle.dump(copied_data, f)

    # 分割数据集为训练集和测试集
    # total_samples = len(copied_data)
    # train_size = int(total_samples * 0.5)  # 训练集大小为 7/10
    train_data = copied_data
    test_data = processed_data

                    # # 保存为 .pkl
                    # output_path_1 = os.path.join(output_dir, "samples.pkl")
                    # with open(output_path_1, "wb") as f:
                    #     pickle.dump(processed_data, f)

                    # # 分割数据集为训练集和测试集

                    # train_data = processed_data
                    # test_data = processed_data

    # 保存训练集为 train_samples.pkl
    output_path_2 = os.path.join(output_dir, "train_samples.pkl")
    with open(output_path_2, "wb") as f:
        pickle.dump(train_data, f)

    # 保存测试集为 test_samples.pkl
    output_path_3 = os.path.join(output_dir, "test_samples.pkl")
    with open(output_path_3, "wb") as f:
        pickle.dump(test_data, f)

    # output_path_2 = os.path.join(output_dir, "train_samples.pkl")
    # with open(output_path_2, "wb") as f:
    #     pickle.dump(processed_data, f)

    # output_path_3 = os.path.join(output_dir, "test_samples.pkl")
    # with open(output_path_3, "wb") as f:
    #     pickle.dump(processed_data, f)


    print(f"✅ 数据已成功保存至 {output_path}")

# 调用函数
save_to_pkl(
    "./data/D3/processed_metric.json",
    "./data/D3/processed_log.json",
    "./data/D3/processed_trace.json",
    "./data/D3/samples/samples.pkl"
)
