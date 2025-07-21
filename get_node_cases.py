# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from multiprocessing import Pool
# import re
# import utils_load as U
# import os
# import pickle
# import json
# from config import train_ticket
# import os
# import json
# from typing import *
# from datetime import datetime
# from tqdm import tqdm
# import pandas as pd
# import utils_load as U



# fault_output_file=f"./data/D1"        # 抽取groundtruch为所需cases.csv文件的存储目录

# def process_gt_df(gt_df: pd.DataFrame) -> pd.DataFrame:     # 抽取groundtruch为所需cases.csv文件的函数
#     # 创建原始数据的副本
#     result_df = gt_df.copy()
#     gt_df["level"] = "WARNING"
#     # # 计算中间时间戳并添加到副本中（其实就是原始groundtruth的st_time注入时间）
#     # result_df['st_time'] = (result_df['st_time'] + result_df['ed_time']) / 2

#     # 重命名列并提取需要的字段
#     result_df = gt_df.rename(columns={
#         'st_time': 'timestamp',
#         'instance': 'cmdb_id'       # 具体的实例作为cmdb_id，符合DeepHunt要求
#     })[['timestamp', 'level', 'cmdb_id', 'failure_type']]
    
#     # 返回新的 DataFrame
#     return result_df


# def __get_files__(root: str, keyword: str):
#         r"""
#         :root: 搜索的根目录
#         :keyword：文件含有的关键词
#         :return 含有关键字的文件列表['file1', 'file2']
#         """
#         # print(root)
#         # print(keyword)
#         files = []
#         for dirpath, _, filenames in os.walk(root):
#             for filename in filenames:
#                 if filename.find(keyword) != -1:
#                     # print(os.path.join(dirpath, filename))
#                     files.append(os.path.join(dirpath, filename))

#         return files

# def __add_by_date__(files, dates):
#         r"""
#         :files 一堆含有时间信息的文件列表
#         :dates 日期
#         :return 返回一个按dates排的二维列表
#         >>> files = ["05-02/demo.csv", "05-03/demo.csv"]
#         >>> dates = ["05-02", "05-03"]
#         >>> rt = self.__add_by_date(files, dates)
#         >>> # output: rt = [["05-02/demo.csv"], ["05-03/demo.csv"]]
#         """
#         _files = [[] for _ in dates]
#         # print(_files)
#         for index, date in enumerate(dates):
#             for file in files:
#                 if file.find(date) != -1:
#                     _files[index].append(file)
#         # print(_files)
#         return _files

# def __load_df__(file_list: List[str], is_json=False):
#     r"""
#     :file_list 同一 columns 的csv文件列表
#     :is_json 是否为json文件
#     :return Dataframe
#     """
#     if is_json:
#         dfs = [
#             pd.read_json(file, keep_default_dates=False) for file in tqdm(file_list)
#         ]
#     else:
#         dfs = [pd.read_csv(file) for file in tqdm(file_list)]
#     return pd.concat(dfs)

# def __load_groundtruth_df__(file_list):
#     groundtruth_df = __load_df__(file_list).rename(
#         columns={
#             "anomaly_type": "failure_type",
#         }
#     )

#     groundtruth_df = groundtruth_df.rename(columns={"service": "root_cause"})
#     duration = 600
#     from datetime import datetime

#     groundtruth_df["st_time"] = groundtruth_df["st_time"].apply(
#         lambda x: int(datetime.strptime(
#             x.split(".")[0], "%Y-%m-%d %H:%M:%S"
#         ).timestamp())
#     )
#     groundtruth_df["ed_time"] = groundtruth_df["st_time"] + duration    # 注入时间(原st_time)左右拓展时间窗口作为st_time和ed_time
#     # groundtruth_df["st_time"] = groundtruth_df["st_time"] - duration
#     groundtruth_df = groundtruth_df.reset_index(drop=True)
#     # groundtruth_df.loc[:, "failure_type"] = groundtruth_df["failure_type"].apply(
#     #     lambda x: self.ANOMALY_DICT[x]
#     # )
#     return groundtruth_df.loc[
#         :, ["st_time", "ed_time", "failure_type", "root_cause", "instance"]
#     ]

# def load():
#     # 先加载config.py中所有的实例，保存为data/gaia/graphs_info/node_hash.pkl
#     graphs_info_path = os.path.join(fault_output_file, "graphs_info")
#     os.makedirs(graphs_info_path, exist_ok=True)

#     json_file_path = os.path.join(graphs_info_path, "node.json")
#     pkl_file_path = os.path.join(graphs_info_path, "node_hash.pkl")

#     instances = train_ticket["instances"].split()
#     instances_dict = {f"{name}": i for i, name in enumerate(instances)}
#     # 保存为 JSON 文件
#     with open(json_file_path, "w") as json_file:
#         json.dump(instances_dict, json_file, indent=4)

#     # 保存为 .pkl 文件
#     with open(pkl_file_path, "wb") as pkl_file:
#         pickle.dump(instances_dict, pkl_file)

#     # read run_table
#     groundtruth_files = __get_files__(train_ticket["dataset_dir"], "groundtruth.csv")
#     # metric_files = __get_files__(self.dataset_dir, "docker_")
#     # self.__get_kpi_list__(metric_files)
#     dates = [
#         "2025-03-03 18_57_36",
#         "2025-03-04 13_49_54",
#         "2025-03-04 16_07_12",
#         "2025-03-04 18_15_07",
#         "2025-03-04 20_17_18",
#         "2025-03-05 09_29_09",
#         "2025-03-05 11_35_31",
#         "2025-03-05 13_40_20",
#         "2025-03-05 16_42_21",
#         "2025-03-05 21_03_15"
#     ]
#     groundtruths = __add_by_date__(groundtruth_files, dates)
#     # metrics = self.__add_by_date__(metric_files, dates)
#     all_gt_df = []  # 用于存储所有 gt_df 的列表
#     for index, date in enumerate(dates):
#         U.notice(f"Loading... {date}")
#         gt_df = __load_groundtruth_df__(groundtruths[index])
#         all_gt_df.append(gt_df)  # 将当前 gt_df 添加到列表中
#         # metric_df = self.__load_metric_df__(metrics[index])
#         # self.__load_labels__(gt_df)
#         # self.__load_metric__(gt_df, metric_df)

#     # 将所有 gt_df 合并成一个 DataFrame,并处理后保存
#     combined_gt_df = pd.concat(all_gt_df, ignore_index=True)
#     processed_gt_df = process_gt_df(combined_gt_df)
#     output_path = os.path.join(fault_output_file, 'cases_test.csv')
#     processed_gt_df.to_csv(output_path, index=False)

# load()

import os
import pandas as pd
from datetime import datetime

def process_gt_df(gt_df: pd.DataFrame) -> pd.DataFrame:
    # 时间字符串转时间戳（秒）
    def str_to_timestamp(x):
        return int(datetime.strptime(x.split(".")[0], "%Y-%m-%d %H:%M:%S").timestamp())

    result_df = gt_df.copy()
    result_df["level"] = "WARNING"
    result_df["timestamp"] = result_df["st_time"].apply(str_to_timestamp)
    result_df = result_df.rename(columns={
        "instance": "cmdb_id",
        "anomaly_type": "failure_type"
    })
    # 保留目标字段顺序
    return result_df[["timestamp", "level", "cmdb_id", "failure_type"]]

def extract_all_cases(root_dir: str, output_path: str):
    all_gt_df = []
    # 遍历所有服务子目录
    for service in os.listdir(root_dir):
        service_path = os.path.join(root_dir, service)
        if not os.path.isdir(service_path):
            continue
        # 遍历所有日期子目录
        for date_dir in os.listdir(service_path):
            date_path = os.path.join(service_path, date_dir)
            if not os.path.isdir(date_path):
                continue
            gt_file = os.path.join(date_path, "groundtruth.csv")
            if os.path.exists(gt_file):
                gt_df = pd.read_csv(gt_file)
                all_gt_df.append(gt_df)
    if not all_gt_df:
        print("No groundtruth.csv files found.")
        return
    combined_gt_df = pd.concat(all_gt_df, ignore_index=True)
    processed_gt_df = process_gt_df(combined_gt_df)
    processed_gt_df.to_csv(output_path, index=False)
    print(f"cases.csv saved to {output_path}")

if __name__ == "__main__":
    # 修改你的根目录和输出目录
    root_dir = "/home/fuxian/DataSet/NewDataset_ByService/aiops"  # 根目录
    output_path = "./data/D1/cases_test.csv"  # 输出路径
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    extract_all_cases(root_dir, output_path)