# # import os
# # import pandas as pd
# # import json
# # import numpy as np

# # SERVICES = [
# #     "logservice1", "logservice2",
# #     "mobservice1", "mobservice2",
# #     "redisservice1", "redisservice2",
# #     "dbservice1", "dbservice2",
# #     "webservice1", "webservice2"
# # ]

# # DATA_PATH = "/home/fuxian/ART-master/classified_data/new_GAIA"
# # # MODALITY_FILES = ["low_log.csv", "METRIC_DATA.csv", "TRACE_DATA.csv"]
# # MODALITY_FILES = ["LOG_DATA.csv"]
# # # OUT_JSON = ["processed_log1.json", "processed_metric.json", "processed_trace.json"]
# # OUT_JSON = ["processed_log1.json"]
# # def parse_value(val):
# #     """解析字符串形式的list为Python list"""
# #     if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
# #         try:
# #             return [float(x) for x in val.strip("[]").split(",")]
# #         except Exception:
# #             return [float(x.strip()) for x in val.strip("[]").split(",")]
# #     return val

# # def aggregate_modality(modality_file):
# #     # {timestamp: {service: feature_vector}}
# #     tmp_dict = dict()
# #     feature_dim = None

# #     # 先收集所有数据
# #     for service in SERVICES:
# #         file_path = os.path.join(DATA_PATH, service, modality_file)
# #         if not os.path.exists(file_path):
# #             print(f"Warning: {file_path} not found, skip.")
# #             continue
# #         df = pd.read_csv(file_path)
# #         for _, row in df.iterrows():
# #             timestamp = str(int(row['timestamp']))
# #             features = parse_value(row['value'])
# #             if feature_dim is None:
# #                 feature_dim = len(features)
# #             if timestamp not in tmp_dict:
# #                 tmp_dict[timestamp] = {}
# #             tmp_dict[timestamp][service] = features

# #     # 统一成二维list格式
# #     feature_dict = dict()
# #     for timestamp in sorted(tmp_dict.keys(), key=int):
# #         vector_list = []
# #         for svc in SERVICES:
# #             if svc in tmp_dict[timestamp]:
# #                 vector_list.append(tmp_dict[timestamp][svc])
# #             else:
# #                 # 用全0填充
# #                 vector_list.append([0.0] * feature_dim if feature_dim else [])
# #         feature_dict[timestamp] = vector_list
# #     return feature_dict

# # def main():
# #     for idx, modality_file in enumerate(MODALITY_FILES):
# #         feature_dict = aggregate_modality(modality_file)
# #         out_path = os.path.join(DATA_PATH, OUT_JSON[idx])
# #         with open(out_path, "w") as f:
# #             json.dump(feature_dict, f, indent=2)
# #         print(f"Saved {OUT_JSON[idx]} to {out_path}")

# # if __name__ == "__main__":
# #     main()
# import os
# import pandas as pd
# import json
# import re

# SERVICES = [
#     "logservice1", "logservice2",
#     "mobservice1", "mobservice2",
#     "redisservice1", "redisservice2",
#     "dbservice1", "dbservice2",
#     "webservice1", "webservice2"
# ]

# DATA_PATH = "/home/fuxian/ART-master/classified_data/new_GAIA"
# MODALITY_FILES = ["LOG_DATA.csv"]
# OUT_JSON = ["processed_log1.json"]

# def parse_value(val):
#     """解析字符串形式的list为Python float列表"""
#     if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
#         return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", val)]
#     # 兼容已是list/array场景
#     if isinstance(val, (list, tuple)):
#         return [float(x) for x in val]
#     return [float(val)]

# def aggregate_modality(modality_file):
#     # {timestamp: {service: feature_vector}}
#     tmp_dict = dict()
#     feature_dim = None

#     for service in SERVICES:
#         file_path = os.path.join(DATA_PATH, service, modality_file)
#         if not os.path.exists(file_path):
#             print(f"Warning: {file_path} not found, skip.")
#             continue
#         df = pd.read_csv(file_path)
#         for _, row in df.iterrows():
#             timestamp = str(int(row['timestamp']))
#             features = parse_value(row['value'])
#             if feature_dim is None:
#                 feature_dim = len(features)
#             if timestamp not in tmp_dict:
#                 tmp_dict[timestamp] = {}
#             tmp_dict[timestamp][service] = features

#     # 统一成二维list格式，并转为标准 Python float
#     feature_dict = dict()
#     for timestamp in sorted(tmp_dict.keys(), key=int):
#         vector_list = []
#         for svc in SERVICES:
#             if svc in tmp_dict[timestamp]:
#                 vector_list.append([float(x) for x in tmp_dict[timestamp][svc]])
#             else:
#                 vector_list.append([0.0] * feature_dim if feature_dim else [])
#         feature_dict[timestamp] = vector_list
#     return feature_dict

# def main():
#     for idx, modality_file in enumerate(MODALITY_FILES):
#         feature_dict = aggregate_modality(modality_file)
#         # 保证所有为标准 Python float，兼容 numpy
#         def convert(obj):
#             if isinstance(obj, float):
#                 return obj
#             elif isinstance(obj, (list, tuple)):
#                 return [convert(x) for x in obj]
#             elif hasattr(obj, 'item'):  # numpy scalar
#                 return obj.item()
#             else:
#                 return obj
#         feature_dict = {k: convert(v) for k, v in feature_dict.items()}

#         out_path = os.path.join(DATA_PATH, OUT_JSON[idx])
#         with open(out_path, "w") as f:
#             json.dump(feature_dict, f, indent=2)
#         print(f"Saved {OUT_JSON[idx]} to {out_path}")

# if __name__ == "__main__":
#     main()

import json

def print_first_100_lines(json_path):
    # 尝试以 JSON Lines 格式逐行读取
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            f.seek(0)
            if first_line.lstrip().startswith('{') or first_line.lstrip().startswith('['):
                # 普通 JSON 格式
                data = json.load(f)
                if isinstance(data, dict):
                    keys = list(data.keys())[:100]
                    for i, k in enumerate(keys, 1):
                        print(f"{i:03d} Key: {k}\nValue: {data[k]}\n{'-'*40}")
                elif isinstance(data, list):
                    for i, item in enumerate(data[:100], 1):
                        print(f"{i:03d}: {item}\n{'-'*40}")
                else:
                    print("Unsupported JSON structure.")
            else:
                # JSON Lines 格式
                for i, line in enumerate(f, 1):
                    if i > 100:
                        break
                    print(f"{i:03d}: {line.strip()}\n{'-'*40}")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    json_file = "/home/fuxian/ART-master/classified_data/new_GAIA/processed_log1.json"  
    print_first_100_lines(json_file)