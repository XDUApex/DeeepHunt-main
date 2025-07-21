import pickle

pkl_path = "/home/fuxian/DeepHunt-main/data/D1/graphs_info/node_hash/node_hash1.pkl"
with open(pkl_path, "rb") as f:
    obj = pickle.load(f)

print("对象类型:", type(obj)) 
if isinstance(obj, dict):
    print("字典长度:", len(obj))
    print("前5项内容示例:")
    for i, (k, v) in enumerate(obj.items()):
        print(f"{i+1}. {k} : {v}")
        if i >= 4:
            break
elif isinstance(obj, list):
    print("列表长度:", len(obj))
    print("前5项内容示例:", obj[:60])
else:
    print("内容示例:", obj)