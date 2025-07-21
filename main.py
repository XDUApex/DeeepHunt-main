import yaml
import os
import pickle
import json
import pandas as pd

from utils.public_functions import load_samples
from models.train import train
from models.evaluation import get_eval_df
import warnings
warnings.filterwarnings('ignore')

# config
# 1.读取配置文件GAIA.yaml
dataset = 'AIOPS'
config_file = f'{dataset}.yaml'
config = yaml.load(open(f'config_D12/{config_file}', 'r'), Loader=yaml.FullLoader)
print('load config.')
# 2.创建结果保存的目录res/GAIA
res_dir = f'res_test/{dataset}'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
# 3.定义各种结果的文件路径
naive_model_path = f'{res_dir}/naive_model.pkl'     # 基础模型序列化结果
fd_model_path = f'{res_dir}/fd_model.pkl'           # 反馈模型序列化结果
test_df_path = f'{res_dir}/test_df.csv'             # 基础模型的测试结果（CSV 文件）
fd_test_df_path = f'{res_dir}/fd_test_df.csv'       # 反馈模型的测试结果（CSV 文件）
res_path = f'{res_dir}/res.json'                    # 模型评估结果的 JSON 文件

# load samples
# 加载样本数据
train_samples, test_samples = load_samples(config['path']['sample_dir'])
print('load samples.')


# # 去模态（直接处理为0版本）
# for train_sample in train_samples:
#     for data in train_sample[2]:
#         data[24:798] = 0

# for test_sample in test_samples:
#     for data in test_sample[2]:
#         data[24:798] = 0


input_samples = train_samples if config['train_samples_num'] == 'whole' else train_samples[: config['train_samples_num']]


# train naive model
# 训练基础模型
print(f"train samples num: {len(input_samples)}, aug_multiple: {config['model_param']['aug_multiple']}")
model = train(input_samples, config['model_param'])
print('naive model trained.')
with open(naive_model_path, 'wb') as f:
    pickle.dump(model, f)

# train feedback model and use the model with feedback for evaluation
# 训练反馈模型
cases = pd.read_csv(config['path']['case_dir'])
print("samples length:", len(test_samples))
# 测试+训练反馈模型+使用反馈模型测试+模型评估
test_df, res_dict = get_eval_df(model, cases, test_samples, config)

# save the result
# 保存结果
# with open(fd_model_path, 'wb') as f:
#     pickle.dump(fd_model, f)
test_df.to_csv(test_df_path, index=False)
# fd_test_df.to_csv(fd_test_df_path, index=False)
with open(res_path, 'w') as f:
    json.dump(res_dict, f)
