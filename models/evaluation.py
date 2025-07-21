import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from utils.public_functions import load_init
from models.layers import ModalLoss
from models.rc_scorer import naive_scorer, feedback, get_feedback_samples
import time

# test: output the loss for each node
def test(model, samples, node_hash, feat_span):

    test_start = time.time()
    count = 0
    mse = nn.MSELoss()
    with torch.no_grad():
        # 只统计当前 g 的节点
        res = {}
        for ts, g, feats in samples:
            count += 1
            outputs = model.transform(g, feats)
            # 当前 g 的节点 id 范围
            for local_idx in range(outputs.shape[0]):
                # 获取全局名称（如果有）
                global_name = None
                for k, v in node_hash.items():
                    if v == local_idx:
                        global_name = k
                        break
                if global_name is None:
                    global_name = f"node_{local_idx}"
                loss = mse(outputs[local_idx], feats[local_idx])
                if global_name not in res:
                    res[global_name] = []
                res[global_name].append((ts, loss.item()))
    loss_df = pd.concat([pd.DataFrame(res[node], columns=['timestamp', node]).set_index('timestamp') for node in res], axis=1).reset_index()
    test_time = time.time() - test_start
    avg_test_time = test_time / count if count > 0 else 0
    print(f'Testing completed in {test_time:.8f} seconds')
    print(f'Average time per test: {avg_test_time:.8f} seconds')
    return loss_df

def fd_test(test_cases, fd_model, samples, node_hash, window_size):
    fd_test_df = test_cases.copy(deep=True).reset_index(drop=True)
    dataloader = iter(get_feedback_samples(test_cases, samples, node_hash, window_size, batch_size=1))
    for case_id in range(len(fd_test_df)):
        batched_graphs,  batched_feats, _ = next(dataloader)
        scores = fd_model(batched_graphs, batched_feats).tolist()

        final_score_map = {i: scores[i] for i in range(len(scores))}
        ranks = sorted([(node, final_score_map[node_hash[node]]) for node in node_hash], key=lambda x: x[1], reverse=True)
        for i in range(len(ranks)):
            fd_test_df.loc[case_id, f'Top{i+1}'] = '%s:%s' % ranks[i]
    return fd_test_df

# train the feedback model and evaluate the model
def get_eval_df(model, cases, samples, config):
    res_dict = dict()
    node_hash, _, _ = load_init(config['path']['graph_dir'])
    fd_num = config['feedback']['sample_num']
    # test_index = - int(len(cases) * 0.3) # split the test set

    # 测试数据分割
    n_cases = len(cases)  # 总数据量
    start_idx = int(n_cases * 0)  # 从 25% 处开始
    end_idx = int(n_cases * 1)  # 到 75% 处结束

    if isinstance(fd_num, int):
        fd_cases, test_cases = cases.iloc[: fd_num], cases.iloc[start_idx:end_idx]
    elif isinstance(fd_num, float) and (0 < fd_num < 1):
        n_cases = len(cases)
        split_pos = int(n_cases * fd_num)
        fd_cases, test_cases = cases.iloc[: split_pos], cases.iloc[start_idx:end_idx]
    else:
        raise Exception('invalid sample_num')
    print('feedback sample nums: %s; test sample nums: %s' % (len(fd_cases), len(test_cases)))
    
    print('Using the naive model / the model with default scorer for evaluation.')
    loss_df = test(model, samples, node_hash, config['model_param']['feat_span'])
    test_df = naive_scorer(test_cases, samples, loss_df, node_hash, window_size=config['feedback']['window_size'], pre=0, suc=-0)
    res_all = evaluation(test_df, 5)
    calculate_mrr(test_df)
    test_df = naive_scorer(cases, samples, loss_df, node_hash, window_size=config['feedback']['window_size'], pre=0, suc=-0)
    res_dict['naive_res_all'] = res_all.tolist()
    print('res_all: ', res_all)

    # print('Using the model with feedback for evaluation.')
    # fd_model = feedback(model, fd_cases, samples, node_hash, config['feedback'])
    # fd_test_df = fd_test(test_cases, fd_model, samples, node_hash, config['feedback']['window_size'])
    # res_all = evaluation(fd_test_df, 5)
    # res_dict['fd_res_all'] = res_all.tolist()
    # print('res_all: ', res_all)

    return test_df, res_dict

def evaluation(cases, k=5):
    topks = np.zeros(k)
    for _, case in cases.iterrows():
        for i in range(k):
            top_val = case.get(f'Top{i+1}', '')  # 获取 Top 值，若不存在则返回 ''
            if isinstance(top_val, str) and str(case['cmdb_id']) in top_val:
                topks[i:] += 1
                break
    return np.round(topks / len(cases), 8)

def calculate_mrr(cases):
    reciprocal_ranks = []  # 用于存储 1/Rank_i 的列表

    for _, case in cases.iterrows():
        found = False
        for i in range(10):  # Top-10 范围内查找
            top_val = case.get(f'Top{i+1}', '')  # 获取 Top 值，若不存在则返回 ''
            if isinstance(top_val, str) and str(case['cmdb_id']) in top_val:
                reciprocal_ranks.append(1 / (i + 1))  # 计算 MRR 贡献
                found = True
                break
        if not found:
            reciprocal_ranks.append(0)  # 若根因不在 Top-10，贡献为 0
    
    mrr = np.round(np.mean(reciprocal_ranks), 8)  # 计算 MRR
    print(f'MRR: {mrr}')
    return mrr
