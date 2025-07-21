#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import json
import os

def regenerate_node_hash():
    """
    重新生成node_hash.pkl文件
    基于新的服务实例列表创建节点哈希映射
    """
    
    # 新的服务实例列表
    instances = [
        "adservice", "adservice-0", "adservice-1", "adservice-2", "adservice2-0",
        "cartservice", "cartservice-0", "cartservice-1", "cartservice-2", "cartservice2-0",
        "checkoutservice", "checkoutservice-0", "checkoutservice-1", "checkoutservice-2", "checkoutservice2-0",
        "currencyservice", "currencyservice-0", "currencyservice-1", "currencyservice-2", "currencyservice2-0",
        "emailservice", "emailservice-0", "emailservice-1", "emailservice-2", "emailservice2-0",
        "frontend", "frontend-0", "frontend-1", "frontend-2", "frontend2-0",
        "paymentservice", "paymentservice-0", "paymentservice-1", "paymentservice-2", "paymentservice2-0",
        "productcatalogservice", "productcatalogservice-0", "productcatalogservice-1", "productcatalogservice-2", "productcatalogservice2-0",
        "recommendationservice", "recommendationservice-0", "recommendationservice-1", "recommendationservice-2", "recommendationservice2-0",
        "redis-cart-0", "redis-cart2-0",
        "shippingservice", "shippingservice-0", "shippingservice-1", "shippingservice-2", "shippingservice2-0"
    ]
    
    # 创建节点哈希字典：实例名 -> 索引
    node_hash = {}
    for i, instance in enumerate(instances):
        node_hash[instance] = i
    
    # 文件路径
    pkl_path = "/home/fuxian/DeepHunt-main/data/D1/graphs_info/node_hash.pkl"
    json_path = "/home/fuxian/DeepHunt-main/data/D1/graphs_info/node_hash.json"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    
    # 保存到pickle文件
    with open(pkl_path, 'wb') as f:
        pickle.dump(node_hash, f)
    
    # 保存到JSON文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(node_hash, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 成功创建 {pkl_path}")
    print(f"✅ 成功创建 {json_path}")
    print(f"📊 节点总数: {len(node_hash)}")
    print(f"🔍 前5项内容:")
    for i, (instance, idx) in enumerate(list(node_hash.items())[:5]):
        print(f"   {i+1}. {instance} : {idx}")
    
    # 验证文件是否正确创建
    pkl_success = False
    json_success = False
    
    try:
        with open(pkl_path, 'rb') as f:
            loaded_pkl = pickle.load(f)
        print(f"✅ PKL文件验证成功，数据类型: {type(loaded_pkl)}, 长度: {len(loaded_pkl)}")
        pkl_success = True
    except Exception as e:
        print(f"❌ PKL文件验证失败: {e}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded_json = json.load(f)
        print(f"✅ JSON文件验证成功，数据类型: {type(loaded_json)}, 长度: {len(loaded_json)}")
        json_success = True
    except Exception as e:
        print(f"❌ JSON文件验证失败: {e}")
    
    return pkl_success and json_success

def show_node_mapping():
    """
    显示完整的节点映射关系
    """
    instances = [
        "adservice", "adservice-0", "adservice-1", "adservice-2", "adservice2-0",
        "cartservice", "cartservice-0", "cartservice-1", "cartservice-2", "cartservice2-0",
        "checkoutservice", "checkoutservice-0", "checkoutservice-1", "checkoutservice-2", "checkoutservice2-0",
        "currencyservice", "currencyservice-0", "currencyservice-1", "currencyservice-2", "currencyservice2-0",
        "emailservice", "emailservice-0", "emailservice-1", "emailservice-2", "emailservice2-0",
        "frontend", "frontend-0", "frontend-1", "frontend-2", "frontend2-0",
        "paymentservice", "paymentservice-0", "paymentservice-1", "paymentservice-2", "paymentservice2-0",
        "productcatalogservice", "productcatalogservice-0", "productcatalogservice-1", "productcatalogservice-2", "productcatalogservice2-0",
        "recommendationservice", "recommendationservice-0", "recommendationservice-1", "recommendationservice-2", "recommendationservice2-0",
        "redis-cart-0", "redis-cart2-0",
        "shippingservice", "shippingservice-0", "shippingservice-1", "shippingservice-2", "shippingservice2-0"
    ]
    
    print("🗂️  完整的节点映射关系:")
    print("=" * 50)
    
    current_service = ""
    for i, instance in enumerate(instances):
        service_name = instance.split('-')[0].split('2')[0]  # 提取服务名
        if service_name != current_service:
            if current_service:  # 不是第一个服务时添加分隔
                print()
            current_service = service_name
            print(f"📦 {service_name.upper()}:")
        
        print(f"   {instance:<25} -> {i}")

if __name__ == "__main__":
    print("🚀 开始重新生成 node_hash.pkl 和 node_hash.json 文件...")
    print()
    
    # 显示节点映射关系
    show_node_mapping()
    print()
    
    # 重新生成文件
    success = regenerate_node_hash()
    
    if success:
        print("\n🎉 node_hash.pkl 和 node_hash.json 文件重新生成完成！")
        print("📁 生成的文件:")
        print("   - node_hash.pkl  (Python pickle格式)")
        print("   - node_hash.json (JSON格式，可读性更好)")
    else:
        print("\n💥 生成过程中出现错误，请检查！")