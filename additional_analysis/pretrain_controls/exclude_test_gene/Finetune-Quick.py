"""
@description: A quick finetune script
@author: chen ye
"""

# torch相关 
import torch
from torch.utils.data import DataLoader, ConcatDataset

# 常规包
import os
import sys
import yaml
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# 配置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Debug] Using device: {device}")
print(f"[Debug] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"[Debug] torch.cuda.current_device() = {torch.cuda.current_device()}")
print(f"[Debug] torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")

# 导入训练所需文件
from functions.Dataset import Dataset, concat_and_loader, MyDataset, LMDBDataset
from functions.General_train import train_model
from functions.General_evaluate import test_model_detail
from functions.ModelScaffold import model_fn
from functions.DyPositionTransformer import DyPositionTransformer
from functions.Utils import set_full_deterministic_seed

# 控制随机种子
SEEDS = list(range(42, 52))

# 导入训练超参数配置
with open('../config/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)['baseline']

# 导入微调训练数据路径配置
with open('../config/config_finetune.yaml', 'r', encoding='utf-8') as file:
    finetune_config = yaml.safe_load(file)


def finetune_func(source_name, pretrained_task, feature_name, backbone_name, transfer):
    print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\
        | {source_name} | {feature_name} | {backbone_name}")

    # 配置
    tmp_config = finetune_config[feature_name]

    # 初始化任务目录
    task_dir = f"../pretrained/{pretrained_task}/"
    print(f"[INFO] Job starts at: {task_dir}")
    auc_list = []
    aupr_list = []
    groups = []
    indices = []
    metric_path = f"{task_dir}/Finetune.{backbone_name}.metric.summary.txt"

    # 1 导入训练数据
    if tmp_config["data_type"] == "lmdb": # lmdb 数据类型
        train_bn = LMDBDataset(
                lmdb_path=f"{tmp_config['data_root']}/{tmp_config['train_bn']}",
                feature_keys=[tmp_config['feature_name']],
                preload=False,
                label=0
            )
        train_pt = LMDBDataset(
                lmdb_path=f"{tmp_config['data_root']}/{tmp_config['train_pt']}",
                feature_keys=[tmp_config['feature_name']],
                preload=False,
                label=1
            )
    elif tmp_config["data_type"] == "torch": # torch 数据类型
        train_bn = MyDataset(f"{tmp_config['data_root']}/{tmp_config['train_bn']}", label=0)
        train_pt = MyDataset(f"{tmp_config['data_root']}/{tmp_config['train_pt']}", label=1)
    else: # single feature, 没有使用字典包装
        train_bn = MyDataset(f"{tmp_config['data_root']}/{tmp_config['train_bn']}", label=0, single_feature_name='alphag_alt')
        train_pt = MyDataset(f"{tmp_config['data_root']}/{tmp_config['train_pt']}", label=1, single_feature_name='alphag_alt')
    # 合并数据
    train_data = ConcatDataset([train_bn, train_pt])

    # 2 导入测试集数据
    if tmp_config["data_type"] == "lmdb": # lmdb 数据类型
        test_bn = LMDBDataset(
                lmdb_path=f"{tmp_config['data_root']}/{tmp_config['test_bn']}",
                feature_keys=[tmp_config['feature_name']],
                preload=False,
                label=0
            )
        test_pt = LMDBDataset(
                lmdb_path=f"{tmp_config['data_root']}/{tmp_config['test_pt']}",
                feature_keys=[tmp_config['feature_name']],
                preload=False,
                label=1
            )
    elif tmp_config["data_type"] == "torch": # torch 数据类型
        test_bn = MyDataset(f"{tmp_config['data_root']}/{tmp_config['test_bn']}", label=0)
        test_pt = MyDataset(f"{tmp_config['data_root']}/{tmp_config['test_pt']}", label=1)
    else:
        test_bn = MyDataset(f"{tmp_config['data_root']}/{tmp_config['test_bn']}", label=0, single_feature_name='alphag_alt')
        test_pt = MyDataset(f"{tmp_config['data_root']}/{tmp_config['test_pt']}", label=1, single_feature_name='alphag_alt')
    
    # 合并数据
    test_data = ConcatDataset([test_bn, test_pt])
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)


    for SEED in SEEDS: # 重复随机种子
        set_full_deterministic_seed(SEED)
        # 每次SEED 使用不同的划分
        train_loader, val_loader = concat_and_loader(train_data, split_rate=0.8, batch_size=32, seed=SEED)
        
        # 构造模型
        print(f"[INFO] Backbone: {backbone_name} | Model initialization...")
        if backbone_name == 'PositionTransformer':
            my_model = DyPositionTransformer(
                        input_dim=tmp_config['embed_dim'],
                        num_layers=4,
                        d_model=256,
                        num_heads=4,
                        dff=512,
                        dropout_rate=0.1,
                        head_hidden=128,
                    ).to(device)
        else:
            my_model = model_fn(
                    backbone_name=backbone_name,
                    seq_len=tmp_config['seq_len'],
                    embed_dim=tmp_config['embed_dim']).to(device)

        # 导入预训练权重作为模型起点
        pretrained_weight_path = f"../pretrained/{pretrained_task}/source_{backbone_name}_model.pt"
        print(f"[INFO] Loading pretrained weight from: {pretrained_weight_path}...")
        my_model.load_state_dict(torch.load(pretrained_weight_path, map_location='cpu'))
        
        # 微调权重保存路径
        weight_path = f"{task_dir}/finetune.model.pt"

        # 训练
        print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Start training...")
        model, train_losses, val_losses = train_model(
                                                        model=my_model,
                                                        train_loader=train_loader, 
                                                        val_loader=val_loader, 
                                                        epochs=config['epochs'], 
                                                        lr=config['lr'], 
                                                        patience=config['patience'],
                                                        save_path=weight_path, 
                                                        device='cuda',
                                                        strategy_mode=transfer,
                                                        feature_name=tmp_config['feature_name']
                                                    )
        # 评估
        auc, aupr = test_model_detail(
                                        model=model,
                                        dataloader=test_loader, 
                                        device=device, 
                                        feature_name=tmp_config['feature_name']
                                      )
        auc_list.append(auc)
        aupr_list.append(aupr)
        groups.append(transfer)
        indices.append(SEED)
        print(f"[SUCCESS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Training completed!")
    metrics_df = pd.DataFrame({'group': groups, 'seed': indices, 'AUC': auc_list, 'AUPR': aupr_list})
    metrics_df.to_csv(metric_path, sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quick Run Finetune Script')
    parser.add_argument('--source_name', type=str, help='noncoding or missense')
    parser.add_argument('--feature_name', type=str, help='CaLM or GPN-MSA')
    parser.add_argument('--pretrained_task', type=str, help='CaLM-shuffle')
    parser.add_argument('--backbone_name', type=str, help='PositionTransformer or ResNet1D')
    parser.add_argument('--transfer', type=str, help='finetune or scratch')
    args = parser.parse_args()
    finetune_func(args.source_name, args.pretrained_task, args.feature_name, args.backbone_name, args.transfer)
