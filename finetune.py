"""
@description: This script is used for fine-tuning
@author: chen ye
"""

# torch pkgs
import torch
from torch.utils.data import DataLoader, ConcatDataset

# common pkgs
import os
import sys
import yaml
import random
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# setting GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Debug] Using device: {device}")
print(f"[Debug] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"[Debug] torch.cuda.current_device() = {torch.cuda.current_device()}")
print(f"[Debug] torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")

# import training required files
from functions.Dataset import Dataset, concat_and_loader, MyDataset, LMDBDataset
from functions.General_train import train_model
from functions.General_evaluate import test_model_detail
from functions.ModelScaffold import model_fn
from functions.DyPositionTransformer import DyPositionTransformer
from functions.Utils import set_full_deterministic_seed

# random seed
SEEDS = list(range(42, 52))

# import training hyperparameter configuration
with open('./config/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)['baseline']

# import path configuration
source_names = ["missense"] # missense | noncoding
feature_names = ["RiNALMo"] # HyenaDNA | NT | ...
backbone_names = ["PositionTransformer"] # ResNet1D | PositionTransformer
transfer_controls = ['finetune', 'scratch']
with open('./config/config_finetune.yaml', 'r', encoding='utf-8') as file:
    finetune_config = yaml.safe_load(file)

for source_name in source_names:
    for feature_name in feature_names:
        tmp_config = finetune_config[feature_name]
        for backbone_name in backbone_names:
            print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {source_name} | {feature_name} | {backbone_name}")
            
            # init task dir
            task_dir = f"./pretrained/{source_name}-{feature_name}/"
            auc_list = []
            aupr_list = []
            groups = []
            indices = []
            metric_path = f"{task_dir}/Finetune.{backbone_name}.metric.summary.txt"

            for SEED in SEEDS:
                set_full_deterministic_seed(SEED)
                for transfer in transfer_controls: 
                    if tmp_config["data_type"] == "lmdb":
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
                    elif tmp_config["data_type"] == "torch":
                        train_bn = MyDataset(f"{tmp_config['data_root']}/{tmp_config['train_bn']}", label=0)
                        train_pt = MyDataset(f"{tmp_config['data_root']}/{tmp_config['train_pt']}", label=1)
                    else:
                        train_bn = MyDataset(f"{tmp_config['data_root']}/{tmp_config['train_bn']}", label=0, single_feature_name='alphag_alt')
                        train_pt = MyDataset(f"{tmp_config['data_root']}/{tmp_config['train_pt']}", label=1, single_feature_name='alphag_alt')
                    # concat data
                    train_data = ConcatDataset([train_bn, train_pt])
                    train_loader, val_loader = concat_and_loader(train_data, split_rate=0.8, batch_size=32, seed=SEED)
                    
                    # test data for zero-shot
                    if tmp_config["data_type"] == "lmdb":
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
                    elif tmp_config["data_type"] == "torch":
                        test_bn = MyDataset(f"{tmp_config['data_root']}/{tmp_config['test_bn']}", label=0)
                        test_pt = MyDataset(f"{tmp_config['data_root']}/{tmp_config['test_pt']}", label=1)
                    else:
                        test_bn = MyDataset(f"{tmp_config['data_root']}/{tmp_config['test_bn']}", label=0, single_feature_name='alphag_alt')
                        test_pt = MyDataset(f"{tmp_config['data_root']}/{tmp_config['test_pt']}", label=1, single_feature_name='alphag_alt')
                    # concat data
                    test_data = ConcatDataset([test_bn, test_pt])
                    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

                    # build model
                    print(f"[INFO] Backbone: {backbone_name} | Model initialization...")
                    # my_model = model_fn(backbone_name=backbone_name, seq_len=tmp_config['seq_len'], embed_dim=tmp_config['embed_dim']).to(device)
                    my_model = DyPositionTransformer(
                                    input_dim=tmp_config['embed_dim'],
                                    num_layers=4,
                                    d_model=256,
                                    num_heads=4,
                                    dff=512,
                                    dropout_rate=0.1,
                                    head_hidden=128,
                                ).to(device)

                    # load pre-trained weight
                    # pretrained_weight_path = f"./pretrained_weight/{source_name}-{feature_name}/source_{backbone_name}_calm_alt_feature_model.pt"
                    pretrained_weight_path = f"./pretrained_weight/{source_name}-{feature_name}/source_{backbone_name}_model.pt"
                    
                    if transfer=='finetune':
                        print(f"[INFO] Loading pretrained weight from: {pretrained_weight_path}...")
                        my_model.load_state_dict(torch.load(pretrained_weight_path, map_location='cpu'))

                    print(f"[INFO] Job inited at {task_dir}...")
                    weight_path = f"{task_dir}/model.pt"

                    # train
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
                    # evaluate
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