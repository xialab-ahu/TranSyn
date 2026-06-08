"""
@description: This script is used for pre-training
@author: chen ye
"""

# torch related imports
import torch
from torch.utils.data import DataLoader, ConcatDataset

# general packages
import os
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Debug] Using device: {device}")
print(f"[Debug] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"[Debug] torch.cuda.current_device() = {torch.cuda.current_device()}")
print(f"[Debug] torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")

# import training required files
from functions.Dataset import concat_and_loader, MyDataset, LMDBDataset
from functions.General_train import train_model
from functions.General_evaluate import test_model_detail
from functions.DyPositionTransformer import DyPositionTransformer
from functions.Utils import set_full_deterministic_seed

# control random seed
SEED = 42
set_full_deterministic_seed(SEED)

# import training hyperparameter configuration
with open('./config/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)['baseline']
task_names = ["missense-RiNALMo"] 
backbone_names = ["PositionTransformer"]  # ResNet1D | PositionTransformer

# import pre-training data path configuration
with open('./config/config_pretrain.yaml', 'r', encoding='utf-8') as file:
    pretrain_config = yaml.safe_load(file)

for task_name in task_names:
    tmp_config = pretrain_config[task_name]
    for backbone_name in backbone_names:
        print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Loading data...")
        
        # import source domain data
        if tmp_config["source_data_type"] == "lmdb":  # lmdb data type
            src_train_bn = LMDBDataset(
                lmdb_path=f"{tmp_config['source_root']}/{tmp_config['source_bn']}",
                feature_keys=[tmp_config['feature_name']],
                preload=False,
                label=0
            )
            src_train_pt = LMDBDataset(
                lmdb_path=f"{tmp_config['source_root']}/{tmp_config['source_pt']}",
                feature_keys=[tmp_config['feature_name']],
                preload=False,
                label=1
            )
        else:  # torch data type
            src_train_bn = MyDataset(f"{tmp_config['source_root']}/{tmp_config['source_bn']}", label=0)
            src_train_pt = MyDataset(f"{tmp_config['source_root']}/{tmp_config['source_pt']}", label=1)
            # uncomment when use alphaG feature
            # src_train_bn = MyDataset(f"{tmp_config['source_root']}/{tmp_config['source_bn']}", label=0, single_feature_name='alphag_alt')
            # src_train_pt = MyDataset(f"{tmp_config['source_root']}/{tmp_config['source_pt']}", label=1, single_feature_name='alphag_alt')
        
        # merge data
        train_data = ConcatDataset([src_train_bn, src_train_pt])
        train_loader, val_loader = concat_and_loader(train_data, split_rate=0.8, batch_size=32, seed=SEED)
        
        # import target domain test data for zero-shot evaluation
        if tmp_config["target_data_type"] == "lmdb":  # lmdb data type
            tgt_test_bn = LMDBDataset(
                lmdb_path=f"{tmp_config['target_root']}/{tmp_config['target_bn']}",
                feature_keys=[tmp_config['feature_name']],
                preload=False,
                label=0
            )
            tgt_test_pt = LMDBDataset(
                lmdb_path=f"{tmp_config['target_root']}/{tmp_config['target_pt']}",
                feature_keys=[tmp_config['feature_name']],
                preload=False,
                label=1
            )
        else:  # torch data type
            tgt_test_bn = MyDataset(f"{tmp_config['target_root']}/{tmp_config['target_bn']}", label=0)
            tgt_test_pt = MyDataset(f"{tmp_config['target_root']}/{tmp_config['target_pt']}", label=1)
            # uncomment when use alphaG feature
            # tgt_test_bn = MyDataset(f"{tmp_config['target_root']}/{tmp_config['target_bn']}", label=0, single_feature_name='alphag_alt')
            # tgt_test_pt = MyDataset(f"{tmp_config['target_root']}/{tmp_config['target_pt']}", label=1, single_feature_name='alphag_alt')
        
        # merge data
        test_data = ConcatDataset([tgt_test_bn, tgt_test_pt])
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

        print(f"[INFO] Backbone: {backbone_name} | Model initialization...")
        
        # when pre-trained the ResNet architecture
        # my_model = model_fn(backbone_name=backbone_name, seq_len=tmp_config['seq_len'], embed_dim=tmp_config['embed_dim']).to(device)
        
        # when pre-trained the Transformer architecture
        my_model = DyPositionTransformer(
            input_dim=tmp_config['embed_dim'],
            num_layers=4,
            d_model=256,
            num_heads=4,
            dff=512,
            dropout_rate=0.1,
            head_hidden=128,
        ).to(device) 

        # initialize task directory
        task_dir = f"./pretrained_weight/{task_name}/"
        print(f"[INFO] Job inited at {task_dir}...")
        os.makedirs(task_dir, exist_ok=True)
        weight_path = f"{task_dir}/source_{backbone_name}_{tmp_config['feature_name']}_model.pt"
        metric_path = f"{task_dir}/source_{backbone_name}_{tmp_config['feature_name']}_metric.txt"
        pred_path = f"{task_dir}/source_{backbone_name}_{tmp_config['feature_name']}_pred.txt"

        # training
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
            strategy_mode='notransfer',
            feature_name=tmp_config['feature_name']
        )
        
        # zero-shot evaluation
        test_model_detail(
            model=model,
            dataloader=test_loader, 
            savepath=metric_path, 
            save_preds_path=pred_path, 
            device=device, 
            feature_name=tmp_config['feature_name']
        )
        
        print(
            f"[SUCCESS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Training completed!"
        )