"""
@description: This script performs model architecture ablation experiments
1. TranSyn (DNA + RNA + Bio)
2. DNA
3. RNA
4. Bio
5. DNA + RNA
6. DNA + Bio
7. RNA + Bio
"""


import os
import torch
from torch.utils.data import DataLoader, ConcatDataset

import yaml
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Debug] Using device: {device}")
print(f"[Debug] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"[Debug] torch.cuda.current_device() = {torch.cuda.current_device()}")
print(f"[Debug] torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")

from functions.Dataset import MyDataset, concat_and_loader
from functions.TranSyn_train import train_model_ablation
from functions.TranSyn_evaluate import test_model_ablation
from functions.Utils import set_full_deterministic_seed

from functions.TranSynAttention import TranSynAttentionAblation
from functions.ModelScaffold import model_fn, build_backbone
from functions.DyPositionTransformer import DyPositionTransformer


# Import configuration
model_combo = "GPN_CaLM_NewBio"
task_dir = f"./section_4/model_ablation/{model_combo}/"
os.makedirs(task_dir, exist_ok=True)
metric_log = f"{task_dir}/metrics.txt"
with open('./config/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
    base_config = config['base']
    spc_config = config[model_combo]

# Import training data
TARGET_ROOT_PATH = "./feature/"
train_data_bn = MyDataset(data_path=f"{TARGET_ROOT_PATH}/train_target_neg.pth", label=0)
train_data_pt = MyDataset(data_path=f"{TARGET_ROOT_PATH}/train_target_pos.pth", label=1)
train_data = ConcatDataset([train_data_bn, train_data_pt])
test_data_bn = MyDataset(data_path=f"{TARGET_ROOT_PATH}/test_target_neg.pth", label=0)
test_data_pt = MyDataset(data_path=f"{TARGET_ROOT_PATH}/test_target_pos.pth", label=1)
test_data = ConcatDataset([test_data_bn, test_data_pt])

# Pre-trained weight files
# Non-coding pre-trained weights
source_nc_pretrain_path = "./pretrained_weight/noncoding-GPN-MSA/source_ResNet1D_model.pt"
# Missense pre-trained weights
source_missense_pretrain_path = "./pretrained_weight/missense-CaLM/source_PositionTransformer_model.pt"
print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Loading data...")

tasks = ['dna', 'rna', 'bio', 'dna+rna', 'dna+bio', 'rna+bio']
SEEDS = list(range(42, 53))
results = []
# Import training data
print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Loading data...") 
for SEED in SEEDS:
    set_full_deterministic_seed(SEED)
    for task_name in tasks:
        print(f"{SEED}|{task_name}")
        test_data = ConcatDataset([test_data_bn, test_data_pt])
        train_loader, val_loader = concat_and_loader(train_data, 
                                                     split_rate=0.8, 
                                                     batch_size=spc_config['BATCH_SIZE'], 
                                                     seed=SEED)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

        # Initialize backbone network
        ## DNA module
        tsa_dna_module = build_backbone(name="ResNet1D", seq_len=129, embed_dim=768)
        source_dna_module = model_fn(backbone_name="ResNet1D", seq_len=129, embed_dim=768).to(device)
        source_dna_module.load_state_dict(torch.load(source_nc_pretrain_path, map_location='cpu'))
        missing_keys, unexpected_keys = tsa_dna_module.load_state_dict(source_dna_module.backbone.state_dict(), strict=False)
        print("DNA backbone load_state_dict:")
        print("  Missing keys:", missing_keys)
        print("  Unexpected keys:", unexpected_keys)
    
        ## RNA model
        tsa_rna_module = DyPositionTransformer(
            input_dim=768,
            num_layers=4,
            d_model=256,
            num_heads=4,
            dff=512,
            dropout_rate=0.1,
            head_hidden=128,
        ).to(device)
        state_dict = torch.load(source_missense_pretrain_path, map_location='cpu')
        missing_keys, unexpected_keys = tsa_rna_module.load_state_dict(state_dict, strict=False)
        print("RNA backbone load_state_dict:")
        print("  Missing keys:", missing_keys)
        print("  Unexpected keys:", unexpected_keys)

        ## Biological feature MLP
        tsa_bio_module = build_backbone(name="MLP", embed_dim=49, output_dim=spc_config['BIO_HIDDEN']).to(device) 
        
        ## Merge network
        my_model = TranSynAttentionAblation(
            dna_module=tsa_dna_module,
            rna_module=tsa_rna_module,
            bio_module=tsa_bio_module,
            hidden_layer=spc_config['HEAD_HIDDEN'],
            dropout=spc_config['DROPOUT_RATE'],
            mode=task_name
        ).to(device) 
        
        # Training
        model, best_val_loss, best_epoch = train_model_ablation(
            model=my_model,
            train_loader=train_loader, 
            val_loader=val_loader, 
            epochs=base_config['epochs'],  
            lr=spc_config['LEARNING_RATE'], 
            patience=base_config['patience'],
            save_path=f"./model_ablation/{task_name}.model", 
            loss=spc_config['LOSS_FN'],
            trsr_trade_off=spc_config['TRADE_OFF_TRSR'],
            mode=task_name
        )
        
        # Evaluation
        print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Finished! Evaluating...")
        auc, aupr, f2, mcc = test_model_ablation(
            model=model, 
            dataloader=test_loader, 
            device=device,
            mode=task_name
        )
        print(f"[SUCCESS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Training completed") 
        
        # Record current results
        results.append({
            "SEED": SEED,
            "submodule": task_name,
            "AUC": auc,
            "AUPR": aupr,
        })

df = pd.DataFrame(results)
df.to_csv(metric_log, sep="\t", index=False)
print(f"[INFO] Results saved to {metric_log}")