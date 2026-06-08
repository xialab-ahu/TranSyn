"""
@description: This script explores which layers are general and which layers are specific
@author: chen ye
"""

# torch related
import torch
from torch.utils.data import DataLoader, ConcatDataset

# common packages
import os
import yaml
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Debug] Using device: {device}")
print(f"[Debug] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"[Debug] torch.cuda.current_device() = {torch.cuda.current_device()}")
print(f"[Debug] torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")

# load custom packages
from functions.Dataset import MyDataset, concat_and_loader
from functions.General_train import train_model
from functions.General_evaluate import test_model_detail
from functions.ModelScaffold import model_fn
from functions.Utils import load_partial_resnet, set_full_deterministic_seed

# load configs
model_combo = "GPN_CaLM_NewBio"
with open('../config/config.yaml', 'r', encoding='utf-8') as file:
	config = yaml.safe_load(file)
	base_config = config['base']
	spc_config = config[model_combo]
	 
feature_name = 'gpn_msa'
backbone_name = 'ResNet1D'

# load target data
print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Job starts!")
TARGET_ROOT_PATH = "./feature/"
train_data_bn = MyDataset(data_path=f"{TARGET_ROOT_PATH}/train_target_neg.pth", label=0)
train_data_pt = MyDataset(data_path=f"{TARGET_ROOT_PATH}/train_target_pos.pth", label=1)
train_data = ConcatDataset([train_data_bn, train_data_pt])
test_data_bn = MyDataset(data_path=f"{TARGET_ROOT_PATH}/test_target_neg.pth", label=0)
test_data_pt = MyDataset(data_path=f"{TARGET_ROOT_PATH}/test_target_pos.pth", label=1)
test_data = ConcatDataset([test_data_bn, test_data_pt])
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

# repeat random seeds
SEEDS = list(range(42, 53))
layers = 4
pretrain_task = ['source', 'target']
result_list = []

# begin test
for SEED in SEEDS:
	for layer in range(1, layers+1):
		for pt_task in pretrain_task:
			print(f"\n[INFO] ======= Running with SEED={SEED} =======")
			set_full_deterministic_seed(SEED)
			train_loader, val_loader = concat_and_loader(train_data, split_rate=0.8, batch_size=spc_config['BATCH_SIZE'], seed=SEED)
			_, seq_len, embed_dim = next(iter(val_loader))[feature_name].shape

			# load partial weight
			my_model = model_fn(backbone_name, seq_len, embed_dim).to(device)
			pretrained_weight_path = f'./pretrained_weight/layer-wise-experiment/{pt_task}_ResNet1D.pt'
			my_model = load_partial_resnet(my_model, pretrained_weight_path, n=layer)

			# model save path
			weight_path = f"./result/section_3/layer_wise/model.pt"

			# train
			print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Start training...")
			if pt_task == 'source':
				strategy_mode='finetune'
			else:
				strategy_mode='baseline'
			trained_model, _, _ = train_model(
											model=my_model,
											train_loader=train_loader, 
											val_loader=val_loader, 
											epochs=base_config['epochs'], 
											lr=spc_config['LEARNING_RATE'], 
											patience=base_config['patience'],
											save_path=weight_path, 
											loss=spc_config['LOSS_FN'],
											optim_name=spc_config['OPTIMIZE'],
											tsrs_trade_off=spc_config['TRADE_OFF_TRSR'],
											mute=True,
											strategy_mode=strategy_mode,
											feature_name=feature_name
											)
			# test
			auc, aupr = test_model_detail(
											model=trained_model, 
											dataloader=test_loader, 
											device=device, 
											feature_name=feature_name
										)
			result_list.append({
				"SEED": SEED,
				"TRANSFER": pt_task,
				"AUC": auc,
				"AUPR": aupr,
				"Layer": layer
			})
			
# save the result
print(f"[SUCCESS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Job completed!")
metrics_df = pd.DataFrame(result_list)
metrics_df.to_csv(f"./result/section_3/layer_wise/metrics.txt", sep='\t', index=False)