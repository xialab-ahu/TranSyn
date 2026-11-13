"""
@description: A common exclude gene source-domain trained script
@author: chen ye
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from functions.Dataset import LMDBDataset, concat_and_loader
from functions.DyPositionTransformer import DyPositionTransformer
from functions.General_evaluate import test_model_detail
from functions.General_train import train_model
from functions.ModelScaffold import model_fn
from functions.Utils import set_full_deterministic_seed


class TorchFeatures(Dataset):
    def __init__(self, path, label, feature_name):
        self.data = torch.load(path, map_location='cpu', mmap=True)
        self.keys = list(self.data)
        self.label = label
        self.feature_name = feature_name
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        key = self.keys[index]
        feature = torch.as_tensor(self.data[key][self.feature_name]).float()
        return {
                    'vid': key,
                    self.feature_name: feature,
                    'label': torch.tensor(self.label, dtype=torch.float32)
                }


def load_dataset(settings, domain, suffix, label):
    """
    Desc:
        按显式配置加载数据
    Args:
        settings, yaml配置文件
        domain, source/target domain
        suffix, 后缀
        label, 标签
    """
    path = Path(settings[f'{domain}_root']) / settings[f'{domain}_{suffix}']
    # 如果特征是lmdb格式
    if settings[f'{domain}_data_type'] == 'lmdb':
        return LMDBDataset(
                            str(path),
                            label=label,
                            feature_keys=[settings['feature_name']],
                            preload=False
                           )
    return TorchFeatures(path, label, settings['feature_name'])


def build_model(task_name, settings):
    """
    Desc:
        预指定的两个模型定义
    Args:
        task_name, 任务名
        settings, config的yaml配置文件
    """
    if task_name == 'missense-CaLM': # 错义突变利用Transformer架构
        backbone_names = ['PositionTransformer']
        model = DyPositionTransformer(
            input_dim=settings['embed_dim'], num_layers=4, d_model=256,
            num_heads=4, dff=512, dropout_rate=0.1, head_hidden=128,
        )
    else: # 非编码突变利用ResNet架构
        backbone_names = ['ResNet1D']
        model = model_fn(
            backbone_name=backbone_names[0], seq_len=settings['seq_len'],
            embed_dim=settings['embed_dim'],
        )
    return model, backbone_names[0]


def resolve_paths(task_name, hit_filename, arguments, output_name=None):
    """解析排除表和输出目录；替换hit表时自动分离默认结果目录。"""
    hit_path = (arguments.hit_file or Path(hit_filename)).expanduser()
    if not hit_path.is_absolute():
        hit_path = HERE / hit_path
    hit_path = hit_path.resolve()
    if arguments.hit_file is not None:
        default_name = f'{task_name}-exclude-{hit_path.stem}'
    else:
        default_name = output_name or f'{task_name}-exclude-test-gene'
    output = arguments.output or ROOT / 'pretrained' / default_name
    # 保留原有约定：仅默认输出自动追加smoke_test；显式--output按原样使用。
    if arguments.smoke_test and arguments.output is None:
        output = output / 'smoke_test'
    return hit_path, output.expanduser().resolve()


def read_hit_ids(hit_path):
    """读取带variant38表头的TSV，拒绝缺失、空ID和空表，重复ID仅计一次。"""
    hits = pd.read_csv(hit_path, sep='\t', dtype=str)
    if 'variant38' not in hits.columns:
        raise ValueError(f'排除表必须包含variant38列: {hit_path}')
    if hits['variant38'].isna().any():
        raise ValueError(f'variant38中存在缺失值: {hit_path}')
    variants = hits['variant38'].str.strip()
    if variants.empty or variants.eq('').any():
        raise ValueError(f'variant38不能为空: {hit_path}')
    return pd.Index(variants.unique())


def run(task_name, hit_filename, output_name=None):
    """
    Desc:
        模型训练、评估
    Args:
        task_name, 任务名
        hit_filename, 和测试基因重叠的数据, 这部分数据应该去除
        output_name, 可选的默认输出目录名，便于复用不同排除策略
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test', action='store_true') # 是否进行冒烟测试
    parser.add_argument('--output', type=Path) # 输出路径
    parser.add_argument('--hit-file', type=Path, help='替换排除TSV；相对路径以脚本所在目录为基准')
    parser.add_argument('--check-config', action='store_true', help='仅核对排除表和输出路径，不加载特征、不训练、不创建目录')
    arguments = parser.parse_args()
    hit_path, output = resolve_paths(task_name, hit_filename, arguments, output_name)
    hit_ids = read_hit_ids(hit_path)
    # 在读取大体积特征前检查输出，避免覆盖完整或中断实验的任何已有文件。
    if output.exists() and (not output.is_dir() or any(path.is_file() for path in output.iterdir())):
        raise FileExistsError(f'Refusing to overwrite existing experiment: {output}')
    if arguments.check_config:
        print(json.dumps({'task': task_name, 'hit_file': str(hit_path),
                          'hit_unique_ids': len(hit_ids), 'output': str(output),
                          'smoke_test': arguments.smoke_test}, ensure_ascii=False, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; set CUDA_VISIBLE_DEVICES before running on GPU host')
    
    # 设定随机种子和线程数
    torch.set_num_threads(4)
    seed = 42
    set_full_deterministic_seed(seed)

    # 读取预训练的config文件
    settings = yaml.safe_load((ROOT / 'config/config_pretrain.yaml').read_text())[task_name]
    baseline = yaml.safe_load((ROOT / 'config/config.yaml').read_text())['baseline']

    # 输出路径
    if (output / 'run_config.json').exists() or (output / 'SUCCESS.json').exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment: {output}')
    output.mkdir(parents=True, exist_ok=True)
    print(f'START pid={os.getpid()} task={task_name} smoke={arguments.smoke_test} GPU={torch.cuda.get_device_name(0)}', flush=True)

    # 每个脚本读取对应的hit表
    # hit_path和hit_ids已由入口默认值或--hit-file解析并校验。

    # 根据突变类型选择对应的hit_df
    source_datasets = []
    audit_tables = []
    raw_offset = 0
    for suffix, label in [('bn', 0), ('pt', 1)]:
        dataset = load_dataset(settings, 'source', suffix, label) # 加载数据集
        variants = [key.decode() if isinstance(key, bytes) else str(key) for key in dataset.keys] # 获取对应的突变键
        audit = pd.DataFrame({
            'source_index_original': np.arange(len(dataset)) + raw_offset,
            'variant38': variants,
            'label': label,
            'source_partition': suffix,
        }) # 构建审计表，包括原始索引、突变ID、标签、前缀
        audit['excluded'] = audit['variant38'].isin(hit_ids) # 根据hit_ids获取排除的突变
        retained_indices = np.flatnonzero(~audit['excluded'].to_numpy()) # 获取保留数据的索引
        source_datasets.append(Subset(dataset, retained_indices.tolist())) # 分别append对应标签保留索引的数据
        audit_tables.append(audit) # 分别append审计数据
        raw_offset += len(dataset) # 
        print(f'{suffix}: original={len(audit)} excluded={int(audit.excluded.sum())} retained={len(retained_indices)}', flush=True)
    audit = pd.concat(audit_tables, ignore_index=True) # 审计表
    excluded = audit.loc[audit['excluded']].copy() # 排除表
    retained = audit.loc[~audit['excluded']].copy().reset_index(drop=True) # 保留表
    if retained['variant38'].isin(hit_ids).any() or retained.empty:
        raise ValueError('Filtering failed or no source samples remain')
    if excluded.empty:
        raise ValueError('No source IDs matched the hit table; verify variant38 format')
    
    source = ConcatDataset(source_datasets)
    if len(retained) != len(source):
        raise ValueError('Source dataset and filtering audit disagree')
    
    # 保留统计结果
    excluded.to_csv(output / 'excluded_source_samples.tsv', sep='\t', index=False)
    hit_report = pd.DataFrame({'variant38': hit_ids})
    hit_report['source_occurrences'] = hit_report['variant38'].map(audit['variant38'].value_counts()).fillna(0).astype(int)
    hit_report.to_csv(output / 'hit_match_report.tsv', sep='\t', index=False)
    filtering = audit.groupby('source_partition', sort=False).agg(original=('variant38', 'size'), excluded=('excluded', 'sum'))
    filtering['retained'] = filtering['original'] - filtering['excluded']
    filtering.to_csv(output / 'filtering_statistics.tsv', sep='\t')

    # 先过滤后划分：复用原始非分层80/20函数和seed
    original_train, original_val = concat_and_loader(source, split_rate=0.8, batch_size=32, seed=seed)
    retained['source_index_filtered'] = np.arange(len(retained))
    retained['split'] = 'validation'
    retained.loc[original_train.dataset.indices, 'split'] = 'train'
    retained.to_csv(output / 'source_split.tsv.gz', sep='\t', index=False, compression='gzip')
    if retained.groupby('source_index_filtered').size().ne(1).any():
        raise ValueError('Duplicate source sample indices')
    train_loader, val_loader = original_train, original_val
    if arguments.smoke_test:
        train_loader = DataLoader(Subset(original_train.dataset, range(32)), batch_size=32, shuffle=True, num_workers=1, drop_last=True)
        val_loader = DataLoader(Subset(original_val.dataset, range(64)), batch_size=32, shuffle=False, num_workers=1, drop_last=True)
    feature = settings['feature_name']
    expected_shape = (settings['seq_len'], settings['embed_dim'])
    if tuple(source[0][feature].shape) != expected_shape:
        raise ValueError(f'Feature shape differs from configuration: {source[0][feature].shape}')
    model, backbone = build_model(task_name, settings)
    model = model.cuda()
    epochs = 1 if arguments.smoke_test else int(baseline['epochs'])
    weight_path = output / f'source_{backbone}_{feature}_model.pt'
    metadata = {
        'task': task_name, 'seed': seed, 'settings': settings, 'baseline': baseline,
        'epochs_limit': epochs, 'patience': int(baseline['patience']), 'backbone': backbone,
        'smoke_test': arguments.smoke_test, 'hit_table': str(hit_path.resolve()),
        'hit_table_sha256': hashlib.sha256(hit_path.read_bytes()).hexdigest(),
        'hit_unique_ids': len(hit_ids), 'hit_ids_matched': int(hit_report.source_occurrences.gt(0).sum()),
        'source_original': len(audit), 'source_excluded': len(excluded),
        'source_retained': len(source), 'train_count': len(original_train.dataset),
        'validation_count': len(original_val.dataset), 'filter_before_split': True,
        'labels_shuffled': False, 'training_strategy': 'notransfer; no TSRS',
        'checkpoint': str(weight_path.resolve()), 'gpu': torch.cuda.get_device_name(0),
    }
    (output / 'run_config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    # 直接复用标准train_model，早停规则、优化器、损失函数和最佳权重恢复保持一致。
    start = time.monotonic()
    model, train_losses, val_losses = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader, epochs=epochs,
        lr=float(baseline['lr'] if 'lr' in baseline else baseline['LEARNING_RATE']),
        patience=int(baseline['patience']), save_path=str(weight_path),
        device='cuda', strategy_mode='notransfer', feature_name=feature,
        loss=baseline.get('LOSS_FN', 'bce').lower(),
        optim_name=baseline.get('OPTIMIZE', 'adam').lower(),
    )
    history = pd.DataFrame({'epoch': np.arange(1, len(train_losses) + 1), 'train_loss': train_losses, 'val_loss': val_losses})
    history.to_csv(output / 'training_history.tsv', sep='\t', index=False)
    if not np.isfinite(history[['train_loss', 'val_loss']].to_numpy()).all():
        raise RuntimeError('Nonfinite training/validation losses')
    restored, _ = build_model(task_name, settings)
    restored.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True), strict=True)
    for name, value in restored.state_dict().items():
        if not torch.equal(value, model.state_dict()[name].cpu()):
            raise RuntimeError(f'Checkpoint differs from restored best model: {name}')

    # 目标域保持原数据及真实标签，只在最佳源域权重确定后做zero-shot评估。
    targets = [load_dataset(settings, 'target', suffix, label) for suffix, label in [('bn', 0), ('pt', 1)]]
    target = ConcatDataset(targets)
    if arguments.smoke_test:
        target = Subset(target, list(range(16)) + list(range(len(targets[0]), len(targets[0]) + 16)))
    test_loader = DataLoader(target, batch_size=32, shuffle=False, num_workers=1)
    test_model_detail(
        model=model, dataloader=test_loader, device='cuda', feature_name=feature,
        savepath=str(output / f'source_{backbone}_{feature}_metric.txt'),
        save_preds_path=str(output / f'source_{backbone}_{feature}_pred.txt'),
    )
    result = {
        'epochs_run': len(history), 'best_epoch': int(np.argmin(val_losses)) + 1,
        'best_val_loss': float(min(val_losses)), 'checkpoint': str(weight_path.resolve()),
        'strict_reload_verified': True, 'smoke_test': arguments.smoke_test,
        'seconds': time.monotonic() - start, 'finished_unix': time.time(),
    }
    (output / 'SUCCESS.json').write_text(json.dumps(result, indent=2))
    print(f"{'SMOKE_PASS' if arguments.smoke_test else 'COMPLETE'} {json.dumps(result)}", flush=True)
