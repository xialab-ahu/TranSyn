"""
@description: Select models using five-fold chromosome-based cross-validation without an independent test set, and evaluate stability across five random seeds for each split.
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from functions.Dataset import MyDataset, LMDBDataset
from functions.DyPositionTransformer import DyPositionTransformer
from functions.General_evaluate import test_model_detail
from functions.General_train import train_model
from functions.ModelScaffold import model_fn
from functions.Utils import set_full_deterministic_seed


SEEDS = [42, 43, 44, 45, 46]
SPLIT_SCRIPT = ROOT.parent / 'TranSyn_Revise/split_chromosome_folds.py' # Chromosome-based splitting script


def load_training_data(settings):
    """
    Desc:
        Load the training data.
    Args:
        settings, YAML configuration.
    """
    datasets = []
    metadata = []
    for suffix, label in [('bn', 0), ('pt', 1)]:
        path = Path(settings['data_root']) / settings[f'train_{suffix}']
        if settings['data_type'] == 'lmdb':
            dataset = LMDBDataset(
                lmdb_path=str(path), feature_keys=[settings['feature_name']],
                preload=False, label=label,
            )
        else:
            dataset = MyDataset(
                str(path), label=label, feature_keys=[settings['feature_name']],
                single_feature_name=settings['feature_name'],
            )
        datasets.append(dataset)
        metadata.append(pd.DataFrame({
            'vid': [key.decode() if isinstance(key, bytes) else str(key) for key in dataset.keys],
            'label': label,
        }))
    samples = pd.concat(metadata, ignore_index=True)
    samples.insert(0, 'sample_index', np.arange(len(samples)))
    if samples['vid'].duplicated().any():
        raise ValueError('Duplicate training variant IDs; check input before splitting')
    # Return the dataset and sample metadata.
    return ConcatDataset(datasets), samples


def build_model(backbone_name, settings):
    """
    Desc:
        Build the model.
    Args:
        backbone_name, Backbone network: ResNet1D or PositionTransformer.
        settings, YAML configuration providing values such as embed_dim and
        seq_len.
    """
    if backbone_name == 'PositionTransformer':
        return DyPositionTransformer(
            input_dim=settings['embed_dim'], num_layers=4, d_model=256,
            num_heads=4, dff=512, dropout_rate=0.1, head_hidden=128,
        )
    return model_fn(
        backbone_name=backbone_name,
        seq_len=settings['seq_len'], embed_dim=settings['embed_dim'],
    )


def find_pretrained_weight(arguments, settings):
    """
    Desc:
        Find the pretrained weights.
    Args:
        arguments, Parsed command-line arguments.
        settings, YAML configuration.
    """
    if arguments.pretrained_weight is not None:
        candidates = [arguments.pretrained_weight]
    else:
        directory = ROOT / 'pretrained' / arguments.pretrained_task
        candidates = [
            directory / f'source_{arguments.backbone_name}_model.pt',
            directory / f'source_{arguments.backbone_name}_{settings["feature_name"]}_model.pt',
        ]
    for candidate in candidates: # Prioritize the first candidate, including an explicitly supplied path.
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'No pretrained checkpoint found: {candidates}')


def chromosome_splits(samples, protein=False):
    """
    Desc:
        Call the existing chromosome-based splitting script.
    Args:
        samples, Variant identifiers and sample metadata.
        protein, Whether this is a protein experiment with a sample lacking
        sequence context.
    """
    specification = importlib.util.spec_from_file_location('transyn_chromosome_split', SPLIT_SCRIPT)
    if specification is None or specification.loader is None:
        raise ImportError(f'Cannot import chromosome splitter: {SPLIT_SCRIPT}')
    splitter = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(splitter)
    reference = splitter.assign_folds(splitter.read_table(splitter.DEFAULT_INPUT))
    # Assign folds using the complete training list before excluding the protein
    # sample that lacks sequence context.
    if protein:
        reference = reference.loc[reference['variant_hg38'].ne(splitter.EXCLUDED_PROTEIN_VID)]
        samples = samples.loc[samples['vid'].ne(splitter.EXCLUDED_PROTEIN_VID)].copy()
    if set(samples['vid']) != set(reference['variant_hg38']):
        raise ValueError('Feature variant IDs differ from the reference chromosome split')
    reference = reference[['variant_hg38', '#CHROM', 'label', 'fold']].rename(
        columns={'variant_hg38': 'vid', 'label': 'reference_label'},
    )
    samples = samples.merge(reference, on='vid', how='left', sort=False, validate='one_to_one')
    if not samples['label'].eq(samples['reference_label']).all():
        raise ValueError('Feature labels differ from the reference training list')
    samples = samples.drop(columns='reference_label')
    splits = []
    for fold in range(1, splitter.FOLD_COUNT + 1):
        training = samples.loc[samples['fold'].ne(fold)]
        validation = samples.loc[samples['fold'].eq(fold)]
        if training['#CHROM'].isin(validation['#CHROM']).any():
            raise ValueError(f'Chromosome overlap in fold {fold}')
        splits.append((training['sample_index'].to_numpy(), validation['sample_index'].to_numpy()))
    return samples, splits, str(splitter.DEFAULT_INPUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_name', required=True)
    parser.add_argument('--feature_name', required=True)
    parser.add_argument('--pretrained_task', required=True)
    parser.add_argument('--backbone_name', required=True)
    parser.add_argument('--transfer', choices=['finetune', 'scratch'], default='finetune')
    parser.add_argument('--pretrained-weight', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--check-only', action='store_true', help='Validate data, folds, and weights without training or writing experiment results')
    arguments = parser.parse_args()
    if arguments.workers < 0:
        parser.error('workers must be nonnegative')
    torch.set_num_threads(4)
    set_full_deterministic_seed(SEEDS[0])

    # Load YAML configurations.
    baseline = yaml.safe_load((ROOT / 'config/config.yaml').read_text())['baseline']
    settings = yaml.safe_load((ROOT / 'config/config_finetune.yaml').read_text())[arguments.feature_name]
    
    # Load the training data.
    train_data, samples = load_training_data(settings)
    # Enable the protein flag for protein-related experiments.
    protein = arguments.feature_name.startswith(('CaLM', 'ISM', 'onehot-protein'))
    # Split the dataset.
    samples, splits, training_list = chromosome_splits(samples, protein=protein)
    # Locate the pretrained checkpoint.
    pretrained_weight = find_pretrained_weight(arguments, settings)
    # Build the model.
    probe_model = build_model(arguments.backbone_name, settings)
    # Load the weights.
    probe_model.load_state_dict(torch.load(pretrained_weight, map_location='cpu', weights_only=True), strict=True)
    del probe_model
    print(f'seeds={SEEDS}; train_samples={len(samples)}; 5 fixed chromosome folds; weight={pretrained_weight}', flush=True)
    if arguments.check_only:
        print(samples.groupby(['fold', 'label']).size().unstack(fill_value=0).to_string())
        print('CHECK_PASS: no training or independent-test data access.', flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; set CUDA_VISIBLE_DEVICES=0 and run on GPU host')
    device = torch.device('cuda')
    output = arguments.output or HERE / 'finetune_cv' / arguments.pretrained_task
    if (output / 'run_config.json').exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    output.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output / 'fold_assignments.tsv', sep='\t', index=False)
    metadata = {
        **vars(arguments), 'seeds': SEEDS, 'n_splits': 5, 'total_runs': 25,
        'settings': settings, 'baseline': baseline, 'pretrained_weight': str(pretrained_weight),
        'splitter': str(SPLIT_SCRIPT),
        'split_script_sha256': hashlib.sha256(SPLIT_SCRIPT.read_bytes()).hexdigest(),
        'training_list': training_list,
        'training_list_sha256': hashlib.sha256(Path(training_list).read_bytes()).hexdigest(),
        'folds_depend_on_seed': False, 'protein_exclusion_applied': protein,
        'evaluation': 'same held-out fold for early stopping and final validation metrics',
        'independent_test_data_loaded': False,
        'initialization': 'pretrained for both transfer modes, matching original Quick script',
        'weights_saved_permanently': False,
    }
    (output / 'run_config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
    records = []
    for seed in SEEDS:
        for fold, (train_indices, val_indices) in enumerate(splits, start=1):
            # Reset the seed, rebuild the model, and load the same source
            # checkpoint for every fold; never inherit fine-tuned weights from
            # the preceding fold.
            set_full_deterministic_seed(seed)
            model = build_model(arguments.backbone_name, settings).to(device)
            model.load_state_dict(torch.load(pretrained_weight, map_location='cpu', weights_only=True), strict=True)
            train_loader = DataLoader(
                Subset(train_data, train_indices.tolist()), batch_size=32, shuffle=True,
                num_workers=arguments.workers, drop_last=True,
            )
            val_loader = DataLoader(
                Subset(train_data, val_indices.tolist()), batch_size=64, shuffle=False,
                num_workers=arguments.workers, drop_last=False,
            )
            print(f'SEED {seed} FOLD {fold}/5 train={len(train_indices)} validation={len(val_indices)}', flush=True)
            # train_model requires an on-disk checkpoint to restore the best
            # state. Remove the temporary directory after use to conserve space.
            with tempfile.TemporaryDirectory(prefix=f'seed{seed}_fold{fold}_', dir=output) as temporary:
                trained_model, train_losses, val_losses = train_model(
                    model=model, train_loader=train_loader, val_loader=val_loader,
                    epochs=int(baseline['epochs']),
                    lr=float(baseline['lr'] if 'lr' in baseline else baseline['LEARNING_RATE']),
                    patience=int(baseline['patience']),
                    save_path=str(Path(temporary) / 'best_model.pt'),
                    device='cuda', strategy_mode=arguments.transfer,
                    feature_name=settings['feature_name'],
                )
                # Evaluate only the validation samples for this fold; do not
                # load test_pt or test_bn.
                auc, aupr = test_model_detail(
                    model=trained_model, dataloader=val_loader, device=device,
                    feature_name=settings['feature_name'],
                    savepath=str(output / f'seed_{seed}.fold_{fold}.metrics.tsv'),
                    save_preds_path=str(output / f'seed_{seed}.fold_{fold}.predictions.tsv'),
                )
            pd.DataFrame({
                'epoch': range(1, len(train_losses) + 1),
                'train_loss': train_losses, 'val_loss': val_losses,
            }).to_csv(output / f'seed_{seed}.fold_{fold}.losses.tsv', sep='\t', index=False)
            records.append({
                'source': arguments.source_name, 'feature': arguments.feature_name,
                'pretrained_task': arguments.pretrained_task, 'backbone': arguments.backbone_name,
                'group': arguments.transfer, 'seed': seed, 'fold': fold,
                'train_n': len(train_indices), 'val_n': len(val_indices),
                'AUC': auc, 'AUPR': aupr, 'epochs_run': len(train_losses),
                'best_epoch': int(np.argmin(val_losses)) + 1,
            })
            pd.DataFrame(records).to_csv(output / 'metrics.tsv', sep='\t', index=False)
            del model, trained_model, train_loader, val_loader
            torch.cuda.empty_cache()
    metrics = pd.DataFrame(records)
    # First summarize the five folds for each seed, then summarize metrics over
    # all 25 runs. Every standard deviation is the sample standard deviation.
    seed_summary = metrics.groupby('seed')[['AUC', 'AUPR']].agg(['mean', 'std'])
    seed_summary.columns = [f'{metric}_{statistic}' for metric, statistic in seed_summary.columns]
    seed_summary.reset_index().to_csv(output / 'summary_by_seed.tsv', sep='\t', index=False)
    summary = metrics[['AUC', 'AUPR']].agg(['mean', 'std']).T.rename_axis('metric').reset_index()
    summary['n_folds_per_seed'] = 5
    summary['n_seeds'] = len(SEEDS)
    summary['n_runs'] = len(metrics)
    summary.to_csv(output / 'summary.tsv', sep='\t', index=False)
    if len(metrics) != 25:
        raise RuntimeError(f'Expected 25 completed runs, found {len(metrics)}')
    (output / 'SUCCESS.json').write_text(json.dumps({'seeds': SEEDS, 'folds_per_seed': 5, 'runs_completed': 25, 'test_data_loaded': False}, indent=2))
    print(summary.to_string(index=False), flush=True)
    print(f'SAVED {output}', flush=True)


if __name__ == '__main__':
    main()
