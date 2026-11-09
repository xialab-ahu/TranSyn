"""
@description: Hyperparameter optimization
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader, Subset

from functions.Dataset import MyDataset
from functions.DyPositionTransformer import DyPositionTransformer
from functions.ModelScaffold import build_backbone, model_fn
from functions.TranSynAttention import TranSynAttention
from functions.TranSyn_evaluate import test_model_detail
from functions.TranSyn_train import train_model
from functions.Utils import set_full_deterministic_seed


def build_model(trial: optuna.Trial, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """
    Desc:
        construct TranSyn and initialize its source backbones
    """
    bio_hidden = trial.suggest_categorical("bio_hidden", [64, 128, 256])
    dropout = trial.suggest_categorical("dropout_rate", [0.1, 0.3, 0.5])
    head_hidden = trial.suggest_categorical("head_hidden", [64, 128, 256, 512])
    dna = build_backbone(name="ResNet1D", seq_len=129, embed_dim=768)
    source_dna = model_fn(backbone_name="ResNet1D", seq_len=129, embed_dim=768)
    source_dna.load_state_dict(torch.load(args.dna_weight, map_location="cpu"))
    dna.load_state_dict(source_dna.backbone.state_dict(), strict=False)
    rna = DyPositionTransformer(input_dim=768, num_layers=4, d_model=256, num_heads=4,
                                dff=512, dropout_rate=0.1, head_hidden=128).to(device)
    rna.load_state_dict(torch.load(args.rna_weight, map_location="cpu"), strict=False)
    biological = build_backbone(name="MLP", embed_dim=49, output_dim=bio_hidden).to(device)
    return TranSynAttention(dna_module=dna, rna_module=rna, bio_module=biological,
                            hidden_layer=head_hidden, dropout=dropout).to(device)


def parse_args() -> argparse.Namespace:
    """Parse data, weight, output, and reproducibility settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-features", type=Path, required=True)
    parser.add_argument("--positive-features", type=Path, required=True)
    parser.add_argument("--dna-weight", type=Path, required=True)
    parser.add_argument("--rna-weight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    """Create and optimize a reproducible Optuna study."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = ConcatDataset([MyDataset(str(args.negative_features), label=0),
                             MyDataset(str(args.positive_features), label=1)])
    summary_path = args.output_dir / "trials_summary.tsv"

    def objective(trial: optuna.Trial) -> float:
        set_full_deterministic_seed(args.seed)
        # parameter search grid
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        trade_off = trial.suggest_categorical("trade_off_trsr", [0.1, 0.3, 0.5, 1.0])
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        loss = trial.suggest_categorical("loss_fn", ["bce", "focal"])
        aucs: list[float] = []
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for fold, (train_indices, validation_indices) in enumerate(splitter.split(dataset)):
            generator = torch.Generator().manual_seed(args.seed + fold)
            train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size,
                                      shuffle=True, num_workers=args.num_workers,
                                      generator=generator)
            validation_loader = DataLoader(Subset(dataset, validation_indices),
                                           batch_size=batch_size,
                                           num_workers=args.num_workers)
            model = build_model(trial, args, device)
            checkpoint = args.output_dir / f"trial_{trial.number:04d}_fold_{fold}.pt"
            train_model(model=model, train_loader=train_loader, val_loader=validation_loader,
                        epochs=args.epochs, lr=learning_rate, patience=args.patience,
                        save_path=str(checkpoint), loss=loss, optim_name=optimizer,
                        trsr_trade_off=trade_off)
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            auc, *_ = test_model_detail(model=model, dataloader=validation_loader)
            aucs.append(float(auc))
            trial.report(float(np.mean(aucs)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        mean_auc = float(np.mean(aucs))
        write_header = not summary_path.exists()
        with summary_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["trial", "mean_auc", "parameters"],
                                    delimiter="\t")
            if write_header:
                writer.writeheader()
            writer.writerow({"trial": trial.number, "mean_auc": mean_auc,
                             "parameters": trial.params})
        return mean_auc

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.trials)
    (args.output_dir / "best_trial.txt").write_text(
        f"value\t{study.best_value}\nparameters\t{study.best_params}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
