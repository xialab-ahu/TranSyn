"""
@description: run the label-shuffle pretraining control
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import yaml

from functions.Dataset import MyDataset, LMDBDataset, concat_and_loader
from functions.DyPositionTransformer import DyPositionTransformer
from functions.General_evaluate import test_model_detail
from functions.Utils import set_full_deterministic_seed


class ShuffledLabels(Dataset):
    def __init__(self, subset, negative_count, seed):
        """
        Desc:
            Execute the   init   operation.
        Args:
            subset, Value supplied for subset.
            negative_count, Value supplied for negative_count.
            seed, Value supplied for seed.
        """
        self.subset = subset
        self.original = np.asarray([int(i >= negative_count) for i in subset.indices])
        self.permutation = np.random.RandomState(seed).permutation(len(subset))
        self.labels = self.original[self.permutation]
        assert np.array_equal(np.sort(self.original), np.sort(self.labels))

    def __len__(self):
        """
        Desc:
            Execute the   len   operation.
        """
        return len(self.subset)

    def __getitem__(self, index):
        """
        Desc:
            Execute the   getitem   operation.
        Args:
            index, Value supplied for index.
        """
        item = dict(self.subset[index])
        item["label"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


def load_pair(config, domain, is_substract=False):
    """
    Desc:
        Load the positive and negative datasets for one domain.
    Args:
        config, Workflow configuration mapping.
        domain, Source or target domain name.
        is_substract, Value supplied for is_substract.
    """
    cls = LMDBDataset if config.get(f"{domain}_data_type", "torch") == "lmdb" else MyDataset
    datasets = []
    for suffix, label in [("bn", 0), ("pt", 1)]:
        path = Path(config[f"{domain}_root"]) / config[f"{domain}_{suffix}"]
        if not path.exists():
            raise FileNotFoundError(path)
        datasets.append(
            cls(
                str(path),
                label=label,
                feature_keys=[config["feature_name"]],
                is_substract=is_substract,
            )
        )
    return ConcatDataset(datasets), len(datasets[0])


def main():
    """
    Desc:
        Run the command-line workflow.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=["missense-CaLM", "noncoding-GPN-MSA"], default="missense-CaLM"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=13)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="One real training batch, checkpoint reload, and two target batches.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run on a GPU host or explicitly use --device cpu.")
    torch.set_num_threads(4)
    set_full_deterministic_seed(args.seed)
    with (ROOT / "config/config_pretrain.yaml").open() as handle:
        config = yaml.safe_load(handle)[args.task]
    with (ROOT / "config/config.yaml").open() as handle:
        baseline = yaml.safe_load(handle)["baseline"]
    backbone = "PositionTransformer" if args.task == "missense-CaLM" else "ResNet1D"
    out = (
        ROOT
        / "pretrained/label_shuffle"
        / args.task
        / f"seed_{args.seed}_shuffle_{args.shuffle_seed}"
    )
    if args.smoke_test:
        out = (
            ROOT
            / "pretrained/label_shuffle/smoke_test"
            / args.task
            / f"seed_{args.seed}_shuffle_{args.shuffle_seed}_{args.device}"
        )
    if (out / "run_config.json").exists() or (out / "shuffle_mapping.npz").exists():
        raise FileExistsError(f"Experiment already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    source, negative_count = load_pair(config, "source", is_substract=True)
    original_loader, unused_loader = concat_and_loader(
        source, split_rate=0.8, batch_size=32, seed=args.seed
    )
    shuffled = ShuffledLabels(original_loader.dataset, negative_count, args.shuffle_seed)
    np.savez_compressed(
        out / "shuffle_mapping.npz",
        train_indices=shuffled.subset.indices,
        unused_indices=unused_loader.dataset.indices,
        original_labels=shuffled.original,
        shuffled_labels=shuffled.labels,
        permutation=shuffled.permutation,
    )
    assert np.any(shuffled.labels != shuffled.original), "Permutation did not change labels"
    target, target_negative_count = load_pair(config, "target")
    train_set = shuffled
    if args.smoke_test:
        train_set = Subset(shuffled, range(32))
        target = Subset(
            target,
            list(range(min(16, target_negative_count)))
            + list(range(target_negative_count, min(target_negative_count + 16, len(target)))),
        )
    train_loader = DataLoader(
        train_set, batch_size=32, shuffle=True, num_workers=args.workers, drop_last=True
    )
    test_loader = DataLoader(target, batch_size=32, shuffle=False, num_workers=args.workers)
    feature = config["feature_name"]
    shape = tuple(source[0][feature].shape)
    if len(shape) != 2:
        raise ValueError(f"Expected (sequence, embedding), got {shape}")
    if args.task == "missense-CaLM":
        if shape != (config["seq_len"], config["embed_dim"]):
            raise ValueError(f"CaLM shape differs from config: {shape}")
        model = DyPositionTransformer(
            input_dim=config["embed_dim"],
            num_layers=4,
            d_model=256,
            num_heads=4,
            dff=512,
            dropout_rate=0.1,
            head_hidden=128,
        )
    else:
        from functions.ModelScaffold import model_fn

        model = model_fn(backbone_name=backbone, seq_len=shape[0], embed_dim=shape[1])
    model = model.to(args.device)
    metadata = dict(
        vars(args),
        config=config,
        baseline=baseline,
        feature_shape=shape,
        source_count=len(source),
        train_count=len(shuffled),
        target_count=len(target),
        changed_labels=int(np.sum(shuffled.labels != shuffled.original)),
        backbone=backbone,
        early_stopping=False,
    )
    (out / "run_config.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=baseline["lr"])
    criterion = torch.nn.BCEWithLogitsLoss()
    weight = out / f"source_{backbone}_model.pt"
    history = []
    for epoch in range(1 if args.smoke_test else args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch[feature].to(args.device))
            loss = criterion(logits, batch["label"].unsqueeze(1).to(args.device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if len(losses) == 1:
                print(
                    f"Epoch {epoch + 1}: first batch completed, loss={loss.item():.6f}", flush=True
                )
        if not losses:
            raise RuntimeError("No training batches")
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
        torch.save(model.state_dict(), weight)
        (out / "training_history.json").write_text(json.dumps(history, indent=2))
        print(
            f'Epoch {epoch + 1}: loss={history[-1]["train_loss"]:.6f}; saved {weight}', flush=True
        )
    model.load_state_dict(torch.load(weight, map_location=args.device))
    test_model_detail(
        model,
        test_loader,
        savepath=str(out / f"source_{backbone}_metric.txt"),
        save_preds_path=str(out / f"source_{backbone}_pred.txt"),
        device=args.device,
        feature_name=feature,
    )
    (out / "SUCCESS").write_text(
        "smoke test passed\n" if args.smoke_test else "training and target evaluation completed\n"
    )
    print(f"SUCCESS: {out}", flush=True)


if __name__ == "__main__":
    main()
