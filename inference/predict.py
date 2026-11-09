#!/usr/bin/env python3
"""Run TranSyn inference from a validated, normalized feature bundle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    """Parse model, feature, output, and runtime options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model-weight", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    """Build the published architecture, load its weights, and score variants."""
    args = parse_args()
    for path in (args.features, args.model_weight, args.model_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    feature_data = torch.load(args.features, map_location="cpu")
    if not feature_data:
        raise ValueError("The feature bundle contains no variants")
    sys.path.insert(0, str(args.project_root.resolve()))
    from functions.Dataset import MyDataset
    from functions.DyPositionTransformer import DyPositionTransformer
    from functions.ModelScaffold import build_backbone
    from functions.TranSynAttention import TranSynAttention
    from functions.TranSyn_evaluate import direct_inference

    configuration = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    configuration = configuration.get("GPN_CaLM_NewBio", configuration)
    device = torch.device(args.device)
    dna = build_backbone(name="ResNet1D", seq_len=129, embed_dim=768)
    rna = DyPositionTransformer(input_dim=768, num_layers=4, d_model=256,
                                num_heads=4, dff=512, dropout_rate=0.1,
                                head_hidden=128).to(device)
    biological = build_backbone(name="MLP", embed_dim=49,
                                output_dim=configuration["BIO_HIDDEN"]).to(device)
    model = TranSynAttention(dna_module=dna, rna_module=rna, bio_module=biological,
                            hidden_layer=configuration["HEAD_HIDDEN"],
                            dropout=configuration["DROPOUT_RATE"]).to(device)
    model.load_state_dict(torch.load(args.model_weight, map_location=device))
    dataset = MyDataset(str(args.features), label=0.0,
                        feature_keys=["gpn_msa", "calm_diff", "biological_feature"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction = direct_inference(model=model, dataloader=loader, label=False,
                                  save_dir=str(args.output.parent), device=device)
    if not isinstance(prediction, pd.DataFrame):
        prediction = pd.DataFrame(prediction)
    required = {"vid", "pred_prob"}
    if not required.issubset(prediction.columns):
        raise RuntimeError(f"Inference output is missing {sorted(required - set(prediction.columns))}")
    prediction[["vid", "pred_prob"]].rename(
        columns={"vid": "Variant38", "pred_prob": "TranSyn_score"}).to_csv(
            args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
