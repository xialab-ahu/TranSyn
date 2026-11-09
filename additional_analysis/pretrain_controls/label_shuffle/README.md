# Label-shuffle pretraining control

This control permutes labels in the 80% source-domain training subset while
preserving class counts, trains without early stopping, and evaluates the target
domain with its original labels.

Run from the TranSyn repository root. The script locates `functions/`,
`config/config.yaml`, and `config/config_pretrain.yaml` relative to itself, so no
server path is required. Data locations remain user-controlled configuration
values and must not be committed when they contain licensed records.

```bash
bash additional_analysis/pretrain_controls/label_shuffle/run_shuffle.sh missense-CaLM 0
bash additional_analysis/pretrain_controls/label_shuffle/run_shuffle.sh noncoding-GPN-MSA 0

python -u additional_analysis/pretrain_controls/label_shuffle/Pretrain_shuffle.py \
  --task missense-CaLM --gpu 0 --epochs 16
```

Set `SHUFFLE_PYTHON` to select a Python interpreter. Dataset-split and
label-shuffle seeds default to 42; batch size is 32. The Python entry point
defaults to 13 epochs, while the wrapper requests 16 epochs.

Outputs include the fixed shuffle mapping, run configuration, losses, target
metrics, predictions, final `state_dict`, and a `SUCCESS` marker. Existing
output directories containing `run_config.json` or `shuffle_mapping.npz` are
not overwritten. Weights, logs, and datasets are excluded from release files.
