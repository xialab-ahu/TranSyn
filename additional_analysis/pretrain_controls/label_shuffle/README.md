# Label-shuffle pretraining control

This control permutes labels in the 80% source-domain training subset while
preserving class counts, trains without early stopping, and evaluates the target
domain with its original labels.

Run from the TranSyn repository root. The script locates `functions/`,
`config/config.yaml`, and `config/config_pretrain.yaml` relative to itself, so no
server path is required.

```bash
bash additional_analysis/pretrain_controls/label_shuffle/run_shuffle.sh missense-CaLM 0
bash additional_analysis/pretrain_controls/label_shuffle/run_shuffle.sh noncoding-GPN-MSA 0

python -u additional_analysis/pretrain_controls/label_shuffle/Pretrain_shuffle.py \
  --task missense-CaLM --gpu 0 --epochs 16
```