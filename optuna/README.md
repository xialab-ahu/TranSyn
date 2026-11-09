# Hyperparameter optimization

All paths, the random seed, fold count, and compute device are command-line
options. The default seed is 42 and is applied before every trial; K-fold
splitting and DataLoader shuffling are seeded independently.

```bash
python optuna/General_optuna.py \
  --negative-features /path/to/train_negative.pth \
  --positive-features /path/to/train_positive.pth \
  --dna-weight /path/to/noncoding_GPN_MSA.pt \
  --rna-weight /path/to/missense_CaLM.pt \
  --output-dir work/optuna --seed 42
```
The script writes per-fold checkpoints, `trials_summary.tsv`, and
`best_trial.txt`. External pretrained weights are not redistributed.
