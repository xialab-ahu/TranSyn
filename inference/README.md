# Inference

Generate normalized features first, then run prediction:

```bash
python inference/predict.py \
  --features work/example/features/features.pth \
  --model-weight model_weight/model.pt \
  --model-config config/config.yaml \
  --output work/example/predictions.tsv
```

`TranSyn_score` is the model probability for the deleterious class. It is a
ranking/prioritization score and is not, by itself, a clinical classification.
Higher values indicate stronger model evidence for functional impact.

The input bundle must already contain GPN-MSA, CaLM-difference, and biological
features under the keys documented in `feature_extraction/README.md`. Biological
features must be transformed with the released training-time scaler; fitting a
new scaler on inference samples changes the model input distribution and is not
supported.
