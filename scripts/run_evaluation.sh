#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 INPUT GRCH38_FASTA PIPELINE_YAML MODEL_WEIGHT MODEL_CONFIG OUTPUT_DIR" >&2
  exit 2
fi

input=$1
reference=$2
pipeline=$3
weight=$4
model_config=$5
output_dir=$6
mkdir -p "$output_dir"

python inference/validate_input.py --input "$input" \
  --reference-fasta "$reference" --output "$output_dir/input.validated.tsv"
python feature_extraction/run_feature_extraction.py --config "$pipeline" \
  --input "$output_dir/input.validated.tsv" --output-dir "$output_dir/features"
python inference/predict.py --features "$output_dir/features/features.pth" \
  --model-weight "$weight" --model-config "$model_config" \
  --output "$output_dir/predictions.tsv"
