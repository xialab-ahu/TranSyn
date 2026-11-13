# TranSyn Independent Calibration and Reference-Threshold Analysis

This directory is independent of `evaluation_stats` and `evaluation_plots`. It
reads only the TranSyn scores from the two original benchmark datasets and does
not modify the original model, scores, or existing comparison results.

## Usage

```bash
cd additional_analysis/calibration_analysis
conda run --no-capture-output -n tllib python compute.py
conda run --no-capture-output -n tllib python plot.py
```

By default, `compute.py` performs 5,000 gene-level bootstrap replicates using a
fixed random seed of `20261028`, five equal-width bins, a reference threshold of
`0.5` (`score > threshold`), and assumed PPV proportions of 0.1%, 1%, 5%, and
10%. These settings can be changed with `--bootstrap`, `--seed`, `--bins`,
`--threshold`, `--prevalences`, and `--output`.

`plot.py` accepts `--input` and redraws only the PNG figures from previously
computed result files.
