# Mutation-spectrum analysis

This directory contains the manuscript mutation-spectrum implementation without
input datasets, reference genomes, results, or historical archives.

- `1_SNV_six_signature.py`: six substitution classes across five CDS bins.
- `2_tri_96_signature.py`: pyrimidine-oriented trinucleotide 96-channel counts.
- `plot_six_signature.py`: plotting from an existing six-class table.
- `plot_tri_96.py`: plotting from an existing 96-channel table.
- `signature_utils.py`: validation, strand normalization, logging, and figures.

Set user-controlled data locations before running:

```bash
export TRANSYN_DATA_ROOT=/path/to/controlled/data
export TRANSYN_PROJECT_DATA=/path/to/derived/project/data
```

The 96-channel workflow additionally requires `pyfaidx`, GRCh38 FASTA, and an
existing `.fai` index. The scripts never create or modify the reference index.

```bash
python 1_SNV_six_signature.py
python 2_tri_96_signature.py

# Plot previously generated statistics without loading source variants.
python 1_SNV_six_signature.py --plot-only --statistics /path/to/statistics.tsv
python 2_tri_96_signature.py --plot-only --statistics /path/to/statistics.tsv
```

Outputs are written under `results/` unless an output directory is supplied.
