# Context-overlap analysis

This directory contains code, tests, and usage instructions only. It does not
redistribute source variants, derived statistics, figures, or logs.

- `calculate_context_overlap.py` calculates directional binary overlap between
  129-bp contexts and writes per-variant audit tables.
- `calculate_overlap_extent.py` calculates the maximum overlap length for every
  denominator-side variant, including zero-overlap variants.
- `plot_context_overlap.py` plots an existing summary table.
- `test_context_overlap.py` and `test_overlap_extent.py` test boundaries,
  duplicate sites, directional denominators, and combined-label behavior.

Each context is `[POS - 64, POS + 64]`. Variants are deduplicated by chromosome,
position, REF, and ALT. Combined groups are formed by union and recalculated;
group percentages are never averaged.

Provide all six user-owned input files explicitly:

```bash
python calculate_context_overlap.py \
  --missense-benign /path/to/missense.benign.tsv \
  --missense-pathogenic /path/to/missense.pathogenic.tsv \
  --synonymous-benign /path/to/synonymous.benign.tsv \
  --synonymous-pathogenic /path/to/synonymous.pathogenic.tsv \
  --noncoding-benign /path/to/noncoding.benign.tsv \
  --noncoding-pathogenic /path/to/noncoding.pathogenic.tsv \
  --output-dir statistics
```

Run the independent tests with:

```bash
python -m unittest test_context_overlap.py test_overlap_extent.py
```
