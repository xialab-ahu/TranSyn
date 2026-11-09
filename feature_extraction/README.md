# Feature extraction

## 1 External software and resources

Install each upstream project in its own Conda environment and obtain its
official model/data files under the upstream license:

- `drp-psm`: SynMall, SilVA/EnDSM, GDI/GC and gene-centric annotations.
- `GPNMSA`: the GPN-MSA Python package, downloaded model directory and
  `89.zarr.zip` GenomeMSA resource.
- `calm`: the CaLM package and its required pretrained weights.
- GRCh38 transcript metadata containing `transcript_id`, `strand`, `refseq_id`
  and `coding_sequence`.

## 2 Configuration

Copy both templates and replace every `/path/to/...` entry:

```bash
cp feature_extraction/pipeline.example.yaml local.pipeline.yaml
cp feature_extraction/biological-resources.example.yaml local.biological-resources.yaml
```

`local.biological-resources.yaml` identifies the exact resources used by
`1-retrieve-biology.py`. `local.pipeline.yaml` defines the Conda environments,
GPN-MSA model/MSA files, transcript table, and released normalization objects.
Keep these local configuration files outside version control.

## 3 Complete workflow

```bash
python feature_extraction/run_feature_extraction.py \
  --config local.pipeline.yaml \
  --input example/input.grch38.tsv \
  --output-dir work/example/features
```

The command runs the actual repository scripts in this order:

1. `1-retrieve-biology.py` creates `input.cds.txt` and `biological.raw.txt`.
2. `1-Local-retrieve-GPN-MSA-Bulk.py` creates `GPN_MSA.pth` using 64 positions
   on each side of the variant.
3. `2-retrieve-cds-info.py` creates 21-codon reference and alternate contexts
   in `input.codon.txt`.
4. `1-Local-LLM-RNA-CaLM-Bulk.py` creates `CaLM.pth` as alternate minus
   reference CaLM representations.
5. The workflow applies `reformat_header.info`, removes the columns recorded in
   `empty_cols.pkl`, transforms missing values with the fitted
   `IterativeImputer` from `imputer.pkl`, and applies the fitted `scaler.pkl`
   before merging the three modalities into `features.pth`.


## 4 Direct stage execution

Each stage also has a standalone CLI. For example:

```bash
conda run -n GPNMSA python feature_extraction/1-Local-retrieve-GPN-MSA-Bulk.py \
  --input_file input.tsv --task_dir work/features \
  --model_path /path/to/GPN-MSA/model \
  --msa_path /path/to/GPN-MSA/89.zarr.zip
```

Use `python SCRIPT --help` for all required arguments. Inputs must use GRCh38
coordinates and preserve `Variant38` identifiers in `CHROM_POS_REF/ALT` form.
