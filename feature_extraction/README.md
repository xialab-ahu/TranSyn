# Feature extraction

## 1 External software and resources

Install each upstream project in its own Conda environment and obtain its
official model/data files under the upstream license:

- `bio_feature`: SynMall, SilVA/EnDSM, GDI/GC and gene-centric annotations.
- `GPNMSA`: the GPN-MSA Python package, downloaded model directory and
  `89.zarr.zip` GenomeMSA resource.
- `calm`: the CaLM package and its required pretrained weights.
- GRCh38 transcript metadata containing `transcript_id`, `strand`, `refseq_id`
  and `coding_sequence`.

### 1.1 Biological-feature environment

Create the reduced environment supplied with TranSyn:

```bash
conda env create -f config/bio_feature.yml
conda run -n bio_feature python -c "import numpy, pandas, sklearn, joblib, yaml"
```

The environment contains only the packages used by the included biological/CDS
scripts. SynMall, SilVA/EnDSM, liftOver, annotation tables, and fitted
preprocessing objects are external resources; set their local paths in a copy
of `biological-resources.example.yaml`.

### 1.2 GPN-MSA environment and assets

Follow the official GPN inference installation and download a compatible
GPN-MSA checkpoint plus GenomeMSA Zarr store. One supported installation route
is:

```bash
conda create -n GPNMSA python=3.11 -y
conda run -n GPNMSA pip install "gpn[inference]" lmdb
conda run -n GPNMSA python -c "import gpn, torch, transformers, lmdb"
```

Set `--model_path` and `--msa_path` in `local.pipeline.yaml`. The model and MSA
must use the same assembly, species order, and preprocessing.

### 1.3 CaLM environment and weights

Install the original codon language model from its official repository:

```bash
git clone https://github.com/oxpig/CaLM.git /path/to/CaLM
conda create -n calm python=3.10 -y
conda run -n calm pip install -e /path/to/CaLM
conda run -n calm pip install pandas tqdm lmdb
conda run -n calm python -c "from calm import CaLM; print('CaLM import: PASS')"
```

Follow the CaLM repository instructions for its pretrained files. TranSyn does
not redistribute GPN-MSA or CaLM weights.

### 1.4 Automatic prerequisite check

Normal execution checks all Conda environments, required Python modules,
configured scripts, GPN-MSA model/MSA paths, transcript metadata, biological
resources, and normalization objects before creating outputs. All missing
requirements are reported together with installation guidance. `--dry-run`
intentionally skips existence checks and only displays the resolved workflow.

## 2 Configuration

Copy both templates and replace every `/path/to/...` entry:

```bash
cp feature_extraction/pipeline.example.yaml feature_extraction/local.pipeline.yaml
cp feature_extraction/biological-resources.example.yaml \
  feature_extraction/local.biological-resources.yaml
```

`local.biological-resources.yaml` identifies the exact resources used by
`1-retrieve-biology.py`. `local.pipeline.yaml` defines the Conda environments,
GPN-MSA model/MSA files, transcript table, and released normalization objects.
Keep these local configuration files outside version control.

## 3 Complete workflow

First inspect the resolved workflow without launching Conda environments,
loading models, or downloading resources:

```bash
python feature_extraction/run_feature_extraction.py \
  --config feature_extraction/pipeline.example.yaml \
  --input example/input.grch38.tsv \
  --output-dir work/example/features \
  --dry-run
```

The dry run prints each command and expected output, followed by the header,
empty-column, `IterativeImputer`, scaler, merge, and skipped-record paths. It
does not require the placeholder `/path/to/...` locations to exist.

After replacing all placeholders in a local YAML file, execute the workflow:

```bash
python feature_extraction/run_feature_extraction.py \
  --config feature_extraction/local.pipeline.yaml \
  --input example/input.grch38.tsv \
  --output-dir work/example/features
```

The command runs the actual repository scripts in this order:

1. `4-Retrieve-biology.py` creates `input.cds.txt` and `biological.raw.txt`.
2. `2-Retrieve-GPN-MSA-Embeddings.py` creates `GPN_MSA.pth` using 64 positions
   on each side of the variant.
3. `3-Retrieve-cds-info.py` creates 21-codon reference and alternate contexts
   in `input.codon.txt`.
4. `1-Retrieve-CaLM-Embeddings.py` creates `CaLM.pth` as alternate minus
   reference CaLM representations.
5. The workflow applies `reformat_header.info`, removes the columns recorded in
   `empty_cols.pkl`, transforms missing values with the fitted
   `IterativeImputer` from `imputer.pkl`, and applies the fitted `scaler.pkl`
   before merging the three modalities into `features.pth`.


## 4 Direct stage execution

Each stage also has a standalone CLI. For example:

```bash
conda run -n GPNMSA python feature_extraction/2-Retrieve-GPN-MSA-Embeddings.py \
  --input_file input.tsv --task_dir work/features \
  --model_path /path/to/GPN-MSA/model \
  --msa_path /path/to/GPN-MSA/89.zarr.zip
```

Use `python SCRIPT --help` for all required arguments. Inputs must use GRCh38
coordinates and preserve `Variant38` identifiers in `CHROM_POS_REF/ALT` form.
