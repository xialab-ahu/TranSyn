# TranSyn

> 💻 **Computational method:** TranSyn is a transfer-learning framework for prioritizing potentially deleterious synonymous single-nucleotide variants. This repository covers input validation, source-domain preprocessing, feature extraction, inference, evaluation, and the analyses reported in the manuscript.

![TranSyn workflow](./asset/Model-Framework.jpg)

> 🌐 **Online web server:** Precomputed genome-wide scores for actionable genes and online variant prediction are available through the [TranSyn web server](https://bioinfo.ahu.edu.cn/transyn/).

![TranSyn web server](./asset/WebServer-Header.jpg)

## Scope and supported inputs

- **Genome assembly:** GRCh38/hg38 only for inference; coordinates are one-based.
- **Variant type:** Biallelic synonymous SNVs with `A`, `C`, `G`, or `T` REF and ALT alleles.
- **Chromosomes:** 1–22, X, and Y.
- **Required columns:** `#CHROM`, `POS`, `REF`, and `ALT` in a tab-separated file.
- **Unsupported inputs:** Indels, multiallelic rows, symbolic or non-ACGT alleles, duplicate rows, other genome assemblies, and reference-allele mismatches.

> 🏥 **Usage note:** TranSyn is a computational prioritization model. Its score is not a clinical classification and should not be used as the sole basis for patient-care decisions.

## Installation and external models

> 🔧 **Prerequisites:** Inference additionally requires independently installed GPN-MSA and CaLM environments, together with the corresponding model weights. Confirm that both environments and all required weights are available before running the complete workflow.

- **GPN-MSA:** <https://github.com/songlab-cal/gpn>
- **CaLM:** <https://github.com/oxpig/CaLM>

Create and activate the TranSyn environment:

```bash
conda env create -f environment.yml
conda activate transyn
```

Recommended hardware:

- A Linux workstation with at least 16 GB RAM.
- An NVIDIA GPU with at least 12 GB VRAM.

## Complete prediction workflow

1. **Prepare the input:** Create a GRCh38/hg38 TSV file and validate its alleles against the same GRCh38 FASTA used by the feature extractors.
2. **Extract features:** Generate biological, GPN-MSA, and CaLM features in their configured environments.
3. **Impute and scale:** Impute missing biological features and apply the fitted scaler.
4. **Merge and predict:** Merge feature modalities by `Variant38`, the hg38 variant identifier formatted as `chromosome_position_reference allele_alternate allele`; run prediction; and inspect any skipped records.
5. **Interpret the output:** `TranSyn_score` estimates the probability of the deleterious class. Higher values indicate a greater likelihood of pathogenicity.

Run the complete evaluation workflow with one command:

```bash
bash scripts/run_evaluation.sh \
  example/input.grch38.tsv /path/to/GRCh38.fa local.pipeline.yaml \
  model_weight/model.pt config/config.yaml work/example
```

A successful run prints `PASS: 1 unique GRCh38 biallelic SNVs` and creates `work/example/predictions.tsv` using the schema shown in `example/expected_output.tsv`.

## Quality control and troubleshooting

| Check or symptom | Recommended action |
|---|---|
| REF does not match the FASTA | Confirm that the input uses GRCh38, consistent contig names, and one-based positions. |
| A variant is absent from the output | Inspect `features.pth.skipped.tsv` for a missing GPN-MSA, CaLM, or biological representation. |
| Biological feature columns differ | Use the released `header.selected.info`; column order is part of the trained model specification. |
| Scores differ between machines | Check model and scaler checksums, package versions, device precision, and external-model versions. |
| CUDA runs out of memory | Reduce `--batch-size` and run each foundation model in its own environment and process. |
| Output is empty | Confirm that input quality control passed and that every external stage produced a non-empty configured output. |

## Licensed data and preprocessing

1. HGMD data are not included in this repository. Users with licensed access can reproduce non-coding or missense source-domain preparation with `1-Process-dataset-pipeline.ipynb`.
2. Random seeds used to reproduce the reported results are specified explicitly.
3. The Optuna hyperparameter-search script and its history log are provided in `./optuna`.

## Manuscript figures and additional analyses

### Main manuscript figures

| Figure | Description | Reproduction procedure |
|---|---|---|
| Figure 1A, B | Supporting analyses for the study background | Run the `Figure 1(A)` and `Figure 1(B)` sections of `./preprocess/1-Process-dataset-pipeline.ipynb`. |
| Figure 2 | Benchmarking against other VEPs on independent test sets | Run `cd ./benchmark/`, followed by `python -m evaluation_stats` and `python -m evaluation_plots`. |
| Figure 3, Figure S1, and Figure S8B | Analysis of why transfer learning improves performance | Run `main.py`, `layer_wise.py`, and the two Section III notebooks: `Result-section-III-NearBy-Source.ipynb` and `Result-section-III.ipynb`. For the UMAP visualizations, see `./additional_analysis/umap_visualizaion/` and run `held_out_umap.py` and `plot_individual_umap.py`. |
| Figure 4, Figure S2, and Figure S8A | Deep representations from language models and one-hot sequence analysis | Run `pretrain.py`, `finetune.py`, `ablation.py`, `zeroshot.py`, and `Result-section-IV.ipynb`. |
| Figure 5 and Figure S3 | Mechanism-annotation analyses | Run `Result-section-V.ipynb` using the released inference output. |

### Supplementary figures

| Figure | Description | Reproduction procedure |
|---|---|---|
| Figure S4A, F | Zero-shot performance on Test Set I using weights trained on a label-shuffled source domain | Run `cd ./additional_analysis/pretrain_controls/label_shuffle`, followed by `Pretrain_shuffle.py`. See the README in that directory for details. |
| Figure S4B, G | Fine-tuned performance on Test Set I using weights trained on a label-shuffled source domain | Run `Finetune.py` after changing the source-domain checkpoint to the checkpoint trained on shuffled source-domain labels. |
| Figure S4C, H | Performance after source-domain pretraining and fine-tuning while excluding samples associated with the target-domain test genes | Run `cd ./additional_analysis/pretrain_controls/exclude_test_gene/`, followed by `pretrain_exclude_genes.py` and `Finetune-Quick.py`. |
| Figure S4D, I | Evidence levels of positive source-domain samples | Run `./preprocess/1-Process-dataset-pipeline.ipynb` and navigate to the `### (D) 🚩 Evidence levels` cell. |
| Figure S4E, J | Source-domain training with and without DMq data | Run `cd ./additional_analysis/pretrain_controls/exclude_test_gene/`, followed by `pretrain_exclude_variants.py`, to obtain source-domain weights trained without DMq data. Then compare chromosome-split training performance with `./additional_analysis/pretrain_controls/exclude_dmq_data/Finetune-Quick-Chromosome-CV.py`. **Note:** This pre-model-construction analysis evaluates the trade-off between source-domain sample size and sample confidence; it is therefore conducted on the training dataset. |
| Figure S5A | Gene-overlap analysis | Run `./additional_analysis/domain_composition/gene_overlap/1-gene-overlap-info.ipynb`. |
| Figure S5B | 129-bp sequence-context overlap analysis | Run `cd ./additional_analysis/domain_composition/context_overlap/`, followed by `calculate_context_overlap.py`. Required arguments are documented in the README in the same directory. |
| Figure S5C | Genomic-coordinate overlap analysis | Run `cd ./additional_analysis/domain_composition/genomic_coordinate/`, followed by `Rscript variant_genomic_position_plot.R`. |
| Figure S6 | Mutation-spectrum analysis of non-coding and synonymous variants | Run `cd ./additional_analysis/domain_composition/mutation_specturm/`, followed by `python 2_tri_96_signature.py`. |
| Figure S7 | Mutation-spectrum analysis of missense and synonymous variants | Run `cd ./additional_analysis/domain_composition/mutation_specturm/`, followed by `python 1_SNV_six_signature.py`. |
| Figure S9 | Calibration analysis on Test Sets I and II | Run `cd ./additional_analysis/calibration_analysis`, followed by `python compute.py` and `python plot.py`. |

### Other code

- Source-domain data processing: `./preprocess/1-Process-dataset-pipeline.ipynb`.

## Output interpretation

The `predictions.tsv` file contains the following principal columns:

- `Variant38`: Variant identifier in the format `CHROM_POS_REF/ALT` on GRCh38.
- `TranSyn_score`: Floating-point deleterious-class probability in `[0, 1]`.

## Citation and contact

> In preparation.
