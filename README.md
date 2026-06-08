# TranSyn

## 0 🌐 Access TranSyn Prediction

We provide genome-wide predictions on actionable genes using TranSyn and a web server to visualize the results. Access it here: [TranSyn Web Server](https://bioinfo.ahu.edu.cn/TranSyn).

![Figure1](asset\WebSerber-header.jpg)

## 1 📢 Introduction

**TranSyn** is a deep transfer learning framework for predicting the functional impact of synonymous variants.  

Although synonymous variants do not change the amino acid sequence, they can influence splicing, translation efficiency, and protein structure, potentially contributing to disease. Accurate prediction remains challenging due to limited labeled pathogenic samples.  

TranSyn addresses this by integrating:

- Deep representations from foundation models pretrained on noncoding and missense variants  
- Biologically informed handcrafted features  

Experiments demonstrate that TranSyn achieves state-of-the-art performance, effectively transferring knowledge from related variant types and showing particularly strong gains under small-sample conditions. Genome-wide inference on actionable genes further highlights potential mechanisms of synonymous variant pathogenicity.

![Figure1](asset\Framework.jpg)

---

## 2 🧱 Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate transyn
```

## 3 📈 Reproducibility

We provide instructions to reproduce all figures and tables from the paper.

1. **Section II:** Comparison with other VEPs | `Figure 2`
   1. `main.py`: Obtain the best model. Run directly: `python main.py`.
   2. Result-section-II.ipynb: Compare with other VEPs.
2. **Section III:** Improvement from transfer knowledge | `Figure 3`
   1. `main.py`: Sample efficiency experiments.
   2. `layer_wise.py`: Layer-wise transfer experiment. Run directly: `python layer_wise.py`.
   3. `Result-section-III-NearBy-Source.ipynb`: Re-select source domains and evaluate improvement.
   4. `Result-section-III.ipynb`: Result visualization.
3. **Section IV:** Contribution from foundation models | `Figure 4`
   1. `pretrain.py`: Pretraining with one-hot features and deep representations.
   2. `finetune.py`: Fine-tuning with one-hot features and deep representations.
   3. `ablation.py`: Ablation experiments. Run directly: `python ablation.py`.
   4. `zeroshot.py`: Zero-shot prediction using foundation models.
   5. 4. `Result-section-IV.ipynb`: Result visualization.
4. **Section V:** Inference on actionable genes | `Figure 5`
   1. `Result-section-V.ipynb`: Analysis and visualization.


## 4 🧾 Citation

> In preparation.

## 5 📩 Contact

> For questions or issues, please open an `Issue` in this repository.