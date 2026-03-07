# Multitask Learning for Membrane Permeability Prediction across Assays with Prediction Reliability Assessment

This repository contains the source code and training data for the paper:

> **Multitask Learning for Membrane Permeability Prediction across Assays with Prediction Reliability Assessment**
> Yuki Doi and Tsuyoshi Esaki
> *[Journal]*, [Year]. DOI: [DOI]

<!-- TODO: Update journal name, year, and DOI when the paper is published. -->

## Overview

This project develops a multitask graph convolutional neural network (MT-GCN) to jointly predict apparent permeability (Papp) across five assays (Caco-2, MDCK, RRCK, LLC-PK1, and PAMPA), and benchmarks it against a fingerprint-based random forest (RF) and a single-task GCN (ST-GCN). In addition, an applicability domain (AD) framework is introduced to quantify prediction confidence at the compound level. Assay-specific error models, trained on ensemble dispersion, learned-space similarity, and local-consistency metrics, provide reliability scores that enable practical accuracy-coverage trade-offs.

## Repository Structure

```
.
├── 01_run_preprocessing.py          # Data preparation and train/test splitting
├── 02_run_cross_validation.py       # 10-fold CV for RF, single-task GCN, and multi-task GCN
├── 03_run_applicability_domain.py   # Applicability domain analysis and error classification
├── 04_run_subsampling_analysis.py   # Training data sufficiency analysis (Caco-2)
├── training_data/
│   └── SupportingInformation_PappValues.csv  # Papp values for ~12,000 compounds (from ChEMBL)
└── utils_for_admet_model/
    ├── __init__.py
    ├── applicability_domain.py      # Applicability domain metric computation
    ├── dataloader_loop.py           # Training/evaluation/inference loops
    ├── datasets.py                  # PyTorch Dataset for molecular graphs
    ├── execute_model.py             # High-level training/evaluation pipeline
    ├── models.py                    # GCN encoder and DNN decoder architectures
    └── utils.py                     # Utilities (seed fixing, early stopping, multi-task loss)
```

## Usage

The scripts are designed to be run sequentially:

### Step 1: Preprocessing

```bash
python 01_run_preprocessing.py
```

Loads compound data, generates Morgan fingerprints (radius=2, 1024 bits), creates a 90/10 train/test split, and prepares 10-fold cross-validation splits.

### Step 2: Cross-Validation

```bash
python 02_run_cross_validation.py
```

Trains and evaluates three model types under 10-fold cross-validation:
- **Random Forest (RF)**: Morgan fingerprint-based baseline (scikit-learn `RandomForestRegressor`)
- **Single-Task GCN (ST-GCN)**: one model per assay
- **Multi-Task GCN (MT-GCN)**: jointly predicts all five assays

### Step 3: Applicability Domain Analysis

```bash
python 03_run_applicability_domain.py
```

Trains a multi-task GCN on a fixed train/test split and computes applicability domain (AD) metrics:
- **PRED_STD**: standard deviation of predictions from the 5-fold ensemble
- **wRMSD1, wRMSD2**: similarity-weighted RMSD metrics in the learned feature space
- **SIM1, SIM5**: cosine similarity to nearest training compounds

Builds error classification models (Random Forest) that output a reliability score (probability that a prediction falls within 2-fold error) and generates SHAP explanations.

### Step 4: Subsampling Analysis

```bash
python 04_run_subsampling_analysis.py
```

Evaluates model performance under reduced Caco-2 training data (N = 100, 200, 500, 1000, 2500) with 5 random seeds per sample size, for all three model types.

## Data

`SupportingInformation_PappValues.csv` contains curated apparent permeability (Papp) values collected from the ChEMBL database (ver. 34). After curation (apical-to-basolateral direction, no overexpressed transporters, unit unification to 10^-6 cm/s, truncation, and MW &le; 700 Da filtering), the dataset contains:

| Assay | # Compounds | Description |
|-------|-------------|-------------|
| Caco-2 | 5,480 | Human intestinal epithelial |
| LLC-PK1 | 451 | Porcine kidney epithelial |
| MDCK | 1,276 | Madin-Darby canine kidney |
| PAMPA | 4,378 | Parallel artificial membrane permeability assay |
| RRCK | 895 | Ralph Russ canine kidney (low-efflux MDCK variant) |

Columns: `ChEMBL_ID`, `SMILES`, `Caco-2`, `LLC-PK1`, `MDCK`, `PAMPA`, `RRCK` (all Papp values in log10 scale). Most compounds have data for only one or two assays; missing values are represented as empty cells.

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{doi2025multitask,
  title   = {Multitask Learning for Membrane Permeability Prediction across Assays with Prediction Reliability Assessment},
  author  = {Doi, Yuki and Esaki, Tsuyoshi},
  journal = {TODO},
  year    = {TODO},
  doi     = {TODO}
}
```

<!-- TODO: Update journal, year, and DOI when the paper is published. -->

## Acknowledgments

This study was supported by JSPS KAKENHI (Grant Number: JP23K14382).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
