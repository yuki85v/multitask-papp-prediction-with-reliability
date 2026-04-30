# Multitask Learning for Membrane Permeability Prediction across Assays with Prediction Reliability Assessment

This repository contains the source code and training data for the paper:

> **Multitask Learning for Membrane Permeability Prediction across Assays with Prediction Reliability Assessment**
> Yuki Doi and Tsuyoshi Esaki
> *[Journal]*, [Year]. DOI: [DOI]

<!-- TODO: Update journal name, year, and DOI when the paper is published. -->

## Overview

This project develops a multitask graph convolutional neural network (MT-GCN) to jointly predict apparent permeability (Papp) across five assays (Caco-2, MDCK, RRCK, LLC-PK1, and PAMPA), and benchmarks it against a fingerprint-based random forest (RF, both default and grid-search-tuned), a single-task GCN (ST-GCN), and a Caco-2-pretrained ST-GCN fine-tuned per assay (transfer learning). All models are evaluated under both random and Murcko-scaffold-grouped 10-fold cross-validation. In addition, an applicability domain (AD) framework is introduced to quantify prediction confidence at the compound level. Assay-specific error models, trained on ensemble dispersion, learned-space similarity, and local-consistency metrics, provide reliability scores that enable practical accuracy-coverage trade-offs. The deployed MT-GCN is further analyzed through dataset-level diagnostics, leave-one-assay-out ablation, Integrated Gradients atom-level attributions, and cross-assay substructure analysis on selected case studies.

## Repository Structure

```
.
├── 01_run_preprocessing.py             # Data prep, fingerprints, random + scaffold splits (hold-out & 10-fold CV)
├── 02_run_cross_validation.py          # 10-fold CV for RF, ST-GCN, MT-GCN (random + scaffold splits)
├── 02b_run_tuned_rf.py                 # Tuned RF baseline via nested GridSearchCV (random + scaffold splits)
├── 02c_run_transfer_learning.py        # Caco-2-pretrained ST-GCN fine-tuned to LLC-PK1/MDCK/PAMPA/RRCK
├── 03_run_applicability_domain.py      # Applicability domain analysis and error classification
├── 04_run_subsampling_analysis.py      # Training data sufficiency analysis (Caco-2; tuned RF + GCNs)
├── 05_run_dataset_analysis.py          # Per-assay counts, overlaps, label/chemspace correlations, physchem
├── 06_run_interpretability.py          # Cross-assay readout/prediction analysis, RRCK error mining, IG attributions
├── 06b_run_case_study_substructures.py # Cross-assay substructure analysis for case-study compounds
├── 07_run_revision_visualizations.py   # Reliability diagrams, coverage-vs-RMSE, parity plots, t-SNE embeddings
├── 08_run_loao_mt_gcn.py               # Leave-one-assay-out ablation of MT-GCN
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

The scripts are designed to be run sequentially. Steps 1–3 produce the deployed model and reliability framework; steps 4–8 reproduce the additional benchmarks, interpretability analyses, and revision figures reported in the paper.

### Step 1: Preprocessing

```bash
python 01_run_preprocessing.py
```

Loads compound data, generates Morgan fingerprints (radius=2, 1024 bits), and creates two evaluation regimes for each of two splitting strategies:
- **Random split** — 90/10 train/test hold-out and 10-fold CV
- **Scaffold split** — Murcko-scaffold-grouped 90/10 hold-out and a greedy 10-fold scaffold CV (largest scaffolds assigned to the smallest fold)

Murcko scaffolds are saved to `training_data/murcko_scaffolds.csv`, and per-fold Papp distribution plots are written under `trained_model/{random_split,scaffold_split}{,_cv}/`.

### Step 2: Cross-Validation

```bash
python 02_run_cross_validation.py
```

Trains and evaluates three model types under 10-fold cross-validation, separately for the random and scaffold splits:
- **Random Forest (RF)**: Morgan fingerprint-based baseline (scikit-learn `RandomForestRegressor`)
- **Single-Task GCN (ST-GCN)**: one model per assay
- **Multi-Task GCN (MT-GCN)**: jointly predicts all five assays

### Step 2b: Tuned Random Forest Baseline

```bash
python 02b_run_tuned_rf.py
```

Runs a nested grid search (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`) inside each outer CV fold to provide a strengthened RF baseline for both random and scaffold splits. Best parameters per fold/assay and held-out metrics are saved alongside the per-fold predictions. Set `GS_N_JOBS` to control parallelism.

### Step 2c: Transfer Learning Baseline

```bash
python 02c_run_transfer_learning.py
```

Fine-tunes the Caco-2-pretrained ST-GCN encoder for each remaining assay (LLC-PK1, MDCK, PAMPA, RRCK) with a small encoder learning rate (`1e-5`) and standard decoder learning rate (`1e-4`). Uses the same 10-fold outer split × 5-fold inner ensemble as ST/MT-GCN, for both random and scaffold splits. Provides a transfer-learning reference point against which MT-GCN is compared.

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

Evaluates model performance under reduced Caco-2 training data (N = 100, 200, 500, 1000, 2500) with 5 random seeds per sample size, for all three model types. The RF baseline is tuned per (N, seed) via the same `GridSearchCV` grid as Step 2b.

### Step 5: Dataset Analysis

```bash
python 05_run_dataset_analysis.py
```

Characterizes the curated dataset and writes tables/figures to `results_from_manuscript/dataset_analysis/`:
- per-assay compound counts and per-compound assay-coverage histogram
- exact and inclusive intersection counts across assay subsets
- pairwise overlap and pairwise Pearson correlation of log Papp on shared compounds
- chemical-space relatedness via nearest-neighbour Tanimoto similarity (Morgan r=2)
- physicochemical descriptor distributions (logP, MW, TPSA, HBD, HBA) by assay
- per-assay label distribution summaries

### Step 6: Interpretability

```bash
python 06_run_interpretability.py
python 06b_run_case_study_substructures.py
```

Uses the deployed 5-fold MT-GCN ensemble to:
- compute cross-assay correlation of MT-GCN predictions on the holdout
- correlate readout dimensions with physicochemical descriptors and report top-loaded dimensions
- mine RRCK failure modes under the scaffold split CV (worst ST-GCN errors and scaffold enrichment)
- select case-study compounds where MT-GCN outperforms both ST-GCN and tuned RF
- generate Integrated Gradients (Captum) atom-level attributions on those case studies

`06b_run_case_study_substructures.py` then performs cross-assay substructure analysis on the selected case-study compounds: top-3 whole-molecule nearest neighbours per other assay, per-bit cross-assay prevalence and enrichment, top-5 enriched bits with fold changes and fragment SMILES, and overlap between enriched-substructure atoms and top-IG atoms. Outputs are written to `results_from_manuscript/interpretability/`.

### Step 7: Revision Visualizations

```bash
python 07_run_revision_visualizations.py
```

Produces calibration and embedding-comparison figures for the revised manuscript under `results_from_manuscript/revision_visualizations/`:
- per-assay reliability diagrams and uncertainty metrics (AUROC, PR-AUC, Brier, ECE)
- coverage-vs-RMSE curves as the reliability threshold tightens
- 10-fold CV parity plots with 2-fold error bands and ensemble error bars (random and scaffold splits)
- t-SNE embeddings of the holdout in MT-GCN readout space vs. Morgan fingerprint space

### Step 8: Leave-One-Assay-Out Ablation

```bash
python 08_run_loao_mt_gcn.py
```

Re-trains MT-GCN on four-of-five assays in turn (using the same 90/10 random hold-out as the deployed model) and contrasts each LOAO model with the full five-assay MT-GCN. Reports per-assay ΔRMSE / ΔR² / ΔMAE and an overall assay-importance ranking, quantifying how much each dropped assay contributes to the shared representation.

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
