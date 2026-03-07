# Training Data

This folder should contain the following file before running the scripts:

## Required File

| File | Description |
|------|-------------|
| `SupportingInformation_PappValues.csv` | Curated Papp values for ~12,000 compounds across 5 assays |

## How to Obtain

`SupportingInformation_PappValues.csv` is provided as Supporting Information of the following paper:

> **Multitask Learning for Membrane Permeability Prediction across Assays with Prediction Reliability Assessment**
> Yuki Doi and Tsuyoshi Esaki

Download the CSV file from the journal's Supporting Information page and place it in this directory (`training_data/`).

## Expected Format

The CSV file should contain the following columns:

| Column | Description |
|--------|-------------|
| `ChEMBL_ID` | Compound identifier from ChEMBL |
| `SMILES` | Canonical SMILES string |
| `Caco-2` | log10 Papp (10^-6 cm/s) |
| `LLC-PK1` | log10 Papp (10^-6 cm/s) |
| `MDCK` | log10 Papp (10^-6 cm/s) |
| `PAMPA` | log10 Papp (10^-6 cm/s) |
| `RRCK` | log10 Papp (10^-6 cm/s) |

Missing assay values are represented as empty cells.
