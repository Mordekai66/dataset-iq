# dataset-iq

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/status-active-green.svg)
![Last Commit](https://img.shields.io/github/last-commit/Mordekai66/dataset-iq)
![Repo Size](https://img.shields.io/github/repo-size/Moredekai66/dataset-iq)
![Visitors](https://komarev.com/ghpvc/?username=Mordekai66&repo=dataset-iq)

---

## Overview

Dataset-IQ is a structured system for organizing machine learning datasets with automated statistics generation, validation, and standardized metadata.

It transforms raw datasets into self-contained units with computed analysis, enabling consistency, reproducibility, and easier dataset comparison.

---

## Structure

Each dataset is stored as a self-contained unit:
```cmd
data/ml/<dataset_name>/
├── data.csv / .xlsx
├── dataset.json
├── stats.json
└── README.md

```

---

## Core Features

- Automatic dataset statistics generation
- Standardized metadata schema
- Machine-readable dataset descriptions
- Support for CSV and Excel datasets
- Self-contained dataset structure

---

## Statistics Generated

Each dataset includes:

- Number of rows and columns
- Missing values count
- Duplicate records
- Numeric vs categorical features
- Class distribution (if applicable)
- Basic quality indicators

---

## Usage

Run stats generation locally:

```bash
python -c "from core.stats import run_all; run_all()"
```

---

## Goal

To create a unified, reproducible, and machine-readable dataset registry for machine learning workflows.

---

## License

MIT
