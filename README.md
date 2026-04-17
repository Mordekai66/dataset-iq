# dataset-iq

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/status-active-green.svg)
![Last Commit](https://img.shields.io/github/last-commit/Mordekai66/dataset-iq)
![Repo Size](https://img.shields.io/github/repo-size/Mordekai66/dataset-iq)
![Visitors](https://komarev.com/ghpvc/?username=Mordekai66&repo=dataset-iq)

---

## Overview

Dataset-IQ is a structured system for organizing machine learning datasets with automated statistics generation, validation, and standardized metadata.

It transforms raw datasets into self-contained units with computed analysis, enabling consistency, reproducibility, and easier dataset comparison.

---

## Demo
![](demo.gif)

---

## Core Features

- Automatic dataset statistics generation via `core/stats.py`
- Standardized metadata schema per dataset
- Machine-readable dataset descriptions
- Support for CSV and Excel datasets
- Flask-based web UI with dataset browsing and detail views
- Data quality scoring with issue detection (missing values, duplicates, high correlation)
- GitHub Actions workflow for auto-generating stats on push

---

## Structure

Datasets are stored flat inside `data/ml/`:

```
data/ml/
├── <dataset_name>.csv / .xlsx
└── <dataset_name>.stats.json
```

Each `.stats.json` file is auto-generated and contains:

```json
{
  "summary": { "rows": ..., "columns": ..., "data_quality_score": ..., "problem_type": ..., "target": ... },
  "issues": { "missing_values_total": ..., "duplicate_rows": ..., "highly_correlated_columns": [...], "columns_with_high_missing": [...] },
  "schema": [ { "name": ..., "type": ..., "missing_pct": ..., "unique_values": ..., "stats": { "min": ..., "max": ..., "mean": ... } } ]
}
```

---

## Repo Structure

```
dataset-iq/
├── .github/
│   └── workflows/
│       └── stats.yml             # GitHub Actions — auto-runs stats on push
├── core/
│   └── stats.py                  # Stats generation logic for any dataset file
├── data/
│    └── ml/
│       ├── <dataset_name>.csv    # Raw dataset file (CSV or Excel)
│       └── <dataset_name>.stats.json  # Auto-generated stats for that dataset
├── app.py                        # Flask app — routes and API endpoints
├── requirements.txt              # Python dependencies
├── static/
│   └── style.css                 # Global stylesheet for all pages
├── templates/
│   ├── index.html                # Homepage — dataset grid with filters
│   ├── dataset.html              # Detail page — schema, issues, data preview
│   └── 404.html                  # 404 error page
├── README.md                     # Main md file
└── LICENSE                       # License file
```

---

## Statistics Generated

Each dataset includes:

- Number of rows and columns
- Data quality score (0–100)
- Problem type detection (classification / regression)
- Target column identification
- Missing values count and percentage per column
- Duplicate records count
- Highly correlated column pairs (threshold > 0.90)
- Columns exceeding 30% missing values

---

## Usage

**Run the web app locally:**

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`.

**Generate stats manually:**

```bash
python -c "from stats import run_all; run_all()"
```

**Stats are also auto-generated via GitHub Actions** on every push that modifies files under `data/ml/`.

---

## Goal

To create a unified, reproducible, and machine-readable dataset registry for machine learning workflows.

---

## Contributing

Fork the repository, add your dataset (CSV or Excel) into `data/ml/`, then open a Pull Request.

Once merged, stats are generated automatically.

This keeps datasets versioned, traceable, and safe to integrate while allowing contributors without write access to still submit work.

Every dataset added improves the registry and makes it easier to reuse structured ML data instead of rebuilding it from scratch.

---

## License

MIT
