# Synthetic Data Benchmark

A reproducible benchmark comparing three approaches to synthetic tabular data
generation — **Gaussian Copula**, **CTGAN**, and **LLM-based few-shot
generation** — across two public datasets (UCI Adult Income, Credit Card
Fraud Detection), evaluated on statistical fidelity, machine learning utility
(Train-Synthetic-Test-Real), and privacy risk (Distance to Closest Record).

This project was built to answer a concrete question directly: **how do you
actually benchmark synthetic data generation methods against each other?**
Rather than a theoretical answer, this repository provides a working,
reproducible implementation.

## Key Finding

Aggregate privacy metrics can hide real risk: Gaussian Copula's *mean*
distance-to-closest-record on the Adult dataset looked safe (0.091), but its
*minimum* was exactly 0 — at least one synthetic row was an exact duplicate
of a real training record. See [`report/REPORT.md`](report/REPORT.md) for
the full write-up, including a discussion of how schema completeness affects
both fidelity and privacy scores.

## Data

Raw, holdout, and synthetic data files are **not included** in this
repository (the Credit Card Fraud dataset alone is ~144MB, well over
GitHub's practical file-size limits, and committing generated data is not
good practice for a reproducible ML project). To regenerate everything
locally:

```bash
kaggle datasets download -d uciml/adult-census-income -p data/raw --unzip
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

Then run `notebooks/01_data_prep.ipynb` to recreate the train/holdout splits,
followed by `02` → `04` to regenerate the synthetic datasets. `results/`
(the metric table and figures) *is* included, since these are the actual
outputs the report is based on and are small enough to version normally.

## Quick Start

```bash
conda create -n synthdata python=3.11 -y
conda activate synthdata
pip install -r requirements.txt
```

Add your API key to `.env` (see `.env.example`), then run the notebooks in
order:

```
notebooks/01_data_prep.ipynb
notebooks/02_gaussian_copula.ipynb
notebooks/03_ctgan.ipynb
notebooks/04_llm_generation.ipynb
notebooks/05_metrics_evaluation.ipynb
notebooks/06_analysis_visualization.ipynb
```

## Project Structure

```
synthetic-data-benchmark/
├── data/
│   ├── raw/            # Kaggle source data + train split
│   ├── holdout/         # Untouched real data, used only for final evaluation
│   └── synthetic/        # Output of each generation method
├── notebooks/            # Analysis pipeline, run in numeric order
├── results/
│   ├── raw_results.csv   # Full metric table
│   └── figures/          # Generated charts
├── report/
│   └── REPORT.md         # Full methodology, results, discussion, limitations
├── requirements.txt
└── .env.example
```

## Methodology (short version)

- **Datasets:** UCI Adult Income, Credit Card Fraud Detection — 70/30
  stratified holdout split, fixed seed, holdout never exposed to generation.
- **Methods:** Gaussian Copula (SDV), CTGAN (SDV), LLM few-shot generation
  (Google Generative AI).
- **Metrics:** SDMetrics fidelity score, TSTR utility gap (Random Forest,
  weighted F1), mean/min Distance to Closest Record, schema coverage.

Full methodology, results, discussion, and disclosed limitations are in
[`report/REPORT.md`](report/REPORT.md).

## License

This project uses publicly available datasets (UCI Adult Income, Credit Card
Fraud Detection via Kaggle) under their respective licenses. Code in this
repository is provided as-is for research and educational purposes.
