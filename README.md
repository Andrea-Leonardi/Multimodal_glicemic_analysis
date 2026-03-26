# Multimodal Glycemic Analysis

## Overview

This project explores whether daily glycemic outcomes can be explained and partially predicted by combining glucose, insulin, carbohydrate intake, sleep, heart-rate, step count, calories, and training data from multiple personal health devices.

The repository is built around a real-world self-tracking dataset collected from:

- Glooko exports containing CGM, insulin, alarms, and meal-related information
- Xiaomi wearable exports containing sleep, activity, heart-rate, calories, and workout signals

The main goal is to turn heterogeneous raw sensor exports into a clean daily analytical dataset and benchmark a set of statistical and machine learning models for glycemic-risk forecasting.

## Why This Project Matters

This is a compact end-to-end data science project that combines:

- multimodal health data integration
- raw CSV and JSON-like payload cleaning
- time-aware feature engineering
- reproducible modeling workflows
- interpretable terminal reporting

It is designed as both a personal research project and a portfolio-ready example of applied machine learning on longitudinal physiological data.

## Problem Statement

The project asks a practical question:

Can daily glycemic quality be forecast from recent behavioral and physiological patterns?

More specifically, the pipeline builds daily targets such as:

- time-weighted glucose mean
- glucose variability
- time in range
- time above range
- rate of area above a hyperglycemic threshold

These glycemic indicators are then paired with lagged features derived from recent days of:

- sleep duration and sleep score
- bedtime timing
- physical activity and training
- carbohydrate intake
- insulin-related events

## Dataset and Modalities

The repository contains personal data exports used for experimentation and model development.

### Glooko-derived signals

- CGM time series
- insulin totals
- bolus-related meal entries
- cartridge replacement events
- alarm and event logs

### Xiaomi-derived signals

- daily calories
- daily steps
- heart-rate summaries
- sleep structure and sleep timing
- training session indicators

## Pipeline Architecture

The workflow is intentionally split into explicit stages.

### 1. Cleaning and harmonization

The cleaning scripts parse raw vendor exports, normalize timestamps, convert numeric fields, unpack JSON-encoded payloads, and aggregate each modality to a daily level.

Relevant scripts:

- `multimodal_glycemic_analysis/Scripts/XiaomiDataCleaning.py`
- `multimodal_glycemic_analysis/Scripts/GlookoDataCleaning.py`

### 2. Dataset merge

The cleaned daily Xiaomi and Glooko tables are aligned on a normalized daily index and merged into one unified analytical dataset.

Relevant script:

- `multimodal_glycemic_analysis/Scripts/MergeData.py`

### 3. Feature engineering

The merged dataset is enriched with lagged predictors so that recent history can be used for forecasting.

Examples:

- previous-day glucose summary
- previous-day carbohydrate intake
- previous-day training indicator
- previous-day sleep score and duration
- day index for temporal ordering

Relevant script:

- `multimodal_glycemic_analysis/Scripts/Processing_and_descriptive.py`

### 4. Modeling

The project benchmarks both regression and classification approaches using time-aware validation.

Regression models currently include:

- Lasso
- Elastic Net
- PCR with Ridge
- Partial Least Squares
- HistGradientBoostingRegressor

Classification models currently include:

- Logistic Regression
- Elastic Net Logistic Regression
- Gaussian Naive Bayes
- Linear Discriminant Analysis
- Linear SVM
- RBF SVM
- HistGradientBoostingClassifier

Relevant script:

- `multimodal_glycemic_analysis/Scripts/MLapplications.py`

### 5. Orchestration

The entire workflow can be launched from a single entry point:

- `multimodal_glycemic_analysis/Scripts/pipeline.py`

## Modeling Strategy

The modeling code follows a deliberately conservative structure:

- lagged features are used to reduce obvious same-day leakage
- train and holdout sets are split chronologically
- cross-validation uses `TimeSeriesSplit`
- missing values are handled explicitly
- results are saved to CSV and displayed in readable terminal tables

The current evaluation focuses on:

- regression `R²` for continuous glycemic targets
- `ROC AUC` for binary risk formulations built from target quantiles

## Project Structure

```text
multimodal_glycemic_analysis/
├── multimodal_glycemic_analysis/
│   ├── Data/
│   │   ├── glooko/
│   │   ├── xiaomi/
│   │   └── processed/
│   └── Scripts/
│       ├── config.py
│       ├── GlookoDataCleaning.py
│       ├── XiaomiDataCleaning.py
│       ├── MergeData.py
│       ├── Processing_and_descriptive.py
│       ├── MLapplications.py
│       └── pipeline.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

### Run the full pipeline

```bash
python multimodal_glycemic_analysis/Scripts/pipeline.py all
```

### Rebuild cleaned and merged data only

```bash
python multimodal_glycemic_analysis/Scripts/pipeline.py clean
```

### Build lagged features only

```bash
python multimodal_glycemic_analysis/Scripts/pipeline.py features --lag-days 2
```

### Train and compare models

```bash
python multimodal_glycemic_analysis/Scripts/pipeline.py train --task all --holdout-days 14
```

### Reuse previously processed data for faster experiments

```bash
python multimodal_glycemic_analysis/Scripts/pipeline.py train --task classification --reuse-processed
```

## Outputs

The pipeline writes intermediate and final artifacts to `multimodal_glycemic_analysis/Data/processed/`.

Main outputs include:

- `xiaomi_data_cleaned.parquet`
- `glooko_data_cleaned.parquet`
- `data_cleaned.parquet`
- `data_processed.parquet`
- `data_lagged.parquet`
- `dataproc.csv`
- `ml_results.csv`

## Technical Highlights

This repository demonstrates:

- practical ETL on messy multi-source health exports
- transformation of mixed raw formats into structured analytical tables
- time-weighted glucose summary statistics
- domain-specific feature rules for insulin and carbohydrate events
- modular experiment orchestration
- maintainable Python scripts instead of notebook-only logic
- lightweight automated testing for critical pipeline helpers

## Tests

```bash
python -m unittest discover -s tests
```

## Current Limitations

This is still a research-oriented project rather than a production medical system.

Important limitations include:

- relatively small sample size
- single-subject longitudinal data
- daily aggregation may hide within-day dynamics
- predictive performance is still modest, especially on regression tasks

## Future Directions

Natural next steps include:

- richer rolling-window features
- explicit next-day forecasting targets
- post-prandial event modeling
- probabilistic or quantile prediction
- SHAP or permutation-based feature importance analysis
- experiment tracking and model comparison dashboards

## Portfolio Positioning

From a CV or portfolio perspective, this project shows the ability to:

- design a complete data pipeline from raw data to model evaluation
- work with noisy real-world physiological data
- write reusable analytical code rather than one-off analyses
- reason about time-series leakage and temporal validation
- communicate technical work clearly through documentation and structured outputs
