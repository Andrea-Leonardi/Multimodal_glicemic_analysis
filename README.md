# Multimodal Glycemic Analysis

Questo repository unisce dati Glooko e Xiaomi per costruire un dataset giornaliero, generare feature laggate e provare modelli statistici e di machine learning su indicatori glicemici.

## Struttura

- `multimodal_glycemic_analysis/Data/`: export grezzi e file processati.
- `multimodal_glycemic_analysis/Scripts/`: cleaning, merge, feature engineering, modeling e runner della pipeline.
- `tests/`: test leggeri sui componenti più fragili della pipeline.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Esecuzione

Pipeline completa:

```bash
python multimodal_glycemic_analysis/Scripts/pipeline.py all
```

Step separati:

```bash
python multimodal_glycemic_analysis/Scripts/pipeline.py clean
python multimodal_glycemic_analysis/Scripts/pipeline.py features --lag-days 2
python multimodal_glycemic_analysis/Scripts/pipeline.py train --task all --holdout-days 14
```

Script singoli:

```bash
python multimodal_glycemic_analysis/Scripts/MergeData.py
python multimodal_glycemic_analysis/Scripts/Processing_and_descriptive.py
python multimodal_glycemic_analysis/Scripts/MLapplications.py --task regression
```

## Miglioramenti già introdotti

- Pipeline esplicita senza side effect impliciti tra gli script.
- Discovery automatica degli export Glooko presenti nella cartella dati.
- Feature laggate costruite in modo riutilizzabile.
- Training con split temporale train/holdout e imputazione esplicita dei valori mancanti.
- Documentazione minima e test di base.

## Test

```bash
python -m unittest discover -s tests
```
