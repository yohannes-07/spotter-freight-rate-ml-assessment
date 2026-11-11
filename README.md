# Freight Rate Prediction

Predicts `posted_rate` for the 12,000 loads in `data/validation.csv` and the fixed December route in `data/december-chart-inputs.csv`.

## Training and validation

The data is split by date, not randomly:

- Model development: January 1 to August 31, 2025
- Holdout: September 1 to October 31, 2025
- Unlabeled validation file: November 1 to December 31, 2025

## Model choice

CatBoost was selected because the dataset is small, contains several categorical features, and has more than 4,000 origin-destination lanes. Some validation lanes do not appear in training. CatBoost handles these categories and unseen values directly, without one-hot encoding or manual target encoding. Hyperparameters are compared on the same chronological holdout before the final fit.

## Features

- Load: distance, weight, equipment, market index, quote signal
- Lane: pickup, delivery, and pickup-delivery pair
- Geography: coordinates, Haversine distance, coordinate deltas, route midpoint
- Time: month, weekday, day/week of year, elapsed days, and cyclical encodings

Negative weights are corrected to their absolute values. Missing numeric values use medians learned from the fitting partition, with indicators retained for missing weight and market index values. Missing coordinates can be recovered from the median coordinate observed for the corresponding city.

## Run

Python 3.10 or newer is required.

```bash
python -m pip install -r requirements.txt
python -m src.train
python -m src.predict
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```

Training writes the following files to `artifacts/`:

- `freight_rate_model.cbm`
- `preprocessor.pkl`
- `metrics.json`
- `experiment_results.csv`
- `feature_importance.csv`

Prediction writes:

- `validation_predictions.csv`
- updated `data/december-chart-inputs.csv`

The scorer checks both CSV files and writes `scorer_results/candidate_december.png`.

## Repository

```text
data/                              Input data and prediction template
notebooks/01_eda.ipynb             Exploratory analysis
notebooks/02_feature_engineering.ipynb  Feature development and validation
notebooks/03_catboost_tuning.ipynb CatBoost hyperparameter results
notebooks/04_model_evaluation.ipynb Error analysis and target-strategy comparison
src/features.py                    Reusable feature pipeline
src/modeling.py                    Validation and model utilities
src/train.py                       Final model fit
src/predict.py                     Submission inference
score.py                           Submission checks and December chart
```

Use `python -m src.train --help` or `python -m src.predict --help` to override paths and training parameters.
