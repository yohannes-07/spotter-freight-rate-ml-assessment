"""Model selection and evaluation utilities."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.features import CATEGORICAL_COLUMNS, FreightPreprocessor


DEFAULT_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "depth_4",
        "depth": 4,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
    },
    {
        "name": "depth_7",
        "depth": 7,
        "learning_rate": 0.04,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.5,
    },
    {
        "name": "depth_6_slow",
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.25,
    },
]

FINAL_CONFIG: dict[str, Any] = {
    "name": "rpm_rmse_distance",
    "depth": 6,
    "learning_rate": 0.04,
    "l2_leaf_reg": 8.0,
    "random_strength": 0.5,
}
FINAL_ITERATIONS = 138


def chronological_split(
    frame: pd.DataFrame, holdout_days: int = 61
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(frame["date"], errors="raise")
    cutoff = dates.max() - pd.Timedelta(days=holdout_days - 1)
    development = frame.loc[dates < cutoff].copy()
    holdout = frame.loc[dates >= cutoff].copy()
    if development.empty or holdout.empty:
        raise ValueError("Chronological split produced an empty partition")
    return development, holdout


def regression_metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    residuals = actual_values - predicted_values
    denominator = np.sum(np.square(actual_values - actual_values.mean()))
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "mae": float(np.mean(np.abs(residuals))),
        "r2": float(1.0 - np.sum(np.square(residuals)) / denominator),
    }


def model_parameters(config: dict[str, Any], iterations: int) -> dict[str, Any]:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": iterations,
        "depth": config["depth"],
        "learning_rate": config["learning_rate"],
        "l2_leaf_reg": config["l2_leaf_reg"],
        "random_strength": config["random_strength"],
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
        "random_seed": 42,
        "allow_writing_files": False,
        "thread_count": -1,
    }


def _normalized_distance_weight(distance: pd.Series, power: int = 1) -> np.ndarray:
    values = pd.to_numeric(distance, errors="raise").to_numpy(dtype=float) ** power
    return values / values.mean()


def evaluate_final_strategy(
    frame: pd.DataFrame,
    *,
    holdout_days: int = 61,
    max_iterations: int = 2_500,
    early_stopping_rounds: int = 150,
    verbose: int | bool = 200,
) -> tuple[dict[str, float], int]:
    development, holdout = chronological_split(frame, holdout_days)
    preprocessor = FreightPreprocessor()
    x_development = preprocessor.fit_transform(development)
    x_holdout = preprocessor.transform(holdout)

    development_rpm = development["posted_rate"].to_numpy() / x_development["distance"].to_numpy()
    holdout_rpm = holdout["posted_rate"].to_numpy() / x_holdout["distance"].to_numpy()
    development_pool = Pool(
        x_development,
        label=development_rpm,
        cat_features=CATEGORICAL_COLUMNS,
        weight=_normalized_distance_weight(x_development["distance"]),
    )
    holdout_pool = Pool(
        x_holdout,
        label=holdout_rpm,
        cat_features=CATEGORICAL_COLUMNS,
        # This makes validation RMSE proportional to squared error in dollars.
        weight=_normalized_distance_weight(x_holdout["distance"], power=2),
    )

    model = CatBoostRegressor(**model_parameters(FINAL_CONFIG, max_iterations))
    model.fit(
        development_pool,
        eval_set=holdout_pool,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )
    prediction = (
        np.clip(model.predict(x_holdout), 0.01, None)
        * x_holdout["distance"].to_numpy()
    )
    metrics = regression_metrics(holdout["posted_rate"], prediction)
    return metrics, max(model.get_best_iteration() + 1, 1)


def run_experiments(
    frame: pd.DataFrame,
    configs: list[dict[str, Any]] | None = None,
    *,
    holdout_days: int = 61,
    iterations: int = 2_000,
    early_stopping_rounds: int = 150,
    verbose: int | bool = 200,
) -> tuple[pd.DataFrame, FreightPreprocessor]:
    configs = configs or DEFAULT_CONFIGS
    development, holdout = chronological_split(frame, holdout_days)

    preprocessor = FreightPreprocessor()
    x_development = preprocessor.fit_transform(development)
    x_holdout = preprocessor.transform(holdout)
    y_development = development["posted_rate"]
    y_holdout = holdout["posted_rate"]

    rows: list[dict[str, Any]] = []
    for config in configs:
        started = perf_counter()
        model = CatBoostRegressor(**model_parameters(config, iterations))
        model.fit(
            x_development,
            y_development,
            cat_features=CATEGORICAL_COLUMNS,
            eval_set=(x_holdout, y_holdout),
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        predictions = np.clip(model.predict(x_holdout), 1.0, None)
        metrics = regression_metrics(y_holdout, predictions)
        best_iteration = model.get_best_iteration()
        rows.append(
            {
                "name": config["name"],
                **metrics,
                "best_iteration": max(int(best_iteration) + 1, 1),
                "training_seconds": round(perf_counter() - started, 2),
                "depth": config["depth"],
                "learning_rate": config["learning_rate"],
                "l2_leaf_reg": config["l2_leaf_reg"],
                "random_strength": config["random_strength"],
            }
        )

    results = pd.DataFrame(rows).sort_values("rmse", ignore_index=True)
    return results, preprocessor


def fit_final_model(
    frame: pd.DataFrame,
    config: dict[str, Any] = FINAL_CONFIG,
    iterations: int = FINAL_ITERATIONS,
    *,
    verbose: int | bool = 200,
) -> tuple[CatBoostRegressor, FreightPreprocessor]:
    preprocessor = FreightPreprocessor()
    features = preprocessor.fit_transform(frame)
    rate_per_mile = frame["posted_rate"].to_numpy() / features["distance"].to_numpy()
    training_pool = Pool(
        features,
        label=rate_per_mile,
        cat_features=CATEGORICAL_COLUMNS,
        weight=_normalized_distance_weight(features["distance"]),
    )
    model = CatBoostRegressor(**model_parameters(config, iterations))
    model.fit(
        training_pool,
        verbose=verbose,
    )
    return model, preprocessor
