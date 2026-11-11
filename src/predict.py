"""Generate submission files."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.features import FreightPreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def load_model(artifact_dir: Path) -> tuple[CatBoostRegressor, FreightPreprocessor]:
    model_path = artifact_dir / "freight_rate_model.cbm"
    preprocessor_path = artifact_dir / "preprocessor.pkl"
    if not model_path.is_file() or not preprocessor_path.is_file():
        raise FileNotFoundError("Model artifacts not found. Run `python -m src.train` first.")
    model = CatBoostRegressor()
    model.load_model(model_path)
    with preprocessor_path.open("rb") as stream:
        preprocessor = pickle.load(stream)
    return model, preprocessor


def predict_rates(
    model: CatBoostRegressor, preprocessor: FreightPreprocessor, frame: pd.DataFrame
) -> np.ndarray:
    features = preprocessor.transform(frame)
    rate_per_mile = np.clip(model.predict(features), 0.01, None)
    predicted_rate = rate_per_mile * features["distance"].to_numpy()
    return np.round(np.clip(predicted_rate, 1.0, None), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate required freight-rate predictions")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--validation-data", type=Path, default=PROJECT_ROOT / "data" / "validation.csv")
    parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_ROOT / "data" / "validation-predictions-template.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "validation_predictions.csv"
    )
    parser.add_argument(
        "--december-data",
        type=Path,
        default=PROJECT_ROOT / "data" / "december-chart-inputs.csv",
    )
    args = parser.parse_args()

    model, preprocessor = load_model(args.artifact_dir)
    validation = pd.read_csv(args.validation_data)
    template = pd.read_csv(args.template)
    if template["load_id"].duplicated().any():
        raise ValueError("Prediction template contains duplicate load_id values")

    predicted = pd.DataFrame(
        {"load_id": validation["load_id"], "predicted_rate": predict_rates(model, preprocessor, validation)}
    )
    submission = template[["load_id"]].merge(predicted, on="load_id", how="left", validate="one_to_one")
    if submission["predicted_rate"].isna().any():
        raise ValueError("Validation data does not cover every template load_id")
    submission.to_csv(args.output, index=False)

    december = pd.read_csv(args.december_data)
    december["predicted_rate"] = predict_rates(model, preprocessor, december)
    december.to_csv(args.december_data, index=False)
    print(f"Wrote {len(submission):,} rows to {args.output}")
    print(f"Wrote {len(december):,} rows to {args.december_data}")


if __name__ == "__main__":
    main()
