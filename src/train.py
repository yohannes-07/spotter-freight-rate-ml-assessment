"""Train and persist the selected CatBoost model."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from src.modeling import (
    FINAL_CONFIG,
    FINAL_ITERATIONS,
    chronological_split,
    evaluate_final_strategy,
    fit_final_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the freight-rate model")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "train-test.csv")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--holdout-days", type=int, default=61)
    parser.add_argument("--iterations", type=int, default=FINAL_ITERATIONS)
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    required = {"date", "posted_rate"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")
    frame["posted_rate"] = pd.to_numeric(frame["posted_rate"], errors="raise")

    holdout_metrics, holdout_best_iteration = evaluate_final_strategy(
        frame,
        holdout_days=args.holdout_days,
        verbose=200,
    )
    print("\nChronological holdout")
    print(pd.Series(holdout_metrics).to_string(float_format=lambda value: f"{value:.3f}"))
    print(f"Holdout best iteration: {holdout_best_iteration}")
    print(f"Final fit: {FINAL_CONFIG['name']} at {args.iterations} iterations")

    model, preprocessor = fit_final_model(frame, FINAL_CONFIG, args.iterations)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(args.artifact_dir / "freight_rate_model.cbm")
    with (args.artifact_dir / "preprocessor.pkl").open("wb") as stream:
        pickle.dump(preprocessor, stream)
    pd.DataFrame(
        [{"strategy": FINAL_CONFIG["name"], **holdout_metrics, "best_iteration": holdout_best_iteration}]
    ).to_csv(args.artifact_dir / "experiment_results.csv", index=False)

    importance = pd.DataFrame(
        {
            "feature": preprocessor.feature_columns_,
            "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(args.artifact_dir / "feature_importance.csv", index=False)

    development, holdout = chronological_split(frame, args.holdout_days)
    metadata = {
        "model": "CatBoostRegressor",
        "target": "rate_per_mile",
        "training_weight": "distance",
        "selected_config": FINAL_CONFIG,
        "final_iterations": args.iterations,
        "holdout_best_iteration": holdout_best_iteration,
        "holdout_metrics": holdout_metrics,
        "development_rows": int(len(development)),
        "holdout_rows": int(len(holdout)),
        "development_date_range": [str(development["date"].min()), str(development["date"].max())],
        "holdout_date_range": [str(holdout["date"].min()), str(holdout["date"].max())],
        "final_training_rows": int(len(frame)),
        "features": preprocessor.feature_columns_,
    }
    (args.artifact_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved model and evaluation artifacts to {args.artifact_dir}")


if __name__ == "__main__":
    main()
