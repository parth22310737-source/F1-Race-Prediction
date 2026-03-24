"""
Train a Random-Forest model to predict F1 race finish positions.

The model is treated as a multi-class classification problem where the
target is the discretised finish position (1–20).

Outputs
-------
models/model.pkl  – trained pipeline (scaler already baked in)
models/metrics.json  – evaluation metrics written for DVC to track
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "grid_position",
    "driver_age",
    "constructor_points",
    "driver_points",
    "laps_completed",
    "qualifying_time_ms",
    "circuit_type",
    "weather",
    "win_rate",
    "podium_rate",
    "dnf_flag",
]


def _available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


def train(
    input_path: str,
    model_dir: str = "models",
    params_path: str = "params.yaml",
) -> dict:
    """Train the model and return evaluation metrics."""
    with open(params_path) as fh:
        params = yaml.safe_load(fh)

    model_cfg = params["model"]
    data_cfg = params["data"]
    target_col = params["features"]["target_column"]

    df = pd.read_csv(input_path)
    feature_cols = _available_features(df)
    logger.info("Training with features: %s", feature_cols)

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=model_cfg["n_estimators"],
                    max_depth=model_cfg["max_depth"],
                    min_samples_split=model_cfg["min_samples_split"],
                    random_state=model_cfg["random_state"],
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
        "mae": float(round(mean_absolute_error(y_test, y_pred), 4)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "n_features": int(len(feature_cols)),
    }
    logger.info("Metrics: %s", metrics)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = f"{model_dir}/model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump({"pipeline": pipeline, "feature_cols": feature_cols}, fh)
    logger.info("Model saved to %s", model_path)

    metrics_path = f"{model_dir}/metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train F1 race prediction model")
    parser.add_argument("--input", default="data/processed/races_features.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    train(args.input, args.model_dir, args.params)
