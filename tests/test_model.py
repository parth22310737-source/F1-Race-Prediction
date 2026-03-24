"""Tests for the ML training and prediction pipeline."""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from src.data.fetch_data import _generate_synthetic
from src.data.preprocess import preprocess
from src.features.build_features import build_features
from src.models.train import train


@pytest.fixture()
def trained_model(tmp_path: Path) -> tuple[str, str]:
    """Run the full pipeline and return (model_dir, features_csv)."""
    raw = tmp_path / "races.csv"
    _generate_synthetic(n_races=50, n_drivers=10).to_csv(raw, index=False)

    processed = tmp_path / "processed.csv"
    preprocess(
        input_path=str(raw),
        output_path=str(processed),
        artifacts_dir=str(tmp_path / "models"),
        params_path="params.yaml",
    )

    features = tmp_path / "features.csv"
    build_features(str(processed), str(features), params_path="params.yaml")

    model_dir = str(tmp_path / "models")
    train(input_path=str(features), model_dir=model_dir, params_path="params.yaml")
    return model_dir, str(features)


def test_model_file_created(trained_model):
    model_dir, _ = trained_model
    assert Path(model_dir, "model.pkl").exists()


def test_metrics_file_created(trained_model):
    model_dir, _ = trained_model
    assert Path(model_dir, "metrics.json").exists()


def test_metrics_keys(trained_model):
    import json

    model_dir, _ = trained_model
    with open(Path(model_dir, "metrics.json")) as fh:
        metrics = json.load(fh)
    assert "accuracy" in metrics
    assert "mae" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_predictor_single(trained_model):
    from src.models.predict import RacePredictor

    model_dir, _ = trained_model
    predictor = RacePredictor(f"{model_dir}/model.pkl")
    features = {
        "grid_position": 0.05,
        "driver_age": 0.3,
        "constructor_points": 0.8,
        "driver_points": 0.7,
        "laps_completed": 0.95,
        "qualifying_time_ms": 0.1,
        "circuit_type": 1,
        "weather": 0,
        "win_rate": 0.3,
        "podium_rate": 0.6,
        "dnf_flag": 0,
    }
    result = predictor.predict(features)
    assert len(result) == 1
    assert 1 <= result[0] <= 20


def test_predictor_batch(trained_model):
    import pandas as pd
    from src.models.predict import RacePredictor

    model_dir, _ = trained_model
    predictor = RacePredictor(f"{model_dir}/model.pkl")
    df = pd.DataFrame(
        [
            {
                "grid_position": 0.05,
                "driver_age": 0.3,
                "constructor_points": 0.8,
                "driver_points": 0.7,
                "laps_completed": 0.95,
                "qualifying_time_ms": 0.1,
            },
            {
                "grid_position": 0.9,
                "driver_age": 0.5,
                "constructor_points": 0.2,
                "driver_points": 0.1,
                "laps_completed": 0.5,
                "qualifying_time_ms": 0.9,
            },
        ]
    )
    result = predictor.predict(df)
    assert len(result) == 2
