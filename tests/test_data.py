"""Tests for the data pipeline (fetch & preprocess)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.fetch_data import _generate_synthetic
from src.data.preprocess import preprocess


@pytest.fixture()
def synthetic_raw_csv(tmp_path: Path) -> str:
    df = _generate_synthetic(n_races=20, n_drivers=10)
    csv_path = tmp_path / "races.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_synthetic_data_shape():
    df = _generate_synthetic(n_races=10, n_drivers=5)
    assert len(df) == 50, "Expected 10 races × 5 drivers"
    expected_cols = {
        "season",
        "circuit",
        "grid_position",
        "finish_position",
        "laps_completed",
    }
    assert expected_cols.issubset(df.columns)


def test_synthetic_finish_positions_valid():
    df = _generate_synthetic(n_races=5, n_drivers=20)
    assert df["finish_position"].between(1, 20).all()


def test_preprocess_creates_output(synthetic_raw_csv: str, tmp_path: Path):
    output_path = str(tmp_path / "processed.csv")
    df = preprocess(
        input_path=synthetic_raw_csv,
        output_path=output_path,
        artifacts_dir=str(tmp_path / "models"),
        params_path="params.yaml",
    )
    assert Path(output_path).exists()
    assert len(df) > 0


def test_preprocess_no_nulls(synthetic_raw_csv: str, tmp_path: Path):
    output_path = str(tmp_path / "processed.csv")
    df = preprocess(
        input_path=synthetic_raw_csv,
        output_path=output_path,
        artifacts_dir=str(tmp_path / "models"),
        params_path="params.yaml",
    )
    required_cols = ["grid_position", "finish_position"]
    assert df[required_cols].isnull().sum().sum() == 0


def test_preprocess_saves_artifacts(synthetic_raw_csv: str, tmp_path: Path):
    artifacts_dir = tmp_path / "models"
    preprocess(
        input_path=synthetic_raw_csv,
        output_path=str(tmp_path / "processed.csv"),
        artifacts_dir=str(artifacts_dir),
        params_path="params.yaml",
    )
    assert (artifacts_dir / "scaler.pkl").exists()
    assert (artifacts_dir / "encoders.pkl").exists()
