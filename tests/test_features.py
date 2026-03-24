"""Tests for feature engineering."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.fetch_data import _generate_synthetic
from src.data.preprocess import preprocess
from src.features.build_features import build_features


@pytest.fixture()
def processed_csv(tmp_path: Path) -> str:
    raw = tmp_path / "races.csv"
    _generate_synthetic(n_races=30, n_drivers=10).to_csv(raw, index=False)
    processed = tmp_path / "processed.csv"
    preprocess(
        input_path=str(raw),
        output_path=str(processed),
        artifacts_dir=str(tmp_path / "models"),
        params_path="params.yaml",
    )
    return str(processed)


def test_build_features_creates_output(processed_csv: str, tmp_path: Path):
    output = str(tmp_path / "features.csv")
    df = build_features(processed_csv, output, params_path="params.yaml")
    assert Path(output).exists()
    assert len(df) > 0


def test_build_features_adds_columns(processed_csv: str, tmp_path: Path):
    output = str(tmp_path / "features.csv")
    df = build_features(processed_csv, output, params_path="params.yaml")
    assert "win_rate" in df.columns
    assert "podium_rate" in df.columns
    assert "dnf_flag" in df.columns


def test_win_rate_range(processed_csv: str, tmp_path: Path):
    output = str(tmp_path / "features.csv")
    df = build_features(processed_csv, output, params_path="params.yaml")
    assert df["win_rate"].between(0, 1).all()
    assert df["podium_rate"].between(0, 1).all()
