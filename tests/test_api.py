"""Tests for the FastAPI prediction service."""
from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---- helpers ---------------------------------------------------------------

def _make_mock_predictor(positions=(3,)):
    mock = MagicMock()
    mock.predict.return_value = list(positions)
    mock.feature_cols = [
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
    return mock


# ---- fixtures --------------------------------------------------------------

@pytest.fixture()
def client():
    import api.app as app_module

    # Reset singleton
    app_module._predictor = _make_mock_predictor()
    from api.app import app
    return TestClient(app)


# ---- tests -----------------------------------------------------------------

def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_single_driver(client: TestClient):
    payload = {
        "drivers": [
            {
                "driver": "Max Verstappen",
                "grid_position": 1,
                "driver_age": 26,
                "constructor_points": 454.0,
                "driver_points": 454.0,
                "laps_completed": 57.0,
                "qualifying_time_ms": 80123.0,
            }
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert data["predictions"][0]["driver"] == "Max Verstappen"
    assert "predicted_finish_position" in data["predictions"][0]


def test_predict_empty_drivers(client: TestClient):
    resp = client.post("/predict", json={"drivers": []})
    assert resp.status_code == 422


def test_strategy_dry(client: TestClient):
    resp = client.post("/strategy", json={"total_laps": 57, "weather": "dry"})
    assert resp.status_code == 200
    data = resp.json()
    assert "strategy" in data
    assert "stints" in data["strategy"]


def test_strategy_wet(client: TestClient):
    resp = client.post("/strategy", json={"total_laps": 57, "weather": "wet"})
    assert resp.status_code == 200


def test_strategy_invalid_weather(client: TestClient):
    resp = client.post("/strategy", json={"total_laps": 57, "weather": "snow"})
    assert resp.status_code == 422


def test_metrics_present(tmp_path, monkeypatch):
    """Metrics endpoint returns 200 when the metrics file exists."""
    import json as _json

    metrics_data = {"accuracy": 0.5, "mae": 3.2}
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(_json.dumps(metrics_data))

    import api.app as app_module
    monkeypatch.setattr(app_module, "METRICS_PATH", str(metrics_file))
    app_module._predictor = _make_mock_predictor()

    from api.app import app
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        resp = c.get("/metrics")
    assert resp.status_code == 200
    assert resp.json()["accuracy"] == 0.5


def test_metrics_missing_returns_404(tmp_path, monkeypatch):
    """Metrics endpoint returns 404 when the metrics file is absent."""
    import api.app as app_module
    monkeypatch.setattr(app_module, "METRICS_PATH", str(tmp_path / "no_metrics.json"))
    app_module._predictor = _make_mock_predictor()

    from api.app import app
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        resp = c.get("/metrics")
    assert resp.status_code == 404
