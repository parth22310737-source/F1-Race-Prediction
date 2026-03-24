"""
FastAPI application exposing F1 race prediction and strategy endpoints.

Routes
------
GET  /health                    – liveness probe
POST /predict                   – predict finish positions for a list of drivers
POST /strategy                  – recommend pit-stop strategy for a race
GET  /metrics                   – last training metrics
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import RacePredictor
from src.strategy.recommend import recommend_strategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="F1 Race Prediction API",
    description=(
        "Predict F1 race finish positions and recommend optimal pit-stop "
        "strategies using machine-learning models."
    ),
    version="1.0.0",
)

_predictor: RacePredictor | None = None

MODEL_PATH = "models/model.pkl"
METRICS_PATH = "models/metrics.json"


def get_predictor() -> RacePredictor:
    global _predictor  # noqa: PLW0603
    if _predictor is None:
        _predictor = RacePredictor(MODEL_PATH)
    return _predictor


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class DriverFeatures(BaseModel):
    driver: str = Field(..., json_schema_extra={"example": "Max Verstappen"})
    grid_position: float = Field(..., ge=1, le=20)
    driver_age: float = Field(..., ge=16, le=60)
    constructor_points: float = Field(..., ge=0)
    driver_points: float = Field(..., ge=0)
    laps_completed: float = Field(..., ge=0)
    qualifying_time_ms: float = Field(..., ge=0)
    circuit_type: float = Field(default=0)
    weather: float = Field(default=0)
    win_rate: float = Field(default=0.0, ge=0, le=1)
    podium_rate: float = Field(default=0.0, ge=0, le=1)
    dnf_flag: int = Field(default=0)


class PredictRequest(BaseModel):
    drivers: list[DriverFeatures]


class PredictResponse(BaseModel):
    predictions: list[dict[str, Any]]


class StrategyRequest(BaseModel):
    total_laps: int = Field(default=57, ge=1, le=100)
    grid_position: int = Field(default=1, ge=1, le=20)
    weather: str = Field(default="dry", pattern="^(dry|wet|mixed)$")
    circuit_type: str = Field(default="mixed")


class StrategyResponse(BaseModel):
    strategy: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest) -> PredictResponse:
    """Predict finish positions for a list of drivers."""
    if not request.drivers:
        raise HTTPException(status_code=422, detail="drivers list must not be empty")
    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    import pandas as pd

    df = pd.DataFrame([d.model_dump() for d in request.drivers])
    positions = predictor.predict(df)

    return PredictResponse(
        predictions=[
            {"driver": d.driver, "predicted_finish_position": pos}
            for d, pos in zip(request.drivers, positions)
        ]
    )


@app.post("/strategy", response_model=StrategyResponse, tags=["Strategy"])
def strategy(request: StrategyRequest) -> StrategyResponse:
    """Recommend an optimal pit-stop strategy."""
    result = recommend_strategy(
        total_laps=request.total_laps,
        grid_position=request.grid_position,
        weather=request.weather,
        circuit_type=request.circuit_type,
    )
    return StrategyResponse(strategy=result)


@app.get("/metrics", tags=["Metrics"])
def metrics() -> dict[str, Any]:
    """Return last model training metrics."""
    if not Path(METRICS_PATH).exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    with open(METRICS_PATH) as fh:
        return json.load(fh)
