"""FastAPI application for F1 Race Prediction and Strategy Recommendation."""

import os
import sys

# Allow running directly: ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from predict import RaceEntry, RacePredictor  # noqa: E402
from strategy import StrategyRecommender  # noqa: E402

app = FastAPI(
    title="F1 Race Prediction API",
    description=(
        "Predict race outcomes and receive optimal pit-stop strategy "
        "recommendations using machine-learning models trained on historical F1 data."
    ),
    version="0.1.0",
)

_predictor: RacePredictor | None = None
_recommender: StrategyRecommender | None = None


def get_predictor() -> RacePredictor:
    global _predictor
    if _predictor is None:
        _predictor = RacePredictor()
    return _predictor


def get_recommender() -> StrategyRecommender:
    global _recommender
    if _recommender is None:
        _recommender = StrategyRecommender()
    return _recommender


# ── Request / Response schemas ──────────────────────────────────────────────


class PredictionRequest(BaseModel):
    driver_id_enc: int = Field(default=0, description="Encoded driver ID")
    constructor_id_enc: int = Field(default=0, description="Encoded constructor ID")
    circuit_id_enc: int = Field(default=0, description="Encoded circuit ID")
    grid: int = Field(default=10, ge=1, le=20, description="Starting grid position")
    quali_position: int = Field(
        default=10, ge=1, le=20, description="Qualifying position"
    )
    total_points: float = Field(
        default=0.0, ge=0, description="Driver's cumulative points from previous season"
    )
    wins: float = Field(default=0.0, ge=0, description="Previous season wins")
    podiums: float = Field(default=0.0, ge=0, description="Previous season podiums")
    total_points_constructor: float = Field(
        default=0.0, ge=0, description="Constructor cumulative points"
    )
    wins_constructor: float = Field(default=0.0, ge=0)
    reliability: float = Field(default=0.9, ge=0, le=1)
    driver_rolling_position: float = Field(default=10.0, ge=1, le=20)
    driver_rolling_points: float = Field(default=5.0, ge=0)
    constructor_rolling_position: float = Field(default=10.0, ge=1, le=20)
    constructor_rolling_points: float = Field(default=5.0, ge=0)


class PredictionResponse(BaseModel):
    top3_probability: float
    predicted_position: float
    is_top3: bool


class StrategyRequest(BaseModel):
    total_laps: int = Field(default=57, ge=1, description="Total race laps")
    current_position: int = Field(
        default=5, ge=1, le=20, description="Current running position"
    )
    safety_car_probability: float = Field(
        default=0.15, ge=0, le=1, description="Estimated probability of safety car"
    )
    current_lap: int = Field(default=1, ge=1, description="Current lap number")
    leader_pit_lap: int | None = Field(
        default=None, description="Expected pit lap of race leader (for undercut)"
    )


class StintPlanResponse(BaseModel):
    compound: str
    start_lap: int
    end_lap: int
    degradation_loss: float
    laps: int


class StrategyResponse(BaseModel):
    strategy_name: str
    total_pit_stops: int
    estimated_time_loss_seconds: float
    rationale: str
    stints: list[StintPlanResponse]


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", tags=["Health"])
def health_check():
    """Check API liveness."""
    return {"status": "ok", "version": app.version}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_race(request: PredictionRequest):
    """Predict race top-3 probability and expected finish position."""
    try:
        predictor = get_predictor()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {exc}. Run the DVC pipeline first.",
        ) from exc

    entry = RaceEntry(**request.model_dump())
    result = predictor.predict(entry)
    return PredictionResponse(
        top3_probability=result.top3_probability,
        predicted_position=result.predicted_position,
        is_top3=result.is_top3,
    )


@app.post("/strategy", response_model=list[StrategyResponse], tags=["Strategy"])
def recommend_strategy(request: StrategyRequest):
    """Return ranked list of pit-stop strategy recommendations."""
    recommender = get_recommender()
    strategies = recommender.recommend(
        total_laps=request.total_laps,
        current_position=request.current_position,
        safety_car_probability=request.safety_car_probability,
        current_lap=request.current_lap,
        leader_pit_lap=request.leader_pit_lap,
    )
    return [
        StrategyResponse(
            strategy_name=s.strategy_name,
            total_pit_stops=s.total_pit_stops,
            estimated_time_loss_seconds=s.estimated_time_loss_seconds,
            rationale=s.rationale,
            stints=[
                StintPlanResponse(
                    compound=st.compound,
                    start_lap=st.start_lap,
                    end_lap=st.end_lap,
                    degradation_loss=st.degradation_loss,
                    laps=st.laps,
                )
                for st in s.stints
            ],
        )
        for s in strategies
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
