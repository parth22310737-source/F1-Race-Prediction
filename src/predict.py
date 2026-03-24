"""Prediction module: loads trained models and makes race predictions."""

import logging
import os
from dataclasses import dataclass, field

import joblib
import numpy as np

from feature_engineering import get_feature_names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@dataclass
class RaceEntry:
    """Input data for a single driver/race entry."""

    driver_id_enc: int = 0
    constructor_id_enc: int = 0
    circuit_id_enc: int = 0
    grid: int = 10
    quali_position: int = 10
    total_points: float = 0.0
    wins: float = 0.0
    podiums: float = 0.0
    total_points_constructor: float = 0.0
    wins_constructor: float = 0.0
    reliability: float = 0.9
    driver_rolling_position: float = 10.0
    driver_rolling_points: float = 5.0
    constructor_rolling_position: float = 10.0
    constructor_rolling_points: float = 5.0
    grid_x_quali: float = field(init=False)
    driver_momentum: float = field(init=False)
    constructor_momentum: float = field(init=False)
    experience_score: float = field(init=False)
    team_strength: float = field(init=False)

    def __post_init__(self):
        self.grid_x_quali = self.grid * self.quali_position
        self.driver_momentum = self.driver_rolling_points / max(
            self.driver_rolling_position, 1
        )
        self.constructor_momentum = self.constructor_rolling_points / max(
            self.constructor_rolling_position, 1
        )
        self.experience_score = self.wins * 3 + self.podiums
        self.team_strength = (
            self.total_points_constructor / (self.wins_constructor + 1)
        ) * self.reliability

    def to_array(self) -> np.ndarray:
        feature_names = get_feature_names()
        return np.array([[getattr(self, f, 0.0) for f in feature_names]])


@dataclass
class PredictionResult:
    top3_probability: float
    predicted_position: float
    is_top3: bool


class RacePredictor:
    """Wrapper around the trained models for inference."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.clf = joblib.load(os.path.join(models_dir, "race_predictor.pkl"))
        self.reg = joblib.load(os.path.join(models_dir, "strategy_recommender.pkl"))
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

    def predict(self, entry: RaceEntry) -> PredictionResult:
        X = entry.to_array()
        X_scaled = self.scaler.transform(X)
        top3_prob = float(self.clf.predict_proba(X_scaled)[0][1])
        predicted_pos = float(self.reg.predict(X_scaled)[0])
        return PredictionResult(
            top3_probability=round(top3_prob, 4),
            predicted_position=round(predicted_pos, 2),
            is_top3=top3_prob >= 0.5,
        )

    def predict_batch(self, entries: list[RaceEntry]) -> list[PredictionResult]:
        return [self.predict(e) for e in entries]
