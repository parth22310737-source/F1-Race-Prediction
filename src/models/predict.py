"""
Load the trained model and produce race finish-position predictions.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class RacePredictor:
    """Wraps the trained sklearn pipeline for inference."""

    def __init__(self, model_path: str = "models/model.pkl"):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. Run the training pipeline first."
            )
        with open(model_path, "rb") as fh:
            artifact = pickle.load(fh)
        self._pipeline = artifact["pipeline"]
        self._feature_cols: list[str] = artifact["feature_cols"]
        logger.info("Model loaded from %s", model_path)

    @property
    def feature_cols(self) -> list[str]:
        return list(self._feature_cols)

    def predict(self, features: dict[str, Any] | pd.DataFrame) -> list[int]:
        """Return predicted finish positions for one or more entries.

        Parameters
        ----------
        features:
            Either a dict of feature values (single prediction) or a
            DataFrame with one row per driver.
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        # Fill missing feature columns with zeros
        for col in self._feature_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[self._feature_cols].values
        predictions = self._pipeline.predict(X)
        return [int(p) for p in predictions]

    def predict_proba(self, features: dict[str, Any] | pd.DataFrame) -> list[dict]:
        """Return class probabilities for finish positions 1-20."""
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        for col in self._feature_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[self._feature_cols].values
        proba_matrix = self._pipeline.predict_proba(X)
        classes = self._pipeline.classes_

        result = []
        for row in proba_matrix:
            result.append({int(cls): float(p) for cls, p in zip(classes, row)})
        return result
