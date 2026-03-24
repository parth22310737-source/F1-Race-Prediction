"""Model training module: trains a race position predictor and strategy recommender."""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_features(processed_dir: str, split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(processed_dir, f"{split}_features.csv"))


def get_xy(df: pd.DataFrame, feature_names: list[str], target: str):
    available = [f for f in feature_names if f in df.columns]
    X = df[available].fillna(0).values
    y = df[target].values
    return X, y


def train_race_predictor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
) -> GradientBoostingClassifier:
    """Train a gradient boosting classifier for top-3 prediction."""
    model = GradientBoostingClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        random_state=params.get("random_state", 42),
    )
    model.fit(X_train, y_train)
    return model


def train_strategy_recommender(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
) -> RandomForestRegressor:
    """Train a random-forest regressor to predict final race position."""
    model = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 6),
        min_samples_leaf=params.get("min_child_weight", 3),
        random_state=params.get("random_state", 42),
    )
    model.fit(X_train, y_train)
    return model


def train(
    processed_dir: str = PROCESSED_DIR,
    models_dir: str = MODELS_DIR,
    params: dict | None = None,
) -> None:
    os.makedirs(models_dir, exist_ok=True)

    with open(
        os.path.join(processed_dir, "feature_names.json"), encoding="utf-8"
    ) as fh:
        feature_names = json.load(fh)

    train_df = load_features(processed_dir, "train")
    val_df = load_features(processed_dir, "val")

    # Combine train + val for final model
    combined = pd.concat([train_df, val_df], ignore_index=True)

    if params is None:
        params = {}

    X_train, y_clf = get_xy(combined, feature_names, "top3")
    _, y_reg = get_xy(combined, feature_names, "target_position")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    logger.info("Training race predictor (classifier)…")
    clf = train_race_predictor(X_scaled, y_clf, params)

    logger.info("Training strategy recommender (regressor)…")
    reg = train_strategy_recommender(X_scaled, y_reg, params)

    joblib.dump(clf, os.path.join(models_dir, "race_predictor.pkl"))
    joblib.dump(reg, os.path.join(models_dir, "strategy_recommender.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    logger.info("Models saved to %s", models_dir)


if __name__ == "__main__":
    with open(
        os.path.join(os.path.dirname(__file__), "..", "params.yaml"), encoding="utf-8"
    ) as fh:
        all_params = yaml.safe_load(fh)
    train(params=all_params["model"])
