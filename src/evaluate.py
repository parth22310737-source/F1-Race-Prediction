"""Evaluation module: computes and persists model performance metrics."""

import json
import logging
import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

from feature_engineering import get_feature_names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "metrics")


def evaluate(
    processed_dir: str = PROCESSED_DIR,
    models_dir: str = MODELS_DIR,
    metrics_dir: str = METRICS_DIR,
) -> dict:
    os.makedirs(metrics_dir, exist_ok=True)

    clf = joblib.load(os.path.join(models_dir, "race_predictor.pkl"))
    reg = joblib.load(os.path.join(models_dir, "strategy_recommender.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

    feature_names = get_feature_names()
    test_df = pd.read_csv(os.path.join(processed_dir, "test_features.csv"))

    available = [f for f in feature_names if f in test_df.columns]
    X_test = test_df[available].fillna(0).values
    X_scaled = scaler.transform(X_test)

    y_clf = test_df["top3"].values
    y_reg = test_df["target_position"].values

    # Classification metrics
    y_pred_clf = clf.predict(X_scaled)
    acc = float(accuracy_score(y_clf, y_pred_clf))
    prec = float(precision_score(y_clf, y_pred_clf, zero_division=0))
    rec = float(recall_score(y_clf, y_pred_clf, zero_division=0))
    f1 = float(f1_score(y_clf, y_pred_clf, zero_division=0))

    # Regression metrics
    y_pred_reg = reg.predict(X_scaled)
    mae = float(mean_absolute_error(y_reg, y_pred_reg))

    metrics = {
        "classifier": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        },
        "regressor": {
            "mae_position": round(mae, 4),
        },
        "test_samples": int(len(test_df)),
    }

    out_path = os.path.join(metrics_dir, "evaluation.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("Evaluation metrics: %s", json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
