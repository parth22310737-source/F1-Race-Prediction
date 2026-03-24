"""
Preprocess raw F1 race CSV into a clean, model-ready DataFrame.

Steps
-----
1. Drop rows with nulls in critical columns.
2. Encode categorical features.
3. Scale numeric features.
4. Persist the processed dataset and the fitted scalers/encoders.
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def preprocess(
    input_path: str,
    output_path: str,
    artifacts_dir: str = "models",
    params_path: str = "params.yaml",
) -> pd.DataFrame:
    """Load raw data, clean and encode it, then save the result."""
    with open(params_path) as fh:
        params = yaml.safe_load(fh)

    feature_cfg = params["features"]
    numeric_features: list[str] = feature_cfg["numeric_features"]
    categorical_features: list[str] = feature_cfg["categorical_features"]
    target_col: str = feature_cfg["target_column"]

    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows from %s", len(df), input_path)

    required = numeric_features + categorical_features + [target_col]
    df = df.dropna(subset=required)
    logger.info("After dropping NaNs: %d rows", len(df))

    # --- encode categoricals ---
    encoders: dict[str, LabelEncoder] = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info("Encoded '%s': %d classes", col, len(le.classes_))

    # --- scale numerics ---
    scaler = MinMaxScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Clip finish position to 1-20 (DNS/DNF assigned to 20)
    df[target_col] = np.clip(df[target_col], 1, 20).astype(int)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Processed data saved to %s", output_path)

    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{artifacts_dir}/scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    with open(f"{artifacts_dir}/encoders.pkl", "wb") as fh:
        pickle.dump(encoders, fh)
    logger.info("Preprocessing artifacts saved to %s/", artifacts_dir)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess F1 race data")
    parser.add_argument("--input", default="data/raw/races.csv")
    parser.add_argument("--output", default="data/processed/races_processed.csv")
    parser.add_argument("--artifacts-dir", default="models")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    preprocess(args.input, args.output, args.artifacts_dir, args.params)
