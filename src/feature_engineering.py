"""Feature engineering module: derives model-ready features from processed data."""

import json
import logging
import os

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

NUMERIC_FEATURES = [
    "grid",
    "quali_position",
    "total_points",
    "wins",
    "podiums",
    "total_points_constructor",
    "wins_constructor",
    "reliability",
    "driver_rolling_position",
    "driver_rolling_points",
    "constructor_rolling_position",
    "constructor_rolling_points",
]

CATEGORICAL_FEATURES = [
    "driver_id_enc",
    "constructor_id_enc",
    "circuit_id_enc",
]

TARGET_CLF = "top3"
TARGET_REG = "target_position"


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-product features that capture compound effects."""
    df = df.copy()
    df["grid_x_quali"] = df["grid"] * df["quali_position"]
    rolling_pos = (
        df.get("driver_rolling_position", df.get("grid", 20))
        .replace(0, float("nan"))
        .fillna(20)
    )
    rolling_pts = df.get("driver_rolling_points", df.get("points", 0))
    df["driver_momentum"] = rolling_pts / rolling_pos
    cons_rolling_pos = (
        df.get("constructor_rolling_position", df.get("grid", 20))
        .replace(0, float("nan"))
        .fillna(20)
    )
    cons_rolling_pts = df.get("constructor_rolling_points", df.get("points", 0))
    df["constructor_momentum"] = cons_rolling_pts / cons_rolling_pos
    df["experience_score"] = df["wins"] * 3 + df["podiums"]
    df["team_strength"] = (
        df["total_points_constructor"] / (df["wins_constructor"] + 1)
    ) * df["reliability"]
    return df


def get_feature_names() -> list[str]:
    return (
        NUMERIC_FEATURES
        + [
            "grid_x_quali",
            "driver_momentum",
            "constructor_momentum",
            "experience_score",
            "team_strength",
        ]
        + CATEGORICAL_FEATURES
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering and return a DataFrame with feature columns + targets."""
    df = add_interaction_features(df)
    feature_cols = get_feature_names()
    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        logger.warning("Missing feature columns, filling with 0: %s", missing)
        for col in missing:
            df[col] = 0

    keep = available + [
        TARGET_CLF,
        TARGET_REG,
        "season",
        "round",
        "driver_id",
        "constructor_id",
        "circuit_id",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def featurize(processed_dir: str = PROCESSED_DIR) -> None:
    os.makedirs(processed_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        path = os.path.join(processed_dir, f"{split}.csv")
        df = pd.read_csv(path)
        df_feat = build_features(df)
        out_path = os.path.join(processed_dir, f"{split}_features.csv")
        df_feat.to_csv(out_path, index=False)
        logger.info(
            "Wrote %s (%d rows, %d cols)", out_path, len(df_feat), len(df_feat.columns)
        )

    feature_names = get_feature_names()
    with open(
        os.path.join(processed_dir, "feature_names.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(feature_names, fh, indent=2)
    logger.info("Feature engineering complete.")


if __name__ == "__main__":
    with open(
        os.path.join(os.path.dirname(__file__), "..", "params.yaml"), encoding="utf-8"
    ) as fh:
        _ = yaml.safe_load(fh)
    featurize()
