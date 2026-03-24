"""Preprocessing module: cleans and splits race data for model training."""

import json
import logging
import os

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_raw(raw_dir: str = RAW_DIR) -> tuple[pd.DataFrame, ...]:
    """Load raw CSV files and return (races, qualifying, drivers, constructors)."""
    races = pd.read_csv(os.path.join(raw_dir, "races.csv"))
    qualifying = pd.read_csv(os.path.join(raw_dir, "qualifying.csv"))
    drivers = pd.read_csv(os.path.join(raw_dir, "drivers.csv"))
    constructors = pd.read_csv(os.path.join(raw_dir, "constructors.csv"))
    return races, qualifying, drivers, constructors


def clean_races(races: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing position and coerce types."""
    df = races.copy()
    df = df.dropna(subset=["position"])
    df["position"] = df["position"].astype(int)
    df["grid"] = df["grid"].fillna(20).astype(int)
    df["points"] = df["points"].fillna(0).astype(float)
    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)
    return df


def encode_categorical(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Label-encode *cols* and return the DataFrame with mapping dict."""
    df = df.copy()
    mappings: dict[str, dict] = {}
    for col in cols:
        categories = sorted(df[col].dropna().unique())
        mapping = {v: i for i, v in enumerate(categories)}
        df[col + "_enc"] = df[col].map(mapping).fillna(-1).astype(int)
        mappings[col] = mapping
    return df, mappings


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling average position and points per driver/constructor."""
    df = df.sort_values(["season", "round"]).copy()

    for col, key in [
        ("position", "driver"),
        ("points", "driver"),
        ("position", "constructor"),
        ("points", "constructor"),
    ]:
        group_col = "driver_id" if key == "driver" else "constructor_id"
        df[f"{key}_rolling_{col}"] = (
            df.groupby(group_col)[col]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            .fillna(df[col].median())
        )
    return df


def build_dataset(
    races: pd.DataFrame,
    qualifying: pd.DataFrame,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all raw data into a single modelling DataFrame."""
    races = clean_races(races)

    # Merge qualifying grid position
    qual_cols = ["season", "round", "driver_id", "quali_position"]
    df = races.merge(
        qualifying[qual_cols], on=["season", "round", "driver_id"], how="left"
    )
    df["quali_position"] = df["quali_position"].fillna(df["grid"])

    # Merge cumulative driver stats (previous season)
    driver_prev = drivers.copy()
    driver_prev["season"] = driver_prev["season"] + 1
    df = df.merge(
        driver_prev[["driver_id", "season", "total_points", "wins", "podiums"]],
        on=["driver_id", "season"],
        how="left",
        suffixes=("", "_driver_prev"),
    )
    df["total_points"] = df["total_points"].fillna(0)
    df["wins"] = df["wins"].fillna(0)
    df["podiums"] = df["podiums"].fillna(0)

    # Merge constructor stats (previous season)
    cons_prev = constructors.copy()
    cons_prev["season"] = cons_prev["season"] + 1
    df = df.merge(
        cons_prev[["constructor_id", "season", "total_points", "wins", "reliability"]],
        on=["constructor_id", "season"],
        how="left",
        suffixes=("", "_constructor"),
    )
    df["total_points_constructor"] = df["total_points_constructor"].fillna(0)
    df["wins_constructor"] = df["wins_constructor"].fillna(0)
    df["reliability"] = df["reliability"].fillna(0.9)

    # Rolling in-season stats
    df = add_rolling_features(df)

    # Binary target: did the driver finish in top 3?
    df["top3"] = (df["position"] <= 3).astype(int)
    # Exact position target for regression
    df["target_position"] = df["position"]

    return df


def split(
    df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split by season."""
    seasons = sorted(df["season"].unique())
    n = len(seasons)
    n_test = max(1, int(np.ceil(n * test_size)))
    n_val = max(1, int(np.ceil(n * val_size)))
    n_train = n - n_test - n_val

    train_seasons = seasons[:n_train]
    val_seasons = seasons[n_train : n_train + n_val]
    test_seasons = seasons[n_train + n_val :]

    train = df[df["season"].isin(train_seasons)].copy()
    val = df[df["season"].isin(val_seasons)].copy()
    test = df[df["season"].isin(test_seasons)].copy()

    logger.info(
        "Split: train=%d, val=%d, test=%d rows",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


def preprocess(
    raw_dir: str = RAW_DIR,
    processed_dir: str = PROCESSED_DIR,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> None:
    os.makedirs(processed_dir, exist_ok=True)
    races, qualifying, drivers, constructors = load_raw(raw_dir)
    df = build_dataset(races, qualifying, drivers, constructors)

    # Encode categoricals
    df, mappings = encode_categorical(df, ["driver_id", "constructor_id", "circuit_id"])
    mapping_path = os.path.join(processed_dir, "encodings.json")
    with open(mapping_path, "w", encoding="utf-8") as fh:
        json.dump(mappings, fh, indent=2)

    train, val, test = split(df, test_size, val_size)
    train.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    with open(
        os.path.join(os.path.dirname(__file__), "..", "params.yaml"), encoding="utf-8"
    ) as fh:
        params = yaml.safe_load(fh)
    preprocess(
        test_size=params["data"]["test_size"],
        val_size=params["data"]["val_size"],
    )
