"""
Build derived features for the F1 race prediction model.

Extra features created
-----------------------
- ``win_rate``    : fraction of races won by this driver in the dataset.
- ``podium_rate`` : fraction of races where the driver finished in the top 3.
- ``dnf_flag``    : binary indicator (1) when laps_completed is below 85 % of
                    the maximum, approximating a Did-Not-Finish event.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_features(
    input_path: str,
    output_path: str,
    params_path: str = "params.yaml",
) -> pd.DataFrame:
    """Read processed CSV, engineer new features, write enriched CSV."""
    with open(params_path) as fh:
        params = yaml.safe_load(fh)

    df = pd.read_csv(input_path)
    logger.info("Building features on %d rows", len(df))

    target_col = params["features"]["target_column"]

    if "driver" in df.columns:
        driver_stats = (
            df.groupby("driver")
            .agg(
                win_rate=(target_col, lambda x: (x == 1).mean()),
                podium_rate=(target_col, lambda x: (x <= 3).mean()),
            )
            .reset_index()
        )
        df = df.merge(driver_stats, on="driver", how="left")
    else:
        df["win_rate"] = 0.0
        df["podium_rate"] = 0.0

    if "laps_completed" in df.columns:
        # laps_completed is already scaled; treat < 0.85 of max as likely DNF
        max_laps = df["laps_completed"].max()
        df["dnf_flag"] = (df["laps_completed"] < 0.85 * max_laps).astype(int)
    else:
        df["dnf_flag"] = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Feature-engineered data saved to %s", output_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build F1 features")
    parser.add_argument("--input", default="data/processed/races_processed.csv")
    parser.add_argument("--output", default="data/processed/races_features.csv")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    build_features(args.input, args.output, args.params)
