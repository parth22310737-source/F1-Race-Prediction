"""
Fetch and generate F1 race data.

When the Ergast API is reachable the script downloads real data; otherwise it
falls back to generating a realistic synthetic dataset so the pipeline can
always run end-to-end (CI / offline environments).
"""
from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ERGAST_BASE = "https://ergast.com/api/f1"
SEASONS = list(range(2018, 2024))

CIRCUITS = [
    "monaco",
    "monza",
    "spa",
    "silverstone",
    "suzuka",
    "bahrain",
    "singapore",
    "australia",
    "brazil",
    "usa",
]
CIRCUIT_TYPES = {
    "monaco": "street",
    "singapore": "street",
    "monza": "high_speed",
    "spa": "mixed",
    "silverstone": "mixed",
    "suzuka": "mixed",
    "bahrain": "desert",
    "australia": "mixed",
    "brazil": "mixed",
    "usa": "mixed",
}
WEATHER_OPTIONS = ["dry", "wet", "mixed"]
CONSTRUCTORS = [
    "Mercedes",
    "Red Bull",
    "Ferrari",
    "McLaren",
    "Alpine",
    "AlphaTauri",
    "Aston Martin",
    "Williams",
    "Alfa Romeo",
    "Haas",
]


def _fetch_ergast(season: int) -> list[dict]:
    """Try to fetch race results from the Ergast REST API."""
    url = f"{ERGAST_BASE}/{season}/results.json?limit=500"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        races = resp.json()["MRData"]["RaceTable"]["Races"]
        records: list[dict] = []
        for race in races:
            circuit = race["Circuit"]["circuitId"]
            for res in race["Results"]:
                driver = res["Driver"]
                constructor = res["Constructor"]["name"]
                records.append(
                    {
                        "season": season,
                        "round": int(race["round"]),
                        "circuit": circuit,
                        "circuit_type": CIRCUIT_TYPES.get(circuit, "mixed"),
                        "weather": random.choice(WEATHER_OPTIONS),
                        "driver": f"{driver['givenName']} {driver['familyName']}",
                        "constructor": constructor,
                        "grid_position": int(res.get("grid", 10)),
                        "finish_position": int(res["position"]),
                        "laps_completed": int(res["laps"]),
                        "driver_age": random.randint(20, 38),
                        "driver_points": float(res.get("points", 0)),
                        "constructor_points": random.uniform(0, 500),
                        "qualifying_time_ms": random.randint(70000, 100000),
                        "fastest_lap_ms": random.randint(72000, 105000),
                    }
                )
        return records
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ergast API unavailable (%s); using synthetic data.", exc)
        return []


def _generate_synthetic(n_races: int = 400, n_drivers: int = 20) -> pd.DataFrame:
    """Generate a realistic synthetic F1 dataset."""
    rng = np.random.default_rng(42)
    records = []
    season = 2018
    for race_id in range(n_races):
        circuit = rng.choice(CIRCUITS)
        weather = rng.choice(WEATHER_OPTIONS, p=[0.7, 0.15, 0.15])
        grid_positions = rng.permutation(n_drivers) + 1  # 1-based
        constructor_idx = rng.integers(0, len(CONSTRUCTORS), size=n_drivers)
        # DNF mask — ~10% chance per driver
        dnf = rng.random(n_drivers) < 0.10
        finish_order = grid_positions.copy().astype(float)
        # Add noise to simulate overtakes
        noise = rng.normal(0, 3, size=n_drivers)
        finish_order = np.clip(finish_order + noise, 1, n_drivers)
        # Assign actual finish positions
        ranked = np.argsort(finish_order) + 1
        for i in range(n_drivers):
            laps = rng.integers(30, 58) if dnf[i] else rng.integers(55, 58)
            records.append(
                {
                    "season": season + race_id // 22,
                    "round": (race_id % 22) + 1,
                    "circuit": circuit,
                    "circuit_type": CIRCUIT_TYPES[circuit],
                    "weather": weather,
                    "driver": f"Driver_{i + 1:02d}",
                    "constructor": CONSTRUCTORS[int(constructor_idx[i])],
                    "grid_position": int(grid_positions[i]),
                    "finish_position": int(ranked[i]),
                    "laps_completed": int(laps),
                    "driver_age": int(rng.integers(20, 39)),
                    "driver_points": float(rng.uniform(0, 300)),
                    "constructor_points": float(rng.uniform(0, 600)),
                    "qualifying_time_ms": int(rng.integers(70000, 100000)),
                    "fastest_lap_ms": int(rng.integers(72000, 105000)),
                }
            )
    return pd.DataFrame(records)


def fetch_data(output_path: str, params_path: str = "params.yaml") -> pd.DataFrame:
    """Fetch or generate F1 data and save to *output_path*."""
    with open(params_path) as fh:
        params = yaml.safe_load(fh)

    records: list[dict] = []
    for season in SEASONS:
        records.extend(_fetch_ergast(season))

    if records:
        df = pd.DataFrame(records)
        logger.info("Fetched %d records from Ergast API.", len(df))
    else:
        df = _generate_synthetic()
        logger.info("Generated %d synthetic records.", len(df))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Raw data saved to %s", output_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch F1 race data")
    parser.add_argument("--output", default="data/raw/races.csv")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    fetch_data(args.output, args.params)
