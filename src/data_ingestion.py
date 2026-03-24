"""Data ingestion module: fetches F1 race data from the Ergast API."""

import logging
import os
import time

import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ERGAST_BASE = "http://ergast.com/api/f1"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _get(url: str, retries: int = 3, delay: float = 1.0) -> dict:
    """Fetch JSON from *url* with simple retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Request failed (%s/%s): %s", attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def fetch_season_results(season: int) -> list[dict]:
    """Return all race results for *season*."""
    url = f"{ERGAST_BASE}/{season}/results.json?limit=1000"
    data = _get(url)
    return data["MRData"]["RaceTable"]["Races"]


def fetch_qualifying(season: int) -> list[dict]:
    """Return qualifying results for *season*."""
    url = f"{ERGAST_BASE}/{season}/qualifying.json?limit=1000"
    data = _get(url)
    return data["MRData"]["RaceTable"]["Races"]


def _parse_results(races: list[dict], season: int) -> pd.DataFrame:
    rows = []
    for race in races:
        circuit = race["Circuit"]["circuitId"]
        round_num = int(race["round"])
        race_name = race["raceName"]
        for res in race.get("Results", []):
            status = res.get("status", "")
            rows.append(
                {
                    "season": season,
                    "round": round_num,
                    "race_name": race_name,
                    "circuit_id": circuit,
                    "driver_id": res["Driver"]["driverId"],
                    "driver_code": res["Driver"].get("code", ""),
                    "constructor_id": res["Constructor"]["constructorId"],
                    "grid": int(res.get("grid", 0)),
                    "position": int(res["position"]) if res.get("position") else None,
                    "points": float(res.get("points", 0)),
                    "laps": int(res.get("laps", 0)),
                    "status": status,
                    "finished": status == "Finished" or status.startswith("+"),
                }
            )
    return pd.DataFrame(rows)


def _parse_qualifying(races: list[dict], season: int) -> pd.DataFrame:
    rows = []
    for race in races:
        round_num = int(race["round"])
        for qual in race.get("QualifyingResults", []):
            rows.append(
                {
                    "season": season,
                    "round": round_num,
                    "driver_id": qual["Driver"]["driverId"],
                    "constructor_id": qual["Constructor"]["constructorId"],
                    "quali_position": int(qual.get("position", 20)),
                    "q1": qual.get("Q1", ""),
                    "q2": qual.get("Q2", ""),
                    "q3": qual.get("Q3", ""),
                }
            )
    return pd.DataFrame(rows)


def ingest(seasons: list[int], output_dir: str = RAW_DIR) -> None:
    """Fetch race results and qualifying for the given *seasons*."""
    os.makedirs(output_dir, exist_ok=True)

    all_results, all_qualifying = [], []

    for season in seasons:
        logger.info("Fetching season %d …", season)
        results = fetch_season_results(season)
        qualifying = fetch_qualifying(season)
        all_results.extend(results)
        all_qualifying.extend(qualifying)
        time.sleep(0.5)

    # Parse per season (season embedded in each race dict from Ergast)
    dfs = []
    for season in seasons:
        logger.info("Parsing season %d …", season)
        season_races = [r for r in all_results if r.get("season") == str(season)]
        if season_races:
            dfs.append(_parse_results(season_races, season))
    if dfs:
        df_results = pd.concat(dfs, ignore_index=True)
    else:
        # Fall back to bulk parse (season embedded in Ergast response)
        df_results = _parse_results_multi(all_results)

    dfs_q = []
    for season in seasons:
        season_races = [r for r in all_qualifying if r.get("season") == str(season)]
        if season_races:
            dfs_q.append(_parse_qualifying(season_races, season))
    if dfs_q:
        df_qual = pd.concat(dfs_q, ignore_index=True)
    else:
        df_qual = _parse_qualifying_multi(all_qualifying)

    df_results.to_csv(os.path.join(output_dir, "races.csv"), index=False)
    df_qual.to_csv(os.path.join(output_dir, "qualifying.csv"), index=False)

    _build_driver_standings(df_results).to_csv(
        os.path.join(output_dir, "drivers.csv"), index=False
    )
    _build_constructor_standings(df_results).to_csv(
        os.path.join(output_dir, "constructors.csv"), index=False
    )
    logger.info("Ingestion complete. Files written to %s", output_dir)


def _parse_results_multi(races: list[dict]) -> pd.DataFrame:
    """Parse results where season is embedded in each race dict."""
    rows = []
    for race in races:
        season = int(race.get("season", 0))
        circuit = race["Circuit"]["circuitId"]
        round_num = int(race["round"])
        race_name = race["raceName"]
        for res in race.get("Results", []):
            status = res.get("status", "")
            rows.append(
                {
                    "season": season,
                    "round": round_num,
                    "race_name": race_name,
                    "circuit_id": circuit,
                    "driver_id": res["Driver"]["driverId"],
                    "driver_code": res["Driver"].get("code", ""),
                    "constructor_id": res["Constructor"]["constructorId"],
                    "grid": int(res.get("grid", 0)),
                    "position": int(res["position"]) if res.get("position") else None,
                    "points": float(res.get("points", 0)),
                    "laps": int(res.get("laps", 0)),
                    "status": res.get("status", ""),
                    "finished": status == "Finished" or status.startswith("+"),
                }
            )
    return pd.DataFrame(rows)


def _parse_qualifying_multi(races: list[dict]) -> pd.DataFrame:
    rows = []
    for race in races:
        season = int(race.get("season", 0))
        round_num = int(race["round"])
        for qual in race.get("QualifyingResults", []):
            rows.append(
                {
                    "season": season,
                    "round": round_num,
                    "driver_id": qual["Driver"]["driverId"],
                    "constructor_id": qual["Constructor"]["constructorId"],
                    "quali_position": int(qual.get("position", 20)),
                    "q1": qual.get("Q1", ""),
                    "q2": qual.get("Q2", ""),
                    "q3": qual.get("Q3", ""),
                }
            )
    return pd.DataFrame(rows)


def _build_driver_standings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["driver_id", "season", "total_points", "wins", "podiums"]
        )
    agg = (
        df.groupby(["driver_id", "season"])
        .agg(
            total_points=("points", "sum"),
            wins=("position", lambda x: (x == 1).sum()),
            podiums=("position", lambda x: (x <= 3).sum()),
        )
        .reset_index()
    )
    return agg


def _build_constructor_standings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "constructor_id",
                "season",
                "total_points",
                "wins",
                "reliability",
            ]
        )
    agg = (
        df.groupby(["constructor_id", "season"])
        .agg(
            total_points=("points", "sum"),
            wins=("position", lambda x: (x == 1).sum()),
            reliability=("finished", "mean"),
        )
        .reset_index()
    )
    return agg


if __name__ == "__main__":
    with open(
        os.path.join(os.path.dirname(__file__), "..", "params.yaml"), encoding="utf-8"
    ) as fh:
        params = yaml.safe_load(fh)
    ingest(params["data"]["seasons"])
