"""Tests for the preprocessing module."""

import pandas as pd

from preprocessing import (
    add_rolling_features,
    build_dataset,
    clean_races,
    encode_categorical,
    split,
)


def _make_races(n: int = 40) -> pd.DataFrame:
    """Return a minimal synthetic races DataFrame with exactly *n* rows."""
    half = n // 2
    seasons = [2021] * half + [2022] * (n - half)
    rounds = list(range(1, half + 1)) + list(range(1, n - half + 1))
    return pd.DataFrame(
        {
            "season": seasons,
            "round": rounds,
            "race_name": ["GP"] * n,
            "circuit_id": ["monaco", "silverstone"] * (n // 2) + ["monaco"] * (n % 2),
            "driver_id": ["hamilton", "verstappen"] * (n // 2) + ["hamilton"] * (n % 2),
            "driver_code": ["HAM", "VER"] * (n // 2) + ["HAM"] * (n % 2),
            "constructor_id": ["mercedes", "red_bull"] * (n // 2)
            + ["mercedes"] * (n % 2),
            "grid": list(range(1, n + 1)),
            "position": ([1, 2] * (n // 2 + 1))[:n],
            "points": ([25.0, 18.0] * (n // 2 + 1))[:n],
            "laps": [57] * n,
            "status": ["Finished"] * n,
            "finished": [True] * n,
        }
    )


def _make_qualifying(n: int = 40) -> pd.DataFrame:
    seasons = [2021] * 20 + [2022] * 20
    return pd.DataFrame(
        {
            "season": seasons,
            "round": list(range(1, 21)) * 2,
            "driver_id": ["hamilton", "verstappen"] * (n // 2),
            "constructor_id": ["mercedes", "red_bull"] * (n // 2),
            "quali_position": [1, 2] * (n // 2),
        }
    )


def _make_driver_standings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_id": ["hamilton", "verstappen"],
            "season": [2020, 2020],
            "total_points": [347.0, 214.0],
            "wins": [11, 3],
            "podiums": [17, 18],
        }
    )


def _make_constructor_standings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "constructor_id": ["mercedes", "red_bull"],
            "season": [2020, 2020],
            "total_points": [573.0, 319.0],
            "wins": [13, 2],
            "reliability": [0.95, 0.90],
        }
    )


class TestCleanRaces:
    def test_drops_nan_position(self):
        df = _make_races(10)
        df.loc[0, "position"] = None
        cleaned = clean_races(df)
        assert len(cleaned) == 9

    def test_grid_filled_with_20(self):
        df = _make_races(5)
        df.loc[0, "grid"] = None
        cleaned = clean_races(df)
        assert cleaned.loc[cleaned.index[0], "grid"] == 20

    def test_position_is_int(self):
        cleaned = clean_races(_make_races(10))
        assert cleaned["position"].dtype == int


class TestEncodeCategorical:
    def test_encoding_creates_new_column(self):
        df = _make_races(10)
        df_enc, mappings = encode_categorical(df, ["driver_id"])
        assert "driver_id_enc" in df_enc.columns
        assert "driver_id" in mappings

    def test_mapping_is_contiguous(self):
        df = _make_races(10)
        _, mappings = encode_categorical(df, ["driver_id"])
        values = sorted(mappings["driver_id"].values())
        assert values == list(range(len(values)))


class TestSplit:
    def test_split_sizes(self):
        df = pd.DataFrame({"season": [2018, 2019, 2020, 2021, 2022, 2023] * 5})
        train, val, test = split(df, test_size=0.2, val_size=0.1)
        assert len(train) + len(val) + len(test) == len(df)
        assert len(test) > 0
        assert len(val) > 0

    def test_no_data_leakage(self):
        df = pd.DataFrame({"season": [2018, 2019, 2020, 2021, 2022, 2023] * 5})
        train, val, test = split(df, test_size=0.2, val_size=0.1)
        assert set(train["season"]).isdisjoint(set(test["season"]))
        assert set(val["season"]).isdisjoint(set(test["season"]))


class TestAddRollingFeatures:
    def test_creates_rolling_columns(self):
        df = _make_races(20)
        df = add_rolling_features(df)
        for col in [
            "driver_rolling_position",
            "driver_rolling_points",
            "constructor_rolling_position",
            "constructor_rolling_points",
        ]:
            assert col in df.columns

    def test_no_nans(self):
        df = _make_races(20)
        df = add_rolling_features(df)
        for col in ["driver_rolling_position", "driver_rolling_points"]:
            assert df[col].isna().sum() == 0


class TestBuildDataset:
    def test_top3_column_created(self):
        races = _make_races(40)
        qualifying = _make_qualifying(40)
        drivers = _make_driver_standings()
        constructors = _make_constructor_standings()
        df = build_dataset(races, qualifying, drivers, constructors)
        assert "top3" in df.columns

    def test_target_position_column(self):
        races = _make_races(40)
        qualifying = _make_qualifying(40)
        drivers = _make_driver_standings()
        constructors = _make_constructor_standings()
        df = build_dataset(races, qualifying, drivers, constructors)
        assert "target_position" in df.columns
        assert df["target_position"].notna().all()
