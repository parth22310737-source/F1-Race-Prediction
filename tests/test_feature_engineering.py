"""Tests for the feature engineering module."""

import pandas as pd

from feature_engineering import (
    add_interaction_features,
    build_features,
    get_feature_names,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def _make_processed_df(n: int = 20) -> pd.DataFrame:
    """Create a synthetic processed DataFrame with all required columns."""
    return pd.DataFrame(
        {
            "season": [2022] * n,
            "round": list(range(1, n + 1)),
            "driver_id": ["hamilton"] * n,
            "constructor_id": ["mercedes"] * n,
            "circuit_id": ["monaco"] * n,
            "driver_id_enc": [0] * n,
            "constructor_id_enc": [0] * n,
            "circuit_id_enc": [0] * n,
            "grid": list(range(1, n + 1)),
            "quali_position": list(range(1, n + 1)),
            "total_points": [300.0] * n,
            "wins": [10.0] * n,
            "podiums": [15.0] * n,
            "total_points_constructor": [570.0] * n,
            "wins_constructor": [12.0] * n,
            "reliability": [0.95] * n,
            "driver_rolling_position": [3.0] * n,
            "driver_rolling_points": [20.0] * n,
            "constructor_rolling_position": [2.0] * n,
            "constructor_rolling_points": [22.0] * n,
            "top3": [1] * n,
            "target_position": list(range(1, n + 1)),
        }
    )


class TestGetFeatureNames:
    def test_returns_list(self):
        names = get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_contains_numeric_features(self):
        names = get_feature_names()
        for f in NUMERIC_FEATURES:
            assert f in names

    def test_contains_categorical_features(self):
        names = get_feature_names()
        for f in CATEGORICAL_FEATURES:
            assert f in names


class TestAddInteractionFeatures:
    def test_new_columns_created(self):
        df = _make_processed_df(10)
        df = add_interaction_features(df)
        for col in [
            "grid_x_quali",
            "driver_momentum",
            "constructor_momentum",
            "experience_score",
            "team_strength",
        ]:
            assert col in df.columns

    def test_grid_x_quali_correct(self):
        df = _make_processed_df(5)
        df = add_interaction_features(df)
        expected = df["grid"] * df["quali_position"]
        assert (df["grid_x_quali"] == expected).all()

    def test_no_nans_in_interaction_features(self):
        df = _make_processed_df(10)
        df = add_interaction_features(df)
        for col in [
            "driver_momentum",
            "constructor_momentum",
            "experience_score",
            "team_strength",
        ]:
            assert df[col].isna().sum() == 0


class TestBuildFeatures:
    def test_output_has_feature_columns(self):
        df = _make_processed_df(10)
        df_feat = build_features(df)
        for name in get_feature_names():
            assert name in df_feat.columns

    def test_output_has_target_columns(self):
        df = _make_processed_df(10)
        df_feat = build_features(df)
        assert "top3" in df_feat.columns
        assert "target_position" in df_feat.columns

    def test_handles_missing_features_gracefully(self):
        df = _make_processed_df(10)
        # Drop a column that should be created by interaction
        df_minimal = df.drop(
            columns=["driver_rolling_position", "driver_rolling_points"]
        )
        df_feat = build_features(df_minimal)
        # Should not raise; missing cols filled with 0
        assert "grid" in df_feat.columns

    def test_row_count_preserved(self):
        df = _make_processed_df(15)
        df_feat = build_features(df)
        assert len(df_feat) == 15
