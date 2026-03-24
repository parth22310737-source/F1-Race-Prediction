"""Tests for the predict module (no model files required – uses mocked models)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from predict import RaceEntry, RacePredictor, PredictionResult


class TestRaceEntry:
    def test_defaults(self):
        entry = RaceEntry()
        assert entry.grid == 10
        assert entry.quali_position == 10

    def test_derived_fields(self):
        entry = RaceEntry(grid=3, quali_position=4)
        assert entry.grid_x_quali == 12

    def test_driver_momentum(self):
        entry = RaceEntry(driver_rolling_points=20.0, driver_rolling_position=5.0)
        assert entry.driver_momentum == pytest.approx(20.0 / 5.0)

    def test_experience_score(self):
        entry = RaceEntry(wins=10, podiums=15)
        assert entry.experience_score == pytest.approx(10 * 3 + 15)

    def test_to_array_shape(self):
        entry = RaceEntry()
        arr = entry.to_array()
        assert arr.ndim == 2
        assert arr.shape[0] == 1

    def test_zero_division_guard_driver_momentum(self):
        entry = RaceEntry(driver_rolling_position=0.0)
        # Should not raise; momentum = points / max(pos, 1)
        assert entry.driver_momentum >= 0


class TestRacePredictor:
    def _make_predictor(self):
        """Create a RacePredictor with mocked internal models."""
        predictor = object.__new__(RacePredictor)
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_reg = MagicMock()
        mock_reg.predict.return_value = np.array([3.2])
        mock_scaler = MagicMock()
        mock_scaler.transform.side_effect = lambda x: x
        predictor.clf = mock_clf
        predictor.reg = mock_reg
        predictor.scaler = mock_scaler
        return predictor

    def test_predict_returns_result(self):
        predictor = self._make_predictor()
        result = predictor.predict(RaceEntry())
        assert isinstance(result, PredictionResult)

    def test_top3_probability_range(self):
        predictor = self._make_predictor()
        result = predictor.predict(RaceEntry())
        assert 0.0 <= result.top3_probability <= 1.0

    def test_is_top3_when_prob_high(self):
        predictor = self._make_predictor()
        result = predictor.predict(RaceEntry())
        assert result.is_top3 is True  # mock returns prob=0.7

    def test_is_not_top3_when_prob_low(self):
        predictor = self._make_predictor()
        predictor.clf.predict_proba.return_value = np.array([[0.8, 0.2]])
        result = predictor.predict(RaceEntry())
        assert result.is_top3 is False

    def test_predicted_position_positive(self):
        predictor = self._make_predictor()
        result = predictor.predict(RaceEntry())
        assert result.predicted_position > 0

    def test_predict_batch(self):
        predictor = self._make_predictor()
        entries = [RaceEntry(grid=i) for i in range(1, 6)]
        results = predictor.predict_batch(entries)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, PredictionResult)
