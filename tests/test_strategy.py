"""Tests for the strategy recommendation module."""

from strategy import (
    StrategyRecommender,
    StrategyRecommendation,
    recommend_one_stop,
    recommend_two_stop,
    recommend_undercut,
    _degradation_loss,
    _pit_stop_time_loss,
)

DEG = {"soft": 0.12, "medium": 0.07, "hard": 0.04}
MAX_STINT = {"soft": 25, "medium": 40, "hard": 55}


class TestDegradationLoss:
    def test_soft_greater_than_hard(self):
        assert _degradation_loss("soft", 20, DEG) > _degradation_loss("hard", 20, DEG)

    def test_zero_laps(self):
        assert _degradation_loss("medium", 0, DEG) == 0.0

    def test_positive_result(self):
        assert _degradation_loss("medium", 30, DEG) > 0


class TestPitStopTimeLoss:
    def test_zero_stops(self):
        assert _pit_stop_time_loss(0) == 0.0

    def test_two_stops_double_one(self):
        assert _pit_stop_time_loss(2) == 2 * _pit_stop_time_loss(1)


class TestRecommendOneStop:
    def test_returns_recommendation(self):
        rec = recommend_one_stop(57, DEG, MAX_STINT)
        assert isinstance(rec, StrategyRecommendation)

    def test_two_stints(self):
        rec = recommend_one_stop(57, DEG, MAX_STINT)
        assert len(rec.stints) == 2

    def test_one_pit_stop(self):
        rec = recommend_one_stop(57, DEG, MAX_STINT)
        assert rec.total_pit_stops == 1

    def test_stints_cover_all_laps(self):
        total = 57
        rec = recommend_one_stop(total, DEG, MAX_STINT)
        assert rec.stints[0].start_lap == 1
        assert rec.stints[-1].end_lap == total

    def test_strategy_name(self):
        rec = recommend_one_stop(57, DEG, MAX_STINT)
        assert "One-Stop" in rec.strategy_name


class TestRecommendTwoStop:
    def test_three_stints(self):
        rec = recommend_two_stop(57, DEG, MAX_STINT)
        assert len(rec.stints) == 3

    def test_two_pit_stops(self):
        rec = recommend_two_stop(57, DEG, MAX_STINT)
        assert rec.total_pit_stops == 2

    def test_stints_cover_all_laps(self):
        total = 57
        rec = recommend_two_stop(total, DEG, MAX_STINT)
        assert rec.stints[0].start_lap == 1
        assert rec.stints[-1].end_lap == total


class TestRecommendUndercut:
    def test_undercut_pits_before_leader(self):
        rec = recommend_undercut(
            current_lap=20,
            leader_pit_lap=25,
            total_laps=57,
            deg_rates=DEG,
            max_stint=MAX_STINT,
        )
        # Undercut lap should be <= leader_pit_lap - 1
        undercut_pit_lap = rec.stints[0].end_lap
        assert undercut_pit_lap < 25

    def test_returns_recommendation(self):
        rec = recommend_undercut(10, 20, 57, DEG, MAX_STINT)
        assert isinstance(rec, StrategyRecommendation)


class TestStrategyRecommender:
    def setup_method(self):
        self.recommender = StrategyRecommender(
            params={
                "soft_deg_rate": 0.12,
                "medium_deg_rate": 0.07,
                "hard_deg_rate": 0.04,
                "max_stint_soft": 25,
                "max_stint_medium": 40,
                "max_stint_hard": 55,
                "safety_car_probability_threshold": 0.3,
            }
        )

    def test_returns_list(self):
        strategies = self.recommender.recommend(total_laps=57, current_position=3)
        assert isinstance(strategies, list)
        assert len(strategies) >= 1

    def test_all_strategies_are_recommendations(self):
        strategies = self.recommender.recommend(total_laps=57, current_position=5)
        for s in strategies:
            assert isinstance(s, StrategyRecommendation)

    def test_undercut_included_when_leader_pit_lap_provided(self):
        strategies = self.recommender.recommend(
            total_laps=57,
            current_position=5,
            current_lap=15,
            leader_pit_lap=20,
        )
        names = [s.strategy_name for s in strategies]
        assert any("Undercut" in n for n in names)

    def test_conservative_strategy_for_top5_low_sc(self):
        strategies = self.recommender.recommend(
            total_laps=57,
            current_position=2,
            safety_car_probability=0.05,
        )
        # First strategy should be one-stop (conservative)
        assert "One-Stop" in strategies[0].strategy_name

    def test_aggressive_strategy_for_lower_position(self):
        strategies = self.recommender.recommend(
            total_laps=57,
            current_position=15,
            safety_car_probability=0.05,
        )
        # First strategy should be two-stop (aggressive)
        assert "Two-Stop" in strategies[0].strategy_name

    def test_estimated_time_loss_positive(self):
        strategies = self.recommender.recommend(total_laps=57, current_position=5)
        for s in strategies:
            assert s.estimated_time_loss_seconds > 0
