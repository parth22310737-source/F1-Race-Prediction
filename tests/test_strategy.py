"""Tests for the strategy recommendation system."""
from __future__ import annotations

import pytest

from src.strategy.recommend import recommend_strategy


def test_strategy_dry_returns_valid():
    result = recommend_strategy(total_laps=57, weather="dry")
    assert "stints" in result
    assert "total_time_seconds" in result
    assert "pit_stops" in result
    assert result["pit_stops"] >= 1


def test_strategy_wet_conditions():
    result = recommend_strategy(total_laps=57, weather="wet")
    assert "recommendation" in result
    assert "wet" in result["recommendation"].lower() or "intermediate" in result["recommendation"].lower()


def test_strategy_mixed_conditions():
    result = recommend_strategy(total_laps=57, weather="mixed")
    assert result["pit_stops"] >= 1


def test_strategy_stints_sum_to_total_laps():
    total = 57
    result = recommend_strategy(total_laps=total, weather="dry")
    laps_in_stints = sum(s["laps"] for s in result["stints"])
    assert laps_in_stints == total


def test_strategy_two_compounds_minimum():
    """F1 regulations require at least two different compounds in dry races."""
    result = recommend_strategy(total_laps=57, weather="dry")
    compounds_used = {s["compound"] for s in result["stints"]}
    assert len(compounds_used) >= 2


def test_strategy_positive_time():
    result = recommend_strategy(total_laps=57, weather="dry")
    assert result["total_time_seconds"] > 0


def test_strategy_different_laps():
    r30 = recommend_strategy(total_laps=30, weather="dry")
    r70 = recommend_strategy(total_laps=70, weather="dry")
    assert r30["total_time_seconds"] < r70["total_time_seconds"]
