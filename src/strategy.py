"""Race strategy recommendation module.

Recommends optimal pit-stop strategies based on predicted position, tire
degradation rates, and safety-car probability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import yaml
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TireCompound = Literal["soft", "medium", "hard"]

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "params.yaml")

_DEFAULT_DEG = {"soft": 0.12, "medium": 0.07, "hard": 0.04}
_DEFAULT_MAX_STINT = {"soft": 25, "medium": 40, "hard": 55}


def _load_strategy_params() -> dict:
    try:
        with open(PARAMS_PATH, encoding="utf-8") as fh:
            return yaml.safe_load(fh).get("strategy", {})
    except FileNotFoundError:
        return {}


@dataclass
class StintPlan:
    compound: TireCompound
    start_lap: int
    end_lap: int
    degradation_loss: float

    @property
    def laps(self) -> int:
        return self.end_lap - self.start_lap + 1


@dataclass
class StrategyRecommendation:
    stints: list[StintPlan]
    total_pit_stops: int
    estimated_time_loss_seconds: float
    strategy_name: str
    rationale: str


def _degradation_loss(compound: TireCompound, laps: int, deg_rates: dict) -> float:
    """Estimate cumulative lap-time loss from tire degradation (seconds)."""
    rate = deg_rates.get(compound, 0.07)
    # Quadratic degradation model: loss = rate * laps^1.5
    return round(rate * (laps**1.5), 2)


def _pit_stop_time_loss(n_stops: int, pit_delta: float = 22.0) -> float:
    """Time lost in pit lane per stop (seconds)."""
    return n_stops * pit_delta


def recommend_one_stop(
    total_laps: int,
    deg_rates: dict,
    max_stint: dict,
) -> StrategyRecommendation:
    """Classic medium–hard one-stop strategy."""
    # Optimal pit window ~35% into the race
    pit_lap = max(10, int(total_laps * 0.35))
    stint1 = StintPlan(
        compound="medium",
        start_lap=1,
        end_lap=pit_lap,
        degradation_loss=_degradation_loss("medium", pit_lap, deg_rates),
    )
    stint2 = StintPlan(
        compound="hard",
        start_lap=pit_lap + 1,
        end_lap=total_laps,
        degradation_loss=_degradation_loss("hard", total_laps - pit_lap, deg_rates),
    )
    total_deg = stint1.degradation_loss + stint2.degradation_loss
    time_loss = _pit_stop_time_loss(1) + total_deg
    return StrategyRecommendation(
        stints=[stint1, stint2],
        total_pit_stops=1,
        estimated_time_loss_seconds=round(time_loss, 2),
        strategy_name="One-Stop (Medium → Hard)",
        rationale=(
            "Conservative one-stop strategy. Pit around lap "
            f"{pit_lap} for maximum tire life and predictability."
        ),
    )


def recommend_two_stop(
    total_laps: int,
    deg_rates: dict,
    max_stint: dict,
) -> StrategyRecommendation:
    """Aggressive soft–medium–medium two-stop strategy."""
    pit1 = max(10, int(total_laps * 0.28))
    pit2 = max(pit1 + 10, int(total_laps * 0.60))
    stint1 = StintPlan(
        compound="soft",
        start_lap=1,
        end_lap=pit1,
        degradation_loss=_degradation_loss("soft", pit1, deg_rates),
    )
    stint2 = StintPlan(
        compound="medium",
        start_lap=pit1 + 1,
        end_lap=pit2,
        degradation_loss=_degradation_loss("medium", pit2 - pit1, deg_rates),
    )
    stint3 = StintPlan(
        compound="medium",
        start_lap=pit2 + 1,
        end_lap=total_laps,
        degradation_loss=_degradation_loss("medium", total_laps - pit2, deg_rates),
    )
    total_deg = (
        stint1.degradation_loss + stint2.degradation_loss + stint3.degradation_loss
    )
    time_loss = _pit_stop_time_loss(2) + total_deg
    return StrategyRecommendation(
        stints=[stint1, stint2, stint3],
        total_pit_stops=2,
        estimated_time_loss_seconds=round(time_loss, 2),
        strategy_name="Two-Stop (Soft → Medium → Medium)",
        rationale=(
            f"Aggressive two-stop to maximize early pace on softs. "
            f"Pit laps: {pit1}, {pit2}."
        ),
    )


def recommend_undercut(
    current_lap: int,
    leader_pit_lap: int,
    total_laps: int,
    deg_rates: dict,
    max_stint: dict,
) -> StrategyRecommendation:
    """Undercut strategy: pit before the leader."""
    undercut_lap = max(current_lap + 1, leader_pit_lap - 3)
    stint1 = StintPlan(
        compound="medium",
        start_lap=current_lap,
        end_lap=undercut_lap,
        degradation_loss=_degradation_loss(
            "medium", undercut_lap - current_lap + 1, deg_rates
        ),
    )
    stint2 = StintPlan(
        compound="soft",
        start_lap=undercut_lap + 1,
        end_lap=total_laps,
        degradation_loss=_degradation_loss(
            "soft", total_laps - undercut_lap, deg_rates
        ),
    )
    total_deg = stint1.degradation_loss + stint2.degradation_loss
    time_loss = _pit_stop_time_loss(1) + total_deg
    return StrategyRecommendation(
        stints=[stint1, stint2],
        total_pit_stops=1,
        estimated_time_loss_seconds=round(time_loss, 2),
        strategy_name=f"Undercut (pit lap {undercut_lap})",
        rationale=(
            f"Undercut the car ahead by pitting {leader_pit_lap - undercut_lap} "
            "laps earlier. Fresh soft tyres should recover the position."
        ),
    )


class StrategyRecommender:
    """High-level interface for race strategy recommendations."""

    def __init__(self, params: dict | None = None):
        loaded = _load_strategy_params()
        p = {**loaded, **(params or {})}
        self.deg_rates: dict = {
            "soft": p.get("soft_deg_rate", 0.12),
            "medium": p.get("medium_deg_rate", 0.07),
            "hard": p.get("hard_deg_rate", 0.04),
        }
        self.max_stint: dict = {
            "soft": p.get("max_stint_soft", 25),
            "medium": p.get("max_stint_medium", 40),
            "hard": p.get("max_stint_hard", 55),
        }
        self.sc_threshold: float = float(p.get("safety_car_probability_threshold", 0.3))

    def recommend(
        self,
        total_laps: int,
        current_position: int,
        safety_car_probability: float = 0.15,
        current_lap: int = 1,
        leader_pit_lap: int | None = None,
    ) -> list[StrategyRecommendation]:
        """Return ranked list of strategy recommendations."""
        strategies: list[StrategyRecommendation] = []

        if current_position <= 5 and safety_car_probability < self.sc_threshold:
            # Favour conservative one-stop for points-scoring positions
            strategies.append(
                recommend_one_stop(total_laps, self.deg_rates, self.max_stint)
            )
            strategies.append(
                recommend_two_stop(total_laps, self.deg_rates, self.max_stint)
            )
        else:
            # Behind – more aggressive two-stop to gain track position
            strategies.append(
                recommend_two_stop(total_laps, self.deg_rates, self.max_stint)
            )
            strategies.append(
                recommend_one_stop(total_laps, self.deg_rates, self.max_stint)
            )

        if leader_pit_lap is not None and current_lap < leader_pit_lap:
            strategies.insert(
                0,
                recommend_undercut(
                    current_lap,
                    leader_pit_lap,
                    total_laps,
                    self.deg_rates,
                    self.max_stint,
                ),
            )

        return strategies
