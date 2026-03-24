"""
F1 Race Strategy Recommendation System.

Provides optimal tire-compound selection and pit-stop lap recommendations
for a given race scenario using a greedy lap-time simulation.

Tire compounds modelled
-----------------------
- soft   : fast but short-lived
- medium : balanced
- hard   : slow but long-lasting

The simulator estimates total race time for a set of candidate strategies and
returns the one with the lowest projected time including pit-stop time loss.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COMPOUND_BASE_LAPTIME = {
    "soft": 90.0,    # seconds per lap (baseline)
    "medium": 91.5,
    "hard": 93.0,
}
COMPOUND_DEGRADATION = {
    "soft": 0.08,    # seconds lost per lap of tyre age
    "medium": 0.05,
    "hard": 0.03,
}
MAX_TIRE_LIFE = {
    "soft": 25,
    "medium": 35,
    "hard": 50,
}


@dataclass
class Strategy:
    """Represents a pit-stop strategy for one stint."""

    stints: list[tuple[str, int]]  # [(compound, n_laps), …]
    total_time: float = 0.0
    pit_stops: int = 0
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stints": [{"compound": c, "laps": l} for c, l in self.stints],
            "total_time_seconds": round(self.total_time, 2),
            "pit_stops": self.pit_stops,
            "recommendation": self.recommendation,
        }


def _simulate_stint(compound: str, n_laps: int) -> float:
    """Estimate total stint time including tyre degradation."""
    base = COMPOUND_BASE_LAPTIME[compound]
    deg = COMPOUND_DEGRADATION[compound]
    return sum(base + deg * lap for lap in range(n_laps))


def _build_candidate_strategies(
    total_laps: int,
    pit_stop_time: float,
    params: dict,
) -> list[Strategy]:
    """Enumerate candidate 1- and 2-stop strategies."""
    strategy_cfg = params.get("strategy", {})
    soft_life = strategy_cfg.get("soft_tire_life", MAX_TIRE_LIFE["soft"])
    medium_life = strategy_cfg.get("medium_tire_life", MAX_TIRE_LIFE["medium"])
    hard_life = strategy_cfg.get("hard_tire_life", MAX_TIRE_LIFE["hard"])
    effective_life = {"soft": soft_life, "medium": medium_life, "hard": hard_life}

    compounds = ["soft", "medium", "hard"]
    candidates: list[Strategy] = []

    # --- 1-stop strategies ---
    for c1, c2 in product(compounds, compounds):
        if c1 == c2:
            continue  # F1 rules: must use ≥2 compounds
        for split in range(5, total_laps - 4):
            s1_laps = split
            s2_laps = total_laps - split
            if s1_laps > effective_life[c1] or s2_laps > effective_life[c2]:
                continue
            time = (
                _simulate_stint(c1, s1_laps)
                + pit_stop_time
                + _simulate_stint(c2, s2_laps)
            )
            candidates.append(
                Strategy(stints=[(c1, s1_laps), (c2, s2_laps)], total_time=time, pit_stops=1)
            )

    # --- 2-stop strategies ---
    for c1, c2, c3 in product(compounds, compounds, compounds):
        if c1 == c2 == c3:
            continue
        for s1 in range(5, total_laps - 8):
            for s2 in range(5, total_laps - s1 - 4):
                s3 = total_laps - s1 - s2
                if s3 < 4:
                    continue
                if (
                    s1 > effective_life[c1]
                    or s2 > effective_life[c2]
                    or s3 > effective_life[c3]
                ):
                    continue
                time = (
                    _simulate_stint(c1, s1)
                    + pit_stop_time
                    + _simulate_stint(c2, s2)
                    + pit_stop_time
                    + _simulate_stint(c3, s3)
                )
                candidates.append(
                    Strategy(
                        stints=[(c1, s1), (c2, s2), (c3, s3)],
                        total_time=time,
                        pit_stops=2,
                    )
                )

    return candidates


def recommend_strategy(
    total_laps: int = 57,
    grid_position: int = 1,
    weather: str = "dry",
    circuit_type: str = "mixed",
    params_path: str = "params.yaml",
) -> dict[str, Any]:
    """Return the optimal pit-stop strategy for the given race scenario.

    Parameters
    ----------
    total_laps:
        Number of laps in the race.
    grid_position:
        Starting grid position (1 = pole).
    weather:
        Race weather condition: ``"dry"``, ``"wet"``, or ``"mixed"``.
    circuit_type:
        Type of circuit (affects tyre choice heuristics).
    params_path:
        Path to the YAML params file.
    """
    try:
        with open(params_path) as fh:
            params = yaml.safe_load(fh)
    except FileNotFoundError:
        params = {}

    pit_stop_time = params.get("strategy", {}).get("pit_stop_time_loss", 22.0)

    # In wet/mixed conditions force intermediates as the first compound
    if weather in ("wet", "mixed"):
        best = Strategy(
            stints=[("medium", total_laps // 2), ("hard", total_laps - total_laps // 2)],
            total_time=0.0,
            pit_stops=1,
            recommendation=(
                "Wet/mixed conditions detected. Start on intermediates "
                "(modelled as medium) and switch to slicks when track dries."
            ),
        )
        best.total_time = (
            _simulate_stint("medium", total_laps // 2)
            + pit_stop_time
            + _simulate_stint("hard", total_laps - total_laps // 2)
        )
        return best.to_dict()

    candidates = _build_candidate_strategies(total_laps, pit_stop_time, params)
    if not candidates:
        # Fallback
        return Strategy(
            stints=[("medium", total_laps)],
            total_time=_simulate_stint("medium", total_laps),
            pit_stops=0,
            recommendation="No valid multi-compound strategy found; run a single stint.",
        ).to_dict()

    best = min(candidates, key=lambda s: s.total_time)

    stint_desc = " → ".join(f"{c.upper()} ({l} laps)" for c, l in best.stints)
    best.recommendation = (
        f"Optimal {best.pit_stops}-stop strategy: {stint_desc}. "
        f"Projected race time: {best.total_time:.1f}s. "
        f"Pit stop{'s' if best.pit_stops > 1 else ''} at lap "
        + ", ".join(
            str(sum(l for _, l in best.stints[:i]))
            for i in range(1, best.pit_stops + 1)
        )
        + "."
    )
    logger.info("Recommended strategy: %s", best.recommendation)
    return best.to_dict()
