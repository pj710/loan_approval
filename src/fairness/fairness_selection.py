"""Strategy selection helpers for fairness mitigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class FairnessStrategyResult:
    name: str
    predictions: Any
    probabilities: Any
    postprocessor: Any | None
    metrics: dict[str, float]
    fairness_components: dict[str, float]
    fairness_score: float
    expected_value: float
    precision: float


def fairness_tradeoff_score(metrics: dict[str, float]) -> float:
    """Lower is better; 0 means parity across all three fairness metrics."""
    demographic_parity = float(metrics.get("demographic_parity_difference", 0.0))
    equalized_odds = float(metrics.get("equalized_odds_difference", 0.0))
    disparate_impact = float(metrics.get("disparate_impact", 0.0))
    return demographic_parity + equalized_odds + abs(1.0 - disparate_impact)


def choose_best_fairness_strategy(
    candidates: Iterable[FairnessStrategyResult],
    min_precision: float,
    precision_tolerance: float = 0.002,
) -> FairnessStrategyResult:
    """Pick the strategy with the best combined fairness score and acceptable precision."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError("At least one fairness candidate is required.")

    compliant = [candidate for candidate in candidates if candidate.precision >= (min_precision - precision_tolerance)]
    ranked = compliant if compliant else candidates
    return sorted(
        ranked,
        key=lambda candidate: (
            candidate.fairness_score,
            -candidate.expected_value,
            -candidate.precision,
        ),
    )[0]
