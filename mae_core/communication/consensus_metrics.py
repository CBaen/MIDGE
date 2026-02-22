"""Consensus Metrics - Statistical confidence for quorum decisions.

Uses Wilson score intervals for small-sample confidence estimation.
Determines when consensus is statistically significant.

Based on: Wilson (1927), Agresti & Coull (1998).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ConfidenceLevel(Enum):
    """Qualitative confidence bands."""

    VERY_LOW = 0.0
    LOW = 0.2
    MODERATE = 0.4
    HIGH = 0.6
    VERY_HIGH = 0.8


@dataclass
class ConfidenceInterval:
    """Statistical confidence in a consensus measurement."""

    point_estimate: float  # proportion of agents supporting
    lower_bound: float  # 95% CI lower
    upper_bound: float  # 95% CI upper
    confidence_width: float  # interval width
    is_significant: bool  # above significance threshold
    confidence_level: ConfidenceLevel
    supporters: int
    total_agents: int


class ConsensusConfidenceCalculator:
    """Calculates statistical confidence for consensus measurements.

    Uses Wilson score interval - better than normal approximation
    for small samples (which is common in agent swarms).
    """

    def __init__(
        self, significance_threshold: float = 0.3, z_score: float = 1.96
    ) -> None:
        self._threshold = significance_threshold
        self._z = z_score

    def wilson_score_interval(
        self, successes: int, total: int
    ) -> tuple[float, float]:
        """Wilson score confidence interval for a binomial proportion."""
        if total == 0:
            return 0.0, 0.0

        p = successes / total
        z2 = self._z ** 2
        denom = 1 + z2 / total

        center = (p + z2 / (2 * total)) / denom
        spread = self._z * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total) / denom

        return max(0.0, center - spread), min(1.0, center + spread)

    def calculate(
        self, supporters: int, total_agents: int, min_votes: int = 3
    ) -> ConfidenceInterval:
        """Calculate confidence interval for a consensus measurement."""
        if total_agents == 0 or supporters < min_votes:
            return ConfidenceInterval(
                point_estimate=supporters / max(total_agents, 1),
                lower_bound=0.0,
                upper_bound=0.0,
                confidence_width=1.0,
                is_significant=False,
                confidence_level=ConfidenceLevel.VERY_LOW,
                supporters=supporters,
                total_agents=total_agents,
            )

        lower, upper = self.wilson_score_interval(supporters, total_agents)
        point = supporters / total_agents
        width = upper - lower

        # Handle edge cases with Rule of Three
        if supporters == 0:
            upper = self._zero_success_upper_bound(total_agents)
            lower = 0.0
        elif supporters == total_agents:
            lower = self._full_success_lower_bound(total_agents)
            upper = 1.0

        is_sig = lower >= self._threshold
        level = self._classify_confidence(lower)

        return ConfidenceInterval(
            point_estimate=point,
            lower_bound=lower,
            upper_bound=upper,
            confidence_width=width,
            is_significant=is_sig,
            confidence_level=level,
            supporters=supporters,
            total_agents=total_agents,
        )

    def compare_consensus(
        self, msg_a_votes: int, msg_b_votes: int, total_agents: int
    ) -> str:
        """Compare two consensus measurements. Returns 'A_STRONGER', 'B_STRONGER', or 'NO_DIFFERENCE'."""
        ci_a = self.calculate(msg_a_votes, total_agents)
        ci_b = self.calculate(msg_b_votes, total_agents)

        if ci_a.lower_bound > ci_b.upper_bound:
            return "A_STRONGER"
        if ci_b.lower_bound > ci_a.upper_bound:
            return "B_STRONGER"
        return "NO_DIFFERENCE"

    def get_consensus_quality_score(
        self, supporters: int, total_agents: int
    ) -> float:
        """Combined quality score [0, 1] considering point estimate and confidence width."""
        ci = self.calculate(supporters, total_agents)
        if not ci.is_significant:
            return 0.0
        # Quality = point estimate * (1 - width), rewarding narrow confident intervals
        return ci.point_estimate * (1.0 - min(1.0, ci.confidence_width))

    def _zero_success_upper_bound(self, total: int) -> float:
        """Rule of Three: upper bound when 0 successes observed."""
        return 3.0 / total if total > 0 else 1.0

    def _full_success_lower_bound(self, total: int) -> float:
        """Rule of Three: lower bound when all succeed."""
        return 1.0 - 3.0 / total if total > 3 else 0.0

    def _classify_confidence(self, lower_bound: float) -> ConfidenceLevel:
        if lower_bound >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        if lower_bound >= 0.6:
            return ConfidenceLevel.HIGH
        if lower_bound >= 0.4:
            return ConfidenceLevel.MODERATE
        if lower_bound >= 0.2:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.VERY_LOW


def is_consensus_reliable(
    supporters: int, total_agents: int, min_confidence: float = 0.5
) -> bool:
    """Quick check: is this consensus statistically reliable?"""
    calc = ConsensusConfidenceCalculator()
    ci = calc.calculate(supporters, total_agents)
    return ci.is_significant and ci.lower_bound >= min_confidence
