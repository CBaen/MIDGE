"""Correlation analysis helpers — find_leading_pairs and related utilities."""

import logging
from typing import List, Tuple

from mae_core.market.intelligence.correlation_models import CorrelationPair

logger = logging.getLogger(__name__)


def find_leading_pairs(
    correlations: dict,
    correlation_history: dict,
    signal_domains: dict,
    window_size: int,
    anomaly_threshold: float,
    min_observations: int,
    correlation_threshold: float = 0.7,
    anomaly_only: bool = True,
) -> List[Tuple[CorrelationPair, str]]:
    """Find pairs that may indicate leading indicator relationships.

    A leading pair shows:
    - High current correlation (signals moving together)
    - This correlation is unusual (wasn't there before)

    Returns:
        List of (pair, reason) tuples
    """
    leading = []
    for pair in correlations.values():
        if pair.observation_count < min_observations:
            continue

        reasons = []

        if abs(pair.current_correlation) >= correlation_threshold:
            if pair.correlation_zscore > anomaly_threshold:
                reasons.append(
                    f"unusual positive correlation ({pair.current_correlation:.2f}, "
                    f"z={pair.correlation_zscore:.2f})"
                )
            elif pair.correlation_zscore < -anomaly_threshold:
                reasons.append(
                    f"unusual negative correlation ({pair.current_correlation:.2f}, "
                    f"z={pair.correlation_zscore:.2f})"
                )
            elif not anomaly_only:
                reasons.append(f"strong correlation ({pair.current_correlation:.2f})")

        if reasons:
            leading.append((pair, "; ".join(reasons)))

    leading.sort(key=lambda x: abs(x[0].correlation_zscore), reverse=True)
    return leading
