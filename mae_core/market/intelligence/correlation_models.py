"""CorrelationPair dataclass — shared model for correlation tracking."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class CorrelationPair:
    """Tracks correlation between two signals."""
    signal_a: str
    signal_b: str
    current_correlation: float = 0.0
    historical_mean: float = 0.0
    historical_std: float = 0.1
    correlation_zscore: float = 0.0
    is_anomalous: bool = False
    observation_count: int = 0
    last_updated: datetime = None
