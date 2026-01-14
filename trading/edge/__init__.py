"""
Edge detection modules - finding patterns before they become obvious.
"""

from .politician_tracker import (
    PoliticianTracker,
    CorrelationSignal,
    PoliticianProfile,
    find_correlations,
    get_daily_alerts,
)

__all__ = [
    "PoliticianTracker",
    "CorrelationSignal",
    "PoliticianProfile",
    "find_correlations",
    "get_daily_alerts",
]
