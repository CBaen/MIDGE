"""Post-Mortem Utility Functions — shared helpers for post_mortem.py and post_mortem_analysis.py.

Extracted from post_mortem.py to avoid circular imports and stay under the
500-line monolith cap.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional


def _safe_mean(values: list) -> Optional[float]:
    """Mean of a list, None if empty."""
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _payoff_ratio(recs: List[dict]) -> Optional[float]:
    """Avg winning return / avg losing return magnitude."""
    winners = [abs(r.get("return_pct", 0.0)) for r in recs if r.get("was_correct")]
    losers = [abs(r.get("return_pct", 0.0)) for r in recs if not r.get("was_correct")]
    if not winners or not losers:
        return None
    avg_win = sum(winners) / len(winners)
    avg_loss = sum(losers) / len(losers)
    if avg_loss == 0:
        return None
    return round(avg_win / avg_loss, 3)


def _timing_insight(buckets: dict, total: int) -> str:
    """Plain-language summary of timing distribution."""
    if total == 0:
        return "No outcomes to analyze"
    missed = buckets.get("missed", 0)
    early = buckets.get("early", 0)
    late = buckets.get("late", 0)
    parts = []
    if missed / total > 0.5:
        parts.append(f"majority ({missed}/{total}) missed entirely")
    if early / total > 0.2:
        parts.append(f"{early} moved early (watch for faster-moving opportunities)")
    if late / total > 0.2:
        parts.append(f"{late} moved late (window may be too short)")
    return "; ".join(parts) if parts else "Timing distribution looks normal"


def _json_default(obj: Any) -> Any:
    """JSON serializer for non-standard types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
