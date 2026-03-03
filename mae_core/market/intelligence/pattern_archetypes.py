"""Pattern Archetypes — Gift 8 (Council Gift).

Canonical market pattern templates: accumulation, distribution, Wyckoff
phases, squeeze/expansion, capitulation/exhaustion. MIDGE learns to
recognize these higher-order structural patterns across any ticker.

"Patterns that others wouldn't see" — these are the deep structural
patterns that experienced traders recognize intuitively. Accumulation
looks like nothing happening. Distribution looks like normal trading.
Only someone who has seen the archetype hundreds of times recognizes
it unfolding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PatternArchetype:
    """A canonical market pattern template."""

    archetype_id: str
    name: str
    required_signals: List[str]  # Signal sources that must be present
    optional_signals: List[str]  # Signals that strengthen the match
    direction_bias: str  # "bullish", "bearish", "neutral"
    typical_duration_days: int
    historical_win_rate: float  # From market research literature
    description: str


@dataclass
class ArchetypeMatch:
    """Result of matching signals against an archetype template."""

    archetype_id: str
    ticker: str
    match_score: float  # 0.0 to 1.0
    matched_signals: List[str]
    missing_signals: List[str]
    estimated_completion_pct: float


@dataclass
class PartialMatch:
    """Tracks a partially-matching archetype for completion hunting (Gift 10)."""

    archetype_id: str
    ticker: str
    matched_so_far: List[str]
    still_needed: List[str]
    first_match_time: datetime
    match_score: float


# --- Built-in archetypes (the pattern library) ---

_BUILTIN_ARCHETYPES = {
    "accumulation": PatternArchetype(
        archetype_id="accumulation",
        name="Accumulation",
        required_signals=["insider", "technical"],
        optional_signals=["regulatory", "institutional"],
        direction_bias="bullish",
        typical_duration_days=30,
        historical_win_rate=0.62,
        description=(
            "Low volume + insider buying + range-bound price + declining "
            "volatility → bullish breakout candidate"
        ),
    ),
    "distribution": PatternArchetype(
        archetype_id="distribution",
        name="Distribution",
        required_signals=["insider", "technical"],
        optional_signals=["sentiment", "social"],
        direction_bias="bearish",
        typical_duration_days=30,
        historical_win_rate=0.58,
        description=(
            "High volume + insider selling + range-bound price + rising "
            "volatility → bearish breakdown candidate"
        ),
    ),
    "squeeze": PatternArchetype(
        archetype_id="squeeze",
        name="Volatility Squeeze",
        required_signals=["technical"],
        optional_signals=["positioning", "institutional"],
        direction_bias="neutral",
        typical_duration_days=14,
        historical_win_rate=0.55,
        description=(
            "Bollinger bandwidth < 20th percentile + declining volume → "
            "explosive move (direction TBD)"
        ),
    ),
    "capitulation": PatternArchetype(
        archetype_id="capitulation",
        name="Capitulation",
        required_signals=["technical", "sentiment"],
        optional_signals=["insider", "volatility"],
        direction_bias="bullish",
        typical_duration_days=5,
        historical_win_rate=0.60,
        description=(
            "Volume spike 5σ + RSI < 20 + price gap down → reversal "
            "candidate (bearish exhaustion)"
        ),
    ),
    "momentum_ignition": PatternArchetype(
        archetype_id="momentum_ignition",
        name="Momentum Ignition",
        required_signals=["technical", "insider"],
        optional_signals=["sentiment", "regulatory"],
        direction_bias="bullish",
        typical_duration_days=10,
        historical_win_rate=0.58,
        description=(
            "Price breakout + volume surge + MACD crossover + insider "
            "buying → trend start"
        ),
    ),
    "failed_breakout": PatternArchetype(
        archetype_id="failed_breakout",
        name="Failed Breakout",
        required_signals=["technical"],
        optional_signals=["sentiment", "insider"],
        direction_bias="bearish",
        typical_duration_days=5,
        historical_win_rate=0.56,
        description=(
            "Price breaks resistance + no volume confirmation + "
            "immediate reversal → bull trap"
        ),
    ),
    "sector_rotation": PatternArchetype(
        archetype_id="sector_rotation",
        name="Sector Rotation",
        required_signals=["technical", "institutional"],
        optional_signals=["positioning", "sentiment"],
        direction_bias="neutral",
        typical_duration_days=21,
        historical_win_rate=0.52,
        description=(
            "Money flowing out of one sector ETF into another — "
            "relative strength shift"
        ),
    ),
    "smart_money_divergence": PatternArchetype(
        archetype_id="smart_money_divergence",
        name="Smart Money Divergence",
        required_signals=["insider", "regulatory"],
        optional_signals=["technical", "positioning"],
        direction_bias="bullish",
        typical_duration_days=21,
        historical_win_rate=0.63,
        description=(
            "Price falling + insider buying + short interest declining → "
            "stealth accumulation"
        ),
    ),
}


class PatternArchetypeEngine:
    """Scan for market pattern archetypes in signal flow.

    Matches incoming signals against canonical pattern templates.
    Tracks partial matches for pattern completion (Gift 10).
    """

    def __init__(self, price_fetcher=None):
        self._price_fetcher = price_fetcher
        self._archetypes: Dict[str, PatternArchetype] = dict(_BUILTIN_ARCHETYPES)
        self._active_matches: Dict[str, PartialMatch] = {}  # ticker -> best partial

    def scan_for_archetypes(
        self,
        ticker: str,
        signals: Optional[List[dict]] = None,
        signal_domains: Optional[List[str]] = None,
    ) -> List[ArchetypeMatch]:
        """Scan signals for archetype matches on a given ticker.

        Args:
            ticker: The ticker symbol to scan.
            signals: List of signal dicts with 'domain' key.
            signal_domains: Alternative: just pass domain names present.

        Returns:
            List of ArchetypeMatch results, sorted by match_score descending.
        """
        # Determine which domains are present
        if signal_domains is not None:
            present_domains = set(signal_domains)
        elif signals is not None:
            present_domains = set()
            for sig in signals:
                if isinstance(sig, dict):
                    d = sig.get("domain", "")
                else:
                    d = getattr(sig, "domain", "")
                if d:
                    present_domains.add(d)
        else:
            return []

        if not present_domains:
            return []

        matches = []
        for archetype in self._archetypes.values():
            match = self._match_archetype(ticker, archetype, present_domains)
            if match is not None:
                matches.append(match)

                # Track partial matches for Gift 10
                if 0.0 < match.match_score < 1.0:
                    existing = self._active_matches.get(ticker)
                    if existing is None or match.match_score > existing.match_score:
                        self._active_matches[ticker] = PartialMatch(
                            archetype_id=match.archetype_id,
                            ticker=ticker,
                            matched_so_far=match.matched_signals,
                            still_needed=match.missing_signals,
                            first_match_time=datetime.now(),
                            match_score=match.match_score,
                        )

        matches.sort(key=lambda m: m.match_score, reverse=True)
        return matches

    def _match_archetype(
        self,
        ticker: str,
        archetype: PatternArchetype,
        present_domains: set,
    ) -> Optional[ArchetypeMatch]:
        """Match a single archetype against present signal domains."""
        required = set(archetype.required_signals)
        optional = set(archetype.optional_signals)

        matched_required = required & present_domains
        matched_optional = optional & present_domains
        missing_required = required - present_domains

        # Must have at least one required signal
        if not matched_required:
            return None

        # Score: required signals are weighted 2x, optional 1x
        total_weight = len(required) * 2 + len(optional)
        matched_weight = len(matched_required) * 2 + len(matched_optional)

        if total_weight == 0:
            return None

        score = matched_weight / total_weight
        completion_pct = len(matched_required) / len(required) if required else 0.0

        matched_signals = list(matched_required | matched_optional)
        missing_signals = list(missing_required)

        return ArchetypeMatch(
            archetype_id=archetype.archetype_id,
            ticker=ticker,
            match_score=round(score, 3),
            matched_signals=matched_signals,
            missing_signals=missing_signals,
            estimated_completion_pct=round(completion_pct, 3),
        )

    def get_partial_matches(self) -> Dict[str, PartialMatch]:
        """Return active partial matches for pattern completion (Gift 10)."""
        return dict(self._active_matches)

    def get_archetype(self, archetype_id: str) -> Optional[PatternArchetype]:
        """Look up an archetype by ID."""
        return self._archetypes.get(archetype_id)

    def get_statistics(self) -> dict:
        """Return engine statistics for holon protocol."""
        return {
            "registered_archetypes": len(self._archetypes),
            "active_partial_matches": len(self._active_matches),
            "archetype_ids": list(self._archetypes.keys()),
        }
