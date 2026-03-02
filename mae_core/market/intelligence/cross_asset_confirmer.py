#!/usr/bin/env python3
"""
cross_asset_confirmer.py - Cross-Asset Regime Confirmation

Checks whether correlated or anti-correlated assets are moving in ways that
confirm or contradict a signal. Well-established market relationships are
hardcoded as CorrelationPair objects.

Core insight: A bullish signal on CL=F (crude oil) is MORE credible when XLE
(energy sector ETF) is also moving up. When they diverge — crude up, XLE down
— something unusual is happening and the signal should be treated with
skepticism.

Built-in pairs cover the most liquid, well-documented relationships:
  - S&P vs Nasdaq (large cap comovement)
  - Gold vs Dollar (classic inverse)
  - Crude vs Energy sector
  - Bonds vs Stocks (risk-on/risk-off)
  - VIX vs S&P (fear gauge)
  - Futures vs their ETF proxies

Confirmation score range: -1.0 (pure contradiction) to +1.0 (full confirmation).

Usage:
    from mae_core.market.intelligence.cross_asset_confirmer import (
        CrossAssetConfirmer,
    )

    confirmer = CrossAssetConfirmer(price_fetcher=price_fetcher)
    result = confirmer.check_confirmation("CL=F", direction="bullish")
    print(result.confirmation_score)  # e.g. 0.75
    print(result.confirming_pairs)    # ["CL=F → XLE (positive, XLE +1.2%)"]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CorrelationPair:
    """
    A known market relationship between two instruments.

    relationship:
        "positive" — the two instruments normally move in the same direction.
        "inverse"  — they normally move in opposite directions.

    weight:
        Confidence in the relationship (0–1). Lower weight = less influence
        on the final confirmation score. Well-established pairs use 1.0;
        sector correlations use 0.7–0.8.
    """
    ticker_a: str
    ticker_b: str
    relationship: str   # "positive" or "inverse"
    weight: float = 1.0


@dataclass
class CrossAssetResult:
    """
    Result of cross-asset confirmation check for a single ticker + direction.

    confirmation_score:
        -1.0 = all checked pairs contradict the signal.
         0.0 = neutral or no data available.
        +1.0 = all checked pairs confirm the signal.

    confirming_pairs / conflicting_pairs:
        Human-readable descriptions of what confirmed or conflicted.

    pairs_checked:
        Total number of correlation pairs that were evaluated (data was
        available for both instruments). 0 means no data at all.
    """
    confirmation_score: float
    confirming_pairs: List[str] = field(default_factory=list)
    conflicting_pairs: List[str] = field(default_factory=list)
    pairs_checked: int = 0


# ---------------------------------------------------------------------------
# Confirmer
# ---------------------------------------------------------------------------

class CrossAssetConfirmer:
    """
    Validates signals by checking how correlated instruments are moving.

    price_fetcher:
        Optional PriceFetcher instance. When None, check_confirmation()
        always returns a neutral CrossAssetResult with pairs_checked=0.
        This allows the class to exist in the pipeline without breaking
        it when price data is unavailable.

    _lookback_days:
        How many calendar days of daily history to fetch for the secondary
        instrument. A 5-day window captures the current week's movement
        without over-weighting old regime behavior.

    _min_move_pct:
        Moves smaller than this (as an absolute percentage) are treated as
        noise and scored as neutral (0). Prevents false signals from
        instruments that haven't moved meaningfully.
    """

    def __init__(self, price_fetcher=None):
        self._price_fetcher = price_fetcher
        self._pairs: Dict[str, List[CorrelationPair]] = {}
        self._lookback_days: int = 5
        self._min_move_pct: float = 0.5  # percent — moves below this are noise
        self._build_default_pairs()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def check_confirmation(
        self, ticker: str, direction: str
    ) -> CrossAssetResult:
        """
        Check how correlated instruments are moving relative to this signal.

        For each correlation pair involving `ticker`, fetches recent price
        history of the other instrument and computes whether its recent
        movement confirms or contradicts the given direction.

        Args:
            ticker:    The ticker being signaled (e.g. "CL=F", "AAPL").
            direction: Signal direction — "bullish" or "bearish".

        Returns:
            CrossAssetResult with confirmation_score in [-1, 1].
        """
        if self._price_fetcher is None:
            logger.debug(
                "CrossAssetConfirmer: no price_fetcher, returning neutral for %s",
                ticker,
            )
            return CrossAssetResult(confirmation_score=0.0)

        pairs = self.get_pairs_for_ticker(ticker)
        if not pairs:
            logger.debug(
                "CrossAssetConfirmer: no known pairs for %s", ticker
            )
            return CrossAssetResult(confirmation_score=0.0)

        confirming: List[str] = []
        conflicting: List[str] = []
        weighted_score = 0.0
        total_weight = 0.0

        for pair in pairs:
            # Determine which ticker is the "other" one
            other = pair.ticker_b if pair.ticker_a == ticker else pair.ticker_a

            move_pct = self._get_recent_move_pct(other)
            if move_pct is None:
                logger.debug(
                    "CrossAssetConfirmer: no price data for %s (pair with %s)",
                    other,
                    ticker,
                )
                continue

            # Movements too small to be meaningful → treat as neutral
            if abs(move_pct) < self._min_move_pct:
                logger.debug(
                    "CrossAssetConfirmer: %s move %.2f%% below threshold, neutral",
                    other,
                    move_pct,
                )
                # Still count this pair as checked, contributing 0 weight
                total_weight += pair.weight
                weighted_score += 0.0
                continue

            # Determine direction of the other instrument's recent move
            other_direction = "bullish" if move_pct > 0 else "bearish"

            # Score the pair
            pair_score = self._score_pair(
                pair.relationship, direction, other_direction
            )

            total_weight += pair.weight
            weighted_score += pair_score * pair.weight

            # Build human-readable description
            sign_str = f"+{move_pct:.1f}%" if move_pct > 0 else f"{move_pct:.1f}%"
            desc = (
                f"{ticker} → {other} ({pair.relationship}, {other} {sign_str})"
            )
            if pair_score > 0:
                confirming.append(desc)
            elif pair_score < 0:
                conflicting.append(desc)

        if total_weight == 0.0:
            return CrossAssetResult(
                confirmation_score=0.0,
                confirming_pairs=confirming,
                conflicting_pairs=conflicting,
                pairs_checked=0,
            )

        final_score = weighted_score / total_weight
        # Clamp to [-1, 1] for safety
        final_score = max(-1.0, min(1.0, final_score))

        return CrossAssetResult(
            confirmation_score=round(final_score, 4),
            confirming_pairs=confirming,
            conflicting_pairs=conflicting,
            pairs_checked=len(pairs),
        )

    def get_pairs_for_ticker(self, ticker: str) -> List[CorrelationPair]:
        """
        Return all known correlation pairs involving this ticker.

        A pair is returned if ticker appears as either ticker_a or ticker_b.

        Args:
            ticker: Ticker symbol (exact match, case-sensitive).

        Returns:
            List of CorrelationPair objects. Empty if ticker is unknown.
        """
        return list(self._pairs.get(ticker, []))

    # ------------------------------------------------------------------
    # Pair registry
    # ------------------------------------------------------------------

    def _register_pair(self, pair: CorrelationPair) -> None:
        """
        Register a correlation pair bidirectionally.

        Both ticker_a and ticker_b receive an entry in self._pairs so that
        lookups by either symbol work correctly.
        """
        self._pairs.setdefault(pair.ticker_a, []).append(pair)
        # Add the mirrored pair for ticker_b lookups
        mirrored = CorrelationPair(
            ticker_a=pair.ticker_b,
            ticker_b=pair.ticker_a,
            relationship=pair.relationship,
            weight=pair.weight,
        )
        self._pairs.setdefault(pair.ticker_b, []).append(mirrored)

    def _build_default_pairs(self) -> None:
        """
        Register all hardcoded well-established market correlation pairs.

        Sources: decades of academic literature on cross-asset dynamics,
        risk-on/risk-off frameworks, commodity-sector relationships.
        """
        default_pairs = [
            # Large-cap US equity comovement
            CorrelationPair("SPY", "QQQ", relationship="positive", weight=1.0),
            # Gold inverse to US Dollar — gold denominated in USD
            CorrelationPair("GC=F", "DXY", relationship="inverse", weight=1.0),
            # Crude oil and energy sector ETF — direct fundamental link
            CorrelationPair("CL=F", "XLE", relationship="positive", weight=1.0),
            # Risk-off: bonds rally when stocks fall
            CorrelationPair("TLT", "SPY", relationship="inverse", weight=1.0),
            # Fear gauge: VIX rises when SPY falls
            CorrelationPair("VIX", "SPY", relationship="inverse", weight=0.8),
            # S&P futures vs Nasdaq futures — highly correlated
            CorrelationPair("ES=F", "NQ=F", relationship="positive", weight=1.0),
            # S&P futures vs ETF proxy — near-perfect tracking
            CorrelationPair("ES=F", "SPY", relationship="positive", weight=1.0),
            # Gold futures vs gold ETF — near-perfect tracking
            CorrelationPair("GC=F", "GLD", relationship="positive", weight=1.0),
            # Crude futures vs crude ETF — tracks with roll drag
            CorrelationPair("CL=F", "USO", relationship="positive", weight=0.8),
            # Financials sector vs flagship bank — sector leadership
            CorrelationPair("XLF", "JPM", relationship="positive", weight=0.7),
        ]

        for pair in default_pairs:
            self._register_pair(pair)

        logger.debug(
            "CrossAssetConfirmer: registered %d base pairs → %d ticker entries",
            len(default_pairs),
            len(self._pairs),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_recent_move_pct(self, symbol: str) -> Optional[float]:
        """
        Fetch recent price history and compute net percentage move.

        Uses get_daily_history(symbol, self._lookback_days) and computes
        the total move from the first available close to the last:
            (last_close - first_close) / first_close * 100

        Returns None when no data is available or when the price fetcher
        raises an exception.
        """
        try:
            history = self._price_fetcher.get_daily_history(
                symbol, self._lookback_days
            )
        except Exception as exc:
            logger.warning(
                "CrossAssetConfirmer: price fetch failed for %s: %s", symbol, exc
            )
            return None

        if not history:
            return None

        # get_daily_history returns PriceData sorted oldest-first
        first_price = history[0].price
        last_price = history[-1].price

        if first_price == 0:
            return None

        move_pct = (last_price - first_price) / first_price * 100.0
        return move_pct

    @staticmethod
    def _score_pair(
        relationship: str, signal_direction: str, other_direction: str
    ) -> float:
        """
        Compute the pair score given the relationship type and directions.

        Positive relationship:
            Same direction → confirming (+1.0)
            Opposite direction → conflicting (-1.0)

        Inverse relationship:
            Opposite direction → confirming (+1.0)
            Same direction → conflicting (-1.0)

        Returns:
            +1.0 for confirmation, -1.0 for conflict, 0.0 for ambiguous input.
        """
        if relationship == "positive":
            return 1.0 if signal_direction == other_direction else -1.0
        if relationship == "inverse":
            return 1.0 if signal_direction != other_direction else -1.0
        # Unknown relationship type — neutral
        logger.warning(
            "CrossAssetConfirmer: unknown relationship type %r", relationship
        )
        return 0.0
