"""Somatic Anticipation — Gift 9 (Council Gift).

Pre-conscious pattern response via the endocrine system. When multiple
signal domains become active on the same ticker but haven't yet formally
converged, MIDGE's "body" starts responding — releasing DOPAMINE (curiosity)
and adjusting exploration bias toward that ticker.

This is Antonio Damasio's somatic marker hypothesis applied to markets.
The body (endocrine system) responds to partial pattern recognition before
the conscious mind (convergence alerter) has enough data to declare
convergence. This creates a preparatory state — heightened attention,
focused sensing — that makes formal convergence more likely to be captured.

"Knowing things before knowing things."
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Default anticipation parameters
_DEFAULT_THRESHOLD = 2  # domains before somatic response fires
_DEFAULT_WINDOW_HOURS = 48.0  # signals older than this are forgotten

# Persistence path — resolves to data/market/ at the project root
_STATE_DIR = Path(__file__).resolve().parents[3] / "data" / "market"


@dataclass
class TickerState:
    """Accumulated signal state for a single ticker."""

    ticker: str
    domains_active: Set[str] = field(default_factory=set)
    directions: Dict[str, int] = field(default_factory=lambda: Counter())
    total_strength: float = 0.0
    signal_count: int = 0
    first_signal_time: Optional[datetime] = None
    last_signal_time: Optional[datetime] = None


@dataclass
class AnticipationEvent:
    """Somatic anticipation event — pre-convergence pattern detection."""

    ticker: str
    domains_active: int
    domain_names: List[str]
    dominant_direction: str
    anticipation_strength: float  # 0.0 to 1.0
    hormone_response: Dict[str, float]  # hormone_name -> intensity


class SomaticAnticipation:
    """Pre-conscious pattern response via endocrine system.

    Records signals per ticker, and when multiple domains start
    activating on the same ticker within a time window, fires
    somatic responses (hormone releases) BEFORE the convergence
    alerter reaches its min_domains threshold.

    Runs at cadence 25 in sensing_hook — faster than convergence
    (cadence 50) to create a preparatory state.
    """

    def __init__(
        self,
        endocrine_system=None,
        event_bus=None,
        anticipation_threshold: int = _DEFAULT_THRESHOLD,
        response_window_hours: float = _DEFAULT_WINDOW_HOURS,
    ):
        self._endocrine = endocrine_system
        self._bus = event_bus
        self._anticipation_threshold = anticipation_threshold
        self._response_window_hours = response_window_hours

        # Per-ticker signal accumulator
        self._ticker_states: Dict[str, TickerState] = {}

        # Track recently fired events to avoid spamming (ticker -> last_fire_time)
        self._last_fired: Dict[str, float] = {}
        self._fire_cooldown_seconds = 300.0  # 5 min cooldown per ticker

    def record_signal(
        self,
        ticker: str,
        domain: str,
        direction: str,
        strength: float,
        timestamp: datetime,
    ) -> None:
        """Record a signal for somatic anticipation tracking."""
        if not ticker:
            return

        now = datetime.now()
        cutoff = now - timedelta(hours=self._response_window_hours)

        state = self._ticker_states.get(ticker)
        if state is None:
            state = TickerState(ticker=ticker)
            self._ticker_states[ticker] = state

        # If the state is stale (first_signal too old), reset it
        if state.first_signal_time is not None and state.first_signal_time < cutoff:
            state = TickerState(ticker=ticker)
            self._ticker_states[ticker] = state

        state.domains_active.add(domain)
        state.directions[direction] += 1
        state.total_strength += strength
        state.signal_count += 1
        if state.first_signal_time is None:
            state.first_signal_time = timestamp
        state.last_signal_time = timestamp

    def check_anticipation(self) -> List[AnticipationEvent]:
        """Check for pre-convergence patterns and fire somatic responses.

        Returns list of anticipation events for any ticker that has
        crossed the domain threshold within the response window.
        """
        events = []
        now = time.time()
        cutoff = datetime.now() - timedelta(hours=self._response_window_hours)

        for ticker, state in list(self._ticker_states.items()):
            # Skip stale states
            if state.first_signal_time is not None and state.first_signal_time < cutoff:
                del self._ticker_states[ticker]
                continue

            # Check domain threshold
            active_count = len(state.domains_active)
            if active_count < self._anticipation_threshold:
                continue

            # Cooldown check
            last = self._last_fired.get(ticker, 0.0)
            if (now - last) < self._fire_cooldown_seconds:
                continue

            # Determine dominant direction
            if state.directions:
                dominant = max(state.directions, key=state.directions.get)
                dominant_count = state.directions[dominant]
                total_directional = sum(state.directions.values())
            else:
                dominant = "neutral"
                dominant_count = 0
                total_directional = 0

            # Compute anticipation strength
            # Higher with more domains and stronger signal agreement
            domain_factor = min(1.0, active_count / 4.0)
            agreement = dominant_count / total_directional if total_directional > 0 else 0.0
            avg_strength = state.total_strength / state.signal_count if state.signal_count > 0 else 0.0
            anticipation_strength = domain_factor * 0.4 + agreement * 0.3 + avg_strength * 0.3

            # Compute hormone response
            hormone_response = self._compute_somatic_response(
                active_count, dominant, agreement,
            )

            event = AnticipationEvent(
                ticker=ticker,
                domains_active=active_count,
                domain_names=sorted(state.domains_active),
                dominant_direction=dominant,
                anticipation_strength=round(anticipation_strength, 3),
                hormone_response=hormone_response,
            )
            events.append(event)
            self._last_fired[ticker] = now

            # Release hormones if endocrine system is available
            self._release_hormones(hormone_response)

            # Publish to EventBus
            if self._bus is not None:
                try:
                    from mae_core.market.channels import CH_SOMATIC_ANTICIPATION
                    self._bus.publish(CH_SOMATIC_ANTICIPATION, {
                        "ticker": ticker,
                        "domains_active": active_count,
                        "dominant_direction": dominant,
                        "anticipation_strength": anticipation_strength,
                        "hormones": hormone_response,
                    })
                except Exception:
                    logger.debug("Somatic anticipation publish failed", exc_info=True)

        return events

    def _compute_somatic_response(
        self,
        domain_count: int,
        dominant_direction: str,
        agreement: float,
    ) -> Dict[str, float]:
        """Map pattern state to hormone cocktail.

        2 domains same direction → DOPAMINE 0.15 (curiosity/attention)
        3+ domains same direction → DOPAMINE 0.25 + ADRENALINE 0.10
        Mixed directions (low agreement) → CORTISOL 0.15 (caution)
        """
        hormones: Dict[str, float] = {}

        if agreement > 0.6:
            # Mostly aligned — curiosity and attention
            if domain_count >= 3:
                hormones["DOPAMINE"] = 0.25
                hormones["ADRENALINE"] = 0.10
            else:
                hormones["DOPAMINE"] = 0.15
        elif agreement < 0.4:
            # Mixed signals — caution and conflict detection
            hormones["CORTISOL"] = 0.15
        else:
            # Moderate agreement — mild curiosity
            hormones["DOPAMINE"] = 0.10

        return hormones

    def _release_hormones(self, hormone_response: Dict[str, float]) -> None:
        """Release hormones via endocrine system if available."""
        if self._endocrine is None:
            return

        try:
            from mae_core.coordination.endocrine_system import HormoneType

            hormone_map = {
                "DOPAMINE": HormoneType.DOPAMINE,
                "ADRENALINE": HormoneType.ADRENALINE,
                "CORTISOL": HormoneType.CORTISOL,
            }

            for name, intensity in hormone_response.items():
                ht = hormone_map.get(name)
                if ht is not None and intensity > 0:
                    self._endocrine.release_hormone(
                        ht, intensity, f"somatic_anticipation_{name.lower()}"
                    )
        except Exception:
            logger.debug("Somatic hormone release failed", exc_info=True)

    def get_statistics(self) -> dict:
        """Return somatic anticipation stats."""
        active_tickers = [
            t for t, s in self._ticker_states.items()
            if len(s.domains_active) >= self._anticipation_threshold
        ]
        return {
            "tracked_tickers": len(self._ticker_states),
            "active_anticipations": len(active_tickers),
            "active_tickers": active_tickers[:5],
            "anticipation_threshold": self._anticipation_threshold,
            "window_hours": self._response_window_hours,
        }
