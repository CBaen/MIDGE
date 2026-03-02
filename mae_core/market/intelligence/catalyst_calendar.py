#!/usr/bin/env python3
"""
catalyst_calendar.py - Upcoming Catalyst Tracker

Tracks known upcoming catalysts (earnings, FOMC, dividends, FDA events) and
provides confidence modifiers for signal evaluation.

The core insight: signals near a known catalyst carry MORE information, not less.
Insider buying 3 days before earnings = 1.5x multiplier because they know something.
No upcoming catalyst = mild 0.9x discount (no catalytic context for the move).

FOMC dates for 2026 are hardcoded from the publicly announced Federal Reserve
meeting schedule. These affect all tickers regardless of their earnings schedules.

Usage:
    from mae_core.market.intelligence.catalyst_calendar import CatalystCalendar

    cal = CatalystCalendar(finnhub_client=finnhub_client)
    cal.refresh(tickers=["AAPL", "SPY"])

    modifier = cal.compute_catalyst_modifier("AAPL", signal_domain="insider")
    # → 1.5 if earnings within 14 days and domain is "insider"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Catalyst:
    """A single upcoming market-moving event for a ticker."""
    ticker: str
    event_type: str          # "earnings", "fomc", "dividend", "fda"
    date: datetime
    estimated_impact: float  # 0–1 estimated volatility impact
    source: str = ""         # "finnhub", "hardcoded"


# ---------------------------------------------------------------------------
# Modifier thresholds
# ---------------------------------------------------------------------------

_NEAR_DAYS = 3      # catalyst within 0-3 days → amplify most
_MID_DAYS = 7       # catalyst within 3-7 days → amplify less
_FAR_DAYS = 14      # catalyst within 7-14 days → neutral
# > 14 days or none → mild discount

_MOD_NEAR = 1.3
_MOD_MID = 1.1
_MOD_FAR = 1.0
_MOD_NONE = 0.9
_MOD_INSIDER_SPECIAL = 1.5   # insider domain + catalyst < 14 days

# FOMC dates affect all tickers regardless of individual calendars
_FOMC_IMPACT = 0.7           # High estimated impact (FOMC moves indices)

# Domains that get the insider-special multiplier
_INSIDER_DOMAINS = {"insider", "sec_form4", "congressional", "politician"}


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

class CatalystCalendar:
    """
    Tracks upcoming catalysts and returns confidence modifiers.

    Requires no dependencies to instantiate — finnhub_client is optional.
    When no finnhub_client is provided, only FOMC dates are tracked (free,
    hardcoded from the publicly announced Federal Reserve schedule).

    Thread safety: refresh() and compute_catalyst_modifier() share
    self._upcoming. They are not explicitly locked; refresh() replaces the
    dict atomically. For production multi-threaded use, callers should
    serialize refresh() calls.
    """

    def __init__(self, finnhub_client=None):
        self._finnhub = finnhub_client
        self._upcoming: Dict[str, List[Catalyst]] = {}  # ticker → sorted by date
        self._cache_date: str = ""

        # 2026 FOMC meeting dates (announced by the Federal Reserve).
        # Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
        self._fomc_dates: List[datetime] = [
            datetime(2026, 1, 28),
            datetime(2026, 3, 18),
            datetime(2026, 5, 6),
            datetime(2026, 6, 17),
            datetime(2026, 7, 29),
            datetime(2026, 9, 16),
            datetime(2026, 11, 4),
            datetime(2026, 12, 16),
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def refresh(self, tickers: Optional[List[str]] = None) -> None:
        """
        Pull latest catalyst data and cache for today.

        FOMC dates are always included for every ticker. Finnhub earnings
        are included when a finnhub_client is available.

        Args:
            tickers: Ticker symbols to fetch earnings for. None fetches
                     only FOMC dates (still useful for index instruments).
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if self._cache_date == today and self._upcoming:
            logger.debug("CatalystCalendar: cache hit for %s", today)
            return

        new_upcoming: Dict[str, List[Catalyst]] = {}
        tickers = tickers or []

        # 1. Load FOMC dates — affect all tickers
        fomc_catalysts = self._build_fomc_catalysts()

        # 2. Fetch earnings from Finnhub if available
        earnings_by_ticker: Dict[str, List[Catalyst]] = {}
        if self._finnhub is not None and tickers:
            earnings_by_ticker = self._fetch_finnhub_earnings(tickers)

        # 3. Merge: each ticker gets its earnings + all FOMC dates
        all_tickers = set(tickers) | {"__fomc__"}
        for ticker in all_tickers:
            combined: List[Catalyst] = list(fomc_catalysts)
            if ticker in earnings_by_ticker:
                combined.extend(earnings_by_ticker[ticker])
            combined.sort(key=lambda c: c.date)
            new_upcoming[ticker] = combined

        self._upcoming = new_upcoming
        self._cache_date = today
        logger.info(
            "CatalystCalendar refreshed: %d tickers, %d FOMC dates",
            len(tickers),
            len(fomc_catalysts),
        )

    def get_upcoming(self, ticker: str, days: int = 14) -> List[Catalyst]:
        """
        Return all catalysts for this ticker within the next N days.

        FOMC catalysts are always included when within the window.
        Results are sorted by date ascending.

        Args:
            ticker: Ticker symbol (e.g. "AAPL", "ES=F").
            days:   Look-ahead window in calendar days.

        Returns:
            List of Catalyst objects sorted by date ascending.
        """
        now = datetime.now()
        cutoff = now + timedelta(days=days)

        # Collect from ticker-specific list + FOMC fallback
        sources: List[Catalyst] = []
        if ticker in self._upcoming:
            sources.extend(self._upcoming[ticker])
        elif "__fomc__" in self._upcoming:
            # Ticker not refreshed; still provide FOMC context
            sources.extend(self._upcoming["__fomc__"])

        upcoming = [c for c in sources if now <= c.date <= cutoff]
        # Deduplicate by (event_type, date) in case FOMC appears in both lists
        seen: set = set()
        deduped: List[Catalyst] = []
        for c in upcoming:
            key = (c.event_type, c.date.date())
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        deduped.sort(key=lambda c: c.date)
        return deduped

    def compute_catalyst_modifier(
        self, ticker: str, signal_domain: str = ""
    ) -> float:
        """
        Return a confidence multiplier based on upcoming catalyst proximity.

        Modifiers:
            0–3 days   → 1.3x  (imminent: signal has urgency)
            3–7 days   → 1.1x  (near-term: elevated relevance)
            7–14 days  → 1.0x  (neutral window)
            > 14 days  → 0.9x  (mild discount: no catalyst context)
            insider domain + catalyst < 14 days → 1.5x special case

        The insider-special case supersedes all other tiers: smart-money
        insiders buying ahead of a known event is an extremely high-signal
        combination.

        Args:
            ticker:        Ticker symbol.
            signal_domain: Domain of the originating signal (e.g. "insider").

        Returns:
            Float multiplier in [0.5, 1.5].
        """
        days_away = self.days_to_nearest_catalyst(ticker)

        # No catalyst context
        if days_away is None:
            return _MOD_NONE

        # Insider + any catalyst within 14 days → special multiplier
        domain_normalized = (signal_domain or "").lower().strip()
        if domain_normalized in _INSIDER_DOMAINS and days_away <= _FAR_DAYS:
            logger.debug(
                "CatalystCalendar: insider+catalyst modifier for %s (%.1f days)",
                ticker,
                days_away,
            )
            return _MOD_INSIDER_SPECIAL

        # Standard proximity tiers
        if days_away <= _NEAR_DAYS:
            return _MOD_NEAR
        if days_away <= _MID_DAYS:
            return _MOD_MID
        if days_away <= _FAR_DAYS:
            return _MOD_FAR
        return _MOD_NONE

    def days_to_nearest_catalyst(self, ticker: str) -> Optional[float]:
        """
        Return the number of days until the nearest upcoming catalyst.

        Args:
            ticker: Ticker symbol.

        Returns:
            Float days (can be fractional for intraday precision) or None
            when no upcoming catalyst is known.
        """
        now = datetime.now()

        # Collect catalyst candidates: ticker-specific + FOMC
        sources: List[Catalyst] = []
        if ticker in self._upcoming:
            sources.extend(self._upcoming[ticker])
        if "__fomc__" in self._upcoming:
            sources.extend(self._upcoming["__fomc__"])

        future = [c for c in sources if c.date >= now]
        if not future:
            return None

        nearest = min(future, key=lambda c: c.date)
        delta = (nearest.date - now).total_seconds() / 86400.0
        return max(0.0, delta)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_fomc_catalysts(self) -> List[Catalyst]:
        """Build Catalyst objects for future FOMC dates only."""
        now = datetime.now()
        catalysts: List[Catalyst] = []
        for dt in self._fomc_dates:
            if dt >= now:
                catalysts.append(
                    Catalyst(
                        ticker="__fomc__",
                        event_type="fomc",
                        date=dt,
                        estimated_impact=_FOMC_IMPACT,
                        source="hardcoded",
                    )
                )
        return catalysts

    def _fetch_finnhub_earnings(
        self, tickers: List[str]
    ) -> Dict[str, List[Catalyst]]:
        """
        Fetch upcoming earnings from Finnhub and convert to Catalyst objects.

        FinnhubClient.get_upcoming_earnings() returns a list of EarningsEvent
        objects with .symbol and .date (YYYY-MM-DD string) attributes.

        Returns dict mapping ticker → list of Catalysts.
        """
        by_ticker: Dict[str, List[Catalyst]] = {}
        try:
            events = self._finnhub.get_upcoming_earnings(days=30)
        except Exception as exc:
            logger.warning("CatalystCalendar: Finnhub earnings fetch failed: %s", exc)
            return by_ticker

        for event in events:
            symbol = getattr(event, "symbol", None)
            if symbol is None:
                continue
            symbol = symbol.upper()
            if symbol not in [t.upper() for t in tickers]:
                continue

            # Parse date string to datetime
            date_str = getattr(event, "date", None)
            if not date_str:
                continue
            try:
                event_dt = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                logger.debug(
                    "CatalystCalendar: could not parse date %r for %s",
                    date_str,
                    symbol,
                )
                continue

            catalyst = Catalyst(
                ticker=symbol,
                event_type="earnings",
                date=event_dt,
                estimated_impact=0.6,   # Earnings are high-impact events
                source="finnhub",
            )
            by_ticker.setdefault(symbol, []).append(catalyst)

        return by_ticker
