"""Market Clock — Time-aware source availability for MIDGE's sensing pipeline.

Knows when US equity markets, futures markets, and crypto markets are open.
Used by MarketSensingHook to filter source rotation to available sources only.

Biological analogy: Circadian rhythm. The organism doesn't try to see in the
dark — it knows when its senses are useful and rests the others.

Session definitions (all times US/Eastern):
  - pre_market:  4:00 AM  – 9:30 AM  (business days)
  - market:      9:30 AM  – 4:00 PM  (business days, ex-holidays)
  - after_hours: 4:00 PM  – 8:00 PM  (business days)
  - overnight:   8:00 PM  – 4:00 AM  (business days with futures open, or nights)
  - weekend:     Saturday 12:00 AM – Sunday 6:00 PM ET

Futures session (CME Globex):
  Opens:  Sunday 6:00 PM ET
  Closes: Friday 5:00 PM ET
  Daily maintenance halt: 5:00 PM – 6:00 PM ET (Mon–Thu only)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

logger = logging.getLogger("midge.market.clock")

# ---------------------------------------------------------------------------
# Source availability categories
# These mirror the SOURCE_ROTATION list in sensing_hook.py. Update here if
# new sources are added to the rotation.
# ---------------------------------------------------------------------------

ALWAYS_AVAILABLE: Set[str] = {
    "sec_form4",
    "sec_form8k",
    "congressional",
    "senate",
    "hiring",
    "usa_spending",
    "sam_gov_and_prices",
    "social_sentiment",
    "sec_efts",
    "fred_macro",
    "stocktwits",
    "google_trends",
    "finnhub_extras",
    "crypto_prices",       # CoinGecko API serves data 24/7
    "crypto_exchange",     # CoinCap API serves data 24/7
    "openinsider",         # Website serves cached SEC data anytime
    "economic_calendar",   # Hardcoded calendar, always available
    "eia_energy",          # EIA API serves historical data anytime
    "congress_legislation",  # Congress.gov API serves data anytime
    "binance_funding",       # Binance funding rates — crypto runs 24/7
    "crypto_fear_greed",     # Fear & Greed index — always available
    "kalshi_market",         # Kalshi prediction markets — always available
    "yahoo_rss",             # Yahoo Finance RSS feeds — always available
    "fred_yields",           # FRED yield curve data — always available
    "usda_agriculture",      # USDA agricultural data — always available
    "news_headlines",        # News headline feeds — always available
    "edgar_xbrl",            # SEC EDGAR XBRL data — always available
    "cboe_options",          # CBOE options data — always available
    "social_text",           # Social text analysis — always available
}

# Requires CME Globex futures session (Sun 6PM – Fri 5PM ET, daily halt 5–6 PM Mon–Thu)
FUTURES_HOURS: Set[str] = {
    "session_sweep",
    "order_flow",
    "ta_indicators",
}

# Requires US equity market hours (9:30 AM – 4:00 PM ET, Mon–Fri, ex-holidays)
MARKET_HOURS: Set[str] = {
    "finnhub",
    "fractal_resonance",
    "vix_structure",
    "finviz",              # FinViz data is most useful during market hours
    "massive_snapshot",    # Massive/Polygon EOD data — market hours for freshness
}

# Available on business days only (T+2 delay / weekly cadence don't change this)
PERIODIC: Set[str] = {
    "cot_positioning",
    "finra_short",
    "institutional_13f",   # SEC EDGAR serves data on business days
}

# ---------------------------------------------------------------------------
# US market holidays 2026
# Source: NYSE / CME holiday schedule.
# Extend this set each year — it's the only maintenance required.
# ---------------------------------------------------------------------------

_HOLIDAYS_2026: Set[date] = {
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day (observed — July 4 is Saturday)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas Day
}

_HOLIDAYS_2027: Set[date] = {
    date(2027, 1, 1),   # New Year's Day
    date(2027, 1, 18),  # MLK Day
    date(2027, 2, 15),  # Presidents' Day
    date(2027, 3, 26),  # Good Friday
    date(2027, 5, 31),  # Memorial Day
    date(2027, 6, 18),  # Juneteenth (observed — June 19 is Saturday)
    date(2027, 7, 5),   # Independence Day (observed — July 4 is Sunday)
    date(2027, 9, 6),   # Labor Day
    date(2027, 11, 25), # Thanksgiving
    date(2027, 12, 24), # Christmas (observed — Dec 25 is Saturday)
}

_ALL_HOLIDAYS: Set[date] = _HOLIDAYS_2026 | _HOLIDAYS_2027

# Session boundary times (naive, applied in ET)
_EQUITY_OPEN = time(9, 30)
_EQUITY_CLOSE = time(16, 0)
_PRE_MARKET_START = time(4, 0)
_AFTER_HOURS_END = time(20, 0)   # 8:00 PM ET

# Futures boundaries
_FUTURES_SUNDAY_OPEN = time(18, 0)   # Sunday 6:00 PM ET
_FUTURES_FRIDAY_CLOSE = time(17, 0)  # Friday 5:00 PM ET
_FUTURES_DAILY_HALT_START = time(17, 0)  # 5:00 PM ET (Mon–Thu)
_FUTURES_DAILY_HALT_END = time(18, 0)    # 6:00 PM ET (Mon–Thu)

_ET = ZoneInfo("US/Eastern")


class MarketClock:
    """Track market sessions and source availability.

    All public methods accept an optional ``at`` parameter so callers can
    check availability at an arbitrary time (critical for unit tests).
    If ``at`` is None the current wall-clock time in US/Eastern is used.
    If ``at`` is a naive datetime it is assumed to be in US/Eastern.
    """

    def __init__(self, timezone: str = "US/Eastern"):
        self._tz = ZoneInfo(timezone)
        self._holidays = _ALL_HOLIDAYS

    # ------------------------------------------------------------------
    # Core clock
    # ------------------------------------------------------------------

    def now(self) -> datetime:
        """Current time in Eastern timezone."""
        return datetime.now(self._tz)

    def _resolve(self, at: Optional[datetime]) -> datetime:
        """Return a timezone-aware ET datetime from the optional ``at`` arg."""
        if at is None:
            return self.now()
        if at.tzinfo is None:
            # Naive → assume ET
            return at.replace(tzinfo=self._tz)
        # Already aware — convert to ET for comparison
        return at.astimezone(self._tz)

    # ------------------------------------------------------------------
    # Holiday / business-day helpers
    # ------------------------------------------------------------------

    def is_holiday(self, at: Optional[datetime] = None) -> bool:
        """Return True if the given date is a US market holiday."""
        dt = self._resolve(at)
        return dt.date() in self._holidays

    def is_weekend(self, at: Optional[datetime] = None) -> bool:
        """Return True if the given time falls on a Saturday or Sunday."""
        dt = self._resolve(at)
        return dt.weekday() >= 5  # 5=Saturday, 6=Sunday

    def is_business_day(self, at: Optional[datetime] = None) -> bool:
        """Return True if today is a trading business day (Mon–Fri, ex-holidays)."""
        dt = self._resolve(at)
        return not self.is_weekend(dt) and not self.is_holiday(dt)

    # ------------------------------------------------------------------
    # Equity market
    # ------------------------------------------------------------------

    def is_equity_market_open(self, at: Optional[datetime] = None) -> bool:
        """Return True if US equity markets are currently open.

        Open window: 9:30 AM – 4:00 PM ET, Monday–Friday, excluding holidays.
        """
        dt = self._resolve(at)
        if not self.is_business_day(dt):
            return False
        t = dt.time().replace(tzinfo=None)
        return _EQUITY_OPEN <= t < _EQUITY_CLOSE

    # ------------------------------------------------------------------
    # Futures market
    # ------------------------------------------------------------------

    def is_futures_open(self, at: Optional[datetime] = None) -> bool:
        """Return True if CME Globex futures are currently trading.

        Session: Sunday 6:00 PM ET through Friday 5:00 PM ET.
        Daily maintenance halt: 5:00 PM – 6:00 PM ET, Monday–Thursday only.
        Futures are closed all weekend from Friday 5 PM to Sunday 6 PM.
        """
        dt = self._resolve(at)
        weekday = dt.weekday()  # 0=Monday … 6=Sunday
        t = dt.time().replace(tzinfo=None)

        # Saturday: always closed
        if weekday == 5:
            return False

        # Sunday: open only from 6:00 PM onward
        if weekday == 6:
            return t >= _FUTURES_SUNDAY_OPEN

        # Friday: open until 5:00 PM, then closed for the weekend
        if weekday == 4:
            return t < _FUTURES_FRIDAY_CLOSE

        # Monday–Thursday: open all day except the 5–6 PM maintenance halt
        in_halt = _FUTURES_DAILY_HALT_START <= t < _FUTURES_DAILY_HALT_END
        return not in_halt

    # ------------------------------------------------------------------
    # Session name
    # ------------------------------------------------------------------

    def get_session(self, at: Optional[datetime] = None) -> str:
        """Return the current named market session.

        Possible values:
          "pre_market"  — 4:00 AM to 9:30 AM ET, business day
          "market"      — 9:30 AM to 4:00 PM ET, business day (ex-holidays)
          "after_hours" — 4:00 PM to 8:00 PM ET, business day
          "overnight"   — 8:00 PM to 4:00 AM ET on a business day,
                          OR any hour on a non-holiday weeknight when
                          futures are open
          "weekend"     — Saturday or Sunday before 6:00 PM ET Sunday
        """
        dt = self._resolve(at)
        weekday = dt.weekday()
        t = dt.time().replace(tzinfo=None)

        # Weekend window: Saturday midnight through Sunday 6 PM
        if weekday == 5:
            return "weekend"
        if weekday == 6 and t < _FUTURES_SUNDAY_OPEN:
            return "weekend"

        if self.is_business_day(dt):
            if _PRE_MARKET_START <= t < _EQUITY_OPEN:
                return "pre_market"
            if _EQUITY_OPEN <= t < _EQUITY_CLOSE:
                return "market"
            if _EQUITY_CLOSE <= t < _AFTER_HOURS_END:
                return "after_hours"

        # Everything else where futures are open (or just overnight non-business)
        return "overnight"

    # ------------------------------------------------------------------
    # Source availability
    # ------------------------------------------------------------------

    def source_available(self, source_name: str, at: Optional[datetime] = None) -> bool:
        """Return True if the named source can return useful data right now.

        Unknown sources default to True (fail-open is safer than fail-closed:
        a new source that isn't categorised yet should not be silently dropped).
        """
        if source_name in ALWAYS_AVAILABLE:
            return True
        if source_name in FUTURES_HOURS:
            return self.is_futures_open(at)
        if source_name in MARKET_HOURS:
            return self.is_equity_market_open(at)
        if source_name in PERIODIC:
            return self.is_business_day(at)
        # Unknown source — assume available (log once so operators can categorise it)
        logger.debug(
            "MarketClock: unknown source '%s' — defaulting to available. "
            "Add it to one of the availability sets in market_clock.py.",
            source_name,
        )
        return True

    def get_available_sources(
        self,
        source_list: Optional[List[str]] = None,
        at: Optional[datetime] = None,
    ) -> List[str]:
        """Filter a source list to those currently available.

        If ``source_list`` is None, all known sources are used as the default.
        The returned list preserves the original order.
        """
        if source_list is None:
            source_list = sorted(
                ALWAYS_AVAILABLE | FUTURES_HOURS | MARKET_HOURS | PERIODIC
            )
        return [s for s in source_list if self.source_available(s, at)]

    def get_unavailable_sources(
        self,
        source_list: Optional[List[str]] = None,
        at: Optional[datetime] = None,
    ) -> List[str]:
        """Return sources from ``source_list`` that are NOT available right now.

        Useful for logging / debugging when sources are skipped.
        """
        if source_list is None:
            source_list = sorted(
                ALWAYS_AVAILABLE | FUTURES_HOURS | MARKET_HOURS | PERIODIC
            )
        return [s for s in source_list if not self.source_available(s, at)]

    # ------------------------------------------------------------------
    # Look-ahead helpers
    # ------------------------------------------------------------------

    def next_equity_open(self, at: Optional[datetime] = None) -> datetime:
        """Return the datetime of the next equity session open (9:30 AM ET).

        If markets are already open ``at`` the given time, returns the NEXT
        open after today's close — not the current session.
        """
        dt = self._resolve(at)

        # Start searching from the next calendar day's 9:30, or today's 9:30
        # if we haven't reached it yet (and today is a business day).
        candidate = dt.replace(
            hour=_EQUITY_OPEN.hour,
            minute=_EQUITY_OPEN.minute,
            second=0,
            microsecond=0,
        )

        # If we're still before today's open AND today is a business day, that's
        # our answer.
        if candidate > dt and self.is_business_day(candidate):
            return candidate

        # Otherwise walk forward one day at a time until we land on a business day.
        candidate = candidate + timedelta(days=1)
        for _ in range(14):  # Safety: never loop more than two weeks
            if self.is_business_day(candidate):
                return candidate
            candidate += timedelta(days=1)

        # Should never reach here (no 14-day holiday stretch exists)
        raise RuntimeError(
            "MarketClock.next_equity_open: could not find a business day "
            "within 14 days of %s" % dt.isoformat()
        )

    def next_futures_open(self, at: Optional[datetime] = None) -> datetime:
        """Return the datetime of the next futures session open.

        If futures are already open, returns the open after the next closure
        (i.e., next Sunday 6 PM following the current session end).
        If futures are closed and we are in the Friday-close → Sunday-open
        gap, returns the upcoming Sunday 6 PM ET.
        """
        dt = self._resolve(at)

        if self.is_futures_open(dt):
            # Already open — compute when this session ends, then find next open
            session_end = self._current_futures_close(dt)
            return self.next_futures_open(session_end + timedelta(minutes=1))

        weekday = dt.weekday()
        t = dt.time().replace(tzinfo=None)

        # In the daily maintenance halt (Mon–Thu 5–6 PM): open resumes at 6 PM today
        if weekday < 4 and _FUTURES_DAILY_HALT_START <= t < _FUTURES_DAILY_HALT_END:
            return dt.replace(
                hour=_FUTURES_DAILY_HALT_END.hour,
                minute=_FUTURES_DAILY_HALT_END.minute,
                second=0,
                microsecond=0,
            )

        # Weekend gap: walk forward to Sunday
        days_until_sunday = (6 - weekday) % 7
        if days_until_sunday == 0 and t >= _FUTURES_SUNDAY_OPEN:
            # Already past Sunday open — this shouldn't happen (is_futures_open
            # would have returned True), but handle defensively
            days_until_sunday = 7
        target = (dt + timedelta(days=days_until_sunday)).replace(
            hour=_FUTURES_SUNDAY_OPEN.hour,
            minute=_FUTURES_SUNDAY_OPEN.minute,
            second=0,
            microsecond=0,
        )
        return target

    def _current_futures_close(self, at: datetime) -> datetime:
        """Return when the current futures session closes (internal helper).

        Assumes futures ARE open at ``at``.
        """
        dt = self._resolve(at)
        weekday = dt.weekday()

        # Friday: closes at 5 PM today
        if weekday == 4:
            return dt.replace(
                hour=_FUTURES_FRIDAY_CLOSE.hour,
                minute=_FUTURES_FRIDAY_CLOSE.minute,
                second=0,
                microsecond=0,
            )

        # Mon–Thu: next daily halt at 5 PM today
        if weekday < 4:
            halt = dt.replace(
                hour=_FUTURES_DAILY_HALT_START.hour,
                minute=_FUTURES_DAILY_HALT_START.minute,
                second=0,
                microsecond=0,
            )
            if dt < halt:
                return halt
            # Past today's halt — next close is Friday 5 PM
            days_to_friday = (4 - weekday) % 7
            return (dt + timedelta(days=days_to_friday)).replace(
                hour=_FUTURES_FRIDAY_CLOSE.hour,
                minute=_FUTURES_FRIDAY_CLOSE.minute,
                second=0,
                microsecond=0,
            )

        # Sunday: session runs until Monday's 5 PM halt
        # (weekday == 6, futures open after 6 PM)
        return (dt + timedelta(days=1)).replace(
            hour=_FUTURES_DAILY_HALT_START.hour,
            minute=_FUTURES_DAILY_HALT_START.minute,
            second=0,
            microsecond=0,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self, at: Optional[datetime] = None) -> Dict[str, object]:
        """Return a human-readable status dict for logging and monitoring."""
        dt = self._resolve(at)
        return {
            "at": dt.isoformat(),
            "session": self.get_session(dt),
            "equity_open": self.is_equity_market_open(dt),
            "futures_open": self.is_futures_open(dt),
            "business_day": self.is_business_day(dt),
            "is_holiday": self.is_holiday(dt),
            "is_weekend": self.is_weekend(dt),
            "next_equity_open": self.next_equity_open(dt).isoformat()
            if not self.is_equity_market_open(dt)
            else None,
        }
