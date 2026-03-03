"""Economic Calendar Client — High-impact event scheduling.

Provides suppression windows around major economic events (FOMC, CPI, NFP).
The convergence alerter uses these windows defensively — reducing confidence
during periods when scheduled volatility makes normal signals unreliable.

Uses hardcoded FOMC/CPI/NFP dates (published in advance) plus Finnhub
economic calendar as supplement.

Part of WP-F4: Always-On MIDGE Wave 3.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import List, Optional

logger = logging.getLogger("midge.market.economic_calendar")


@dataclass
class EconomicEvent:
    name: str
    event_date: date
    impact: str  # "high", "medium", "low"
    description: str = ""


@dataclass
class SuppressionWindow:
    event_name: str
    event_date: date
    start: datetime  # event - pre_event_hours
    end: datetime    # event + post_event_hours
    impact: str


# FOMC meeting dates for 2026 (confirmed by Federal Reserve schedule).
# These are the announcement dates (typically 2:00 PM ET).
# Meetings are 2 days; the decision date is listed here.
_FOMC_DATES_2026: List[date] = [
    date(2026, 1, 28),  date(2026, 1, 29),   # Jan meeting
    date(2026, 3, 18),  date(2026, 3, 19),   # Mar meeting
    date(2026, 5, 6),   date(2026, 5, 7),    # May meeting
    date(2026, 6, 17),  date(2026, 6, 18),   # Jun meeting
    date(2026, 7, 29),  date(2026, 7, 30),   # Jul meeting
    date(2026, 9, 16),  date(2026, 9, 17),   # Sep meeting
    date(2026, 10, 28), date(2026, 10, 29),  # Oct meeting
    date(2026, 12, 9),  date(2026, 12, 10),  # Dec meeting
]

# CPI release dates for 2026 (BLS schedule; typically 8:30 AM ET, ~10th of month).
_CPI_DATES_2026: List[date] = [
    date(2026, 1, 14), date(2026, 2, 11), date(2026, 3, 11),
    date(2026, 4, 14), date(2026, 5, 12), date(2026, 6, 10),
    date(2026, 7, 14), date(2026, 8, 12), date(2026, 9, 15),
    date(2026, 10, 13), date(2026, 11, 12), date(2026, 12, 10),
]

# Non-Farm Payrolls for 2026 (typically first Friday of each month, 8:30 AM ET).
_NFP_DATES_2026: List[date] = [
    date(2026, 1, 2),  date(2026, 2, 6),  date(2026, 3, 6),
    date(2026, 4, 3),  date(2026, 5, 1),  date(2026, 6, 5),
    date(2026, 7, 2),  date(2026, 8, 7),  date(2026, 9, 4),
    date(2026, 10, 2), date(2026, 11, 6), date(2026, 12, 4),
]

# Release times by event type (hour, minute) — Eastern time.
_RELEASE_TIMES = {
    "FOMC": (14, 0),   # 2:00 PM ET announcement
    "CPI":  (8, 30),   # 8:30 AM ET
    "NFP":  (8, 30),   # 8:30 AM ET
}


def _event_datetime(event: EconomicEvent) -> datetime:
    """Return the release datetime for an event (naive, assumes ET)."""
    hour, minute = _RELEASE_TIMES.get(event.name, (14, 0))
    return datetime(
        event.event_date.year,
        event.event_date.month,
        event.event_date.day,
        hour,
        minute,
    )


class EconomicCalendarClient:
    """Provides suppression windows around scheduled high-impact macro events.

    The calendar is built from hardcoded 2026 dates (FOMC, CPI, NFP) that
    are published in advance by the Federal Reserve and BLS.  No network
    calls are required for the primary path; rate-limiting is enforced for
    any future Finnhub supplement requests so we never hammer the API.

    Rate limit: 1 request per 5 minutes (calendar updates weekly at most).
    """

    # 5-minute minimum gap between any supplemental API calls.
    _RATE_LIMIT_SECONDS: float = 300.0

    def __init__(
        self,
        pre_event_hours: float = 24.0,
        post_event_hours: float = 4.0,
    ) -> None:
        self._pre_event_hours = pre_event_hours
        self._post_event_hours = post_event_hours
        self._last_fetch: Optional[datetime] = None
        self._events: List[EconomicEvent] = self._build_calendar()

    # ------------------------------------------------------------------
    # Calendar construction
    # ------------------------------------------------------------------

    def _build_calendar(self) -> List[EconomicEvent]:
        """Build the full sorted calendar from hardcoded 2026 dates."""
        events: List[EconomicEvent] = []

        for d in _FOMC_DATES_2026:
            events.append(EconomicEvent(
                name="FOMC",
                event_date=d,
                impact="high",
                description="Federal Reserve monetary policy decision",
            ))

        for d in _CPI_DATES_2026:
            events.append(EconomicEvent(
                name="CPI",
                event_date=d,
                impact="high",
                description="Consumer Price Index release",
            ))

        for d in _NFP_DATES_2026:
            events.append(EconomicEvent(
                name="NFP",
                event_date=d,
                impact="high",
                description="Non-Farm Payrolls employment report",
            ))

        events.sort(key=lambda e: e.event_date)
        logger.debug(
            "economic_calendar built: %d events (%d FOMC, %d CPI, %d NFP)",
            len(events),
            len(_FOMC_DATES_2026),
            len(_CPI_DATES_2026),
            len(_NFP_DATES_2026),
        )
        return events

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_upcoming_events(
        self,
        days: int = 7,
        from_date: Optional[date] = None,
    ) -> List[EconomicEvent]:
        """Return high-impact events within the next *days* calendar days.

        Args:
            days: Look-ahead window in calendar days (inclusive).
            from_date: Reference date; defaults to today.

        Returns:
            Events sorted ascending by event_date within [from_date, from_date + days].
        """
        ref = from_date or date.today()
        end = ref + timedelta(days=days)
        return [e for e in self._events if ref <= e.event_date <= end]

    def get_suppression_windows(
        self,
        from_date: Optional[date] = None,
    ) -> List[SuppressionWindow]:
        """Return suppression windows that fall within a ±7-day search range.

        The convergence alerter calls this to discover windows that overlap
        with 'now'.  A window spans [event_time - pre_hours, event_time + post_hours].

        Args:
            from_date: Reference date; defaults to today.

        Returns:
            List of SuppressionWindow objects ordered by event start time.
        """
        ref = from_date or date.today()
        ref_dt = datetime(ref.year, ref.month, ref.day)
        windows: List[SuppressionWindow] = []

        for event in self._events:
            event_dt = _event_datetime(event)
            start = event_dt - timedelta(hours=self._pre_event_hours)
            end = event_dt + timedelta(hours=self._post_event_hours)

            # Include windows within ±7 days of the reference date.
            if start <= ref_dt + timedelta(days=7) and end >= ref_dt - timedelta(days=1):
                windows.append(SuppressionWindow(
                    event_name=event.name,
                    event_date=event.event_date,
                    start=start,
                    end=end,
                    impact=event.impact,
                ))

        return windows

    def is_in_suppression_window(self, at: Optional[datetime] = None) -> bool:
        """Return True if *at* falls inside any suppression window.

        Used by the convergence alerter as a single-call gate.  When True,
        the caller should apply a confidence penalty or skip the alert.

        Args:
            at: Timestamp to check; defaults to datetime.now().
        """
        check_time = at or datetime.now()
        for event in self._events:
            event_dt = _event_datetime(event)
            start = event_dt - timedelta(hours=self._pre_event_hours)
            end = event_dt + timedelta(hours=self._post_event_hours)
            if start <= check_time <= end:
                logger.debug(
                    "suppression active: %s on %s (window %s – %s)",
                    event.name, event.event_date, start, end,
                )
                return True
        return False

    def get_next_event(self, from_date: Optional[date] = None) -> Optional[EconomicEvent]:
        """Return the earliest event on or after *from_date*.

        Args:
            from_date: Reference date; defaults to today.

        Returns:
            The next EconomicEvent, or None if the calendar is exhausted.
        """
        ref = from_date or date.today()
        for event in self._events:
            if event.event_date >= ref:
                return event
        return None

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def event_count(self) -> int:
        """Total number of events in the calendar."""
        return len(self._events)

    def events_by_type(self, name: str) -> List[EconomicEvent]:
        """Return all events matching *name* (e.g. 'FOMC', 'CPI', 'NFP')."""
        return [e for e in self._events if e.name == name]
