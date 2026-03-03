"""Tests for MarketClock — time-aware source availability for MIDGE's sensing pipeline.

Coverage:
  - Core equity market open/close queries
  - Futures session open/close queries and edge cases
  - Session name detection
  - Source availability filtering (individual and batch)
  - Business-day / holiday / weekend helpers
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from mae_core.market.market_clock import (
    ALWAYS_AVAILABLE,
    FUTURES_HOURS,
    MARKET_HOURS,
    PERIODIC,
    MarketClock,
)

_ET = ZoneInfo("US/Eastern")


def dt(year: int, month: int, day: int, hour: int = 12, minute: int = 0) -> datetime:
    """Convenience builder: always returns a timezone-aware US/Eastern datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=_ET)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clock() -> MarketClock:
    return MarketClock()


# ---------------------------------------------------------------------------
# Core time queries — equity market
# ---------------------------------------------------------------------------


def test_is_equity_market_open_during_hours(clock):
    """Mon-Fri 9:30-16:00 ET on a normal business day returns True."""
    # Monday 2026-03-02 at 10:00 ET (confirmed non-holiday weekday)
    assert clock.is_equity_market_open(at=dt(2026, 3, 2, 10, 0)) is True


def test_is_equity_market_open_before_open(clock):
    """9:00 AM on a weekday is pre-market — equity market is closed."""
    assert clock.is_equity_market_open(at=dt(2026, 3, 2, 9, 0)) is False


def test_is_equity_market_open_after_close(clock):
    """4:30 PM on a weekday is after-hours — equity market is closed."""
    assert clock.is_equity_market_open(at=dt(2026, 3, 2, 16, 30)) is False


def test_is_equity_market_open_at_exact_open(clock):
    """9:30 AM exactly is the first tick — market IS open."""
    assert clock.is_equity_market_open(at=dt(2026, 3, 2, 9, 30)) is True


def test_is_equity_market_open_at_exact_close(clock):
    """4:00 PM exactly is the close tick — market is NOT open (half-open interval)."""
    assert clock.is_equity_market_open(at=dt(2026, 3, 2, 16, 0)) is False


def test_is_equity_market_open_weekend(clock):
    """Saturday at noon — equity market is never open on weekends."""
    # 2026-03-07 is a Saturday
    assert clock.is_equity_market_open(at=dt(2026, 3, 7, 12, 0)) is False


def test_is_equity_market_open_holiday(clock):
    """MLK Day 2026 (2026-01-19) is a market holiday — equity closed."""
    assert clock.is_equity_market_open(at=dt(2026, 1, 19, 10, 0)) is False


# ---------------------------------------------------------------------------
# Core time queries — futures market
# ---------------------------------------------------------------------------


def test_is_futures_open_weekday(clock):
    """Tuesday 10:00 AM ET — futures are open during normal weekday hours."""
    # 2026-03-03 is a Tuesday
    assert clock.is_futures_open(at=dt(2026, 3, 3, 10, 0)) is True


def test_is_futures_open_weekend(clock):
    """Saturday noon — futures are always closed on Saturdays."""
    # 2026-03-07 is a Saturday
    assert clock.is_futures_open(at=dt(2026, 3, 7, 12, 0)) is False


def test_is_futures_open_sunday_evening(clock):
    """Sunday 6:30 PM ET — futures reopen at 6:00 PM Sunday, so True."""
    # 2026-03-08 is a Sunday
    assert clock.is_futures_open(at=dt(2026, 3, 8, 18, 30)) is True


# ---------------------------------------------------------------------------
# Futures edge cases
# ---------------------------------------------------------------------------


def test_futures_daily_halt_monday(clock):
    """Monday 5:00 PM – 6:00 PM ET is the daily maintenance halt — futures closed."""
    # 2026-03-02 is a Monday; halt is [17:00, 18:00)
    assert clock.is_futures_open(at=dt(2026, 3, 2, 17, 0)) is False
    assert clock.is_futures_open(at=dt(2026, 3, 2, 17, 30)) is False


def test_futures_daily_halt_thursday(clock):
    """Thursday is also subject to the daily 5–6 PM maintenance halt."""
    # 2026-03-05 is a Thursday
    assert clock.is_futures_open(at=dt(2026, 3, 5, 17, 15)) is False


def test_futures_friday_close(clock):
    """Friday 5:30 PM ET — futures close at 5:00 PM Friday for the weekend."""
    # 2026-03-06 is a Friday
    assert clock.is_futures_open(at=dt(2026, 3, 6, 17, 30)) is False


def test_futures_late_night(clock):
    """Tuesday 2:00 AM ET — overnight session is active."""
    # 2026-03-03 is a Tuesday
    assert clock.is_futures_open(at=dt(2026, 3, 3, 2, 0)) is True


def test_futures_open_boundary(clock):
    """Sunday 6:00 PM ET exactly — futures reopen at this instant (inclusive)."""
    # 2026-03-08 is a Sunday
    assert clock.is_futures_open(at=dt(2026, 3, 8, 18, 0)) is True


def test_futures_before_sunday_open(clock):
    """Sunday 5:59 PM ET — just before the 6:00 PM open, still 'weekend'."""
    # 2026-03-08 is a Sunday
    assert clock.is_futures_open(at=dt(2026, 3, 8, 17, 59)) is False


def test_futures_friday_before_close(clock):
    """Friday 4:59 PM ET — one minute before the 5:00 PM Friday close, still open."""
    # 2026-03-06 is a Friday
    assert clock.is_futures_open(at=dt(2026, 3, 6, 16, 59)) is True


# ---------------------------------------------------------------------------
# Session detection
# ---------------------------------------------------------------------------


def test_session_pre_market(clock):
    """Weekday 7:00 AM ET falls in the pre-market window (4:00–9:30 AM)."""
    assert clock.get_session(at=dt(2026, 3, 2, 7, 0)) == "pre_market"


def test_session_market(clock):
    """Weekday 10:00 AM ET falls in the regular market session."""
    assert clock.get_session(at=dt(2026, 3, 2, 10, 0)) == "market"


def test_session_after_hours(clock):
    """Weekday 6:00 PM ET falls in after-hours (4:00–8:00 PM)."""
    assert clock.get_session(at=dt(2026, 3, 2, 18, 0)) == "after_hours"


def test_session_weekend_saturday(clock):
    """Saturday — always 'weekend'."""
    # 2026-03-07 is a Saturday
    assert clock.get_session(at=dt(2026, 3, 7, 12, 0)) == "weekend"


def test_session_weekend_sunday_before_open(clock):
    """Sunday before 6:00 PM ET is still 'weekend' (futures not yet open)."""
    # 2026-03-08 is a Sunday
    assert clock.get_session(at=dt(2026, 3, 8, 14, 0)) == "weekend"


def test_session_overnight_sunday_after_open(clock):
    """Sunday after 6:00 PM ET is 'overnight' — futures are open."""
    # 2026-03-08 is a Sunday
    assert clock.get_session(at=dt(2026, 3, 8, 20, 0)) == "overnight"


# ---------------------------------------------------------------------------
# Source availability — batch queries
# ---------------------------------------------------------------------------


def test_always_available_sources_always_returned(clock):
    """ALWAYS_AVAILABLE sources appear in get_available_sources at any time."""
    # Saturday noon — most sources would be unavailable
    saturday_noon = dt(2026, 3, 7, 12, 0)
    available = clock.get_available_sources(at=saturday_noon)
    for source in ALWAYS_AVAILABLE:
        assert source in available, f"Expected always-available source '{source}' to be present"


def test_market_sources_during_market_hours(clock):
    """MARKET_HOURS sources are included when equity market is open."""
    market_time = dt(2026, 3, 2, 10, 0)  # Monday 10:00 AM ET
    available = clock.get_available_sources(at=market_time)
    for source in MARKET_HOURS:
        assert source in available, f"Expected market-hours source '{source}' during market hours"


def test_market_sources_excluded_after_hours(clock):
    """MARKET_HOURS sources are excluded after the 4:00 PM equity close."""
    after_close = dt(2026, 3, 2, 17, 0)  # Monday 5:00 PM ET
    unavailable = clock.get_unavailable_sources(at=after_close)
    for source in MARKET_HOURS:
        assert source in unavailable, (
            f"Expected market-hours source '{source}' to be unavailable after close"
        )


def test_futures_sources_during_futures_hours(clock):
    """FUTURES_HOURS sources are included when futures are open."""
    futures_time = dt(2026, 3, 3, 2, 0)  # Tuesday 2:00 AM ET (overnight session)
    available = clock.get_available_sources(at=futures_time)
    for source in FUTURES_HOURS:
        assert source in available, (
            f"Expected futures-hours source '{source}' during overnight session"
        )


def test_futures_sources_excluded_weekend(clock):
    """FUTURES_HOURS sources are excluded on Saturday (futures fully closed)."""
    saturday = dt(2026, 3, 7, 12, 0)  # Saturday noon
    unavailable = clock.get_unavailable_sources(at=saturday)
    for source in FUTURES_HOURS:
        assert source in unavailable, (
            f"Expected futures-hours source '{source}' to be unavailable on Saturday"
        )


# ---------------------------------------------------------------------------
# Source availability — individual queries
# ---------------------------------------------------------------------------


def test_source_available_always_available(clock):
    """source_available() returns True for ALWAYS_AVAILABLE sources at any time."""
    saturday = dt(2026, 3, 7, 3, 0)  # Saturday 3:00 AM — worst case
    for source in ALWAYS_AVAILABLE:
        assert clock.source_available(source, at=saturday) is True


def test_source_available_market_hours_open(clock):
    """source_available() returns True for a MARKET_HOURS source during market hours."""
    market_time = dt(2026, 3, 2, 11, 0)
    assert clock.source_available("finnhub", at=market_time) is True


def test_source_available_market_hours_closed(clock):
    """source_available() returns False for a MARKET_HOURS source outside market hours."""
    overnight = dt(2026, 3, 2, 23, 0)
    assert clock.source_available("finnhub", at=overnight) is False


def test_source_available_futures_hours_open(clock):
    """source_available() returns True for a FUTURES_HOURS source during futures session."""
    futures_open = dt(2026, 3, 4, 1, 0)  # Wednesday 1:00 AM ET
    assert clock.source_available("session_sweep", at=futures_open) is True


def test_source_available_futures_hours_during_halt(clock):
    """source_available() returns False for FUTURES_HOURS during the daily maintenance halt."""
    halt = dt(2026, 3, 2, 17, 30)  # Monday 5:30 PM ET
    assert clock.source_available("ta_indicators", at=halt) is False


def test_source_available_unknown_source_defaults_true(clock):
    """Unknown sources fail open — source_available() returns True for unrecognised names."""
    any_time = dt(2026, 3, 7, 3, 0)  # Saturday deep-night
    assert clock.source_available("some_future_source_not_yet_categorised", at=any_time) is True


# ---------------------------------------------------------------------------
# Business-day helpers
# ---------------------------------------------------------------------------


def test_is_business_day_weekday(clock):
    """Monday 2026-03-02 is a normal trading day."""
    assert clock.is_business_day(at=dt(2026, 3, 2, 12, 0)) is True


def test_is_business_day_weekend(clock):
    """Saturday 2026-03-07 is not a business day."""
    assert clock.is_business_day(at=dt(2026, 3, 7, 12, 0)) is False


def test_is_business_day_holiday(clock):
    """MLK Day 2026 (2026-01-19) is not a business day."""
    assert clock.is_business_day(at=dt(2026, 1, 19, 10, 0)) is False


def test_is_business_day_day_after_holiday(clock):
    """The Tuesday after MLK Day (2026-01-20) IS a business day."""
    assert clock.is_business_day(at=dt(2026, 1, 20, 10, 0)) is True


# ---------------------------------------------------------------------------
# Look-ahead helpers (smoke tests — verify type and basic direction)
# ---------------------------------------------------------------------------


def test_next_equity_open_from_weekend(clock):
    """next_equity_open called on Saturday returns a datetime on a weekday at 9:30 AM ET."""
    saturday = dt(2026, 3, 7, 12, 0)
    nxt = clock.next_equity_open(at=saturday)
    assert isinstance(nxt, datetime)
    assert nxt.weekday() < 5          # weekday (not weekend)
    assert nxt.hour == 9 and nxt.minute == 30


def test_next_futures_open_from_saturday(clock):
    """next_futures_open from Saturday returns the upcoming Sunday 6:00 PM ET."""
    saturday = dt(2026, 3, 7, 12, 0)  # Saturday noon
    nxt = clock.next_futures_open(at=saturday)
    assert isinstance(nxt, datetime)
    # Should be Sunday at 18:00
    assert nxt.weekday() == 6         # Sunday
    assert nxt.hour == 18 and nxt.minute == 0


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_status_returns_dict_with_expected_keys(clock):
    """status() returns a dict containing all expected diagnostic keys."""
    result = clock.status(at=dt(2026, 3, 2, 10, 0))
    expected_keys = {
        "at", "session", "equity_open", "futures_open",
        "business_day", "is_holiday", "is_weekend",
    }
    assert expected_keys.issubset(result.keys())


def test_status_during_market_hours_is_consistent(clock):
    """status() fields are mutually consistent during regular market hours."""
    market_time = dt(2026, 3, 2, 10, 0)
    s = clock.status(at=market_time)
    assert s["equity_open"] is True
    assert s["futures_open"] is True
    assert s["business_day"] is True
    assert s["is_holiday"] is False
    assert s["is_weekend"] is False
    assert s["session"] == "market"
    # next_equity_open should be None when market is already open
    assert s.get("next_equity_open") is None
