#!/usr/bin/env python3
"""Tests for mae_core.market.intelligence.catalyst_calendar.

All tests are self-contained — no real network calls are made.
FinnhubClient is mocked and datetime.now() is patched where precise
control over the current time is needed.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from mae_core.market.intelligence.catalyst_calendar import (
    Catalyst,
    CatalystCalendar,
    _FOMC_IMPACT,
    _MOD_INSIDER_SPECIAL,
    _MOD_MID,
    _MOD_NEAR,
    _MOD_NONE,
    _MOD_FAR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_finnhub_event(symbol: str, date_str: str) -> MagicMock:
    """Create a mock EarningsEvent from FinnhubClient."""
    event = MagicMock()
    event.symbol = symbol
    event.date = date_str
    event.eps_actual = None   # Not yet reported — still upcoming
    return event


def _make_calendar(finnhub_client=None) -> CatalystCalendar:
    return CatalystCalendar(finnhub_client=finnhub_client)


# ---------------------------------------------------------------------------
# TestCatalystRefresh
# ---------------------------------------------------------------------------

class TestCatalystRefresh(unittest.TestCase):
    """refresh() correctly loads earnings and always includes FOMC dates."""

    def test_refresh_without_finnhub_populates_fomc_only(self):
        cal = _make_calendar(finnhub_client=None)
        cal.refresh(tickers=["AAPL"])
        # No ticker-specific data without finnhub, but FOMC bucket should exist
        # FOMC catalysts are in __fomc__ bucket
        assert "__fomc__" in cal._upcoming

    def test_refresh_with_finnhub_loads_earnings(self):
        finnhub = MagicMock()
        finnhub.get_upcoming_earnings.return_value = [
            _make_finnhub_event("AAPL", "2026-05-01"),
        ]
        cal = _make_calendar(finnhub_client=finnhub)
        cal.refresh(tickers=["AAPL"])
        # AAPL should have at least one earnings catalyst
        aapl_catalysts = cal._upcoming.get("AAPL", [])
        types = [c.event_type for c in aapl_catalysts]
        assert "earnings" in types

    def test_refresh_caches_for_today(self):
        """Second refresh on same day does not call finnhub again."""
        finnhub = MagicMock()
        finnhub.get_upcoming_earnings.return_value = []
        cal = _make_calendar(finnhub_client=finnhub)
        cal.refresh(tickers=["MSFT"])
        cal.refresh(tickers=["MSFT"])
        # Should only have been called once
        assert finnhub.get_upcoming_earnings.call_count == 1

    def test_refresh_includes_fomc_for_all_tickers(self):
        """FOMC catalysts are added regardless of which ticker is being refreshed."""
        finnhub = MagicMock()
        finnhub.get_upcoming_earnings.return_value = []
        cal = _make_calendar(finnhub_client=finnhub)
        cal.refresh(tickers=["SPY", "QQQ"])
        # __fomc__ bucket should contain FOMC events
        fomc_catalysts = cal._upcoming.get("__fomc__", [])
        assert all(c.event_type == "fomc" for c in fomc_catalysts)

    def test_refresh_ignores_earnings_for_unlisted_tickers(self):
        """Finnhub returns TSLA earnings but we only asked for AAPL."""
        finnhub = MagicMock()
        finnhub.get_upcoming_earnings.return_value = [
            _make_finnhub_event("TSLA", "2026-05-01"),
        ]
        cal = _make_calendar(finnhub_client=finnhub)
        cal.refresh(tickers=["AAPL"])
        aapl_catalysts = cal._upcoming.get("AAPL", [])
        # No earnings for AAPL — TSLA earnings should not appear
        types = [c.event_type for c in aapl_catalysts]
        assert "earnings" not in types

    def test_refresh_finnhub_exception_does_not_crash(self):
        """If finnhub raises, refresh continues with FOMC only."""
        finnhub = MagicMock()
        finnhub.get_upcoming_earnings.side_effect = RuntimeError("API down")
        cal = _make_calendar(finnhub_client=finnhub)
        cal.refresh(tickers=["AAPL"])   # Should not raise
        assert "__fomc__" in cal._upcoming


# ---------------------------------------------------------------------------
# TestUpcoming
# ---------------------------------------------------------------------------

class TestUpcoming(unittest.TestCase):
    """get_upcoming() returns correctly filtered and sorted catalysts."""

    def _cal_with_catalysts(self, days_away_list, event_type="earnings") -> CatalystCalendar:
        """Build a calendar with synthetic catalysts at the given offsets."""
        cal = _make_calendar()
        now = datetime.now()
        catalysts = [
            Catalyst(
                ticker="TEST",
                event_type=event_type,
                date=now + timedelta(days=d),
                estimated_impact=0.5,
                source="test",
            )
            for d in days_away_list
        ]
        # Inject directly into the internal dict (no finnhub call needed)
        cal._upcoming["TEST"] = sorted(catalysts, key=lambda c: c.date)
        cal._cache_date = now.strftime("%Y-%m-%d")
        return cal

    def test_returns_catalysts_within_window(self):
        cal = self._cal_with_catalysts([3, 10, 20])
        upcoming = cal.get_upcoming("TEST", days=14)
        days_away = [(c.date - datetime.now()).days for c in upcoming]
        # Only 3-day and 10-day should be within 14-day window
        assert len(upcoming) == 2

    def test_excludes_catalysts_beyond_window(self):
        cal = self._cal_with_catalysts([20, 30])
        upcoming = cal.get_upcoming("TEST", days=14)
        assert len(upcoming) == 0

    def test_results_sorted_by_date(self):
        cal = self._cal_with_catalysts([7, 2, 12])
        upcoming = cal.get_upcoming("TEST", days=14)
        dates = [c.date for c in upcoming]
        assert dates == sorted(dates)

    def test_unknown_ticker_returns_fomc_if_available(self):
        """An unregistered ticker still gets FOMC dates from the __fomc__ bucket."""
        cal = _make_calendar()
        now = datetime.now()
        # Manually inject a near-term FOMC date
        fomc_dt = now + timedelta(days=5)
        cal._upcoming["__fomc__"] = [
            Catalyst(ticker="__fomc__", event_type="fomc", date=fomc_dt,
                     estimated_impact=_FOMC_IMPACT, source="hardcoded")
        ]
        cal._cache_date = now.strftime("%Y-%m-%d")
        upcoming = cal.get_upcoming("UNKNOWN_TICKER", days=14)
        assert len(upcoming) == 1
        assert upcoming[0].event_type == "fomc"

    def test_no_catalysts_returns_empty_list(self):
        cal = _make_calendar()
        cal._cache_date = datetime.now().strftime("%Y-%m-%d")
        upcoming = cal.get_upcoming("NOCAT", days=14)
        assert upcoming == []


# ---------------------------------------------------------------------------
# TestModifier
# ---------------------------------------------------------------------------

class TestModifier(unittest.TestCase):
    """compute_catalyst_modifier() returns correct multiplier for each tier."""

    def _cal_with_days_away(self, days: float) -> CatalystCalendar:
        """Calendar with a single catalyst at `days` offset from now."""
        cal = _make_calendar()
        now = datetime.now()
        cat = Catalyst(
            ticker="TEST",
            event_type="earnings",
            date=now + timedelta(days=days),
            estimated_impact=0.5,
            source="test",
        )
        cal._upcoming["TEST"] = [cat]
        cal._cache_date = now.strftime("%Y-%m-%d")
        return cal

    def test_near_catalyst_returns_1_3(self):
        """0–3 days → 1.3x multiplier."""
        cal = self._cal_with_days_away(1.5)
        modifier = cal.compute_catalyst_modifier("TEST")
        assert modifier == _MOD_NEAR

    def test_mid_catalyst_returns_1_1(self):
        """3–7 days → 1.1x multiplier."""
        cal = self._cal_with_days_away(5)
        modifier = cal.compute_catalyst_modifier("TEST")
        assert modifier == _MOD_MID

    def test_far_catalyst_returns_1_0(self):
        """7–14 days → 1.0x multiplier."""
        cal = self._cal_with_days_away(10)
        modifier = cal.compute_catalyst_modifier("TEST")
        assert modifier == _MOD_FAR

    def test_no_catalyst_returns_0_9(self):
        """No upcoming catalysts → 0.9x mild discount."""
        cal = _make_calendar()
        cal._cache_date = datetime.now().strftime("%Y-%m-%d")
        modifier = cal.compute_catalyst_modifier("TEST")
        assert modifier == _MOD_NONE

    def test_catalyst_at_exactly_3_days_is_near(self):
        """Boundary: exactly 3 days = near tier (≤ 3)."""
        cal = self._cal_with_days_away(3.0)
        modifier = cal.compute_catalyst_modifier("TEST")
        assert modifier == _MOD_NEAR

    def test_catalyst_at_exactly_7_days_is_mid(self):
        """Boundary: exactly 7 days = mid tier (3 < d ≤ 7)."""
        cal = self._cal_with_days_away(7.0)
        modifier = cal.compute_catalyst_modifier("TEST")
        assert modifier == _MOD_MID

    def test_beyond_14_days_returns_0_9(self):
        """Catalyst exists but is > 14 days away → treat as no context."""
        cal = self._cal_with_days_away(20)
        modifier = cal.compute_catalyst_modifier("TEST")
        assert modifier == _MOD_NONE


# ---------------------------------------------------------------------------
# TestInsiderSpecial
# ---------------------------------------------------------------------------

class TestInsiderSpecial(unittest.TestCase):
    """insider domain + catalyst < 14 days → 1.5x multiplier."""

    def _cal_with_days_away(self, days: float) -> CatalystCalendar:
        cal = _make_calendar()
        now = datetime.now()
        cat = Catalyst(
            ticker="TEST",
            event_type="earnings",
            date=now + timedelta(days=days),
            estimated_impact=0.5,
            source="test",
        )
        cal._upcoming["TEST"] = [cat]
        cal._cache_date = now.strftime("%Y-%m-%d")
        return cal

    def test_insider_domain_with_near_catalyst(self):
        cal = self._cal_with_days_away(2)
        modifier = cal.compute_catalyst_modifier("TEST", signal_domain="insider")
        assert modifier == _MOD_INSIDER_SPECIAL

    def test_insider_domain_with_14_day_catalyst(self):
        """Insider + exactly 14 days → still special (boundary inclusive)."""
        cal = self._cal_with_days_away(14)
        modifier = cal.compute_catalyst_modifier("TEST", signal_domain="insider")
        assert modifier == _MOD_INSIDER_SPECIAL

    def test_sec_form4_domain_triggers_special(self):
        """sec_form4 is also in the insider domains set."""
        cal = self._cal_with_days_away(5)
        modifier = cal.compute_catalyst_modifier("TEST", signal_domain="sec_form4")
        assert modifier == _MOD_INSIDER_SPECIAL

    def test_congressional_domain_triggers_special(self):
        cal = self._cal_with_days_away(8)
        modifier = cal.compute_catalyst_modifier("TEST", signal_domain="congressional")
        assert modifier == _MOD_INSIDER_SPECIAL

    def test_insider_domain_beyond_14_days_does_not_trigger_special(self):
        """Insider + catalyst > 14 days → no special treatment."""
        cal = self._cal_with_days_away(20)
        modifier = cal.compute_catalyst_modifier("TEST", signal_domain="insider")
        # Beyond 14 days → no-catalyst path → 0.9
        assert modifier == _MOD_NONE

    def test_non_insider_domain_near_catalyst_uses_standard_tier(self):
        """technical domain near earnings → standard 1.3x, not 1.5x."""
        cal = self._cal_with_days_away(2)
        modifier = cal.compute_catalyst_modifier("TEST", signal_domain="technical")
        assert modifier == _MOD_NEAR


# ---------------------------------------------------------------------------
# TestGracefulDegradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation(unittest.TestCase):
    """CatalystCalendar works when finnhub is None or data is missing."""

    def test_no_finnhub_client_does_not_crash(self):
        cal = _make_calendar(finnhub_client=None)
        cal.refresh(tickers=["AAPL"])  # Should not raise

    def test_modifier_with_no_data_returns_0_9(self):
        cal = _make_calendar()
        cal._cache_date = datetime.now().strftime("%Y-%m-%d")
        modifier = cal.compute_catalyst_modifier("UNKNOWN")
        assert modifier == _MOD_NONE

    def test_days_to_nearest_with_no_data_returns_none(self):
        cal = _make_calendar()
        cal._cache_date = datetime.now().strftime("%Y-%m-%d")
        result = cal.days_to_nearest_catalyst("UNKNOWN")
        assert result is None

    def test_empty_tickers_list_does_not_crash(self):
        cal = _make_calendar()
        cal.refresh(tickers=[])  # Should not raise

    def test_get_upcoming_empty_calendar_returns_empty(self):
        cal = _make_calendar()
        cal._cache_date = datetime.now().strftime("%Y-%m-%d")
        result = cal.get_upcoming("AAPL", days=14)
        assert result == []


# ---------------------------------------------------------------------------
# TestFOMC
# ---------------------------------------------------------------------------

class TestFOMC(unittest.TestCase):
    """FOMC dates are returned for any ticker and match known 2026 dates."""

    def test_fomc_dates_exist_after_refresh(self):
        cal = _make_calendar()
        cal.refresh(tickers=[])
        fomc_bucket = cal._upcoming.get("__fomc__", [])
        assert len(fomc_bucket) > 0

    def test_fomc_events_have_correct_event_type(self):
        cal = _make_calendar()
        cal.refresh(tickers=[])
        fomc_bucket = cal._upcoming.get("__fomc__", [])
        assert all(c.event_type == "fomc" for c in fomc_bucket)

    def test_fomc_source_is_hardcoded(self):
        cal = _make_calendar()
        cal.refresh(tickers=[])
        fomc_bucket = cal._upcoming.get("__fomc__", [])
        assert all(c.source == "hardcoded" for c in fomc_bucket)

    def test_fomc_appears_in_upcoming_for_any_ticker(self):
        """get_upcoming() for a ticker that was not refreshed still returns FOMC."""
        cal = _make_calendar()
        now = datetime.now()
        # Inject a near-term FOMC date directly
        fomc_dt = now + timedelta(days=4)
        cal._upcoming["__fomc__"] = [
            Catalyst(ticker="__fomc__", event_type="fomc", date=fomc_dt,
                     estimated_impact=_FOMC_IMPACT, source="hardcoded")
        ]
        cal._cache_date = now.strftime("%Y-%m-%d")
        upcoming = cal.get_upcoming("RANDOM_TICKER", days=14)
        assert any(c.event_type == "fomc" for c in upcoming)

    def test_fomc_dates_are_in_future(self):
        """All hardcoded FOMC dates from 2026 that are in the future have correct year."""
        cal = _make_calendar()
        now = datetime.now()
        future_fomc = [dt for dt in cal._fomc_dates if dt > now]
        assert all(dt.year == 2026 for dt in future_fomc)

    def test_fomc_nearest_catalyst_computed_correctly(self):
        """days_to_nearest returns a value when FOMC is upcoming."""
        cal = _make_calendar()
        now = datetime.now()
        fomc_dt = now + timedelta(days=6)
        cal._upcoming["__fomc__"] = [
            Catalyst(ticker="__fomc__", event_type="fomc", date=fomc_dt,
                     estimated_impact=_FOMC_IMPACT, source="hardcoded")
        ]
        cal._cache_date = now.strftime("%Y-%m-%d")
        days = cal.days_to_nearest_catalyst("ANY_TICKER")
        assert days is not None
        assert 5.9 <= days <= 6.1


if __name__ == "__main__":
    unittest.main()
