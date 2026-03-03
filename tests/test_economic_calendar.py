"""Tests for the Economic Calendar Client.

Covers: dataclass creation, calendar construction, get_upcoming_events,
get_suppression_windows, is_in_suppression_window, get_next_event,
release times (FOMC 2PM / CPI+NFP 8:30AM), window boundaries,
custom pre/post hours, and full date-count assertions.

All tests use deterministic dates from the hardcoded 2026 calendar —
no network calls, no datetime.now() patching required for date-specific
assertions.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from mae_core.market.apis.economic_calendar_client import (
    EconomicCalendarClient,
    EconomicEvent,
    SuppressionWindow,
    _CPI_DATES_2026,
    _FOMC_DATES_2026,
    _NFP_DATES_2026,
)


# ---------------------------------------------------------------------------
# Dataclass creation
# ---------------------------------------------------------------------------

class TestEconomicEventDataclass:
    def test_create_with_all_fields(self):
        event = EconomicEvent(
            name="FOMC",
            event_date=date(2026, 1, 29),
            impact="high",
            description="Federal Reserve monetary policy decision",
        )
        assert event.name == "FOMC"
        assert event.event_date == date(2026, 1, 29)
        assert event.impact == "high"
        assert "Federal Reserve" in event.description

    def test_description_has_default(self):
        event = EconomicEvent(name="CPI", event_date=date(2026, 2, 11), impact="high")
        assert event.description == ""

    def test_create_medium_impact(self):
        event = EconomicEvent(name="JOLTS", event_date=date(2026, 3, 1), impact="medium")
        assert event.impact == "medium"


class TestSuppressionWindowDataclass:
    def test_create_with_all_fields(self):
        start = datetime(2026, 1, 28, 14, 0)
        end = datetime(2026, 1, 29, 18, 0)
        window = SuppressionWindow(
            event_name="FOMC",
            event_date=date(2026, 1, 29),
            start=start,
            end=end,
            impact="high",
        )
        assert window.event_name == "FOMC"
        assert window.event_date == date(2026, 1, 29)
        assert window.start == start
        assert window.end == end
        assert window.impact == "high"

    def test_start_before_end(self):
        start = datetime(2026, 3, 10, 8, 30) - timedelta(hours=24)
        end = datetime(2026, 3, 10, 8, 30) + timedelta(hours=4)
        window = SuppressionWindow(
            event_name="CPI",
            event_date=date(2026, 3, 11),
            start=start,
            end=end,
            impact="high",
        )
        assert window.start < window.end


# ---------------------------------------------------------------------------
# Calendar construction — counts
# ---------------------------------------------------------------------------

class TestCalendarCounts:
    def test_fomc_date_count(self):
        """FOMC list must have exactly 16 entries (8 meetings × 2 days)."""
        assert len(_FOMC_DATES_2026) == 16

    def test_cpi_date_count(self):
        """CPI list must have exactly 12 entries (one per month)."""
        assert len(_CPI_DATES_2026) == 12

    def test_nfp_date_count(self):
        """NFP list must have exactly 12 entries (one per month)."""
        assert len(_NFP_DATES_2026) == 12

    def test_calendar_total_events(self):
        client = EconomicCalendarClient()
        assert client.event_count() == 16 + 12 + 12  # 40

    def test_calendar_contains_fomc(self):
        client = EconomicCalendarClient()
        fomc_events = client.events_by_type("FOMC")
        assert len(fomc_events) == 16

    def test_calendar_contains_cpi(self):
        client = EconomicCalendarClient()
        cpi_events = client.events_by_type("CPI")
        assert len(cpi_events) == 12

    def test_calendar_contains_nfp(self):
        client = EconomicCalendarClient()
        nfp_events = client.events_by_type("NFP")
        assert len(nfp_events) == 12

    def test_all_events_are_high_impact(self):
        client = EconomicCalendarClient()
        for event in client._events:
            assert event.impact == "high"

    def test_events_sorted_ascending(self):
        client = EconomicCalendarClient()
        dates = [e.event_date for e in client._events]
        assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# get_upcoming_events
# ---------------------------------------------------------------------------

class TestGetUpcomingEvents:
    def test_returns_events_within_range(self):
        client = EconomicCalendarClient()
        # CPI on 2026-01-14; query from 2026-01-14, 7-day window.
        events = client.get_upcoming_events(days=7, from_date=date(2026, 1, 14))
        names = {e.name for e in events}
        assert "CPI" in names

    def test_returns_empty_for_far_future(self):
        client = EconomicCalendarClient()
        # 2030 is well past the hardcoded calendar.
        events = client.get_upcoming_events(days=7, from_date=date(2030, 6, 1))
        assert events == []

    def test_inclusive_boundary_start(self):
        client = EconomicCalendarClient()
        # NFP on 2026-01-02; querying from that exact date should include it.
        events = client.get_upcoming_events(days=0, from_date=date(2026, 1, 2))
        assert any(e.event_date == date(2026, 1, 2) for e in events)

    def test_inclusive_boundary_end(self):
        client = EconomicCalendarClient()
        # 7-day window from 2026-01-07 should reach 2026-01-14 (CPI).
        events = client.get_upcoming_events(days=7, from_date=date(2026, 1, 7))
        assert any(e.name == "CPI" and e.event_date == date(2026, 1, 14) for e in events)

    def test_excludes_event_just_outside_window(self):
        client = EconomicCalendarClient()
        # CPI on 2026-01-14; window from 2026-01-08, 5 days → ends 2026-01-13.
        events = client.get_upcoming_events(days=5, from_date=date(2026, 1, 8))
        assert not any(e.event_date == date(2026, 1, 14) for e in events)

    def test_results_sorted_by_date(self):
        client = EconomicCalendarClient()
        events = client.get_upcoming_events(days=60, from_date=date(2026, 1, 1))
        dates = [e.event_date for e in events]
        assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# Release times — FOMC at 2PM, CPI/NFP at 8:30AM
# ---------------------------------------------------------------------------

class TestReleaseTimes:
    def test_fomc_event_time_is_2pm(self):
        client = EconomicCalendarClient()
        windows = client.get_suppression_windows(from_date=date(2026, 1, 29))
        fomc_windows = [w for w in windows if w.event_name == "FOMC" and w.event_date == date(2026, 1, 29)]
        assert fomc_windows, "Expected at least one FOMC window for 2026-01-29"
        w = fomc_windows[0]
        # event_time = 14:00; start = event_time - 24h
        expected_start = datetime(2026, 1, 29, 14, 0) - timedelta(hours=24)
        assert w.start == expected_start

    def test_cpi_event_time_is_8_30am(self):
        client = EconomicCalendarClient()
        windows = client.get_suppression_windows(from_date=date(2026, 1, 14))
        cpi_windows = [w for w in windows if w.event_name == "CPI" and w.event_date == date(2026, 1, 14)]
        assert cpi_windows, "Expected at least one CPI window for 2026-01-14"
        w = cpi_windows[0]
        expected_start = datetime(2026, 1, 14, 8, 30) - timedelta(hours=24)
        assert w.start == expected_start

    def test_nfp_event_time_is_8_30am(self):
        client = EconomicCalendarClient()
        windows = client.get_suppression_windows(from_date=date(2026, 1, 2))
        nfp_windows = [w for w in windows if w.event_name == "NFP" and w.event_date == date(2026, 1, 2)]
        assert nfp_windows, "Expected at least one NFP window for 2026-01-02"
        w = nfp_windows[0]
        expected_start = datetime(2026, 1, 2, 8, 30) - timedelta(hours=24)
        assert w.start == expected_start


# ---------------------------------------------------------------------------
# Suppression window boundaries
# ---------------------------------------------------------------------------

class TestSuppressionWindowBoundaries:
    def test_start_is_event_minus_24h(self):
        client = EconomicCalendarClient()
        # FOMC on 2026-03-19 at 14:00
        windows = client.get_suppression_windows(from_date=date(2026, 3, 19))
        w = next((x for x in windows if x.event_name == "FOMC" and x.event_date == date(2026, 3, 19)), None)
        assert w is not None
        assert w.start == datetime(2026, 3, 19, 14, 0) - timedelta(hours=24)

    def test_end_is_event_plus_4h(self):
        client = EconomicCalendarClient()
        # FOMC on 2026-03-19 at 14:00
        windows = client.get_suppression_windows(from_date=date(2026, 3, 19))
        w = next((x for x in windows if x.event_name == "FOMC" and x.event_date == date(2026, 3, 19)), None)
        assert w is not None
        assert w.end == datetime(2026, 3, 19, 14, 0) + timedelta(hours=4)

    def test_cpi_window_end_is_event_plus_4h(self):
        client = EconomicCalendarClient()
        # CPI on 2026-03-11 at 08:30
        windows = client.get_suppression_windows(from_date=date(2026, 3, 11))
        w = next((x for x in windows if x.event_name == "CPI" and x.event_date == date(2026, 3, 11)), None)
        assert w is not None
        assert w.end == datetime(2026, 3, 11, 8, 30) + timedelta(hours=4)


# ---------------------------------------------------------------------------
# Custom pre/post hours
# ---------------------------------------------------------------------------

class TestCustomHours:
    def test_custom_pre_event_hours(self):
        client = EconomicCalendarClient(pre_event_hours=48.0, post_event_hours=4.0)
        # FOMC on 2026-01-29 at 14:00; start should be 48h before.
        windows = client.get_suppression_windows(from_date=date(2026, 1, 29))
        w = next((x for x in windows if x.event_name == "FOMC" and x.event_date == date(2026, 1, 29)), None)
        assert w is not None
        assert w.start == datetime(2026, 1, 29, 14, 0) - timedelta(hours=48)

    def test_custom_post_event_hours(self):
        client = EconomicCalendarClient(pre_event_hours=24.0, post_event_hours=8.0)
        # NFP on 2026-01-02 at 08:30; end should be 8h after.
        windows = client.get_suppression_windows(from_date=date(2026, 1, 2))
        w = next((x for x in windows if x.event_name == "NFP" and x.event_date == date(2026, 1, 2)), None)
        assert w is not None
        assert w.end == datetime(2026, 1, 2, 8, 30) + timedelta(hours=8)

    def test_default_hours_are_24_and_4(self):
        client = EconomicCalendarClient()
        assert client._pre_event_hours == 24.0
        assert client._post_event_hours == 4.0


# ---------------------------------------------------------------------------
# is_in_suppression_window
# ---------------------------------------------------------------------------

class TestIsInSuppressionWindow:
    def test_true_during_fomc_window(self):
        client = EconomicCalendarClient()
        # FOMC on 2026-01-29 14:00; check 1h before the event = inside window.
        inside = datetime(2026, 1, 29, 13, 0)
        assert client.is_in_suppression_window(at=inside) is True

    def test_true_at_exact_event_start(self):
        client = EconomicCalendarClient()
        # Window starts exactly at event_time - 24h.
        window_start = datetime(2026, 1, 29, 14, 0) - timedelta(hours=24)
        assert client.is_in_suppression_window(at=window_start) is True

    def test_true_during_post_event_period(self):
        client = EconomicCalendarClient()
        # FOMC on 2026-01-29 14:00; 2h after the event = still in window.
        after = datetime(2026, 1, 29, 16, 0)
        assert client.is_in_suppression_window(at=after) is True

    def test_false_outside_any_window(self):
        client = EconomicCalendarClient()
        # Mid-March, well between events.
        outside = datetime(2026, 3, 25, 12, 0)
        assert client.is_in_suppression_window(at=outside) is False

    def test_false_after_window_end(self):
        client = EconomicCalendarClient()
        # FOMC on 2026-01-29 14:00; window ends at 18:00; check at 19:00.
        after_window = datetime(2026, 1, 29, 19, 0)
        assert client.is_in_suppression_window(at=after_window) is False

    def test_true_during_cpi_window(self):
        client = EconomicCalendarClient()
        # CPI on 2026-01-14 08:30; check 2h before event = inside window.
        inside = datetime(2026, 1, 14, 6, 30)
        assert client.is_in_suppression_window(at=inside) is True


# ---------------------------------------------------------------------------
# get_next_event
# ---------------------------------------------------------------------------

class TestGetNextEvent:
    def test_from_january_returns_january_event(self):
        client = EconomicCalendarClient()
        event = client.get_next_event(from_date=date(2026, 1, 1))
        assert event is not None
        assert event.event_date.year == 2026
        assert event.event_date.month == 1

    def test_from_exact_event_date_returns_that_event(self):
        client = EconomicCalendarClient()
        # NFP on 2026-01-02 is the earliest 2026 event.
        event = client.get_next_event(from_date=date(2026, 1, 2))
        assert event is not None
        assert event.event_date == date(2026, 1, 2)

    def test_returns_none_past_end_of_calendar(self):
        client = EconomicCalendarClient()
        event = client.get_next_event(from_date=date(2030, 1, 1))
        assert event is None

    def test_next_event_from_mid_year(self):
        client = EconomicCalendarClient()
        # After June FOMC (2026-06-18), next should be in July.
        event = client.get_next_event(from_date=date(2026, 6, 19))
        assert event is not None
        assert event.event_date >= date(2026, 6, 19)
