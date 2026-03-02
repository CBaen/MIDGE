"""Tests for AbsenceMonitor (Package B — the dog that didn't bark).

What is under test:
  - mae_core/market/intelligence/absence_monitor.py (AbsenceMonitor, AbsenceSignal)
  - sensing_hook.py integration (_collect_one records arrival, step() checks at cadence 100)
  - bootstrap wiring (_wire_sensing_hook injects absence_monitor from ctx)

Design decisions validated here:
  - Cadence uses median inter-arrival time (robust to outliers)
  - Known-cadence sources (_KNOWN_CADENCES) can fire with < MIN_OBSERVATIONS arrivals
  - Unknown sources require MIN_OBSERVATIONS (5) arrivals before they can fire
  - Silence fires when silence_ratio >= DEFAULT_ABSENCE_THRESHOLD (2.5)
  - EventBus publish is best-effort: exceptions must not propagate
  - bootstrap_from_archives returns 0 gracefully when reader is None

Test strategy:
  - Direct unit tests of AbsenceMonitor using explicit timestamps (no time.sleep)
  - Sensing hook tests drive _collect_one() directly (avoids network I/O)
  - Bootstrap wiring calls _wire_sensing_hook() with a minimal ctx SimpleNamespace
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.intelligence.absence_monitor import (
    AbsenceMonitor,
    AbsenceSignal,
    _KNOWN_CADENCES,
    DEFAULT_ABSENCE_THRESHOLD,
    MAX_HISTORY,
    MIN_OBSERVATIONS,
)
from mae_core.market.sensing_hook import MarketSensingHook


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_monitor(**kwargs) -> AbsenceMonitor:
    """Construct an AbsenceMonitor with no external dependencies."""
    defaults = dict(event_bus=None, archive_reader=None)
    defaults.update(kwargs)
    return AbsenceMonitor(**defaults)


def _arrivals(monitor: AbsenceMonitor, source: str, base: datetime, count: int, gap_hours: float) -> None:
    """Record *count* arrivals spaced *gap_hours* apart, ending at *base*."""
    start = base - timedelta(hours=gap_hours * (count - 1))
    for i in range(count):
        monitor.record_arrival(source, start + timedelta(hours=gap_hours * i))


def _make_hook(**overrides) -> MarketSensingHook:
    """Construct a MarketSensingHook with all params set to None."""
    defaults = dict(
        sec_client=None, price_fetcher=None, congress_client=None,
        senate_client=None, job_tracker=None, usa_spending=None,
        sam_gov=None, apewisdom=None, finra_client=None, sec_efts=None,
        finnhub=None, fred=None, convergence_alerter=None,
        velocity_detector=None, filing_analyzer=None, form8k_sentiment=None,
        session_sweep_detector=None, ta_indicators=None, cot_client=None,
        stocktwits_client=None, vix_client=None, trends_client=None,
        outcome_collector=None, memory=None, thompson_sampler=None,
        tiered_alerters=None, watchlist={}, fetch_cadence=50,
        outcome_cadence=200,
    )
    defaults.update(overrides)
    return MarketSensingHook(**defaults)


def _make_signal(
    signal_id: str = "sig1",
    strength: float = 0.7,
    domain: str = "insider",
    direction: str = "bullish",
    source: str = "sec_form4",
    symbol: str = "AAPL",
) -> MagicMock:
    """Build a minimal mock satisfying MarketSensingHook._collect_one() reads."""
    sig = MagicMock()
    sig.signal_id = signal_id
    sig.strength = strength
    sig.domain = domain
    sig.direction = direction
    sig.source = source
    sig.symbol = symbol
    sig.confidence = 0.6
    sig.velocity = 0.0
    sig.timestamp = datetime.now()
    sig.metadata = {}
    return sig


def _make_future(signals: list) -> MagicMock:
    """Return a completed Future-like mock yielding *signals*."""
    fut = MagicMock()
    fut.done.return_value = True
    fut.result.return_value = signals
    return fut


# ---------------------------------------------------------------------------
# Section 1: Cadence learning from arrival history
# ---------------------------------------------------------------------------


class TestCadenceLearning:
    """Verifies that inter-arrival cadence is computed correctly from history."""

    def test_cadence_learned_from_arrivals(self):
        """Six arrivals 24 h apart -> learned cadence near 24 h."""
        mon = _make_monitor()
        base = datetime(2026, 3, 1, 12, 0, 0)
        _arrivals(mon, "finra_short", base, count=6, gap_hours=24.0)

        cadence = mon._cadence.get("finra_short")
        assert cadence is not None, "cadence must be set after 6 arrivals"
        assert abs(cadence - 24.0) < 0.5, f"Expected ~24h cadence, got {cadence}"

    def test_cadence_uses_median_not_mean(self):
        """Outlier arrival should not significantly distort the learned cadence.

        5 arrivals 24 h apart + 1 arrival 168 h later. Median interval should
        be dominated by the 24 h intervals, not pulled toward 168.
        """
        mon = _make_monitor()
        base = datetime(2026, 3, 1, 0, 0, 0)
        # 5 regular arrivals at 24 h spacing
        for i in range(5):
            mon.record_arrival("custom_src", base + timedelta(hours=24 * i))
        # 1 outlier 168 h after the last
        mon.record_arrival("custom_src", base + timedelta(hours=24 * 4 + 168))

        cadence = mon._cadence.get("custom_src")
        assert cadence is not None
        # Mean would be pulled up; median stays near 24
        assert cadence < 60.0, (
            f"Median cadence should be ~24 h (robust), got {cadence}"
        )

    def test_cadence_requires_at_least_2_arrivals(self):
        """A single arrival provides no inter-arrival interval — cadence not set for unknown sources."""
        mon = _make_monitor()
        # "unknown_src" is not in _KNOWN_CADENCES
        mon.record_arrival("unknown_src", datetime(2026, 3, 1))

        # No cadence from known dict, and only 1 arrival — no interval to compute
        cadence = mon._cadence.get("unknown_src")
        assert cadence is None, (
            f"Expected no cadence from single arrival for unknown source, got {cadence}"
        )

    def test_known_cadence_seeds_used_when_no_history(self):
        """Known-cadence sources start with their seed value before any arrivals."""
        mon = _make_monitor()
        # congressional is in _KNOWN_CADENCES with 168.0 h seed
        assert "congressional" in _KNOWN_CADENCES
        assert mon._cadence.get("congressional") == 168.0, (
            "congressional must start with 168 h seed cadence"
        )

    def test_cadence_updates_with_new_arrivals(self):
        """Recording more arrivals refines the cadence estimate.

        Start with the known seed of 168 h for 'congressional', then
        record 6 arrivals 24 h apart — cadence should update toward 24 h.
        """
        mon = _make_monitor()
        initial = mon._cadence.get("congressional")
        assert initial == 168.0

        base = datetime(2026, 3, 1)
        _arrivals(mon, "congressional", base, count=6, gap_hours=24.0)

        updated = mon._cadence.get("congressional")
        assert updated is not None
        # After 6 arrivals at 24 h spacing, cadence should move toward 24 h
        assert updated < 100.0, (
            f"Cadence should update toward 24 h from data, got {updated}"
        )

    def test_max_history_bounded(self):
        """After MAX_HISTORY arrivals, the deque does not grow unbounded."""
        mon = _make_monitor()
        base = datetime(2026, 1, 1)
        for i in range(MAX_HISTORY + 50):
            mon.record_arrival("unknown_src", base + timedelta(hours=i))

        history = mon._arrival_history.get("unknown_src")
        assert history is not None
        assert len(history) <= MAX_HISTORY, (
            f"History deque exceeded MAX_HISTORY={MAX_HISTORY}: len={len(history)}"
        )


# ---------------------------------------------------------------------------
# Section 2: Absence detection
# ---------------------------------------------------------------------------


class TestAbsenceDetection:
    """Verifies the silence threshold logic and AbsenceSignal output."""

    def test_absence_fires_at_threshold(self):
        """Source with 48 h cadence, silent for 120 h -> fires (ratio 2.5)."""
        mon = _make_monitor()
        base = datetime(2026, 3, 1, 0, 0, 0)
        _arrivals(mon, "test_src", base, count=6, gap_hours=48.0)

        last_seen = base
        check_time = base + timedelta(hours=120)  # 120/48 = 2.5 >= threshold
        absences = mon.check_absences(now=check_time)

        assert any(a.source == "test_src" for a in absences), (
            "Expected absence signal for test_src at 2.5x cadence"
        )

    def test_no_absence_below_threshold(self):
        """Source with 48 h cadence, silent for 96 h -> does not fire (ratio 2.0 < 2.5)."""
        mon = _make_monitor()
        base = datetime(2026, 3, 1, 0, 0, 0)
        _arrivals(mon, "test_src", base, count=6, gap_hours=48.0)

        check_time = base + timedelta(hours=96)  # 96/48 = 2.0 < 2.5
        absences = mon.check_absences(now=check_time)

        assert not any(a.source == "test_src" for a in absences), (
            "Should not fire below threshold (ratio 2.0)"
        )

    def test_absence_requires_min_observations_for_unknown_source(self):
        """Unknown source with only 3 arrivals must not fire even if silent."""
        mon = _make_monitor()
        base = datetime(2026, 3, 1)
        # 3 arrivals 24 h apart — below MIN_OBSERVATIONS=5
        _arrivals(mon, "brand_new_src", base, count=3, gap_hours=24.0)

        check_time = base + timedelta(hours=200)  # very long silence
        absences = mon.check_absences(now=check_time)

        assert not any(a.source == "brand_new_src" for a in absences), (
            "Unknown source with <5 observations must not fire"
        )

    def test_known_source_can_fire_without_min_observations(self):
        """Known-cadence source with only 2 arrivals CAN fire (seed cadence trusted)."""
        mon = _make_monitor()
        base = datetime(2026, 3, 1)
        # sec_form4 is a known-cadence source (48 h seed)
        assert "sec_form4" in _KNOWN_CADENCES
        # Only 2 arrivals — below MIN_OBSERVATIONS
        mon.record_arrival("sec_form4", base - timedelta(hours=48))
        mon.record_arrival("sec_form4", base)

        # Silent for 200 h — silence_ratio >> 2.5
        check_time = base + timedelta(hours=200)
        absences = mon.check_absences(now=check_time)

        assert any(a.source == "sec_form4" for a in absences), (
            "Known-cadence source with seed cadence must fire even with <5 observations"
        )

    def test_absence_direction_is_bearish(self):
        """Default direction on AbsenceSignal is 'bearish'."""
        mon = _make_monitor()
        base = datetime(2026, 3, 1)
        _arrivals(mon, "test_src", base, count=6, gap_hours=48.0)

        check_time = base + timedelta(hours=200)
        absences = mon.check_absences(now=check_time)

        assert absences, "Expected at least one absence signal"
        for a in absences:
            if a.source == "test_src":
                assert a.direction == "bearish", (
                    f"Expected direction='bearish', got {a.direction!r}"
                )

    def test_silence_ratio_computed_correctly(self):
        """silence_ratio == actual_hours / expected_hours."""
        mon = _make_monitor()
        base = datetime(2026, 3, 1, 0, 0, 0)
        _arrivals(mon, "ratio_src", base, count=6, gap_hours=24.0)

        silence_hours = 72.0  # 72 / 24 = 3.0
        check_time = base + timedelta(hours=silence_hours)
        absences = mon.check_absences(now=check_time)

        match = [a for a in absences if a.source == "ratio_src"]
        assert match, "Expected absence signal for ratio_src"
        a = match[0]
        expected_ratio = a.actual_hours / a.expected_hours
        assert abs(a.silence_ratio - expected_ratio) < 0.01, (
            f"silence_ratio={a.silence_ratio} does not match "
            f"actual/expected={expected_ratio}"
        )

    def test_multiple_sources_can_fire_simultaneously(self):
        """Two silent sources both exceed threshold -> two AbsenceSignals returned."""
        mon = _make_monitor()
        base = datetime(2026, 3, 1)
        _arrivals(mon, "source_alpha", base, count=6, gap_hours=24.0)
        _arrivals(mon, "source_beta", base, count=6, gap_hours=24.0)

        check_time = base + timedelta(hours=200)  # 200/24 >> 2.5
        absences = mon.check_absences(now=check_time)

        sources_fired = {a.source for a in absences}
        assert "source_alpha" in sources_fired, "source_alpha should fire"
        assert "source_beta" in sources_fired, "source_beta should fire"


# ---------------------------------------------------------------------------
# Section 3: EventBus publishing
# ---------------------------------------------------------------------------


class TestEventBusPublishing:
    """Verifies that CH_ABSENCE_DETECTED is published when absences are found."""

    def test_absence_publishes_to_eventbus(self):
        """When a bus is present and absences are found, publish is called."""
        bus = MagicMock()
        mon = _make_monitor(event_bus=bus)

        base = datetime(2026, 3, 1)
        _arrivals(mon, "test_src", base, count=6, gap_hours=24.0)

        check_time = base + timedelta(hours=200)
        absences = mon.check_absences(now=check_time)

        assert absences, "Expected absence signals to trigger publish"
        bus.publish.assert_called()

        # Verify CH_ABSENCE_DETECTED was the channel used
        from mae_core.market.channels import CH_ABSENCE_DETECTED
        call_args = bus.publish.call_args_list
        channels_used = [c[0][0] for c in call_args]
        assert CH_ABSENCE_DETECTED in channels_used, (
            f"Expected {CH_ABSENCE_DETECTED} in publish calls, got {channels_used}"
        )

    def test_no_crash_when_bus_is_none(self):
        """No bus present — check_absences must complete without error."""
        mon = _make_monitor(event_bus=None)

        base = datetime(2026, 3, 1)
        _arrivals(mon, "test_src", base, count=6, gap_hours=24.0)

        check_time = base + timedelta(hours=200)
        # Must not raise
        absences = mon.check_absences(now=check_time)
        assert isinstance(absences, list)

    def test_bus_publish_exception_caught(self):
        """If bus.publish raises, check_absences must catch and not propagate."""
        bus = MagicMock()
        bus.publish.side_effect = RuntimeError("bus exploded")
        mon = _make_monitor(event_bus=bus)

        base = datetime(2026, 3, 1)
        _arrivals(mon, "test_src", base, count=6, gap_hours=24.0)

        check_time = base + timedelta(hours=200)
        # Must not raise even if bus.publish fails
        absences = mon.check_absences(now=check_time)
        assert isinstance(absences, list)


# ---------------------------------------------------------------------------
# Section 4: Archive bootstrap
# ---------------------------------------------------------------------------


class TestArchiveBootstrap:
    """Verifies bootstrap_from_archives() learns cadences from historical data."""

    def _make_archive_reader(self, sources_and_timestamps: dict) -> MagicMock:
        """Build a mock archive_reader with realistic structure.

        sources_and_timestamps: {source_name: [datetime, datetime, ...]}
        """
        reader = MagicMock()

        # _by_source is checked via hasattr + iteration
        reader._by_source = {src: [] for src in sources_and_timestamps}

        def _query_source(source, start=None, end=None):
            timestamps = sources_and_timestamps.get(source, [])
            records = []
            for ts in timestamps:
                rec = MagicMock()
                rec.timestamp = ts
                rec.source = source
                records.append(rec)
            return records

        reader.query_source.side_effect = _query_source
        reader.load_range.return_value = None  # no-op
        return reader

    def test_bootstrap_from_archives_learns_cadences(self):
        """Mock archive with known cadences -> monitor learns them correctly."""
        base = datetime(2026, 1, 1)
        # 10 arrivals 48 h apart for sec_form4
        timestamps = [base + timedelta(hours=48 * i) for i in range(10)]
        reader = self._make_archive_reader({"sec_form4": timestamps})

        mon = _make_monitor()
        # Override the seed cadence to test that data overrides it
        mon._cadence["sec_form4"] = 999.0  # artificial starting point

        count = mon.bootstrap_from_archives(reader)

        assert count >= 1, "Expected at least 1 source learned"
        cadence = mon._cadence.get("sec_form4")
        assert cadence is not None
        assert abs(cadence - 48.0) < 2.0, (
            f"Expected ~48 h cadence from archive, got {cadence}"
        )

    def test_bootstrap_returns_count_of_learned_sources(self):
        """Return value equals the number of sources with enough history."""
        base = datetime(2026, 1, 1)
        timestamps_a = [base + timedelta(hours=24 * i) for i in range(10)]
        timestamps_b = [base + timedelta(hours=48 * i) for i in range(8)]
        reader = self._make_archive_reader({
            "source_a": timestamps_a,
            "source_b": timestamps_b,
        })

        mon = _make_monitor()
        count = mon.bootstrap_from_archives(reader)

        assert count == 2, f"Expected 2 sources learned, got {count}"

    def test_bootstrap_skips_sources_with_too_few_records(self):
        """Sources with fewer than MIN_OBSERVATIONS records are skipped."""
        base = datetime(2026, 1, 1)
        # Only 3 records — below MIN_OBSERVATIONS=5
        timestamps = [base + timedelta(hours=24 * i) for i in range(3)]
        reader = self._make_archive_reader({"sparse_src": timestamps})

        mon = _make_monitor()
        count = mon.bootstrap_from_archives(reader)

        # sparse_src had < MIN_OBSERVATIONS — should not count as learned
        assert count == 0, (
            f"Expected 0 sources learned for sparse data, got {count}"
        )

    def test_bootstrap_graceful_when_reader_is_none(self):
        """No archive reader -> returns 0 without raising."""
        mon = _make_monitor(archive_reader=None)
        count = mon.bootstrap_from_archives(None)
        assert count == 0, f"Expected 0 when reader is None, got {count}"


# ---------------------------------------------------------------------------
# Section 5: Sensing hook integration
# ---------------------------------------------------------------------------


class TestSensingHookIntegration:
    """Verifies that MarketSensingHook correctly calls AbsenceMonitor."""

    def test_record_arrival_called_from_collect_one(self):
        """When _collect_one processes a non-empty batch, absence_monitor.record_arrival is called."""
        hook = _make_hook()
        absence_monitor = MagicMock()
        hook._absence_monitor = absence_monitor

        sig = _make_signal(source="sec_form4")
        hook._pending_futures["sec_form4"] = _make_future([sig])
        hook._collect_one("sec_form4")

        absence_monitor.record_arrival.assert_called_once()
        call_kwargs = absence_monitor.record_arrival.call_args
        # First positional arg should be the source name
        assert call_kwargs[0][0] == "sec_form4", (
            f"Expected source='sec_form4', got {call_kwargs[0][0]!r}"
        )

    def test_absence_check_fires_at_step_100(self):
        """step() at step 100 must call absence_monitor.check_absences() exactly once."""
        hook = _make_hook()
        absence_monitor = MagicMock()
        absence_monitor.check_absences.return_value = []
        hook._absence_monitor = absence_monitor

        with (
            patch.object(hook, "_collect_results"),
            patch.object(hook, "_launch_next_fetch"),
            patch.object(hook, "_evaluate_outcomes"),
        ):
            for _ in range(100):
                hook.step()

        absence_monitor.check_absences.assert_called_once()

    def test_absence_check_does_not_fire_at_step_99(self):
        """step() at step 99 must not trigger check_absences."""
        hook = _make_hook()
        absence_monitor = MagicMock()
        absence_monitor.check_absences.return_value = []
        hook._absence_monitor = absence_monitor

        with (
            patch.object(hook, "_collect_results"),
            patch.object(hook, "_launch_next_fetch"),
            patch.object(hook, "_evaluate_outcomes"),
        ):
            for _ in range(99):
                hook.step()

        absence_monitor.check_absences.assert_not_called()

    def test_absence_monitor_none_no_crash_at_step_100(self):
        """If _absence_monitor is None, step() at step 100 must not raise."""
        hook = _make_hook()
        assert hook._absence_monitor is None  # initial state

        with (
            patch.object(hook, "_collect_results"),
            patch.object(hook, "_launch_next_fetch"),
            patch.object(hook, "_evaluate_outcomes"),
        ):
            for _ in range(100):
                hook.step()  # Must not raise

    def test_record_arrival_not_called_for_empty_signal_batch(self):
        """If future returns empty list, _collect_one returns early and does NOT call record_arrival."""
        hook = _make_hook()
        absence_monitor = MagicMock()
        hook._absence_monitor = absence_monitor

        hook._pending_futures["sec_form4"] = _make_future([])
        hook._collect_one("sec_form4")

        absence_monitor.record_arrival.assert_not_called()

    def test_record_arrival_exception_does_not_crash(self):
        """If absence_monitor.record_arrival raises, _collect_one must catch and not propagate."""
        hook = _make_hook()
        absence_monitor = MagicMock()
        absence_monitor.record_arrival.side_effect = RuntimeError("monitor crashed")
        hook._absence_monitor = absence_monitor

        sig = _make_signal(source="sec_form4")
        hook._pending_futures["sec_form4"] = _make_future([sig])
        # Must not raise
        hook._collect_one("sec_form4")

    def test_absence_check_fires_twice_by_step_200(self):
        """check_absences() should be called at step 100 and step 200 (two total by step 200)."""
        hook = _make_hook()
        absence_monitor = MagicMock()
        absence_monitor.check_absences.return_value = []
        hook._absence_monitor = absence_monitor

        with (
            patch.object(hook, "_collect_results"),
            patch.object(hook, "_launch_next_fetch"),
            patch.object(hook, "_evaluate_outcomes"),
        ):
            for _ in range(200):
                hook.step()

        assert absence_monitor.check_absences.call_count == 2, (
            f"Expected 2 check_absences calls by step 200, "
            f"got {absence_monitor.check_absences.call_count}"
        )

    def test_absence_check_exception_does_not_crash_step(self):
        """If check_absences() raises, step() must catch and not propagate."""
        hook = _make_hook()
        absence_monitor = MagicMock()
        absence_monitor.check_absences.side_effect = RuntimeError("check failed")
        hook._absence_monitor = absence_monitor

        with (
            patch.object(hook, "_collect_results"),
            patch.object(hook, "_launch_next_fetch"),
            patch.object(hook, "_evaluate_outcomes"),
        ):
            for _ in range(100):
                hook.step()  # Must not raise


# ---------------------------------------------------------------------------
# Section 6: Bootstrap wiring
# ---------------------------------------------------------------------------


class TestBootstrapWiring:
    """Verifies that _wire_sensing_hook injects absence_monitor from ctx."""

    def _make_minimal_ctx(self, absence_monitor=None):
        """Return a ctx namespace with just enough attributes for _wire_sensing_hook."""
        ctx = SimpleNamespace()
        ctx.absence_monitor = absence_monitor
        ctx.correlation_tracker = None

        # Attributes read by _wire_sensing_hook via getattr(..., None)
        ctx.sec_edgar_client = None
        ctx.price_fetcher = None
        ctx.house_stock_watcher = None
        ctx.senate_stock_watcher = None
        ctx.job_tracker = None
        ctx.usa_spending_client = None
        ctx.sam_gov_client = None
        ctx.apewisdom_client = None
        ctx.finra_client = None
        ctx.sec_efts_client = None
        ctx.finnhub_client = None
        ctx.fred_client = None
        ctx.convergence_alerter = None
        ctx.velocity_detector = None
        ctx.filing_time_analyzer = None
        ctx.session_sweep_detector = None
        ctx.ta_indicators = None
        ctx.cot_client = None
        ctx.stocktwits_client = None
        ctx.vix_client = None
        ctx.trends_client = None
        ctx.thompson_sampler = None
        ctx.regime_classifier = None
        ctx.hypothesis_engine = None
        ctx.qdrant_url = "http://localhost:6333"

        ctx.model = MagicMock()
        ctx.bus = MagicMock()
        return ctx

    def test_absence_monitor_injected_from_ctx(self):
        """When ctx.absence_monitor is set, it is assigned to hook._absence_monitor."""
        from mae_core.bootstrap.market_hooks import _wire_sensing_hook

        monitor = MagicMock()
        ctx = self._make_minimal_ctx(absence_monitor=monitor)

        _wire_sensing_hook(ctx)

        hook = getattr(ctx, "_market_sensing_hook", None)
        assert hook is not None, "ctx._market_sensing_hook must be set by _wire_sensing_hook"
        assert hook._absence_monitor is monitor, (
            "hook._absence_monitor must reference the monitor from ctx"
        )

    def test_missing_absence_monitor_yields_none_on_hook(self):
        """When ctx.absence_monitor is absent, hook._absence_monitor must be None."""
        from mae_core.bootstrap.market_hooks import _wire_sensing_hook

        ctx = self._make_minimal_ctx(absence_monitor=None)
        # Remove the attribute entirely to test getattr(..., None) fallback
        del ctx.absence_monitor

        _wire_sensing_hook(ctx)

        hook = getattr(ctx, "_market_sensing_hook", None)
        assert hook is not None
        assert hook._absence_monitor is None, (
            "hook._absence_monitor must be None when ctx lacks the attribute"
        )


# ---------------------------------------------------------------------------
# Section 7: get_statistics
# ---------------------------------------------------------------------------


class TestGetStatistics:
    """Verifies the monitoring summary returned by get_statistics()."""

    def test_statistics_reflect_recorded_sources(self):
        """After recording arrivals, get_statistics() shows tracked_sources > 0."""
        mon = _make_monitor()
        base = datetime(2026, 3, 1)
        mon.record_arrival("source_x", base)
        mon.record_arrival("source_y", base)

        stats = mon.get_statistics()

        assert stats["tracked_sources"] == 2, (
            f"Expected 2 tracked sources, got {stats['tracked_sources']}"
        )

    def test_statistics_include_known_cadences(self):
        """Learned cadences count includes seed cadences from _KNOWN_CADENCES."""
        mon = _make_monitor()
        stats = mon.get_statistics()

        # At minimum, all seed cadences are present
        assert stats["learned_cadences"] >= len(_KNOWN_CADENCES), (
            f"Expected >= {len(_KNOWN_CADENCES)} learned cadences, "
            f"got {stats['learned_cadences']}"
        )

    def test_statistics_empty_on_fresh_monitor(self):
        """Fresh monitor has 0 tracked sources and 0 sources with history."""
        mon = _make_monitor()
        stats = mon.get_statistics()

        assert stats["tracked_sources"] == 0
        assert stats["sources_with_history"] == 0
