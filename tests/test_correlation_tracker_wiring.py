"""Tests for CorrelationTracker wiring in MarketSensingHook.

Two changes under test:

1. sensing_hook.py (_collect_one):
   After feeding signals to the convergence engine, if
   self._correlation_tracker is not None AND signals is non-empty,
   calls:
     self._correlation_tracker.record(
         signal_id=source_name,
         value=max(sig.strength for sig in signals),
         timestamp=datetime.now(),
         domain=signals[0].domain,
     )

2. sensing_hook.py (step):
   At step_counter % 200 == 0, if self._correlation_tracker is not None,
   calls:
     self._correlation_tracker.detect_cross_domain_anomalies()

3. market_hooks.py (_wire_sensing_hook):
   After MarketSensingHook is constructed:
     hook._correlation_tracker = getattr(ctx, "correlation_tracker", None)

Test strategy:
- Construct MarketSensingHook with all constructor params set to None
  (the class gracefully degrades on None). Inject mocks manually
  for the two dependencies under test: _correlation_tracker and
  any future passed to _pending_futures.
- Never call _launch_next_fetch() or touch ThreadPoolExecutor — tests
  drive _collect_one() and step() directly to avoid network I/O.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.sensing_hook import MarketSensingHook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hook(**overrides) -> MarketSensingHook:
    """Construct a MarketSensingHook with all params set to None.

    Passes fetch_cadence=50 and outcome_cadence=200 (defaults).  Any
    kwarg in overrides is forwarded to the constructor.  The caller is
    responsible for setting _correlation_tracker after construction.
    """
    defaults = dict(
        sec_client=None,
        price_fetcher=None,
        congress_client=None,
        senate_client=None,
        job_tracker=None,
        usa_spending=None,
        sam_gov=None,
        apewisdom=None,
        finra_client=None,
        sec_efts=None,
        finnhub=None,
        fred=None,
        convergence_alerter=None,
        velocity_detector=None,
        filing_analyzer=None,
        form8k_sentiment=None,
        session_sweep_detector=None,
        ta_indicators=None,
        cot_client=None,
        stocktwits_client=None,
        vix_client=None,
        trends_client=None,
        outcome_collector=None,
        memory=None,
        thompson_sampler=None,
        tiered_alerters=None,
        watchlist={},
        fetch_cadence=50,
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
    """Build a minimal mock that satisfies MarketSensingHook's _collect_one() reads.

    Attributes accessed inside _collect_one:
      sig.signal_id, sig.strength, sig.domain, sig.direction,
      sig.confidence, sig.velocity, sig.timestamp, sig.metadata,
      sig.symbol, sig.source
    """
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
    """Return a completed Future-like mock that yields *signals*."""
    fut = MagicMock()
    fut.done.return_value = True
    fut.result.return_value = signals
    return fut


# ---------------------------------------------------------------------------
# Class 1: CorrelationTracker receives data via _collect_one
# ---------------------------------------------------------------------------


class TestCorrelationTrackerRecording:
    """_collect_one() must call tracker.record() with correct args when
    a non-empty signal batch arrives and the tracker is not None."""

    # ── 1. record() called when signals arrive ─────────────────────────────

    def test_record_called_on_signal_batch(self):
        """When _collect_one processes a non-empty batch, tracker.record()
        must be called exactly once."""
        hook = _make_hook()
        tracker = MagicMock()
        hook._correlation_tracker = tracker

        sig = _make_signal(strength=0.75, domain="insider", source="sec_form4")
        hook._pending_futures["sec_form4"] = _make_future([sig])
        hook._collect_one("sec_form4")

        tracker.record.assert_called_once()

    # ── 2. signal_id argument is the source name ───────────────────────────

    def test_record_signal_id_is_source_name(self):
        """tracker.record() must be called with signal_id=source_name,
        NOT the individual signal's signal_id field."""
        hook = _make_hook()
        tracker = MagicMock()
        hook._correlation_tracker = tracker

        sig = _make_signal(signal_id="individual-id-123", source="finra_short")
        hook._pending_futures["finra_short"] = _make_future([sig])
        hook._collect_one("finra_short")

        call_kwargs = tracker.record.call_args.kwargs
        assert call_kwargs["signal_id"] == "finra_short", (
            f"Expected signal_id='finra_short', got {call_kwargs['signal_id']!r}"
        )

    # ── 3. value is max strength across the batch ──────────────────────────

    def test_record_value_is_max_strength(self):
        """When multiple signals arrive, record() must receive the maximum
        strength value across all signals in the batch."""
        hook = _make_hook()
        tracker = MagicMock()
        hook._correlation_tracker = tracker

        sigs = [
            _make_signal(signal_id="s1", strength=0.60, source="congressional"),
            _make_signal(signal_id="s2", strength=0.85, source="congressional"),
            _make_signal(signal_id="s3", strength=0.72, source="congressional"),
        ]
        hook._pending_futures["congressional"] = _make_future(sigs)
        hook._collect_one("congressional")

        call_kwargs = tracker.record.call_args.kwargs
        assert call_kwargs["value"] == pytest.approx(0.85), (
            f"Expected value=0.85 (max strength), got {call_kwargs['value']}"
        )

    # ── 4. domain comes from first signal in the batch ─────────────────────

    def test_record_domain_from_first_signal(self):
        """tracker.record() must receive domain=signals[0].domain.
        The first signal in the list determines the domain key."""
        hook = _make_hook()
        tracker = MagicMock()
        hook._correlation_tracker = tracker

        sig = _make_signal(domain="positioning", source="cot_positioning")
        hook._pending_futures["cot_positioning"] = _make_future([sig])
        hook._collect_one("cot_positioning")

        call_kwargs = tracker.record.call_args.kwargs
        assert call_kwargs["domain"] == "positioning", (
            f"Expected domain='positioning', got {call_kwargs['domain']!r}"
        )

    # ── 5. domain fallback when signal lacks domain attribute ────────────

    def test_record_domain_none_when_signal_lacks_domain_attr(self):
        """If the first signal has no 'domain' attribute, record() must
        receive domain=None rather than crashing. This exercises the
        hasattr(signals[0], 'domain') guard at sensing_hook.py:484."""
        hook = _make_hook()
        tracker = MagicMock()
        hook._correlation_tracker = tracker

        # Build a signal mock without a 'domain' attribute
        sig = MagicMock(spec=[])  # empty spec = no attributes
        sig.signal_id = "sig_no_domain"
        sig.strength = 0.65
        sig.direction = "bullish"
        sig.source = "sec_form4"
        sig.symbol = "AAPL"
        sig.confidence = 0.6
        sig.velocity = 0.0
        sig.timestamp = datetime.now()
        sig.metadata = {}
        # Explicitly do NOT set sig.domain

        hook._pending_futures["sec_form4"] = _make_future([sig])
        hook._collect_one("sec_form4")

        tracker.record.assert_called_once()
        call_kwargs = tracker.record.call_args.kwargs
        assert call_kwargs["domain"] is None, (
            f"Expected domain=None when signal lacks domain attr, "
            f"got {call_kwargs['domain']!r}"
        )

    # ── 6. Empty signal list does not call record ──────────────────────────

    def test_empty_signals_does_not_call_record(self):
        """If the completed future returns an empty list, _collect_one returns
        early before the CorrelationTracker path is reached."""
        hook = _make_hook()
        tracker = MagicMock()
        hook._correlation_tracker = tracker

        hook._pending_futures["sec_form4"] = _make_future([])
        hook._collect_one("sec_form4")

        tracker.record.assert_not_called()

    # ── 6. None tracker does not crash ────────────────────────────────────

    def test_none_tracker_does_not_crash(self):
        """When _correlation_tracker is None, _collect_one must complete
        without raising. This is the default state before bootstrap wires it."""
        hook = _make_hook()
        assert hook._correlation_tracker is None  # verifies initial state

        sig = _make_signal(strength=0.7, source="sec_form4")
        hook._pending_futures["sec_form4"] = _make_future([sig])

        # Must not raise
        hook._collect_one("sec_form4")

    # ── 7. record() exception is swallowed, no crash ───────────────────────

    def test_record_exception_does_not_crash(self):
        """If tracker.record() raises (e.g. malformed data), _collect_one
        must catch and log the error rather than propagating it."""
        hook = _make_hook()
        tracker = MagicMock()
        tracker.record.side_effect = RuntimeError("tracker exploded")
        hook._correlation_tracker = tracker

        sig = _make_signal(strength=0.8, source="sec_form4")
        hook._pending_futures["sec_form4"] = _make_future([sig])

        # Must not raise
        hook._collect_one("sec_form4")


# ---------------------------------------------------------------------------
# Class 2: Anomaly detection cadence in step()
# ---------------------------------------------------------------------------


class TestAnomalyDetectionCadence:
    """step() must call detect_cross_domain_anomalies() at multiples of 200
    and must NOT call it at any other step."""

    # ── 8. Anomaly scan runs exactly at step 200 ──────────────────────────

    def test_anomaly_scan_fires_at_step_200(self):
        """Advancing the hook to step 200 must trigger one call to
        detect_cross_domain_anomalies() on the tracker."""
        hook = _make_hook()
        tracker = MagicMock()
        tracker.detect_cross_domain_anomalies.return_value = []
        hook._correlation_tracker = tracker

        # Drive the step counter to 200 by calling step() 200 times.
        # Patch _collect_results and _launch_next_fetch to avoid I/O.
        with (
            patch.object(hook, "_collect_results"),
            patch.object(hook, "_launch_next_fetch"),
            patch.object(hook, "_evaluate_outcomes"),
        ):
            for _ in range(200):
                hook.step()

        tracker.detect_cross_domain_anomalies.assert_called_once()

    # ── 9. Anomaly scan does NOT fire at step 199 ─────────────────────────

    def test_anomaly_scan_does_not_fire_at_step_199(self):
        """At step 199, detect_cross_domain_anomalies() must not be called."""
        hook = _make_hook()
        tracker = MagicMock()
        tracker.detect_cross_domain_anomalies.return_value = []
        hook._correlation_tracker = tracker

        with (
            patch.object(hook, "_collect_results"),
            patch.object(hook, "_launch_next_fetch"),
            patch.object(hook, "_evaluate_outcomes"),
        ):
            for _ in range(199):
                hook.step()

        tracker.detect_cross_domain_anomalies.assert_not_called()

    # ── 10. Anomaly scan fires at step 400 (second multiple) ──────────────

    def test_anomaly_scan_fires_at_step_400(self):
        """At step 400, detect_cross_domain_anomalies() must be called a
        second time (200 + 400 = two calls total by step 400)."""
        hook = _make_hook()
        tracker = MagicMock()
        tracker.detect_cross_domain_anomalies.return_value = []
        hook._correlation_tracker = tracker

        with (
            patch.object(hook, "_collect_results"),
            patch.object(hook, "_launch_next_fetch"),
            patch.object(hook, "_evaluate_outcomes"),
        ):
            for _ in range(400):
                hook.step()

        assert tracker.detect_cross_domain_anomalies.call_count == 2, (
            f"Expected 2 anomaly scans by step 400, "
            f"got {tracker.detect_cross_domain_anomalies.call_count}"
        )

    # ── 11. Anomaly scan does not fire when tracker is None ───────────────

    def test_anomaly_scan_skipped_when_tracker_is_none(self):
        """step() at step 200 with _correlation_tracker=None must not raise
        and must not attempt any anomaly scan."""
        hook = _make_hook()
        assert hook._correlation_tracker is None

        with (
            patch.object(hook, "_collect_results"),
            patch.object(hook, "_launch_next_fetch"),
            patch.object(hook, "_evaluate_outcomes"),
        ):
            for _ in range(200):
                hook.step()  # Must not raise

    # ── 12. detect_cross_domain_anomalies exception is swallowed ──────────

    def test_anomaly_scan_exception_does_not_crash_step(self):
        """If detect_cross_domain_anomalies() raises, step() must catch and
        log the error rather than propagating it to the caller."""
        hook = _make_hook()
        tracker = MagicMock()
        tracker.detect_cross_domain_anomalies.side_effect = RuntimeError("scan failed")
        hook._correlation_tracker = tracker

        with (
            patch.object(hook, "_collect_results"),
            patch.object(hook, "_launch_next_fetch"),
            patch.object(hook, "_evaluate_outcomes"),
        ):
            for _ in range(200):
                hook.step()  # Must not raise


# ---------------------------------------------------------------------------
# Class 3: Bootstrap wiring (_wire_sensing_hook injects the tracker)
# ---------------------------------------------------------------------------


class TestBootstrapWiring:
    """_wire_sensing_hook() must set hook._correlation_tracker from ctx.

    These tests verify the wiring contract without running the full
    33-layer bootstrap: we call _wire_sensing_hook() directly with a
    minimal ctx SimpleNamespace.
    """

    def _make_minimal_ctx(self, correlation_tracker=None):
        """Return a ctx namespace with just enough attributes to satisfy
        _wire_sensing_hook() without crashing."""
        ctx = SimpleNamespace()
        ctx.correlation_tracker = correlation_tracker

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

        # model.add_step_hook must be callable (captures the wrapped hook)
        ctx.model = MagicMock()

        # bus.register_callback must be callable
        ctx.bus = MagicMock()

        return ctx

    # ── 13. tracker from ctx is injected into hook ────────────────────────

    def test_tracker_injected_from_ctx(self):
        """When ctx.correlation_tracker is set, _wire_sensing_hook() must
        assign it to hook._correlation_tracker before registering the step
        hook."""
        from mae_core.bootstrap.market_hooks import _wire_sensing_hook

        tracker = MagicMock()
        ctx = self._make_minimal_ctx(correlation_tracker=tracker)

        _wire_sensing_hook(ctx)

        hook = getattr(ctx, "_market_sensing_hook", None)
        assert hook is not None, "ctx._market_sensing_hook must be set by _wire_sensing_hook"
        assert hook._correlation_tracker is tracker, (
            "hook._correlation_tracker must reference the tracker from ctx"
        )

    # ── 14. None tracker on ctx yields None on hook ────────────────────────

    def test_none_tracker_on_ctx_yields_none_on_hook(self):
        """When ctx.correlation_tracker is absent (None), _wire_sensing_hook()
        must leave hook._correlation_tracker as None rather than raising."""
        from mae_core.bootstrap.market_hooks import _wire_sensing_hook

        ctx = self._make_minimal_ctx(correlation_tracker=None)
        # Remove the attribute entirely to test getattr(..., None) fallback
        del ctx.correlation_tracker

        _wire_sensing_hook(ctx)

        hook = getattr(ctx, "_market_sensing_hook", None)
        assert hook is not None
        assert hook._correlation_tracker is None
