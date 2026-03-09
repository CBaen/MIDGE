"""Tests for signal-triggered reactive convergence in MarketSensingHook.

Verifies that _trigger_reactive_convergence() fires check_convergence() and
check_ticker_convergence() immediately after signals are ingested in _collect_one(),
without requiring a separate step tick.
"""

from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime
from unittest.mock import MagicMock, patch, call
from typing import List

import pytest

from mae_core.market.sensing_hook import MarketSensingHook
from mae_core.market.signal import MarketSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(symbol: str = "AAPL", domain: str = "insider") -> MarketSignal:
    now = datetime.now()
    return MarketSignal(
        signal_id=f"test-{symbol}-{domain}",
        source="test_source",
        symbol=symbol,
        asset_class="stock",
        domain=domain,
        direction="bullish",
        strength=0.7,
        confidence=0.7,
        decay_rate=0.1,
        timestamp=now,
        received_at=now,
        velocity=0.0,
        metadata={"symbol": symbol},
    )


def _make_alert(ticker: str = "AAPL", direction: str = "bullish") -> MagicMock:
    alert = MagicMock()
    alert.direction = direction
    alert.ticker = ticker
    sig = MagicMock()
    sig.metadata = {"symbol": ticker}
    alert.signals = [sig]
    alert.to_dict.return_value = {"ticker": ticker, "direction": direction, "strength": 0.8}
    return alert


def _minimal_hook(**kwargs) -> MarketSensingHook:
    """Build a MarketSensingHook with all optional args defaulted to None."""
    defaults = dict(
        sec_client=None, price_fetcher=None, congress_client=None,
        senate_client=None, job_tracker=None, usa_spending=None,
        sam_gov=None, apewisdom=None, finra_client=None, sec_efts=None,
        finnhub=None, fred=None, cot_client=None, stocktwits_client=None,
        vix_client=None, trends_client=None, convergence_alerter=None,
        velocity_detector=None, filing_analyzer=None, form8k_sentiment=None,
        session_sweep_detector=None, ta_indicators=None, outcome_collector=None,
        memory=None, thompson_sampler=None,
    )
    defaults.update(kwargs)
    return MarketSensingHook(**defaults)


# ---------------------------------------------------------------------------
# _cached_reactive_alerts initialises to empty list
# ---------------------------------------------------------------------------

class TestCachedReactiveAlertsAttribute:
    def test_attribute_exists_on_init(self):
        hook = _minimal_hook()
        assert hasattr(hook, "_cached_reactive_alerts")
        assert hook._cached_reactive_alerts == []


# ---------------------------------------------------------------------------
# _trigger_reactive_convergence — no-ops when alerter is None or signals empty
# ---------------------------------------------------------------------------

class TestTriggerReactiveConvergenceGuards:
    def test_no_alerter_does_nothing(self):
        hook = _minimal_hook()
        signals = [_make_signal()]
        # Should not raise
        hook._trigger_reactive_convergence(signals)
        assert hook._cached_reactive_alerts == []

    def test_empty_signals_does_nothing(self):
        alerter = MagicMock()
        hook = _minimal_hook(convergence_alerter=alerter)
        hook._trigger_reactive_convergence([])
        alerter.check_convergence.assert_not_called()
        alerter.check_ticker_convergence.assert_not_called()

    def test_none_signals_does_nothing(self):
        alerter = MagicMock()
        hook = _minimal_hook(convergence_alerter=alerter)
        # Empty list counts as falsy
        hook._trigger_reactive_convergence([])
        alerter.check_convergence.assert_not_called()


# ---------------------------------------------------------------------------
# _trigger_reactive_convergence — calls check_convergence and caches alerts
# ---------------------------------------------------------------------------

class TestTriggerReactiveConvergenceGlobalCheck:
    def test_calls_check_convergence(self):
        alerter = MagicMock()
        alerter.check_convergence.return_value = []
        alerter.check_ticker_convergence.return_value = []
        hook = _minimal_hook(convergence_alerter=alerter)
        signals = [_make_signal()]
        hook._trigger_reactive_convergence(signals)
        alerter.check_convergence.assert_called_once()

    def test_global_alerts_cached(self):
        alert = _make_alert("AAPL", "bullish")
        alerter = MagicMock()
        alerter.check_convergence.return_value = [alert]
        alerter.check_ticker_convergence.return_value = []
        hook = _minimal_hook(convergence_alerter=alerter)
        hook._trigger_reactive_convergence([_make_signal()])
        assert alert in hook._cached_reactive_alerts

    def test_no_bus_still_caches(self):
        alert = _make_alert()
        alerter = MagicMock()
        alerter.check_convergence.return_value = [alert]
        alerter.check_ticker_convergence.return_value = []
        hook = _minimal_hook(convergence_alerter=alerter)
        hook._bus = None
        hook._trigger_reactive_convergence([_make_signal()])
        assert alert in hook._cached_reactive_alerts


# ---------------------------------------------------------------------------
# _trigger_reactive_convergence — publishes to EventBus when bus is set
# ---------------------------------------------------------------------------

class TestTriggerReactiveConvergenceBusPublish:
    def test_publishes_global_alert_to_bus(self):
        from mae_core.market.channels import CH_CONVERGENCE
        alert = _make_alert("TSLA", "bearish")
        alerter = MagicMock()
        alerter.check_convergence.return_value = [alert]
        alerter.check_ticker_convergence.return_value = []

        bus = MagicMock()
        hook = _minimal_hook(convergence_alerter=alerter)
        hook._bus = bus

        hook._trigger_reactive_convergence([_make_signal("TSLA", "technical")])

        published_channels = [c[0][0] for c in bus.publish.call_args_list]
        assert CH_CONVERGENCE in published_channels

    def test_publishes_alert_dict(self):
        from mae_core.market.channels import CH_CONVERGENCE
        expected_dict = {"ticker": "NVDA", "direction": "bullish", "strength": 0.9}
        alert = MagicMock()
        alert.direction = "bullish"
        alert.to_dict.return_value = expected_dict
        sig = MagicMock()
        sig.metadata = {"symbol": "NVDA"}
        alert.signals = [sig]

        alerter = MagicMock()
        alerter.check_convergence.return_value = [alert]
        alerter.check_ticker_convergence.return_value = []

        bus = MagicMock()
        hook = _minimal_hook(convergence_alerter=alerter)
        hook._bus = bus

        hook._trigger_reactive_convergence([_make_signal("NVDA")])

        bus.publish.assert_called_once_with(CH_CONVERGENCE, expected_dict)


# ---------------------------------------------------------------------------
# _trigger_reactive_convergence — per-ticker check
# ---------------------------------------------------------------------------

class TestTriggerReactiveConvergenceTickerCheck:
    def test_calls_check_ticker_convergence(self):
        alerter = MagicMock()
        alerter.check_convergence.return_value = []
        alerter.check_ticker_convergence.return_value = []
        hook = _minimal_hook(convergence_alerter=alerter)
        hook._trigger_reactive_convergence([_make_signal("SPY")])
        alerter.check_ticker_convergence.assert_called_once()

    def test_ticker_alert_for_matching_symbol_cached(self):
        ticker_alert = _make_alert("MSFT", "bullish")
        alerter = MagicMock()
        alerter.check_convergence.return_value = []
        alerter.check_ticker_convergence.return_value = [ticker_alert]
        hook = _minimal_hook(convergence_alerter=alerter)
        hook._trigger_reactive_convergence([_make_signal("MSFT", "macro")])
        assert ticker_alert in hook._cached_reactive_alerts

    def test_ticker_alert_for_non_matching_symbol_not_cached(self):
        # Alert is for AMZN but signal is for GOOG — should be filtered out
        ticker_alert = _make_alert("AMZN", "bullish")
        alerter = MagicMock()
        alerter.check_convergence.return_value = []
        alerter.check_ticker_convergence.return_value = [ticker_alert]
        hook = _minimal_hook(convergence_alerter=alerter)
        hook._trigger_reactive_convergence([_make_signal("GOOG", "macro")])
        assert ticker_alert not in hook._cached_reactive_alerts

    def test_ticker_alert_publishes_to_bus(self):
        from mae_core.market.channels import CH_CONVERGENCE
        ticker_alert = _make_alert("META", "bearish")
        alerter = MagicMock()
        alerter.check_convergence.return_value = []
        alerter.check_ticker_convergence.return_value = [ticker_alert]
        bus = MagicMock()
        hook = _minimal_hook(convergence_alerter=alerter)
        hook._bus = bus
        hook._trigger_reactive_convergence([_make_signal("META")])
        published_channels = [c[0][0] for c in bus.publish.call_args_list]
        assert CH_CONVERGENCE in published_channels


# ---------------------------------------------------------------------------
# Exception safety — convergence errors must NOT block signal collection
# ---------------------------------------------------------------------------

class TestTriggerReactiveConvergenceExceptionSafety:
    def test_check_convergence_exception_does_not_raise(self):
        alerter = MagicMock()
        alerter.check_convergence.side_effect = RuntimeError("convergence exploded")
        alerter.check_ticker_convergence.return_value = []
        hook = _minimal_hook(convergence_alerter=alerter)
        # Should not raise — try/except must swallow the error
        hook._trigger_reactive_convergence([_make_signal()])
        # Cache should remain empty
        assert hook._cached_reactive_alerts == []

    def test_check_ticker_convergence_exception_does_not_raise(self):
        alerter = MagicMock()
        alerter.check_convergence.return_value = []
        alerter.check_ticker_convergence.side_effect = ValueError("ticker boom")
        hook = _minimal_hook(convergence_alerter=alerter)
        hook._trigger_reactive_convergence([_make_signal()])
        # Global alerts still empty; no raise
        assert hook._cached_reactive_alerts == []

    def test_bus_publish_exception_does_not_raise(self):
        alert = _make_alert()
        alerter = MagicMock()
        alerter.check_convergence.return_value = [alert]
        alerter.check_ticker_convergence.return_value = []
        bus = MagicMock()
        bus.publish.side_effect = RuntimeError("bus dead")
        hook = _minimal_hook(convergence_alerter=alerter)
        hook._bus = bus
        hook._trigger_reactive_convergence([_make_signal()])
        # Alert should still be cached even though publish failed
        assert alert in hook._cached_reactive_alerts


# ---------------------------------------------------------------------------
# Integration: _collect_one calls _trigger_reactive_convergence
# ---------------------------------------------------------------------------

class TestCollectOneCallsTrigger:
    def _make_future(self, signals: list) -> Future:
        f: Future = Future()
        f.set_result(signals)
        return f

    def test_collect_one_triggers_convergence(self, tmp_path):
        alerter = MagicMock()
        alerter.check_convergence.return_value = []
        alerter.check_ticker_convergence.return_value = []

        hook = _minimal_hook(convergence_alerter=alerter)

        signals = [_make_signal("AAPL")]
        hook._pending_futures["test_src"] = self._make_future(signals)

        with (
            patch.object(hook, "_trigger_reactive_convergence") as mock_trigger,
            patch("mae_core.market.sensing_hook.store_signals"),
            patch("mae_core.market.sensing_hook.enrich_signal", side_effect=lambda s, *a, **kw: s),
        ):
            hook._collect_one("test_src")
            mock_trigger.assert_called_once_with(signals)

    def test_collect_one_skips_trigger_on_empty_signals(self):
        alerter = MagicMock()
        hook = _minimal_hook(convergence_alerter=alerter)
        hook._pending_futures["empty_src"] = self._make_future([])

        with patch.object(hook, "_trigger_reactive_convergence") as mock_trigger:
            hook._collect_one("empty_src")
            mock_trigger.assert_not_called()

    def test_collect_one_no_alerter_trigger_still_safe(self, tmp_path):
        hook = _minimal_hook()
        signals = [_make_signal()]
        hook._pending_futures["src"] = self._make_future(signals)

        with (
            patch("mae_core.market.sensing_hook.store_signals"),
            patch("mae_core.market.sensing_hook.enrich_signal", side_effect=lambda s, *a, **kw: s),
        ):
            # Should not raise even without alerter
            hook._collect_one("src")
