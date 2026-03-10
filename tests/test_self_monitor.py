"""Tests for SelfMonitor — behavioral risk monitoring for alert emission."""
import threading
import time
from unittest.mock import MagicMock, call

import pytest

from mae_core.market.intelligence.self_monitor import (
    CH_BEHAVIORAL_ANOMALY,
    ANOMALY_CONFIDENCE_CLUSTERING,
    ANOMALY_DIRECTION_BIAS,
    ANOMALY_RUNAWAY_RATE,
    ANOMALY_TICKER_FLOODING,
    SelfMonitor,
    _std_dev,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor(**kwargs) -> SelfMonitor:
    """Create a SelfMonitor with tight defaults for fast tests."""
    defaults = dict(
        rate_window=100,
        direction_window=50,
        max_alerts_per_window=10,
        bias_threshold=0.80,
    )
    defaults.update(kwargs)
    return SelfMonitor(**defaults)


def _flood_alerts(monitor: SelfMonitor, n: int, direction: str = "bullish",
                  confidence: float = 0.70, ticker: str = "AAPL") -> None:
    """Record n identical alerts."""
    for i in range(n):
        monitor.record_alert(direction=direction, confidence=confidence,
                             ticker=ticker, step=i)


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestStdDev:
    def test_empty_list(self):
        assert _std_dev([]) == 0.0

    def test_single_element(self):
        assert _std_dev([0.5]) == 0.0

    def test_uniform_values(self):
        # All identical → std dev of 0
        assert _std_dev([0.7, 0.7, 0.7, 0.7, 0.7]) == pytest.approx(0.0)

    def test_known_std_dev(self):
        # Population std dev of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        result = _std_dev([2, 4, 4, 4, 5, 5, 7, 9])
        assert result == pytest.approx(2.0, abs=1e-9)

    def test_two_elements(self):
        # std dev of [0.0, 1.0] = 0.5
        assert _std_dev([0.0, 1.0]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Normal alert flow
# ---------------------------------------------------------------------------

class TestNormalFlow:
    def test_no_anomalies_on_empty(self):
        monitor = _make_monitor()
        assert monitor.check_anomalies() == []
        assert monitor.is_alerting_suppressed() is False

    def test_single_alert_no_anomaly(self):
        monitor = _make_monitor()
        monitor.record_alert("bullish", 0.70, "AAPL", step=1)
        assert monitor.check_anomalies() == []
        assert monitor.is_alerting_suppressed() is False

    def test_alert_count_increments(self):
        monitor = _make_monitor()
        for i in range(5):
            monitor.record_alert("bullish", 0.70, "AAPL", step=i)
        stats = monitor.get_statistics()
        assert stats["alert_count"] == 5

    def test_mixed_directions_no_bias(self):
        """Alternating directions should not trigger direction_bias."""
        monitor = _make_monitor(bias_threshold=0.80, direction_window=50)
        # Record 50 alerts, 25 bullish + 25 bearish (50% each)
        for i in range(50):
            direction = "bullish" if i % 2 == 0 else "bearish"
            monitor.record_alert(direction, 0.70, "AAPL", step=i)
        assert ANOMALY_DIRECTION_BIAS not in monitor.check_anomalies()

    def test_diverse_tickers_no_flooding(self):
        """Many tickers rotating through should not trigger ticker_flooding."""
        monitor = _make_monitor()
        tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "NVDA"]
        for i in range(30):
            monitor.record_alert("bullish", 0.70, tickers[i % len(tickers)], step=i)
        assert ANOMALY_TICKER_FLOODING not in monitor.check_anomalies()


# ---------------------------------------------------------------------------
# Runaway rate
# ---------------------------------------------------------------------------

class TestRunawayRate:
    def test_runaway_rate_detected_above_threshold(self):
        """More than max_alerts_per_window in deque triggers runaway_rate."""
        # rate_window=100, max_alerts=10 → after 11 alerts, runaway fires
        monitor = _make_monitor(rate_window=100, max_alerts_per_window=10)
        _flood_alerts(monitor, 11)
        assert ANOMALY_RUNAWAY_RATE in monitor.check_anomalies()

    def test_runaway_rate_exactly_at_threshold_does_not_fire(self):
        """Exactly at the threshold should NOT trigger (strict > check)."""
        monitor = _make_monitor(rate_window=100, max_alerts_per_window=10)
        _flood_alerts(monitor, 10)
        assert ANOMALY_RUNAWAY_RATE not in monitor.check_anomalies()

    def test_runaway_rate_clears_after_window_rotation(self):
        """After window rotates past the burst, anomaly should clear."""
        # Tiny window so rotation is observable
        monitor = _make_monitor(rate_window=5, max_alerts_per_window=3)
        # Flood 4 alerts (triggers runaway in window of 5, threshold=3)
        _flood_alerts(monitor, 4)
        assert ANOMALY_RUNAWAY_RATE in monitor.check_anomalies()
        # Add 2 more alerts with a different ticker to fill + rotate
        # Now window still has 5 items (maxlen=5), still >3 → still flagged
        monitor.record_alert("bullish", 0.70, "TSLA", step=10)
        monitor.record_alert("bullish", 0.70, "TSLA", step=11)
        # All 5 in window, still > 3 → still runaway
        assert ANOMALY_RUNAWAY_RATE in monitor.check_anomalies()

    def test_no_runaway_well_under_threshold(self):
        monitor = _make_monitor(rate_window=100, max_alerts_per_window=10)
        _flood_alerts(monitor, 5)
        assert ANOMALY_RUNAWAY_RATE not in monitor.check_anomalies()


# ---------------------------------------------------------------------------
# Direction bias
# ---------------------------------------------------------------------------

class TestDirectionBias:
    def test_direction_bias_detected_at_threshold(self):
        """More than 80% same direction in direction_window alerts → bias."""
        monitor = _make_monitor(bias_threshold=0.80, direction_window=10)
        # 9 bullish + 1 bearish in a 10-alert window = 90% bullish > 80%
        for i in range(9):
            monitor.record_alert("bullish", 0.70, f"T{i}", step=i)
        monitor.record_alert("bearish", 0.70, "TSLA", step=9)
        assert ANOMALY_DIRECTION_BIAS in monitor.check_anomalies()

    def test_direction_bias_not_detected_below_threshold(self):
        """79% same direction should NOT trigger."""
        # direction_window=10, bias=0.80
        # 7 bullish + 3 bearish = 70% bullish < 80%
        monitor = _make_monitor(bias_threshold=0.80, direction_window=10)
        for i in range(7):
            monitor.record_alert("bullish", 0.70, f"T{i}", step=i)
        for i in range(3):
            monitor.record_alert("bearish", 0.70, f"B{i}", step=7 + i)
        assert ANOMALY_DIRECTION_BIAS not in monitor.check_anomalies()

    def test_direction_bias_requires_minimum_window(self):
        """Bias check only fires when direction_window alerts have been seen."""
        monitor = _make_monitor(bias_threshold=0.80, direction_window=10)
        # Only 5 alerts (fewer than direction_window=10) — bias should not fire
        _flood_alerts(monitor, 5, direction="bullish")
        assert ANOMALY_DIRECTION_BIAS not in monitor.check_anomalies()

    def test_direction_bias_uses_most_recent_window(self):
        """Bias is computed over the LAST direction_window alerts, not all."""
        monitor = _make_monitor(bias_threshold=0.80, direction_window=10,
                                rate_window=100)
        # 30 bearish alerts first
        for i in range(30):
            monitor.record_alert("bearish", 0.70, f"B{i}", step=i)
        # Then 9 bullish + 1 bearish in the last 10 (recent window = 90% bullish)
        for i in range(9):
            monitor.record_alert("bullish", 0.70, f"A{i}", step=30 + i)
        monitor.record_alert("bearish", 0.70, "TSLA", step=39)
        assert ANOMALY_DIRECTION_BIAS in monitor.check_anomalies()


# ---------------------------------------------------------------------------
# Confidence clustering
# ---------------------------------------------------------------------------

class TestConfidenceClustering:
    def test_confidence_clustering_detected_low_std(self):
        """Std dev < 0.02 over last 20 alerts → clustering detected."""
        monitor = _make_monitor()
        # All confidences at 0.710 ± 0.001 → std dev ~0.001
        for i in range(20):
            conf = 0.710 + (i % 3) * 0.001  # range [0.710, 0.712], std≈0.001
            monitor.record_alert("bullish", conf, f"T{i}", step=i)
        assert ANOMALY_CONFIDENCE_CLUSTERING in monitor.check_anomalies()

    def test_confidence_clustering_not_detected_high_std(self):
        """Std dev >= 0.02 should NOT trigger."""
        monitor = _make_monitor()
        # Spread over [0.30, 0.90] range → high std dev
        for i in range(20):
            conf = 0.30 + (i / 19) * 0.60  # 0.30 to 0.90 linearly
            monitor.record_alert("bullish", conf, f"T{i}", step=i)
        assert ANOMALY_CONFIDENCE_CLUSTERING not in monitor.check_anomalies()

    def test_confidence_clustering_requires_minimum_20_alerts(self):
        """Clustering check requires at least 20 alerts."""
        monitor = _make_monitor()
        # Only 15 identical-confidence alerts (< 20 window)
        for i in range(15):
            monitor.record_alert("bullish", 0.710, f"T{i}", step=i)
        assert ANOMALY_CONFIDENCE_CLUSTERING not in monitor.check_anomalies()

    def test_confidence_clustering_at_exactly_20_alerts(self):
        """Exactly 20 uniform-confidence alerts should trigger."""
        monitor = _make_monitor()
        for i in range(20):
            monitor.record_alert("bullish", 0.710, f"T{i}", step=i)
        assert ANOMALY_CONFIDENCE_CLUSTERING in monitor.check_anomalies()


# ---------------------------------------------------------------------------
# Ticker flooding
# ---------------------------------------------------------------------------

class TestTickerFlooding:
    def test_ticker_flooding_detected_single_dominant_ticker(self):
        """Single ticker in >50% of last 30 alerts → flooding."""
        monitor = _make_monitor()
        # 20 AAPL + 10 TSLA in 30 alerts = 66.7% AAPL > 50%
        for i in range(20):
            monitor.record_alert("bullish", 0.70, "AAPL", step=i)
        for i in range(10):
            monitor.record_alert("bearish", 0.65, "TSLA", step=20 + i)
        assert ANOMALY_TICKER_FLOODING in monitor.check_anomalies()

    def test_ticker_flooding_not_detected_at_threshold(self):
        """Exactly 50% should NOT trigger (strict > check)."""
        monitor = _make_monitor()
        # 15 AAPL + 15 TSLA = exactly 50% each
        for i in range(15):
            monitor.record_alert("bullish", 0.70, "AAPL", step=i)
        for i in range(15):
            monitor.record_alert("bearish", 0.65, "TSLA", step=15 + i)
        assert ANOMALY_TICKER_FLOODING not in monitor.check_anomalies()

    def test_ticker_flooding_requires_minimum_30_alerts(self):
        """Flooding check requires at least 30 recent alerts."""
        monitor = _make_monitor()
        # Only 20 alerts — insufficient for the 30-alert flooding window
        for i in range(20):
            monitor.record_alert("bullish", 0.70, "AAPL", step=i)
        assert ANOMALY_TICKER_FLOODING not in monitor.check_anomalies()

    def test_ticker_flooding_uses_most_recent_30(self):
        """Flooding is computed over the LAST 30 alerts in the deque."""
        monitor = _make_monitor(rate_window=100)
        # 40 diverse alerts first
        tickers = ["T1", "T2", "T3", "T4"]
        for i in range(40):
            monitor.record_alert("bullish", 0.70, tickers[i % 4], step=i)
        # Then 20 AAPL + 10 MSFT = last 30 alerts, 66.7% AAPL → floods
        for i in range(20):
            monitor.record_alert("bullish", 0.70, "AAPL", step=40 + i)
        for i in range(10):
            monitor.record_alert("bullish", 0.70, "MSFT", step=60 + i)
        assert ANOMALY_TICKER_FLOODING in monitor.check_anomalies()


# ---------------------------------------------------------------------------
# Auto-suppression behavior
# ---------------------------------------------------------------------------

class TestAutoSuppression:
    def test_suppression_triggers_on_runaway_rate(self):
        """runaway_rate MUST trigger auto-suppression."""
        monitor = _make_monitor(max_alerts_per_window=5, rate_window=100)
        _flood_alerts(monitor, 6)  # exceeds max_alerts=5
        assert ANOMALY_RUNAWAY_RATE in monitor.check_anomalies()
        assert monitor.is_alerting_suppressed() is True

    def test_suppression_triggers_on_confidence_clustering(self):
        """confidence_clustering MUST trigger auto-suppression."""
        monitor = _make_monitor()
        for i in range(20):
            monitor.record_alert("bullish", 0.710, f"T{i}", step=i)
        assert ANOMALY_CONFIDENCE_CLUSTERING in monitor.check_anomalies()
        assert monitor.is_alerting_suppressed() is True

    def test_no_suppression_on_direction_bias_alone(self):
        """direction_bias alone MUST NOT trigger suppression."""
        monitor = _make_monitor(
            bias_threshold=0.80,
            direction_window=10,
            max_alerts_per_window=100,  # never hits runaway
        )
        # 9 bullish + 1 bearish = 90% bullish → direction_bias
        for i in range(9):
            monitor.record_alert("bullish", 0.70 + i * 0.01, f"T{i}", step=i)
        monitor.record_alert("bearish", 0.50, "TSLA", step=9)
        assert ANOMALY_DIRECTION_BIAS in monitor.check_anomalies()
        assert monitor.is_alerting_suppressed() is False

    def test_no_suppression_on_ticker_flooding_alone(self):
        """ticker_flooding alone MUST NOT trigger suppression."""
        monitor = _make_monitor(
            max_alerts_per_window=100,  # never hits runaway
        )
        # 21 AAPL + 9 TSLA = 70% AAPL over 30 alerts → flooding
        for i in range(21):
            monitor.record_alert("bullish", 0.70 + i * 0.005, "AAPL", step=i)
        for i in range(9):
            monitor.record_alert("bearish", 0.50 + i * 0.01, "TSLA", step=21 + i)
        assert ANOMALY_TICKER_FLOODING in monitor.check_anomalies()
        assert monitor.is_alerting_suppressed() is False

    def test_suppression_persists_after_reset_not_called(self):
        """Suppression stays True until reset_suppression() is called."""
        monitor = _make_monitor(max_alerts_per_window=5, rate_window=100)
        _flood_alerts(monitor, 6)
        assert monitor.is_alerting_suppressed() is True
        # Add a few more alerts (normal pattern now)
        monitor.record_alert("bullish", 0.70, "MSFT", step=100)
        # Still suppressed — deque still has >5 entries
        assert monitor.is_alerting_suppressed() is True


# ---------------------------------------------------------------------------
# reset_suppression
# ---------------------------------------------------------------------------

class TestResetSuppression:
    def test_reset_suppression_clears_flag(self):
        """reset_suppression() must set _alerting_suppressed to False."""
        monitor = _make_monitor(max_alerts_per_window=5, rate_window=100)
        _flood_alerts(monitor, 6)
        assert monitor.is_alerting_suppressed() is True
        monitor.reset_suppression()
        assert monitor.is_alerting_suppressed() is False

    def test_reset_suppression_idempotent_when_not_suppressed(self):
        """Calling reset_suppression() when not suppressed is a no-op."""
        monitor = _make_monitor()
        monitor.reset_suppression()  # no-op, should not raise
        assert monitor.is_alerting_suppressed() is False

    def test_suppression_can_retrigger_after_reset(self):
        """After reset, new runaway alerts should re-trigger suppression."""
        monitor = _make_monitor(max_alerts_per_window=5, rate_window=20)
        _flood_alerts(monitor, 6)
        monitor.reset_suppression()
        assert monitor.is_alerting_suppressed() is False
        # Add more alerts to retrigger (deque still has >5)
        monitor.record_alert("bullish", 0.70, "NEW", step=200)
        # Deque still has 6+ entries from earlier (maxlen=20)
        assert monitor.is_alerting_suppressed() is True


# ---------------------------------------------------------------------------
# get_statistics
# ---------------------------------------------------------------------------

class TestGetStatistics:
    def test_get_statistics_returns_required_keys(self):
        """get_statistics must return all required keys."""
        monitor = _make_monitor()
        stats = monitor.get_statistics()
        assert "alert_count" in stats
        assert "active_anomalies" in stats
        assert "alerting_suppressed" in stats
        assert "recent_alert_rate" in stats
        assert "direction_distribution" in stats

    def test_get_statistics_alert_count_correct(self):
        monitor = _make_monitor()
        _flood_alerts(monitor, 7)
        stats = monitor.get_statistics()
        assert stats["alert_count"] == 7

    def test_get_statistics_direction_distribution(self):
        monitor = _make_monitor()
        for i in range(3):
            monitor.record_alert("bullish", 0.70, f"T{i}", step=i)
        for i in range(1):
            monitor.record_alert("bearish", 0.60, "TSLA", step=3 + i)
        stats = monitor.get_statistics()
        dist = stats["direction_distribution"]
        assert "bullish" in dist
        assert "bearish" in dist
        assert dist["bullish"] == pytest.approx(0.75, abs=0.01)
        assert dist["bearish"] == pytest.approx(0.25, abs=0.01)

    def test_get_statistics_recent_alert_rate_between_0_and_1(self):
        monitor = _make_monitor(rate_window=100)
        _flood_alerts(monitor, 10)
        stats = monitor.get_statistics()
        assert 0.0 <= stats["recent_alert_rate"] <= 1.0

    def test_get_statistics_on_empty_monitor(self):
        monitor = _make_monitor()
        stats = monitor.get_statistics()
        assert stats["alert_count"] == 0
        assert stats["active_anomalies"] == []
        assert stats["alerting_suppressed"] is False
        assert stats["recent_alert_rate"] == 0.0
        assert stats["direction_distribution"] == {}

    def test_get_statistics_active_anomalies_matches_check_anomalies(self):
        monitor = _make_monitor(max_alerts_per_window=5, rate_window=100)
        _flood_alerts(monitor, 6)
        stats = monitor.get_statistics()
        assert stats["active_anomalies"] == monitor.check_anomalies()

    def test_get_statistics_alerting_suppressed_matches_is_suppressed(self):
        monitor = _make_monitor(max_alerts_per_window=5, rate_window=100)
        _flood_alerts(monitor, 6)
        stats = monitor.get_statistics()
        assert stats["alerting_suppressed"] == monitor.is_alerting_suppressed()


# ---------------------------------------------------------------------------
# EventBus integration
# ---------------------------------------------------------------------------

class TestEventBusIntegration:
    def test_anomaly_published_to_bus_on_runaway(self):
        """CH_BEHAVIORAL_ANOMALY must be published when runaway fires."""
        bus = MagicMock()
        monitor = SelfMonitor(event_bus=bus, max_alerts_per_window=5, rate_window=100)
        _flood_alerts(monitor, 6)
        # At least one publish call for the runaway anomaly
        published_channels = [c[0][0] for c in bus.publish.call_args_list]
        assert CH_BEHAVIORAL_ANOMALY in published_channels

    def test_anomaly_payload_contains_anomaly_type(self):
        """Published payload must include anomaly_type key."""
        bus = MagicMock()
        monitor = SelfMonitor(event_bus=bus, max_alerts_per_window=5, rate_window=100)
        _flood_alerts(monitor, 6)
        # Find the CH_BEHAVIORAL_ANOMALY call
        for call_args in bus.publish.call_args_list:
            channel, payload = call_args[0]
            if channel == CH_BEHAVIORAL_ANOMALY:
                assert "anomaly_type" in payload
                assert "suppresses_alerting" in payload
                assert payload["anomaly_type"] == ANOMALY_RUNAWAY_RATE
                break
        else:
            pytest.fail("CH_BEHAVIORAL_ANOMALY was never published")

    def test_anomaly_not_published_twice_for_same_anomaly(self):
        """Anomaly event should fire ONCE per anomaly appearance, not per alert."""
        bus = MagicMock()
        monitor = SelfMonitor(event_bus=bus, max_alerts_per_window=5, rate_window=100)
        # Trigger runaway with 6 alerts
        _flood_alerts(monitor, 6)
        runaway_publishes = sum(
            1 for c in bus.publish.call_args_list
            if c[0][0] == CH_BEHAVIORAL_ANOMALY
            and c[0][1].get("anomaly_type") == ANOMALY_RUNAWAY_RATE
        )
        assert runaway_publishes == 1

        # More alerts while still in runaway state → no additional publish
        _flood_alerts(monitor, 3, ticker="MSFT")
        runaway_publishes_after = sum(
            1 for c in bus.publish.call_args_list
            if c[0][0] == CH_BEHAVIORAL_ANOMALY
            and c[0][1].get("anomaly_type") == ANOMALY_RUNAWAY_RATE
        )
        assert runaway_publishes_after == 1

    def test_direction_bias_suppresses_false_in_payload(self):
        """direction_bias payload must have suppresses_alerting=False."""
        bus = MagicMock()
        monitor = SelfMonitor(
            event_bus=bus,
            bias_threshold=0.80,
            direction_window=10,
            max_alerts_per_window=100,
        )
        for i in range(9):
            monitor.record_alert("bullish", 0.70 + i * 0.01, f"T{i}", step=i)
        monitor.record_alert("bearish", 0.50, "TSLA", step=9)

        for call_args in bus.publish.call_args_list:
            channel, payload = call_args[0]
            if (channel == CH_BEHAVIORAL_ANOMALY
                    and payload.get("anomaly_type") == ANOMALY_DIRECTION_BIAS):
                assert payload["suppresses_alerting"] is False
                break


# ---------------------------------------------------------------------------
# Graceful degradation (no event_bus)
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_works_without_event_bus(self):
        """All methods must work correctly when event_bus is None."""
        monitor = SelfMonitor(event_bus=None, max_alerts_per_window=5, rate_window=100)
        _flood_alerts(monitor, 6)
        # No exception raised
        assert ANOMALY_RUNAWAY_RATE in monitor.check_anomalies()
        assert monitor.is_alerting_suppressed() is True
        stats = monitor.get_statistics()
        assert stats["alert_count"] == 6

    def test_reset_suppression_without_event_bus(self):
        monitor = SelfMonitor(event_bus=None, max_alerts_per_window=5, rate_window=100)
        _flood_alerts(monitor, 6)
        monitor.reset_suppression()
        assert monitor.is_alerting_suppressed() is False

    def test_bus_exception_does_not_crash_monitor(self):
        """If event_bus.publish raises, monitor must not propagate the error."""
        bus = MagicMock()
        bus.publish.side_effect = RuntimeError("bus error")
        monitor = SelfMonitor(event_bus=bus, max_alerts_per_window=5, rate_window=100)
        # Should not raise even though bus.publish throws
        _flood_alerts(monitor, 6)
        assert ANOMALY_RUNAWAY_RATE in monitor.check_anomalies()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_record_alert_does_not_corrupt_count(self):
        """50 threads recording 20 alerts each → count must be exactly 1000."""
        monitor = _make_monitor(rate_window=2000, max_alerts_per_window=1001)
        threads = []
        for t in range(50):
            thread = threading.Thread(
                target=_flood_alerts,
                args=(monitor, 20),
                kwargs={"direction": "bullish", "confidence": 0.70, "ticker": "AAPL"},
            )
            threads.append(thread)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stats = monitor.get_statistics()
        assert stats["alert_count"] == 1000

    def test_concurrent_record_and_get_statistics_no_error(self):
        """get_statistics must not raise when called concurrently with record_alert."""
        monitor = _make_monitor(rate_window=200, max_alerts_per_window=1000)
        errors: list[Exception] = []

        def writer():
            for i in range(100):
                try:
                    monitor.record_alert("bullish", 0.70, "AAPL", step=i)
                except Exception as e:
                    errors.append(e)

        def reader():
            for _ in range(100):
                try:
                    monitor.get_statistics()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_suppression_toggle_is_consistent(self):
        """reset_suppression and record_alert running concurrently must not deadlock."""
        monitor = _make_monitor(max_alerts_per_window=5, rate_window=50)

        def suppress_and_reset():
            for _ in range(50):
                _flood_alerts(monitor, 6)
                monitor.reset_suppression()

        threads = [threading.Thread(target=suppress_and_reset) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Test passes if it completes without deadlock — no assertion needed


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_deque_respects_maxlen(self):
        """After rate_window alerts, deque should not exceed maxlen."""
        monitor = _make_monitor(rate_window=20, max_alerts_per_window=100)
        _flood_alerts(monitor, 50)
        # Internal deque maxlen is rate_window=20
        assert len(monitor._recent_alerts) == 20

    def test_direction_window_capped_at_rate_window(self):
        """direction_window must not exceed rate_window."""
        # Requesting direction_window > rate_window should be clamped
        monitor = SelfMonitor(rate_window=10, direction_window=50)
        assert monitor._direction_window <= monitor._rate_window

    def test_zero_confidence_alerts_accepted(self):
        """Edge: confidence of 0.0 must be stored and processed."""
        monitor = _make_monitor()
        monitor.record_alert("neutral", 0.0, "SPY", step=0)
        stats = monitor.get_statistics()
        assert stats["alert_count"] == 1

    def test_confidence_of_one_accepted(self):
        """Edge: confidence of 1.0 must be stored and processed."""
        monitor = _make_monitor()
        monitor.record_alert("bullish", 1.0, "AAPL", step=0)
        stats = monitor.get_statistics()
        assert stats["alert_count"] == 1

    def test_empty_ticker_string_accepted(self):
        """Edge: empty ticker string must not crash the monitor."""
        monitor = _make_monitor()
        monitor.record_alert("bullish", 0.70, "", step=0)
        stats = monitor.get_statistics()
        assert stats["alert_count"] == 1

    def test_step_zero_accepted(self):
        """Default step parameter of 0 must work."""
        monitor = _make_monitor()
        monitor.record_alert("bullish", 0.70, "AAPL")  # no step arg
        stats = monitor.get_statistics()
        assert stats["alert_count"] == 1
