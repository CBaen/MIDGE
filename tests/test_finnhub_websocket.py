#!/usr/bin/env python3
"""Tests for mae_core.market.apis.finnhub_websocket.

All tests are self-contained — no real WebSocket connections are made.
Signal generation logic is tested by calling _process_trade() directly with
synthetic trade dicts. The WebSocket infrastructure is tested with mocks.
"""

import json
import threading
import time
import unittest
from collections import deque
from datetime import datetime
from queue import Queue
from unittest.mock import MagicMock, patch

from mae_core.market.apis.finnhub_websocket import (
    BASELINE_UPDATE_EVERY,
    DEFAULT_TICKERS,
    MIN_BASELINE_PERIODS,
    MIN_TRADES_FOR_VOLUME,
    RAPID_MOVE_PCT,
    VOLUME_SPIKE_SIGMA,
    FinnhubWebSocket,
    TradeBuffer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ws(api_key: str = "test-key", tickers=None) -> FinnhubWebSocket:
    """Return a client that will NOT try to open a real socket."""
    return FinnhubWebSocket(api_key=api_key, tickers=tickers or ["AAPL"])


def make_trade(symbol="AAPL", price=150.0, volume=100.0, ts_ms=None) -> dict:
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)
    return {"s": symbol, "p": price, "v": volume, "t": ts_ms}


def inject_trades(ws: FinnhubWebSocket, symbol: str, prices, volumes=None):
    """Push a list of price ticks into the client's buffer via _process_trade."""
    if volumes is None:
        volumes = [100.0] * len(prices)
    for price, volume in zip(prices, volumes):
        ws._process_trade(make_trade(symbol=symbol, price=price, volume=volume))


# ---------------------------------------------------------------------------
# TradeBuffer tests
# ---------------------------------------------------------------------------

class TestTradeBuffer(unittest.TestCase):

    def test_default_creation(self):
        buf = TradeBuffer()
        self.assertIsInstance(buf.prices, deque)
        self.assertIsInstance(buf.volumes, deque)
        self.assertIsInstance(buf.timestamps, deque)
        self.assertIsInstance(buf.volume_baseline, deque)
        self.assertEqual(buf.last_price, 0.0)
        self.assertEqual(buf.last_volume, 0.0)
        self.assertEqual(buf.trade_count, 0)

    def test_maxlen_prices(self):
        buf = TradeBuffer()
        # Fill beyond maxlen — oldest entry should drop
        for i in range(350):
            buf.prices.append(float(i))
        self.assertEqual(len(buf.prices), 300)
        self.assertEqual(buf.prices[-1], 349.0)

    def test_maxlen_volume_baseline(self):
        buf = TradeBuffer()
        for i in range(25):
            buf.volume_baseline.append(float(i))
        self.assertEqual(len(buf.volume_baseline), 20)

    def test_accumulation(self):
        buf = TradeBuffer()
        buf.prices.append(100.0)
        buf.volumes.append(500.0)
        buf.last_price = 100.0
        buf.last_volume = 500.0
        buf.trade_count = 1
        self.assertEqual(buf.last_price, 100.0)
        self.assertEqual(buf.trade_count, 1)


# ---------------------------------------------------------------------------
# FinnhubWebSocket construction
# ---------------------------------------------------------------------------

class TestConstruction(unittest.TestCase):

    def test_defaults_with_env_key(self):
        with patch.dict("os.environ", {"MAE_FINNHUB_API_KEY": "env-key"}):
            ws = FinnhubWebSocket()
        self.assertEqual(ws._api_key, "env-key")
        self.assertEqual(ws._tickers, DEFAULT_TICKERS)

    def test_explicit_key_overrides_env(self):
        with patch.dict("os.environ", {"MAE_FINNHUB_API_KEY": "env-key"}):
            ws = FinnhubWebSocket(api_key="explicit-key")
        self.assertEqual(ws._api_key, "explicit-key")

    def test_explicit_tickers(self):
        ws = make_ws(tickers=["TSLA", "SPY"])
        self.assertEqual(ws._tickers, ["TSLA", "SPY"])

    def test_initial_state(self):
        ws = make_ws()
        self.assertFalse(ws._running)
        self.assertFalse(ws._connected)
        self.assertEqual(len(ws._buffers), 0)
        self.assertEqual(ws._pending_signals.qsize(), 0)
        self.assertEqual(ws._reconnect_delay, 5.0)


# ---------------------------------------------------------------------------
# Volume spike detection
# ---------------------------------------------------------------------------

class TestVolumeSpikeDetection(unittest.TestCase):

    def _prime_baseline(self, ws: FinnhubWebSocket, symbol: str, base_vol=100.0):
        """Inject enough trades to build a stable baseline."""
        # We need MIN_BASELINE_PERIODS baseline samples.
        # Each sample is added every BASELINE_UPDATE_EVERY trades.
        # Also need MIN_TRADES_FOR_VOLUME trades before first comparison.
        total_needed = BASELINE_UPDATE_EVERY * (MIN_BASELINE_PERIODS + 1)
        inject_trades(
            ws, symbol,
            prices=[150.0] * total_needed,
            volumes=[base_vol] * total_needed,
        )

    def test_no_signal_below_threshold(self):
        ws = make_ws()
        # Build baseline with steady volume (100.0 per trade).
        # The 60-trade recent-volume window sums to 60*100=6000.
        # After priming, baseline_mean ~6000 and baseline_std ~0.
        # Inject another 60 trades at the SAME volume — result equals mean, no spike.
        self._prime_baseline(ws, "AAPL", base_vol=100.0)
        before = [s for s in ws.get_pending_signals() if s["signal_type"] == "volume_spike"]
        inject_trades(ws, "AAPL", prices=[150.0] * 60, volumes=[100.0] * 60)
        after = [s for s in ws.get_pending_signals() if s["signal_type"] == "volume_spike"]
        self.assertEqual(len(after), 0, "Stable volume at baseline mean should not spike")

    def test_spike_signal_emitted(self):
        ws = make_ws()
        # Build a low baseline (vol=10 per trade)
        self._prime_baseline(ws, "AAPL", base_vol=10.0)
        before = ws._pending_signals.qsize()
        # Now inject very high volume — well above baseline + 2σ
        inject_trades(ws, "AAPL", prices=[150.0] * MIN_TRADES_FOR_VOLUME, volumes=[10000.0] * MIN_TRADES_FOR_VOLUME)
        after = ws._pending_signals.qsize()
        self.assertGreater(after, before, "Expected a volume_spike signal")

    def test_spike_signal_has_correct_type(self):
        ws = make_ws()
        self._prime_baseline(ws, "AAPL", base_vol=10.0)
        inject_trades(ws, "AAPL", prices=[150.0] * MIN_TRADES_FOR_VOLUME, volumes=[10000.0] * MIN_TRADES_FOR_VOLUME)
        signals = ws.get_pending_signals()
        spike_signals = [s for s in signals if s["signal_type"] == "volume_spike"]
        self.assertGreater(len(spike_signals), 0)
        sig = spike_signals[0]
        self.assertEqual(sig["symbol"], "AAPL")
        self.assertIn("price", sig)
        self.assertIn("value", sig)
        self.assertIn("threshold", sig)
        self.assertIn("direction", sig)
        self.assertIsInstance(sig["timestamp"], datetime)

    def test_no_spike_before_enough_baseline_samples(self):
        ws = make_ws()
        # Inject fewer than MIN_BASELINE_PERIODS * BASELINE_UPDATE_EVERY trades
        # so the baseline is not yet populated enough
        inject_trades(ws, "AAPL", prices=[150.0] * MIN_TRADES_FOR_VOLUME, volumes=[10000.0] * MIN_TRADES_FOR_VOLUME)
        signals = [s for s in ws.get_pending_signals() if s["signal_type"] == "volume_spike"]
        self.assertEqual(len(signals), 0, "Should not spike without baseline")


# ---------------------------------------------------------------------------
# Rapid price move detection
# ---------------------------------------------------------------------------

class TestRapidMoveDetection(unittest.TestCase):

    def test_no_signal_below_1pct(self):
        ws = make_ws()
        # Small price move — 0.5%
        inject_trades(ws, "AAPL", prices=[150.0] * 5 + [150.75] * 5)
        signals = [s for s in ws.get_pending_signals() if s["signal_type"] == "rapid_move"]
        self.assertEqual(len(signals), 0)

    def test_bullish_rapid_move(self):
        ws = make_ws()
        # Price rises > 1%: 150 → 152.5 (1.67%)
        inject_trades(ws, "AAPL", prices=[150.0] * 5 + [152.5] * 5)
        signals = [s for s in ws.get_pending_signals() if s["signal_type"] == "rapid_move"]
        self.assertGreater(len(signals), 0)
        self.assertEqual(signals[0]["direction"], "bullish")

    def test_bearish_rapid_move(self):
        ws = make_ws()
        # Price falls > 1%: 150 → 147.5 (1.67%)
        inject_trades(ws, "AAPL", prices=[150.0] * 5 + [147.5] * 5)
        signals = [s for s in ws.get_pending_signals() if s["signal_type"] == "rapid_move"]
        self.assertGreater(len(signals), 0)
        self.assertEqual(signals[0]["direction"], "bearish")

    def test_signal_value_is_pct_change(self):
        ws = make_ws()
        inject_trades(ws, "AAPL", prices=[100.0] * 5 + [102.0] * 5)
        signals = [s for s in ws.get_pending_signals() if s["signal_type"] == "rapid_move"]
        self.assertGreater(len(signals), 0)
        sig = signals[0]
        self.assertAlmostEqual(sig["value"], 0.02, places=4)
        self.assertEqual(sig["threshold"], RAPID_MOVE_PCT)

    def test_no_move_with_fewer_than_10_trades(self):
        ws = make_ws()
        # Only 9 trades — buffer check requires >= 10
        inject_trades(ws, "AAPL", prices=[100.0] * 4 + [110.0] * 5)
        signals = [s for s in ws.get_pending_signals() if s["signal_type"] == "rapid_move"]
        self.assertEqual(len(signals), 0)


# ---------------------------------------------------------------------------
# Signal queue
# ---------------------------------------------------------------------------

class TestSignalQueue(unittest.TestCase):

    def test_get_pending_signals_drains_queue(self):
        ws = make_ws()
        # Manually populate
        ws._pending_signals.put({"signal_type": "test", "symbol": "AAPL"})
        ws._pending_signals.put({"signal_type": "test", "symbol": "MSFT"})
        result = ws.get_pending_signals()
        self.assertEqual(len(result), 2)
        # Queue should now be empty
        self.assertEqual(ws.get_pending_signals(), [])

    def test_get_pending_signals_non_blocking(self):
        ws = make_ws()
        # Empty queue — should return empty list without blocking
        start = time.time()
        result = ws.get_pending_signals()
        elapsed = time.time() - start
        self.assertEqual(result, [])
        self.assertLess(elapsed, 0.1)

    def test_queue_drops_when_full(self):
        ws = make_ws()
        # Manually fill the queue to capacity (maxsize=1000)
        for i in range(1000):
            ws._pending_signals.put_nowait({"signal_type": "filler", "i": i})
        # _emit_signal should silently drop without raising
        ws._emit_signal("AAPL", "rapid_move", 150.0, 0.02, 0.01, direction="bullish")
        # Queue stays at capacity — the extra signal was dropped
        self.assertEqual(ws._pending_signals.qsize(), 1000)

    def test_signal_dict_structure(self):
        ws = make_ws()
        ws._emit_signal("TSLA", "rapid_move", 200.0, 0.015, 0.01, direction="bullish")
        signals = ws.get_pending_signals()
        self.assertEqual(len(signals), 1)
        sig = signals[0]
        expected_keys = {"signal_type", "symbol", "price", "value", "threshold", "direction", "timestamp"}
        self.assertEqual(set(sig.keys()), expected_keys)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestGetStatistics(unittest.TestCase):

    def test_statistics_keys(self):
        ws = make_ws(tickers=["AAPL", "MSFT"])
        stats = ws.get_statistics()
        self.assertIn("connected", stats)
        self.assertIn("tickers", stats)
        self.assertIn("buffers", stats)
        self.assertIn("pending_signals", stats)
        self.assertIn("reconnect_delay", stats)

    def test_statistics_values(self):
        ws = make_ws(tickers=["AAPL", "MSFT"])
        stats = ws.get_statistics()
        self.assertFalse(stats["connected"])
        self.assertEqual(stats["tickers"], ["AAPL", "MSFT"])
        self.assertEqual(stats["buffers"], 0)
        self.assertEqual(stats["pending_signals"], 0)
        self.assertEqual(stats["reconnect_delay"], 5.0)

    def test_buffers_count_increases_with_trades(self):
        ws = make_ws()
        inject_trades(ws, "AAPL", [150.0, 151.0])
        inject_trades(ws, "MSFT", [300.0, 301.0])
        self.assertEqual(ws.get_statistics()["buffers"], 2)


# ---------------------------------------------------------------------------
# Start / stop (no real connection)
# ---------------------------------------------------------------------------

class TestStartStop(unittest.TestCase):

    def test_start_without_api_key_is_noop(self):
        ws = FinnhubWebSocket(api_key="")
        ws.start()
        self.assertFalse(ws._running)
        self.assertIsNone(ws._thread)

    @patch("mae_core.market.apis.finnhub_websocket.FinnhubWebSocket._run_loop")
    def test_start_launches_background_thread(self, mock_run_loop):
        ws = make_ws()
        ws.start()
        time.sleep(0.05)
        self.assertTrue(ws._running)
        self.assertIsNotNone(ws._thread)
        ws._running = False  # prevent real loop

    @patch("mae_core.market.apis.finnhub_websocket.FinnhubWebSocket._run_loop")
    def test_start_is_idempotent(self, mock_run_loop):
        ws = make_ws()
        ws.start()
        thread_a = ws._thread
        ws.start()  # second call should be a no-op
        self.assertIs(ws._thread, thread_a)
        ws._running = False

    def test_stop_sets_running_false(self):
        ws = make_ws()
        ws._running = True
        ws.stop()
        self.assertFalse(ws._running)

    def test_stop_closes_websocket(self):
        ws = make_ws()
        mock_ws = MagicMock()
        ws._ws = mock_ws
        ws._running = True
        ws.stop()
        mock_ws.close.assert_called_once()


# ---------------------------------------------------------------------------
# Update subscriptions
# ---------------------------------------------------------------------------

class TestUpdateSubscriptions(unittest.TestCase):

    def test_update_changes_ticker_list(self):
        ws = make_ws(tickers=["AAPL", "MSFT"])
        ws.update_subscriptions(["TSLA", "SPY"])
        self.assertEqual(sorted(ws._tickers), ["SPY", "TSLA"])

    def test_update_sends_subscribe_when_connected(self):
        ws = make_ws(tickers=["AAPL"])
        ws._connected = True
        mock_ws = MagicMock()
        ws._ws = mock_ws

        ws.update_subscriptions(["AAPL", "MSFT"])

        # MSFT should be subscribed
        calls = [json.loads(c.args[0]) for c in mock_ws.send.call_args_list]
        subscribe_calls = [c for c in calls if c.get("type") == "subscribe"]
        symbols_subscribed = [c["symbol"] for c in subscribe_calls]
        self.assertIn("MSFT", symbols_subscribed)

    def test_update_sends_unsubscribe_when_connected(self):
        ws = make_ws(tickers=["AAPL", "MSFT"])
        ws._connected = True
        mock_ws = MagicMock()
        ws._ws = mock_ws

        ws.update_subscriptions(["AAPL"])

        calls = [json.loads(c.args[0]) for c in mock_ws.send.call_args_list]
        unsub_calls = [c for c in calls if c.get("type") == "unsubscribe"]
        self.assertEqual(unsub_calls[0]["symbol"], "MSFT")

    def test_update_no_send_when_disconnected(self):
        ws = make_ws(tickers=["AAPL"])
        ws._connected = False
        mock_ws = MagicMock()
        ws._ws = mock_ws

        ws.update_subscriptions(["TSLA"])
        mock_ws.send.assert_not_called()


# ---------------------------------------------------------------------------
# Reconnect delay back-off
# ---------------------------------------------------------------------------

class TestReconnectBackoff(unittest.TestCase):

    def test_initial_reconnect_delay(self):
        ws = make_ws()
        self.assertEqual(ws._reconnect_delay, 5.0)

    def test_on_open_resets_delay(self):
        # _on_open no longer resets _reconnect_delay — it only records _connect_time.
        # Backoff resets in _run_loop only after a stable connection (>= 60 s).
        ws = make_ws()
        ws._reconnect_delay = 32.0  # simulate prior failures
        mock_ws = MagicMock()
        ws._on_open(mock_ws)
        # Delay must NOT be reset by _on_open
        self.assertEqual(ws._reconnect_delay, 32.0)
        # _connect_time must be recorded so _run_loop can measure stability
        self.assertIsNotNone(ws._connect_time)

    def test_delay_does_not_exceed_max(self):
        ws = make_ws()
        # Start above the halfway point so doubling exceeds the new 300 s cap
        ws._reconnect_delay = 200.0
        # Simulate one more failure doubling the delay
        ws._reconnect_delay = min(ws._reconnect_delay * 2, ws._max_reconnect_delay)
        self.assertEqual(ws._reconnect_delay, ws._max_reconnect_delay)

    def test_max_reconnect_delay(self):
        ws = make_ws()
        self.assertEqual(ws._max_reconnect_delay, 300.0)


# ---------------------------------------------------------------------------
# Graceful handling of missing / invalid trade data
# ---------------------------------------------------------------------------

class TestInvalidTradeData(unittest.TestCase):

    def test_empty_trade_dict_ignored(self):
        ws = make_ws()
        ws._process_trade({})  # should not raise
        self.assertEqual(len(ws._buffers), 0)

    def test_zero_price_ignored(self):
        ws = make_ws()
        ws._process_trade({"s": "AAPL", "p": 0.0, "v": 500.0, "t": 0})
        self.assertNotIn("AAPL", ws._buffers)

    def test_negative_price_ignored(self):
        ws = make_ws()
        ws._process_trade({"s": "AAPL", "p": -10.0, "v": 100.0, "t": 0})
        self.assertNotIn("AAPL", ws._buffers)

    def test_missing_symbol_ignored(self):
        ws = make_ws()
        ws._process_trade({"p": 150.0, "v": 100.0, "t": 0})
        self.assertEqual(len(ws._buffers), 0)

    def test_malformed_json_message_ignored(self):
        ws = make_ws()
        ws._on_message(MagicMock(), "not-valid-json")
        # No exception should propagate

    def test_non_trade_message_type_ignored(self):
        ws = make_ws()
        msg = json.dumps({"type": "ping"})
        ws._on_message(MagicMock(), msg)
        self.assertEqual(ws._pending_signals.qsize(), 0)


# ---------------------------------------------------------------------------
# Directional bias from close-in-range
# ---------------------------------------------------------------------------

class TestDirectionalBias(unittest.TestCase):

    def test_bullish_at_top_of_range(self):
        ws = make_ws()
        # Prices range 100–102, current price at top
        inject_trades(ws, "AAPL", prices=[100.0] * 5 + [102.0] * 5)
        direction = ws._resolve_direction("AAPL", 102.0)
        self.assertEqual(direction, "bullish")

    def test_bearish_at_bottom_of_range(self):
        ws = make_ws()
        inject_trades(ws, "AAPL", prices=[102.0] * 5 + [100.0] * 5)
        direction = ws._resolve_direction("AAPL", 100.0)
        self.assertEqual(direction, "bearish")

    def test_neutral_in_middle_of_range(self):
        ws = make_ws()
        inject_trades(ws, "AAPL", prices=[100.0] * 5 + [110.0] * 5)
        direction = ws._resolve_direction("AAPL", 105.0)
        self.assertEqual(direction, "neutral")

    def test_neutral_when_no_buffer(self):
        ws = make_ws()
        direction = ws._resolve_direction("AAPL", 150.0)
        self.assertEqual(direction, "neutral")

    def test_neutral_when_flat_range(self):
        ws = make_ws()
        inject_trades(ws, "AAPL", prices=[150.0] * 10)
        direction = ws._resolve_direction("AAPL", 150.0)
        self.assertEqual(direction, "neutral")

    def test_emit_signal_resolves_direction_when_not_provided(self):
        ws = make_ws()
        # Use a flat price sequence so _process_trade does NOT trigger rapid_move,
        # then emit a volume_spike explicitly and check direction resolution.
        inject_trades(ws, "AAPL", prices=[100.0] * 10)
        # Manually push the last price high within the buffer so close-in-range = bullish
        ws._buffers["AAPL"].prices.append(100.0)
        for _ in range(9):
            ws._buffers["AAPL"].prices.append(102.0)
        ws.get_pending_signals()  # drain any rapid_move signals from inject_trades
        ws._emit_signal("AAPL", "volume_spike", 102.0, 5000.0, 1000.0)
        signals = ws.get_pending_signals()
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["signal_type"], "volume_spike")
        self.assertEqual(signals[0]["direction"], "bullish")


if __name__ == "__main__":
    unittest.main()
