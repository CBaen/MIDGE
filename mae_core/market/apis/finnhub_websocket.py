"""Finnhub WebSocket — Real-time trade data streaming.

Connects to Finnhub's free WebSocket API for real-time US stock trades.
Generates signal dicts for volume spikes and rapid price moves.
Runs in a background thread — signals are queued for main thread consumption.

Part of WP-D: Always-On MIDGE Wave 2.

API endpoint: wss://ws.finnhub.io?token={api_key}
Auth: MAE_FINNHUB_API_KEY environment variable (same key used by the REST client).
Requires: pip install websocket-client
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from queue import Empty, Queue
from typing import Dict, List, Optional

logger = logging.getLogger("midge.market.finnhub_ws")

# ---------------------------------------------------------------------------
# Default ticker list — broad US equities + index proxy
# ---------------------------------------------------------------------------
DEFAULT_TICKERS: List[str] = ["AAPL", "MSFT", "TSLA", "NVDA", "SPY"]

# Signal thresholds
VOLUME_SPIKE_SIGMA = 2.0          # Standard deviations above baseline to flag a spike
RAPID_MOVE_PCT = 0.01             # 1 % price change across the rolling buffer triggers signal
MIN_TRADES_FOR_VOLUME = 60        # Need at least 1 min of trades before volume comparison
MIN_BASELINE_PERIODS = 5          # Minimum baseline samples before spike detection is active
BASELINE_UPDATE_EVERY = 60        # Append a new baseline sample every N trades
DIRECTIONAL_BULLISH_THRESHOLD = 0.6   # Close-in-range position >= this → bullish
DIRECTIONAL_BEARISH_THRESHOLD = 0.4   # Close-in-range position <= this → bearish


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeBuffer:
    """Rolling buffer for real-time trade aggregation per ticker."""

    # 5-minute window at ~1 trade/sec cadence
    prices: deque = field(default_factory=lambda: deque(maxlen=300))
    volumes: deque = field(default_factory=lambda: deque(maxlen=300))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=300))

    # 20-period rolling baseline for volume spike detection
    volume_baseline: deque = field(default_factory=lambda: deque(maxlen=20))

    last_price: float = 0.0
    last_volume: float = 0.0
    trade_count: int = 0  # total trades seen (not bounded by maxlen)


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class FinnhubWebSocket:
    """Background WebSocket client that streams Finnhub real-time trades.

    Usage::

        ws = FinnhubWebSocket(tickers=["AAPL", "MSFT"])
        ws.start()

        # Inside the step loop:
        for sig in ws.get_pending_signals():
            process(sig)

        ws.stop()

    Signal dict keys:
        signal_type: "volume_spike" | "rapid_move"
        symbol:      ticker string
        price:       trade price that triggered the signal
        value:       spike measure (recent_vol for volume_spike, pct_change for rapid_move)
        threshold:   the threshold that was crossed
        direction:   "bullish" | "bearish" | "neutral"
        timestamp:   datetime of signal creation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        raw_store=None,
    ) -> None:
        self._api_key: str = api_key if api_key is not None else os.environ.get("MAE_FINNHUB_API_KEY", "")
        self._tickers: List[str] = list(tickers or DEFAULT_TICKERS)
        self._raw_store = raw_store

        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._connected: bool = False

        # Per-ticker rolling buffers
        self._buffers: Dict[str, TradeBuffer] = {}

        # Thread-safe signal queue consumed by the main step loop
        self._pending_signals: Queue = Queue(maxsize=1000)

        # Reconnect back-off (doubles on each consecutive failure, caps at 300 s)
        self._reconnect_delay: float = 5.0
        self._max_reconnect_delay: float = 300.0
        self._consecutive_failures: int = 0
        self._connect_time: Optional[float] = None  # wall-clock time of last successful open
        self._last_429: bool = False  # flag set by _on_error / _on_message on rate-limit

        # Protects _tickers list during update_subscriptions
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect and subscribe in a daemon background thread."""
        if self._running:
            return
        if not self._api_key:
            logger.warning("FinnhubWebSocket: no API key — skipping start")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="finnhub-ws",
        )
        self._thread.start()
        logger.info("FinnhubWebSocket: background thread started")

    def stop(self) -> None:
        """Gracefully disconnect and wait for the background thread to exit."""
        self._running = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("FinnhubWebSocket: stopped")

    def is_connected(self) -> bool:
        """True when the WebSocket handshake has completed and we are subscribed."""
        return self._connected

    def get_pending_signals(self) -> list:
        """Drain the signal queue non-blocking. Call from the main step loop.

        Returns a list of signal dicts (may be empty). Each call returns only
        signals that arrived since the previous call.
        """
        signals = []
        while True:
            try:
                signals.append(self._pending_signals.get_nowait())
            except Empty:
                break
        return signals

    def update_subscriptions(self, tickers: List[str]) -> None:
        """Dynamically update the set of subscribed tickers.

        New tickers are subscribed immediately if connected.
        Removed tickers are unsubscribed immediately if connected.
        Buffers for removed tickers are retained (history is still useful).
        """
        with self._lock:
            current = set(self._tickers)
            updated = set(tickers)
            to_remove = current - updated
            to_add = updated - current
            self._tickers = list(tickers)

        if self._connected and self._ws is not None:
            for ticker in to_remove:
                try:
                    self._ws.send(json.dumps({"type": "unsubscribe", "symbol": ticker}))
                    logger.debug("Unsubscribed from %s", ticker)
                except Exception:
                    pass
            for ticker in to_add:
                try:
                    self._ws.send(json.dumps({"type": "subscribe", "symbol": ticker}))
                    logger.debug("Subscribed to %s", ticker)
                except Exception:
                    pass

    def get_statistics(self) -> dict:
        """Snapshot of client health — safe to call from any thread."""
        return {
            "connected": self._connected,
            "tickers": list(self._tickers),
            "buffers": len(self._buffers),
            "pending_signals": self._pending_signals.qsize(),
            "reconnect_delay": self._reconnect_delay,
        }

    # ------------------------------------------------------------------
    # WebSocket event handlers (called from WebSocket thread)
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Connect, reconnect with exponential back-off, until stop() is called."""
        try:
            import websocket  # type: ignore[import]
        except ImportError:
            logger.error(
                "websocket-client not installed — pip install websocket-client"
            )
            self._running = False
            return

        while self._running:
            try:
                url = f"wss://ws.finnhub.io?token={self._api_key}"
                ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws = ws
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception:
                logger.debug("WebSocket run_forever raised", exc_info=True)

            # If the connection stayed alive for >= 60 s, consider it stable and
            # reset the backoff counter so the next disconnect starts fresh.
            if self._connect_time is not None:
                alive_seconds = time.monotonic() - self._connect_time
                if alive_seconds >= 60.0:
                    self._reconnect_delay = 5.0
                    self._consecutive_failures = 0
                    logger.debug(
                        "FinnhubWebSocket: connection was stable (%.0fs) — backoff reset",
                        alive_seconds,
                    )
            self._connect_time = None
            self._connected = False

            if self._running:
                self._consecutive_failures += 1

                # 429 rate-limit: enforce a hard minimum of 60 s regardless of
                # where the backoff counter currently sits.
                if self._last_429:
                    delay = max(self._reconnect_delay, 60.0)
                    logger.warning(
                        "FinnhubWebSocket: rate-limited (429) — backing off %ds "
                        "after %d consecutive failure(s). Slow down.",
                        int(delay),
                        self._consecutive_failures,
                    )
                    self._last_429 = False
                else:
                    delay = self._reconnect_delay
                    logger.info(
                        "FinnhubWebSocket: backing off %ds after %d consecutive failure(s)",
                        int(delay),
                        self._consecutive_failures,
                    )

                time.sleep(delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )

    def _on_open(self, ws) -> None:
        """Called once the WebSocket handshake completes."""
        self._connected = True
        self._connect_time = time.monotonic()
        # Do NOT reset backoff here — wait until the connection proves stable
        # (>= 60 s alive).  The _run_loop checks this after run_forever returns.
        with self._lock:
            tickers = list(self._tickers)
        logger.info(
            "FinnhubWebSocket connected — subscribing to %d tickers", len(tickers)
        )
        for ticker in tickers:
            ws.send(json.dumps({"type": "subscribe", "symbol": ticker}))

    def _on_close(self, ws, close_status, close_msg) -> None:
        """Called when the connection is closed (cleanly or otherwise)."""
        self._connected = False
        logger.info(
            "FinnhubWebSocket disconnected: status=%s msg=%s",
            close_status,
            close_msg,
        )

    def _on_error(self, ws, error) -> None:
        """Called on non-fatal WebSocket errors."""
        error_str = str(error)
        if "429" in error_str:
            self._last_429 = True
            logger.warning(
                "FinnhubWebSocket: received 429 Too Many Requests — "
                "rate limit hit, will enforce minimum 60s backoff"
            )
        else:
            logger.debug("FinnhubWebSocket error: %s", error)

    def _on_message(self, ws, message: str) -> None:
        """Parse incoming Finnhub trade messages. Called in WebSocket thread."""
        try:
            # 429 can arrive as a plain-text or JSON error message before the
            # server closes the connection.
            if "429" in message:
                self._last_429 = True
                logger.warning(
                    "FinnhubWebSocket: 429 Too Many Requests in message — "
                    "rate limit hit, will enforce minimum 60s backoff"
                )
                return
            data = json.loads(message)
            if data.get("type") != "trade":
                return
            for trade in data.get("data", []):
                self._process_trade(trade)
        except Exception:
            logger.debug("Failed to process WebSocket message", exc_info=True)

    # ------------------------------------------------------------------
    # Trade processing and signal generation
    # ------------------------------------------------------------------

    def _process_trade(self, trade: dict) -> None:
        """Accumulate a single trade tick and emit signals when thresholds are crossed.

        Finnhub trade dict keys:
            s  — symbol
            p  — price (float)
            v  — volume (float, shares in this trade)
            t  — timestamp milliseconds since epoch
        """
        symbol: str = trade.get("s", "")
        price: float = float(trade.get("p", 0.0))
        volume: float = float(trade.get("v", 0.0))
        ts: float = trade.get("t", 0) / 1000.0  # ms → seconds

        if not symbol or price <= 0:
            return

        # Persist raw tick to SQLite (best-effort; never block the WebSocket thread)
        if self._raw_store is not None:
            try:
                self._raw_store.store_finnhub_ticks([{
                    "symbol": symbol,
                    "price": price,
                    "volume": volume,
                    "timestamp_ms": trade.get("t", 0),
                    "conditions": trade.get("c", []),
                }])
            except Exception:
                pass

        # Lazy-create buffer per ticker
        buf = self._buffers.get(symbol)
        if buf is None:
            buf = TradeBuffer()
            self._buffers[symbol] = buf

        buf.prices.append(price)
        buf.volumes.append(volume)
        buf.timestamps.append(ts)
        buf.trade_count += 1
        buf.last_price = price
        buf.last_volume = volume

        self._check_volume_spike(symbol, buf, price)
        self._check_rapid_move(symbol, buf, price)

    def _check_volume_spike(self, symbol: str, buf: TradeBuffer, price: float) -> None:
        """Emit a volume_spike signal when recent volume exceeds baseline by 2σ."""
        if len(buf.volumes) < MIN_TRADES_FOR_VOLUME:
            return

        recent_vol = sum(list(buf.volumes)[-MIN_TRADES_FOR_VOLUME:])

        # Update baseline periodically (every BASELINE_UPDATE_EVERY trades)
        if buf.trade_count % BASELINE_UPDATE_EVERY == 0:
            buf.volume_baseline.append(recent_vol)

        if len(buf.volume_baseline) < MIN_BASELINE_PERIODS:
            return

        baseline_mean = statistics.mean(buf.volume_baseline)
        baseline_std = (
            statistics.stdev(buf.volume_baseline)
            if len(buf.volume_baseline) > 1
            else baseline_mean * 0.5
        )

        if baseline_std <= 0:
            return

        if recent_vol > baseline_mean + VOLUME_SPIKE_SIGMA * baseline_std:
            self._emit_signal(
                symbol=symbol,
                signal_type="volume_spike",
                price=price,
                value=recent_vol,
                threshold=baseline_mean + VOLUME_SPIKE_SIGMA * baseline_std,
            )

    def _check_rapid_move(self, symbol: str, buf: TradeBuffer, price: float) -> None:
        """Emit a rapid_move signal when the rolling-window price change exceeds 1%."""
        if len(buf.prices) < 10:
            return

        oldest_price = buf.prices[0]
        if oldest_price <= 0:
            return

        pct_change = abs(price - oldest_price) / oldest_price
        if pct_change > RAPID_MOVE_PCT:
            direction = "bullish" if price > oldest_price else "bearish"
            self._emit_signal(
                symbol=symbol,
                signal_type="rapid_move",
                price=price,
                value=pct_change,
                threshold=RAPID_MOVE_PCT,
                direction=direction,
            )

    def _resolve_direction(self, symbol: str, price: float) -> str:
        """Determine directional bias from close-in-range position.

        Measures where the current price sits within the recent high-low range.
        Above 60% of range → bullish. Below 40% → bearish. Otherwise → neutral.
        """
        buf = self._buffers.get(symbol)
        if buf is None or len(buf.prices) < 10:
            return "neutral"

        prices = list(buf.prices)[-10:]
        low = min(prices)
        high = max(prices)
        price_range = high - low

        if price_range <= 0:
            return "neutral"

        position = (price - low) / price_range
        if position >= DIRECTIONAL_BULLISH_THRESHOLD:
            return "bullish"
        if position <= DIRECTIONAL_BEARISH_THRESHOLD:
            return "bearish"
        return "neutral"

    def _emit_signal(
        self,
        symbol: str,
        signal_type: str,
        price: float,
        value: float,
        threshold: float,
        direction: Optional[str] = None,
    ) -> None:
        """Build and enqueue a signal dict for consumption by the main step loop.

        If direction is not provided (volume_spike), it is resolved from the
        close-in-range position of the current price within the rolling buffer.
        """
        if direction is None:
            direction = self._resolve_direction(symbol, price)

        signal = {
            "signal_type": signal_type,
            "symbol": symbol,
            "price": price,
            "value": value,
            "threshold": threshold,
            "direction": direction,
            "timestamp": datetime.now(),
        }

        try:
            self._pending_signals.put_nowait(signal)
        except Exception:
            # Queue full — drop signal rather than block the WebSocket thread
            logger.debug(
                "FinnhubWebSocket: signal queue full, dropping %s %s",
                signal_type,
                symbol,
            )
