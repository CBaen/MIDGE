"""Technical analysis fetch functions — TA indicators, session sweeps, order flow, fractals.

Pure local computation or price-history-only calls. No social or macro data.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("midge.market.sensing")


def fetch_ta_indicators(
    ta_indicators: Any,
    price_fetcher: Any,
    watchlist: dict,
    converter: Callable,
) -> list:
    """Compute technical analysis indicators for watchlist tickers.

    Uses price_fetcher.get_daily_history() for OHLCV data, then runs
    RSI, MACD, Bollinger, Market Structure, and Candlestick detection.
    Pure local computation — no external API calls beyond yfinance history.
    """
    if ta_indicators is None or price_fetcher is None:
        return []

    from mae_core.market.edge.ta_indicators import compute_all

    signals = []
    for ticker in watchlist.get("tickers", []):
        try:
            history = price_fetcher.get_daily_history(ticker, days=90)
            if not history:
                continue
            ta_signals = compute_all(ticker, history)
            for ta_sig in ta_signals:
                try:
                    signals.append(converter(ta_sig))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("TA indicators failed for %s: %s", ticker, e)
    return signals


def fetch_session_sweep(session_sweep_detector: Any, converter: Callable) -> list:
    """Fetch ICT session sweep signals for futures.

    Kill-zone time guard: returns early if not within 90 min of a
    kill zone window. Prevents wasting yfinance rate limit during
    dead hours.
    """
    if session_sweep_detector is None:
        return []

    # Time-of-day guard (Eastern time)
    try:
        from zoneinfo import ZoneInfo
        from datetime import time as _time
        from datetime import datetime
        now_et = datetime.now(ZoneInfo("America/New_York")).time()
        # Kill zone windows with ±90 min buffer
        kz_windows = [
            (_time(18, 30), _time(23, 59)),  # Asia buffer (evening)
            (_time(0, 0), _time(6, 30)),     # Asia + London buffer
            (_time(5, 30), _time(11, 30)),   # NY kill zone buffer
        ]
        in_window = any(s <= now_et <= e for s, e in kz_windows)
        if not in_window:
            logger.debug("Session sweep: outside kill zone window, skipping")
            return []
    except Exception:
        pass  # If timezone check fails, proceed anyway

    signals = []
    futures_symbols = ["ES=F", "NQ=F"]
    for symbol in futures_symbols:
        try:
            sweeps = session_sweep_detector.detect_sweeps(symbol)
            for sweep in sweeps:
                try:
                    signals.append(converter(sweep))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Session sweep fetch failed for %s: %s", symbol, e)
    return signals


def fetch_fractal_resonance(
    fractal_detector: Any,
    watchlist: dict,
    converter: Callable,
) -> list:
    """Fetch fractal resonance signals — multi-timeframe structure alignment.

    Calls FractalResonanceDetector.detect_resonance() for each ticker.
    Cadence 200 in SOURCE_ROTATION (weekly refresh rate).
    """
    if fractal_detector is None:
        return []

    signals = []
    tickers = watchlist.get("tickers", [])[:10]
    for ticker in tickers:
        try:
            result = fractal_detector.detect_resonance(ticker)
            if result is not None and result.resonance_score > 0.3:
                try:
                    signals.append(converter(result))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Fractal resonance fetch failed for %s: %s", ticker, e)
    return signals


def fetch_order_flow(
    order_flow_detector: Any,
    watchlist: dict,
    converter: Callable,
) -> list:
    """Fetch order flow imbalance signals for watchlist tickers.

    Calls OrderFlowDetector.detect_imbalance() for each ticker, then converts
    detected imbalances into MarketSignal via the from_order_flow adapter.
    """
    if order_flow_detector is None:
        return []

    signals = []
    tickers = watchlist.get("tickers", [])[:10]
    for ticker in tickers:
        try:
            results = order_flow_detector.detect_imbalance(ticker)
            for result in results:
                try:
                    signals.append(converter(result))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Order flow fetch failed for %s: %s", ticker, e)
    return signals
