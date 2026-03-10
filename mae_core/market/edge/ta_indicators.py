#!/usr/bin/env python3
"""
ta_indicators.py - Technical Analysis Indicators for MIDGE

Re-export hub. All computation lives in sub-modules:
  ta_models.py      — TASignalBase, RSISignal, MACDSignal, BollingerSignal,
                       StructureSignal, CandleSignal, _extract_ohlcv
  ta_oscillators.py — compute_rsi, compute_macd, compute_bollinger
  ta_structure.py   — compute_market_structure, compute_candlestick_patterns, compute_atr

These signals flow through the same convergence pipeline as all other
MIDGE signals — Thompson Sampling learns which indicators actually work.
"""

import logging
from typing import List

# Re-export all models
from mae_core.market.edge.ta_models import (  # noqa: F401
    TASignalBase, RSISignal, MACDSignal, BollingerSignal,
    StructureSignal, CandleSignal, _extract_ohlcv,
)

# Re-export oscillators
from mae_core.market.edge.ta_oscillators import (  # noqa: F401
    compute_rsi, compute_macd, compute_bollinger,
)

# Re-export structure + ATR
from mae_core.market.edge.ta_structure import (  # noqa: F401
    compute_market_structure, compute_candlestick_patterns, compute_atr,
)

logger = logging.getLogger(__name__)


def compute_all(symbol: str, price_history) -> List[TASignalBase]:
    """Run all TA indicators and return actionable signals.

    Args:
        symbol: Ticker symbol
        price_history: List of PriceData (oldest first, from price_fetcher.get_daily_history)

    Returns:
        List of TASignalBase subclass instances (only actionable, not neutral)
    """
    if not price_history:
        return []

    signals: List[TASignalBase] = []

    try:
        rsi = compute_rsi(symbol, price_history)
        if rsi is not None:
            signals.append(rsi)
    except Exception:
        logger.debug("RSI computation failed for %s", symbol, exc_info=True)

    try:
        macd = compute_macd(symbol, price_history)
        if macd is not None:
            signals.append(macd)
    except Exception:
        logger.debug("MACD computation failed for %s", symbol, exc_info=True)

    try:
        bollinger = compute_bollinger(symbol, price_history)
        if bollinger is not None:
            signals.append(bollinger)
    except Exception:
        logger.debug("Bollinger computation failed for %s", symbol, exc_info=True)

    try:
        structure = compute_market_structure(symbol, price_history)
        if structure is not None:
            signals.append(structure)
    except Exception:
        logger.debug("Market structure computation failed for %s", symbol, exc_info=True)

    try:
        candle = compute_candlestick_patterns(symbol, price_history)
        if candle is not None:
            signals.append(candle)
    except Exception:
        logger.debug("Candlestick pattern detection failed for %s", symbol, exc_info=True)

    return signals


if __name__ == "__main__":
    import time
    import numpy as np

    class _FakePrice:
        __slots__ = ("price", "open", "high", "low", "volume", "timestamp")

        def __init__(self, price: float, i: int):
            self.price = price
            self.open = price * 0.999
            self.high = price * 1.005
            self.low = price * 0.995
            self.volume = 1_000_000
            self.timestamp = f"2025-{(i // 30) % 12 + 1:02d}-{i % 28 + 1:02d}"

    rng = np.random.default_rng(42)
    raw_prices = rng.uniform(100, 200, 1000).tolist()
    fake_history = [_FakePrice(p, i) for i, p in enumerate(raw_prices)]

    compute_rsi("BENCH", fake_history)
    compute_bollinger("BENCH", fake_history)
    compute_macd("BENCH", fake_history)

    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        compute_rsi("BENCH", fake_history, 14)
        compute_bollinger("BENCH", fake_history, 20, 2.0)
        compute_macd("BENCH", fake_history, 12, 26, 9)
    elapsed = time.perf_counter() - start
    print(f"Vectorized TA — {iterations} iterations: {elapsed:.3f}s  ({elapsed / iterations * 1000:.1f}ms each)")
