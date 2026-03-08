"""Motif Detector — STUMPY streaming matrix profile for per-symbol pattern detection.

Uses stumpy.stumpi (streaming mode) to maintain a live matrix profile for each
tracked symbol. Detects two signal types:

  - Motifs: repeated patterns (current price action matches a known historical shape)
  - Discords: anomalous subsequences (price action that is unlike anything seen before)

One STUMPI stream per symbol. Each stream holds the last N prices in a sliding window
(egress=True keeps length constant). When a new price arrives, the matrix profile
updates and the top-k distances indicate pattern matches or novel behavior.

Design constraints:
  - Max 50 tracked symbols (memory)
  - Window size 20 bars (~1 month of daily data)
  - Graceful degradation if stumpy not installed
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import stumpy

    HAS_STUMPY = True
except ImportError:
    HAS_STUMPY = False
    logger.warning("stumpy not installed — MotifDetector will run in no-op mode")

_WINDOW_SIZE = 20          # bars (~1 month of daily data)
_MAX_SYMBOLS = 50          # memory guard
_TOP_K_MOTIFS = 5          # motifs tracked per symbol
_DISCORD_THRESHOLD = 2.0   # MP z-score above this = discord
_MOTIF_THRESHOLD = 0.5     # normalised distance below this = motif match
_WARMUP_BARS = 40          # need 2x window to initialise a stream


@dataclass
class MotifSignal:
    """A detected motif or discord for a symbol."""

    symbol: str
    signal_type: str           # "motif" or "discord"
    mp_value: float            # current matrix profile value at detection point
    mp_index: int              # index in current profile where signal was found
    strength: float            # 0-1 normalised signal strength
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class _SymbolStream:
    """One per symbol. Wraps a STUMPI incremental stream + price buffer."""

    def __init__(self, initial_prices: List[float], window: int = _WINDOW_SIZE) -> None:
        self._window = window
        self._prices: deque = deque(initial_prices, maxlen=_WARMUP_BARS * 3)
        self._stream: Any = None
        self._mp_history: deque = deque(maxlen=50)   # recent min(MP) values for stats
        self._initialised = False
        self._total_updates = 0

        if HAS_STUMPY and len(initial_prices) >= window + 1:
            self._init_stream(initial_prices)

    def _init_stream(self, prices: List[float]) -> None:
        arr = np.array(prices, dtype=float)
        try:
            self._stream = stumpy.stumpi(arr, m=self._window, egress=True)
            self._initialised = True
        except Exception:
            logger.debug("STUMPI init failed", exc_info=True)
            self._initialised = False

    def update(self, price: float) -> Optional[MotifSignal]:
        """Feed one price. Returns a MotifSignal if a pattern fires, else None."""
        self._prices.append(price)
        self._total_updates += 1

        if not HAS_STUMPY:
            return None

        # Lazy initialisation: wait until we have enough data
        if not self._initialised:
            if len(self._prices) >= _WARMUP_BARS:
                self._init_stream(list(self._prices))
            return None

        try:
            self._stream.update(price)
        except Exception:
            logger.debug("STUMPI update failed", exc_info=True)
            return None

        return self._evaluate()

    def _evaluate(self) -> Optional[MotifSignal]:
        """Check the current matrix profile for motifs and discords."""
        try:
            mp = self._stream.P_           # matrix profile values
            if mp is None or len(mp) == 0:
                return None

            min_val = float(np.nanmin(mp))
            max_val = float(np.nanmax(mp))
            self._mp_history.append(min_val)

            # Compute rolling stats for discord detection
            history = list(self._mp_history)
            if len(history) < 3:
                return None

            mean_mp = sum(history) / len(history)
            variance = sum((v - mean_mp) ** 2 for v in history) / max(1, len(history) - 1)
            std_mp = math.sqrt(variance) if variance > 0 else 1.0

            # Discord: current min is unusually HIGH (far from all other subsequences)
            current_discord_idx = int(np.argmax(mp))
            current_discord_val = float(mp[current_discord_idx])
            discord_zscore = (current_discord_val - mean_mp) / std_mp if std_mp > 0 else 0.0

            if discord_zscore >= _DISCORD_THRESHOLD:
                strength = min(1.0, discord_zscore / (_DISCORD_THRESHOLD * 2))
                return MotifSignal(
                    symbol="",          # filled by caller
                    signal_type="discord",
                    mp_value=current_discord_val,
                    mp_index=current_discord_idx,
                    strength=strength,
                    metadata={"zscore": round(discord_zscore, 3), "mp_mean": round(mean_mp, 4)},
                )

            # Motif: current min is unusually LOW (very close match to a past subsequence)
            current_motif_idx = int(np.argmin(mp))
            current_motif_val = float(mp[current_motif_idx])
            if current_motif_val <= _MOTIF_THRESHOLD and max_val > 0:
                # Normalise: 0 = perfect match, 1 = worst match in profile
                strength = 1.0 - (current_motif_val / max(max_val, 1e-9))
                strength = max(0.0, min(1.0, strength))
                if strength >= 0.3:   # filter weak matches
                    return MotifSignal(
                        symbol="",
                        signal_type="motif",
                        mp_value=current_motif_val,
                        mp_index=current_motif_idx,
                        strength=strength,
                        metadata={"match_distance": round(current_motif_val, 4)},
                    )
        except Exception:
            logger.debug("STUMPI evaluate failed", exc_info=True)

        return None

    @property
    def active_motifs(self) -> List[dict]:
        """Summary of current matrix profile top motifs."""
        if not self._initialised or not HAS_STUMPY:
            return []
        try:
            mp = self._stream.P_
            if mp is None or len(mp) == 0:
                return []
            top_k = min(_TOP_K_MOTIFS, len(mp))
            indices = np.argsort(mp)[:top_k]
            return [
                {"index": int(i), "distance": round(float(mp[i]), 4)}
                for i in indices
                if not math.isnan(float(mp[i]))
            ]
        except Exception:
            return []


class MotifDetector:
    """Per-symbol streaming motif and discord detector using STUMPY.

    Constructor args:
        raw_store: Optional RawStore (unused directly; reserved for future persistence).

    Usage:
        md = MotifDetector()
        signals = md.update("AAPL", 185.50, datetime.now())
        active = md.get_active_motifs("AAPL")
    """

    def __init__(self, raw_store=None) -> None:
        self._raw_store = raw_store
        self._streams: Dict[str, _SymbolStream] = {}
        self._symbol_order: deque = deque(maxlen=_MAX_SYMBOLS)  # LRU eviction
        self._total_signals = 0

    def update(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> List[MotifSignal]:
        """Feed one price observation for a symbol.

        Returns a list of MotifSignal objects (empty if nothing fires).
        """
        if not symbol or not isinstance(price, (int, float)) or math.isnan(price):
            return []

        timestamp = timestamp or datetime.now()

        # Evict oldest symbol if at capacity
        if symbol not in self._streams and len(self._streams) >= _MAX_SYMBOLS:
            oldest = self._symbol_order.popleft()
            if oldest != symbol:
                del self._streams[oldest]
                logger.debug("MotifDetector: evicted %s (capacity=%d)", oldest, _MAX_SYMBOLS)

        if symbol not in self._streams:
            self._streams[symbol] = _SymbolStream([], _WINDOW_SIZE)
            self._symbol_order.append(symbol)

        signal = self._streams[symbol].update(price)
        if signal is None:
            return []

        signal.symbol = symbol
        signal.timestamp = timestamp
        self._total_signals += 1

        logger.info(
            "MotifDetector: %s %s detected (strength=%.2f, mp=%.4f)",
            symbol, signal.signal_type, signal.strength, signal.mp_value,
        )
        return [signal]

    def get_active_motifs(self, symbol: str) -> List[dict]:
        """Return the top-k current motif candidates for a symbol."""
        stream = self._streams.get(symbol)
        if stream is None:
            return []
        return stream.active_motifs

    def get_statistics(self) -> dict:
        """For HolonProxy.sense() delegation."""
        return {
            "tracked_symbols": len(self._streams),
            "total_signals": self._total_signals,
            "has_stumpy": HAS_STUMPY,
            "initialised_streams": sum(
                1 for s in self._streams.values() if s._initialised
            ),
        }
