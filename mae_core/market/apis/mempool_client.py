"""mempool_client.py — Bitcoin on-chain data from mempool.space.

Free, no auth, US-accessible. Provides fee pressure, mempool congestion,
and mining pool data as on-chain signals for crypto convergence.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("midge.market.apis.mempool_client")

_BASE = "https://mempool.space/api"
_CACHE_TTL = 60  # seconds


@dataclass
class MempoolData:
    """Snapshot of Bitcoin mempool and fee state."""
    mempool_count: int = 0
    mempool_vsize: int = 0
    fastest_fee: int = 0
    half_hour_fee: int = 0
    hour_fee: int = 0
    economy_fee: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class MempoolClient:
    """Bitcoin on-chain data client using mempool.space public API."""

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=15, follow_redirects=True)
        self._cache: Optional[MempoolData] = None
        self._cache_time: float = 0
        self._call_count = 0
        self._error_count = 0
        logger.info("MempoolClient initialized (free API, no key required)")

    def get_mempool_state(self) -> MempoolData:
        """Fetch current mempool state. Cached for 60s."""
        if self._cache and (time.time() - self._cache_time) < _CACHE_TTL:
            return self._cache

        data = MempoolData()
        try:
            # Mempool stats
            self._call_count += 1
            resp = self._client.get(f"{_BASE}/mempool")
            if resp.status_code == 200:
                j = resp.json()
                data.mempool_count = j.get("count", 0)
                data.mempool_vsize = j.get("vsize", 0)

            # Fee estimates
            self._call_count += 1
            resp = self._client.get(f"{_BASE}/v1/fees/recommended")
            if resp.status_code == 200:
                j = resp.json()
                data.fastest_fee = j.get("fastestFee", 0)
                data.half_hour_fee = j.get("halfHourFee", 0)
                data.hour_fee = j.get("hourFee", 0)
                data.economy_fee = j.get("economyFee", 0)

            data.timestamp = datetime.now()
            self._cache = data
            self._cache_time = time.time()
        except Exception as e:
            self._error_count += 1
            logger.debug("MempoolClient fetch failed: %s", e)

        return data

    def get_on_chain_signals(self) -> List[Dict[str, Any]]:
        """Return convergence-ready signal dicts for BTC on-chain state."""
        state = self.get_mempool_state()
        signals: List[Dict[str, Any]] = []
        now = datetime.now().isoformat()

        # Fee spike detection
        if state.fastest_fee > 0:
            if state.fastest_fee > 50:
                direction = "bullish"
                strength = min(1.0, (state.fastest_fee - 50) / 150)
            elif state.fastest_fee < 5:
                direction = "bearish"
                strength = 0.3
            else:
                direction = "neutral"
                strength = 0.1

            if direction != "neutral":
                signals.append({
                    "source": "mempool_btc",
                    "symbol": "BTC",
                    "asset_class": "crypto",
                    "domain": "on_chain",
                    "direction": direction,
                    "strength": round(strength, 3),
                    "confidence": 0.55,
                    "timestamp": now,
                    "metadata": {
                        "symbol": "BTC",
                        "fastest_fee": state.fastest_fee,
                        "hour_fee": state.hour_fee,
                        "mempool_count": state.mempool_count,
                        "signal_type": "fee_pressure",
                    },
                })

        # Mempool congestion
        if state.mempool_count > 100000:
            signals.append({
                "source": "mempool_btc",
                "symbol": "BTC",
                "asset_class": "crypto",
                "domain": "on_chain",
                "direction": "bullish",
                "strength": min(1.0, (state.mempool_count - 100000) / 200000),
                "confidence": 0.50,
                "timestamp": now,
                "metadata": {
                    "symbol": "BTC",
                    "mempool_count": state.mempool_count,
                    "mempool_vsize_mb": round(state.mempool_vsize / 1_000_000, 1),
                    "signal_type": "congestion",
                },
            })

        return signals

    def get_statistics(self) -> Dict[str, Any]:
        """Return client statistics for HolonProxy integration."""
        return {
            "calls_made": self._call_count,
            "errors": self._error_count,
            "cached": self._cache is not None,
        }
