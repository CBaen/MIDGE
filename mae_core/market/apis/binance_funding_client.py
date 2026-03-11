"""Binance Funding Rate Client — Crypto derivatives positioning signal.

Fetches perpetual futures funding rates from Binance's public API.
No authentication required. Rate limit: 2400 requests/min (IP-based).

Funding rates reveal leveraged positioning:
  - High positive rate (>0.03%) = longs paying shorts = overleveraged bullish
  - High negative rate (<-0.03%) = shorts paying longs = overleveraged bearish
  - Extreme rates predict liquidation cascades (mean-reversion signal)

This is an independent signal from crypto prices — it measures derivatives
positioning, not spot market activity. Convergence with spot data creates
a "crypto_positioning" domain distinct from "crypto" (price/volume).

Funding rate is settled every 8 hours (00:00, 08:00, 16:00 UTC).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import requests

logger = logging.getLogger("midge.market.binance_funding")

_BASE_URL = "https://fapi.binance.com"
_REQUEST_DELAY = 0.1  # 100ms between requests (conservative for 2400/min)

# Symbols to track — major perpetual futures
TRACKED_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]

# Thresholds for directional signals
# Normal funding: -0.01% to 0.01% (neutral)
# Elevated: 0.01% to 0.05% (mild positioning)
# Extreme: >0.05% (liquidation risk)
EXTREME_THRESHOLD = 0.0005  # 0.05% per 8h = overleveraged
ELEVATED_THRESHOLD = 0.0001  # 0.01% per 8h = positioning building


@dataclass
class FundingRate:
    """Single funding rate observation for a perpetual futures contract."""

    symbol: str
    rate: float  # e.g. 0.0001 = 0.01%
    timestamp: datetime
    mark_price: Optional[float] = None

    @property
    def direction(self) -> str:
        """Contrarian signal — extreme longs predict bearish reversal."""
        if self.rate > EXTREME_THRESHOLD:
            return "bearish"  # Overleveraged longs = liquidation risk
        elif self.rate < -EXTREME_THRESHOLD:
            return "bullish"  # Overleveraged shorts = squeeze risk
        return "neutral"

    @property
    def strength(self) -> float:
        """Signal strength based on funding rate magnitude."""
        abs_rate = abs(self.rate)
        if abs_rate > EXTREME_THRESHOLD:
            return min(1.0, abs_rate / (EXTREME_THRESHOLD * 3))
        elif abs_rate > ELEVATED_THRESHOLD:
            return 0.3 + 0.3 * (abs_rate - ELEVATED_THRESHOLD) / (EXTREME_THRESHOLD - ELEVATED_THRESHOLD)
        return 0.1


class BinanceFundingClient:
    """Client for Binance Futures public API — funding rates only.

    No API key needed. Uses public endpoints only.
    """

    def __init__(self, raw_store=None):
        self._raw_store = raw_store
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "MIDGE Trading Research"
        self._last_request: float = 0.0
        self._cache: dict[str, tuple[list[FundingRate], float]] = {}
        self._cache_ttl = 4 * 3600  # 4h — funding settles every 8h

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < _REQUEST_DELAY:
            time.sleep(_REQUEST_DELAY - elapsed)
        self._last_request = time.time()

    def get_funding_rates(self, symbols: Optional[List[str]] = None) -> List[FundingRate]:
        """Fetch latest funding rates for tracked symbols.

        Args:
            symbols: Override default symbol list. None = use TRACKED_SYMBOLS.

        Returns:
            List of FundingRate objects, one per symbol.
        """
        targets = symbols or TRACKED_SYMBOLS
        results: List[FundingRate] = []

        for symbol in targets:
            # Check cache
            if symbol in self._cache:
                cached, cached_at = self._cache[symbol]
                if time.time() - cached_at < self._cache_ttl:
                    results.extend(cached)
                    continue

            rate = self._fetch_one(symbol)
            if rate:
                results.append(rate)
                self._cache[symbol] = ([rate], time.time())

        if results:
            logger.info(
                "Binance funding: %d/%d symbols fetched, %d extreme",
                len(results), len(targets),
                sum(1 for r in results if abs(r.rate) > EXTREME_THRESHOLD),
            )
        return results

    def _fetch_one(self, symbol: str) -> Optional[FundingRate]:
        """Fetch latest funding rate for one symbol."""
        self._rate_limit()
        try:
            resp = self._session.get(
                f"{_BASE_URL}/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": 1},
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning("Binance funding HTTP %s for %s", resp.status_code, symbol)
                return None

            data = resp.json()
            if not data:
                return None

            # Store raw response
            if self._raw_store:
                try:
                    self._raw_store.store_binance_funding(symbol, data)
                except Exception:
                    pass

            entry = data[0]
            rate = float(entry["fundingRate"])
            ts = datetime.fromtimestamp(
                int(entry["fundingTime"]) / 1000, tz=timezone.utc
            )
            mark_price = float(entry.get("markPrice", 0)) or None

            return FundingRate(
                symbol=symbol,
                rate=rate,
                timestamp=ts,
                mark_price=mark_price,
            )
        except Exception as exc:
            logger.warning("Binance funding error for %s: %s", symbol, exc)
            return None

    def get_extreme_rates(self, symbols: Optional[List[str]] = None) -> List[FundingRate]:
        """Return only symbols with extreme funding rates (positioning signal)."""
        return [r for r in self.get_funding_rates(symbols) if abs(r.rate) > EXTREME_THRESHOLD]
