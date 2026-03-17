"""Kraken Futures Client — Funding rates and open interest from Kraken Perpetuals.

US-accessible replacement for the geo-blocked Binance derivatives API.
All data is fully public — no authentication required.

Endpoint: https://futures.kraken.com/derivatives/api/v3/tickers
Returns all perpetual contract tickers in a single call.

Funding rate interpretation (contrarian):
  - rate > +0.01%: longs crowded → bearish (longs will get squeezed)
  - rate < -0.01%: shorts crowded → bullish (shorts will get squeezed)
  - |rate| in [-0.01%, +0.01%]: neutral — no structural positioning imbalance

Signals carry domain="positioning" for convergence with other institutional
flow data. 30-second cache avoids API hammering.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("midge.market.apis.kraken_futures")

_TICKERS_URL = "https://futures.kraken.com/derivatives/api/v3/tickers"
_CACHE_TTL = 30.0  # 30 seconds — funding rates update frequently

# Thresholds — fraction (not percent)
_NEUTRAL_BAND = 0.0001   # ±0.01%
_STRENGTH_MAX = 0.001    # 0.10% → strength 1.0

# Kraken perpetual symbol → clean ticker
_SYMBOL_MAP: Dict[str, str] = {
    "PI_XBTUSD":  "BTC",
    "PI_ETHUSD":  "ETH",
    "PI_SOLUSD":  "SOL",
    "PI_XRPUSD":  "XRP",
    "PI_ADAUSD":  "ADA",
    "PI_DOGEUSD": "DOGE",
    "PI_LINKUSD": "LINK",
    "PI_LTCUSD":  "LTC",
    "PI_AVAXUSD": "AVAX",
}


@dataclass
class KrakenDerivativesData:
    """Derivatives positioning snapshot for a single perpetual contract.

    direction is CONTRARIAN:
      - positive funding → longs crowded → bearish signal
      - negative funding → shorts crowded → bullish signal
      - neutral band → no directional signal
    """

    symbol: str           # Clean ticker, e.g. "BTC"
    funding_rate: float   # Current funding rate fraction (e.g. 0.0001 = 0.01%)
    open_interest: float  # Number of open contracts
    mark_price: float     # Current mark price
    volume_24h: float     # 24-hour trading volume
    direction: str        # "bullish", "bearish", or "neutral"
    strength: float       # 0.0 (neutral band) → 1.0 (0.10%+ extreme)
    timestamp: datetime


def _derive_direction(funding_rate: float) -> str:
    """Contrarian direction from funding rate sign."""
    if funding_rate > _NEUTRAL_BAND:
        return "bearish"   # Longs crowded — will get squeezed down
    if funding_rate < -_NEUTRAL_BAND:
        return "bullish"   # Shorts crowded — will get squeezed up
    return "neutral"


def _derive_strength(funding_rate: float) -> float:
    """Scale |rate| from neutral band → max into 0.0 → 1.0."""
    excess = max(0.0, abs(funding_rate) - _NEUTRAL_BAND)
    scale = _STRENGTH_MAX - _NEUTRAL_BAND
    if scale <= 0:
        return 0.0
    return min(1.0, excess / scale)


class KrakenFuturesClient:
    """Kraken Futures public derivatives data — funding rates and open interest.

    Single endpoint fetches all perpetual tickers in one call.
    9 mapped symbols: BTC, ETH, SOL, XRP, ADA, DOGE, LINK, LTC, AVAX.
    30-second cache. No authentication required.
    """

    def __init__(self, raw_store=None):
        self._raw_store = raw_store
        self._client = httpx.Client(
            headers={"User-Agent": "MIDGE Trading Research"},
            timeout=10.0,
        )
        self._cache: Optional[Tuple[List[KrakenDerivativesData], float]] = None
        self._total_fetches: int = 0
        self._total_errors: int = 0
        self._last_symbol_count: int = 0

    def _fetch_raw(self) -> Optional[dict]:
        """GET the tickers endpoint. Returns parsed JSON or None."""
        try:
            resp = self._client.get(_TICKERS_URL)
            if resp.status_code != 200:
                logger.warning("Kraken Futures HTTP %s", resp.status_code)
                self._total_errors += 1
                return None
            return resp.json()
        except httpx.RequestError as exc:
            logger.warning("Kraken Futures request error: %s", exc)
            self._total_errors += 1
            return None

    def _parse(self, raw: dict) -> List[KrakenDerivativesData]:
        """Parse tickers response into KrakenDerivativesData for mapped symbols."""
        tickers = raw.get("tickers", [])
        now = datetime.now(tz=timezone.utc)
        results: List[KrakenDerivativesData] = []

        for entry in tickers:
            kraken_sym = entry.get("symbol", "")
            clean = _SYMBOL_MAP.get(kraken_sym)
            if clean is None:
                continue  # Not a mapped perpetual

            try:
                rate = float(entry.get("fundingRate", 0) or 0)
                oi = float(entry.get("openInterest", 0) or 0)
                mark = float(entry.get("markPrice", 0) or 0)
                vol = float(entry.get("vol24h", 0) or 0)
            except (TypeError, ValueError) as exc:
                logger.debug("Kraken Futures parse error for %s: %s", kraken_sym, exc)
                continue

            results.append(KrakenDerivativesData(
                symbol=clean,
                funding_rate=rate,
                open_interest=oi,
                mark_price=mark,
                volume_24h=vol,
                direction=_derive_direction(rate),
                strength=_derive_strength(rate),
                timestamp=now,
            ))

        return results

    def get_derivatives_data(self) -> List[KrakenDerivativesData]:
        """Fetch funding rates and open interest for all mapped perpetuals.

        Returns cached data if within 30-second TTL.
        Returns empty list on any failure.
        """
        if self._cache is not None:
            cached_data, fetched_at = self._cache
            if time.time() - fetched_at < _CACHE_TTL:
                return cached_data

        raw = self._fetch_raw()
        if raw is None:
            return []

        self._total_fetches += 1

        if self._raw_store is not None:
            try:
                self._raw_store.store_kraken_futures(raw)
            except Exception as exc:
                logger.debug("RawStore kraken_futures write failed: %s", exc)

        results = self._parse(raw)
        self._last_symbol_count = len(results)
        self._cache = (results, time.time())

        logger.debug(
            "Kraken Futures: %d symbols parsed, %d actionable",
            len(results),
            sum(1 for r in results if r.direction != "neutral"),
        )
        return results

    def get_statistics(self) -> dict:
        """HolonProxy integration — returns operational health snapshot."""
        return {
            "total_fetches": self._total_fetches,
            "total_errors": self._total_errors,
            "last_symbol_count": self._last_symbol_count,
            "cache_age_seconds": (
                round(time.time() - self._cache[1], 1)
                if self._cache is not None else None
            ),
            "mapped_symbols": list(_SYMBOL_MAP.values()),
        }
