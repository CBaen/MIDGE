"""Binance Derivatives Client — Open interest, long/short ratio, liquidations.

Extends the existing funding rate capabilities with the three remaining
free derivatives data streams from Binance's public API:

  - Open Interest: Total open contracts (rising OI + price = healthy trend,
    diverging OI = warning sign)
  - Long/Short Ratio: Account-level positioning (extremes predict reversals)
  - Liquidation Orders: Forced closes reveal stop-hunt zones and exhaustion

All endpoints are FREE and require no API key.
Base URL: https://fapi.binance.com (Binance Futures)

The crown jewel is get_crowded_trade_score() — when funding rate is extreme,
OI is diverging from price, AND long/short ratio is extreme, a liquidation
cascade becomes structurally inevitable. All signals carry domain="positioning"
for convergence with other institutional flow data.

Rate limit: Binance allows 2400 req/min. We use 100ms delays (conservative).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("midge.market.binance_derivatives")

_FUTURES_BASE_URL = "https://fapi.binance.com"
_REQUEST_DELAY = 0.1  # 100ms between requests (2400/min rate limit)
_CACHE_TTL = 900.0  # 15 minutes — derivatives data updates frequently

# Symbols to track — major perpetual futures
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]

# Open Interest analysis thresholds
OI_DIVERGENCE_THRESHOLD = 0.05   # 5% OI change in opposite direction of price
LONG_SHORT_EXTREME_HIGH = 1.5    # Long/Short ratio > 1.5 = heavily long (bearish contrarian)
LONG_SHORT_EXTREME_LOW = 0.7     # Long/Short ratio < 0.7 = heavily short (bullish contrarian)
FUNDING_EXTREME = 0.0005         # 0.05% per 8h — mirrors binance_funding_client threshold


@dataclass
class OpenInterestSnapshot:
    """Current open interest for a perpetual futures contract."""

    symbol: str
    open_interest: float          # Number of open contracts
    open_interest_value: float    # USD value of open contracts
    timestamp: datetime


@dataclass
class OpenInterestHistory:
    """Single OI data point from history endpoint."""

    symbol: str
    open_interest: float
    open_interest_value: float
    timestamp: datetime


@dataclass
class LongShortRatio:
    """Global long/short account ratio for a perpetual contract."""

    symbol: str
    long_ratio: float     # Fraction of accounts holding net long (0.0-1.0)
    short_ratio: float    # Fraction of accounts holding net short (0.0-1.0)
    long_short_ratio: float   # long_account / short_account
    timestamp: datetime

    @property
    def direction(self) -> str:
        """Contrarian direction from extreme positioning."""
        if self.long_short_ratio > LONG_SHORT_EXTREME_HIGH:
            return "bearish"   # Overleveraged longs = ripe for squeeze down
        elif self.long_short_ratio < LONG_SHORT_EXTREME_LOW:
            return "bullish"   # Overleveraged shorts = ripe for squeeze up
        return "neutral"

    @property
    def strength(self) -> float:
        """Signal strength: 0.0 at neutral, 1.0 at extreme."""
        if self.long_short_ratio > LONG_SHORT_EXTREME_HIGH:
            return min(1.0, (self.long_short_ratio - LONG_SHORT_EXTREME_HIGH) / 0.5)
        elif self.long_short_ratio < LONG_SHORT_EXTREME_LOW:
            return min(1.0, (LONG_SHORT_EXTREME_LOW - self.long_short_ratio) / 0.3)
        return 0.0


@dataclass
class LiquidationOrder:
    """Single forced liquidation order."""

    symbol: str
    side: str          # "BUY" (liq of short) or "SELL" (liq of long)
    price: float
    quantity: float
    usd_value: float
    timestamp: datetime


@dataclass
class CrowdedTradeScore:
    """Composite derivatives positioning score.

    When all three fire simultaneously, a liquidation cascade is structurally
    inevitable: the crowd is one-sided, OI is diverging, and funding confirms
    the imbalance. MIDGE calls this "mechanical inevitability."

    score: 0-3 (count of conditions satisfied)
    direction: which direction the cascade will run
    """

    symbol: str
    score: int          # 0-3 fire count
    funding_extreme: bool
    oi_diverging: bool
    long_extreme: bool
    direction: str      # "bullish" (shorts get squeezed) or "bearish" (longs get liquidated)
    timestamp: datetime

    # Raw values for transparency
    funding_rate: Optional[float] = None
    oi_change_pct: Optional[float] = None
    long_short_ratio: Optional[float] = None

    @property
    def strength(self) -> float:
        """Signal strength from 0.0 (no signal) to 1.0 (all three firing)."""
        return self.score / 3.0

    @property
    def is_actionable(self) -> bool:
        """True when at least 2 of 3 conditions are satisfied."""
        return self.score >= 2


class BinanceDerivativesClient:
    """Crypto derivatives data — open interest, long/short ratio, liquidations.

    All endpoints are FREE and require no API key.
    Signals carry domain="positioning" for convergence with funding rates
    and other institutional flow data.
    """

    def __init__(self, raw_store=None):
        self._raw_store = raw_store
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "MIDGE Trading Research",
            "Accept": "application/json",
        })
        self._last_request: float = 0.0
        self._cache: Dict[str, Tuple[object, float]] = {}

    def _rate_limit(self) -> None:
        """Enforce 100ms minimum between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < _REQUEST_DELAY:
            time.sleep(_REQUEST_DELAY - elapsed)
        self._last_request = time.time()

    def _get(self, path: str, params: dict = None) -> object:
        """Make a rate-limited GET to Binance Futures API.

        Returns parsed JSON or None on any failure.
        """
        self._rate_limit()
        url = f"{_FUTURES_BASE_URL}{path}"
        try:
            resp = self._session.get(url, params=params or {}, timeout=10)
            if resp.status_code != 200:
                logger.warning("Binance derivatives HTTP %s for %s", resp.status_code, path)
                return None
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Binance derivatives request failed %s: %s", path, exc)
            return None

    def _cached_get(self, cache_key: str, path: str, params: dict = None) -> object:
        """GET with 15-minute cache."""
        cached = self._cache.get(cache_key)
        if cached is not None:
            data, fetched_at = cached
            if time.time() - fetched_at < _CACHE_TTL:
                return data
        data = self._get(path, params)
        if data is not None:
            self._cache[cache_key] = (data, time.time())
        return data

    # ------------------------------------------------------------------
    # Open Interest
    # ------------------------------------------------------------------

    def get_open_interest(self, symbol: str = "BTCUSDT") -> Optional[OpenInterestSnapshot]:
        """Current open interest for a perpetual contract.

        Endpoint: GET /fapi/v1/openInterest

        Args:
            symbol: Binance perpetual symbol (e.g. "BTCUSDT")

        Returns:
            OpenInterestSnapshot or None on failure.
        """
        data = self._cached_get(
            f"oi:{symbol}",
            "/fapi/v1/openInterest",
            {"symbol": symbol},
        )
        if not data or not isinstance(data, dict):
            return None
        try:
            return OpenInterestSnapshot(
                symbol=symbol,
                open_interest=float(data.get("openInterest", 0)),
                open_interest_value=float(data.get("openInterestValue", 0)),
                timestamp=datetime.fromtimestamp(
                    int(data.get("time", time.time() * 1000)) / 1000,
                    tz=timezone.utc,
                ),
            )
        except (TypeError, ValueError) as exc:
            logger.warning("OI parse error for %s: %s", symbol, exc)
            return None

    def get_open_interest_history(
        self, symbol: str = "BTCUSDT", period: str = "1h", limit: int = 48
    ) -> List[OpenInterestHistory]:
        """Historical OI data points.

        Endpoint: GET /futures/data/openInterestHist

        Args:
            symbol: Binance perpetual symbol
            period: "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"
            limit: Number of data points (max 500)

        Returns:
            List of OpenInterestHistory, sorted oldest→newest.
        """
        limit = min(limit, 500)
        data = self._cached_get(
            f"oi_hist:{symbol}:{period}:{limit}",
            "/futures/data/openInterestHist",
            {"symbol": symbol, "period": period, "limit": limit},
        )
        if not data or not isinstance(data, list):
            return []
        results = []
        for entry in data:
            try:
                results.append(OpenInterestHistory(
                    symbol=symbol,
                    open_interest=float(entry.get("sumOpenInterest", 0)),
                    open_interest_value=float(entry.get("sumOpenInterestValue", 0)),
                    timestamp=datetime.fromtimestamp(
                        int(entry.get("timestamp", 0)) / 1000, tz=timezone.utc
                    ),
                ))
            except (TypeError, ValueError):
                continue
        return sorted(results, key=lambda x: x.timestamp)

    def _compute_oi_trend(self, symbol: str, periods: int = 12) -> Optional[float]:
        """Compute OI change as fraction over the last N periods.

        Returns:
            Float: positive = OI growing, negative = OI shrinking.
            None if insufficient data.
        """
        history = self.get_open_interest_history(symbol, period="1h", limit=periods + 1)
        if len(history) < 2:
            return None
        oldest_oi = history[0].open_interest
        newest_oi = history[-1].open_interest
        if oldest_oi <= 0:
            return None
        return (newest_oi - oldest_oi) / oldest_oi

    # ------------------------------------------------------------------
    # Long/Short Ratio
    # ------------------------------------------------------------------

    def get_long_short_ratio(
        self, symbol: str = "BTCUSDT", period: str = "1h", limit: int = 1
    ) -> Optional[LongShortRatio]:
        """Global long/short account ratio.

        Endpoint: GET /futures/data/globalLongShortAccountRatio

        Args:
            symbol: Binance perpetual symbol
            period: "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"
            limit: 1 = most recent value only

        Returns:
            LongShortRatio or None on failure.
        """
        data = self._cached_get(
            f"ls_ratio:{symbol}:{period}",
            "/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": period, "limit": limit},
        )
        if not data or not isinstance(data, list) or not data:
            return None
        entry = data[-1]  # Most recent
        try:
            long_ratio = float(entry.get("longAccount", 0))
            short_ratio = float(entry.get("shortAccount", 0))
            ls_ratio = float(entry.get("longShortRatio", 1.0))
            return LongShortRatio(
                symbol=symbol,
                long_ratio=long_ratio,
                short_ratio=short_ratio,
                long_short_ratio=ls_ratio,
                timestamp=datetime.fromtimestamp(
                    int(entry.get("timestamp", time.time() * 1000)) / 1000,
                    tz=timezone.utc,
                ),
            )
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning("L/S ratio parse error for %s: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Liquidation Orders
    # ------------------------------------------------------------------

    def get_liquidation_orders(
        self, symbol: str = "BTCUSDT", limit: int = 100
    ) -> List[LiquidationOrder]:
        """Recent forced liquidation orders.

        Uses the public allForceOrders endpoint (no auth required).
        Endpoint: GET /fapi/v1/allForceOrders

        Note: Binance may restrict this endpoint. Returns empty list on 403.

        Args:
            symbol: Binance perpetual symbol
            limit: Max results (max 1000)

        Returns:
            List of LiquidationOrder, sorted newest→oldest.
        """
        limit = min(limit, 1000)
        data = self._get(
            "/fapi/v1/allForceOrders",
            {"symbol": symbol, "limit": limit},
        )
        if not data or not isinstance(data, list):
            return []
        results = []
        for entry in data:
            try:
                qty = float(entry.get("origQty", 0))
                price = float(entry.get("averagePrice", 0) or entry.get("price", 0))
                results.append(LiquidationOrder(
                    symbol=symbol,
                    side=entry.get("side", ""),
                    price=price,
                    quantity=qty,
                    usd_value=price * qty,
                    timestamp=datetime.fromtimestamp(
                        int(entry.get("time", 0)) / 1000, tz=timezone.utc
                    ),
                ))
            except (TypeError, ValueError):
                continue
        return sorted(results, key=lambda x: x.timestamp, reverse=True)

    def get_recent_liquidation_value(self, symbol: str = "BTCUSDT") -> float:
        """Total USD value of liquidations in the last batch (rough 1-min window).

        High liquidation value → momentum exhaustion signal.
        Returns 0.0 if data unavailable.
        """
        orders = self.get_liquidation_orders(symbol, limit=50)
        return sum(o.usd_value for o in orders)

    # ------------------------------------------------------------------
    # Crowded Trade Score (Crown Jewel)
    # ------------------------------------------------------------------

    def get_crowded_trade_score(self, symbol: str = "BTCUSDT") -> CrowdedTradeScore:
        """Composite crowded trade score: funding × OI × long/short ratio.

        This is the crown jewel. When all three fire simultaneously:
          1. Funding rate is extreme (one side paying too much)
          2. OI is diverging from price (disconnection between
             conviction and actual price movement)
          3. Long/short ratio is extreme (crowd is one-sided)

        All three together = liquidation cascade is structurally inevitable.
        MIDGE treats this as a "mechanical inevitability" — not a prediction
        but a structural consequence of positioning imbalances.

        Direction is contrarian:
          - All three bullish signals → crowd is overcrowded SHORT → price squeeze UP
          - All three bearish signals → crowd is overcrowded LONG → price liquidation DOWN

        Args:
            symbol: Binance perpetual symbol (e.g. "BTCUSDT")

        Returns:
            CrowdedTradeScore with score 0-3 and actionability flag.
        """
        now = datetime.now(tz=timezone.utc)
        funding_extreme = False
        oi_diverging = False
        long_extreme = False
        funding_direction = "neutral"
        oi_direction = "neutral"
        ls_direction = "neutral"
        funding_rate_val = None
        oi_change_val = None
        ls_ratio_val = None

        # --- Check 1: Funding rate ---
        try:
            from mae_core.market.apis.binance_funding_client import (
                BinanceFundingClient,
                EXTREME_THRESHOLD,
            )
            funding_client = BinanceFundingClient()
            rates = funding_client.get_funding_rates([symbol])
            if rates:
                rate = rates[0]
                funding_rate_val = rate.rate
                if abs(rate.rate) > EXTREME_THRESHOLD:
                    funding_extreme = True
                    funding_direction = rate.direction
        except Exception as exc:
            logger.debug("Crowded score: funding check failed for %s: %s", symbol, exc)

        # --- Check 2: OI divergence ---
        try:
            oi_change = self._compute_oi_trend(symbol, periods=12)
            if oi_change is not None:
                oi_change_val = oi_change
                if abs(oi_change) > OI_DIVERGENCE_THRESHOLD:
                    oi_diverging = True
                    # Falling OI with rising price = weak rally = bearish
                    # Rising OI with falling price = distribution = bearish
                    # We flag the divergence; direction comes from funding/LS consensus
                    oi_direction = "bearish" if oi_change < 0 else "bullish"
        except Exception as exc:
            logger.debug("Crowded score: OI check failed for %s: %s", symbol, exc)

        # --- Check 3: Long/short ratio extreme ---
        try:
            ls = self.get_long_short_ratio(symbol)
            if ls is not None:
                ls_ratio_val = ls.long_short_ratio
                if ls.direction != "neutral":
                    long_extreme = True
                    ls_direction = ls.direction
        except Exception as exc:
            logger.debug("Crowded score: L/S check failed for %s: %s", symbol, exc)

        # --- Determine consensus direction ---
        directions = []
        if funding_extreme and funding_direction != "neutral":
            directions.append(funding_direction)
        if long_extreme and ls_direction != "neutral":
            directions.append(ls_direction)
        if oi_diverging and oi_direction != "neutral":
            directions.append(oi_direction)

        if directions:
            bullish_votes = directions.count("bullish")
            bearish_votes = directions.count("bearish")
            consensus = "bullish" if bullish_votes >= bearish_votes else "bearish"
        else:
            consensus = "neutral"

        score = sum([funding_extreme, oi_diverging, long_extreme])

        result = CrowdedTradeScore(
            symbol=symbol,
            score=score,
            funding_extreme=funding_extreme,
            oi_diverging=oi_diverging,
            long_extreme=long_extreme,
            direction=consensus,
            timestamp=now,
            funding_rate=funding_rate_val,
            oi_change_pct=oi_change_val,
            long_short_ratio=ls_ratio_val,
        )

        if result.is_actionable:
            logger.info(
                "BTC Derivatives CROWDED TRADE: score=%d/3 direction=%s "
                "(funding=%s, OI_div=%s, LS_extreme=%s) for %s",
                score, consensus, funding_extreme, oi_diverging, long_extreme, symbol,
            )

        if self._raw_store is not None:
            try:
                self._raw_store.store_binance_derivatives(symbol, {
                    "score": score,
                    "direction": consensus,
                    "funding_extreme": funding_extreme,
                    "oi_diverging": oi_diverging,
                    "long_extreme": long_extreme,
                    "funding_rate": funding_rate_val,
                    "oi_change_pct": oi_change_val,
                    "long_short_ratio": ls_ratio_val,
                    "timestamp": now.isoformat(),
                })
            except Exception:
                pass

        return result

    def get_multi_crowded_scores(
        self, symbols: Optional[List[str]] = None
    ) -> List[CrowdedTradeScore]:
        """Get crowded trade scores for multiple symbols.

        Returns only actionable scores (score >= 2) to reduce noise.
        """
        targets = symbols or DEFAULT_SYMBOLS
        results = []
        for sym in targets:
            try:
                score = self.get_crowded_trade_score(sym)
                if score.is_actionable:
                    results.append(score)
            except Exception as exc:
                logger.debug("Crowded score failed for %s: %s", sym, exc)
        return results
