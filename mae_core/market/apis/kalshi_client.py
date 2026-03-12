"""Kalshi Prediction Market Client — crowd probability signals.

Kalshi is a CFTC-regulated prediction market where contract prices directly
represent crowd-estimated probabilities (1-99 cents = 1-99%). When 10,000
traders price "Fed raises rates" at 85 cents, that's 85% crowd consensus.

MIDGE uses this as a signal source: prediction market movers are signals
that crowd consensus is shifting. Combined with MIDGE's other domains
(macro, insider, technical, etc.), this creates powerful convergence.

Dual use planned:
  Phase 1 (now): Signal source — read market data for convergence
  Phase 2 (future): Execution venue — trade on inevitabilities

Auth: RSA-PSS key pair (daemon-compatible, no session tokens).
Env vars: KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH
Demo: KALSHI_DEMO=1 uses demo-api.kalshi.co (default until production ready)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEMO_HOST = "https://demo-api.kalshi.co/trade-api/v2"
PROD_HOST = "https://api.kalshi.com/trade-api/v2"

# Categories of Kalshi markets relevant to financial convergence
_FINANCIAL_CATEGORIES = {
    "economics", "fed", "inflation", "cpi", "gdp", "unemployment",
    "interest", "recession", "treasury", "fomc",
}
_GEOPOLITICAL_CATEGORIES = {
    "election", "regulation", "government", "policy", "tariff",
    "sanction", "legislation",
}


@dataclass
class KalshiMarket:
    """A single Kalshi prediction market."""
    ticker: str                    # e.g. "FED-RATE-25MAR12"
    title: str                     # e.g. "Will the Fed raise rates in March?"
    event_ticker: str              # parent event
    yes_price: float               # 0.0-1.0 probability (converted from cents)
    volume_24h: int                # 24h trading volume
    open_interest: int             # outstanding contracts
    status: str                    # "active", "closed", "settled"
    close_time: Optional[datetime] = None
    result: Optional[str] = None   # "yes", "no", None (unsettled)
    category: str = ""             # derived: "macro", "geopolitical", "other"
    previous_yes_price: float = 0.0  # for calculating price moves


@dataclass
class KalshiMover:
    """A market with significant recent price movement."""
    market: KalshiMarket
    price_change: float            # absolute change in probability (0.0-1.0)
    direction: str                 # "bullish" (price up) or "bearish" (price down)
    strength: float                # 0.0-1.0 signal strength


class KalshiMarketClient:
    """Kalshi prediction market client for MIDGE's sensing pipeline.

    Reads market data and identifies movers (significant probability shifts).
    Does NOT place trades — that's Phase 2.

    Usage:
        client = KalshiMarketClient()  # demo mode by default
        movers = client.get_market_movers()
    """

    def __init__(
        self,
        api_key_id: str = "",
        private_key_path: str = "",
        demo: bool = True,
        raw_store=None,
        min_volume: int = 100,
        min_move_pct: float = 5.0,
    ):
        self._api_key_id = api_key_id or os.environ.get("KALSHI_API_KEY_ID", "") or os.environ.get("KALSHI_API_KEY", "")
        self._private_key_path = private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
        self._demo = demo if not os.environ.get("KALSHI_DEMO", "1") == "0" else False
        self._raw_store = raw_store
        self._min_volume = min_volume
        self._min_move_pct = min_move_pct
        self._client = None
        self._last_prices: Dict[str, float] = {}  # ticker → last seen yes_price
        self._call_count = 0
        self._error_count = 0
        self._initialized = False

    def _load_private_key(self) -> Optional[str]:
        """Load RSA private key from file path or env var.

        Supports three sources (checked in order):
          1. File path via KALSHI_PRIVATE_KEY_PATH
          2. Raw PEM content via KALSHI_RSA_PRIVATE_KEY
          3. Constructor private_key_path argument (file or inline)
        """
        # Try file path first
        key_path = self._private_key_path
        if key_path and os.path.exists(key_path):
            with open(key_path, "r") as f:
                return f.read()

        # Try raw key from env var
        raw_key = os.environ.get("KALSHI_RSA_PRIVATE_KEY", "")
        if raw_key:
            # Wrap in PEM headers if missing
            if "-----BEGIN" not in raw_key:
                raw_key = (
                    "-----BEGIN RSA PRIVATE KEY-----\n"
                    + raw_key.strip()
                    + "\n-----END RSA PRIVATE KEY-----\n"
                )
            return raw_key

        return None

    def _ensure_client(self) -> bool:
        """Lazy-init the SDK client. Returns True if ready."""
        if self._initialized:
            return self._client is not None
        self._initialized = True

        if not self._api_key_id:
            logger.debug("Kalshi client: no API key configured")
            return False

        pem = self._load_private_key()
        if not pem:
            logger.debug("Kalshi client: no private key found (checked file path + KALSHI_RSA_PRIVATE_KEY)")
            return False

        try:
            from kalshi_python_sync import Configuration, KalshiClient

            host = DEMO_HOST if self._demo else PROD_HOST
            config = Configuration(host=host)
            config.api_key_id = self._api_key_id
            config.private_key_pem = pem

            self._client = KalshiClient(config)
            mode = "DEMO" if self._demo else "PRODUCTION"
            logger.info("Kalshi client initialized (%s mode)", mode)
            return True

        except ImportError:
            logger.debug("kalshi_python_sync not installed")
            return False
        except Exception:
            logger.debug("Kalshi client init failed", exc_info=True)
            return False

    def get_active_markets(
        self,
        limit: int = 200,
        status: str = "open",
    ) -> List[KalshiMarket]:
        """Fetch active markets from Kalshi.

        Returns list of KalshiMarket with prices converted to 0.0-1.0 range.
        Filters to markets with minimum volume threshold.
        """
        if not self._ensure_client():
            return []

        try:
            response = self._client.get_markets(
                limit=min(limit, 1000),
                status=status,
            )
            self._call_count += 1

            markets = []
            for m in getattr(response, "markets", []) or []:
                ticker = getattr(m, "ticker", "") or ""
                title = getattr(m, "title", "") or ""
                event_ticker = getattr(m, "event_ticker", "") or ""

                # Price: cents → probability (0.0-1.0)
                yes_price = (getattr(m, "yes_bid", None) or 0) / 100.0
                volume = getattr(m, "volume_24h", 0) or 0
                oi = getattr(m, "open_interest", 0) or 0

                if volume < self._min_volume:
                    continue

                close_ts = getattr(m, "close_time", None)
                close_time = None
                if close_ts:
                    try:
                        close_time = datetime.fromisoformat(str(close_ts).replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        pass

                category = self._categorize(title, event_ticker)

                markets.append(KalshiMarket(
                    ticker=ticker,
                    title=title,
                    event_ticker=event_ticker,
                    yes_price=yes_price,
                    volume_24h=volume,
                    open_interest=oi,
                    status=getattr(m, "status", "active") or "active",
                    close_time=close_time,
                    category=category,
                    previous_yes_price=self._last_prices.get(ticker, yes_price),
                ))

            # Store raw response
            if self._raw_store is not None:
                try:
                    self._raw_store.store_kalshi_markets(
                        [{"ticker": mk.ticker, "title": mk.title,
                          "yes_price": mk.yes_price, "volume_24h": mk.volume_24h,
                          "category": mk.category}
                         for mk in markets[:50]]
                    )
                except Exception:
                    pass

            # Update price cache
            for mk in markets:
                self._last_prices[mk.ticker] = mk.yes_price

            return markets

        except Exception:
            self._error_count += 1
            logger.debug("Kalshi get_active_markets failed", exc_info=True)
            return []

    def get_market_movers(self) -> List[KalshiMover]:
        """Identify markets with significant probability shifts.

        Compares current prices against cached previous prices.
        Returns movers sorted by absolute price change (largest first).
        """
        markets = self.get_active_markets()
        if not markets:
            return []

        movers = []
        for mk in markets:
            price_change = mk.yes_price - mk.previous_yes_price
            abs_change = abs(price_change)

            # Filter: minimum move threshold (default 5%)
            if abs_change < self._min_move_pct / 100.0:
                continue

            # Direction: price going UP means crowd consensus is strengthening
            # for the "yes" outcome. This is "bullish" for the event.
            direction = "bullish" if price_change > 0 else "bearish"

            # Strength: scales with move size. 5% move = 0.25, 20% move = 1.0
            strength = min(1.0, abs_change / 0.20)

            movers.append(KalshiMover(
                market=mk,
                price_change=price_change,
                direction=direction,
                strength=round(strength, 3),
            ))

        movers.sort(key=lambda m: abs(m.price_change), reverse=True)

        if movers:
            logger.info(
                "Kalshi: %d market movers detected (top: %s %.1f%%)",
                len(movers),
                movers[0].market.ticker,
                movers[0].price_change * 100,
            )

        return movers

    def _categorize(self, title: str, event_ticker: str) -> str:
        """Categorize a market by its title/ticker into MIDGE domains."""
        combined = (title + " " + event_ticker).lower()

        for keyword in _FINANCIAL_CATEGORIES:
            if keyword in combined:
                return "macro"

        for keyword in _GEOPOLITICAL_CATEGORIES:
            if keyword in combined:
                return "geopolitical"

        return "other"

    def get_statistics(self) -> dict:
        """Return client statistics."""
        return {
            "calls_made": self._call_count,
            "errors": self._error_count,
            "cached_prices": len(self._last_prices),
            "mode": "demo" if self._demo else "production",
            "initialized": self._initialized,
            "connected": self._client is not None,
        }
