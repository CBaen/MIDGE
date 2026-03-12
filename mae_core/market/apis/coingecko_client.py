"""CoinGecko Client — Free crypto market data.

Provides cryptocurrency prices, global market data, and trending coins
via CoinGecko's free API. Used for 24/7 market exposure when equity
markets are closed.

Rate limit: 30 calls/min (free), 500/min (demo key).
Part of WP-E: Always-On MIDGE Wave 2.
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import requests

logger = logging.getLogger("midge.market.coingecko")

_BASE_URL = "https://api.coingecko.com/api/v3"
_RATE_LIMIT = 30  # calls per minute


@dataclass
class CryptoPrice:
    coin_id: str
    symbol: str
    price_usd: float
    volume_24h: float
    change_24h_pct: float
    change_7d_pct: float
    market_cap: float
    last_updated: str
    # Extended fields from CoinGecko /coins/markets response
    high_24h: Optional[float] = None            # 24h high price — intraday range context
    low_24h: Optional[float] = None             # 24h low price — intraday range context
    ath_change_percentage: Optional[float] = None  # % distance from all-time high (mean reversion signal)
    circulating_supply: Optional[float] = None  # Tokens in circulation (scarcity signal)
    total_supply: Optional[float] = None        # Total token supply (inflation signal)


@dataclass
class GlobalCryptoData:
    total_market_cap_usd: float
    btc_dominance: float
    eth_dominance: float
    market_cap_change_24h_pct: float
    active_cryptocurrencies: int


@dataclass
class TrendingCoin:
    coin_id: str
    symbol: str
    name: str
    market_cap_rank: int
    price_btc: float


class CoinGeckoClient:
    def __init__(self, api_key: str = None, raw_store=None):
        self._api_key = api_key or os.environ.get("COINGECKO_API_KEY", "")
        self._raw_store = raw_store
        self._session = requests.Session()
        if self._api_key:
            self._session.headers["x-cg-demo-api-key"] = self._api_key
        self._session.headers["Accept"] = "application/json"
        self._call_times: deque = deque(maxlen=_RATE_LIMIT)

    def _rate_limit(self):
        """Enforce 30 calls/min rate limit by sleeping when the window is full."""
        now = time.time()
        if len(self._call_times) >= _RATE_LIMIT:
            oldest = self._call_times[0]
            elapsed = now - oldest
            if elapsed < 60:
                sleep_time = 60 - elapsed + 0.1
                time.sleep(sleep_time)
        self._call_times.append(time.time())

    def _get(self, path: str, params: dict = None) -> dict:
        """Make a rate-limited GET request. Returns empty dict on any failure."""
        self._rate_limit()
        url = f"{_BASE_URL}{path}"
        try:
            resp = self._session.get(url, params=params or {}, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            logger.debug("CoinGecko request failed: %s", path, exc_info=True)
            return {}

    def get_prices(self, coins: List[str] = None) -> List[CryptoPrice]:
        """Get current prices for a list of coins.

        Default coins: BTC, ETH, SOL, XRP, ADA (top 5 by market cap).
        Returns an empty list on any network or parse failure.
        """
        coins = coins or ["bitcoin", "ethereum", "solana", "ripple", "cardano"]
        ids = ",".join(coins)
        data = self._get("/coins/markets", {
            "vs_currency": "usd",
            "ids": ids,
            "order": "market_cap_desc",
            "price_change_percentage": "7d",
            "sparkline": "false",
        })
        if not isinstance(data, list):
            return []
        results = []
        for coin in data:
            try:
                results.append(CryptoPrice(
                    coin_id=coin.get("id", ""),
                    symbol=coin.get("symbol", "").upper(),
                    price_usd=float(coin.get("current_price", 0)),
                    volume_24h=float(coin.get("total_volume", 0)),
                    change_24h_pct=float(coin.get("price_change_percentage_24h", 0) or 0),
                    change_7d_pct=float(
                        coin.get("price_change_percentage_7d_in_currency", 0) or 0
                    ),
                    market_cap=float(coin.get("market_cap", 0)),
                    last_updated=coin.get("last_updated", ""),
                    high_24h=float(coin["high_24h"]) if coin.get("high_24h") is not None else None,
                    low_24h=float(coin["low_24h"]) if coin.get("low_24h") is not None else None,
                    ath_change_percentage=(
                        float(coin["ath_change_percentage"])
                        if coin.get("ath_change_percentage") is not None else None
                    ),
                    circulating_supply=(
                        float(coin["circulating_supply"])
                        if coin.get("circulating_supply") is not None else None
                    ),
                    total_supply=(
                        float(coin["total_supply"])
                        if coin.get("total_supply") is not None else None
                    ),
                ))
            except (TypeError, ValueError):
                continue
        if self._raw_store is not None and results:
            try:
                self._raw_store.store_coingecko_prices(results)
            except Exception:
                pass
        return results

    def get_market_data(self) -> Optional[GlobalCryptoData]:
        """Get global crypto market data: BTC dominance, total market cap, etc.

        Returns None on any network or parse failure.
        """
        data = self._get("/global")
        if not data or "data" not in data:
            return None
        d = data["data"]
        try:
            total_mcap = sum(d.get("total_market_cap", {}).values())
            return GlobalCryptoData(
                total_market_cap_usd=float(total_mcap),
                btc_dominance=float(d.get("market_cap_percentage", {}).get("btc", 0)),
                eth_dominance=float(d.get("market_cap_percentage", {}).get("eth", 0)),
                market_cap_change_24h_pct=float(
                    d.get("market_cap_change_percentage_24h_usd", 0)
                ),
                active_cryptocurrencies=int(d.get("active_cryptocurrencies", 0)),
            )
        except (TypeError, ValueError):
            return None

    def get_trending(self) -> List[TrendingCoin]:
        """Get trending coins — most searched in the last 24 hours.

        Returns an empty list on any network or parse failure.
        """
        data = self._get("/search/trending")
        if not data or "coins" not in data:
            return []
        results = []
        for item in data["coins"]:
            coin = item.get("item", {})
            try:
                results.append(TrendingCoin(
                    coin_id=coin.get("id", ""),
                    symbol=coin.get("symbol", "").upper(),
                    name=coin.get("name", ""),
                    market_cap_rank=int(coin.get("market_cap_rank", 0) or 0),
                    price_btc=float(coin.get("price_btc", 0) or 0),
                ))
            except (TypeError, ValueError):
                continue
        return results
