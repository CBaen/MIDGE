"""CoinCap Client — Free cryptocurrency market data.

Simple, no-auth API for basic crypto price and volume data.
Complements CoinGecko (different source = domain diversity for convergence).

No API key required. Rate limit: 200 requests/min.
Part of WP-E: Always-On MIDGE Wave 2.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import requests

logger = logging.getLogger("midge.market.coincap")

_BASE_URL = "https://api.coincap.io/v2"
_RATE_LIMIT = 200


@dataclass
class CryptoAsset:
    asset_id: str
    symbol: str
    name: str
    rank: int
    price_usd: float
    volume_24h_usd: float
    change_24h_pct: float
    market_cap_usd: float
    supply: float
    max_supply: Optional[float]
    vwap_24h: Optional[float]


@dataclass
class CryptoCandle:
    open: float
    high: float
    low: float
    close: float
    volume: float
    period: int  # Unix timestamp


class CoinCapClient:
    def __init__(self, raw_store=None):
        self._raw_store = raw_store
        self._session = requests.Session()
        self._session.headers["Accept-Encoding"] = "gzip"
        self._call_times: deque = deque(maxlen=_RATE_LIMIT)

    def _rate_limit(self):
        now = time.time()
        if len(self._call_times) >= _RATE_LIMIT:
            oldest = self._call_times[0]
            elapsed = now - oldest
            if elapsed < 60:
                time.sleep(60 - elapsed + 0.1)
        self._call_times.append(time.time())

    def _get(self, path: str, params: dict = None) -> dict:
        self._rate_limit()
        url = f"{_BASE_URL}{path}"
        try:
            resp = self._session.get(url, params=params or {}, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            logger.debug("CoinCap request failed: %s", path, exc_info=True)
            return {}

    def get_assets(self, limit: int = 20) -> List[CryptoAsset]:
        """Get top N crypto assets by market cap."""
        data = self._get("/assets", {"limit": limit})
        if not data or "data" not in data:
            return []
        results = []
        for asset in data["data"]:
            try:
                results.append(CryptoAsset(
                    asset_id=asset.get("id", ""),
                    symbol=asset.get("symbol", ""),
                    name=asset.get("name", ""),
                    rank=int(asset.get("rank", 0)),
                    price_usd=float(asset.get("priceUsd", 0) or 0),
                    volume_24h_usd=float(asset.get("volumeUsd24Hr", 0) or 0),
                    change_24h_pct=float(asset.get("changePercent24Hr", 0) or 0),
                    market_cap_usd=float(asset.get("marketCapUsd", 0) or 0),
                    supply=float(asset.get("supply", 0) or 0),
                    max_supply=float(asset["maxSupply"]) if asset.get("maxSupply") else None,
                    vwap_24h=float(asset["vwap24Hr"]) if asset.get("vwap24Hr") else None,
                ))
            except (TypeError, ValueError, KeyError):
                continue
        if self._raw_store is not None and results:
            try:
                self._raw_store.store_coincap_assets(results)
            except Exception:
                pass
        return results

    def get_asset_history(self, asset_id: str, interval: str = "h1") -> List[CryptoCandle]:
        """Get historical price data for an asset.

        interval options: m1, m5, m15, m30, h1, h2, h6, h12, d1
        Note: CoinCap history endpoint returns one price per interval (no OHLC).
        open/high/low/close are all set to the same priceUsd value.
        """
        data = self._get(f"/assets/{asset_id}/history", {"interval": interval})
        if not data or "data" not in data:
            return []
        results = []
        for entry in data["data"]:
            try:
                results.append(CryptoCandle(
                    open=float(entry.get("priceUsd", 0)),
                    high=float(entry.get("priceUsd", 0)),  # CoinCap only gives one price per interval
                    low=float(entry.get("priceUsd", 0)),
                    close=float(entry.get("priceUsd", 0)),
                    volume=0.0,  # Not available in history endpoint
                    period=int(entry.get("time", 0)),
                ))
            except (TypeError, ValueError):
                continue
        return results
