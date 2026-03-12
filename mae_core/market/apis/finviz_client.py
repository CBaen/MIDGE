"""FinViz Client — Insider trades, unusual volume, short squeeze screening.

Uses finvizfinance library for FinViz data access. Provides:
- Latest insider transactions
- Unusual volume stocks (volume > 2x average)
- High short float stocks (squeeze setup detection)

Rate limit: 1 request per 5 seconds.
Part of WP-F3: Always-On MIDGE Wave 3.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("midge.market.finviz")

_RATE_LIMIT_SECONDS = 5.0

# Try importing finvizfinance — graceful degradation if not available
_HAS_FINVIZ = False
try:
    from finvizfinance.insider import Insider
    from finvizfinance.screener.overview import Overview
    _HAS_FINVIZ = True
except ImportError:
    logger.info("finvizfinance not installed — FinViz client will return empty results")


@dataclass
class FinVizInsiderTrade:
    ticker: str
    owner_name: str
    relationship: str  # CEO, CFO, Director, 10% Owner
    transaction_type: str  # Buy, Sale
    date: str
    shares_traded: int
    value: float
    shares_owned: int


@dataclass
class UnusualVolume:
    ticker: str
    company: str
    sector: str
    price: float
    change_pct: float
    volume: int
    avg_volume: int
    volume_ratio: float  # volume / avg_volume


@dataclass
class ShortSqueezeCandidate:
    ticker: str
    company: str
    sector: str
    price: float
    short_float_pct: float
    short_ratio: float  # days to cover
    float_shares: int
    short_interest: int


class FinVizClient:
    def __init__(self, raw_store=None):
        self._raw_store = raw_store
        self._last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < _RATE_LIMIT_SECONDS:
            time.sleep(_RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()

    def get_insider_trades(self) -> List[FinVizInsiderTrade]:
        """Get latest insider transactions from FinViz."""
        if not _HAS_FINVIZ:
            return []
        self._rate_limit()
        try:
            insider = Insider()
            df = insider.getInsider()
            if df is None or df.empty:
                return []
            results = []
            for _, row in df.head(50).iterrows():
                try:
                    results.append(FinVizInsiderTrade(
                        ticker=str(row.get("Ticker", "")),
                        owner_name=str(row.get("Owner", "")),
                        relationship=str(row.get("Relationship", "")),
                        transaction_type=str(row.get("Transaction", "")),
                        date=str(row.get("Date", "")),
                        shares_traded=int(float(str(row.get("Shares Traded", 0)).replace(",", "") or 0)),
                        value=float(str(row.get("Value ($)", 0)).replace(",", "").replace("$", "") or 0),
                        shares_owned=int(float(str(row.get("Shares Owned", 0)).replace(",", "") or 0)),
                    ))
                except (ValueError, TypeError):
                    continue
            if self._raw_store is not None and results:
                try:
                    self._raw_store.store_finviz_insider_trades(results)
                except Exception:
                    pass
            return results
        except Exception:
            logger.debug("FinViz insider trades fetch failed", exc_info=True)
            return []

    def get_unusual_volume(self) -> List[UnusualVolume]:
        """Get stocks with unusual volume (>2x average)."""
        if not _HAS_FINVIZ:
            return []
        self._rate_limit()
        try:
            overview = Overview()
            # Filter: unusual volume over 200% of average
            overview.set_filter(filters_dict={"Relative Volume": "Over 2"})
            df = overview.screener_view()
            if df is None or df.empty:
                return []
            results = []
            for _, row in df.head(30).iterrows():
                try:
                    volume = int(float(str(row.get("Volume", 0)).replace(",", "") or 0))
                    avg_vol = int(float(str(row.get("Avg Volume", 1)).replace(",", "") or 1))
                    results.append(UnusualVolume(
                        ticker=str(row.get("Ticker", "")),
                        company=str(row.get("Company", "")),
                        sector=str(row.get("Sector", "")),
                        price=float(str(row.get("Price", 0)).replace(",", "") or 0),
                        change_pct=float(str(row.get("Change", "0%")).replace("%", "").replace(",", "") or 0),
                        volume=volume,
                        avg_volume=avg_vol,
                        volume_ratio=volume / max(avg_vol, 1),
                    ))
                except (ValueError, TypeError):
                    continue
            if self._raw_store is not None and results:
                try:
                    self._raw_store.store_finviz_unusual_volume(results)
                except Exception:
                    pass
            return results
        except Exception:
            logger.debug("FinViz unusual volume fetch failed", exc_info=True)
            return []

    def get_high_short_float(self, min_pct: float = 20.0) -> List[ShortSqueezeCandidate]:
        """Get stocks with high short interest (potential squeeze setups)."""
        if not _HAS_FINVIZ:
            return []
        self._rate_limit()
        try:
            overview = Overview()
            overview.set_filter(filters_dict={"Short Float": "Over 20%"})
            df = overview.screener_view()
            if df is None or df.empty:
                return []
            results = []
            for _, row in df.head(30).iterrows():
                try:
                    results.append(ShortSqueezeCandidate(
                        ticker=str(row.get("Ticker", "")),
                        company=str(row.get("Company", "")),
                        sector=str(row.get("Sector", "")),
                        price=float(str(row.get("Price", 0)).replace(",", "") or 0),
                        short_float_pct=float(str(row.get("Short Float", "0%")).replace("%", "") or 0),
                        short_ratio=float(str(row.get("Short Ratio", 0)).replace(",", "") or 0),
                        float_shares=int(float(str(row.get("Float", 0)).replace(",", "").replace("M", "e6").replace("B", "e9").replace("K", "e3") or 0)),
                        short_interest=int(float(str(row.get("Short Interest", 0)).replace(",", "").replace("M", "e6").replace("B", "e9").replace("K", "e3") or 0)),
                    ))
                except (ValueError, TypeError):
                    continue
            if self._raw_store is not None and results:
                try:
                    self._raw_store.store_finviz_short_squeeze(results)
                except Exception:
                    pass
            return results
        except Exception:
            logger.debug("FinViz high short float fetch failed", exc_info=True)
            return []
