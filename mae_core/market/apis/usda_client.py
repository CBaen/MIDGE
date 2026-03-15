"""
usda_client.py - USDA World Agricultural Supply and Demand Estimates (WASDE) Client

Fetches agricultural supply/demand data from the USDA Foreign Agricultural Service
Production, Supply and Distribution (PSD) Open Data API.

Free API — no authentication required for public data.
API base: https://apps.fas.usda.gov/OpenData/api/psd/

Agricultural data matters for:
  - Commodity futures: wheat (ZW), corn (ZC), soybeans (ZS), cotton (CT)
  - Sector ETFs: DBA (agriculture ETF), MOO (agribusiness)
  - Individual names: ADM, BG, CTVA, MOS, NTR

Supply surplus = bearish for that commodity. Supply deficit = bullish.
WASDE is released monthly (usually 2nd Tuesday). PSD database is near real-time.

Rate limiting: 1 req/sec conservative to respect free tier.
Cache: 6 hours — monthly data updates rarely intraday.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import httpx

logger = logging.getLogger(__name__)

USDA_PSD_BASE = "https://apps.fas.usda.gov/OpenData/api/psd"

# Rate limiting — conservative for free tier
REQUEST_DELAY = 1.0  # 1 second between requests

# Cache duration — monthly reports, update at most daily
CACHE_DURATION = 6 * 3600  # 6 hours

# USDA commodity codes (PSD commodity codes)
# https://apps.fas.usda.gov/OpenData/api/psd/commodities
USDA_COMMODITIES: Dict[str, dict] = {
    "wheat": {
        "code": "0410000",
        "name": "Wheat",
        "futures_ticker": "ZW",
        "affected_tickers": ["WEAT", "DBA", "ADM", "BG"],
    },
    "corn": {
        "code": "0440000",
        "name": "Corn",
        "futures_ticker": "ZC",
        "affected_tickers": ["CORN", "DBA", "ADM", "BG", "CTVA"],
    },
    "soybeans": {
        "code": "2222000",
        "name": "Soybeans",
        "futures_ticker": "ZS",
        "affected_tickers": ["SOYB", "DBA", "ADM", "BG", "MOS"],
    },
    "cotton": {
        "code": "0813100",
        "name": "Upland Cotton",
        "futures_ticker": "CT",
        "affected_tickers": ["BAL", "DBA"],
    },
}

# PSD attribute IDs for supply/demand components
# https://apps.fas.usda.gov/OpenData/api/psd/commodityAttributes
ATTR_PRODUCTION = 20              # Production (million metric tons or thousand 480-lb bales)
ATTR_DOMESTIC_CONSUMPTION = 176   # Total domestic consumption
ATTR_ENDING_STOCKS = 176          # Ending stocks (surrogate for surplus/deficit)
# Note: ending stocks = production + imports - consumption - exports
# We compute supply_surplus_ratio = ending_stocks / consumption

# World region code
WORLD_REGION = "0"   # Global aggregate


@dataclass
class AgriculturalIndicator:
    """A single agricultural supply/demand reading from USDA PSD.

    Carries MIDGE signal metadata for convergence weighting.
    Agricultural signals decay moderately — monthly reports supersede.

    The supply_surplus_ratio is the core signal:
      > 1.0 = surplus (bearish for commodity price)
      < 0.7 = deficit (bullish for commodity price)
      0.7-1.0 = neutral
    """
    commodity_key: str           # e.g. "wheat"
    commodity_name: str          # e.g. "Wheat"
    market_year: str             # e.g. "2025/2026"
    production: float            # Production in metric tons (millions)
    consumption: float           # Total domestic consumption
    ending_stocks: float         # Ending stocks
    supply_surplus_ratio: float  # ending_stocks / consumption (stocks-to-use ratio)
    direction: str               # "bullish", "bearish", "neutral"
    strength: float              # 0.0-1.0
    affected_tickers: list       # Commodity-sensitive tickers
    futures_ticker: str          # Primary futures instrument
    signal_source: str = "usda_agriculture"
    decay_rate: float = 0.03     # ~23 day half-life (monthly reports, slow-moving)
    confidence: float = 0.65     # Government data reliable but lags reality


def _compute_direction_strength(stocks_to_use: float) -> tuple:
    """Derive direction and strength from the stocks-to-use ratio.

    Stocks-to-use ratio is the canonical agricultural supply/demand measure.
    Higher ratio = more surplus = bearish for commodity price.

    Historical tight supply: wheat <0.25, corn <0.12, soy <0.10.
    Historical loose supply: wheat >0.35, corn >0.18, soy >0.16.

    We use conservative thresholds that work across all four crops.

    Returns:
        (direction, strength) tuple
    """
    if stocks_to_use < 0.15:
        # Very tight — significant bullish
        strength = min(1.0, (0.15 - stocks_to_use) / 0.15)
        return "bullish", max(0.5, strength)
    elif stocks_to_use < 0.25:
        # Mild tightness
        strength = (0.25 - stocks_to_use) / 0.10
        return "bullish", round(max(0.2, min(0.5, strength)), 3)
    elif stocks_to_use > 0.40:
        # Loose supply — bearish
        strength = min(1.0, (stocks_to_use - 0.40) / 0.20)
        return "bearish", max(0.3, strength)
    elif stocks_to_use > 0.30:
        # Mild surplus
        strength = (stocks_to_use - 0.30) / 0.10
        return "bearish", round(max(0.2, min(0.4, strength)), 3)
    else:
        return "neutral", 0.2


class USDAClient:
    """Client for USDA Foreign Agricultural Service PSD Open Data API.

    Fetches global supply/demand balance for key agricultural commodities
    (wheat, corn, soybeans, cotton). Translates stocks-to-use ratios into
    AgriculturalIndicator objects that carry MIDGE signal metadata.

    The PSD API is free and requires no authentication. It covers
    official WASDE monthly estimates — the most authoritative agricultural
    supply/demand data in the world.

    Usage:
        client = USDAClient()
        snapshot = client.get_agricultural_snapshot()   # all 4 commodities
        wheat = client.get_commodity("wheat")           # single commodity

    Rate limits: Free tier — client enforces 1.0s delay.
    Cache: 6 hours — monthly reports rarely change intraday.
    """

    def __init__(self, raw_store=None, provider=None):
        """Initialize USDA client.

        Args:
            raw_store: Optional RawStore for persisting API responses.
            provider: Unused — USDAClient uses httpx directly. Accepted so
                      the bootstrap can pass a uniform kwargs dict.
        """
        self._raw_store = raw_store
        self._last_request_time: float = 0.0
        self._cache: Dict[str, tuple] = {}  # commodity_key -> (indicator, timestamp)

        # Reusable httpx client (follow existing project pattern: httpx not requests)
        self._client = httpx.Client(
            headers={"User-Agent": "MIDGE Trading Research"},
            timeout=30.0,
            follow_redirects=True,
        )
        logger.info("USDAClient initialized (free API, no key required)")

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _request(self, path: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Make a GET request to USDA PSD API.

        Args:
            path: URL path after base (e.g. "/commodityData")
            params: Query parameters

        Returns:
            Parsed JSON (list or dict) or None on failure
        """
        url = f"{USDA_PSD_BASE}{path}"
        try:
            self._rate_limit()
            response = self._client.get(url, params=params or {})
            if response.status_code == 200:
                return response.json()
            logger.warning(
                "USDA HTTP %s for %s: %s",
                response.status_code, url, response.text[:200],
            )
            return None
        except Exception as exc:
            logger.error("USDA request exception for %s: %s", url, exc)
            return None

    def get_commodity(self, commodity_key: str) -> Optional[AgriculturalIndicator]:
        """Fetch the latest supply/demand data for a single commodity.

        Queries global (world aggregate) production, consumption, and ending
        stocks for the current and most recent complete market year. Computes
        the stocks-to-use ratio as the primary direction signal.

        Args:
            commodity_key: One of "wheat", "corn", "soybeans", "cotton"

        Returns:
            AgriculturalIndicator or None on failure / unknown commodity
        """
        if commodity_key not in USDA_COMMODITIES:
            logger.warning("Unknown USDA commodity key: %s", commodity_key)
            return None

        # Cache check
        if commodity_key in self._cache:
            indicator, cached_at = self._cache[commodity_key]
            if time.time() - cached_at < CACHE_DURATION:
                return indicator

        comm_def = USDA_COMMODITIES[commodity_key]
        commodity_code = comm_def["code"]

        # Fetch commodity data for world region, all recent market years
        # PSD API: /commodityData?commodityCode={code}&countryCode=0000
        # countryCode=0000 = World
        data = self._request(
            "/commodityData",
            params={
                "commodityCode": commodity_code,
                "countryCode": "0000",  # World aggregate
            },
        )

        if data is None:
            return None

        if self._raw_store:
            try:
                self._raw_store.store_usda_data(commodity_key, data if isinstance(data, list) else [])
            except Exception as exc:
                logger.debug("RawStore USDA write failed: %s", exc)

        if not isinstance(data, list) or not data:
            logger.warning("USDA returned empty/non-list data for %s", commodity_key)
            return None

        # Data is a list of records with marketYear, attributeId, value, unitDescription
        # Group by market year, then pick the most recent year with all three attributes
        by_year: Dict[str, Dict[int, float]] = {}
        for record in data:
            year = str(record.get("marketYear", ""))
            attr_id = int(record.get("attributeId", 0))
            raw_val = record.get("value")
            if year and raw_val is not None:
                try:
                    val = float(raw_val)
                    if year not in by_year:
                        by_year[year] = {}
                    by_year[year][attr_id] = val
                except (TypeError, ValueError):
                    pass

        # Pick most recent year that has production (20), consumption (176), ending stocks (125)
        # PSD attribute 125 = Ending Stocks, 176 = Total Dom. Consumption, 20 = Production
        ATTR_PROD = 20
        ATTR_CONS = 176
        ATTR_STOCKS = 125

        best_indicator = None
        for year in sorted(by_year.keys(), reverse=True):
            attrs = by_year[year]
            prod = attrs.get(ATTR_PROD)
            cons = attrs.get(ATTR_CONS)
            stocks = attrs.get(ATTR_STOCKS)

            if prod is None or cons is None or stocks is None:
                continue
            if cons <= 0:
                continue

            stocks_to_use = stocks / cons
            direction, strength = _compute_direction_strength(stocks_to_use)

            best_indicator = AgriculturalIndicator(
                commodity_key=commodity_key,
                commodity_name=comm_def["name"],
                market_year=year,
                production=round(prod, 2),
                consumption=round(cons, 2),
                ending_stocks=round(stocks, 2),
                supply_surplus_ratio=round(stocks_to_use, 4),
                direction=direction,
                strength=round(strength, 3),
                affected_tickers=comm_def["affected_tickers"],
                futures_ticker=comm_def["futures_ticker"],
            )
            break  # Take the most recent complete year

        if best_indicator is None:
            logger.warning(
                "USDA: could not compute indicator for %s (missing attributes in all years)",
                commodity_key,
            )
            return None

        self._cache[commodity_key] = (best_indicator, time.time())
        logger.debug(
            "USDA %s (%s): stocks_to_use=%.3f → %s (strength=%.3f)",
            commodity_key, best_indicator.market_year,
            best_indicator.supply_surplus_ratio,
            best_indicator.direction, best_indicator.strength,
        )
        return best_indicator

    def get_agricultural_snapshot(self) -> List[AgriculturalIndicator]:
        """Fetch supply/demand indicators for all four key commodities.

        Returns:
            List of AgriculturalIndicators, one per available commodity.
            Failures are skipped silently (logged at WARNING level).
        """
        results = []
        for commodity_key in USDA_COMMODITIES:
            indicator = self.get_commodity(commodity_key)
            if indicator is not None:
                results.append(indicator)
            else:
                logger.warning(
                    "get_agricultural_snapshot: failed to fetch %s", commodity_key
                )
        return results


def get_agricultural_snapshot() -> List[AgriculturalIndicator]:
    """Convenience function: get all key agricultural indicators."""
    client = USDAClient()
    return client.get_agricultural_snapshot()


if __name__ == "__main__":
    import sys

    print("USDA Agricultural Data Test")
    print("=" * 60)

    client = USDAClient()

    commodity = sys.argv[1] if len(sys.argv) > 1 else "wheat"
    print(f"\nFetching: {commodity}")
    indicator = client.get_commodity(commodity)
    if indicator:
        arrow = {"bullish": "+", "bearish": "-", "neutral": "~"}[indicator.direction]
        print(f"  {indicator.commodity_name} ({indicator.market_year})")
        print(f"  Production:      {indicator.production:,.1f} MMT")
        print(f"  Consumption:     {indicator.consumption:,.1f} MMT")
        print(f"  Ending Stocks:   {indicator.ending_stocks:,.1f} MMT")
        print(f"  Stocks-to-Use:   {indicator.supply_surplus_ratio:.3f}")
        print(f"  Direction:       [{arrow}] {indicator.direction}")
        print(f"  Strength:        {indicator.strength:.3f}")
        print(f"  Futures:         {indicator.futures_ticker}")
        print(f"  Affected:        {', '.join(indicator.affected_tickers[:4])}")
    else:
        print(f"  Failed to fetch {commodity}")

    print("\nAgricultural Snapshot (all key commodities):")
    snapshot = client.get_agricultural_snapshot()
    if snapshot:
        for m in snapshot:
            arrow = {"bullish": "+", "bearish": "-", "neutral": "~"}[m.direction]
            print(
                f"  [{arrow}] {m.commodity_key:<12} "
                f"stocks/use={m.supply_surplus_ratio:.3f}  "
                f"({m.market_year})  {m.direction}"
            )
    else:
        print("  No indicators available")

    print("\nDone.")
