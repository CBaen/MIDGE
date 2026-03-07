"""
eia_client.py - U.S. Energy Information Administration (EIA) API Client

Fetches energy market data from the EIA Open Data API v2.
Free registration: https://www.eia.gov/opendata/register.php
Rate limits: ~9,000 requests/hour, burst <5/second, 5,000 rows per response.

Energy data is a real-economy signal that precedes market moves. A crude
inventory build (supply > demand) is bearish for energy stocks. A natural
gas storage draw in winter is bullish for gas producers. These are signals
nobody else cross-references with insider trades and congressional activity.

Key data:
  - Weekly Petroleum Status Report (every Wednesday 10:30 AM ET)
  - Weekly Natural Gas Storage Report (every Thursday 10:30 AM ET)
  - Monthly production data

API base: https://api.eia.gov/v2
Auth: ?api_key={KEY} (query parameter)
Response: {"response": {"data": [...]}}
"""

import os
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import requests

logger = logging.getLogger(__name__)

EIA_BASE_URL = "https://api.eia.gov/v2"

# Rate limiting — conservative to stay well under ~9,000/hour
REQUEST_DELAY = 0.7  # 700ms between requests

# Cache duration — weekly data updates on specific days
CACHE_DURATION = 6 * 3600  # 6 hours

# --- Series definitions ---
# Each entry: route, facets, human name, signal_type, frequency
EIA_SERIES: Dict[str, dict] = {
    "crude_stocks": {
        "route": "/petroleum/stoc/wstk/data",
        "facets": {"duoarea": "NUS", "product": "EPC0", "process": "SAX"},
        "name": "US Commercial Crude Oil Stocks (Weekly)",
        "signal_type": "crude_inventory",
        "frequency": "weekly",
        "unit": "thousand_barrels",
    },
    "gasoline_stocks": {
        "route": "/petroleum/stoc/wstk/data",
        "facets": {"duoarea": "NUS", "product": "EPM0F", "process": "SAX"},
        "name": "US Motor Gasoline Stocks (Weekly)",
        "signal_type": "gasoline_inventory",
        "frequency": "weekly",
        "unit": "thousand_barrels",
    },
    "distillate_stocks": {
        "route": "/petroleum/stoc/wstk/data",
        "facets": {"duoarea": "NUS", "product": "EPD0", "process": "SAX"},
        "name": "US Distillate Fuel Oil Stocks (Weekly)",
        "signal_type": "distillate_inventory",
        "frequency": "weekly",
        "unit": "thousand_barrels",
    },
    "natgas_storage": {
        "route": "/natural-gas/stor/wkly/data",
        "facets": {"process": "SAX"},
        "name": "US Natural Gas Working Storage (Weekly)",
        "signal_type": "natgas_storage",
        "frequency": "weekly",
        "unit": "billion_cubic_feet",
    },
    "crude_production": {
        "route": "/petroleum/crd/crpdn/data",
        "facets": {"duoarea": "NUS"},
        "name": "US Crude Oil Production (Monthly)",
        "signal_type": "crude_production",
        "frequency": "monthly",
        "unit": "thousand_barrels_per_day",
    },
}

# Tickers most sensitive to each signal type
ENERGY_TICKER_MAP: Dict[str, List[str]] = {
    "crude_inventory": ["XLE", "XOP", "OIH", "USO", "XOM", "CVX", "COP", "EOG", "PXD", "DVN"],
    "gasoline_inventory": ["VLO", "MPC", "PSX", "XLE", "HFC"],
    "distillate_inventory": ["VLO", "MPC", "PSX", "XLE"],
    "natgas_storage": ["UNG", "EQT", "AR", "CHK", "RRC", "CTRA"],
    "crude_production": ["XLE", "XOP", "XOM", "CVX", "COP"],
}

# Typical weekly change ranges for direction/strength calibration
# Values in thousands of barrels (crude/products) or Bcf (natgas)
_TYPICAL_CHANGES: Dict[str, float] = {
    "crude_inventory": 5000,      # ~5M barrel weekly swing is typical
    "gasoline_inventory": 2000,   # ~2M barrel weekly swing
    "distillate_inventory": 2000, # ~2M barrel weekly swing
    "natgas_storage": 80,         # ~80 Bcf weekly swing
    "crude_production": 100,      # ~100K bbl/d monthly change
}


def _determine_direction(signal_type: str, change: float) -> str:
    """Classify an energy data change as bullish/bearish/neutral for energy stocks.

    Inventory BUILD (positive change) = bearish (supply > demand)
    Inventory DRAW (negative change) = bullish (demand > supply)
    Production INCREASE = bearish (more supply)
    Production DECREASE = bullish (less supply)
    """
    if abs(change) < 0.001:
        return "neutral"

    if signal_type in ("crude_inventory", "gasoline_inventory", "distillate_inventory"):
        # Inventory: build = bearish, draw = bullish
        return "bearish" if change > 0 else "bullish"
    elif signal_type == "natgas_storage":
        # Storage: injection (build) = bearish, withdrawal (draw) = bullish
        return "bearish" if change > 0 else "bullish"
    elif signal_type == "crude_production":
        # Production increase = more supply = bearish
        return "bearish" if change > 0 else "bullish"

    return "neutral"


def _compute_strength(signal_type: str, change: float) -> float:
    """Compute signal strength from magnitude of change.

    Normalizes against typical weekly/monthly change ranges.
    Returns 0.0-1.0 (clamped).
    """
    typical = _TYPICAL_CHANGES.get(signal_type, 5000)
    if typical == 0:
        return 0.5
    raw = abs(change) / typical
    return min(1.0, max(0.1, raw))


@dataclass
class EnergyIndicator:
    """A single energy data reading from EIA.

    Carries the same signal metadata fields as other MIDGE signals
    so the convergence alerter can weight energy data against other domains.

    Energy signals decay moderately — inventory data is relevant for
    ~2-4 weeks (the next report supersedes it).
    """
    series_key: str          # e.g. "crude_stocks"
    series_name: str         # e.g. "US Commercial Crude Oil Stocks (Weekly)"
    value: float             # Current level (absolute)
    prior_value: float       # Previous reading (for change calculation)
    change: float            # value - prior_value
    change_pct: float        # Percentage change
    date: str                # YYYY-MM-DD of the observation
    signal_type: str         # "crude_inventory", "natgas_storage", etc.
    direction: str           # "bullish", "bearish", "neutral"
    strength: float          # 0-1 signal strength
    affected_tickers: list   # Tickers most sensitive to this data
    signal_source: str = "eia_energy"
    decay_rate: float = 0.05    # ~14 day half-life (next report supersedes)
    confidence: float = 0.75    # Government data is reliable
    unit: str = ""


class EIAClient:
    """Client for the U.S. Energy Information Administration API v2.

    Fetches energy inventory, storage, and production data. Translates
    raw readings into EnergyIndicator objects with direction and strength
    signals that the convergence engine can cross-reference with other domains.

    Usage:
        client = EIAClient()                          # key from env
        snapshot = client.get_energy_snapshot()        # all key series
        crude = client.get_crude_stocks()              # just crude

    Rate limits: ~9,000/hour. Client enforces 0.7s delay.
    Cache: 6 hours — weekly data updates on specific days.
    """

    def __init__(self, api_key: Optional[str] = None, provider=None):
        self._provider = provider
        self.api_key = api_key or os.environ.get("EIA_API_KEY")

        if self.api_key:
            logger.info("EIAClient initialized with API key")
        else:
            logger.warning(
                "No EIA_API_KEY found. Set EIA_API_KEY env var or pass api_key=. "
                "Register free at https://www.eia.gov/opendata/register.php"
            )

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MIDGE Trading Research"})
        self._last_request_time: float = 0.0
        self._cache: Dict[str, tuple] = {}  # key -> (EnergyIndicator, timestamp)

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _request(self, route: str, params: Dict) -> Optional[dict]:
        """Make a GET request to EIA API v2."""
        if not self.api_key:
            logger.error("Cannot make EIA request: no API key configured")
            return None

        full_params = dict(params)
        full_params["api_key"] = self.api_key

        url = f"{EIA_BASE_URL}{route}"

        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus

            resp = market_request(
                self._provider, url,
                headers={"User-Agent": "MIDGE Trading Research"},
                params=full_params,
                source_name="eia_energy",
                timeout_ms=30000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                return resp.payload
            logger.warning("EIA request failed via provider: %s", resp.error_message)
            return None

        try:
            self._rate_limit()
            response = self.session.get(url, params=full_params, timeout=30)
            if response.status_code == 200:
                return response.json()
            logger.warning(
                "EIA HTTP %s for %s: %s",
                response.status_code, url, response.text[:200]
            )
            return None
        except Exception as exc:
            logger.error("EIA request exception for %s: %s", url, exc)
            return None

    def get_series(self, series_key: str) -> Optional[EnergyIndicator]:
        """Fetch the latest two observations for a series (to compute change).

        Returns an EnergyIndicator with the current value, prior value,
        and computed change/direction/strength.
        """
        if series_key not in EIA_SERIES:
            logger.warning("Unknown EIA series key: %s", series_key)
            return None

        # Check cache
        if series_key in self._cache:
            indicator, cached_at = self._cache[series_key]
            if time.time() - cached_at < CACHE_DURATION:
                return indicator

        series_def = EIA_SERIES[series_key]
        route = series_def["route"]

        # Build facet parameters for v2 API
        params: Dict[str, Any] = {
            "frequency": series_def["frequency"],
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 5,  # Get 5 observations to handle missing values
        }
        for facet_name, facet_value in series_def.get("facets", {}).items():
            params[f"facets[{facet_name}][]"] = facet_value

        data = self._request(route, params)
        if data is None:
            return None

        response_data = data.get("response", {}).get("data", [])
        if len(response_data) < 2:
            logger.warning("EIA returned fewer than 2 observations for %s", series_key)
            return None

        # Extract current and prior values
        try:
            current = response_data[0]
            prior = response_data[1]
            value = float(current.get("value", 0))
            prior_value = float(prior.get("value", 0))
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("EIA data parse error for %s: %s", series_key, e)
            return None

        change = value - prior_value
        change_pct = (change / prior_value * 100) if prior_value != 0 else 0.0

        signal_type = series_def["signal_type"]
        direction = _determine_direction(signal_type, change)
        strength = _compute_strength(signal_type, change)

        indicator = EnergyIndicator(
            series_key=series_key,
            series_name=series_def["name"],
            value=value,
            prior_value=prior_value,
            change=change,
            change_pct=round(change_pct, 2),
            date=current.get("period", ""),
            signal_type=signal_type,
            direction=direction,
            strength=round(strength, 3),
            affected_tickers=ENERGY_TICKER_MAP.get(signal_type, []),
            unit=series_def.get("unit", ""),
        )

        self._cache[series_key] = (indicator, time.time())
        logger.debug(
            "EIA %s = %.0f (change: %+.0f, %s) [%s]",
            series_key, value, change, direction, indicator.date,
        )
        return indicator

    def get_crude_stocks(self) -> Optional[EnergyIndicator]:
        """Convenience: fetch just crude oil inventory."""
        return self.get_series("crude_stocks")

    def get_natgas_storage(self) -> Optional[EnergyIndicator]:
        """Convenience: fetch just natural gas storage."""
        return self.get_series("natgas_storage")

    def get_energy_snapshot(self) -> List[EnergyIndicator]:
        """Fetch all key energy series. Returns list of EnergyIndicators."""
        results = []
        for series_key in EIA_SERIES:
            indicator = self.get_series(series_key)
            if indicator is not None:
                results.append(indicator)
            else:
                logger.warning("get_energy_snapshot: failed to fetch %s", series_key)
        return results

    def get_historical_series(
        self, series_key: str, days: int = 365
    ) -> List[EnergyIndicator]:
        """Fetch historical observations for a series.

        Returns one EnergyIndicator per observation pair (current/prior)
        within the lookback window. Used for excavation and backtesting.
        """
        if series_key not in EIA_SERIES:
            return []

        series_def = EIA_SERIES[series_key]
        route = series_def["route"]

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        params: Dict[str, Any] = {
            "frequency": series_def["frequency"],
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "start": start_date,
            "length": 5000,
        }
        for facet_name, facet_value in series_def.get("facets", {}).items():
            params[f"facets[{facet_name}][]"] = facet_value

        data = self._request(route, params)
        if data is None:
            return []

        response_data = data.get("response", {}).get("data", [])
        if len(response_data) < 2:
            return []

        signal_type = series_def["signal_type"]
        results: List[EnergyIndicator] = []

        for i in range(1, len(response_data)):
            try:
                current = response_data[i]
                prior = response_data[i - 1]
                value = float(current.get("value", 0))
                prior_value = float(prior.get("value", 0))
            except (ValueError, TypeError):
                continue

            change = value - prior_value
            change_pct = (change / prior_value * 100) if prior_value != 0 else 0.0
            direction = _determine_direction(signal_type, change)
            strength = _compute_strength(signal_type, change)

            results.append(EnergyIndicator(
                series_key=series_key,
                series_name=series_def["name"],
                value=value,
                prior_value=prior_value,
                change=change,
                change_pct=round(change_pct, 2),
                date=current.get("period", ""),
                signal_type=signal_type,
                direction=direction,
                strength=round(strength, 3),
                affected_tickers=ENERGY_TICKER_MAP.get(signal_type, []),
                unit=series_def.get("unit", ""),
            ))

        logger.info("EIA %s: %d observations over %d days", series_key, len(results), days)
        return results


def get_energy_snapshot() -> List[EnergyIndicator]:
    """Convenience function: get all key energy indicators."""
    client = EIAClient()
    return client.get_energy_snapshot()


if __name__ == "__main__":
    import sys

    print("EIA Energy Data Test")
    print("=" * 60)

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print("WARNING: EIA_API_KEY not set. Requests will fail.")
        print("Register free at: https://www.eia.gov/opendata/register.php")
        print()

    client = EIAClient()

    # Single series test
    series = sys.argv[1] if len(sys.argv) > 1 else "crude_stocks"
    print(f"\nFetching: {series}")
    indicator = client.get_series(series)
    if indicator:
        arrow = {"bullish": "+", "bearish": "-", "neutral": "~"}[indicator.direction]
        print(f"  {indicator.series_name}")
        print(f"  Value:     {indicator.value:,.0f} {indicator.unit}")
        print(f"  Prior:     {indicator.prior_value:,.0f}")
        print(f"  Change:    {indicator.change:+,.0f} ({indicator.change_pct:+.1f}%)")
        print(f"  Direction: [{arrow}] {indicator.direction}")
        print(f"  Strength:  {indicator.strength:.3f}")
        print(f"  Date:      {indicator.date}")
        print(f"  Tickers:   {', '.join(indicator.affected_tickers[:5])}")
    else:
        print(f"  Failed to fetch {series}")

    # Full snapshot
    print("\nEnergy Snapshot (all key series):")
    snapshot = client.get_energy_snapshot()
    if snapshot:
        for m in snapshot:
            arrow = {"bullish": "+", "bearish": "-", "neutral": "~"}[m.direction]
            print(f"  [{arrow}] {m.series_key:<20} {m.change:>+10,.0f}  ({m.date})  {m.direction}")
    else:
        print("  No indicators available")

    print("\nDone.")
