#!/usr/bin/env python3
"""
fred_client.py - Federal Reserve Economic Data (FRED) Client

Fetches macroeconomic indicators from the St. Louis Fed FRED API.
Free tier: 120 calls/minute. Requires FRED_API_KEY environment variable.

Macro indicators are slow-moving signals — they context-set the market
regime rather than trigger immediate trades. A rising VIX suppresses
confidence in all other signals. An inverted yield curve shifts the
interpretation of insider buying patterns. These are the background
conditions Mae uses to weight everything else.

API base: https://api.stlouisfed.org/fred
Auth: &api_key={KEY}&file_type=json (query parameters)
Key endpoint: /series/observations?series_id={ID}&sort_order=desc&limit={N}
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import requests

logger = logging.getLogger(__name__)


# API base
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# Rate limiting — 120 calls/min free tier
REQUEST_DELAY = 0.5  # 500ms between requests

# Cache duration — macro data updates at most daily
CACHE_DURATION = 4 * 3600  # 4 hours in seconds

# Series definitions: id -> (human_readable_name, signal_type)
FRED_SERIES: Dict[str, tuple] = {
    "T10Y2Y":      ("10Y-2Y Treasury Spread (Yield Curve)", "yield_curve"),
    "BAMLH0A0HYM2": ("ICE BofA High Yield Spread", "credit_spread"),
    "VIXCLS":      ("VIX Volatility Index", "volatility"),
    "DFF":         ("Federal Funds Rate (Effective)", "rates"),
    "UNRATE":      ("Unemployment Rate", "employment"),
    "CPIAUCSL":    ("Consumer Price Index (All Urban)", "inflation"),
}


def _determine_direction(series_id: str, value: float) -> str:
    """
    Classify a macro value as bullish, bearish, or neutral.

    Rules are based on historically significant thresholds — not precise
    predictions, but reliable regime signals.

    Args:
        series_id: FRED series identifier
        value: Numeric value of the observation

    Returns:
        "bullish", "bearish", or "neutral"
    """
    if series_id == "T10Y2Y":
        # Yield curve inversion (negative spread) = recession signal = bearish
        # Healthy spread above 0.5 = bullish
        if value < 0:
            return "bearish"
        elif value > 0.5:
            return "bullish"
        return "neutral"

    elif series_id == "BAMLH0A0HYM2":
        # Credit spread: high = credit stress = fear = bearish
        # Low spread = risk-on, credit healthy = bullish
        if value > 5.0:
            return "bearish"
        elif value < 3.0:
            return "bullish"
        return "neutral"

    elif series_id == "VIXCLS":
        # VIX: high = fear = bearish. Low = complacency/calm = bullish
        if value > 30:
            return "bearish"
        elif value < 15:
            return "bullish"
        return "neutral"

    elif series_id == "DFF":
        # Rates: just reporting — direction is context-dependent
        # High rates = tighter conditions but not directly bullish/bearish
        return "neutral"

    elif series_id == "UNRATE":
        # Unemployment: high = economic weakness = bearish
        # Low = tight labor market = bullish
        if value > 5.0:
            return "bearish"
        elif value < 4.0:
            return "bullish"
        return "neutral"

    elif series_id == "CPIAUCSL":
        # CPI level: raw level is not directly directional
        # Month-over-month change would be more useful, but keep simple
        return "neutral"

    return "neutral"


@dataclass
class MacroIndicator:
    """
    A single macroeconomic indicator reading from FRED.

    Carries the same signal metadata fields as other MIDGE signals
    (signal_source, decay_rate, confidence) so the convergence alerter
    can weight macro context against micro signals uniformly.

    Macro signals decay slowly (~70 day half-life via decay_rate=0.01)
    because economic regimes persist for months, not days.
    """
    series_id: str          # e.g. "T10Y2Y"
    series_name: str        # e.g. "10Y-2Y Treasury Spread (Yield Curve)"
    value: float            # Numeric observation value
    date: str               # YYYY-MM-DD of the observation
    signal_type: str        # "yield_curve", "credit_spread", "volatility", "rates", "employment", "inflation"
    direction: str          # "bullish", "bearish", or "neutral"
    signal_source: str = "fred_macro"
    decay_rate: float = 0.01    # Slow decay — macro regimes persist ~70 days
    confidence: float = 0.70    # Government data is reliable but lags reality


class FREDClient:
    """
    Client for the St. Louis Fed FRED API.

    Fetches macroeconomic indicators and translates them into MacroIndicator
    objects that carry MIDGE signal metadata for convergence weighting.

    Usage:
        client = FREDClient()                         # key from env
        snapshot = client.get_macro_snapshot()        # all key series
        yield_curve = client.get_yield_curve_status() # just T10Y2Y

    Rate limits: 120 calls/min (free tier). Client enforces 0.5s delay.
    Cache: 4 hours — macro data updates at most daily.
    """

    def __init__(self, api_key: Optional[str] = None, provider=None, raw_store=None):
        """
        Initialize FRED client.

        Args:
            api_key: FRED API key. Falls back to FRED_API_KEY env var.
            provider: Optional MarketDataProvider for ApiGateway routing.
            raw_store: Optional RawStore for persisting all observations.
        """
        self._provider = provider
        self._raw_store = raw_store
        self.api_key = api_key or os.environ.get("FRED_API_KEY")

        if self.api_key:
            logger.info("FREDClient initialized with API key")
        else:
            logger.warning(
                "No FRED_API_KEY found. Set FRED_API_KEY env var or pass api_key=. "
                "Register free at https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MIDGE Trading Research"})

        self._last_request_time: float = 0.0

        # Per-series cache: series_id -> (MacroIndicator, cache_timestamp)
        self._cache: Dict[str, tuple] = {}

    def _rate_limit(self) -> None:
        """Enforce minimum 0.5s between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _request(self, endpoint: str, params: Dict) -> Optional[dict]:
        """
        Make a GET request to FRED API through provider or direct session.

        Appends api_key and file_type=json to all requests.

        Args:
            endpoint: Path after base URL (e.g. "/series/observations")
            params: Query parameters (without api_key/file_type)

        Returns:
            Parsed JSON dict or None on failure
        """
        if not self.api_key:
            logger.error("Cannot make FRED request: no API key configured")
            return None

        full_params = dict(params)
        full_params["api_key"] = self.api_key
        full_params["file_type"] = "json"

        url = f"{FRED_BASE_URL}{endpoint}"

        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus

            resp = market_request(
                self._provider, url,
                headers={"User-Agent": "MIDGE Trading Research"},
                params=full_params,
                source_name="fred_macro",
                timeout_ms=30000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                return resp.payload
            logger.warning("FRED request failed via provider: %s", resp.error_message)
            return None

        try:
            response = self.session.get(url, params=full_params, timeout=30)
            if response.status_code == 200:
                return response.json()
            logger.warning(
                "FRED HTTP %s for %s: %s",
                response.status_code, url, response.text[:200]
            )
            return None
        except Exception as exc:
            logger.error("FRED request exception for %s: %s", url, exc)
            return None

    def get_series(
        self, series_id: str, limit: int = 1
    ) -> Optional[MacroIndicator]:
        """
        Get the latest observation(s) for a FRED series.

        Uses a 4-hour cache per series to avoid hammering the API on
        repeated calls within a session (e.g. get_macro_snapshot calling
        multiple series in sequence).

        Args:
            series_id: FRED series ID (e.g. "T10Y2Y", "VIXCLS")
            limit: Number of observations to fetch (1 = latest only)

        Returns:
            MacroIndicator for the most recent observation, or None on failure
        """
        # Check cache
        if series_id in self._cache:
            indicator, cached_at = self._cache[series_id]
            if time.time() - cached_at < CACHE_DURATION:
                logger.debug("FRED cache hit for %s", series_id)
                return indicator

        self._rate_limit()

        data = self._request(
            "/series/observations",
            params={
                "series_id": series_id,
                "sort_order": "desc",
                "limit": limit,
            },
        )

        if data is None:
            return None

        observations = data.get("observations", [])
        if not observations:
            logger.warning("FRED returned no observations for %s", series_id)
            return None

        # Store ALL observations before extracting latest
        if self._raw_store:
            try:
                self._raw_store.store_fred_observations(series_id, observations)
            except Exception:
                pass

        # Most recent observation is first (sort_order=desc)
        obs = observations[0]
        raw_value = obs.get("value", "")

        # FRED uses "." for missing values
        if raw_value == "." or raw_value == "":
            # Try the next observation if available
            for fallback_obs in observations[1:]:
                fallback_val = fallback_obs.get("value", "")
                if fallback_val not in (".", ""):
                    raw_value = fallback_val
                    obs = fallback_obs
                    break
            else:
                logger.warning("FRED has no valid value for %s (all missing)", series_id)
                return None

        try:
            value = float(raw_value)
        except ValueError:
            logger.warning("FRED non-numeric value for %s: %r", series_id, raw_value)
            return None

        series_name, signal_type = FRED_SERIES.get(
            series_id, (series_id, "macro")
        )
        direction = _determine_direction(series_id, value)

        indicator = MacroIndicator(
            series_id=series_id,
            series_name=series_name,
            value=value,
            date=obs.get("date", ""),
            signal_type=signal_type,
            direction=direction,
        )

        self._cache[series_id] = (indicator, time.time())
        logger.debug(
            "FRED %s = %.4f on %s [%s]",
            series_id, value, indicator.date, direction
        )
        return indicator

    def get_historical_series(
        self, series_id: str, days: int = 90
    ) -> List[MacroIndicator]:
        """
        Fetch multiple observations for a FRED series over a date range.

        Unlike get_series() which returns only the latest observation, this
        returns one MacroIndicator per valid observation within the lookback
        window. Used by the backfill script to populate signal archives.

        Args:
            series_id: FRED series ID (e.g. "T10Y2Y", "VIXCLS")
            days: Number of calendar days to look back

        Returns:
            List of MacroIndicator objects, sorted oldest-first.
            Empty list on failure or if no API key is configured.
        """
        from datetime import datetime, timedelta

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        self._rate_limit()

        data = self._request(
            "/series/observations",
            params={
                "series_id": series_id,
                "sort_order": "asc",
                "observation_start": start_date,
            },
        )

        if data is None:
            return []

        observations = data.get("observations", [])
        if not observations:
            logger.warning("FRED returned no observations for %s (days=%d)", series_id, days)
            return []

        # Store ALL observations
        if self._raw_store:
            try:
                self._raw_store.store_fred_observations(series_id, observations)
            except Exception:
                pass

        series_name, signal_type = FRED_SERIES.get(series_id, (series_id, "macro"))
        results: List[MacroIndicator] = []

        for obs in observations:
            raw_value = obs.get("value", "")
            if raw_value in (".", ""):
                continue

            try:
                value = float(raw_value)
            except ValueError:
                continue

            direction = _determine_direction(series_id, value)
            results.append(MacroIndicator(
                series_id=series_id,
                series_name=series_name,
                value=value,
                date=obs.get("date", ""),
                signal_type=signal_type,
                direction=direction,
            ))

        logger.info("FRED %s: %d observations over %d days", series_id, len(results), days)
        return results

    def get_macro_snapshot(self) -> List[MacroIndicator]:
        """
        Fetch the latest reading for all key macro series.

        Returns all five primary trading-relevant series:
        T10Y2Y, BAMLH0A0HYM2, VIXCLS, DFF, UNRATE.
        (CPIAUCSL omitted from snapshot — raw level is not actionable
        without month-over-month delta context.)

        Returns:
            List of MacroIndicator objects, one per available series.
            Failures are skipped silently (logged at WARNING level).
        """
        snapshot_series = ["T10Y2Y", "BAMLH0A0HYM2", "VIXCLS", "DFF", "UNRATE"]
        results = []

        for series_id in snapshot_series:
            indicator = self.get_series(series_id)
            if indicator is not None:
                results.append(indicator)
            else:
                logger.warning("get_macro_snapshot: failed to fetch %s", series_id)

        return results

    def get_yield_curve_status(self) -> Optional[MacroIndicator]:
        """
        Convenience method: fetch just the yield curve spread (T10Y2Y).

        The yield curve is the most watched recession indicator. Negative
        values signal an inverted curve — historically predictive of
        recessions within 12-18 months.

        Returns:
            MacroIndicator for T10Y2Y, or None on failure
        """
        return self.get_series("T10Y2Y")


def get_macro_snapshot() -> List[MacroIndicator]:
    """
    Convenience function: get all key macro indicators.

    Returns:
        List of MacroIndicator objects
    """
    client = FREDClient()
    return client.get_macro_snapshot()


if __name__ == "__main__":
    import sys

    print("FRED Macro Indicator Test")
    print("=" * 50)

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("WARNING: FRED_API_KEY not set. Requests will fail.")
        print("Register free at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print()

    client = FREDClient()

    # Single series test
    series_id = sys.argv[1] if len(sys.argv) > 1 else "T10Y2Y"
    print(f"\nFetching single series: {series_id}")
    indicator = client.get_series(series_id)
    if indicator:
        print(f"  {indicator.series_name}")
        print(f"  Value:     {indicator.value:.4f}")
        print(f"  Date:      {indicator.date}")
        print(f"  Direction: {indicator.direction}")
        print(f"  Type:      {indicator.signal_type}")
    else:
        print(f"  Failed to fetch {series_id}")

    # Yield curve convenience method
    print("\nYield Curve Status (T10Y2Y):")
    yc = client.get_yield_curve_status()
    if yc:
        status = "INVERTED (recession signal)" if yc.direction == "bearish" else yc.direction.upper()
        print(f"  Spread: {yc.value:+.2f}% — {status}")
    else:
        print("  Could not fetch yield curve data")

    # Full macro snapshot
    print("\nMacro Snapshot (all key series):")
    snapshot = client.get_macro_snapshot()
    if snapshot:
        for m in snapshot:
            arrow = {"bullish": "+", "bearish": "-", "neutral": "~"}[m.direction]
            print(f"  [{arrow}] {m.series_id:<15} {m.value:>10.4f}  ({m.date})  {m.direction}")
    else:
        print("  No indicators available")

    print("\nDone.")
