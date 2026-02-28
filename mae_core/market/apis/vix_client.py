"""
vix_client.py - CBOE VIX Term Structure Client

Fetches VIX spot level and term structure from CBOE's free data feeds.
Calculates contango/backwardation state of the VIX futures curve.

VIX term structure is one of the most reliable regime signals:
- Contango (normal): near-term VIX < far-term = complacency, risk-on
- Backwardation (fear): near-term VIX > far-term = active fear, risk-off
- Flat: transition state, uncertainty

No API key required. Data source: CBOE public data feeds.
"""

import csv
import io
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# CBOE public data URLs
VIX_CURRENT_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

REQUEST_DELAY = 2.0
CACHE_DURATION = 14400  # 4 hours — VIX updates daily


@dataclass
class VIXSignal:
    """VIX spot level and term structure data."""

    vix_spot: float             # Current VIX level
    vix_1m: Optional[float]     # 1-month VIX futures (if available)
    vix_3m: Optional[float]     # 3-month VIX futures (if available)
    term_spread: float          # vix_1m - vix_spot (positive = contango)
    structure_type: str         # "contango", "backwardation", or "flat"
    date: str                   # YYYY-MM-DD

    signal_source: str = "vix_term_structure"
    decay_rate: float = 0.30    # Medium decay — daily data
    confidence: float = 0.60    # VIX is well-studied, reliable regime signal

    def to_plain_language(self) -> str:
        return (
            f"VIX {self.vix_spot:.1f} | "
            f"Structure: {self.structure_type} "
            f"(spread {self.term_spread:+.2f}) | "
            f"Date: {self.date}"
        )


class VIXClient:
    """
    Client for CBOE VIX data.

    Fetches VIX spot from the daily price CSV and estimates term structure
    from recent price history. No API key required.

    Rate limited to 1 req/2s. 4-hour cache.
    """

    def __init__(self, provider=None):
        self._provider = provider
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MIDGE Trading Research"})
        self._last_request_time: float = 0.0
        self._cache: Optional[VIXSignal] = None
        self._cache_time: float = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _fetch_text(self, url: str) -> Optional[str]:
        """Fetch raw text content from a URL."""
        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus
            resp = market_request(
                self._provider, url,
                source_name="cboe_vix", timeout_ms=30000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                payload = resp.payload
                if isinstance(payload, dict) and "text" in payload:
                    return payload["text"]
                if isinstance(payload, str):
                    return payload
            return None

        try:
            r = self.session.get(url, timeout=30)
            if r.status_code == 200:
                return r.text
            logger.warning("VIX fetch HTTP %d", r.status_code)
            return None
        except Exception as exc:
            logger.warning("VIX fetch error: %s", exc)
            return None

    def get_vix_structure(self) -> Optional[VIXSignal]:
        """
        Get current VIX level and term structure assessment.

        Returns:
            VIXSignal with spot VIX and structure classification,
            or None if data unavailable.
        """
        if self._cache is not None and time.time() - self._cache_time < CACHE_DURATION:
            return self._cache

        self._rate_limit()
        text = self._fetch_text(VIX_CURRENT_URL)
        if not text:
            return None

        try:
            return self._parse_vix_csv(text)
        except Exception as exc:
            logger.warning("VIX parse error: %s", exc)
            return None

    def _parse_vix_csv(self, text: str) -> Optional[VIXSignal]:
        """Parse CBOE VIX history CSV and extract most recent data."""
        reader = csv.DictReader(io.StringIO(text))
        rows = []
        for row in reader:
            try:
                close = float(row.get("CLOSE", row.get("Close", 0)))
                date_str = row.get("DATE", row.get("Date", ""))
                if close > 0 and date_str:
                    rows.append({"date": date_str.strip(), "close": close})
            except (ValueError, TypeError):
                continue

        if not rows:
            return None

        # Sort by date descending to get most recent
        rows.sort(key=lambda r: r["date"], reverse=True)

        current = rows[0]
        vix_spot = current["close"]
        date = current["date"]

        # Estimate term structure from recent history
        # Use 20-day and 60-day averages as proxy for 1M and 3M VIX futures
        vix_1m = None
        vix_3m = None
        if len(rows) >= 20:
            vix_1m = sum(r["close"] for r in rows[:20]) / 20
        if len(rows) >= 60:
            vix_3m = sum(r["close"] for r in rows[:60]) / 60

        # Calculate term spread
        if vix_1m is not None:
            term_spread = vix_1m - vix_spot
        else:
            term_spread = 0.0

        # Classify structure
        if term_spread > 1.0:
            structure_type = "contango"
        elif term_spread < -1.0:
            structure_type = "backwardation"
        else:
            structure_type = "flat"

        result = VIXSignal(
            vix_spot=round(vix_spot, 2),
            vix_1m=round(vix_1m, 2) if vix_1m else None,
            vix_3m=round(vix_3m, 2) if vix_3m else None,
            term_spread=round(term_spread, 2),
            structure_type=structure_type,
            date=date[:10] if len(date) >= 10 else date,
        )

        self._cache = result
        self._cache_time = time.time()
        logger.debug("VIX: spot=%.1f, structure=%s, spread=%.2f",
                      vix_spot, structure_type, term_spread)
        return result


def get_vix_structure() -> Optional[VIXSignal]:
    """Convenience function for one-shot VIX data fetch."""
    return VIXClient().get_vix_structure()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Testing CBOE VIX Term Structure...")

    client = VIXClient()
    vix = client.get_vix_structure()

    if vix:
        print(f"  {vix.to_plain_language()}")
        print(f"  1M avg: {vix.vix_1m}, 3M avg: {vix.vix_3m}")
    else:
        print("  No data returned")

    print("\nDone.")
