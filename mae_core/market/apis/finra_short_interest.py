#!/usr/bin/env python3
"""
finra_short_interest.py - FINRA Daily Short Volume Client

Fetches daily short volume data from FINRA's publicly available RegSHO files.
No API key required.

Short volume data is a leading indicator for:
- Squeeze setups: very high short ratio + low float = potential explosive move
- Distribution: increasing short ratio on a rising stock = smart money hedging
- Conviction signals: persistent high short ratio = sustained bearish pressure

Data source: FINRA CDN (updated each trading day after market close)
  https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt

File format: pipe-delimited
  Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

Coverage: All FINRA-member broker-dealer short sales reported under RegSHO.
Note: This is SHORT VOLUME (trades that day that were short), not SHORT INTEREST
(total open short positions). Short volume is a daily flow metric; short interest
is a twice-monthly snapshot. Both measure bearish pressure, different timescales.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# FINRA CDN endpoint — no auth required, updated each trading day
FINRA_CDN_BASE = "https://cdn.finra.org/equity/regsho/daily"
FINRA_FILE_PATTERN = "CNMSshvol{date}.txt"  # date = YYYYMMDD

# Rate limiting — be respectful to a free public resource
REQUEST_DELAY = 1.0  # 1 request/second

# Cache configuration
CACHE_DURATION = 3600  # 1 hour — daily data won't change once published

# Ticker validation — standard exchange symbols are 1-5 uppercase letters/digits
# Skip symbols with spaces, slashes, dots with extra parts (preferred shares etc.),
# or other special chars that indicate non-standard instruments
_VALID_TICKER_RE = re.compile(r'^[A-Z]{1,5}$')


def _is_valid_ticker(symbol: str) -> bool:
    """
    Return True if symbol looks like a plain exchange ticker.

    Filters out things like 'AAPL.W' (warrants), 'BRK/B' (share class slash),
    'ACME UNIT', or numeric-only symbols that appear in the FINRA file.
    """
    return bool(_VALID_TICKER_RE.match(symbol))


@dataclass
class ShortInterestData:
    """
    Daily short volume record for a single ticker.

    short_ratio is the actionable signal: fraction of total volume that was short.
    A ratio above 0.50 means more than half of all trades that day were short sales,
    which is unusually high bearish flow.
    """
    symbol: str
    date: str                       # YYYY-MM-DD
    short_volume: int               # Shares sold short that day
    total_volume: int               # Total shares traded that day
    short_ratio: float              # short_volume / max(1, total_volume)

    # MIDGE signal metadata — consumed by ConvergenceAlerter / ThompsonSampler
    signal_source: str = "finra_short"
    decay_rate: float = 0.04        # ~17 day half-life (short vol is a flow, fades quickly)
    confidence: float = 0.55        # Moderate base confidence — combine with other signals


class FINRAShortInterestClient:
    """
    Client for FINRA daily short volume data (RegSHO).

    Pulls pipe-delimited text files from FINRA's public CDN. No API key needed.
    Each file covers a single trading day and lists every symbol's short volume.

    Typical usage:
        client = FINRAShortInterestClient()
        high_shorts = client.get_high_short_ratio(min_ratio=0.5)
        for row in high_shorts[:10]:
            print(f"{row.symbol}: {row.short_ratio:.1%} short ({row.short_volume:,} / {row.total_volume:,})")
    """

    def __init__(self, provider=None):
        """
        Initialize client.

        Args:
            provider: Optional MarketDataProvider for gateway routing.
                      If None, uses a direct requests.Session.
        """
        self._provider = provider
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MIDGE Trading Research"
        })
        self._last_request_time: float = 0.0

        # Per-date cache: date_str -> (List[ShortInterestData], fetched_at_timestamp)
        self._cache: Dict[str, Tuple[List[ShortInterestData], float]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Enforce 1 request/second rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _fetch_raw_text(self, url: str) -> Optional[str]:
        """
        GET a URL and return its text body, or None on failure.

        Routes through provider if configured, otherwise uses direct session.
        """
        self._rate_limit()

        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus

            resp = market_request(
                self._provider, url,
                headers={"User-Agent": "MIDGE Trading Research"},
                source_name="finra_short",
                timeout_ms=30000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                payload = resp.payload
                # Provider returns JSON; raw text is stored under "text" key
                if isinstance(payload, dict) and "text" in payload:
                    return payload["text"]
                if isinstance(payload, str):
                    return payload
            logger.warning("FINRA request via provider failed: %s", getattr(resp, "error_message", ""))
            return None

        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            logger.warning("FINRA HTTP %d for %s", response.status_code, url)
            return None
        except Exception as exc:
            logger.error("FINRA request failed for %s: %s", url, exc)
            return None

    @staticmethod
    def _date_to_finra_str(date_obj: datetime) -> str:
        """Convert datetime to YYYYMMDD string used in FINRA filenames."""
        return date_obj.strftime("%Y%m%d")

    @staticmethod
    def _finra_date_to_iso(finra_date: str) -> str:
        """
        Convert FINRA date column to ISO format.

        FINRA uses YYYYMMDD in the Date column.
        """
        try:
            return datetime.strptime(finra_date.strip(), "%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            return finra_date.strip()

    def _parse_file(self, text: str) -> List[ShortInterestData]:
        """
        Parse FINRA pipe-delimited short volume text.

        Expected format (first line is header):
          Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

        Returns only records with valid ticker symbols.
        """
        records: List[ShortInterestData] = []
        lines = text.strip().splitlines()

        if not lines:
            return records

        # Skip header line
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 5:
                continue

            date_raw, symbol, short_vol_raw, _exempt, total_vol_raw = parts[:5]

            # Only keep plain exchange tickers
            if not _is_valid_ticker(symbol):
                continue

            try:
                short_volume = int(short_vol_raw)
                total_volume = int(total_vol_raw)
            except ValueError:
                continue

            short_ratio = short_volume / max(1, total_volume)
            iso_date = FINRAShortInterestClient._finra_date_to_iso(date_raw)

            records.append(ShortInterestData(
                symbol=symbol,
                date=iso_date,
                short_volume=short_volume,
                total_volume=total_volume,
                short_ratio=short_ratio,
            ))

        return records

    def _load_date(self, date_obj: datetime) -> List[ShortInterestData]:
        """
        Fetch and cache short volume data for a single trading day.

        Checks 1-hour cache first. Returns empty list if the file is not yet
        available (i.e., market hasn't closed yet for that day).
        """
        date_str = self._date_to_finra_str(date_obj)
        cache_key = date_str

        # Return from cache if still fresh
        if cache_key in self._cache:
            records, fetched_at = self._cache[cache_key]
            if time.time() - fetched_at < CACHE_DURATION:
                logger.debug("Cache hit for FINRA date %s (%d records)", date_str, len(records))
                return records

        url = f"{FINRA_CDN_BASE}/{FINRA_FILE_PATTERN.format(date=date_str)}"
        logger.debug("Fetching FINRA short volume: %s", url)

        text = self._fetch_raw_text(url)
        if text is None:
            logger.info("FINRA file not available for %s (market may not have closed yet)", date_str)
            return []

        records = self._parse_file(text)
        self._cache[cache_key] = (records, time.time())
        logger.info("Loaded %d FINRA short volume records for %s", len(records), date_str)
        return records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_daily_short_volume(self, date: Optional[str] = None) -> List[ShortInterestData]:
        """
        Fetch short volume for a trading day.

        If today's file is not yet available (published after market close ~6pm ET),
        automatically falls back to yesterday's file.

        Args:
            date: Date in YYYY-MM-DD format. Defaults to today.

        Returns:
            List of ShortInterestData, one per valid ticker. Empty on failure.
        """
        if date is not None:
            try:
                target = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                logger.error("Invalid date format '%s' — expected YYYY-MM-DD", date)
                return []
            return self._load_date(target)

        # Default: try today, fall back to yesterday if not yet available
        today = datetime.now()
        records = self._load_date(today)

        if not records:
            yesterday = today - timedelta(days=1)
            logger.info("Today's FINRA file not available, trying yesterday (%s)",
                        yesterday.strftime("%Y-%m-%d"))
            records = self._load_date(yesterday)

        return records

    def get_high_short_ratio(
        self,
        min_ratio: float = 0.5,
        date: Optional[str] = None,
    ) -> List[ShortInterestData]:
        """
        Return tickers where short volume exceeds min_ratio of total volume.

        A short ratio above 0.50 means more than half of all recorded volume
        that day was short — a statistically unusual bearish signal worth
        flagging for convergence analysis.

        Args:
            min_ratio: Minimum short_ratio threshold (default 0.5 = 50%).
            date: Date in YYYY-MM-DD format. Defaults to most recent available.

        Returns:
            Filtered list sorted by short_ratio descending (highest first).
        """
        all_records = self.get_daily_short_volume(date)
        filtered = [r for r in all_records if r.short_ratio >= min_ratio]
        filtered.sort(key=lambda r: r.short_ratio, reverse=True)
        logger.debug(
            "FINRA high-short filter (>= %.0f%%): %d / %d tickers",
            min_ratio * 100, len(filtered), len(all_records),
        )
        return filtered

    def get_ticker_short_history(
        self,
        ticker: str,
        days: int = 5,
    ) -> List[ShortInterestData]:
        """
        Get short volume for a specific ticker over multiple recent trading days.

        Walks backwards day-by-day, skipping weekends, until it has collected
        `days` worth of data or exhausted a reasonable search window.

        Args:
            ticker: Uppercase ticker symbol (e.g. "NVDA").
            days: Number of trading days to retrieve (default 5 = one week).

        Returns:
            List of ShortInterestData for the ticker, sorted oldest-first.
            Days with no data (holidays, no trading) are omitted silently.
        """
        ticker = ticker.upper().strip()
        results: List[ShortInterestData] = []

        # Search up to 3x days to account for weekends and holidays
        search_limit = days * 3
        current = datetime.now()
        days_searched = 0

        while len(results) < days and days_searched < search_limit:
            # Skip weekends (Saturday=5, Sunday=6)
            if current.weekday() >= 5:
                current -= timedelta(days=1)
                days_searched += 1
                continue

            day_records = self._load_date(current)

            # Find this ticker in the day's data
            match = next((r for r in day_records if r.symbol == ticker), None)
            if match:
                results.append(match)
            elif day_records:
                # File loaded fine, ticker just wasn't traded that day — still counts
                pass
            # If day_records is empty the file may not exist (future date or holiday)

            current -= timedelta(days=1)
            days_searched += 1

        # Return in chronological order (oldest first)
        results.sort(key=lambda r: r.date)
        return results


# ---------------------------------------------------------------------------
# Module-level convenience functions (mirror pattern from other clients)
# ---------------------------------------------------------------------------

def get_daily_short_volume(date: Optional[str] = None) -> List[ShortInterestData]:
    """Convenience: fetch today's (or specified date's) FINRA short volume."""
    client = FINRAShortInterestClient()
    return client.get_daily_short_volume(date)


def get_high_short_ratio(min_ratio: float = 0.5, date: Optional[str] = None) -> List[ShortInterestData]:
    """Convenience: return tickers with short ratio above threshold."""
    client = FINRAShortInterestClient()
    return client.get_high_short_ratio(min_ratio=min_ratio, date=date)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Testing FINRA Short Interest Client...")
    print()

    client = FINRAShortInterestClient()

    # Allow optional date argument: python finra_short_interest.py 2026-02-21
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    date_label = target_date or "most recent available"

    print(f"Fetching daily short volume for: {date_label}")
    all_records = client.get_daily_short_volume(target_date)

    if not all_records:
        print("No data returned. File may not be available yet (try a past trading date).")
        sys.exit(0)

    print(f"Loaded {len(all_records):,} ticker records.")

    # Show high short ratio tickers
    high = client.get_high_short_ratio(min_ratio=0.5, date=target_date)
    print(f"\nHigh short ratio (>= 50%): {len(high)} tickers")
    for row in high[:15]:
        print(
            f"  {row.symbol:<6}  ratio={row.short_ratio:.1%}  "
            f"short={row.short_volume:>10,}  total={row.total_volume:>10,}  "
            f"date={row.date}"
        )

    # Show ticker history if a second arg provided
    if len(sys.argv) > 2:
        ticker = sys.argv[2].upper()
        print(f"\nShort history for {ticker} (last 5 trading days):")
        history = client.get_ticker_short_history(ticker, days=5)
        if history:
            for row in history:
                print(
                    f"  {row.date}  ratio={row.short_ratio:.1%}  "
                    f"short={row.short_volume:>10,}  total={row.total_volume:>10,}"
                )
        else:
            print(f"  No data found for {ticker}")

    print("\nDone.")
