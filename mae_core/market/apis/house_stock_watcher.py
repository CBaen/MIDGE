#!/usr/bin/env python3
"""
house_stock_watcher.py - Congressional Stock Trade Tracker

Fetches stock trades made by US House Representatives from housestockwatcher.com.
This is FREE data based on STOCK Act disclosures.

Congressional trades can be early indicators:
- Committee members may trade before legislation
- Representatives may have industry knowledge
- 45-day disclosure delay means some trades are already "old news"

Data source: https://housestockwatcher.com/api
Updated daily as new disclosures are filed.
"""

import os
import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)


# API endpoints (in priority order)
# Primary: RapidAPI US Congress Insider Trading (requires RAPIDAPI_KEY env var)
RAPIDAPI_HOST = "us-congress-insider-trading-data.p.rapidapi.com"
RAPIDAPI_BASE = f"https://{RAPIDAPI_HOST}"

# Fallback: Free sources (may be unreliable)
# S3 bucket where data was stored
HOUSE_STOCK_API = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
# GitHub raw data (community mirror)
HOUSE_STOCK_API_BACKUP = "https://raw.githubusercontent.com/timothycarambat/house-stock-watcher-data/master/data/all_transactions.json"
# Original site API
HOUSE_STOCK_API_TERTIARY = "https://housestockwatcher.com/api"

# Rate limiting
REQUEST_DELAY = 1.0  # Be respectful to free service


@dataclass
class CongressionalTrade:
    """Stock trade by a US House Representative."""
    # Politician info
    representative: str  # Full name
    district: str  # Office/district
    party: str  # R, D, I (if available)

    # Trade details
    ticker: str
    asset_description: str
    transaction_type: str  # "purchase", "sale", "sale_partial", "sale_full", "exchange"
    transaction_date: str
    amount_range: str  # "$1,001 - $15,000" format
    amount_low: float  # Parsed lower bound
    amount_high: float  # Parsed upper bound

    # Filing info
    disclosure_date: str
    disclosure_url: str

    # Ownership
    owner: str  # "Self", "Spouse", "Joint", "Child"

    # Signal metadata for MIDGE
    signal_source: str = "congressional"
    decay_rate: float = 0.03  # ~23 day half-life (slow - political info persists)
    confidence: float = 0.65  # Base confidence

    def to_plain_language(self) -> str:
        """Format for dashboard."""
        action = "bought" if "purchase" in self.transaction_type.lower() else "sold"
        return (
            f"{self.representative} ({self.party}) {action} "
            f"{self.amount_range} of {self.ticker or self.asset_description[:30]} "
            f"on {self.transaction_date}"
        )

    @staticmethod
    def parse_amount_range(amount_str: str) -> tuple:
        """
        Parse amount range string into (low, high) floats.

        Examples:
            "$1,001 - $15,000" -> (1001, 15000)
            "$50,001 - $100,000" -> (50001, 100000)
            "$1,000,001 - $5,000,000" -> (1000001, 5000000)
        """
        if not amount_str or amount_str == "--":
            return (0, 0)

        try:
            # Remove $ and commas, split on " - "
            cleaned = amount_str.replace("$", "").replace(",", "")
            parts = cleaned.split(" - ")

            if len(parts) == 2:
                return (float(parts[0]), float(parts[1]))
            elif len(parts) == 1:
                val = float(parts[0])
                return (val, val)
            else:
                return (0, 0)
        except Exception:
            return (0, 0)


class HouseStockWatcherClient:
    """
    Client for Congressional Stock Trade data.

    Priority:
    1. RapidAPI US Congress Insider Trading (requires RAPIDAPI_KEY env var)
    2. Fallback to free House Stock Watcher sources

    Data is based on STOCK Act disclosures.
    """

    def __init__(self, rapidapi_key: Optional[str] = None, provider=None):
        self._provider = provider
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MIDGE Trading Research"
        })
        self._last_request_time = 0
        self._cache = None
        self._cache_time = None
        self._cache_duration = 3600  # 1 hour cache

        # RapidAPI key from param or env
        self.rapidapi_key = rapidapi_key or os.environ.get("RAPIDAPI_KEY")
        if self.rapidapi_key:
            logger.info("RapidAPI key configured - using US Congress Insider Trading API")
        else:
            logger.info("No RAPIDAPI_KEY - falling back to free sources (may be unreliable)")

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _request(self, url: str, headers: Optional[Dict] = None,
                 params: Optional[Dict] = None, source_name: str = "rapidapi") -> Optional[dict]:
        """Make a GET request through provider or session. Returns parsed JSON or None."""
        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus
            all_headers = {"User-Agent": "MIDGE Trading Research"}
            if headers:
                all_headers.update(headers)
            resp = market_request(
                self._provider, url, headers=all_headers,
                params=params, source_name=source_name, timeout_ms=30000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                return resp.payload
            return None

        try:
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    def _get_trades_from_rapidapi(self, limit: int = 100, offset: int = 0) -> List[dict]:
        """
        Fetch trades from RapidAPI US Congress Insider Trading.

        Returns list of trades or empty list on failure.
        """
        if not self.rapidapi_key:
            return []

        self._rate_limit()

        data = self._request(
            f"{RAPIDAPI_BASE}/trades",
            headers={"x-rapidapi-host": RAPIDAPI_HOST, "x-rapidapi-key": self.rapidapi_key},
            params={"limit": limit, "offset": offset},
        )
        if data is not None:
            trades = data if isinstance(data, list) else data.get("trades", data.get("data", []))
            logger.debug(f"Loaded {len(trades)} trades from RapidAPI Congress Trading")
            return trades
        return []

    def _get_all_trades(self, use_cache: bool = True) -> List[dict]:
        """
        Fetch all trades from API.

        Uses caching to avoid repeated requests.
        Priority: RapidAPI -> Free sources
        """
        # Check cache
        if use_cache and self._cache and self._cache_time:
            if time.time() - self._cache_time < self._cache_duration:
                return self._cache

        # Try RapidAPI first
        if self.rapidapi_key:
            trades = self._get_trades_from_rapidapi(limit=100)  # API caps at 100
            if trades:
                self._cache = trades
                self._cache_time = time.time()
                return trades

        # Fallback to free sources
        self._rate_limit()

        endpoints = [
            (HOUSE_STOCK_API, "S3 bucket"),
            (HOUSE_STOCK_API_BACKUP, "GitHub mirror"),
            (HOUSE_STOCK_API_TERTIARY, "housestockwatcher.com"),
        ]

        for url, name in endpoints:
            data = self._request(url, source_name="congressional_free")
            if data is not None:
                self._cache = data
                self._cache_time = time.time()
                logger.debug(f"Loaded {len(data)} trades from {name}")
                return data
            logger.warning(f"{name} failed, trying next...")

        logger.warning("All congressional trade endpoints failed")
        return []

    def _normalize_trade(self, item: dict) -> Optional[dict]:
        """
        Normalize trade data from different API formats.

        Handles both RapidAPI and House Stock Watcher formats.
        """
        # Detect format by checking for RapidAPI-specific fields
        if "pubDate" in item or "txDate" in item:
            # RapidAPI format
            pub_date = item.get("pubDate", "")
            if pub_date and "T" in pub_date:
                pub_date = pub_date.split("T")[0]  # Extract date part

            # Parse ticker from "HUBS:US" format
            ticker_raw = item.get("issuer_issuerTicker", "")
            ticker = ticker_raw.split(":")[0] if ticker_raw else ""

            # Build representative name
            first = item.get("politician_firstName", "")
            last = item.get("politician_lastName", "")
            rep_name = f"{first} {last}".strip() or "Unknown"

            # Map party
            party_raw = item.get("politician_party", "").lower()
            party = "D" if "democrat" in party_raw else "R" if "republican" in party_raw else ""

            # Get value
            value = item.get("value", 0) or 0

            return {
                "disclosure_date": pub_date,
                "transaction_date": item.get("txDate", ""),
                "representative": rep_name,
                "party": party,
                "district": f"{item.get('politician_stateId', '').upper()} ({item.get('chamber', '')})",
                "ticker": ticker,
                "asset_description": item.get("issuer_issuerName", ""),
                "transaction_type": item.get("txType", "unknown"),
                "amount_low": value,
                "amount_high": value,
                "amount_str": f"${value:,.0f}" if value else "$0",
                "owner": item.get("ownership", "Self"),
                "url": ""
            }
        else:
            # House Stock Watcher format (original)
            return {
                "disclosure_date": item.get("disclosure_date", ""),
                "transaction_date": item.get("transaction_date", ""),
                "representative": item.get("representative", "Unknown"),
                "party": item.get("party", ""),
                "district": item.get("district", ""),
                "ticker": item.get("ticker", ""),
                "asset_description": item.get("asset_description", ""),
                "transaction_type": item.get("type", "unknown"),
                "amount_low": 0,
                "amount_high": 0,
                "amount_str": item.get("amount", "$0 - $0"),
                "owner": item.get("owner", "Self"),
                "url": item.get("ptr_link", "")
            }

    def get_recent_trades(self, days: int = 30) -> List[CongressionalTrade]:
        """
        Get trades from the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of CongressionalTrade objects
        """
        all_trades = self._get_all_trades()

        if not all_trades:
            return []

        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        trades = []

        for item in all_trades:
            # Normalize the data format
            normalized = self._normalize_trade(item)
            if not normalized:
                continue

            # Filter by disclosure date
            disc_date = normalized["disclosure_date"]
            if not disc_date or disc_date < cutoff:
                continue

            # Parse amount if string format
            amount_str = normalized["amount_str"]
            if normalized["amount_low"] == 0 and "-" in amount_str:
                amount_low, amount_high = CongressionalTrade.parse_amount_range(amount_str)
            else:
                amount_low = normalized["amount_low"]
                amount_high = normalized["amount_high"]

            # Clean ticker
            ticker = normalized["ticker"]
            if ticker == "--" or not ticker:
                ticker = ""

            trade = CongressionalTrade(
                representative=normalized["representative"],
                district=normalized["district"],
                party=normalized["party"],
                ticker=ticker,
                asset_description=normalized["asset_description"],
                transaction_type=normalized["transaction_type"].lower(),
                transaction_date=normalized["transaction_date"],
                amount_range=amount_str,
                amount_low=amount_low,
                amount_high=amount_high,
                disclosure_date=disc_date,
                disclosure_url=normalized["url"],
                owner=normalized["owner"]
            )
            trades.append(trade)

        # Sort by disclosure date (newest first)
        trades.sort(key=lambda t: t.disclosure_date, reverse=True)

        return trades

    def search_by_ticker(self, ticker: str, days: int = 365) -> List[CongressionalTrade]:
        """
        Find all trades of a specific ticker.

        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back

        Returns:
            List of matching trades
        """
        all_trades = self.get_recent_trades(days)
        ticker_upper = ticker.upper()
        return [t for t in all_trades if t.ticker.upper() == ticker_upper]

    def search_by_representative(
        self,
        name: str,
        days: int = 365
    ) -> List[CongressionalTrade]:
        """
        Find all trades by a representative.

        Args:
            name: Partial or full name of representative
            days: Number of days to look back

        Returns:
            List of matching trades
        """
        all_trades = self.get_recent_trades(days)
        name_lower = name.lower()
        return [t for t in all_trades if name_lower in t.representative.lower()]

    def get_large_trades(
        self,
        min_amount: float = 50000,
        days: int = 30
    ) -> List[CongressionalTrade]:
        """
        Find trades above a minimum dollar amount.

        Args:
            min_amount: Minimum lower bound of amount range
            days: Number of days to look back

        Returns:
            List of large trades
        """
        all_trades = self.get_recent_trades(days)
        return [t for t in all_trades if t.amount_low >= min_amount]

    def get_purchases(self, days: int = 30) -> List[CongressionalTrade]:
        """Get only purchase transactions."""
        all_trades = self.get_recent_trades(days)
        return [t for t in all_trades if "purchase" in t.transaction_type.lower()]

    def get_sales(self, days: int = 30) -> List[CongressionalTrade]:
        """Get only sale transactions."""
        all_trades = self.get_recent_trades(days)
        return [t for t in all_trades if "sale" in t.transaction_type.lower()]

    def get_top_politicians(self, limit: int = 10) -> List[Dict]:
        """
        Get top politicians by trading activity.

        Requires RapidAPI key. Returns empty list if not configured.

        Returns:
            List of dicts with politician name, party, trade counts
        """
        if not self.rapidapi_key:
            logger.warning("get_top_politicians requires RAPIDAPI_KEY")
            return []

        self._rate_limit()

        data = self._request(
            f"{RAPIDAPI_BASE}/stats/top-politicians",
            headers={"x-rapidapi-host": RAPIDAPI_HOST, "x-rapidapi-key": self.rapidapi_key},
            params={"limit": limit},
        )
        if data is not None:
            return data if isinstance(data, list) else data.get("data", [])
        return []

    def get_top_issuers(self, limit: int = 10) -> List[Dict]:
        """
        Get top stocks traded by congress.

        Requires RapidAPI key. Returns empty list if not configured.

        Returns:
            List of dicts with ticker, company name, trade counts
        """
        if not self.rapidapi_key:
            logger.warning("get_top_issuers requires RAPIDAPI_KEY")
            return []

        self._rate_limit()

        data = self._request(
            f"{RAPIDAPI_BASE}/stats/top-issuers",
            headers={"x-rapidapi-host": RAPIDAPI_HOST, "x-rapidapi-key": self.rapidapi_key},
            params={"limit": limit},
        )
        if data is not None:
            return data if isinstance(data, list) else data.get("data", [])
        return []


def get_recent_trades(days: int = 30) -> List[CongressionalTrade]:
    """
    Convenience function to get recent congressional trades.

    Args:
        days: Number of days to look back

    Returns:
        List of CongressionalTrade objects
    """
    client = HouseStockWatcherClient()
    return client.get_recent_trades(days)


if __name__ == "__main__":
    import sys

    print("Testing House Stock Watcher API...")
    print()

    client = HouseStockWatcherClient()

    # Get recent trades
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"Fetching trades from last {days} days...")

    trades = client.get_recent_trades(days)

    if trades:
        print(f"\nFound {len(trades)} congressional trades:")
        for trade in trades[:10]:
            print(f"  {trade.to_plain_language()}")

        # Show purchases
        purchases = client.get_purchases(days)
        print(f"\nOf these, {len(purchases)} are purchases")

        # Show large trades
        large = client.get_large_trades(min_amount=100000, days=days)
        if large:
            print(f"\nLarge trades (>$100k): {len(large)}")
            for trade in large[:5]:
                print(f"  {trade.to_plain_language()}")
    else:
        print("No trades found (API may be down)")

    # Search by ticker if provided
    if len(sys.argv) > 2:
        ticker = sys.argv[2]
        print(f"\nSearching for {ticker} trades...")
        ticker_trades = client.search_by_ticker(ticker, days=365)
        if ticker_trades:
            for trade in ticker_trades[:5]:
                print(f"  {trade.to_plain_language()}")
        else:
            print(f"  No {ticker} trades found")

    print("\nDone.")
