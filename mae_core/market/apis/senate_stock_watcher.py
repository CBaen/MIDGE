#!/usr/bin/env python3
"""
senate_stock_watcher.py - Senate Stock Trade Tracker

Fetches stock trades made by US Senators from senatestockwatcher.com.
This is FREE data based on STOCK Act disclosures.

Senate trades can be early indicators:
- Committee chairs may trade before legislation (more concentrated power than House)
- Senators serve 6-year terms — longer-horizon positions are meaningful
- 45-day disclosure delay means some trades are already "old news"

Data source: https://senatestockwatcher.com/api
GitHub mirror: https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/master/data/all_transactions.json
Updated daily as new disclosures are filed.
"""

import time
import logging
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import requests

from mae_core.market.apis.house_stock_watcher import CongressionalTrade

logger = logging.getLogger(__name__)


# API endpoints (in priority order)
# Primary: GitHub raw data (most reliable free source)
SENATE_STOCK_API = "https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/master/data/all_transactions.json"
# Fallback: Original site API
SENATE_STOCK_API_FALLBACK = "https://senatestockwatcher.com/api"

# Rate limiting
REQUEST_DELAY = 1.0  # Be respectful to free service


class SenateStockWatcherClient:
    """
    Client for Senate Stock Trade data.

    Priority:
    1. GitHub raw JSON (timothycarambat/senate-stock-watcher-data)
    2. Fallback to senatestockwatcher.com/api

    Data is based on STOCK Act disclosures.
    Party affiliation is not included in the Senate data source.
    District is set to "Senate" for all records.
    """

    def __init__(self, provider=None, raw_store=None):
        self._provider = provider
        self._raw_store = raw_store
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MIDGE Trading Research"
        })
        self._last_request_time = 0
        self._cache = None
        self._cache_time = None
        self._cache_duration = 3600  # 1 hour cache

        logger.info("SenateStockWatcherClient initialized - using free GitHub/API sources")

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _request(self, url: str, headers: Optional[Dict] = None,
                 params: Optional[Dict] = None, source_name: str = "senate_free") -> Optional[dict]:
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

    def _get_all_trades(self, use_cache: bool = True) -> List[dict]:
        """
        Fetch all trades from API.

        Uses caching to avoid repeated requests.
        Priority: GitHub mirror -> senatestockwatcher.com
        """
        # Check cache
        if use_cache and self._cache and self._cache_time:
            if time.time() - self._cache_time < self._cache_duration:
                return self._cache

        self._rate_limit()

        endpoints = [
            (SENATE_STOCK_API, "GitHub mirror"),
            (SENATE_STOCK_API_FALLBACK, "senatestockwatcher.com"),
        ]

        for url, name in endpoints:
            data = self._request(url, source_name="senate_free")
            if data is not None:
                if self._raw_store:
                    try:
                        self._raw_store.store_congressional_trades(
                            [{**t, "chamber": "senate"} for t in (data if isinstance(data, list) else [])])
                    except Exception:
                        pass
                self._cache = data
                self._cache_time = time.time()
                logger.debug(f"Loaded {len(data)} trades from {name}")
                return data
            logger.warning(f"{name} failed, trying next...")

        logger.warning("All Senate trade endpoints failed")
        return []

    def _normalize_trade(self, item: dict) -> Optional[dict]:
        """
        Normalize trade data from the Senate Stock Watcher JSON format.

        Senate JSON fields:
            transaction_date, owner, ticker, asset_description, asset_type,
            type (Purchase/Sale), amount, comment, senator, ptr_link, disclosure_date
        """
        return {
            "disclosure_date": item.get("disclosure_date", ""),
            "transaction_date": item.get("transaction_date", ""),
            "representative": item.get("senator", "Unknown"),
            "party": "",          # Senate data does not include party
            "district": "Senate",
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

            # Clean ticker — guard against NaN, None, dashes, and nonsense values
            ticker = normalized["ticker"]
            if not ticker or ticker == "--" or str(ticker).strip().upper() in ("NAN", "N/A", "NONE"):
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

    def search_by_senator(self, name: str, days: int = 365) -> List[CongressionalTrade]:
        """
        Find all trades by a senator.

        Args:
            name: Partial or full name of senator
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


def get_recent_trades(days: int = 30) -> List[CongressionalTrade]:
    """
    Convenience function to get recent Senate trades.

    Args:
        days: Number of days to look back

    Returns:
        List of CongressionalTrade objects
    """
    client = SenateStockWatcherClient()
    return client.get_recent_trades(days)


if __name__ == "__main__":
    import sys

    print("Testing Senate Stock Watcher API...")
    print()

    client = SenateStockWatcherClient()

    # Get recent trades
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"Fetching trades from last {days} days...")

    trades = client.get_recent_trades(days)

    if trades:
        print(f"\nFound {len(trades)} Senate trades:")
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
