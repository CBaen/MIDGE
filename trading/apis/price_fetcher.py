#!/usr/bin/env python3
"""
price_fetcher.py - Stock Price Data Fetcher

Fetches current and historical stock prices for outcome tracking.
Uses multiple free sources with fallback.

Primary: Yahoo Finance (yfinance)
Fallback: Alpha Vantage (free tier), Polygon (if configured)
"""

import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import requests

# Try to import yfinance, but don't fail if not installed
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("yfinance not installed. Run: pip install yfinance")


@dataclass
class PriceData:
    """Stock price data point."""
    symbol: str
    price: float
    timestamp: str
    source: str
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: int = 0
    change_pct: float = 0.0


class PriceFetcher:
    """
    Multi-source price fetcher with fallback.

    Usage:
        fetcher = PriceFetcher()
        price = fetcher.get_current_price("AAPL")
        print(f"AAPL: ${price.price}")
    """

    def __init__(self, alpha_vantage_key: str = None):
        """
        Initialize price fetcher.

        Args:
            alpha_vantage_key: Optional Alpha Vantage API key for fallback
        """
        self.alpha_vantage_key = alpha_vantage_key
        self._cache: Dict[str, Tuple[PriceData, datetime]] = {}
        self._cache_ttl = 60  # Cache prices for 60 seconds

    def get_current_price(self, symbol: str) -> Optional[PriceData]:
        """
        Get current price for a symbol.

        Args:
            symbol: Stock ticker (e.g., "AAPL", "MSFT")

        Returns:
            PriceData or None if unavailable
        """
        # Check cache
        if symbol in self._cache:
            data, cached_at = self._cache[symbol]
            if (datetime.now() - cached_at).seconds < self._cache_ttl:
                return data

        # Try sources in order
        price = None

        if YFINANCE_AVAILABLE:
            price = self._fetch_yfinance(symbol)

        if price is None and self.alpha_vantage_key:
            price = self._fetch_alpha_vantage(symbol)

        if price:
            self._cache[symbol] = (price, datetime.now())

        return price

    def get_historical_price(self, symbol: str, date: str) -> Optional[PriceData]:
        """
        Get closing price for a specific date.

        Args:
            symbol: Stock ticker
            date: Date in YYYY-MM-DD format

        Returns:
            PriceData or None if unavailable
        """
        if not YFINANCE_AVAILABLE:
            return None

        try:
            ticker = yf.Ticker(symbol)
            # Fetch data for date range
            target_date = datetime.strptime(date, "%Y-%m-%d")
            start = target_date - timedelta(days=1)
            end = target_date + timedelta(days=1)

            hist = ticker.history(start=start.strftime("%Y-%m-%d"),
                                 end=end.strftime("%Y-%m-%d"))

            if hist.empty:
                return None

            # Get closest date's closing price
            row = hist.iloc[-1]
            return PriceData(
                symbol=symbol,
                price=float(row["Close"]),
                timestamp=date,
                source="yfinance_historical",
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                volume=int(row["Volume"])
            )

        except Exception as e:
            print(f"Historical price error for {symbol}: {e}")
            return None

    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Optional[PriceData]]:
        """
        Get current prices for multiple symbols efficiently.

        Args:
            symbols: List of stock tickers

        Returns:
            Dict mapping symbol -> PriceData
        """
        results = {}

        if YFINANCE_AVAILABLE:
            try:
                # yfinance can fetch multiple symbols at once
                tickers = yf.Tickers(" ".join(symbols))
                for symbol in symbols:
                    try:
                        ticker = tickers.tickers.get(symbol)
                        if ticker:
                            info = ticker.info
                            results[symbol] = PriceData(
                                symbol=symbol,
                                price=info.get("currentPrice") or info.get("regularMarketPrice", 0),
                                timestamp=datetime.now().isoformat(),
                                source="yfinance",
                                open=info.get("open", 0),
                                high=info.get("dayHigh", 0),
                                low=info.get("dayLow", 0),
                                volume=info.get("volume", 0),
                                change_pct=info.get("regularMarketChangePercent", 0)
                            )
                    except:
                        results[symbol] = None
            except Exception as e:
                print(f"Batch price fetch error: {e}")
                # Fall back to individual fetches
                for symbol in symbols:
                    results[symbol] = self.get_current_price(symbol)
        else:
            for symbol in symbols:
                results[symbol] = self.get_current_price(symbol)

        return results

    def _fetch_yfinance(self, symbol: str) -> Optional[PriceData]:
        """Fetch price from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if not price:
                # Try getting from history
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])

            if price:
                return PriceData(
                    symbol=symbol,
                    price=float(price),
                    timestamp=datetime.now().isoformat(),
                    source="yfinance",
                    open=info.get("open", 0) or 0,
                    high=info.get("dayHigh", 0) or 0,
                    low=info.get("dayLow", 0) or 0,
                    volume=info.get("volume", 0) or 0,
                    change_pct=info.get("regularMarketChangePercent", 0) or 0
                )

        except Exception as e:
            print(f"yfinance error for {symbol}: {e}")

        return None

    def _fetch_alpha_vantage(self, symbol: str) -> Optional[PriceData]:
        """Fetch price from Alpha Vantage (free tier: 5 calls/min)."""
        if not self.alpha_vantage_key:
            return None

        try:
            url = (
                f"https://www.alphavantage.co/query"
                f"?function=GLOBAL_QUOTE"
                f"&symbol={symbol}"
                f"&apikey={self.alpha_vantage_key}"
            )

            response = requests.get(url, timeout=10)
            data = response.json()

            quote = data.get("Global Quote", {})
            if quote:
                return PriceData(
                    symbol=symbol,
                    price=float(quote.get("05. price", 0)),
                    timestamp=datetime.now().isoformat(),
                    source="alpha_vantage",
                    open=float(quote.get("02. open", 0)),
                    high=float(quote.get("03. high", 0)),
                    low=float(quote.get("04. low", 0)),
                    volume=int(quote.get("06. volume", 0)),
                    change_pct=float(quote.get("10. change percent", "0").replace("%", ""))
                )

        except Exception as e:
            print(f"Alpha Vantage error for {symbol}: {e}")

        return None


def get_price(symbol: str) -> Optional[float]:
    """Convenience function to get current price."""
    fetcher = PriceFetcher()
    data = fetcher.get_current_price(symbol)
    return data.price if data else None


def get_prices(symbols: List[str]) -> Dict[str, Optional[float]]:
    """Convenience function to get multiple prices."""
    fetcher = PriceFetcher()
    results = fetcher.get_multiple_prices(symbols)
    return {s: (d.price if d else None) for s, d in results.items()}


# Price fetcher function for outcome tracker
def price_fetcher_for_outcomes(symbol: str) -> float:
    """
    Price fetcher compatible with OutcomeTracker.check_and_record_outcomes().

    Raises:
        ValueError: If price cannot be fetched
    """
    fetcher = PriceFetcher()
    data = fetcher.get_current_price(symbol)
    if data:
        return data.price
    raise ValueError(f"Could not fetch price for {symbol}")


if __name__ == "__main__":
    print("Price Fetcher Test")
    print("=" * 50)

    if not YFINANCE_AVAILABLE:
        print("yfinance not installed. Install with: pip install yfinance")
        print("Testing with placeholder...")
    else:
        fetcher = PriceFetcher()

        # Test single price
        print("\nSingle price fetch:")
        symbols = ["AAPL", "MSFT", "GOOGL"]
        for symbol in symbols:
            price = fetcher.get_current_price(symbol)
            if price:
                print(f"  {symbol}: ${price.price:.2f} ({price.change_pct:+.2f}%)")
            else:
                print(f"  {symbol}: Could not fetch")

        # Test batch fetch
        print("\nBatch price fetch:")
        prices = fetcher.get_multiple_prices(["LMT", "BA", "RTX"])
        for symbol, data in prices.items():
            if data:
                print(f"  {symbol}: ${data.price:.2f}")
            else:
                print(f"  {symbol}: Could not fetch")

        # Test historical
        print("\nHistorical price fetch:")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        hist = fetcher.get_historical_price("AAPL", yesterday)
        if hist:
            print(f"  AAPL on {yesterday}: ${hist.price:.2f}")
