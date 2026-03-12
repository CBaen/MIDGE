#!/usr/bin/env python3
"""
price_fetcher.py - Stock Price Data Fetcher

Fetches current and historical stock prices for outcome tracking.
Uses multiple free sources with fallback.

Primary: Yahoo Finance (yfinance)
Fallback: Alpha Vantage (free tier), Polygon (if configured)
"""

import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import requests

logger = logging.getLogger(__name__)

# Try to import yfinance, but don't fail if not installed
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Run: pip install yfinance")


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
    # Extended fundamental fields from yfinance ticker.info (80+ fields available)
    # These are populated by _fetch_yfinance() and get_multiple_prices() when
    # info dict is available. All Optional — not present for Alpha Vantage results.
    short_ratio: Optional[float] = None           # Short interest / avg daily volume
    held_pct_insiders: Optional[float] = None     # Insider ownership fraction (0-1)
    held_pct_institutions: Optional[float] = None # Institutional ownership fraction (0-1)
    beta: Optional[float] = None                  # 5-year monthly beta vs S&P 500
    forward_pe: Optional[float] = None            # Forward price-to-earnings ratio
    sector: Optional[str] = None                  # e.g. "Technology", "Healthcare"
    industry: Optional[str] = None                # e.g. "Semiconductors"
    fifty_two_week_high: Optional[float] = None   # 52-week high price
    fifty_two_week_low: Optional[float] = None    # 52-week low price
    shares_short: Optional[int] = None            # Current short interest (shares)
    target_mean_price: Optional[float] = None     # Analyst consensus target price
    recommendation_mean: Optional[float] = None   # Analyst rating (1=Strong Buy, 5=Sell)


class PriceFetcher:
    """
    Multi-source price fetcher with fallback.

    Usage:
        fetcher = PriceFetcher()
        price = fetcher.get_current_price("AAPL")
        print(f"AAPL: ${price.price}")
    """

    def __init__(self, alpha_vantage_key: str = None, provider=None, raw_store=None):
        """
        Initialize price fetcher.

        Args:
            alpha_vantage_key: Optional Alpha Vantage API key for fallback
            provider: Optional MarketDataProvider for gateway routing
            raw_store: Optional RawStore for persisting full ticker.info data
        """
        self.alpha_vantage_key = alpha_vantage_key
        self._provider = provider
        self._raw_store = raw_store
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

    def get_intraday_candles(
        self,
        symbol: str,
        interval: str = "1m",
        period: str = "5d",
    ):
        """Fetch intraday OHLCV candles for a symbol.

        Args:
            symbol: Ticker (e.g., "ES=F", "NQ=F", "AAPL")
            interval: Bar interval (1m, 2m, 5m, 15m, 30m, 60m, 1h)
            period: Lookback period. For 1m: max 7 days (use "5d" for safety).

        Returns:
            DataFrame with lowercase columns (open, high, low, close, volume)
            indexed by datetime. Returns None on failure.
        """
        if not YFINANCE_AVAILABLE:
            return None
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(interval=interval, period=period)
            if df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            logger.warning("Intraday candles error for %s: %s", symbol, e)
            return None

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
            logger.warning(f"Historical price error for {symbol}: {e}")
            return None

    def get_daily_history(self, symbol: str, days: int = 90) -> List[PriceData]:
        """
        Fetch daily OHLCV history for a symbol using yfinance.

        Returns one PriceData per trading day with change_pct calculated as
        intraday (close - open) / open * 100. Used by the backfill script
        to populate signal archives with price action data.

        Args:
            symbol: Stock ticker (e.g. "AAPL")
            days: Calendar days of history to fetch

        Returns:
            List of PriceData sorted oldest-first. Empty on failure.
        """
        if not YFINANCE_AVAILABLE:
            return []

        try:
            start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            end = datetime.now().strftime("%Y-%m-%d")

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)

            if df.empty:
                return []

            results: List[PriceData] = []
            for idx, row in df.iterrows():
                # Handle timezone-aware DatetimeIndex
                if hasattr(idx, 'tz') and idx.tz is not None:
                    idx = idx.tz_localize(None)

                open_price = float(row["Open"])
                close_price = float(row["Close"])
                change_pct = ((close_price - open_price) / open_price * 100) if open_price != 0 else 0.0

                results.append(PriceData(
                    symbol=symbol,
                    price=close_price,
                    timestamp=idx.strftime("%Y-%m-%d"),
                    source="yfinance_history",
                    open=open_price,
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    volume=int(row["Volume"]),
                    change_pct=change_pct,
                ))

            return results

        except Exception as e:
            logger.warning("Daily history error for %s: %s", symbol, e)
            return []

    def get_weekly_history(self, symbol: str, weeks: int = 52) -> List[PriceData]:
        """Fetch weekly OHLCV history for a symbol using yfinance.

        Returns one PriceData per trading week. Used by fractal resonance
        detector for multi-timeframe pattern analysis.

        Args:
            symbol: Stock ticker (e.g. "AAPL", "ES=F")
            weeks: Number of weeks of history to fetch

        Returns:
            List of PriceData sorted oldest-first. Empty on failure.
        """
        if not YFINANCE_AVAILABLE:
            return []
        try:
            days = weeks * 7
            start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            end = datetime.now().strftime("%Y-%m-%d")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval="1wk")
            if df.empty:
                return []
            results: List[PriceData] = []
            for idx, row in df.iterrows():
                if hasattr(idx, "tz") and idx.tz is not None:
                    idx = idx.tz_localize(None)
                open_price = float(row["Open"])
                close_price = float(row["Close"])
                change_pct = ((close_price - open_price) / open_price * 100) if open_price != 0 else 0.0
                results.append(PriceData(
                    symbol=symbol,
                    price=close_price,
                    timestamp=idx.strftime("%Y-%m-%d"),
                    source="yfinance_weekly",
                    open=open_price,
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    volume=int(row["Volume"]),
                    change_pct=change_pct,
                ))
            return results
        except Exception as e:
            logger.warning("Weekly history error for %s: %s", symbol, e)
            return []

    def get_monthly_history(self, symbol: str, months: int = 24) -> List[PriceData]:
        """Fetch monthly OHLCV history for a symbol using yfinance.

        Returns one PriceData per calendar month. Used by fractal resonance
        detector for multi-timeframe pattern analysis.

        Args:
            symbol: Stock ticker (e.g. "AAPL", "ES=F")
            months: Number of months of history to fetch

        Returns:
            List of PriceData sorted oldest-first. Empty on failure.
        """
        if not YFINANCE_AVAILABLE:
            return []
        try:
            days = months * 31
            start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            end = datetime.now().strftime("%Y-%m-%d")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval="1mo")
            if df.empty:
                return []
            results: List[PriceData] = []
            for idx, row in df.iterrows():
                if hasattr(idx, "tz") and idx.tz is not None:
                    idx = idx.tz_localize(None)
                open_price = float(row["Open"])
                close_price = float(row["Close"])
                change_pct = ((close_price - open_price) / open_price * 100) if open_price != 0 else 0.0
                results.append(PriceData(
                    symbol=symbol,
                    price=close_price,
                    timestamp=idx.strftime("%Y-%m-%d"),
                    source="yfinance_monthly",
                    open=open_price,
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    volume=int(row["Volume"]),
                    change_pct=change_pct,
                ))
            return results
        except Exception as e:
            logger.warning("Monthly history error for %s: %s", symbol, e)
            return []

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
                            # Store full info before extracting
                            if self._raw_store and info:
                                try:
                                    self._raw_store.store_price_snapshot(symbol, info)
                                except Exception:
                                    pass
                            results[symbol] = PriceData(
                                symbol=symbol,
                                price=info.get("currentPrice") or info.get("regularMarketPrice", 0),
                                timestamp=datetime.now().isoformat(),
                                source="yfinance",
                                open=info.get("open", 0),
                                high=info.get("dayHigh", 0),
                                low=info.get("dayLow", 0),
                                volume=info.get("volume", 0),
                                change_pct=info.get("regularMarketChangePercent", 0),
                                short_ratio=info.get("shortRatio"),
                                held_pct_insiders=info.get("heldPercentInsiders"),
                                held_pct_institutions=info.get("heldPercentInstitutions"),
                                beta=info.get("beta"),
                                forward_pe=info.get("forwardPE"),
                                sector=info.get("sector"),
                                industry=info.get("industry"),
                                fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
                                fifty_two_week_low=info.get("fiftyTwoWeekLow"),
                                shares_short=info.get("sharesShort"),
                                target_mean_price=info.get("targetMeanPrice"),
                                recommendation_mean=info.get("recommendationMean"),
                            )
                    except:
                        results[symbol] = None
            except Exception as e:
                logger.warning(f"Batch price fetch error: {e}")
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

            # Store ALL 80+ fields before extracting the 6 we use
            if self._raw_store and info:
                try:
                    self._raw_store.store_price_snapshot(symbol, info)
                except Exception:
                    pass

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
                    change_pct=info.get("regularMarketChangePercent", 0) or 0,
                    short_ratio=info.get("shortRatio"),
                    held_pct_insiders=info.get("heldPercentInsiders"),
                    held_pct_institutions=info.get("heldPercentInstitutions"),
                    beta=info.get("beta"),
                    forward_pe=info.get("forwardPE"),
                    sector=info.get("sector"),
                    industry=info.get("industry"),
                    fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
                    fifty_two_week_low=info.get("fiftyTwoWeekLow"),
                    shares_short=info.get("sharesShort"),
                    target_mean_price=info.get("targetMeanPrice"),
                    recommendation_mean=info.get("recommendationMean"),
                )

        except Exception as e:
            logger.warning(f"yfinance error for {symbol}: {e}")

        return None

    def _fetch_alpha_vantage(self, symbol: str) -> Optional[PriceData]:
        """Fetch price from Alpha Vantage (free tier: 5 calls/min)."""
        if not self.alpha_vantage_key:
            return None

        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.alpha_vantage_key,
            }

            if self._provider is not None:
                from mae_core.market.apis.market_data_provider import market_request
                from mae_core.external.api_client import ApiResponseStatus
                url = "https://www.alphavantage.co/query"
                resp = market_request(
                    self._provider, url, params=params,
                    source_name="alpha_vantage", timeout_ms=10000.0,
                )
                if resp.status != ApiResponseStatus.SUCCESS:
                    return None
                data = resp.payload or {}
            else:
                url = "https://www.alphavantage.co/query"
                response = requests.get(url, params=params, timeout=10)
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
            logger.warning(f"Alpha Vantage error for {symbol}: {e}")

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
