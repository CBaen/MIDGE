"""
regime_classifier.py - Market Regime Detection

Classifies the current market regime from price data so that
ThompsonSampler can maintain separate reliability distributions
per regime. Different signals have different reliability in
bull vs bear vs volatile markets.

Regimes:
    "bull"     — 20-day return > +2% and volatility < 25%
    "bear"     — 20-day return < -2% and volatility < 25%
    "volatile" — 20-day annualized volatility > 25%
    "sideways" — abs(20-day return) <= 2% and volatility <= 25%
    "default"  — no price data available (graceful degradation)

Uses SPY (S&P 500 ETF) as the reference index. When price data
is unavailable, returns "default" so the system works identically
to the pre-regime-aware state.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Regime thresholds
_TREND_THRESHOLD = 0.02      # 2% over 20 days
_VOLATILITY_THRESHOLD = 0.25  # 25% annualized
_REFERENCE_SYMBOL = "SPY"
_LOOKBACK_DAYS = 20


class RegimeClassifier:
    """Classifies market regime from price data.

    Requires a PriceFetcher instance. When price data is unavailable,
    gracefully returns "default" — the system behaves identically to
    the pre-regime-aware state.
    """

    def __init__(self, price_fetcher=None, reference_symbol: str = _REFERENCE_SYMBOL):
        self._price_fetcher = price_fetcher
        self._reference_symbol = reference_symbol
        self._cached_regime: Optional[str] = None
        self._cache_date: Optional[str] = None

    def classify(self) -> str:
        """Return the current market regime.

        Returns one of: "bull", "bear", "volatile", "sideways", "default".
        Caches result for the current calendar day.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if self._cache_date == today and self._cached_regime is not None:
            return self._cached_regime

        regime = self._detect()
        self._cached_regime = regime
        self._cache_date = today
        return regime

    def _detect(self) -> str:
        """Detect regime from recent price data."""
        if self._price_fetcher is None:
            return "default"

        try:
            prices = self._get_recent_prices()
            if prices is None or len(prices) < 5:
                return "default"

            # Calculate 20-day return
            ret = (prices[-1] - prices[0]) / prices[0]

            # Calculate annualized volatility from daily returns
            daily_returns = [
                (prices[i] - prices[i - 1]) / prices[i - 1]
                for i in range(1, len(prices))
            ]
            if not daily_returns:
                return "default"

            mean_ret = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_ret) ** 2 for r in daily_returns) / max(1, len(daily_returns) - 1)
            daily_vol = math.sqrt(variance)
            annual_vol = daily_vol * math.sqrt(252)

            # Classify
            if annual_vol > _VOLATILITY_THRESHOLD:
                regime = "volatile"
            elif ret > _TREND_THRESHOLD:
                regime = "bull"
            elif ret < -_TREND_THRESHOLD:
                regime = "bear"
            else:
                regime = "sideways"

            logger.debug(
                "Regime classified: %s (20d return=%.3f, annual_vol=%.3f)",
                regime, ret, annual_vol,
            )
            return regime

        except Exception:
            logger.debug("Regime classification failed, using default", exc_info=True)
            return "default"

    def _get_recent_prices(self):
        """Fetch recent closing prices for the reference symbol."""
        prices = []
        today = datetime.now()

        for offset in range(_LOOKBACK_DAYS, -1, -1):
            date = today - timedelta(days=offset)
            date_str = date.strftime("%Y-%m-%d")
            try:
                data = self._price_fetcher.get_historical_price(
                    self._reference_symbol, date_str,
                )
                if data is not None and data.price > 0:
                    prices.append(data.price)
            except Exception:
                continue

        return prices if len(prices) >= 5 else None

    def to_dict(self) -> dict:
        """Export state for monitoring."""
        return {
            "current_regime": self._cached_regime or "unknown",
            "cache_date": self._cache_date,
            "reference_symbol": self._reference_symbol,
        }

    def get_statistics(self) -> dict:
        """For HolonProxy.sense() delegation."""
        return self.to_dict()
