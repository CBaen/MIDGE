"""Tests for mae_core.market.intelligence.regime_classifier."""

import pytest
from unittest.mock import MagicMock
from mae_core.market.intelligence.regime_classifier import RegimeClassifier
from mae_core.market.apis.price_fetcher import PriceData


def _make_price(price: float) -> PriceData:
    return PriceData(symbol="SPY", price=price, timestamp="2026-01-01", source="test")


class TestRegimeClassifier:
    """Test regime classification logic."""

    def test_no_price_fetcher_returns_default(self):
        rc = RegimeClassifier(price_fetcher=None)
        assert rc.classify() == "default"

    def test_bull_regime(self):
        """Steady uptrend > 2% return, low volatility."""
        fetcher = MagicMock()
        # 21 prices: 100 to 103 (3% up, low vol)
        prices = [100 + i * 0.15 for i in range(21)]
        fetcher.get_historical_price.side_effect = [
            _make_price(p) for p in prices
        ]
        rc = RegimeClassifier(price_fetcher=fetcher)
        assert rc.classify() == "bull"

    def test_bear_regime(self):
        """Steady downtrend < -2% return, low volatility."""
        fetcher = MagicMock()
        # 21 prices: 100 to 97 (3% down, low vol)
        prices = [100 - i * 0.15 for i in range(21)]
        fetcher.get_historical_price.side_effect = [
            _make_price(p) for p in prices
        ]
        rc = RegimeClassifier(price_fetcher=fetcher)
        assert rc.classify() == "bear"

    def test_sideways_regime(self):
        """Low return, low volatility."""
        fetcher = MagicMock()
        # 21 prices oscillating around 100 (tiny net change)
        prices = [100 + (0.1 if i % 2 == 0 else -0.1) for i in range(21)]
        fetcher.get_historical_price.side_effect = [
            _make_price(p) for p in prices
        ]
        rc = RegimeClassifier(price_fetcher=fetcher)
        assert rc.classify() == "sideways"

    def test_volatile_regime(self):
        """High volatility regardless of trend."""
        fetcher = MagicMock()
        # 21 prices with large swings
        prices = [100 + (5 if i % 2 == 0 else -5) for i in range(21)]
        fetcher.get_historical_price.side_effect = [
            _make_price(p) for p in prices
        ]
        rc = RegimeClassifier(price_fetcher=fetcher)
        assert rc.classify() == "volatile"

    def test_price_fetch_failure_returns_default(self):
        """When price fetcher raises, gracefully return default."""
        fetcher = MagicMock()
        fetcher.get_historical_price.side_effect = Exception("network error")
        rc = RegimeClassifier(price_fetcher=fetcher)
        assert rc.classify() == "default"

    def test_insufficient_data_returns_default(self):
        """Fewer than 5 prices should return default."""
        fetcher = MagicMock()
        fetcher.get_historical_price.side_effect = [
            _make_price(100), _make_price(101), None, None, None,
            None, None, None, None, None,
            None, None, None, None, None,
            None, None, None, None, None, None,
        ]
        rc = RegimeClassifier(price_fetcher=fetcher)
        assert rc.classify() == "default"

    def test_caching_within_same_day(self):
        """Classify should only call price_fetcher once per day."""
        fetcher = MagicMock()
        prices = [100 + i * 0.15 for i in range(21)]
        fetcher.get_historical_price.side_effect = [
            _make_price(p) for p in prices
        ]
        rc = RegimeClassifier(price_fetcher=fetcher)

        result1 = rc.classify()
        result2 = rc.classify()

        assert result1 == result2
        # Second call should not hit price_fetcher again
        assert fetcher.get_historical_price.call_count == 21  # Only from first call

    def test_to_dict(self):
        rc = RegimeClassifier(price_fetcher=None)
        rc.classify()
        d = rc.to_dict()
        assert "current_regime" in d
        assert "reference_symbol" in d
