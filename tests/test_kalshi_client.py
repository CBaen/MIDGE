"""Tests for Kalshi prediction market client and signal adapter.

Tests cover:
  - KalshiMarket/KalshiMover dataclasses
  - KalshiMarketClient initialization and graceful degradation
  - Market categorization (macro vs geopolitical vs other)
  - Price mover detection logic
  - from_kalshi_mover signal adapter
  - Signal domain, source, confidence scaling
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.apis.kalshi_client import (
    KalshiMarket,
    KalshiMarketClient,
    KalshiMover,
)
from mae_core.market.signal_adapters.wave2_3_technical import from_kalshi_mover


# ─────────────────────────────────────────────────────────────
# Dataclass tests
# ─────────────────────────────────────────────────────────────

class TestKalshiMarket:
    def test_basic_construction(self):
        m = KalshiMarket(
            ticker="FED-RATE-25MAR", title="Fed raises rates?",
            event_ticker="FED-RATE", yes_price=0.85,
            volume_24h=5000, open_interest=12000, status="active",
        )
        assert m.ticker == "FED-RATE-25MAR"
        assert m.yes_price == 0.85
        assert m.category == ""
        assert m.result is None

    def test_previous_price_defaults_zero(self):
        m = KalshiMarket(
            ticker="T", title="T", event_ticker="E",
            yes_price=0.5, volume_24h=100, open_interest=100, status="active",
        )
        assert m.previous_yes_price == 0.0


class TestKalshiMover:
    def test_mover_construction(self):
        market = KalshiMarket(
            ticker="CPI-FEB", title="CPI above 3%?",
            event_ticker="CPI", yes_price=0.72,
            volume_24h=8000, open_interest=5000, status="active",
        )
        mover = KalshiMover(
            market=market, price_change=0.12,
            direction="bullish", strength=0.6,
        )
        assert mover.direction == "bullish"
        assert mover.price_change == 0.12
        assert mover.market.ticker == "CPI-FEB"


# ─────────────────────────────────────────────────────────────
# Client initialization
# ─────────────────────────────────────────────────────────────

class TestKalshiClientInit:
    def test_no_credentials_returns_empty(self):
        client = KalshiMarketClient(api_key_id="", private_key_path="")
        assert client.get_active_markets() == []
        assert client.get_market_movers() == []

    def test_statistics_before_init(self):
        client = KalshiMarketClient()
        stats = client.get_statistics()
        assert stats["calls_made"] == 0
        assert stats["errors"] == 0
        assert stats["mode"] == "demo"

    def test_demo_mode_default(self):
        client = KalshiMarketClient()
        assert client._demo is True

    def test_missing_key_file_graceful(self):
        client = KalshiMarketClient(
            api_key_id="test-key", private_key_path="/nonexistent/key.pem",
        )
        result = client._ensure_client()
        assert result is False
        assert client._client is None


# ─────────────────────────────────────────────────────────────
# Market categorization
# ─────────────────────────────────────────────────────────────

class TestCategorization:
    def _client(self):
        return KalshiMarketClient()

    def test_macro_category_fed(self):
        c = self._client()
        assert c._categorize("Will the Fed raise rates?", "FED-RATE") == "macro"

    def test_macro_category_cpi(self):
        c = self._client()
        assert c._categorize("CPI above 3% in March?", "CPI-MAR") == "macro"

    def test_macro_category_gdp(self):
        c = self._client()
        assert c._categorize("GDP growth above 2%?", "GDP-Q1") == "macro"

    def test_macro_category_unemployment(self):
        c = self._client()
        assert c._categorize("Unemployment below 4%?", "UNEMP") == "macro"

    def test_macro_category_inflation(self):
        c = self._client()
        assert c._categorize("Inflation stays above target", "INF") == "macro"

    def test_geopolitical_election(self):
        c = self._client()
        assert c._categorize("Who wins the election?", "ELEC") == "geopolitical"

    def test_geopolitical_tariff(self):
        c = self._client()
        assert c._categorize("New tariff on imports?", "TRADE") == "geopolitical"

    def test_other_category(self):
        c = self._client()
        assert c._categorize("Will it snow in NYC?", "WEATHER") == "other"

    def test_case_insensitive(self):
        c = self._client()
        assert c._categorize("FEDERAL RESERVE FOMC meeting", "FOMC") == "macro"


# ─────────────────────────────────────────────────────────────
# Price mover detection
# ─────────────────────────────────────────────────────────────

class TestMoverDetection:
    def test_mover_direction_bullish(self):
        mover = KalshiMover(
            market=KalshiMarket(
                ticker="T", title="T", event_ticker="E",
                yes_price=0.7, volume_24h=1000, open_interest=500,
                status="active", previous_yes_price=0.5,
            ),
            price_change=0.2, direction="bullish", strength=1.0,
        )
        assert mover.direction == "bullish"

    def test_mover_direction_bearish(self):
        mover = KalshiMover(
            market=KalshiMarket(
                ticker="T", title="T", event_ticker="E",
                yes_price=0.3, volume_24h=1000, open_interest=500,
                status="active", previous_yes_price=0.5,
            ),
            price_change=-0.2, direction="bearish", strength=1.0,
        )
        assert mover.direction == "bearish"

    def test_strength_scaling(self):
        """5% move = 0.25 strength, 20% move = 1.0 strength."""
        # 5% move
        s1 = min(1.0, 0.05 / 0.20)
        assert s1 == 0.25
        # 20% move
        s2 = min(1.0, 0.20 / 0.20)
        assert s2 == 1.0
        # 30% move caps at 1.0
        s3 = min(1.0, 0.30 / 0.20)
        assert s3 == 1.0


# ─────────────────────────────────────────────────────────────
# Signal adapter: from_kalshi_mover
# ─────────────────────────────────────────────────────────────

class TestFromKalshiMover:
    def _make_mover(self, **overrides):
        defaults = {
            "market": SimpleNamespace(
                ticker="FED-RATE-25MAR", title="Fed raises rates?",
                event_ticker="FED-RATE", yes_price=0.85,
                volume_24h=10000, open_interest=5000,
                status="active", category="macro",
            ),
            "price_change": 0.12,
            "direction": "bullish",
            "strength": 0.6,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_source(self):
        sig = from_kalshi_mover(self._make_mover())
        assert sig.source == "kalshi_market"

    def test_domain_is_prediction_market(self):
        sig = from_kalshi_mover(self._make_mover())
        assert sig.domain == "prediction_market"

    def test_asset_class(self):
        sig = from_kalshi_mover(self._make_mover())
        assert sig.asset_class == "prediction_market"

    def test_direction_passthrough(self):
        sig = from_kalshi_mover(self._make_mover(direction="bearish"))
        assert sig.direction == "bearish"

    def test_strength_passthrough(self):
        sig = from_kalshi_mover(self._make_mover(strength=0.8))
        assert sig.strength == 0.8

    def test_confidence_scales_with_volume(self):
        """Low volume = lower confidence, high volume = higher."""
        low_vol = self._make_mover()
        low_vol.market.volume_24h = 100
        sig_low = from_kalshi_mover(low_vol)

        high_vol = self._make_mover()
        high_vol.market.volume_24h = 50000
        sig_high = from_kalshi_mover(high_vol)

        assert sig_low.confidence < sig_high.confidence
        assert sig_high.confidence <= 0.75

    def test_confidence_minimum(self):
        mover = self._make_mover()
        mover.market.volume_24h = 0
        sig = from_kalshi_mover(mover)
        assert sig.confidence >= 0.45

    def test_signal_id_format(self):
        sig = from_kalshi_mover(self._make_mover())
        assert sig.signal_id.startswith("kalshi:FED-RATE-25MAR:")

    def test_metadata_has_title(self):
        sig = from_kalshi_mover(self._make_mover())
        assert sig.metadata["title"] == "Fed raises rates?"

    def test_metadata_has_price_change(self):
        sig = from_kalshi_mover(self._make_mover(price_change=0.12))
        assert sig.metadata["price_change_pct"] == 12.0

    def test_metadata_has_category(self):
        sig = from_kalshi_mover(self._make_mover())
        assert sig.metadata["category"] == "macro"

    def test_metadata_has_yes_price(self):
        sig = from_kalshi_mover(self._make_mover())
        assert sig.metadata["yes_price"] == 0.85

    def test_decay_rate(self):
        sig = from_kalshi_mover(self._make_mover())
        assert sig.decay_rate == 0.15

    def test_outcome_window(self):
        sig = from_kalshi_mover(self._make_mover())
        assert sig.outcome_window_days == 7

    def test_raw_type(self):
        sig = from_kalshi_mover(self._make_mover())
        assert sig.raw_type == "KalshiMover"

    def test_handles_none_market_gracefully(self):
        """If mover.market is None, falls back to mover itself."""
        mover = SimpleNamespace(
            market=None, ticker="DIRECT", title="Direct",
            event_ticker="E", yes_price=0.5, volume_24h=1000,
            category="other", price_change=0.1,
            direction="bullish", strength=0.5,
        )
        sig = from_kalshi_mover(mover)
        assert sig.source == "kalshi_market"
