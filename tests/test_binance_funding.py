"""Tests for Binance funding rate client, signal adapter, and pipeline wiring."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from mae_core.market.apis.binance_funding_client import (
    BinanceFundingClient,
    FundingRate,
    EXTREME_THRESHOLD,
    ELEVATED_THRESHOLD,
    TRACKED_SYMBOLS,
)
from mae_core.market.signal_adapters.wave2_3_technical import from_binance_funding
from mae_core.market.fetchers_crypto import fetch_binance_funding
from mae_core.market.sensing_constants import (
    TIER_ROUTING,
    SOURCE_ROTATION,
    _ROTATION_TO_THOMPSON,
    _ABSENCE_SOURCE_DOMAINS,
    _DOMAIN_TO_SOURCES,
)


# ---------------------------------------------------------------------------
# FundingRate dataclass tests
# ---------------------------------------------------------------------------

class TestFundingRate:
    def test_extreme_positive_is_bearish(self):
        rate = FundingRate(symbol="BTCUSDT", rate=0.001, timestamp=datetime.now())
        assert rate.direction == "bearish"

    def test_extreme_negative_is_bullish(self):
        rate = FundingRate(symbol="BTCUSDT", rate=-0.001, timestamp=datetime.now())
        assert rate.direction == "bullish"

    def test_neutral_rate(self):
        rate = FundingRate(symbol="BTCUSDT", rate=0.0001, timestamp=datetime.now())
        assert rate.direction == "neutral"

    def test_zero_rate_is_neutral(self):
        rate = FundingRate(symbol="BTCUSDT", rate=0.0, timestamp=datetime.now())
        assert rate.direction == "neutral"

    def test_strength_extreme(self):
        rate = FundingRate(symbol="BTCUSDT", rate=0.0015, timestamp=datetime.now())
        assert rate.strength == 1.0

    def test_strength_moderate(self):
        rate = FundingRate(symbol="BTCUSDT", rate=0.0003, timestamp=datetime.now())
        assert 0.3 < rate.strength < 1.0

    def test_strength_low(self):
        rate = FundingRate(symbol="BTCUSDT", rate=0.00005, timestamp=datetime.now())
        assert rate.strength == 0.1

    def test_mark_price_optional(self):
        rate = FundingRate(symbol="BTCUSDT", rate=0.0001, timestamp=datetime.now())
        assert rate.mark_price is None

    def test_mark_price_set(self):
        rate = FundingRate(
            symbol="BTCUSDT", rate=0.0001,
            timestamp=datetime.now(), mark_price=67000.0,
        )
        assert rate.mark_price == 67000.0


# ---------------------------------------------------------------------------
# Signal adapter tests
# ---------------------------------------------------------------------------

class TestFromBinanceFunding:
    def _make_rate(self, symbol="BTCUSDT", rate=0.001, mark_price=67000.0):
        return FundingRate(
            symbol=symbol,
            rate=rate,
            timestamp=datetime(2026, 3, 10, 8, 0, tzinfo=timezone.utc),
            mark_price=mark_price,
        )

    def test_returns_market_signal(self):
        sig = from_binance_funding(self._make_rate())
        from mae_core.market.signal import MarketSignal
        assert isinstance(sig, MarketSignal)

    def test_source_is_binance_funding(self):
        sig = from_binance_funding(self._make_rate())
        assert sig.source == "binance_funding"

    def test_domain_is_positioning(self):
        sig = from_binance_funding(self._make_rate())
        assert sig.domain == "positioning"

    def test_asset_class_is_crypto(self):
        sig = from_binance_funding(self._make_rate())
        assert sig.asset_class == "crypto"

    def test_symbol_strips_usdt(self):
        sig = from_binance_funding(self._make_rate(symbol="BTCUSDT"))
        assert sig.symbol == "BTC"

    def test_symbol_strips_busd(self):
        sig = from_binance_funding(self._make_rate(symbol="ETHBUSD"))
        assert sig.symbol == "ETH"

    def test_outcome_symbol_has_usd_suffix(self):
        sig = from_binance_funding(self._make_rate(symbol="SOLUSDT"))
        assert sig.outcome_symbol == "SOL-USD"

    def test_positive_rate_is_bearish(self):
        sig = from_binance_funding(self._make_rate(rate=0.001))
        assert sig.direction == "bearish"

    def test_negative_rate_is_bullish(self):
        sig = from_binance_funding(self._make_rate(rate=-0.001))
        assert sig.direction == "bullish"

    def test_small_rate_is_neutral(self):
        sig = from_binance_funding(self._make_rate(rate=0.0001))
        assert sig.direction == "neutral"

    def test_confidence_is_055(self):
        sig = from_binance_funding(self._make_rate())
        assert sig.confidence == 0.55

    def test_decay_rate_is_070(self):
        sig = from_binance_funding(self._make_rate())
        assert sig.decay_rate == 0.70

    def test_outcome_window_3_days(self):
        sig = from_binance_funding(self._make_rate())
        assert sig.outcome_window_days == 3

    def test_metadata_has_funding_rate(self):
        sig = from_binance_funding(self._make_rate(rate=0.0005))
        assert sig.metadata["funding_rate"] == 0.0005
        assert sig.metadata["funding_rate_pct"] == pytest.approx(0.05)

    def test_metadata_has_annualized_rate(self):
        sig = from_binance_funding(self._make_rate(rate=0.0001))
        # 0.01% * 3 * 365 = 10.95%
        assert sig.metadata["annualized_rate"] == pytest.approx(10.95)

    def test_metadata_has_mark_price(self):
        sig = from_binance_funding(self._make_rate(mark_price=67000.0))
        assert sig.metadata["mark_price"] == 67000.0

    def test_signal_id_format(self):
        sig = from_binance_funding(self._make_rate(symbol="ETHUSDT"))
        assert sig.signal_id.startswith("binance_funding:ETHUSDT:")

    def test_raw_type_is_funding_rate(self):
        sig = from_binance_funding(self._make_rate())
        assert sig.raw_type == "FundingRate"

    def test_raw_id_is_original_symbol(self):
        sig = from_binance_funding(self._make_rate(symbol="BTCUSDT"))
        assert sig.raw_id == "BTCUSDT"


# ---------------------------------------------------------------------------
# Fetcher tests
# ---------------------------------------------------------------------------

class TestFetchBinanceFunding:
    def test_none_client_returns_empty(self):
        assert fetch_binance_funding(None, from_binance_funding) == []

    def test_fetches_and_converts(self):
        client = MagicMock()
        client.get_funding_rates.return_value = [
            FundingRate("BTCUSDT", 0.001, datetime.now()),
            FundingRate("ETHUSDT", -0.0008, datetime.now()),
        ]
        signals = fetch_binance_funding(client, from_binance_funding)
        assert len(signals) == 2
        assert signals[0].symbol == "BTC"
        assert signals[1].symbol == "ETH"

    def test_handles_client_exception(self):
        client = MagicMock()
        client.get_funding_rates.side_effect = ConnectionError("timeout")
        signals = fetch_binance_funding(client, from_binance_funding)
        assert signals == []

    def test_skips_failing_converter(self):
        client = MagicMock()
        client.get_funding_rates.return_value = [
            FundingRate("BTCUSDT", 0.001, datetime.now()),
            FundingRate("ETHUSDT", 0.0002, datetime.now()),
        ]

        def bad_converter(rate):
            if rate.symbol == "BTCUSDT":
                raise ValueError("bad")
            return from_binance_funding(rate)

        signals = fetch_binance_funding(client, bad_converter)
        assert len(signals) == 1
        assert signals[0].symbol == "ETH"


# ---------------------------------------------------------------------------
# Pipeline wiring tests
# ---------------------------------------------------------------------------

class TestBinanceFundingWiring:
    def test_in_tier_routing(self):
        assert "binance_funding" in TIER_ROUTING
        assert TIER_ROUTING["binance_funding"] == "tactical"

    def test_in_source_rotation(self):
        assert "binance_funding" in SOURCE_ROTATION

    def test_in_thompson_mapping(self):
        assert "binance_funding" in _ROTATION_TO_THOMPSON
        assert _ROTATION_TO_THOMPSON["binance_funding"] == "binance_funding"

    def test_in_absence_domains(self):
        assert "binance_funding" in _ABSENCE_SOURCE_DOMAINS
        assert _ABSENCE_SOURCE_DOMAINS["binance_funding"] == "positioning"

    def test_in_domain_to_sources(self):
        positioning_sources = _DOMAIN_TO_SOURCES.get("positioning", [])
        assert "binance_funding" in positioning_sources

    def test_shares_positioning_domain_with_cot(self):
        positioning_sources = _DOMAIN_TO_SOURCES.get("positioning", [])
        assert "cot_positioning" in positioning_sources
        assert "binance_funding" in positioning_sources

    def test_import_from_signal_module(self):
        from mae_core.market.signal import from_binance_funding as fn
        assert callable(fn)

    def test_import_from_sensing_fetchers(self):
        from mae_core.market.sensing_fetchers import fetch_binance_funding as fn
        assert callable(fn)


# ---------------------------------------------------------------------------
# Client tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestBinanceFundingClient:
    def test_constructor(self):
        client = BinanceFundingClient()
        assert client._cache == {}

    def test_constructor_with_raw_store(self):
        store = MagicMock()
        client = BinanceFundingClient(raw_store=store)
        assert client._raw_store is store

    def test_tracked_symbols_has_10(self):
        assert len(TRACKED_SYMBOLS) == 10
        assert "BTCUSDT" in TRACKED_SYMBOLS
        assert "ETHUSDT" in TRACKED_SYMBOLS

    @patch.object(BinanceFundingClient, "_fetch_one")
    def test_get_funding_rates_calls_fetch(self, mock_fetch):
        mock_fetch.return_value = FundingRate("BTCUSDT", 0.0001, datetime.now())
        client = BinanceFundingClient()
        rates = client.get_funding_rates(symbols=["BTCUSDT"])
        assert len(rates) == 1
        mock_fetch.assert_called_once_with("BTCUSDT")

    @patch.object(BinanceFundingClient, "_fetch_one")
    def test_caching_prevents_refetch(self, mock_fetch):
        mock_fetch.return_value = FundingRate("BTCUSDT", 0.0001, datetime.now())
        client = BinanceFundingClient()
        client.get_funding_rates(symbols=["BTCUSDT"])
        client.get_funding_rates(symbols=["BTCUSDT"])
        # Only called once — second call uses cache
        assert mock_fetch.call_count == 1

    @patch.object(BinanceFundingClient, "_fetch_one")
    def test_get_extreme_rates_filters(self, mock_fetch):
        mock_fetch.side_effect = [
            FundingRate("BTCUSDT", 0.001, datetime.now()),  # Extreme
            FundingRate("ETHUSDT", 0.0001, datetime.now()),  # Not extreme
        ]
        client = BinanceFundingClient()
        extreme = client.get_extreme_rates(symbols=["BTCUSDT", "ETHUSDT"])
        assert len(extreme) == 1
        assert extreme[0].symbol == "BTCUSDT"

    @patch.object(BinanceFundingClient, "_fetch_one")
    def test_none_fetch_skipped(self, mock_fetch):
        mock_fetch.return_value = None
        client = BinanceFundingClient()
        rates = client.get_funding_rates(symbols=["BADUSDT"])
        assert rates == []
