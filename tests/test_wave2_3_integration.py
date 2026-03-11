"""Integration tests for Wave 2+3 signal adapters, fetch functions, and sensing hook.

Covers:
- Signal adapters (wave2_3.py): crypto, openinsider, 13f, activist, finviz, realtime, suppression
- Fetch functions (sensing_fetchers.py): crypto, openinsider, 13f, finviz, economic_calendar
- Sensing hook constants: SOURCE_ROTATION, TIER_ROUTING, _ROTATION_TO_THOMPSON
- Market clock availability sets: ALWAYS_AVAILABLE, MARKET_HOURS, PERIODIC
- Thompson keys in learning_config.py
- Economic calendar suppression integration
- WebSocket realtime signal processing
"""

import pytest
from datetime import datetime, date
from unittest.mock import MagicMock, patch


# ============================================================================
# 1. Crypto Adapters
# ============================================================================

class TestCryptoAdapters:
    """Tests for from_crypto_signal — CoinGecko and CoinCap paths."""

    def _make_coingecko_price(self, change_24h_pct=5.0):
        """Return a duck-typed CoinGecko CryptoPrice mock."""
        m = MagicMock()
        m.coin_id = "bitcoin"
        m.symbol = "BTC"
        m.price_usd = 50000.0
        m.volume_24h = 1e9
        m.change_24h_pct = change_24h_pct
        m.change_7d_pct = 10.0
        m.market_cap = 1e12
        m.last_updated = "2026-03-02T00:00:00Z"
        # Ensure hasattr("coin_id") returns True (MagicMock already does this)
        return m

    def _make_coincap_asset(self, change_24h_pct=5.0):
        """Return a duck-typed CoinCap CryptoAsset mock — has asset_id, not coin_id."""
        m = MagicMock(spec=[
            "asset_id", "symbol", "name", "rank", "price_usd",
            "volume_24h_usd", "change_24h_pct", "market_cap_usd",
            "supply", "max_supply", "vwap_24h",
        ])
        m.asset_id = "bitcoin"
        m.symbol = "BTC"
        m.name = "Bitcoin"
        m.rank = 1
        m.price_usd = 50000.0
        m.volume_24h_usd = 1e9
        m.change_24h_pct = change_24h_pct
        m.market_cap_usd = 1e12
        m.supply = 19_000_000.0
        m.max_supply = 21_000_000.0
        m.vwap_24h = 49500.0
        return m

    def test_coingecko_bullish_direction(self):
        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        crypto = self._make_coingecko_price(change_24h_pct=5.0)
        sig = from_crypto_signal(crypto)
        assert sig.direction == "bullish"

    def test_coingecko_source_field(self):
        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        crypto = self._make_coingecko_price(change_24h_pct=5.0)
        sig = from_crypto_signal(crypto)
        assert sig.source == "crypto_coingecko"

    def test_coingecko_domain_and_asset_class(self):
        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        crypto = self._make_coingecko_price(change_24h_pct=5.0)
        sig = from_crypto_signal(crypto)
        assert sig.domain == "crypto"
        assert sig.asset_class == "crypto"

    def test_coingecko_bearish_direction(self):
        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        crypto = self._make_coingecko_price(change_24h_pct=-4.0)
        sig = from_crypto_signal(crypto)
        assert sig.direction == "bearish"

    def test_coingecko_neutral_direction(self):
        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        crypto = self._make_coingecko_price(change_24h_pct=1.0)
        sig = from_crypto_signal(crypto)
        assert sig.direction == "neutral"

    def test_coingecko_strength_scaling(self):
        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        crypto = self._make_coingecko_price(change_24h_pct=5.0)
        sig = from_crypto_signal(crypto)
        # strength = min(1.0, abs(5.0) / 10.0) = 0.5
        assert sig.strength == pytest.approx(0.5)

    def test_coingecko_strength_caps_at_one(self):
        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        crypto = self._make_coingecko_price(change_24h_pct=15.0)
        sig = from_crypto_signal(crypto)
        assert sig.strength == pytest.approx(1.0)

    def test_coingecko_metadata_has_coin_id(self):
        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        crypto = self._make_coingecko_price(change_24h_pct=5.0)
        sig = from_crypto_signal(crypto)
        assert sig.metadata.get("coin_id") == "bitcoin"

    def test_coincap_source_field(self):
        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        asset = self._make_coincap_asset(change_24h_pct=5.0)
        sig = from_crypto_signal(asset)
        assert sig.source == "crypto_coincap"

    def test_coincap_domain_and_asset_class(self):
        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        asset = self._make_coincap_asset(change_24h_pct=5.0)
        sig = from_crypto_signal(asset)
        assert sig.domain == "crypto"
        assert sig.asset_class == "crypto"

    def test_coincap_bullish_direction(self):
        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        asset = self._make_coincap_asset(change_24h_pct=4.0)
        sig = from_crypto_signal(asset)
        assert sig.direction == "bullish"

    def test_coincap_metadata_has_asset_id(self):
        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        asset = self._make_coincap_asset(change_24h_pct=5.0)
        sig = from_crypto_signal(asset)
        assert sig.metadata.get("asset_id") == "bitcoin"

    def test_outcome_symbol_format(self):
        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        crypto = self._make_coingecko_price(change_24h_pct=5.0)
        sig = from_crypto_signal(crypto)
        # outcome_symbol should be SYMBOL-USD
        assert sig.outcome_symbol == "BTC-USD"


# ============================================================================
# 2. Insider Adapters (OpenInsider)
# ============================================================================

class TestInsiderAdapters:
    """Tests for from_openinsider."""

    def _make_purchase(self, ticker="AAPL", value=250_000.0):
        m = MagicMock()
        m.filing_date = "2026-03-01"
        m.trade_date = "2026-02-28"
        m.ticker = ticker
        m.company_name = "Apple Inc."
        m.insider_name = "Tim Cook"
        m.title = "CEO"
        m.trade_type = "P - Purchase"
        m.price = 175.50
        m.quantity = 1424
        m.owned = 10_000
        m.delta_owned_pct = 16.6
        m.value = value
        return m

    def test_direction_always_bullish(self):
        from mae_core.market.signal_adapters.wave2_3 import from_openinsider
        p = self._make_purchase()
        sig = from_openinsider(p)
        assert sig.direction == "bullish"

    def test_source_field(self):
        from mae_core.market.signal_adapters.wave2_3 import from_openinsider
        p = self._make_purchase()
        sig = from_openinsider(p)
        assert sig.source == "openinsider_purchase"

    def test_domain(self):
        from mae_core.market.signal_adapters.wave2_3 import from_openinsider
        p = self._make_purchase()
        sig = from_openinsider(p)
        assert sig.domain == "insider"

    def test_strength_250k(self):
        from mae_core.market.signal_adapters.wave2_3 import from_openinsider
        p = self._make_purchase(value=250_000.0)
        sig = from_openinsider(p)
        # strength = min(1.0, 250_000 / 500_000) = 0.5
        assert sig.strength == pytest.approx(0.5)

    def test_strength_1m_caps_at_one(self):
        from mae_core.market.signal_adapters.wave2_3 import from_openinsider
        p = self._make_purchase(value=1_000_000.0)
        sig = from_openinsider(p)
        assert sig.strength == pytest.approx(1.0)

    def test_symbol_matches_ticker(self):
        from mae_core.market.signal_adapters.wave2_3 import from_openinsider
        p = self._make_purchase(ticker="AAPL")
        sig = from_openinsider(p)
        assert sig.symbol == "AAPL"

    def test_metadata_contains_value(self):
        from mae_core.market.signal_adapters.wave2_3 import from_openinsider
        p = self._make_purchase(value=500_000.0)
        sig = from_openinsider(p)
        assert sig.metadata["value"] == pytest.approx(500_000.0)


# ============================================================================
# 3. Institutional Adapters (13F, Activist, Short Squeeze)
# ============================================================================

class TestInstitutionalAdapters:
    """Tests for from_13f_holding, from_activist_filing, from_finviz_short_squeeze."""

    def _make_holding(self, ticker="MSFT", value_usd=5000.0):
        """value_usd is in thousands (SEC convention)."""
        m = MagicMock()
        m.filer_name = "Vanguard Group"
        m.filer_cik = "0000102909"
        m.ticker = ticker
        m.company_name = "Microsoft Corp"
        m.shares = 1_000_000
        m.value_usd = value_usd  # thousands
        m.filing_date = "2026-02-15"
        m.period_of_report = "2025-12-31"
        return m

    def _make_activist_filing(self, subject_ticker="TSLA"):
        m = MagicMock()
        m.filer_name = "Activist Fund LP"
        m.filer_cik = "0001234567"
        m.subject_company = "Tesla Inc"
        m.subject_ticker = subject_ticker
        m.filing_date = "2026-02-20"
        m.form_type = "SC 13D"
        m.percent_owned = 5.5
        m.purpose = "Activist position (>5% ownership)"
        return m

    def _make_short_squeeze(self, ticker="AMC", short_float_pct=30.0):
        m = MagicMock()
        m.ticker = ticker
        m.company = "AMC Entertainment"
        m.sector = "Entertainment"
        m.price = 4.25
        m.short_float_pct = short_float_pct
        m.short_ratio = 3.5
        m.float_shares = 500_000_000
        m.short_interest = 150_000_000
        return m

    # ---- 13F tests ----

    def test_13f_source(self):
        from mae_core.market.signal_adapters.wave2_3 import from_13f_holding
        h = self._make_holding()
        sig = from_13f_holding(h)
        assert sig.source == "institutional_13f"

    def test_13f_domain(self):
        from mae_core.market.signal_adapters.wave2_3 import from_13f_holding
        h = self._make_holding()
        sig = from_13f_holding(h)
        assert sig.domain == "institutional"

    def test_13f_direction_always_bullish(self):
        from mae_core.market.signal_adapters.wave2_3 import from_13f_holding
        h = self._make_holding()
        sig = from_13f_holding(h)
        assert sig.direction == "bullish"

    def test_13f_strength_5m_position(self):
        from mae_core.market.signal_adapters.wave2_3 import from_13f_holding
        # value_usd=5000 thousands => actual $5M. strength = 5M/10M = 0.5
        h = self._make_holding(value_usd=5000.0)
        sig = from_13f_holding(h)
        assert sig.strength == pytest.approx(0.5)

    def test_13f_strength_10m_caps_at_one(self):
        from mae_core.market.signal_adapters.wave2_3 import from_13f_holding
        # value_usd=10000 thousands => $10M. strength = 10M/10M = 1.0
        h = self._make_holding(value_usd=10_000.0)
        sig = from_13f_holding(h)
        assert sig.strength == pytest.approx(1.0)

    def test_13f_symbol(self):
        from mae_core.market.signal_adapters.wave2_3 import from_13f_holding
        h = self._make_holding(ticker="MSFT")
        sig = from_13f_holding(h)
        assert sig.symbol == "MSFT"

    # ---- Activist 13D tests ----

    def test_activist_source(self):
        from mae_core.market.signal_adapters.wave2_3 import from_activist_filing
        f = self._make_activist_filing()
        sig = from_activist_filing(f)
        assert sig.source == "activist_13d"

    def test_activist_domain(self):
        from mae_core.market.signal_adapters.wave2_3 import from_activist_filing
        f = self._make_activist_filing()
        sig = from_activist_filing(f)
        assert sig.domain == "institutional"

    def test_activist_fixed_strength(self):
        from mae_core.market.signal_adapters.wave2_3 import from_activist_filing
        f = self._make_activist_filing()
        sig = from_activist_filing(f)
        assert sig.strength == pytest.approx(0.85)

    def test_activist_fixed_confidence(self):
        from mae_core.market.signal_adapters.wave2_3 import from_activist_filing
        f = self._make_activist_filing()
        sig = from_activist_filing(f)
        assert sig.confidence == pytest.approx(0.60)

    def test_activist_symbol(self):
        from mae_core.market.signal_adapters.wave2_3 import from_activist_filing
        f = self._make_activist_filing(subject_ticker="TSLA")
        sig = from_activist_filing(f)
        assert sig.symbol == "TSLA"

    # ---- Short squeeze tests ----

    def test_short_squeeze_source(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finviz_short_squeeze
        sq = self._make_short_squeeze(short_float_pct=30.0)
        sig = from_finviz_short_squeeze(sq)
        assert sig.source == "finviz_short_squeeze"

    def test_short_squeeze_domain(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finviz_short_squeeze
        sq = self._make_short_squeeze()
        sig = from_finviz_short_squeeze(sq)
        assert sig.domain == "institutional"

    def test_short_squeeze_direction_always_bullish(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finviz_short_squeeze
        sq = self._make_short_squeeze()
        sig = from_finviz_short_squeeze(sq)
        assert sig.direction == "bullish"

    def test_short_squeeze_strength_30pct(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finviz_short_squeeze
        # strength = min(1.0, 30 / 40) = 0.75
        sq = self._make_short_squeeze(short_float_pct=30.0)
        sig = from_finviz_short_squeeze(sq)
        assert sig.strength == pytest.approx(0.75)

    def test_short_squeeze_strength_caps_at_one(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finviz_short_squeeze
        sq = self._make_short_squeeze(short_float_pct=60.0)
        sig = from_finviz_short_squeeze(sq)
        assert sig.strength == pytest.approx(1.0)


# ============================================================================
# 4. Technical Adapters (Unusual Volume, Finnhub Realtime)
# ============================================================================

class TestTechnicalAdapters:
    """Tests for from_finviz_unusual_volume and from_finnhub_realtime."""

    def _make_unusual_volume(self, ticker="GME", change_pct=3.0, volume_ratio=3.0):
        m = MagicMock()
        m.ticker = ticker
        m.company = "GameStop Corp"
        m.sector = "Retail"
        m.price = 12.50
        m.change_pct = change_pct
        m.volume = 30_000_000
        m.avg_volume = 10_000_000
        m.volume_ratio = volume_ratio
        return m

    def test_unusual_volume_source(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finviz_unusual_volume
        uv = self._make_unusual_volume()
        sig = from_finviz_unusual_volume(uv)
        assert sig.source == "finviz_unusual_volume"

    def test_unusual_volume_domain(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finviz_unusual_volume
        uv = self._make_unusual_volume()
        sig = from_finviz_unusual_volume(uv)
        assert sig.domain == "technical"

    def test_unusual_volume_bullish_direction(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finviz_unusual_volume
        uv = self._make_unusual_volume(change_pct=3.0)
        sig = from_finviz_unusual_volume(uv)
        assert sig.direction == "bullish"

    def test_unusual_volume_bearish_direction(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finviz_unusual_volume
        uv = self._make_unusual_volume(change_pct=-3.0)
        sig = from_finviz_unusual_volume(uv)
        assert sig.direction == "bearish"

    def test_unusual_volume_neutral_direction(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finviz_unusual_volume
        uv = self._make_unusual_volume(change_pct=1.0)
        sig = from_finviz_unusual_volume(uv)
        assert sig.direction == "neutral"

    def test_unusual_volume_strength_3x(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finviz_unusual_volume
        # volume_ratio=3.0 → strength = min(1.0, (3.0 - 1) / 3) ≈ 0.667
        uv = self._make_unusual_volume(volume_ratio=3.0)
        sig = from_finviz_unusual_volume(uv)
        assert sig.strength == pytest.approx((3.0 - 1.0) / 3.0)

    def test_unusual_volume_strength_caps_at_one(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finviz_unusual_volume
        uv = self._make_unusual_volume(volume_ratio=10.0)
        sig = from_finviz_unusual_volume(uv)
        assert sig.strength == pytest.approx(1.0)

    # ---- Finnhub realtime tests ----

    def test_finnhub_realtime_volume_spike_source(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finnhub_realtime
        d = {
            "signal_type": "volume_spike",
            "symbol": "AAPL",
            "price": 185.0,
            "value": 100.0,
            "threshold": 40.0,
            "direction": "bullish",
            "timestamp": datetime(2026, 3, 2, 10, 30, 0),
        }
        sig = from_finnhub_realtime(d)
        assert sig.source == "finnhub_realtime"

    def test_finnhub_realtime_volume_spike_direction(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finnhub_realtime
        d = {
            "signal_type": "volume_spike",
            "symbol": "AAPL",
            "value": 100.0,
            "threshold": 40.0,
            "direction": "bullish",
        }
        sig = from_finnhub_realtime(d)
        assert sig.direction == "bullish"

    def test_finnhub_realtime_volume_spike_strength(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finnhub_realtime
        # strength = min(1.0, 100 / (40 * 3)) = min(1.0, 100/120) ≈ 0.833
        d = {
            "signal_type": "volume_spike",
            "symbol": "AAPL",
            "value": 100.0,
            "threshold": 40.0,
            "direction": "bullish",
        }
        sig = from_finnhub_realtime(d)
        assert sig.strength == pytest.approx(100.0 / (40.0 * 3.0))

    def test_finnhub_realtime_rapid_move_bearish(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finnhub_realtime
        d = {
            "signal_type": "rapid_move",
            "symbol": "TSLA",
            "value": -0.02,
            "threshold": 1.0,
            "direction": "bearish",
        }
        sig = from_finnhub_realtime(d)
        assert sig.direction == "bearish"
        # strength = min(1.0, abs(-0.02) / 0.03) ≈ 0.667
        assert sig.strength == pytest.approx(0.02 / 0.03)

    def test_finnhub_realtime_domain(self):
        from mae_core.market.signal_adapters.wave2_3 import from_finnhub_realtime
        d = {"signal_type": "volume_spike", "symbol": "AAPL", "value": 50.0, "threshold": 20.0, "direction": "bullish"}
        sig = from_finnhub_realtime(d)
        assert sig.domain == "technical"


# ============================================================================
# 5. Suppression Adapter
# ============================================================================

class TestSuppressionAdapter:
    """Tests for from_suppression_event."""

    def _make_event(self, name="FOMC", impact="high"):
        m = MagicMock()
        m.name = name
        m.event_date = date(2026, 3, 18)
        m.impact = impact
        m.description = "Federal Reserve monetary policy decision"
        return m

    def test_direction_always_neutral(self):
        from mae_core.market.signal_adapters.wave2_3 import from_suppression_event
        event = self._make_event()
        sig = from_suppression_event(event)
        assert sig.direction == "neutral"

    def test_source_field(self):
        from mae_core.market.signal_adapters.wave2_3 import from_suppression_event
        event = self._make_event()
        sig = from_suppression_event(event)
        assert sig.source == "economic_calendar"

    def test_metadata_suppress_true(self):
        from mae_core.market.signal_adapters.wave2_3 import from_suppression_event
        event = self._make_event()
        sig = from_suppression_event(event)
        assert sig.metadata.get("suppress") is True

    def test_strength_high_impact(self):
        from mae_core.market.signal_adapters.wave2_3 import from_suppression_event
        event = self._make_event(impact="high")
        sig = from_suppression_event(event)
        assert sig.strength == pytest.approx(0.8)

    def test_strength_medium_impact(self):
        from mae_core.market.signal_adapters.wave2_3 import from_suppression_event
        event = self._make_event(impact="medium")
        sig = from_suppression_event(event)
        assert sig.strength == pytest.approx(0.5)

    def test_strength_low_impact(self):
        from mae_core.market.signal_adapters.wave2_3 import from_suppression_event
        event = self._make_event(impact="low")
        sig = from_suppression_event(event)
        assert sig.strength == pytest.approx(0.3)

    def test_metadata_impact_preserved(self):
        from mae_core.market.signal_adapters.wave2_3 import from_suppression_event
        event = self._make_event(impact="high")
        sig = from_suppression_event(event)
        assert sig.metadata["impact"] == "high"

    def test_metadata_name_preserved(self):
        from mae_core.market.signal_adapters.wave2_3 import from_suppression_event
        event = self._make_event(name="FOMC")
        sig = from_suppression_event(event)
        assert sig.metadata["name"] == "FOMC"


# ============================================================================
# 6. Fetch Functions
# ============================================================================

class TestFetchFunctions:
    """Tests for Wave 2+3 fetch functions in sensing_fetchers.py."""

    # ---- fetch_crypto_prices ----

    def test_crypto_prices_none_client_returns_empty(self):
        from mae_core.market.sensing_fetchers import fetch_crypto_prices
        result = fetch_crypto_prices(None, MagicMock())
        assert result == []

    def test_crypto_prices_calls_get_prices(self):
        from mae_core.market.sensing_fetchers import fetch_crypto_prices
        client = MagicMock()
        crypto_obj = MagicMock()
        crypto_obj.coin_id = "bitcoin"
        crypto_obj.symbol = "BTC"
        crypto_obj.price_usd = 50000.0
        crypto_obj.volume_24h = 1e9
        crypto_obj.change_24h_pct = 5.0
        crypto_obj.change_7d_pct = 10.0
        crypto_obj.market_cap = 1e12
        crypto_obj.last_updated = "2026-03-02T00:00:00Z"
        client.get_prices.return_value = [crypto_obj]

        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        result = fetch_crypto_prices(client, from_crypto_signal)
        assert len(result) == 1
        client.get_prices.assert_called_once()

    def test_crypto_prices_exception_returns_empty(self):
        from mae_core.market.sensing_fetchers import fetch_crypto_prices
        client = MagicMock()
        client.get_prices.side_effect = RuntimeError("network error")
        result = fetch_crypto_prices(client, MagicMock())
        assert result == []

    # ---- fetch_crypto_exchange ----

    def test_crypto_exchange_none_client_returns_empty(self):
        from mae_core.market.sensing_fetchers import fetch_crypto_exchange
        result = fetch_crypto_exchange(None, MagicMock())
        assert result == []

    def test_crypto_exchange_calls_get_assets(self):
        from mae_core.market.sensing_fetchers import fetch_crypto_exchange
        client = MagicMock()
        asset = MagicMock(spec=[
            "asset_id", "symbol", "name", "rank", "price_usd",
            "volume_24h_usd", "change_24h_pct", "market_cap_usd",
            "supply", "max_supply", "vwap_24h",
        ])
        asset.asset_id = "bitcoin"
        asset.symbol = "BTC"
        asset.price_usd = 50000.0
        asset.volume_24h_usd = 1e9
        asset.change_24h_pct = 5.0
        asset.market_cap_usd = 1e12
        asset.rank = 1
        asset.supply = 19e6
        asset.max_supply = 21e6
        asset.vwap_24h = 49500.0
        client.get_assets.return_value = [asset]

        from mae_core.market.signal_adapters.wave2_3 import from_crypto_signal
        result = fetch_crypto_exchange(client, from_crypto_signal)
        assert len(result) == 1
        client.get_assets.assert_called_once_with(limit=10)

    def test_crypto_exchange_exception_returns_empty(self):
        from mae_core.market.sensing_fetchers import fetch_crypto_exchange
        client = MagicMock()
        client.get_assets.side_effect = RuntimeError("connection refused")
        result = fetch_crypto_exchange(client, MagicMock())
        assert result == []

    # ---- fetch_openinsider ----

    def test_openinsider_none_client_returns_empty(self):
        from mae_core.market.sensing_fetchers import fetch_openinsider
        result = fetch_openinsider(None, MagicMock())
        assert result == []

    def test_openinsider_calls_get_recent_purchases(self):
        from mae_core.market.sensing_fetchers import fetch_openinsider
        client = MagicMock()
        purchase = MagicMock()
        purchase.ticker = "AAPL"
        purchase.value = 250_000.0
        purchase.filing_date = "2026-03-01"
        purchase.trade_date = "2026-02-28"
        purchase.insider_name = "Tim Cook"
        purchase.title = "CEO"
        purchase.trade_type = "P - Purchase"
        purchase.price = 175.5
        purchase.quantity = 1424
        purchase.owned = 10000
        purchase.delta_owned_pct = 16.6
        purchase.company_name = "Apple Inc."
        client.get_recent_purchases.return_value = [purchase]

        from mae_core.market.signal_adapters.wave2_3 import from_openinsider
        result = fetch_openinsider(client, from_openinsider)
        assert len(result) == 1
        client.get_recent_purchases.assert_called_once_with(days=7)

    def test_openinsider_exception_returns_empty(self):
        from mae_core.market.sensing_fetchers import fetch_openinsider
        client = MagicMock()
        client.get_recent_purchases.side_effect = RuntimeError("scraping failed")
        result = fetch_openinsider(client, MagicMock())
        assert result == []

    # ---- fetch_13f_holdings ----

    def test_13f_none_client_returns_empty(self):
        from mae_core.market.sensing_fetchers import fetch_13f_holdings
        result = fetch_13f_holdings(None, MagicMock(), MagicMock())
        assert result == []

    def test_13f_calls_get_13d_filings(self):
        from mae_core.market.sensing_fetchers import fetch_13f_holdings
        client = MagicMock()
        filing = MagicMock()
        filing.filer_name = "Activist Fund"
        filing.filer_cik = "0001234567"
        filing.subject_company = "Tesla Inc"
        filing.subject_ticker = "TSLA"
        filing.filing_date = "2026-02-20"
        filing.form_type = "SC 13D"
        filing.percent_owned = 5.5
        filing.purpose = "Activist position"
        client.get_13d_filings.return_value = [filing]

        from mae_core.market.signal_adapters.wave2_3 import from_activist_filing
        # 13f fetch uses activist_converter for 13D filings
        result = fetch_13f_holdings(client, MagicMock(), from_activist_filing)
        assert len(result) == 1
        client.get_13d_filings.assert_called_once_with(days=30)

    def test_13f_exception_returns_empty(self):
        from mae_core.market.sensing_fetchers import fetch_13f_holdings
        client = MagicMock()
        client.get_13d_filings.side_effect = RuntimeError("SEC API down")
        result = fetch_13f_holdings(client, MagicMock(), MagicMock())
        assert result == []

    # ---- fetch_finviz ----

    def test_finviz_none_client_returns_empty(self):
        from mae_core.market.sensing_fetchers import fetch_finviz
        result = fetch_finviz(None, MagicMock(), MagicMock(), MagicMock())
        assert result == []

    def test_finviz_calls_get_unusual_volume_and_short_float(self):
        from mae_core.market.sensing_fetchers import fetch_finviz
        client = MagicMock()
        uv = MagicMock()
        uv.ticker = "GME"
        uv.company = "GameStop"
        uv.sector = "Retail"
        uv.price = 12.5
        uv.change_pct = 3.0
        uv.volume = 30_000_000
        uv.avg_volume = 10_000_000
        uv.volume_ratio = 3.0
        client.get_unusual_volume.return_value = [uv]
        client.get_high_short_float.return_value = []

        volume_converter = MagicMock(return_value=MagicMock())
        squeeze_converter = MagicMock(return_value=MagicMock())
        insider_converter = MagicMock(return_value=MagicMock())
        client.get_insider_trades.return_value = []
        result = fetch_finviz(client, volume_converter, squeeze_converter, insider_converter)
        assert len(result) == 1
        client.get_unusual_volume.assert_called_once()
        client.get_high_short_float.assert_called_once_with(min_pct=20)

    def test_finviz_volume_exception_continues_to_squeeze(self):
        from mae_core.market.sensing_fetchers import fetch_finviz
        client = MagicMock()
        client.get_unusual_volume.side_effect = RuntimeError("FinViz down")
        client.get_high_short_float.return_value = []
        # Should not propagate exception; squeeze path still runs
        client.get_insider_trades.return_value = []
        result = fetch_finviz(client, MagicMock(), MagicMock(), MagicMock())
        assert isinstance(result, list)
        client.get_high_short_float.assert_called_once()

    # ---- fetch_economic_calendar ----

    def test_economic_calendar_none_client_returns_empty(self):
        from mae_core.market.sensing_fetchers import fetch_economic_calendar
        result = fetch_economic_calendar(None, MagicMock())
        assert result == []

    def test_economic_calendar_calls_get_upcoming_events(self):
        from mae_core.market.sensing_fetchers import fetch_economic_calendar
        client = MagicMock()
        event = MagicMock()
        event.name = "FOMC"
        event.event_date = date(2026, 3, 18)
        event.impact = "high"
        event.description = "Fed decision"
        client.get_upcoming_events.return_value = [event]

        from mae_core.market.signal_adapters.wave2_3 import from_suppression_event
        result = fetch_economic_calendar(client, from_suppression_event)
        assert len(result) == 1
        client.get_upcoming_events.assert_called_once_with(days=7)

    def test_economic_calendar_filters_non_high_impact(self):
        from mae_core.market.sensing_fetchers import fetch_economic_calendar
        client = MagicMock()
        low_event = MagicMock()
        low_event.impact = "low"
        client.get_upcoming_events.return_value = [low_event]

        converter = MagicMock()
        result = fetch_economic_calendar(client, converter)
        # Low-impact event should be filtered out
        assert result == []
        converter.assert_not_called()

    def test_economic_calendar_exception_returns_empty(self):
        from mae_core.market.sensing_fetchers import fetch_economic_calendar
        client = MagicMock()
        client.get_upcoming_events.side_effect = RuntimeError("calendar unavailable")
        result = fetch_economic_calendar(client, MagicMock())
        assert result == []


# ============================================================================
# 7. Sensing Hook Integration
# ============================================================================

class TestSensingHookIntegration:
    """Tests for SOURCE_ROTATION, TIER_ROUTING, _ROTATION_TO_THOMPSON constants."""

    def test_source_rotation_has_34_entries(self):
        from mae_core.market.sensing_hook import SOURCE_ROTATION
        assert len(SOURCE_ROTATION) == 34

    def test_crypto_prices_in_source_rotation(self):
        from mae_core.market.sensing_hook import SOURCE_ROTATION
        assert "crypto_prices" in SOURCE_ROTATION

    def test_crypto_exchange_in_source_rotation(self):
        from mae_core.market.sensing_hook import SOURCE_ROTATION
        assert "crypto_exchange" in SOURCE_ROTATION

    def test_openinsider_in_source_rotation(self):
        from mae_core.market.sensing_hook import SOURCE_ROTATION
        assert "openinsider" in SOURCE_ROTATION

    def test_institutional_13f_in_source_rotation(self):
        from mae_core.market.sensing_hook import SOURCE_ROTATION
        assert "institutional_13f" in SOURCE_ROTATION

    def test_finviz_in_source_rotation(self):
        from mae_core.market.sensing_hook import SOURCE_ROTATION
        assert "finviz" in SOURCE_ROTATION

    def test_economic_calendar_in_source_rotation(self):
        from mae_core.market.sensing_hook import SOURCE_ROTATION
        assert "economic_calendar" in SOURCE_ROTATION

    def test_crypto_coingecko_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert "crypto_coingecko" in TIER_ROUTING
        assert TIER_ROUTING["crypto_coingecko"] == "thematic"

    def test_crypto_coincap_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert "crypto_coincap" in TIER_ROUTING
        assert TIER_ROUTING["crypto_coincap"] == "thematic"

    def test_openinsider_purchase_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert "openinsider_purchase" in TIER_ROUTING
        assert TIER_ROUTING["openinsider_purchase"] == "tactical"

    def test_institutional_13f_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert "institutional_13f" in TIER_ROUTING
        assert TIER_ROUTING["institutional_13f"] == "strategic"

    def test_activist_13d_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert "activist_13d" in TIER_ROUTING
        assert TIER_ROUTING["activist_13d"] == "strategic"

    def test_finviz_unusual_volume_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert "finviz_unusual_volume" in TIER_ROUTING
        assert TIER_ROUTING["finviz_unusual_volume"] == "tactical"

    def test_finviz_short_squeeze_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert "finviz_short_squeeze" in TIER_ROUTING
        assert TIER_ROUTING["finviz_short_squeeze"] == "strategic"

    def test_economic_calendar_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert "economic_calendar" in TIER_ROUTING
        assert TIER_ROUTING["economic_calendar"] == "thematic"

    def test_finnhub_realtime_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert "finnhub_realtime" in TIER_ROUTING
        assert TIER_ROUTING["finnhub_realtime"] == "tactical"

    def test_crypto_prices_in_rotation_to_thompson(self):
        from mae_core.market.sensing_hook import _ROTATION_TO_THOMPSON
        assert "crypto_prices" in _ROTATION_TO_THOMPSON
        assert _ROTATION_TO_THOMPSON["crypto_prices"] == "crypto_coingecko"

    def test_crypto_exchange_in_rotation_to_thompson(self):
        from mae_core.market.sensing_hook import _ROTATION_TO_THOMPSON
        assert "crypto_exchange" in _ROTATION_TO_THOMPSON
        assert _ROTATION_TO_THOMPSON["crypto_exchange"] == "crypto_coincap"

    def test_openinsider_in_rotation_to_thompson(self):
        from mae_core.market.sensing_hook import _ROTATION_TO_THOMPSON
        assert "openinsider" in _ROTATION_TO_THOMPSON
        assert _ROTATION_TO_THOMPSON["openinsider"] == "openinsider_purchase"

    def test_institutional_13f_in_rotation_to_thompson(self):
        from mae_core.market.sensing_hook import _ROTATION_TO_THOMPSON
        assert "institutional_13f" in _ROTATION_TO_THOMPSON
        assert _ROTATION_TO_THOMPSON["institutional_13f"] == "institutional_13f"

    def test_finviz_in_rotation_to_thompson(self):
        from mae_core.market.sensing_hook import _ROTATION_TO_THOMPSON
        assert "finviz" in _ROTATION_TO_THOMPSON
        assert _ROTATION_TO_THOMPSON["finviz"] == "finviz_unusual_volume"

    def test_market_sensing_hook_accepts_new_params(self):
        """MarketSensingHook instantiation with new Wave 2+3 clients should not error."""
        from mae_core.market.sensing_hook import MarketSensingHook
        hook = MarketSensingHook(
            coingecko_client=MagicMock(),
            coincap_client=MagicMock(),
            openinsider_client=MagicMock(),
            edgar_enhanced_client=MagicMock(),
            finviz_client=MagicMock(),
            economic_calendar_client=MagicMock(),
            finnhub_websocket=MagicMock(),
        )
        assert hook is not None
        hook.shutdown()


# ============================================================================
# 8. Market Clock Availability
# ============================================================================

class TestMarketClockAvailability:
    """Tests for ALWAYS_AVAILABLE, MARKET_HOURS, PERIODIC sets."""

    def test_crypto_prices_in_always_available(self):
        from mae_core.market.market_clock import ALWAYS_AVAILABLE
        assert "crypto_prices" in ALWAYS_AVAILABLE

    def test_crypto_exchange_in_always_available(self):
        from mae_core.market.market_clock import ALWAYS_AVAILABLE
        assert "crypto_exchange" in ALWAYS_AVAILABLE

    def test_openinsider_in_always_available(self):
        from mae_core.market.market_clock import ALWAYS_AVAILABLE
        assert "openinsider" in ALWAYS_AVAILABLE

    def test_economic_calendar_in_always_available(self):
        from mae_core.market.market_clock import ALWAYS_AVAILABLE
        assert "economic_calendar" in ALWAYS_AVAILABLE

    def test_finviz_in_market_hours(self):
        from mae_core.market.market_clock import MARKET_HOURS
        assert "finviz" in MARKET_HOURS

    def test_institutional_13f_in_periodic(self):
        from mae_core.market.market_clock import PERIODIC
        assert "institutional_13f" in PERIODIC


# ============================================================================
# 9. Thompson Keys
# ============================================================================

class TestThompsonKeys:
    """Tests for Wave 2+3 Thompson keys in learning_config.py."""

    def _get_reliability(self):
        from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
        return LEARNING_CONFIG["source_reliability"]

    def test_finnhub_realtime_key_exists(self):
        assert "finnhub_realtime" in self._get_reliability()

    def test_crypto_coingecko_key_exists(self):
        assert "crypto_coingecko" in self._get_reliability()

    def test_crypto_coincap_key_exists(self):
        assert "crypto_coincap" in self._get_reliability()

    def test_openinsider_purchase_key_exists(self):
        assert "openinsider_purchase" in self._get_reliability()

    def test_institutional_13f_key_exists(self):
        assert "institutional_13f" in self._get_reliability()

    def test_activist_13d_key_exists(self):
        assert "activist_13d" in self._get_reliability()

    def test_finviz_unusual_volume_key_exists(self):
        assert "finviz_unusual_volume" in self._get_reliability()

    def test_finviz_short_squeeze_key_exists(self):
        assert "finviz_short_squeeze" in self._get_reliability()

    def test_economic_calendar_key_exists(self):
        assert "economic_calendar" in self._get_reliability()

    def test_crypto_decay_rate_key_exists(self):
        from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
        assert "crypto" in LEARNING_CONFIG.get("decay_rates", {})


# ============================================================================
# 10. Economic Calendar Suppression Integration
# ============================================================================

class TestSuppressionIntegration:
    """Test that EconomicCalendarClient's suppression window query works end-to-end."""

    def test_is_in_suppression_window_fomc_date(self):
        """During an FOMC window the client returns True."""
        from mae_core.market.apis.economic_calendar_client import EconomicCalendarClient
        client = EconomicCalendarClient(pre_event_hours=24.0, post_event_hours=4.0)
        # FOMC March 18 2026 at 2:00 PM ET — check inside window
        check_time = datetime(2026, 3, 18, 14, 30)  # 30 min after announcement
        assert client.is_in_suppression_window(at=check_time) is True

    def test_is_not_in_suppression_window_outside_fomc(self):
        """Far from any event, suppression window is False."""
        from mae_core.market.apis.economic_calendar_client import EconomicCalendarClient
        client = EconomicCalendarClient(pre_event_hours=24.0, post_event_hours=4.0)
        # Well between events — pick a mid-April weekday
        check_time = datetime(2026, 4, 22, 12, 0)
        assert client.is_in_suppression_window(at=check_time) is False

    def test_suppression_window_pre_event(self):
        """Pre-event window (24h before FOMC) also triggers suppression."""
        from mae_core.market.apis.economic_calendar_client import EconomicCalendarClient
        client = EconomicCalendarClient(pre_event_hours=24.0, post_event_hours=4.0)
        # 12 hours before FOMC March 18 2026 2PM → March 18 2AM
        check_time = datetime(2026, 3, 18, 2, 0)
        assert client.is_in_suppression_window(at=check_time) is True

    def test_get_upcoming_events_count(self):
        """Calendar should have events within known date ranges."""
        from mae_core.market.apis.economic_calendar_client import EconomicCalendarClient
        client = EconomicCalendarClient()
        # March 2026 has both CPI (March 11) and FOMC (March 18 and 19)
        events = client.get_upcoming_events(days=30, from_date=date(2026, 3, 1))
        assert len(events) >= 3

    def test_from_suppression_event_confidence_is_high(self):
        """Suppression signals have high confidence (events are scheduled)."""
        from mae_core.market.signal_adapters.wave2_3 import from_suppression_event
        from mae_core.market.apis.economic_calendar_client import EconomicEvent
        event = EconomicEvent(
            name="FOMC",
            event_date=date(2026, 3, 18),
            impact="high",
            description="Federal Reserve monetary policy decision",
        )
        sig = from_suppression_event(event)
        assert sig.confidence == pytest.approx(0.70)


# ============================================================================
# 11. WebSocket Integration
# ============================================================================

class TestWebSocketIntegration:
    """Tests for Finnhub WebSocket realtime signal processing in MarketSensingHook."""

    def test_process_realtime_signals_is_callable(self):
        from mae_core.market.sensing_hook import MarketSensingHook
        hook = MarketSensingHook()
        assert callable(hook._process_realtime_signals)
        hook.shutdown()

    def test_process_realtime_signals_feeds_convergence_alerter(self):
        from mae_core.market.sensing_hook import MarketSensingHook
        from mae_core.market.signal_adapters.wave2_3 import from_finnhub_realtime

        alerter = MagicMock()
        hook = MarketSensingHook(convergence_alerter=alerter)

        signal_dict = {
            "signal_type": "volume_spike",
            "symbol": "AAPL",
            "price": 185.0,
            "value": 120.0,
            "threshold": 40.0,
            "direction": "bullish",
            "timestamp": datetime(2026, 3, 2, 10, 30, 0),
        }
        market_signal = from_finnhub_realtime(signal_dict)
        hook._process_realtime_signals([market_signal])

        alerter.record_signal.assert_called_once()
        hook.shutdown()

    def test_process_realtime_signals_increments_total_fed(self):
        from mae_core.market.sensing_hook import MarketSensingHook
        from mae_core.market.signal_adapters.wave2_3 import from_finnhub_realtime

        hook = MarketSensingHook()
        assert hook._total_signals_fed == 0

        sig_dict = {
            "signal_type": "rapid_move",
            "symbol": "TSLA",
            "value": -0.025,
            "threshold": 1.0,
            "direction": "bearish",
        }
        market_signal = from_finnhub_realtime(sig_dict)
        hook._process_realtime_signals([market_signal])

        assert hook._total_signals_fed == 1
        hook.shutdown()

    def test_step_collects_websocket_signals_when_available(self):
        """step() calls get_pending_signals() on the websocket every step."""
        from mae_core.market.sensing_hook import MarketSensingHook

        ws = MagicMock()
        ws.get_pending_signals.return_value = []  # No signals this step

        hook = MarketSensingHook(finnhub_websocket=ws)
        hook.step()

        ws.get_pending_signals.assert_called_once()
        hook.shutdown()

    def test_step_processes_websocket_realtime_signals(self):
        """step() converts and processes WebSocket signal dicts."""
        from mae_core.market.sensing_hook import MarketSensingHook

        alerter = MagicMock()
        ws = MagicMock()
        ws.get_pending_signals.return_value = [
            {
                "signal_type": "volume_spike",
                "symbol": "SPY",
                "value": 80.0,
                "threshold": 30.0,
                "direction": "bullish",
                "timestamp": datetime(2026, 3, 2, 10, 0, 0),
            }
        ]

        hook = MarketSensingHook(finnhub_websocket=ws, convergence_alerter=alerter)
        hook.step()

        # Alerter should have been called with the processed signal
        assert alerter.record_signal.call_count >= 1
        hook.shutdown()
