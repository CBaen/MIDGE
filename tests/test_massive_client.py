"""Tests for Massive/Polygon.io client and signal adapter."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.apis.massive_client import (
    MassiveClient,
    TickerBar,
    TickerSnapshot,
)
from mae_core.market.signal_adapters.wave2_3 import from_massive_snapshot


# ---------------------------------------------------------------------------
# MassiveClient unit tests
# ---------------------------------------------------------------------------

class TestMassiveClient:
    """Test MassiveClient construction and data parsing."""

    def test_init_defaults(self):
        client = MassiveClient(api_key="test_key")
        assert client._api_key == "test_key"
        assert len(client._call_times) == 0
        assert len(client._grouped_cache) == 0

    def test_init_env_var(self, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "env_key")
        client = MassiveClient()
        assert client._api_key == "env_key"

    def test_init_no_key(self, monkeypatch):
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        client = MassiveClient()
        assert client._api_key == ""

    def test_last_business_day_weekday(self):
        # Thursday → Wednesday
        d = MassiveClient._last_business_day(date(2026, 3, 5))  # Thursday
        assert d.weekday() < 5  # Must be a weekday
        assert d <= date(2026, 3, 5)

    def test_last_business_day_saturday(self):
        d = MassiveClient._last_business_day(date(2026, 3, 7))  # Saturday
        assert d == date(2026, 3, 6)  # Friday

    def test_last_business_day_sunday(self):
        d = MassiveClient._last_business_day(date(2026, 3, 8))  # Sunday
        assert d == date(2026, 3, 6)  # Friday

    def test_last_business_day_monday(self):
        d = MassiveClient._last_business_day(date(2026, 3, 2))  # Monday
        assert d == date(2026, 3, 2)  # Monday itself

    def test_ts_to_date(self):
        # 2026-03-02 00:00:00 UTC in ms
        ts_ms = int(datetime(2026, 3, 2).timestamp() * 1000)
        result = MassiveClient._ts_to_date(ts_ms)
        assert result == "2026-03-02"

    def test_ts_to_date_zero(self):
        assert MassiveClient._ts_to_date(0) == ""

    def test_parse_bars(self):
        results = [
            {"T": "AAPL", "o": 150.0, "h": 155.0, "l": 149.0, "c": 154.0, "v": 1000000, "vw": 152.0, "n": 5000},
            {"T": "MSFT", "o": 300.0, "h": 310.0, "l": 298.0, "c": 308.0, "v": 500000, "vw": 304.0, "n": 3000},
        ]
        bars = MassiveClient._parse_bars(results, "2026-03-02")
        assert len(bars) == 2
        assert bars[0].ticker == "AAPL"
        assert bars[0].close == 154.0
        assert bars[0].volume == 1000000
        assert bars[0].date == "2026-03-02"
        assert bars[1].ticker == "MSFT"

    def test_parse_bars_invalid_entry_skipped(self):
        results = [
            {"T": "AAPL", "o": 150.0, "h": 155.0, "l": 149.0, "c": 154.0, "v": 1000000, "vw": 152.0, "n": 5000},
            {"T": "BAD", "o": "not_a_number"},  # Invalid — float() will raise, try/except skips
        ]
        bars = MassiveClient._parse_bars(results, "2026-03-02")
        assert len(bars) == 1  # Bad entry skipped

    def test_get_no_api_key(self, monkeypatch):
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        client = MassiveClient()
        result = client._get("/test")
        assert result == {}

    @patch("mae_core.market.apis.massive_client.requests.Session")
    def test_get_grouped_daily_success(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"T": "AAPL", "o": 150, "h": 155, "l": 149, "c": 154, "v": 1e6, "vw": 152, "n": 5000},
            ]
        }
        mock_resp.raise_for_status.return_value = None

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        client = MassiveClient(api_key="test")
        client._session = mock_session
        bars = client.get_grouped_daily(date(2026, 3, 2))
        assert len(bars) == 1
        assert bars[0].ticker == "AAPL"

    @patch("mae_core.market.apis.massive_client.requests.Session")
    def test_get_grouped_daily_caches(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"T": "AAPL", "o": 150, "h": 155, "l": 149, "c": 154, "v": 1e6, "vw": 152, "n": 5000}]}
        mock_resp.raise_for_status.return_value = None

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        client = MassiveClient(api_key="test")
        client._session = mock_session
        d = date(2026, 3, 2)
        bars1 = client.get_grouped_daily(d)
        bars2 = client.get_grouped_daily(d)
        # Second call uses cache — session.get only called once
        assert mock_session.get.call_count == 1
        assert len(bars1) == len(bars2)

    @patch("mae_core.market.apis.massive_client.requests.Session")
    def test_get_previous_close(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"T": "AAPL", "o": 150, "h": 155, "l": 149, "c": 154, "v": 1e6, "vw": 152, "n": 5000, "t": 1740873600000}
            ]
        }
        mock_resp.raise_for_status.return_value = None
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        client = MassiveClient(api_key="test")
        client._session = mock_session
        bar = client.get_previous_close("AAPL")
        assert bar is not None
        assert bar.ticker == "AAPL"
        assert bar.close == 154.0

    def test_get_previous_close_no_results(self):
        client = MassiveClient(api_key="test")
        client._get = MagicMock(return_value={"results": []})
        bar = client.get_previous_close("AAPL")
        assert bar is None

    def test_get_ticker_bars(self):
        client = MassiveClient(api_key="test")
        fake_results = []
        for i in range(25):
            fake_results.append({
                "T": "AAPL", "o": 150 + i, "h": 155 + i, "l": 149 + i,
                "c": 154 + i, "v": 1e6 + i * 1000, "vw": 152 + i, "n": 5000,
                "t": int((datetime(2026, 2, 1) + timedelta(days=i)).timestamp() * 1000),
            })
        client._get = MagicMock(return_value={"results": fake_results})
        bars = client.get_ticker_bars("AAPL", days=20)
        assert len(bars) == 20  # Trimmed to requested window

    def test_build_snapshots(self):
        client = MassiveClient(api_key="test")

        # Mock grouped daily for two dates (date-agnostic: first call = today, second = previous)
        _call_count = {"n": 0}

        def mock_grouped(for_date=None):
            _call_count["n"] += 1
            if for_date is not None:
                # Second call (previous day)
                return [TickerBar("AAPL", 148, 150, 147, 149, 800000, 148.5, 4000, "2026-02-27")]
            # First call (today)
            return [TickerBar("AAPL", 150, 155, 149, 154, 1200000, 152, 5000, "2026-03-02")]

        client.get_grouped_daily = mock_grouped

        snapshots = client.build_snapshots(["AAPL"])
        assert len(snapshots) == 1
        snap = snapshots[0]
        assert snap.ticker == "AAPL"
        assert snap.close == 154.0
        assert snap.prev_close == 149.0  # From previous day
        assert abs(snap.daily_return_pct - ((154 - 149) / 149 * 100)) < 0.01
        assert abs(snap.gap_pct - ((150 - 149) / 149 * 100)) < 0.01


# ---------------------------------------------------------------------------
# Signal adapter tests
# ---------------------------------------------------------------------------

class TestFromMassiveSnapshot:
    """Test the signal adapter for Massive/Polygon snapshots."""

    def _make_snapshot(self, **overrides):
        defaults = dict(
            ticker="AAPL", close=154.0, prev_close=150.0, volume=2000000,
            avg_volume_20d=1000000, daily_return_pct=2.67, gap_pct=0.33,
            volume_ratio=2.0, high=155.0, low=149.0, open=150.5, vwap=152.0,
            bar_date="2026-03-02",
        )
        defaults.update(overrides)
        return TickerSnapshot(**defaults)

    def test_basic_conversion(self):
        snap = self._make_snapshot()
        sig = from_massive_snapshot(snap)
        assert sig.source == "massive_snapshot"
        assert sig.symbol == "AAPL"
        assert sig.domain == "technical"
        assert sig.asset_class == "stock"
        assert sig.outcome_symbol == "AAPL"

    def test_bullish_direction(self):
        snap = self._make_snapshot(daily_return_pct=3.0)
        sig = from_massive_snapshot(snap)
        assert sig.direction == "bullish"

    def test_bearish_direction(self):
        snap = self._make_snapshot(daily_return_pct=-2.5)
        sig = from_massive_snapshot(snap)
        assert sig.direction == "bearish"

    def test_neutral_direction(self):
        snap = self._make_snapshot(daily_return_pct=0.3)
        sig = from_massive_snapshot(snap)
        assert sig.direction == "neutral"

    def test_volume_anomaly_signal(self):
        snap = self._make_snapshot(volume_ratio=4.0, daily_return_pct=0.5, gap_pct=0.1)
        sig = from_massive_snapshot(snap)
        assert sig.strength > 0.5
        assert sig.metadata["signal_type"] == "volume_anomaly"

    def test_price_move_signal(self):
        snap = self._make_snapshot(volume_ratio=1.0, daily_return_pct=4.0, gap_pct=0.1)
        sig = from_massive_snapshot(snap)
        assert sig.strength > 0.5
        assert sig.metadata["signal_type"] == "price_move"

    def test_gap_signal(self):
        snap = self._make_snapshot(volume_ratio=1.0, daily_return_pct=0.5, gap_pct=2.0)
        sig = from_massive_snapshot(snap)
        assert sig.strength > 0.5
        assert sig.metadata["signal_type"] == "gap"

    def test_sub_threshold_still_creates_signal(self):
        snap = self._make_snapshot(volume_ratio=1.0, daily_return_pct=0.3, gap_pct=0.1)
        sig = from_massive_snapshot(snap)
        assert sig.strength == 0.1  # Minimum strength

    def test_metadata_populated(self):
        snap = self._make_snapshot()
        sig = from_massive_snapshot(snap)
        assert sig.metadata["close"] == 154.0
        assert sig.metadata["prev_close"] == 150.0
        assert sig.metadata["volume"] == 2000000
        assert sig.metadata["data_source"] == "massive_polygon"

    def test_signal_id_format(self):
        snap = self._make_snapshot()
        sig = from_massive_snapshot(snap)
        assert sig.signal_id.startswith("massive:")
        assert "AAPL" in sig.signal_id

    def test_outcome_window(self):
        snap = self._make_snapshot()
        sig = from_massive_snapshot(snap)
        assert sig.outcome_window_days == 5

    def test_decay_rate(self):
        snap = self._make_snapshot()
        sig = from_massive_snapshot(snap)
        assert sig.decay_rate == 0.30


# ---------------------------------------------------------------------------
# Fetcher integration tests
# ---------------------------------------------------------------------------

class TestFetchMassiveSnapshot:
    """Test the sensing fetcher for Massive snapshots."""

    def test_fetch_returns_signals(self):
        from mae_core.market.sensing_fetchers import fetch_massive_snapshot

        mock_client = MagicMock()
        mock_client.build_snapshots.return_value = [
            TickerSnapshot(
                ticker="AAPL", close=154, prev_close=150, volume=2e6,
                avg_volume_20d=1e6, daily_return_pct=2.67, gap_pct=0.33,
                volume_ratio=2.0, high=155, low=149, open=150.5, vwap=152,
                bar_date="2026-03-02",
            ),
        ]
        watchlist = {"tickers": ["AAPL", "MSFT"]}
        signals = fetch_massive_snapshot(mock_client, watchlist, from_massive_snapshot)
        assert len(signals) == 1
        assert signals[0].source == "massive_snapshot"

    def test_fetch_none_client(self):
        from mae_core.market.sensing_fetchers import fetch_massive_snapshot
        signals = fetch_massive_snapshot(None, {"tickers": ["AAPL"]}, from_massive_snapshot)
        assert signals == []

    def test_fetch_empty_watchlist(self):
        from mae_core.market.sensing_fetchers import fetch_massive_snapshot
        mock_client = MagicMock()
        mock_client.build_snapshots.return_value = []
        signals = fetch_massive_snapshot(mock_client, {"tickers": []}, from_massive_snapshot)
        assert signals == []

    def test_fetch_exception_handled(self):
        from mae_core.market.sensing_fetchers import fetch_massive_snapshot
        mock_client = MagicMock()
        mock_client.build_snapshots.side_effect = Exception("API error")
        signals = fetch_massive_snapshot(mock_client, {"tickers": ["AAPL"]}, from_massive_snapshot)
        assert signals == []


# ---------------------------------------------------------------------------
# Sensing hook wiring tests
# ---------------------------------------------------------------------------

class TestSensingHookWiring:
    """Verify Massive is wired into the sensing pipeline."""

    def test_source_in_rotation(self):
        from mae_core.market.sensing_hook import SOURCE_ROTATION
        assert "massive_snapshot" in SOURCE_ROTATION

    def test_source_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert "massive_snapshot" in TIER_ROUTING
        assert TIER_ROUTING["massive_snapshot"] == "tactical"

    def test_source_in_thompson_mapping(self):
        from mae_core.market.sensing_hook import _ROTATION_TO_THOMPSON
        assert "massive_snapshot" in _ROTATION_TO_THOMPSON

    def test_source_in_absence_domains(self):
        from mae_core.market.sensing_hook import _ABSENCE_SOURCE_DOMAINS
        assert "massive_snapshot" in _ABSENCE_SOURCE_DOMAINS

    def test_thompson_key_in_config(self):
        from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
        assert "massive_snapshot" in LEARNING_CONFIG["source_reliability"]

    def test_market_clock_category(self):
        from mae_core.market.market_clock import MARKET_HOURS
        assert "massive_snapshot" in MARKET_HOURS


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------

class TestRateLimiting:
    """Verify rate limiting at 5 calls/min."""

    def test_rate_limit_no_sleep_under_limit(self):
        client = MassiveClient(api_key="test")
        # 4 calls should not trigger sleep
        for _ in range(4):
            client._call_times.append(datetime.now().timestamp())
        # 5th call should not sleep if within window
        import time
        start = time.time()
        client._rate_limit()
        elapsed = time.time() - start
        # Should complete quickly (no forced sleep since we didn't exceed 5)
        assert elapsed < 1.0


# ---------------------------------------------------------------------------
# get_ticker_details tests
# ---------------------------------------------------------------------------

from mae_core.market.apis.massive_client import TickerDetails


class TestGetTickerDetails:
    """Tests for MassiveClient.get_ticker_details() and TickerDetails dataclass."""

    _SAMPLE_RESPONSE = {
        "results": {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "market": "stocks",
            "locale": "us",
            "primary_exchange": "XNAS",
            "type": "CS",
            "active": True,
            "currency_name": "usd",
            "market_cap": 3_000_000_000_000.0,
            "sic_code": "3571",
            "sic_description": "Electronic Computers",
        }
    }

    def test_ticker_details_dataclass(self):
        td = TickerDetails(
            ticker="AAPL", name="Apple Inc.", market="stocks", locale="us",
            primary_exchange="XNAS", type="CS", active=True,
            currency_name="usd", market_cap=3e12,
            sic_code="3571", sic_description="Electronic Computers",
        )
        assert td.ticker == "AAPL"
        assert td.name == "Apple Inc."
        assert td.market == "stocks"
        assert td.primary_exchange == "XNAS"
        assert td.type == "CS"
        assert td.active is True
        assert td.market_cap == pytest.approx(3e12)
        assert td.sic_code == "3571"
        assert td.sic_description == "Electronic Computers"

    def test_get_ticker_details_success(self):
        client = MassiveClient(api_key="test")
        client._get = MagicMock(return_value=self._SAMPLE_RESPONSE)

        details = client.get_ticker_details("AAPL")

        assert details is not None
        assert details.ticker == "AAPL"
        assert details.name == "Apple Inc."
        assert details.market == "stocks"
        assert details.locale == "us"
        assert details.primary_exchange == "XNAS"
        assert details.type == "CS"
        assert details.active is True
        assert details.currency_name == "usd"
        assert details.market_cap == pytest.approx(3e12)
        assert details.sic_code == "3571"
        assert details.sic_description == "Electronic Computers"

    def test_get_ticker_details_correct_endpoint(self):
        client = MassiveClient(api_key="test")
        client._get = MagicMock(return_value=self._SAMPLE_RESPONSE)

        client.get_ticker_details("AAPL")

        client._get.assert_called_once_with("/v3/reference/tickers/AAPL")

    def test_get_ticker_details_uppercases_ticker(self):
        client = MassiveClient(api_key="test")
        client._get = MagicMock(return_value=self._SAMPLE_RESPONSE)

        client.get_ticker_details("aapl")

        client._get.assert_called_once_with("/v3/reference/tickers/AAPL")

    def test_get_ticker_details_no_results(self):
        client = MassiveClient(api_key="test")
        client._get = MagicMock(return_value={})

        details = client.get_ticker_details("NONEXISTENT")

        assert details is None

    def test_get_ticker_details_empty_results_key(self):
        client = MassiveClient(api_key="test")
        client._get = MagicMock(return_value={"results": None})

        details = client.get_ticker_details("AAPL")

        assert details is None

    def test_get_ticker_details_stores_to_raw_store(self):
        mock_store = MagicMock()
        client = MassiveClient(api_key="test", raw_store=mock_store)
        client._get = MagicMock(return_value=self._SAMPLE_RESPONSE)

        details = client.get_ticker_details("AAPL")

        assert details is not None
        mock_store.store_polygon_ticker_details.assert_called_once_with([details])

    def test_get_ticker_details_no_store_when_raw_store_none(self):
        client = MassiveClient(api_key="test", raw_store=None)
        client._get = MagicMock(return_value=self._SAMPLE_RESPONSE)

        details = client.get_ticker_details("AAPL")

        assert details is not None
        # No AttributeError — raw_store is None, branch simply skipped

    def test_get_ticker_details_store_exception_swallowed(self):
        mock_store = MagicMock()
        mock_store.store_polygon_ticker_details.side_effect = RuntimeError("db locked")
        client = MassiveClient(api_key="test", raw_store=mock_store)
        client._get = MagicMock(return_value=self._SAMPLE_RESPONSE)

        # Should not raise
        details = client.get_ticker_details("AAPL")
        assert details is not None

    def test_get_ticker_details_handles_missing_optional_fields(self):
        """Polygon may omit market_cap / sic_code for some tickers."""
        sparse = {
            "results": {
                "ticker": "XYZ",
                "name": "XYZ Corp",
                "market": "stocks",
                "locale": "us",
                "primary_exchange": "ARCX",
                "type": "ETF",
                "active": True,
                "currency_name": "usd",
                # market_cap, sic_code, sic_description omitted
            }
        }
        client = MassiveClient(api_key="test")
        client._get = MagicMock(return_value=sparse)

        details = client.get_ticker_details("XYZ")

        assert details is not None
        assert details.market_cap == 0.0
        assert details.sic_code == ""
        assert details.sic_description == ""
