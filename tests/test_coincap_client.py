"""Tests for CoinCap crypto client.

Covers: dataclass creation, get_assets, get_asset_history,
rate limiter, no-auth design, and null field handling.
All HTTP calls are mocked — no real network traffic.
"""
from __future__ import annotations

import time
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.apis.coincap_client import (
    CoinCapClient,
    CryptoAsset,
    CryptoCandle,
    _RATE_LIMIT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ASSET_PAYLOAD = {
    "data": [
        {
            "id": "bitcoin",
            "symbol": "BTC",
            "name": "Bitcoin",
            "rank": "1",
            "priceUsd": "65000.12",
            "volumeUsd24Hr": "30000000000.00",
            "changePercent24Hr": "2.5",
            "marketCapUsd": "1280000000000.00",
            "supply": "19700000.00",
            "maxSupply": "21000000.00",
            "vwap24Hr": "64800.00",
        },
        {
            "id": "ethereum",
            "symbol": "ETH",
            "name": "Ethereum",
            "rank": "2",
            "priceUsd": "3200.00",
            "volumeUsd24Hr": "15000000000.00",
            "changePercent24Hr": "-1.2",
            "marketCapUsd": "384000000000.00",
            "supply": "120000000.00",
            "maxSupply": None,
            "vwap24Hr": None,
        },
    ]
}

HISTORY_PAYLOAD = {
    "data": [
        {"priceUsd": "65000.00", "time": 1700000000000},
        {"priceUsd": "65500.00", "time": 1700003600000},
        {"priceUsd": "64800.00", "time": 1700007200000},
    ]
}


def _mock_response(payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestCryptoAsset:
    def test_create_with_all_fields(self):
        asset = CryptoAsset(
            asset_id="bitcoin",
            symbol="BTC",
            name="Bitcoin",
            rank=1,
            price_usd=65000.0,
            volume_24h_usd=3e10,
            change_24h_pct=2.5,
            market_cap_usd=1.28e12,
            supply=19_700_000.0,
            max_supply=21_000_000.0,
            vwap_24h=64800.0,
        )
        assert asset.asset_id == "bitcoin"
        assert asset.symbol == "BTC"
        assert asset.rank == 1
        assert asset.price_usd == 65000.0

    def test_optional_fields_can_be_none(self):
        asset = CryptoAsset(
            asset_id="ethereum",
            symbol="ETH",
            name="Ethereum",
            rank=2,
            price_usd=3200.0,
            volume_24h_usd=1.5e10,
            change_24h_pct=-1.2,
            market_cap_usd=3.84e11,
            supply=1.2e8,
            max_supply=None,
            vwap_24h=None,
        )
        assert asset.max_supply is None
        assert asset.vwap_24h is None


class TestCryptoCandle:
    def test_create_candle(self):
        candle = CryptoCandle(
            open=65000.0,
            high=65000.0,
            low=65000.0,
            close=65000.0,
            volume=0.0,
            period=1700000000000,
        )
        assert candle.open == 65000.0
        assert candle.period == 1700000000000
        assert candle.volume == 0.0


# ---------------------------------------------------------------------------
# get_assets tests
# ---------------------------------------------------------------------------

class TestGetAssets:
    def test_returns_parsed_assets(self):
        client = CoinCapClient()
        mock_resp = _mock_response(ASSET_PAYLOAD)
        with patch.object(client._session, "get", return_value=mock_resp):
            assets = client.get_assets(limit=2)
        assert len(assets) == 2
        btc = assets[0]
        assert btc.asset_id == "bitcoin"
        assert btc.symbol == "BTC"
        assert btc.rank == 1
        assert btc.price_usd == pytest.approx(65000.12)
        assert btc.max_supply == pytest.approx(21_000_000.0)
        assert btc.vwap_24h == pytest.approx(64800.0)

    def test_null_optional_fields_become_none(self):
        client = CoinCapClient()
        mock_resp = _mock_response(ASSET_PAYLOAD)
        with patch.object(client._session, "get", return_value=mock_resp):
            assets = client.get_assets(limit=2)
        eth = assets[1]
        assert eth.max_supply is None
        assert eth.vwap_24h is None

    def test_empty_response_returns_empty_list(self):
        client = CoinCapClient()
        mock_resp = _mock_response({})
        with patch.object(client._session, "get", return_value=mock_resp):
            assets = client.get_assets()
        assert assets == []

    def test_missing_data_key_returns_empty_list(self):
        client = CoinCapClient()
        mock_resp = _mock_response({"timestamp": 12345})
        with patch.object(client._session, "get", return_value=mock_resp):
            assets = client.get_assets()
        assert assets == []

    def test_malformed_entries_are_skipped_gracefully(self):
        payload = {
            "data": [
                # Valid entry
                {
                    "id": "bitcoin", "symbol": "BTC", "name": "Bitcoin",
                    "rank": "1", "priceUsd": "65000.00", "volumeUsd24Hr": "3e10",
                    "changePercent24Hr": "2.5", "marketCapUsd": "1.28e12",
                    "supply": "19700000", "maxSupply": None, "vwap24Hr": None,
                },
                # Malformed — priceUsd is non-numeric
                {
                    "id": "bad", "symbol": "BAD", "name": "Bad",
                    "rank": "99", "priceUsd": "not-a-number", "volumeUsd24Hr": "0",
                    "changePercent24Hr": "0", "marketCapUsd": "0",
                    "supply": "0", "maxSupply": None, "vwap24Hr": None,
                },
            ]
        }
        client = CoinCapClient()
        mock_resp = _mock_response(payload)
        with patch.object(client._session, "get", return_value=mock_resp):
            assets = client.get_assets()
        # Only the valid entry survives
        assert len(assets) == 1
        assert assets[0].symbol == "BTC"

    def test_request_exception_returns_empty_list(self):
        import requests as req
        client = CoinCapClient()
        with patch.object(client._session, "get", side_effect=req.ConnectionError("down")):
            assets = client.get_assets()
        assert assets == []


# ---------------------------------------------------------------------------
# get_asset_history tests
# ---------------------------------------------------------------------------

class TestGetAssetHistory:
    def test_returns_parsed_candles(self):
        client = CoinCapClient()
        mock_resp = _mock_response(HISTORY_PAYLOAD)
        with patch.object(client._session, "get", return_value=mock_resp):
            candles = client.get_asset_history("bitcoin", interval="h1")
        assert len(candles) == 3
        c = candles[0]
        assert c.open == pytest.approx(65000.0)
        assert c.close == pytest.approx(65000.0)
        assert c.period == 1700000000000
        # open/high/low/close all equal (CoinCap limitation)
        assert c.open == c.high == c.low == c.close

    def test_volume_is_zero(self):
        client = CoinCapClient()
        mock_resp = _mock_response(HISTORY_PAYLOAD)
        with patch.object(client._session, "get", return_value=mock_resp):
            candles = client.get_asset_history("bitcoin")
        for candle in candles:
            assert candle.volume == 0.0

    def test_empty_history_returns_empty_list(self):
        client = CoinCapClient()
        mock_resp = _mock_response({})
        with patch.object(client._session, "get", return_value=mock_resp):
            candles = client.get_asset_history("bitcoin")
        assert candles == []

    def test_request_exception_returns_empty_list(self):
        import requests as req
        client = CoinCapClient()
        with patch.object(client._session, "get", side_effect=req.Timeout("timeout")):
            candles = client.get_asset_history("ethereum")
        assert candles == []


# ---------------------------------------------------------------------------
# Rate limiter tests
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_call_times_deque_tracks_requests(self):
        client = CoinCapClient()
        mock_resp = _mock_response({"data": []})
        with patch.object(client._session, "get", return_value=mock_resp):
            client.get_assets()
            client.get_assets()
        assert len(client._call_times) == 2

    def test_deque_maxlen_is_rate_limit(self):
        client = CoinCapClient()
        assert client._call_times.maxlen == _RATE_LIMIT

    def test_deque_does_not_exceed_maxlen(self):
        client = CoinCapClient()
        # Manually fill deque beyond maxlen
        for i in range(_RATE_LIMIT + 50):
            client._call_times.append(float(i))
        assert len(client._call_times) == _RATE_LIMIT


# ---------------------------------------------------------------------------
# No-auth design tests
# ---------------------------------------------------------------------------

class TestNoAuthDesign:
    def test_no_authorization_header(self):
        client = CoinCapClient()
        assert "Authorization" not in client._session.headers

    def test_no_api_key_header(self):
        client = CoinCapClient()
        header_keys_lower = {k.lower() for k in client._session.headers}
        assert "x-api-key" not in header_keys_lower

    def test_accept_encoding_gzip_set(self):
        client = CoinCapClient()
        assert client._session.headers.get("Accept-Encoding") == "gzip"
