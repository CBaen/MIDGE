"""Tests for CoinGecko crypto client.

Covers: dataclass creation, get_prices, get_market_data, get_trending,
rate limiter deque tracking, API key header behavior, and graceful
handling of empty or malformed responses.
All HTTP calls are mocked — no real network traffic.
"""
from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import pytest
import requests as req

from mae_core.market.apis.coingecko_client import (
    CoinGeckoClient,
    CryptoPrice,
    GlobalCryptoData,
    TrendingCoin,
    _RATE_LIMIT,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

PRICES_PAYLOAD = [
    {
        "id": "bitcoin",
        "symbol": "btc",
        "current_price": 65000.12,
        "total_volume": 30_000_000_000.0,
        "price_change_percentage_24h": 2.5,
        "price_change_percentage_7d_in_currency": -1.3,
        "market_cap": 1_280_000_000_000.0,
        "last_updated": "2024-01-01T00:00:00.000Z",
    },
    {
        "id": "ethereum",
        "symbol": "eth",
        "current_price": 3200.0,
        "total_volume": 15_000_000_000.0,
        "price_change_percentage_24h": -1.2,
        "price_change_percentage_7d_in_currency": None,  # test None → 0
        "market_cap": 384_000_000_000.0,
        "last_updated": "2024-01-01T00:00:00.000Z",
    },
]

GLOBAL_PAYLOAD = {
    "data": {
        "total_market_cap": {"usd": 2_500_000_000_000.0, "eur": 2_300_000_000_000.0},
        "market_cap_percentage": {"btc": 52.3, "eth": 17.1},
        "market_cap_change_percentage_24h_usd": -0.85,
        "active_cryptocurrencies": 12000,
    }
}

TRENDING_PAYLOAD = {
    "coins": [
        {
            "item": {
                "id": "pepe",
                "symbol": "pepe",
                "name": "Pepe",
                "market_cap_rank": 42,
                "price_btc": 0.00000015,
            }
        },
        {
            "item": {
                "id": "dogecoin",
                "symbol": "doge",
                "name": "Dogecoin",
                "market_cap_rank": None,  # test None → 0
                "price_btc": 0.0000025,
            }
        },
    ]
}


def _mock_response(payload, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Dataclass creation tests
# ---------------------------------------------------------------------------

class TestCryptoPriceDataclass:
    def test_create_with_all_fields(self):
        price = CryptoPrice(
            coin_id="bitcoin",
            symbol="BTC",
            price_usd=65000.0,
            volume_24h=3e10,
            change_24h_pct=2.5,
            change_7d_pct=-1.3,
            market_cap=1.28e12,
            last_updated="2024-01-01T00:00:00.000Z",
        )
        assert price.coin_id == "bitcoin"
        assert price.symbol == "BTC"
        assert price.price_usd == 65000.0
        assert price.change_7d_pct == -1.3

    def test_fields_are_typed(self):
        price = CryptoPrice(
            coin_id="x", symbol="X", price_usd=1.0, volume_24h=1.0,
            change_24h_pct=0.0, change_7d_pct=0.0, market_cap=1.0,
            last_updated="",
        )
        assert isinstance(price.price_usd, float)
        assert isinstance(price.coin_id, str)


class TestGlobalCryptoDataDataclass:
    def test_create_with_all_fields(self):
        gd = GlobalCryptoData(
            total_market_cap_usd=2.5e12,
            btc_dominance=52.3,
            eth_dominance=17.1,
            market_cap_change_24h_pct=-0.85,
            active_cryptocurrencies=12000,
        )
        assert gd.total_market_cap_usd == 2.5e12
        assert gd.btc_dominance == 52.3
        assert gd.active_cryptocurrencies == 12000

    def test_active_cryptocurrencies_is_int(self):
        gd = GlobalCryptoData(
            total_market_cap_usd=1.0, btc_dominance=50.0, eth_dominance=15.0,
            market_cap_change_24h_pct=0.0, active_cryptocurrencies=9999,
        )
        assert isinstance(gd.active_cryptocurrencies, int)


class TestTrendingCoinDataclass:
    def test_create_with_all_fields(self):
        coin = TrendingCoin(
            coin_id="pepe", symbol="PEPE", name="Pepe",
            market_cap_rank=42, price_btc=0.00000015,
        )
        assert coin.coin_id == "pepe"
        assert coin.symbol == "PEPE"
        assert coin.market_cap_rank == 42


# ---------------------------------------------------------------------------
# get_prices tests
# ---------------------------------------------------------------------------

class TestGetPrices:
    def test_returns_parsed_prices(self):
        client = CoinGeckoClient()
        mock_resp = _mock_response(PRICES_PAYLOAD)
        with patch.object(client._session, "get", return_value=mock_resp):
            prices = client.get_prices(["bitcoin", "ethereum"])
        assert len(prices) == 2
        btc = prices[0]
        assert btc.coin_id == "bitcoin"
        assert btc.symbol == "BTC"
        assert btc.price_usd == pytest.approx(65000.12)
        assert btc.change_24h_pct == pytest.approx(2.5)
        assert btc.change_7d_pct == pytest.approx(-1.3)
        assert btc.market_cap == pytest.approx(1.28e12)

    def test_none_change_7d_becomes_zero(self):
        client = CoinGeckoClient()
        mock_resp = _mock_response(PRICES_PAYLOAD)
        with patch.object(client._session, "get", return_value=mock_resp):
            prices = client.get_prices(["bitcoin", "ethereum"])
        eth = prices[1]
        assert eth.change_7d_pct == pytest.approx(0.0)

    def test_symbol_uppercased(self):
        client = CoinGeckoClient()
        mock_resp = _mock_response(PRICES_PAYLOAD)
        with patch.object(client._session, "get", return_value=mock_resp):
            prices = client.get_prices(["bitcoin"])
        assert prices[0].symbol == "BTC"

    def test_default_coins_are_top5(self):
        """Verify default coin list matches the five expected IDs."""
        client = CoinGeckoClient()
        captured_params = {}

        def fake_get(url, params=None, timeout=None):
            captured_params.update(params or {})
            return _mock_response([])

        with patch.object(client._session, "get", side_effect=fake_get):
            client.get_prices()

        ids = captured_params.get("ids", "")
        for expected in ["bitcoin", "ethereum", "solana", "ripple", "cardano"]:
            assert expected in ids

    def test_non_list_response_returns_empty(self):
        client = CoinGeckoClient()
        mock_resp = _mock_response({"error": "not a list"})
        with patch.object(client._session, "get", return_value=mock_resp):
            prices = client.get_prices()
        assert prices == []

    def test_empty_list_response_returns_empty(self):
        client = CoinGeckoClient()
        mock_resp = _mock_response([])
        with patch.object(client._session, "get", return_value=mock_resp):
            prices = client.get_prices()
        assert prices == []

    def test_malformed_entries_skipped_gracefully(self):
        payload = [
            # Valid
            {
                "id": "bitcoin", "symbol": "btc", "current_price": 65000.0,
                "total_volume": 3e10, "price_change_percentage_24h": 2.5,
                "price_change_percentage_7d_in_currency": 0.0,
                "market_cap": 1.28e12, "last_updated": "",
            },
            # Malformed — price is non-numeric string
            {
                "id": "bad", "symbol": "bad", "current_price": "not-a-number",
                "total_volume": 0, "price_change_percentage_24h": 0,
                "price_change_percentage_7d_in_currency": 0,
                "market_cap": 0, "last_updated": "",
            },
        ]
        client = CoinGeckoClient()
        mock_resp = _mock_response(payload)
        with patch.object(client._session, "get", return_value=mock_resp):
            prices = client.get_prices()
        assert len(prices) == 1
        assert prices[0].coin_id == "bitcoin"

    def test_request_exception_returns_empty(self):
        client = CoinGeckoClient()
        with patch.object(client._session, "get", side_effect=req.ConnectionError("down")):
            prices = client.get_prices()
        assert prices == []


# ---------------------------------------------------------------------------
# get_market_data tests
# ---------------------------------------------------------------------------

class TestGetMarketData:
    def test_returns_parsed_global_data(self):
        client = CoinGeckoClient()
        mock_resp = _mock_response(GLOBAL_PAYLOAD)
        with patch.object(client._session, "get", return_value=mock_resp):
            gd = client.get_market_data()
        assert gd is not None
        # total_market_cap sums usd + eur
        assert gd.total_market_cap_usd == pytest.approx(4_800_000_000_000.0)
        assert gd.btc_dominance == pytest.approx(52.3)
        assert gd.eth_dominance == pytest.approx(17.1)
        assert gd.market_cap_change_24h_pct == pytest.approx(-0.85)
        assert gd.active_cryptocurrencies == 12000

    def test_missing_data_key_returns_none(self):
        client = CoinGeckoClient()
        mock_resp = _mock_response({"timestamp": 12345})
        with patch.object(client._session, "get", return_value=mock_resp):
            gd = client.get_market_data()
        assert gd is None

    def test_empty_response_returns_none(self):
        client = CoinGeckoClient()
        mock_resp = _mock_response({})
        with patch.object(client._session, "get", return_value=mock_resp):
            gd = client.get_market_data()
        assert gd is None

    def test_request_exception_returns_none(self):
        client = CoinGeckoClient()
        with patch.object(client._session, "get", side_effect=req.Timeout("timeout")):
            gd = client.get_market_data()
        assert gd is None


# ---------------------------------------------------------------------------
# get_trending tests
# ---------------------------------------------------------------------------

class TestGetTrending:
    def test_returns_parsed_trending(self):
        client = CoinGeckoClient()
        mock_resp = _mock_response(TRENDING_PAYLOAD)
        with patch.object(client._session, "get", return_value=mock_resp):
            coins = client.get_trending()
        assert len(coins) == 2
        pepe = coins[0]
        assert pepe.coin_id == "pepe"
        assert pepe.symbol == "PEPE"
        assert pepe.name == "Pepe"
        assert pepe.market_cap_rank == 42
        assert pepe.price_btc == pytest.approx(0.00000015)

    def test_none_market_cap_rank_becomes_zero(self):
        client = CoinGeckoClient()
        mock_resp = _mock_response(TRENDING_PAYLOAD)
        with patch.object(client._session, "get", return_value=mock_resp):
            coins = client.get_trending()
        doge = coins[1]
        assert doge.market_cap_rank == 0

    def test_symbol_uppercased(self):
        client = CoinGeckoClient()
        mock_resp = _mock_response(TRENDING_PAYLOAD)
        with patch.object(client._session, "get", return_value=mock_resp):
            coins = client.get_trending()
        assert coins[0].symbol == "PEPE"
        assert coins[1].symbol == "DOGE"

    def test_missing_coins_key_returns_empty(self):
        client = CoinGeckoClient()
        mock_resp = _mock_response({"nfts": []})
        with patch.object(client._session, "get", return_value=mock_resp):
            coins = client.get_trending()
        assert coins == []

    def test_empty_response_returns_empty(self):
        client = CoinGeckoClient()
        mock_resp = _mock_response({})
        with patch.object(client._session, "get", return_value=mock_resp):
            coins = client.get_trending()
        assert coins == []

    def test_request_exception_returns_empty(self):
        client = CoinGeckoClient()
        with patch.object(client._session, "get", side_effect=req.ConnectionError("down")):
            coins = client.get_trending()
        assert coins == []


# ---------------------------------------------------------------------------
# Rate limiter tests
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_deque_maxlen_equals_rate_limit(self):
        client = CoinGeckoClient()
        assert client._call_times.maxlen == _RATE_LIMIT

    def test_call_times_deque_tracks_requests(self):
        client = CoinGeckoClient()
        mock_resp = _mock_response([])
        with patch.object(client._session, "get", return_value=mock_resp):
            client.get_prices()
            client.get_prices()
        assert len(client._call_times) == 2

    def test_deque_does_not_exceed_maxlen(self):
        client = CoinGeckoClient()
        for i in range(_RATE_LIMIT + 50):
            client._call_times.append(float(i))
        assert len(client._call_times) == _RATE_LIMIT


# ---------------------------------------------------------------------------
# API key header tests
# ---------------------------------------------------------------------------

class TestApiKeyHeader:
    def test_api_key_header_set_when_provided(self):
        client = CoinGeckoClient(api_key="test-key-123")
        assert client._session.headers.get("x-cg-demo-api-key") == "test-key-123"

    def test_no_api_key_header_when_not_provided(self, monkeypatch):
        monkeypatch.delenv("COINGECKO_API_KEY", raising=False)
        client = CoinGeckoClient()
        assert "x-cg-demo-api-key" not in client._session.headers

    def test_accept_header_always_set(self):
        client = CoinGeckoClient()
        assert client._session.headers.get("Accept") == "application/json"

    def test_env_var_used_as_api_key(self, monkeypatch):
        monkeypatch.setenv("COINGECKO_API_KEY", "env-key-456")
        client = CoinGeckoClient()
        assert client._session.headers.get("x-cg-demo-api-key") == "env-key-456"
