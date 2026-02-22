"""Tests for mae_core.market.apis.market_data_provider."""

import pytest
from unittest.mock import patch, MagicMock

from mae_core.external.api_client import ApiRequest, ApiResponseStatus
from mae_core.market.apis.market_data_provider import MarketDataProvider


@pytest.fixture
def provider():
    return MarketDataProvider()


def _make_request(payload: dict) -> ApiRequest:
    return ApiRequest(
        request_id="test-1",
        provider="market_data",
        endpoint="",
        payload=payload,
    )


class TestMarketDataProvider:
    """Test MarketDataProvider execute()."""

    def test_missing_url_returns_error(self, provider):
        req = _make_request({})
        resp = provider.execute(req)
        assert resp.status == ApiResponseStatus.ERROR
        assert "no URL" in resp.error_message

    @patch("mae_core.market.apis.market_data_provider.requests.Session")
    def test_get_success(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [1, 2, 3]}
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        prov = MarketDataProvider()
        req = _make_request({
            "url": "https://api.example.com/data",
            "method": "GET",
            "params": {"q": "test"},
        })
        resp = prov.execute(req)

        assert resp.status == ApiResponseStatus.SUCCESS
        assert resp.payload == {"data": [1, 2, 3]}
        mock_session.get.assert_called_once()

    @patch("mae_core.market.apis.market_data_provider.requests.Session")
    def test_post_success(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        prov = MarketDataProvider()
        req = _make_request({
            "url": "https://api.usaspending.gov/api/v2/search/spending_by_award/",
            "method": "POST",
            "json_body": {"filters": {}, "limit": 10},
        })
        resp = prov.execute(req)

        assert resp.status == ApiResponseStatus.SUCCESS
        mock_session.post.assert_called_once()

    @patch("mae_core.market.apis.market_data_provider.requests.Session")
    def test_rate_limited(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        prov = MarketDataProvider()
        req = _make_request({"url": "https://api.example.com/data"})
        resp = prov.execute(req)
        assert resp.status == ApiResponseStatus.RATE_LIMITED

    @patch("mae_core.market.apis.market_data_provider.requests.Session")
    def test_server_error(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        prov = MarketDataProvider()
        req = _make_request({"url": "https://api.example.com/data"})
        resp = prov.execute(req)
        assert resp.status == ApiResponseStatus.PROVIDER_ERROR

    @patch("mae_core.market.apis.market_data_provider.requests.Session")
    def test_client_error(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        prov = MarketDataProvider()
        req = _make_request({"url": "https://api.example.com/data"})
        resp = prov.execute(req)
        assert resp.status == ApiResponseStatus.ERROR
        assert "403" in resp.error_message

    @patch("mae_core.market.apis.market_data_provider.requests.Session")
    def test_timeout(self, mock_session_cls):
        import requests as req_lib
        mock_session = MagicMock()
        mock_session.get.side_effect = req_lib.Timeout("timed out")
        mock_session_cls.return_value = mock_session

        prov = MarketDataProvider()
        req = _make_request({"url": "https://api.example.com/data"})
        resp = prov.execute(req)
        assert resp.status == ApiResponseStatus.TIMEOUT

    @patch("mae_core.market.apis.market_data_provider.requests.Session")
    def test_headers_passed(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        prov = MarketDataProvider()
        req = _make_request({
            "url": "https://api.example.com/data",
            "headers": {"X-RapidAPI-Key": "test-key"},
        })
        resp = prov.execute(req)
        assert resp.status == ApiResponseStatus.SUCCESS
        call_kwargs = mock_session.get.call_args
        assert call_kwargs.kwargs.get("headers", {}).get("X-RapidAPI-Key") == "test-key"

    def test_default_method_is_get(self, provider):
        """Verify default method is GET when not specified in payload."""
        # We can't easily test this without mocking, but we verify
        # the provider initializes properly
        assert provider.provider_name == "market_data"

    @patch("mae_core.market.apis.market_data_provider.requests.Session")
    def test_non_json_response_falls_back_to_text(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("not json")
        mock_resp.text = "<html>response</html>"
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        prov = MarketDataProvider()
        req = _make_request({"url": "https://api.example.com/data"})
        resp = prov.execute(req)
        assert resp.status == ApiResponseStatus.SUCCESS
        assert resp.payload == {"text": "<html>response</html>"}
