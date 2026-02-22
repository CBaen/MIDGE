"""Tests for Mae's financial data and web search providers.

All HTTP calls are mocked. No real API keys needed.
Covers: RestDataProvider, TavilyProvider, and bootstrap Layer 31c.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from mae_core.external.api_client import ApiRequest, ApiResponse, ApiResponseStatus
from mae_core.external.rest_data_provider import RestDataProvider
from mae_core.external.tavily_provider import TavilyProvider


# =========================================================================
# Helpers
# =========================================================================

def _make_request(payload: dict, provider: str = "test") -> ApiRequest:
    return ApiRequest(
        request_id="req-001",
        provider=provider,
        endpoint="",
        payload=payload,
        priority=5,
        agent_id="agent-1",
    )


def _mock_httpx_get(status_code: int, json_data: dict):
    """Return a mock httpx response for GET calls."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = str(json_data)
    return resp


def _mock_httpx_post(status_code: int, json_data: dict):
    """Return a mock httpx response for POST calls."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = str(json_data)
    return resp


# =========================================================================
# RestDataProvider
# =========================================================================

class TestRestDataProviderAvailability:
    def test_available_with_key(self):
        p = RestDataProvider("finnhub", "key123", "https://finnhub.io/api/v1", "token")
        assert p.available is True

    def test_not_available_without_key(self):
        p = RestDataProvider("finnhub", "", "https://finnhub.io/api/v1", "token")
        assert p.available is False

    def test_no_key_returns_error(self):
        p = RestDataProvider("finnhub", "", "https://finnhub.io/api/v1", "token")
        req = _make_request({"endpoint": "/quote", "params": {"symbol": "AAPL"}})
        resp = p.execute(req)
        assert resp.status == ApiResponseStatus.ERROR
        assert "no API key" in resp.error_message


class TestRestDataProviderSuccess:
    @patch("mae_core.external.rest_data_provider.httpx.get")
    def test_successful_quote(self, mock_get):
        mock_get.return_value = _mock_httpx_get(200, {"c": 182.5, "h": 184.0, "l": 181.0})
        p = RestDataProvider("finnhub", "key123", "https://finnhub.io/api/v1", "token")
        req = _make_request({"endpoint": "/quote", "params": {"symbol": "AAPL"}}, "finnhub")
        resp = p.execute(req)
        assert resp.status == ApiResponseStatus.SUCCESS
        assert resp.payload["c"] == 182.5
        assert resp.provider == "finnhub"
        assert resp.latency_ms >= 0

    @patch("mae_core.external.rest_data_provider.httpx.get")
    def test_auth_param_injected(self, mock_get):
        """API key should be sent under the correct param name."""
        mock_get.return_value = _mock_httpx_get(200, {"data": []})
        p = RestDataProvider(
            "marketaux", "mytoken", "https://api.marketaux.com/v1", "api_token"
        )
        req = _make_request({"endpoint": "/news/all", "params": {"filter_entities": "true"}})
        p.execute(req)
        _, kwargs = mock_get.call_args
        params = kwargs.get("params", {})
        assert params.get("api_token") == "mytoken"
        assert params.get("filter_entities") == "true"

    @patch("mae_core.external.rest_data_provider.httpx.get")
    def test_default_endpoint_used(self, mock_get):
        """When payload omits 'endpoint', the default_endpoint is used."""
        mock_get.return_value = _mock_httpx_get(200, {"result": "ok"})
        p = RestDataProvider(
            "alphavantage", "avkey", "https://www.alphavantage.co", "apikey",
            default_endpoint="query",
        )
        req = _make_request({"params": {"function": "TIME_SERIES_DAILY", "symbol": "MSFT"}})
        p.execute(req)
        args, _ = mock_get.call_args
        url = args[0]
        assert "query" in url


class TestRestDataProviderErrors:
    @patch("mae_core.external.rest_data_provider.httpx.get")
    def test_rate_limit_429(self, mock_get):
        mock_get.return_value = _mock_httpx_get(429, {})
        p = RestDataProvider("finnhub", "key", "https://finnhub.io/api/v1", "token")
        resp = p.execute(_make_request({"endpoint": "/quote"}))
        assert resp.status == ApiResponseStatus.RATE_LIMITED

    @patch("mae_core.external.rest_data_provider.httpx.get")
    def test_server_error_500(self, mock_get):
        mock_get.return_value = _mock_httpx_get(500, {})
        p = RestDataProvider("finnhub", "key", "https://finnhub.io/api/v1", "token")
        resp = p.execute(_make_request({"endpoint": "/quote"}))
        assert resp.status == ApiResponseStatus.PROVIDER_ERROR

    @patch("mae_core.external.rest_data_provider.httpx.get")
    def test_client_error_400(self, mock_get):
        mock_get.return_value = _mock_httpx_get(401, {})
        p = RestDataProvider("finnhub", "key", "https://finnhub.io/api/v1", "token")
        resp = p.execute(_make_request({"endpoint": "/quote"}))
        assert resp.status == ApiResponseStatus.ERROR

    @patch("mae_core.external.rest_data_provider.httpx.get")
    def test_timeout(self, mock_get):
        import httpx
        mock_get.side_effect = httpx.TimeoutException("timed out")
        p = RestDataProvider("finnhub", "key", "https://finnhub.io/api/v1", "token")
        resp = p.execute(_make_request({"endpoint": "/quote"}))
        assert resp.status == ApiResponseStatus.TIMEOUT
        assert "timed out" in resp.error_message

    @patch("mae_core.external.rest_data_provider.httpx.get")
    def test_unexpected_exception(self, mock_get):
        mock_get.side_effect = RuntimeError("network down")
        p = RestDataProvider("finnhub", "key", "https://finnhub.io/api/v1", "token")
        resp = p.execute(_make_request({"endpoint": "/quote"}))
        assert resp.status == ApiResponseStatus.ERROR
        assert "network down" in resp.error_message


# =========================================================================
# TavilyProvider
# =========================================================================

class TestTavilyProviderAvailability:
    def test_available_with_key(self):
        p = TavilyProvider("tvly-key123")
        assert p.available is True

    def test_not_available_without_key(self):
        p = TavilyProvider("")
        assert p.available is False

    def test_no_key_returns_error(self):
        p = TavilyProvider("")
        req = _make_request({"query": "inflation rate 2026"}, "tavily")
        resp = p.execute(req)
        assert resp.status == ApiResponseStatus.ERROR
        assert "no API key" in resp.error_message

    def test_no_query_returns_error(self):
        p = TavilyProvider("tvly-key123")
        req = _make_request({}, "tavily")
        resp = p.execute(req)
        assert resp.status == ApiResponseStatus.ERROR
        assert "no query" in resp.error_message


class TestTavilyProviderSuccess:
    @patch("mae_core.external.tavily_provider.httpx.post")
    def test_successful_search(self, mock_post):
        mock_post.return_value = _mock_httpx_post(200, {
            "query": "inflation rate 2026",
            "results": [
                {"title": "CPI Data", "url": "https://bls.gov", "content": "CPI rose 3.1%", "score": 0.95}
            ],
            "answer": "Inflation in 2026 is approximately 3.1%.",
        })
        p = TavilyProvider("tvly-key123")
        req = _make_request({"query": "inflation rate 2026"}, "tavily")
        resp = p.execute(req)
        assert resp.status == ApiResponseStatus.SUCCESS
        assert resp.payload["query"] == "inflation rate 2026"
        assert len(resp.payload["results"]) == 1
        assert resp.payload["answer"] == "Inflation in 2026 is approximately 3.1%."
        assert resp.provider == "tavily"

    @patch("mae_core.external.tavily_provider.httpx.post")
    def test_text_field_populated_from_answer(self, mock_post):
        """text field should carry the answer for LLM result reader compatibility."""
        mock_post.return_value = _mock_httpx_post(200, {
            "results": [],
            "answer": "The answer is 42.",
        })
        p = TavilyProvider("tvly-key123")
        req = _make_request({"query": "what is the answer"}, "tavily")
        resp = p.execute(req)
        assert resp.payload["text"] == "The answer is 42."

    @patch("mae_core.external.tavily_provider.httpx.post")
    def test_text_field_falls_back_to_first_result(self, mock_post):
        """If no answer, text should be the first result's content."""
        mock_post.return_value = _mock_httpx_post(200, {
            "results": [{"title": "T", "url": "u", "content": "First result content", "score": 0.9}],
            "answer": "",
        })
        p = TavilyProvider("tvly-key123")
        req = _make_request({"query": "some query"}, "tavily")
        resp = p.execute(req)
        assert resp.payload["text"] == "First result content"

    @patch("mae_core.external.tavily_provider.httpx.post")
    def test_question_key_fallback(self, mock_post):
        """If payload has 'question' but no 'query', it should still work."""
        mock_post.return_value = _mock_httpx_post(200, {"results": [], "answer": "ok"})
        p = TavilyProvider("tvly-key123")
        req = _make_request({"question": "what is inflation?"}, "tavily")
        resp = p.execute(req)
        assert resp.status == ApiResponseStatus.SUCCESS

    @patch("mae_core.external.tavily_provider.httpx.post")
    def test_api_key_sent_in_body(self, mock_post):
        """API key must be in the POST body, not a query param."""
        mock_post.return_value = _mock_httpx_post(200, {"results": [], "answer": ""})
        p = TavilyProvider("tvly-supersecret")
        req = _make_request({"query": "test"}, "tavily")
        p.execute(req)
        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        assert body.get("api_key") == "tvly-supersecret"

    @patch("mae_core.external.tavily_provider.httpx.post")
    def test_max_results_and_depth_forwarded(self, mock_post):
        mock_post.return_value = _mock_httpx_post(200, {"results": [], "answer": ""})
        p = TavilyProvider("tvly-key", default_max_results=5, default_depth="basic")
        req = _make_request({"query": "test", "max_results": 10, "search_depth": "advanced"}, "tavily")
        p.execute(req)
        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        assert body["max_results"] == 10
        assert body["search_depth"] == "advanced"

    @patch("mae_core.external.tavily_provider.httpx.post")
    def test_defaults_applied(self, mock_post):
        """When payload omits optional keys, defaults are used."""
        mock_post.return_value = _mock_httpx_post(200, {"results": [], "answer": ""})
        p = TavilyProvider("tvly-key", default_max_results=3, default_depth="basic")
        req = _make_request({"query": "test"}, "tavily")
        p.execute(req)
        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        assert body["max_results"] == 3
        assert body["search_depth"] == "basic"


class TestTavilyProviderErrors:
    @patch("mae_core.external.tavily_provider.httpx.post")
    def test_rate_limit_429(self, mock_post):
        mock_post.return_value = _mock_httpx_post(429, {})
        p = TavilyProvider("tvly-key")
        resp = p.execute(_make_request({"query": "test"}, "tavily"))
        assert resp.status == ApiResponseStatus.RATE_LIMITED

    @patch("mae_core.external.tavily_provider.httpx.post")
    def test_server_error_500(self, mock_post):
        mock_post.return_value = _mock_httpx_post(500, {})
        p = TavilyProvider("tvly-key")
        resp = p.execute(_make_request({"query": "test"}, "tavily"))
        assert resp.status == ApiResponseStatus.PROVIDER_ERROR

    @patch("mae_core.external.tavily_provider.httpx.post")
    def test_client_error_401(self, mock_post):
        mock_post.return_value = _mock_httpx_post(401, {})
        p = TavilyProvider("tvly-key")
        resp = p.execute(_make_request({"query": "test"}, "tavily"))
        assert resp.status == ApiResponseStatus.ERROR

    @patch("mae_core.external.tavily_provider.httpx.post")
    def test_timeout(self, mock_post):
        import httpx
        mock_post.side_effect = httpx.TimeoutException("timed out")
        p = TavilyProvider("tvly-key")
        resp = p.execute(_make_request({"query": "test"}, "tavily"))
        assert resp.status == ApiResponseStatus.TIMEOUT

    @patch("mae_core.external.tavily_provider.httpx.post")
    def test_unexpected_exception(self, mock_post):
        mock_post.side_effect = RuntimeError("connection refused")
        p = TavilyProvider("tvly-key")
        resp = p.execute(_make_request({"query": "test"}, "tavily"))
        assert resp.status == ApiResponseStatus.ERROR
        assert "connection refused" in resp.error_message


# =========================================================================
# Bootstrap Layer 31c integration
# =========================================================================

class TestBootstrapDataProviders:
    """Verify that bootstrap wires data providers when env vars are set."""

    def _make_ctx(self, api_gateway):
        from types import SimpleNamespace
        ctx = SimpleNamespace()
        ctx.api_gateway = api_gateway
        ctx.boundary_membrane = MagicMock()
        ctx.boundary_membrane.register_source = MagicMock()
        return ctx

    @patch.dict("os.environ", {"MAE_FINNHUB_API_KEY": "fh-key"})
    def test_finnhub_registered_when_key_present(self):
        from mae_core.bootstrap.external import _register_data_providers
        gw = MagicMock()
        gw.has_provider.return_value = False
        ctx = self._make_ctx(gw)
        _register_data_providers(ctx)
        provider_names = [call.args[0] for call in gw.register_provider.call_args_list]
        assert "finnhub" in provider_names

    @patch.dict("os.environ", {"MAE_MARKETAUX_API_KEY": "mx-key"})
    def test_marketaux_registered_when_key_present(self):
        from mae_core.bootstrap.external import _register_data_providers
        gw = MagicMock()
        gw.has_provider.return_value = False
        ctx = self._make_ctx(gw)
        _register_data_providers(ctx)
        provider_names = [call.args[0] for call in gw.register_provider.call_args_list]
        assert "marketaux" in provider_names

    @patch.dict("os.environ", {"MAE_ALPHAVANTAGE_API_KEY": "av-key"})
    def test_alphavantage_registered_when_key_present(self):
        from mae_core.bootstrap.external import _register_data_providers
        gw = MagicMock()
        gw.has_provider.return_value = False
        ctx = self._make_ctx(gw)
        _register_data_providers(ctx)
        provider_names = [call.args[0] for call in gw.register_provider.call_args_list]
        assert "alphavantage" in provider_names

    @patch.dict("os.environ", {"MAE_TAVILY_API_KEY": "tvly-key"})
    def test_tavily_registered_when_key_present(self):
        from mae_core.bootstrap.external import _register_data_providers
        gw = MagicMock()
        gw.has_provider.return_value = False
        ctx = self._make_ctx(gw)
        _register_data_providers(ctx)
        provider_names = [call.args[0] for call in gw.register_provider.call_args_list]
        assert "tavily" in provider_names

    @patch.dict("os.environ", {}, clear=True)
    def test_no_providers_without_keys(self):
        """Graceful degradation: no keys = nothing registered."""
        from mae_core.bootstrap.external import _register_data_providers
        # Clear any accidentally set keys
        import os
        for k in ("MAE_MARKETAUX_API_KEY", "MAE_FINNHUB_API_KEY",
                   "MAE_ALPHAVANTAGE_API_KEY", "MAE_TAVILY_API_KEY"):
            os.environ.pop(k, None)
        gw = MagicMock()
        gw.has_provider.return_value = False
        ctx = self._make_ctx(gw)
        _register_data_providers(ctx)
        assert gw.register_provider.call_count == 0

    @patch.dict("os.environ", {
        "MAE_FINNHUB_API_KEY": "fh-key",
        "MAE_TAVILY_API_KEY": "tvly-key",
    })
    def test_boundary_membrane_trusted_for_each_provider(self):
        from mae_core.bootstrap.external import _register_data_providers
        gw = MagicMock()
        gw.has_provider.return_value = False
        bm = MagicMock()
        bm.register_source = MagicMock()
        ctx = self._make_ctx(gw)
        ctx.boundary_membrane = bm
        _register_data_providers(ctx)
        trusted = [call.args[0] for call in bm.register_source.call_args_list]
        assert "finnhub" in trusted
        assert "tavily" in trusted
