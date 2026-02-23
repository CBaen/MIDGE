"""Tests: All 6 market API clients route through MarketDataProvider when injected.

Each test mocks provider.execute() to return a canned ApiResponse, calls the
client's public method, and verifies the provider was called with correct
URL/method/params and the client correctly parsed the ApiResponse payload.
"""

import pytest
from unittest.mock import MagicMock, patch

from mae_core.external.api_client import ApiResponse, ApiResponseStatus


def _success_response(payload):
    """Build a successful ApiResponse with given payload."""
    return ApiResponse(
        status=ApiResponseStatus.SUCCESS,
        payload=payload,
        provider="market_data",
        latency_ms=50.0,
    )


# ---- USASpendingClient ----

class TestUSASpendingRouting:
    """USASpendingClient routes POST through provider."""

    def test_search_contracts_via_provider(self):
        provider = MagicMock()
        provider.execute.return_value = _success_response({
            "results": [{
                "Award ID": "CONT123",
                "Award Amount": 5000000,
                "Start Date": "2025-01-01",
                "End Date": "2026-01-01",
                "Recipient Name": "TestCorp",
                "recipient_id": "DUNS123",
                "Description": "Test contract",
                "NAICS Code": "541330",
                "NAICS Description": "Engineering",
                "Awarding Agency": "DOD",
                "Awarding Sub Agency": "Army",
                "Funding Agency": "DOD",
                "Place of Performance State Code": "VA",
            }],
            "page_metadata": {"total": 1},
        })

        from mae_core.market.apis.usa_spending import USASpendingClient
        client = USASpendingClient(provider=provider)
        results = client.search_contracts(keyword="test")

        assert provider.execute.called
        req = provider.execute.call_args[0][0]
        assert "usaspending.gov" in req.payload["url"]
        assert req.payload["method"] == "POST"
        assert req.payload["source_name"] == "usa_spending"


# ---- SAMGovClient ----

class TestSAMGovRouting:
    """SAMGovClient routes GET through provider."""

    def test_search_opportunities_via_provider(self):
        provider = MagicMock()
        provider.execute.return_value = _success_response({
            "opportunitiesData": [{
                "noticeId": "OPP-001",
                "title": "Test Opportunity",
                "solicitationNumber": "SOL-001",
                "department": "DOD",
                "subtierAgency": "Navy",
            }],
        })

        from mae_core.market.apis.sam_gov import SAMGovClient
        client = SAMGovClient(api_key="test-key", provider=provider)
        results = client.search_opportunities(keywords="test")

        assert provider.execute.called
        req = provider.execute.call_args[0][0]
        assert "sam.gov" in req.payload["url"]
        assert req.payload["source_name"] == "sam_gov"
        assert len(results) == 1
        assert results[0].notice_id == "OPP-001"


# ---- PriceFetcher ----

class TestPriceFetcherRouting:
    """PriceFetcher routes Alpha Vantage through provider."""

    def test_alpha_vantage_via_provider(self):
        provider = MagicMock()
        provider.execute.return_value = _success_response({
            "Global Quote": {
                "05. price": "150.00",
                "02. open": "148.00",
                "03. high": "151.00",
                "04. low": "147.50",
                "06. volume": "1000000",
                "10. change percent": "1.5%",
            },
        })

        from mae_core.market.apis.price_fetcher import PriceFetcher
        client = PriceFetcher(alpha_vantage_key="test-key", provider=provider)
        # Bypass yfinance by calling _fetch_alpha_vantage directly
        result = client._fetch_alpha_vantage("AAPL")

        assert provider.execute.called
        req = provider.execute.call_args[0][0]
        assert "alphavantage.co" in req.payload["url"]
        assert req.payload["source_name"] == "alpha_vantage"
        assert result is not None
        assert result.price == 150.0
        assert result.symbol == "AAPL"


# ---- HouseStockWatcherClient ----

class TestHouseStockWatcherRouting:
    """HouseStockWatcherClient routes all HTTP through provider."""

    def test_request_via_provider(self):
        provider = MagicMock()
        provider.execute.return_value = _success_response([
            {
                "representative": "Test Rep",
                "district": "TX-01",
                "transaction_date": "2025-01-01",
                "ticker": "AAPL",
                "asset_description": "Apple Inc",
                "type": "purchase",
                "amount": "$1,001 - $15,000",
                "disclosure_date": "2025-02-01",
                "disclosure_url": "https://example.com",
                "owner": "Self",
            },
        ])

        from mae_core.market.apis.house_stock_watcher import HouseStockWatcherClient
        client = HouseStockWatcherClient(provider=provider)
        # _request is the shared internal method
        result = client._request("https://example.com/api", source_name="test")

        assert provider.execute.called
        req = provider.execute.call_args[0][0]
        assert req.payload["url"] == "https://example.com/api"
        assert req.payload["source_name"] == "test"
        assert isinstance(result, list)


# ---- JobTracker ----

class TestJobTrackerRouting:
    """JobTracker routes all HTTP through provider."""

    def test_request_via_provider(self):
        provider = MagicMock()
        provider.execute.return_value = _success_response({
            "data": [
                {"job_title": "Software Engineer", "job_posted_at_datetime_utc": "2025-01-01T00:00:00Z"},
            ],
        })

        from mae_core.market.apis.job_tracker import JobTracker
        client = JobTracker(provider=provider)
        result = client._request(
            "https://jsearch.p.rapidapi.com/search",
            headers={"x-rapidapi-host": "jsearch.p.rapidapi.com"},
            params={"query": "test"},
        )

        assert provider.execute.called
        req = provider.execute.call_args[0][0]
        assert "jsearch" in req.payload["url"]
        assert req.payload["source_name"] == "rapidapi"
        assert result is not None
        assert "data" in result


# ---- SECEdgarClient ----

class TestSECEdgarRouting:
    """SECEdgarClient routes _get() through provider with ResponseShim."""

    def test_get_json_via_provider(self):
        """JSON response: callers use .json() on the shim."""
        provider = MagicMock()
        provider.execute.return_value = _success_response({
            "cik": "0000320193",
            "entityType": "operating",
            "name": "Apple Inc.",
            "tickers": ["AAPL"],
        })

        from mae_core.market.apis.sec_edgar.client import SECEdgarClient
        client = SECEdgarClient(provider=provider)
        # Disable rate limiting for test speed
        client._last_request_time = 0

        response = client._get("https://data.sec.gov/submissions/CIK0000320193.json")

        assert provider.execute.called
        req = provider.execute.call_args[0][0]
        assert "sec.gov" in req.payload["url"]
        assert req.payload["source_name"] == "sec_edgar"

        # Verify shim interface
        assert response is not None
        assert response.json()["cik"] == "0000320193"
        assert response.status_code == 200

    def test_get_text_via_provider(self):
        """Non-JSON response: callers use .text on the shim."""
        provider = MagicMock()
        provider.execute.return_value = _success_response({
            "text": "<XML>form4 data</XML>",
        })

        from mae_core.market.apis.sec_edgar.client import SECEdgarClient
        client = SECEdgarClient(provider=provider)
        client._last_request_time = 0

        response = client._get("https://www.sec.gov/Archives/edgar/data/123/form4.xml")

        assert response is not None
        assert response.text == "<XML>form4 data</XML>"
        assert response.content == b"<XML>form4 data</XML>"
