"""MarketDataProvider - BaseProvider for market intelligence HTTP calls.

Routes market API requests through ApiGateway's BoundaryMembrane and
InputValidator, providing security screening and centralized monitoring
for all outbound market data requests.

Supports GET (SEC EDGAR, HouseStockWatcher, SAM.gov) and POST
(USASpending) with both header-based (RapidAPI) and query-param-based
(SAM.gov, Alpha Vantage) authentication styles.

Payload contract:
    {
        "url": "https://api.usaspending.gov/api/v2/search/...",
        "method": "GET" | "POST",          # default GET
        "headers": {},                     # optional extra headers
        "params": {},                      # query params
        "json_body": {},                   # POST JSON body (only for POST)
        "source_name": "usa_spending",     # for BoundaryMembrane classification
    }
"""

from __future__ import annotations

import logging
import time
import uuid

import requests

from mae_core.external.api_client import (
    ApiRequest,
    ApiResponse,
    ApiResponseStatus,
    BaseProvider,
)

logger = logging.getLogger(__name__)


class MarketDataProvider(BaseProvider):
    """Generic HTTP provider for market data APIs.

    Unlike RestDataProvider (GET-only, query-param auth), this supports
    the full range of patterns used by MIDGE's 6 market API clients.
    """

    provider_name = "market_data"

    def __init__(self, default_timeout_s: float = 30.0) -> None:
        self._session = requests.Session()
        self._default_timeout = default_timeout_s

    def execute(
        self, request: ApiRequest, timeout_ms: float = 30000.0,
    ) -> ApiResponse:
        """Execute an HTTP request for market data."""
        start = time.time()
        payload = request.payload or {}

        url = payload.get("url", "")
        if not url:
            return ApiResponse(
                status=ApiResponseStatus.ERROR,
                error_message="market_data: no URL in payload",
                provider=self.provider_name,
            )

        method = payload.get("method", "GET").upper()
        headers = payload.get("headers", {})
        params = payload.get("params", {})
        json_body = payload.get("json_body")
        timeout_s = min(timeout_ms / 1000.0, self._default_timeout)

        try:
            if method == "POST":
                resp = self._session.post(
                    url, headers=headers, params=params,
                    json=json_body, timeout=timeout_s,
                )
            else:
                resp = self._session.get(
                    url, headers=headers, params=params,
                    timeout=timeout_s,
                )

            latency_ms = (time.time() - start) * 1000

            if resp.status_code == 429:
                return ApiResponse(
                    status=ApiResponseStatus.RATE_LIMITED,
                    error_message=f"market_data: HTTP 429 from {url}",
                    provider=self.provider_name,
                    latency_ms=latency_ms,
                )

            if resp.status_code >= 500:
                return ApiResponse(
                    status=ApiResponseStatus.PROVIDER_ERROR,
                    error_message=f"market_data: HTTP {resp.status_code}",
                    provider=self.provider_name,
                    latency_ms=latency_ms,
                )

            if resp.status_code >= 400:
                return ApiResponse(
                    status=ApiResponseStatus.ERROR,
                    error_message=f"market_data: HTTP {resp.status_code} — {resp.text[:200]}",
                    provider=self.provider_name,
                    latency_ms=latency_ms,
                )

            # Try JSON first, fall back to text
            try:
                data = resp.json()
            except ValueError:
                data = {"text": resp.text}

            return ApiResponse(
                status=ApiResponseStatus.SUCCESS,
                payload=data,
                provider=self.provider_name,
                latency_ms=latency_ms,
            )

        except requests.Timeout:
            latency_ms = (time.time() - start) * 1000
            return ApiResponse(
                status=ApiResponseStatus.TIMEOUT,
                error_message=f"market_data: timed out after {timeout_s:.0f}s",
                provider=self.provider_name,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            logger.warning("market_data request failed: %s", exc)
            return ApiResponse(
                status=ApiResponseStatus.ERROR,
                error_message=str(exc),
                provider=self.provider_name,
                latency_ms=latency_ms,
            )

    def close(self) -> None:
        """Close the underlying session."""
        self._session.close()
