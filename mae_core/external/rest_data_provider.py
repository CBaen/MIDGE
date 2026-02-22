"""RestDataProvider - Generic REST GET provider for financial/data APIs.

Handles MarketAux, Finnhub, and Alpha Vantage with different auth configs.
All three authenticate via a query parameter, so one class covers all three.

Biological analogy: Chemoreceptors — sensors that read the chemical state
of the environment. These providers let Mae sample the real-world signal
environment (financial markets, news sentiment) rather than only reasoning
internally.

Usage by agents (via TaskPool oracle pathway):
    payload = {
        "endpoint": "/quote",          # API path relative to base_url
        "params": {"symbol": "AAPL"},  # Provider-specific query params
    }

Provider configs:
    MarketAux:     base_url="https://api.marketaux.com/v1",  auth_param="api_token"
    Finnhub:       base_url="https://finnhub.io/api/v1",    auth_param="token"
    Alpha Vantage: base_url="https://www.alphavantage.co",  auth_param="apikey"
"""

from __future__ import annotations

import logging
import time

import httpx

from mae_core.external.api_client import (
    ApiRequest,
    ApiResponse,
    ApiResponseStatus,
    BaseProvider,
)

logger = logging.getLogger(__name__)


class RestDataProvider(BaseProvider):
    """Generic REST GET provider for financial/data APIs.

    Supports any API that authenticates via a single query parameter.
    Agent payload specifies the endpoint and any extra query params;
    the provider injects the API key automatically.

    Args:
        name: Provider name used for routing (e.g., "finnhub").
        api_key: API key for authentication.
        base_url: API base URL (no trailing slash).
        auth_param: Name of the query param that carries the API key.
        default_endpoint: Fallback endpoint if payload omits "endpoint".
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        auth_param: str = "apikey",
        default_endpoint: str = "",
    ) -> None:
        self.provider_name = name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._auth_param = auth_param
        self._default_endpoint = default_endpoint

    @property
    def available(self) -> bool:
        """True if an API key is configured."""
        return bool(self._api_key)

    def execute(self, request: ApiRequest, timeout_ms: float = 30000.0) -> ApiResponse:
        """Execute a GET request to the data API.

        Payload keys:
            endpoint (str): API path, e.g. "/quote" or "query". Relative to base_url.
            params (dict): Extra query parameters for the request.
            question (str): Fallback — if no endpoint given, used as a search term.
        """
        if not self._api_key:
            return ApiResponse(
                status=ApiResponseStatus.ERROR,
                error_message=f"{self.provider_name}: no API key configured",
                provider=self.provider_name,
            )

        start = time.time()

        endpoint = request.payload.get("endpoint", self._default_endpoint)
        extra_params: dict = request.payload.get("params", {})

        # Auth key always added; caller params merged on top
        params = {self._auth_param: self._api_key}
        params.update(extra_params)

        url = f"{self._base_url}/{endpoint.lstrip('/')}"

        try:
            response = httpx.get(
                url,
                params=params,
                timeout=timeout_ms / 1000.0,
            )
            latency_ms = (time.time() - start) * 1000

            if response.status_code == 429:
                return ApiResponse(
                    status=ApiResponseStatus.RATE_LIMITED,
                    error_message=f"{self.provider_name}: rate limit exceeded",
                    provider=self.provider_name,
                    latency_ms=latency_ms,
                )

            if response.status_code >= 500:
                return ApiResponse(
                    status=ApiResponseStatus.PROVIDER_ERROR,
                    error_message=f"{self.provider_name}: HTTP {response.status_code}",
                    provider=self.provider_name,
                    latency_ms=latency_ms,
                )

            if response.status_code >= 400:
                return ApiResponse(
                    status=ApiResponseStatus.ERROR,
                    error_message=(
                        f"{self.provider_name}: HTTP {response.status_code} — "
                        f"{response.text[:200]}"
                    ),
                    provider=self.provider_name,
                    latency_ms=latency_ms,
                )

            data = response.json()
            logger.info(
                "%s responded — %.0fms, endpoint=%s",
                self.provider_name, latency_ms, endpoint or "(root)",
            )
            return ApiResponse(
                status=ApiResponseStatus.SUCCESS,
                payload=data,
                provider=self.provider_name,
                latency_ms=latency_ms,
            )

        except httpx.TimeoutException:
            latency_ms = (time.time() - start) * 1000
            return ApiResponse(
                status=ApiResponseStatus.TIMEOUT,
                error_message=f"{self.provider_name}: timed out after {timeout_ms:.0f}ms",
                provider=self.provider_name,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            logger.warning("%s request failed: %s", self.provider_name, exc)
            return ApiResponse(
                status=ApiResponseStatus.ERROR,
                error_message=str(exc),
                provider=self.provider_name,
                latency_ms=latency_ms,
            )
