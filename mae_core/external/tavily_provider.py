"""TavilyProvider - Web search provider for Mae.

Connects Mae to real-time web search via the Tavily API. Unlike the
financial data providers (MarketAux, Finnhub, Alpha Vantage) which return
structured data, Tavily returns web search results with titles, URLs, and
content snippets.

Biological analogy: Olfactory receptors — sampling the scent of the broader
world. Where chemoreceptors (RestDataProvider) sample specific chemicals
(financial signals), olfactory receptors sweep the whole environment for
novel signals. Web search lets Mae discover things she wasn't explicitly
looking for.

Usage by agents (via TaskPool oracle pathway):
    payload = {
        "query": "Fed interest rate decision 2026",  # required
        "max_results": 5,                            # optional, default 5
        "search_depth": "basic",                     # optional: "basic" | "advanced"
        "include_domains": [],                       # optional domain filter
    }

Response payload:
    {
        "query": "...",
        "results": [
            {"title": "...", "url": "...", "content": "...", "score": 0.95},
            ...
        ],
        "answer": "...",   # AI-generated answer if available
    }
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

TAVILY_URL = "https://api.tavily.com/search"


class TavilyProvider(BaseProvider):
    """Web search provider using the Tavily API.

    Agent payload specifies the query and optional search options.
    Returns a structured list of web results with titles, URLs, and snippets.

    Args:
        api_key: Tavily API key.
        default_max_results: Default number of results if not specified in payload.
        default_depth: Default search depth ("basic" or "advanced").
    """

    provider_name = "tavily"

    def __init__(
        self,
        api_key: str,
        default_max_results: int = 5,
        default_depth: str = "basic",
    ) -> None:
        self._api_key = api_key
        self._default_max_results = default_max_results
        self._default_depth = default_depth

    @property
    def available(self) -> bool:
        """True if an API key is configured."""
        return bool(self._api_key)

    def execute(self, request: ApiRequest, timeout_ms: float = 30000.0) -> ApiResponse:
        """Execute a web search via Tavily.

        Payload keys:
            query (str): The search query. Required.
            max_results (int): Number of results to return. Default 5.
            search_depth (str): "basic" (fast) or "advanced" (deeper). Default "basic".
            include_domains (list[str]): Restrict to these domains. Optional.
            exclude_domains (list[str]): Exclude these domains. Optional.
            include_answer (bool): Include AI-generated answer. Default False.
        """
        if not self._api_key:
            return ApiResponse(
                status=ApiResponseStatus.ERROR,
                error_message="tavily: no API key configured",
                provider=self.provider_name,
            )

        query = request.payload.get("query", "")
        if not query:
            # Fallback: use the "question" key (same convention as LlmProvider)
            query = request.payload.get("question", "")

        if not query:
            return ApiResponse(
                status=ApiResponseStatus.ERROR,
                error_message="tavily: no query provided in payload",
                provider=self.provider_name,
            )

        body = {
            "api_key": self._api_key,
            "query": query,
            "max_results": request.payload.get("max_results", self._default_max_results),
            "search_depth": request.payload.get("search_depth", self._default_depth),
            "include_answer": request.payload.get("include_answer", False),
        }

        # Optional domain filters
        include_domains = request.payload.get("include_domains", [])
        if include_domains:
            body["include_domains"] = include_domains

        exclude_domains = request.payload.get("exclude_domains", [])
        if exclude_domains:
            body["exclude_domains"] = exclude_domains

        start = time.time()

        try:
            response = httpx.post(
                TAVILY_URL,
                json=body,
                timeout=timeout_ms / 1000.0,
            )
            latency_ms = (time.time() - start) * 1000

            if response.status_code == 429:
                return ApiResponse(
                    status=ApiResponseStatus.RATE_LIMITED,
                    error_message="tavily: rate limit exceeded",
                    provider=self.provider_name,
                    latency_ms=latency_ms,
                )

            if response.status_code >= 500:
                return ApiResponse(
                    status=ApiResponseStatus.PROVIDER_ERROR,
                    error_message=f"tavily: HTTP {response.status_code}",
                    provider=self.provider_name,
                    latency_ms=latency_ms,
                )

            if response.status_code >= 400:
                return ApiResponse(
                    status=ApiResponseStatus.ERROR,
                    error_message=(
                        f"tavily: HTTP {response.status_code} — "
                        f"{response.text[:200]}"
                    ),
                    provider=self.provider_name,
                    latency_ms=latency_ms,
                )

            data = response.json()

            # Normalize: expose results + answer in a consistent structure
            results = data.get("results", [])
            answer = data.get("answer", "")

            logger.info(
                "tavily responded — %.0fms, %d results for: %s",
                latency_ms, len(results), query[:60],
            )

            return ApiResponse(
                status=ApiResponseStatus.SUCCESS,
                payload={
                    "query": query,
                    "results": results,
                    "answer": answer,
                    # Also expose as "text" for compatibility with LLM result readers
                    "text": answer or (results[0]["content"] if results else ""),
                },
                provider=self.provider_name,
                latency_ms=latency_ms,
            )

        except httpx.TimeoutException:
            latency_ms = (time.time() - start) * 1000
            return ApiResponse(
                status=ApiResponseStatus.TIMEOUT,
                error_message=f"tavily: timed out after {timeout_ms:.0f}ms",
                provider=self.provider_name,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            logger.warning("tavily request failed: %s", exc)
            return ApiResponse(
                status=ApiResponseStatus.ERROR,
                error_message=str(exc),
                provider=self.provider_name,
                latency_ms=latency_ms,
            )
