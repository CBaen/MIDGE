"""ApiClient - Thin HTTP client wrapper for external API calls.

Biological analogy: The neuromuscular junction. Translates nervous system
signals (Mae's requests) into motor actions (HTTP calls) and returns
sensory feedback (responses). Handles retry, timeout, and rate limiting
like a motor neuron handles muscle fatigue.

Used only by ApiGateway — agents never call this directly.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ApiResponseStatus(Enum):
    """Outcome of an API call."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    PROVIDER_ERROR = "provider_error"


@dataclass
class ApiResponse:
    """Result of an external API call.

    Carries the response payload, status, timing info, and token counts
    for LLM calls. Immutable record of what happened.
    """

    status: ApiResponseStatus
    payload: Any = None
    error_message: str = ""
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    provider: str = ""
    attempt: int = 1
    timestamp: float = field(default_factory=time.time)


@dataclass
class ApiRequest:
    """A request to send to an external API.

    Created by agents (via ExternalTask), consumed by ApiGateway,
    executed by ApiClient.
    """

    request_id: str
    provider: str
    endpoint: str
    payload: dict[str, Any]
    method: str = "POST"
    priority: int = 5  # 1=highest, 10=lowest
    timeout_ms: float = 30000.0
    max_retries: int = 2
    agent_id: str = ""
    created_at: float = field(default_factory=time.time)


class ApiClient:
    """Thin HTTP client wrapper with retry logic, timeouts, and rate limiting.

    Biological analogy: The motor end plate. Converts neural signals into
    physical actions. Handles the mechanics of reliable communication
    with the outside world.

    This class is provider-agnostic — it delegates actual API formatting
    to provider instances (LlmProvider, etc.). The client handles:
    - Retry with exponential backoff
    - Per-provider rate limiting (token bucket)
    - Timeout enforcement
    - Response wrapping
    """

    def __init__(
        self,
        max_retries: int = 2,
        base_timeout_ms: float = 30000.0,
        rate_limit_rpm: int = 60,
    ) -> None:
        self._max_retries = max_retries
        self._base_timeout_ms = base_timeout_ms
        self._rate_limit_rpm = rate_limit_rpm

        # Token bucket for rate limiting
        self._tokens = float(rate_limit_rpm)
        self._max_tokens = float(rate_limit_rpm)
        self._last_refill = time.time()
        self._refill_rate = rate_limit_rpm / 60.0  # tokens per second

        # Statistics
        self._total_calls: int = 0
        self._total_successes: int = 0
        self._total_failures: int = 0
        self._total_retries: int = 0
        self._total_rate_limited: int = 0
        self._total_latency_ms: float = 0.0

    def call(
        self,
        provider: "BaseProvider",
        request: ApiRequest,
    ) -> ApiResponse:
        """Execute an API call through a provider with retry and rate limiting.

        Args:
            provider: The provider instance that knows how to format/parse
            request: The request to execute

        Returns:
            ApiResponse with status, payload, and metadata
        """
        self._total_calls += 1

        # Rate limit check
        if not self._try_consume_token():
            self._total_rate_limited += 1
            logger.warning(
                "ApiClient: rate limited (provider=%s, request=%s)",
                request.provider, request.request_id,
            )
            return ApiResponse(
                status=ApiResponseStatus.RATE_LIMITED,
                error_message="Rate limit exceeded",
                provider=request.provider,
            )

        max_retries = min(request.max_retries, self._max_retries)
        timeout_ms = min(request.timeout_ms, self._base_timeout_ms)

        last_error = ""
        for attempt in range(1, max_retries + 1):
            start = time.time()
            try:
                result = provider.execute(request, timeout_ms=timeout_ms)
                latency = (time.time() - start) * 1000
                self._total_latency_ms += latency

                if result.status == ApiResponseStatus.SUCCESS:
                    self._total_successes += 1
                    result.latency_ms = latency
                    result.attempt = attempt
                    return result
                else:
                    last_error = result.error_message
                    if attempt < max_retries:
                        self._total_retries += 1
                        backoff = 0.1 * (2 ** (attempt - 1)) * (0.5 + random.random())
                        time.sleep(backoff)

            except Exception as exc:
                latency = (time.time() - start) * 1000
                self._total_latency_ms += latency
                last_error = str(exc)
                logger.debug(
                    "ApiClient: attempt %d/%d failed: %s",
                    attempt, max_retries, exc,
                )
                if attempt < max_retries:
                    self._total_retries += 1
                    backoff = 0.1 * (2 ** (attempt - 1))
                    time.sleep(backoff)

        self._total_failures += 1
        return ApiResponse(
            status=ApiResponseStatus.ERROR,
            error_message=f"All {max_retries} attempts failed: {last_error}",
            provider=request.provider,
            attempt=max_retries,
        )

    def _try_consume_token(self) -> bool:
        """Token bucket rate limiter. Returns True if a token was consumed."""
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._max_tokens,
            self._tokens + elapsed * self._refill_rate,
        )
        self._last_refill = now

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def get_statistics(self) -> dict[str, Any]:
        """Return client statistics for monitoring."""
        return {
            "total_calls": self._total_calls,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "total_retries": self._total_retries,
            "total_rate_limited": self._total_rate_limited,
            "avg_latency_ms": (
                self._total_latency_ms / max(1, self._total_calls)
            ),
            "success_rate": (
                self._total_successes / max(1, self._total_calls)
            ),
            "rate_limit_rpm": self._rate_limit_rpm,
            "tokens_remaining": self._tokens,
        }


class BaseProvider:
    """Base class for API providers.

    Each provider knows how to format requests for its specific API
    and parse responses back into Mae's internal format.
    """

    provider_name: str = "base"

    def execute(
        self, request: ApiRequest, timeout_ms: float = 30000.0,
    ) -> ApiResponse:
        """Execute a request. Subclasses must override."""
        raise NotImplementedError
