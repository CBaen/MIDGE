"""OpenAI-compatible provider - Mae's polyglot voice.

Biological analogy: Multiple oracle pathways. Where LlmProvider speaks
only one dialect (Anthropic), this provider speaks the lingua franca
of LLM APIs (OpenAI format). Groq, Mistral, DeepSeek, and many others
all understand this common tongue.

One class, many endpoints. Parameterized by base_url, model, and API key.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from mae_core.external.api_client import (
    ApiRequest,
    ApiResponse,
    ApiResponseStatus,
    BaseProvider,
)

logger = logging.getLogger(__name__)


# Mae-aware system prompt (same intent as LlmProvider)
DEFAULT_SYSTEM_PROMPT = (
    "You are being consulted by Mae, a biologically-inspired multi-agent "
    "AI organism. An agent within Mae has encountered a novel situation "
    "that its internal decision cascade could not resolve. Provide a "
    "concise, actionable response. Focus on the specific question asked."
)


class OpenAIProvider(BaseProvider):
    """Provider for any OpenAI-compatible chat completions API.

    Handles Groq, Mistral, DeepSeek, and any other API that follows
    the OpenAI chat completions format. Uses httpx for HTTP calls.

    Parameterized by:
        provider_name: Identifier for this provider (e.g., "groq")
        api_key: Bearer token for authentication
        base_url: API base URL (e.g., "https://api.groq.com/openai/v1")
        model: Default model name (e.g., "llama-3.3-70b-versatile")
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.provider_name = name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # Persistent HTTP client — reuses TCP connections and SSL sessions.
        # Eliminates ~5s of SSL handshake overhead per run.
        self._http_client = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=120,
            ),
        )

        # Statistics
        self._total_requests: int = 0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0

    @property
    def available(self) -> bool:
        """Whether this provider has a valid API key configured."""
        return bool(self._api_key)

    def build_prompt(
        self,
        question: str,
        agent_context: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, str]]:
        """Build OpenAI-format messages from Mae's internal format.

        Same logic as LlmProvider.build_prompt — assembles agent context
        and question into a clean user message.
        """
        messages = []

        # System message
        messages.append({"role": "system", "content": self._system_prompt})

        # Build user message with context
        parts = []
        if agent_context:
            context_lines = []
            if "agent_id" in agent_context:
                context_lines.append(f"Agent: {agent_context['agent_id']}")
            if "role" in agent_context:
                context_lines.append(f"Role: {agent_context['role']}")
            if "prediction_error" in agent_context:
                context_lines.append(
                    f"Prediction error: {agent_context['prediction_error']:.2f}"
                )
            if "recent_actions" in agent_context:
                actions = agent_context["recent_actions"]
                context_lines.append(
                    f"Recent actions: {', '.join(str(a) for a in actions[-3:])}"
                )
            if "energy_level" in agent_context:
                context_lines.append(f"Energy: {agent_context['energy_level']:.2f}")

            if context_lines:
                parts.append("Context:\n" + "\n".join(context_lines))

        parts.append(f"Question: {question}")
        messages.append({"role": "user", "content": "\n\n".join(parts)})

        return messages

    def execute(
        self,
        request: ApiRequest,
        timeout_ms: float = 30000.0,
    ) -> ApiResponse:
        """Execute a chat completion via OpenAI-compatible API."""
        self._total_requests += 1

        if not self._api_key:
            return ApiResponse(
                status=ApiResponseStatus.ERROR,
                error_message="No API key configured",
                provider=self.provider_name,
            )

        question = request.payload.get("question", "")
        agent_context = request.payload.get("context", {})
        messages = self.build_prompt(question, agent_context)

        start = time.time()

        try:
            response = self._http_client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    "messages": messages,
                    "max_tokens": self._max_tokens,
                },
                timeout=timeout_ms / 1000.0,
            )

            latency = (time.time() - start) * 1000

            if response.status_code == 429:
                return ApiResponse(
                    status=ApiResponseStatus.RATE_LIMITED,
                    error_message=f"{self.provider_name} rate limit",
                    provider=self.provider_name,
                    latency_ms=latency,
                )

            if response.status_code != 200:
                return ApiResponse(
                    status=ApiResponseStatus.PROVIDER_ERROR,
                    error_message=f"HTTP {response.status_code}: {response.text[:200]}",
                    provider=self.provider_name,
                    latency_ms=latency,
                )

            data = response.json()

            # Parse OpenAI-format response
            tokens_in = data.get("usage", {}).get("prompt_tokens", 0)
            tokens_out = data.get("usage", {}).get("completion_tokens", 0)
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

            # Extract text from choices
            text = ""
            choices = data.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")

            return ApiResponse(
                status=ApiResponseStatus.SUCCESS,
                payload={
                    "text": text,
                    "model": data.get("model", self._model),
                    "stop_reason": choices[0].get("finish_reason", "") if choices else "",
                },
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                provider=self.provider_name,
                latency_ms=latency,
            )

        except httpx.TimeoutException:
            latency = (time.time() - start) * 1000
            return ApiResponse(
                status=ApiResponseStatus.TIMEOUT,
                error_message=f"{self.provider_name} timeout",
                provider=self.provider_name,
                latency_ms=latency,
            )
        except Exception as exc:
            latency = (time.time() - start) * 1000
            return ApiResponse(
                status=ApiResponseStatus.ERROR,
                error_message=str(exc),
                provider=self.provider_name,
                latency_ms=latency,
            )

    def close(self) -> None:
        """Close the persistent HTTP client and release connections."""
        try:
            self._http_client.close()
        except Exception:
            pass

    def get_statistics(self) -> dict[str, Any]:
        """Provider statistics for monitoring."""
        return {
            "provider": self.provider_name,
            "model": self._model,
            "base_url": self._base_url,
            "available": self.available,
            "total_requests": self._total_requests,
            "total_tokens_in": self._total_tokens_in,
            "total_tokens_out": self._total_tokens_out,
        }
