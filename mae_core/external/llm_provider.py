"""LlmProvider - Claude API adapter for Mae.

Biological analogy: The oracle pathway. When Mae's internal decision
cascade reaches the prefrontal tier and encounters genuine novelty,
this pathway lets her ask an external intelligence for guidance.

Like asking a wise elder: the question must be well-formed, the answer
must be validated, and the knowledge must be integrated into memory.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from mae_core.external.api_client import (
    ApiRequest,
    ApiResponse,
    ApiResponseStatus,
    BaseProvider,
)

logger = logging.getLogger(__name__)


class LlmProvider(BaseProvider):
    """Claude API adapter. Translates Mae's internal request format
    to Anthropic's API format and parses responses.

    Handles:
    - Prompt construction from agent state + question + context
    - Response parsing and token counting
    - System prompt for Mae-aware responses
    - Graceful fallback when API is unavailable
    """

    provider_name = "claude"

    # Default system prompt: tells Claude it's being consulted by Mae
    DEFAULT_SYSTEM_PROMPT = (
        "You are being consulted by Mae, a biologically-inspired multi-agent "
        "AI organism. An agent within Mae has encountered a novel situation "
        "that its internal decision cascade could not resolve. Provide a "
        "concise, actionable response. Focus on the specific question asked."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250514",
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("MAE_CLAUDE_API_KEY", "")
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

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
        """Build Claude API messages from Mae's internal format.

        Takes the agent's question and optional context (state vector,
        recent experiences, prediction error) and formats them into
        a clean prompt.
        """
        messages = []

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
                context_lines.append(f"Recent actions: {', '.join(str(a) for a in actions[-3:])}")
            if "energy_level" in agent_context:
                context_lines.append(f"Energy: {agent_context['energy_level']:.2f}")

            if context_lines:
                parts.append("Context:\n" + "\n".join(context_lines))

        parts.append(f"Question: {question}")

        messages.append({
            "role": "user",
            "content": "\n\n".join(parts),
        })

        return messages

    def execute(
        self,
        request: ApiRequest,
        timeout_ms: float = 30000.0,
    ) -> ApiResponse:
        """Execute a Claude API call.

        This is the actual HTTP call. Uses the anthropic library if
        available, falls back to httpx for raw HTTP.
        """
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
            # Try anthropic library first
            response = self._call_anthropic(messages, timeout_ms)
        except ImportError:
            # Fall back to httpx
            try:
                response = self._call_httpx(messages, timeout_ms)
            except ImportError:
                return ApiResponse(
                    status=ApiResponseStatus.ERROR,
                    error_message="Neither anthropic nor httpx library available",
                    provider=self.provider_name,
                )

        latency = (time.time() - start) * 1000
        response.latency_ms = latency
        return response

    def _call_anthropic(
        self,
        messages: list[dict[str, str]],
        timeout_ms: float,
    ) -> ApiResponse:
        """Call Claude via the anthropic Python library."""
        import anthropic

        client = anthropic.Anthropic(
            api_key=self._api_key,
            timeout=timeout_ms / 1000.0,
        )

        try:
            result = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                messages=messages,
            )

            tokens_in = getattr(result.usage, "input_tokens", 0)
            tokens_out = getattr(result.usage, "output_tokens", 0)
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

            # Extract text from response
            text = ""
            for block in result.content:
                if hasattr(block, "text"):
                    text += block.text

            return ApiResponse(
                status=ApiResponseStatus.SUCCESS,
                payload={
                    "text": text,
                    "model": result.model,
                    "stop_reason": result.stop_reason,
                },
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                provider=self.provider_name,
            )

        except anthropic.RateLimitError:
            return ApiResponse(
                status=ApiResponseStatus.RATE_LIMITED,
                error_message="Claude API rate limit",
                provider=self.provider_name,
            )
        except anthropic.APITimeoutError:
            return ApiResponse(
                status=ApiResponseStatus.TIMEOUT,
                error_message="Claude API timeout",
                provider=self.provider_name,
            )
        except anthropic.APIError as exc:
            return ApiResponse(
                status=ApiResponseStatus.PROVIDER_ERROR,
                error_message=str(exc),
                provider=self.provider_name,
            )

    def _call_httpx(
        self,
        messages: list[dict[str, str]],
        timeout_ms: float,
    ) -> ApiResponse:
        """Call Claude via raw httpx HTTP client (fallback)."""
        import httpx

        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": self._max_tokens,
                    "system": self._system_prompt,
                    "messages": messages,
                },
                timeout=timeout_ms / 1000.0,
            )

            if response.status_code == 429:
                return ApiResponse(
                    status=ApiResponseStatus.RATE_LIMITED,
                    error_message="Claude API rate limit",
                    provider=self.provider_name,
                )

            if response.status_code != 200:
                return ApiResponse(
                    status=ApiResponseStatus.PROVIDER_ERROR,
                    error_message=f"HTTP {response.status_code}: {response.text[:200]}",
                    provider=self.provider_name,
                )

            data = response.json()
            tokens_in = data.get("usage", {}).get("input_tokens", 0)
            tokens_out = data.get("usage", {}).get("output_tokens", 0)
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

            # Extract text
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")

            return ApiResponse(
                status=ApiResponseStatus.SUCCESS,
                payload={
                    "text": text,
                    "model": data.get("model", self._model),
                    "stop_reason": data.get("stop_reason", ""),
                },
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                provider=self.provider_name,
            )

        except httpx.TimeoutException:
            return ApiResponse(
                status=ApiResponseStatus.TIMEOUT,
                error_message="Claude API timeout (httpx)",
                provider=self.provider_name,
            )

    def get_statistics(self) -> dict[str, Any]:
        """Provider statistics for monitoring."""
        return {
            "provider": self.provider_name,
            "model": self._model,
            "available": self.available,
            "total_requests": self._total_requests,
            "total_tokens_in": self._total_tokens_in,
            "total_tokens_out": self._total_tokens_out,
        }
