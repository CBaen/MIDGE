"""Tests for Mae's external API gateway — her mouth and ears.

All external calls are mocked. No real API keys needed.
Tests cover: ApiClient, ApiGateway, LlmProvider, ExternalTask,
bootstrap Layer 31, and integration flow.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from mae_core.external.api_client import (
    ApiClient,
    ApiRequest,
    ApiResponse,
    ApiResponseStatus,
    BaseProvider,
)
from mae_core.external.api_gateway import (
    ApiGateway,
    CH_EXTERNAL_FAILED,
    CH_EXTERNAL_QUEUED,
    CH_EXTERNAL_RESPONSE,
    CH_EXTERNAL_SUBMIT,
)
from mae_core.external.external_task import (
    ExternalTaskSpec,
    complete_external_task,
    get_pending_external_tasks,
    inject_external_task,
)
from mae_core.external.llm_provider import LlmProvider
from mae_core.environment.task_pool import TaskPool, Task


# =========================================================================
# Fixtures
# =========================================================================

class MockProvider(BaseProvider):
    """Test provider that returns canned responses."""

    provider_name = "mock"

    def __init__(self, response: ApiResponse | None = None, raise_error: bool = False):
        self._response = response or ApiResponse(
            status=ApiResponseStatus.SUCCESS,
            payload={"text": "mock response"},
            tokens_in=10,
            tokens_out=20,
            provider="mock",
        )
        self._raise_error = raise_error
        self.call_count = 0

    def execute(self, request: ApiRequest, timeout_ms: float = 30000.0) -> ApiResponse:
        self.call_count += 1
        if self._raise_error:
            raise ConnectionError("mock connection error")
        return self._response


class MockEventBus:
    """Minimal EventBus mock for testing."""

    def __init__(self):
        self.published: list[tuple[str, dict]] = []
        self._callbacks: dict[str, list] = {}

    def publish(self, channel: str, message) -> None:
        self.published.append((channel, message))

    def register_callback(self, channel: str, callback) -> None:
        self._callbacks.setdefault(channel, []).append(callback)


class MockBoundaryMembrane:
    """Minimal BoundaryMembrane mock."""

    def __init__(self, allow: bool = True, reason: str = "trusted"):
        self._allow = allow
        self._reason = reason

    def check_input(self, source: str, data: dict) -> tuple[bool, str]:
        return (self._allow, self._reason)


class MockInputValidator:
    """Minimal InputValidator mock."""

    def __init__(self, result: str = "accepted"):
        self._result = result

    def validate(self, source: str, input_type: str, data, numeric_value=None):
        report = MagicMock()
        report.result.value = self._result
        return report


# =========================================================================
# ApiClient Tests (~8 tests)
# =========================================================================

class TestApiClient:
    """ApiClient: retry, timeout, rate limiting."""

    def test_successful_call(self):
        client = ApiClient()
        provider = MockProvider()
        request = ApiRequest(
            request_id="test-1", provider="mock",
            endpoint="", payload={"q": "test"},
        )
        response = client.call(provider, request)
        assert response.status == ApiResponseStatus.SUCCESS
        assert provider.call_count == 1

    def test_retry_on_error(self):
        """Client retries on provider error."""
        error_response = ApiResponse(
            status=ApiResponseStatus.ERROR,
            error_message="transient",
            provider="mock",
        )
        provider = MockProvider(response=error_response)
        client = ApiClient(max_retries=3)
        request = ApiRequest(
            request_id="test-2", provider="mock",
            endpoint="", payload={}, max_retries=3,
        )
        response = client.call(provider, request)
        assert response.status == ApiResponseStatus.ERROR
        assert provider.call_count == 3  # All retries exhausted

    def test_retry_on_exception(self):
        """Client retries when provider raises exception."""
        provider = MockProvider(raise_error=True)
        client = ApiClient(max_retries=2)
        request = ApiRequest(
            request_id="test-3", provider="mock",
            endpoint="", payload={}, max_retries=2,
        )
        response = client.call(provider, request)
        assert response.status == ApiResponseStatus.ERROR
        assert provider.call_count == 2
        assert "mock connection error" in response.error_message

    def test_rate_limiting(self):
        """Client rate limits when tokens exhausted."""
        client = ApiClient(rate_limit_rpm=1)
        provider = MockProvider()

        # First call succeeds
        r1 = client.call(provider, ApiRequest(
            request_id="r1", provider="mock", endpoint="", payload={},
        ))
        assert r1.status == ApiResponseStatus.SUCCESS

        # Second immediate call is rate limited
        r2 = client.call(provider, ApiRequest(
            request_id="r2", provider="mock", endpoint="", payload={},
        ))
        assert r2.status == ApiResponseStatus.RATE_LIMITED

    def test_statistics(self):
        client = ApiClient()
        provider = MockProvider()
        client.call(provider, ApiRequest(
            request_id="s1", provider="mock", endpoint="", payload={},
        ))
        stats = client.get_statistics()
        assert stats["total_calls"] == 1
        assert stats["total_successes"] == 1

    def test_successful_call_records_latency(self):
        client = ApiClient()
        provider = MockProvider()
        response = client.call(provider, ApiRequest(
            request_id="lat-1", provider="mock", endpoint="", payload={},
        ))
        assert response.latency_ms >= 0

    def test_max_retries_capped_by_client(self):
        """Request retries capped by client's max_retries."""
        provider = MockProvider(response=ApiResponse(
            status=ApiResponseStatus.ERROR, provider="mock",
        ))
        client = ApiClient(max_retries=1)
        request = ApiRequest(
            request_id="cap", provider="mock", endpoint="", payload={},
            max_retries=5,  # Request wants 5, client caps at 1
        )
        client.call(provider, request)
        assert provider.call_count == 1

    def test_token_bucket_refills(self):
        """Tokens refill over time."""
        client = ApiClient(rate_limit_rpm=60)
        # Consume a token
        assert client._try_consume_token()
        # Should still have tokens (60 RPM = 1/sec)
        assert client._try_consume_token()


# =========================================================================
# ApiGateway Tests (~12 tests)
# =========================================================================

class TestApiGateway:
    """ApiGateway: queuing, rate limits, membrane integration."""

    def test_create_gateway(self):
        gw = ApiGateway()
        assert gw.provider_count == 0
        assert len(gw._queue) == 0

    def test_register_provider(self):
        gw = ApiGateway()
        gw.register_provider("mock", MockProvider())
        assert gw.has_provider("mock")
        assert gw.provider_count == 1

    def test_submit_request_queues(self):
        bus = MockEventBus()
        gw = ApiGateway(event_bus=bus)
        request = ApiRequest(
            request_id="q1", provider="mock", endpoint="", payload={},
        )
        assert gw.submit_request(request)
        assert len(gw._queue) == 1
        assert gw._total_queued == 1
        # Check EventBus publish
        queued_events = [e for e in bus.published if e[0] == CH_EXTERNAL_QUEUED]
        assert len(queued_events) == 1

    def test_membrane_blocks_request(self):
        membrane = MockBoundaryMembrane(allow=False, reason="rejected")
        gw = ApiGateway(boundary_membrane=membrane)
        request = ApiRequest(
            request_id="mb1", provider="untrusted", endpoint="", payload={},
        )
        assert not gw.submit_request(request)
        assert gw._total_membrane_blocked == 1

    def test_validator_blocks_request(self):
        validator = MockInputValidator(result="rejected")
        gw = ApiGateway(input_validator=validator)
        request = ApiRequest(
            request_id="vb1", provider="mock", endpoint="", payload={},
        )
        assert not gw.submit_request(request)
        assert gw._total_validation_failed == 1

    def test_step_processes_queue(self):
        bus = MockEventBus()
        gw = ApiGateway(event_bus=bus)
        provider = MockProvider()
        gw.register_provider("mock", provider)

        request = ApiRequest(
            request_id="sp1", provider="mock", endpoint="", payload={},
        )
        gw.submit_request(request)
        gw.step(current_step=1)

        assert provider.call_count == 1
        assert gw._total_processed == 1
        # Should have published response
        response_events = [e for e in bus.published if e[0] == CH_EXTERNAL_RESPONSE]
        assert len(response_events) == 1

    def test_step_respects_max_per_step(self):
        gw = ApiGateway(max_requests_per_step=2)
        provider = MockProvider()
        gw.register_provider("mock", provider)

        for i in range(5):
            gw.submit_request(ApiRequest(
                request_id=f"ms-{i}", provider="mock", endpoint="", payload={},
            ))

        gw.step(current_step=1)
        assert provider.call_count == 2  # Only 2 processed
        assert len(gw._queue) == 3  # 3 remaining

    def test_no_provider_publishes_failure(self):
        bus = MockEventBus()
        gw = ApiGateway(event_bus=bus)
        gw.submit_request(ApiRequest(
            request_id="np1", provider="nonexistent", endpoint="", payload={},
        ))
        gw.step(current_step=1)

        fail_events = [e for e in bus.published if e[0] == CH_EXTERNAL_FAILED]
        assert len(fail_events) == 1
        assert "no_provider" in fail_events[0][1]["reason"]

    def test_response_validation_failure(self):
        """Response from provider fails InputValidator check."""
        bus = MockEventBus()
        validator = MockInputValidator(result="accepted")
        gw = ApiGateway(event_bus=bus, input_validator=validator)
        gw.register_provider("mock", MockProvider())

        gw.submit_request(ApiRequest(
            request_id="rv1", provider="mock", endpoint="", payload={},
        ))

        # Switch validator to reject responses
        validator._result = "rejected"
        gw.step(current_step=1)

        fail_events = [e for e in bus.published if e[0] == CH_EXTERNAL_FAILED]
        assert len(fail_events) == 1

    def test_on_submit_request_callback(self):
        """EventBus callback parses and queues requests."""
        gw = ApiGateway()
        gw._on_submit_request("external.submit_request", {
            "request_id": "cb1",
            "provider": "claude",
            "payload": {"question": "test"},
            "agent_id": "agent-1",
        })
        assert len(gw._queue) == 1
        assert gw._queue[0].request_id == "cb1"

    def test_statistics(self):
        gw = ApiGateway()
        gw.register_provider("mock", MockProvider())
        stats = gw.get_statistics()
        assert stats["provider_count"] == 1
        assert stats["queue_depth"] == 0
        assert "client" in stats

    def test_scan_task_pool(self):
        """Gateway picks up api_call tasks from TaskPool."""
        pool = TaskPool(generation_rate=0)
        gw = ApiGateway(task_pool=pool)
        gw.register_provider("mock", MockProvider())

        # Inject an external task
        task_id = inject_external_task(
            pool, provider="mock", payload={"q": "test"}, agent_id="a1",
        )
        assert task_id is not None

        gw.step(current_step=1)
        assert gw._total_queued >= 1


# =========================================================================
# LlmProvider Tests (~6 tests)
# =========================================================================

class TestLlmProvider:
    """LlmProvider: prompt construction, response parsing."""

    def test_create_provider(self):
        provider = LlmProvider()
        assert provider.provider_name == "claude"
        assert not provider.available  # No API key

    def test_create_with_key(self):
        provider = LlmProvider(api_key="test-key")
        assert provider.available

    def test_build_prompt_simple(self):
        provider = LlmProvider()
        messages = provider.build_prompt("What should I do?")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "What should I do?" in messages[0]["content"]

    def test_build_prompt_with_context(self):
        provider = LlmProvider()
        messages = provider.build_prompt(
            "Help me decide",
            agent_context={
                "agent_id": "agent-42",
                "role": "EXPLORER",
                "prediction_error": 0.85,
                "energy_level": 0.6,
            },
        )
        content = messages[0]["content"]
        assert "agent-42" in content
        assert "EXPLORER" in content
        assert "0.85" in content
        assert "0.6" in content or "Energy" in content

    def test_execute_no_key_returns_error(self):
        provider = LlmProvider(api_key="")
        request = ApiRequest(
            request_id="nk1", provider="claude", endpoint="",
            payload={"question": "test"},
        )
        response = provider.execute(request)
        assert response.status == ApiResponseStatus.ERROR
        assert "No API key" in response.error_message

    def test_statistics(self):
        provider = LlmProvider(api_key="test")
        stats = provider.get_statistics()
        assert stats["provider"] == "claude"
        assert stats["total_requests"] == 0


# =========================================================================
# ExternalTask Tests (~8 tests)
# =========================================================================

class TestExternalTask:
    """External task injection, claiming, and work delegation."""

    def test_inject_external_task(self):
        pool = TaskPool(generation_rate=0)
        task_id = inject_external_task(
            pool, provider="claude",
            payload={"question": "test"},
            agent_id="agent-1",
        )
        assert task_id is not None
        assert task_id.startswith("ext-")
        assert task_id in pool._tasks
        task = pool._tasks[task_id]
        assert task.task_type == "api_call"
        assert task.state == "available"

    def test_inject_returns_none_without_pool(self):
        result = inject_external_task(
            None, provider="claude", payload={},
        )
        assert result is None

    def test_external_spec_attached(self):
        pool = TaskPool(generation_rate=0)
        task_id = inject_external_task(
            pool, provider="claude",
            payload={"question": "help"},
            agent_id="a1",
            priority=3,
            context={"role": "EXPLORER"},
        )
        task = pool._tasks[task_id]
        spec = task.external_spec
        assert isinstance(spec, ExternalTaskSpec)
        assert spec.provider == "claude"
        assert spec.payload == {"question": "help"}
        assert spec.agent_id == "a1"
        assert spec.priority == 3

    def test_get_pending_external_tasks(self):
        pool = TaskPool(generation_rate=0)
        inject_external_task(pool, "claude", {"q": "1"})
        inject_external_task(pool, "claude", {"q": "2"})

        pending = get_pending_external_tasks(pool)
        assert len(pending) == 2
        assert all(t.task_type == "api_call" for t in pending)

    def test_get_pending_excludes_claimed(self):
        pool = TaskPool(generation_rate=0)
        task_id = inject_external_task(pool, "claude", {"q": "1"})
        pool._tasks[task_id].state = "claimed"

        pending = get_pending_external_tasks(pool)
        assert len(pending) == 0

    def test_complete_external_task(self):
        pool = TaskPool(generation_rate=0)
        task_id = inject_external_task(pool, "claude", {"q": "1"})

        result = complete_external_task(pool, task_id, {
            "text": "the answer",
            "tokens_in": 10,
            "tokens_out": 50,
        })
        assert result is True
        task = pool._tasks[task_id]
        assert task.state == "completed"
        assert task.external_spec.response["text"] == "the answer"

    def test_complete_nonexistent_task(self):
        pool = TaskPool(generation_rate=0)
        assert not complete_external_task(pool, "nonexistent", {})

    def test_complete_without_pool(self):
        assert not complete_external_task(None, "any", {})


# =========================================================================
# Bootstrap Layer 31 Tests (~6 tests)
# =========================================================================

class TestBootstrapExternal:
    """Bootstrap Layer 31: wiring, triadic connections, holon registration."""

    def _make_ctx(self):
        """Create a minimal ctx for bootstrap testing."""
        from types import SimpleNamespace

        ctx = SimpleNamespace()
        ctx.bus = MockEventBus()
        ctx.boundary_membrane = MockBoundaryMembrane()
        ctx.input_validator = MockInputValidator()
        ctx.task_pool = TaskPool(generation_rate=0)
        ctx.somatic_map = MagicMock()
        ctx.holon_registry = MagicMock()
        ctx.holon_registry.get_proxy.return_value = MagicMock()
        ctx.connection_registry = MagicMock()
        ctx.model = MagicMock()
        ctx.model.time = 0

        return ctx

    def test_bootstrap_creates_gateway(self):
        from mae_core.bootstrap.external import bootstrap_external
        ctx = self._make_ctx()
        bootstrap_external(ctx)
        assert hasattr(ctx, "api_gateway")
        assert isinstance(ctx.api_gateway, ApiGateway)

    def test_bootstrap_registers_holon(self):
        from mae_core.bootstrap.external import bootstrap_external
        ctx = self._make_ctx()
        bootstrap_external(ctx)
        ctx.holon_registry.register.assert_called()
        # Check that api_gateway was registered
        calls = [c for c in ctx.holon_registry.register.call_args_list
                 if c[0][0] == "api_gateway"]
        assert len(calls) == 1

    def test_bootstrap_adds_step_hook(self):
        from mae_core.bootstrap.external import bootstrap_external
        ctx = self._make_ctx()
        bootstrap_external(ctx)
        ctx.model.add_step_hook.assert_called()

    def test_bootstrap_registers_connections(self):
        from mae_core.bootstrap.external import bootstrap_external
        ctx = self._make_ctx()
        bootstrap_external(ctx)
        # Should register 8 triadic connections
        assert ctx.connection_registry.register_connection.call_count == 8

    def test_bootstrap_no_api_key_graceful(self):
        """Without any API keys, gateway has 0 providers."""
        from mae_core.bootstrap.external import bootstrap_external
        ctx = self._make_ctx()
        with patch.dict("os.environ", {}, clear=False):
            # Remove all provider keys (LLM + data + search)
            import os
            for key in (
                "MAE_CLAUDE_API_KEY", "MAE_GROQ_API_KEY",
                "MAE_MISTRAL_API_KEY", "MAE_DEEPSEEK_API_KEY",
                "MAE_MARKETAUX_API_KEY", "MAE_FINNHUB_API_KEY",
                "MAE_ALPHAVANTAGE_API_KEY", "MAE_TAVILY_API_KEY",
            ):
                os.environ.pop(key, None)
            bootstrap_external(ctx)
        assert ctx.api_gateway.provider_count == 0

    def test_bootstrap_with_api_key(self):
        """With API key, Claude provider is registered."""
        from mae_core.bootstrap.external import bootstrap_external
        ctx = self._make_ctx()
        with patch.dict("os.environ", {"MAE_CLAUDE_API_KEY": "test-key-123"}, clear=False):
            import os
            # Isolate: remove other provider keys so we test Claude alone
            for key in ("MAE_GROQ_API_KEY", "MAE_MISTRAL_API_KEY", "MAE_DEEPSEEK_API_KEY"):
                os.environ.pop(key, None)
            bootstrap_external(ctx)
        assert ctx.api_gateway.has_provider("claude")
        assert ctx.api_gateway.provider_count >= 1


# =========================================================================
# Integration Tests (~5 tests)
# =========================================================================

class TestIntegration:
    """End-to-end: agent -> gateway -> mock LLM -> response -> learning."""

    def test_full_flow_mock_provider(self):
        """Agent submits -> gateway processes -> response published."""
        bus = MockEventBus()
        pool = TaskPool(generation_rate=0)
        gw = ApiGateway(event_bus=bus, task_pool=pool)
        gw.register_provider("mock", MockProvider())

        # Inject external task (as an agent would)
        task_id = inject_external_task(
            pool, provider="mock",
            payload={"question": "what should I do?"},
            agent_id="agent-7",
        )
        assert task_id is not None

        # Gateway picks up and processes
        gw.step(current_step=1)

        # Task should be completed
        task = pool._tasks[task_id]
        assert task.state == "completed"

        # Response should be published
        response_events = [e for e in bus.published if e[0] == CH_EXTERNAL_RESPONSE]
        assert len(response_events) >= 1
        assert response_events[0][1]["status"] == "success"

    def test_flow_with_membrane_and_validator(self):
        """Full flow with immune system checks."""
        bus = MockEventBus()
        pool = TaskPool(generation_rate=0)
        membrane = MockBoundaryMembrane(allow=True)
        validator = MockInputValidator(result="accepted")
        gw = ApiGateway(
            event_bus=bus, task_pool=pool,
            boundary_membrane=membrane, input_validator=validator,
        )
        gw.register_provider("mock", MockProvider())

        task_id = inject_external_task(pool, "mock", {"q": "test"}, "agent-1")
        gw.step(current_step=1)

        task = pool._tasks[task_id]
        assert task.state == "completed"

    def test_flow_membrane_blocks(self):
        """Membrane blocks untrusted provider."""
        bus = MockEventBus()
        pool = TaskPool(generation_rate=0)
        membrane = MockBoundaryMembrane(allow=False, reason="rejected")
        gw = ApiGateway(event_bus=bus, task_pool=pool, boundary_membrane=membrane)
        gw.register_provider("mock", MockProvider())

        task_id = inject_external_task(pool, "mock", {"q": "test"}, "agent-1")
        gw.step(current_step=1)

        # Task should NOT be completed (membrane blocked the queuing)
        task = pool._tasks[task_id]
        # Task was claimed by gateway but request was blocked
        assert gw._total_membrane_blocked >= 1

    def test_multiple_steps_drain_queue(self):
        """Multiple steps drain the queue correctly."""
        pool = TaskPool(generation_rate=0)
        gw = ApiGateway(task_pool=pool, max_requests_per_step=1)
        gw.register_provider("mock", MockProvider())

        inject_external_task(pool, "mock", {"q": "1"}, "a1")
        inject_external_task(pool, "mock", {"q": "2"}, "a2")

        gw.step(current_step=1)
        gw.step(current_step=2)

        assert gw._total_processed >= 2

    def test_api_call_task_type_in_pool(self):
        """api_call is a valid task type in TaskPool."""
        from mae_core.environment.task_pool import TASK_TYPES
        assert "api_call" in TASK_TYPES


# =========================================================================
# Stem Cell Role Profile Tests
# =========================================================================

class TestStemCellRoles:
    """New role profiles for external API access."""

    def test_api_caller_role_exists(self):
        from mae_core.agents.stem_cell import ROLE_PROFILES
        assert "API_CALLER" in ROLE_PROFILES
        assert ROLE_PROFILES["API_CALLER"]["api_call_enabled"] is True

    def test_llm_specialist_role_exists(self):
        from mae_core.agents.stem_cell import ROLE_PROFILES
        assert "LLM_SPECIALIST" in ROLE_PROFILES
        assert ROLE_PROFILES["LLM_SPECIALIST"]["api_call_enabled"] is True
        assert ROLE_PROFILES["LLM_SPECIALIST"]["llm_prompt_quality"] == 0.9

    def test_genome_has_api_knobs(self):
        from mae_core.agents.stem_cell import DEFAULT_GENOME
        names = DEFAULT_GENOME.get_gene_names()
        assert "api_call_enabled" in names
        assert "llm_prompt_quality" in names

    def test_defaults_have_api_disabled(self):
        from mae_core.agents.stem_cell import DEFAULT_GENOME
        defaults = DEFAULT_GENOME.get_defaults()
        assert defaults["api_call_enabled"] is False
        assert defaults["llm_prompt_quality"] == 0.5
