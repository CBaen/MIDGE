"""ApiGateway - Mae's sensory organ for the external world.

Biological analogy: The enteric nervous system (gut-brain). It mediates
between Mae's internal decision-making and external APIs. Like the gut
processes food through multiple validation stages before nutrients enter
the bloodstream, the gateway processes API requests through BoundaryMembrane
and InputValidator before responses enter Mae's consciousness.

Data flow:
    Agent publishes ExternalRequest to "external.submit_request"
    -> ApiGateway dequeues request
    -> BoundaryMembrane.check_input(provider) -> classification
    -> InputValidator.validate(payload) -> validation
    -> Provider.execute(request) -> raw response
    -> InputValidator.validate(response) -> validated response
    -> Publish ExternalResponse to "external.response_received"
    -> TaskPool marks task completed with response

All external calls are mocked in tests. No real API keys needed for testing.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait as futures_wait
from typing import Any, Optional

from mae_core.external.api_client import (
    ApiClient,
    ApiRequest,
    ApiResponse,
    ApiResponseStatus,
    BaseProvider,
)
from mae_core.external.external_task import (
    complete_external_task,
    get_pending_external_tasks,
)

logger = logging.getLogger(__name__)

# EventBus channel constants
CH_EXTERNAL_RESPONSE = "external.response_received"
CH_EXTERNAL_QUEUED = "external.request_queued"
CH_EXTERNAL_FAILED = "external.request_failed"
CH_EXTERNAL_SUBMIT = "external.submit_request"


class ApiGateway:
    """Mae's central gateway for all external API communication.

    Mediates between internal agent requests and external APIs.
    All requests pass through BoundaryMembrane (source classification)
    and InputValidator (payload validation) before and after execution.

    Registered as a holon in Mae's fractal hierarchy.
    Runs on a step hook for periodic queue processing.
    """

    def __init__(
        self,
        event_bus: Any = None,
        boundary_membrane: Any = None,
        input_validator: Any = None,
        task_pool: Any = None,
        max_queue_size: int = 100,
        max_requests_per_step: int = 3,
    ) -> None:
        self._bus = event_bus
        self._boundary_membrane = boundary_membrane
        self._input_validator = input_validator
        self._task_pool = task_pool

        # API client for actual HTTP calls
        self._client = ApiClient()

        # Registered providers
        self._providers: dict[str, BaseProvider] = {}

        # Provider rotation groups (e.g., "groq" -> ["groq_1", "groq_2", ...])
        self._rotations: dict[str, list[str]] = {}
        self._rotation_index: dict[str, int] = {}
        self._rotation_cooldowns: dict[str, float] = {}  # provider -> cooldown_until

        # Request queue (priority-sorted deque)
        self._queue: deque[ApiRequest] = deque(maxlen=max_queue_size)
        self._max_per_step = max_requests_per_step

        # Async processing — fire API calls in background threads,
        # collect results next step. Eliminates blocking the organism.
        self._executor = ThreadPoolExecutor(max_workers=max_requests_per_step, thread_name_prefix="mae-api")
        self._in_flight: dict[str, tuple[Future, ApiRequest]] = {}

        # Statistics
        self._total_queued: int = 0
        self._total_processed: int = 0
        self._total_rejected: int = 0
        self._total_membrane_blocked: int = 0
        self._total_validation_failed: int = 0
        self._step_count: int = 0

        # Subscribe to external request channel
        if self._bus is not None:
            self._bus.register_callback(CH_EXTERNAL_SUBMIT, self._on_submit_request)

        logger.info("ApiGateway initialized (max_queue=%d, max_per_step=%d)",
                     max_queue_size, max_requests_per_step)

    def register_provider(self, name: str, provider: BaseProvider) -> None:
        """Register an API provider (e.g., "claude" -> LlmProvider).

        Biological analogy: Adding a new sensory receptor type.
        """
        self._providers[name] = provider
        logger.info("ApiGateway: registered provider '%s'", name)

    def register_rotation(self, group_name: str, provider_names: list[str]) -> None:
        """Register a rotation group (e.g., "groq" -> ["groq_1", "groq_2", ...]).

        When a request targets group_name, the gateway round-robins through
        the provider_names, skipping any on cooldown (429 in last 60s).
        """
        self._rotations[group_name] = list(provider_names)
        self._rotation_index[group_name] = 0
        logger.info("ApiGateway: rotation '%s' -> %s", group_name, provider_names)

    def _resolve_provider(self, provider_name: str) -> tuple[str, BaseProvider | None]:
        """Resolve a provider name, using rotation if available.

        Returns (actual_provider_name, provider_instance).
        """
        if provider_name in self._rotations:
            members = self._rotations[provider_name]
            now = time.time()
            # Try each member starting from current index
            start_idx = self._rotation_index[provider_name]
            for offset in range(len(members)):
                idx = (start_idx + offset) % len(members)
                name = members[idx]
                cooldown = self._rotation_cooldowns.get(name, 0)
                if now >= cooldown:
                    # Advance index for next call
                    self._rotation_index[provider_name] = (idx + 1) % len(members)
                    return name, self._providers.get(name)
            # All on cooldown — use the next one anyway
            name = members[start_idx % len(members)]
            self._rotation_index[provider_name] = (start_idx + 1) % len(members)
            return name, self._providers.get(name)
        return provider_name, self._providers.get(provider_name)

    def _mark_cooldown(self, provider_name: str, seconds: float = 60.0) -> None:
        """Mark a provider as on cooldown (rate limited)."""
        self._rotation_cooldowns[provider_name] = time.time() + seconds

    def has_provider(self, name: str) -> bool:
        """Check if a provider is registered."""
        return name in self._providers or name in self._rotations

    @property
    def provider_count(self) -> int:
        """Number of registered providers."""
        return len(self._providers)

    def submit_request(self, request: ApiRequest) -> bool:
        """Submit a request to the gateway queue.

        Returns True if queued, False if rejected (membrane/validation).
        """
        # Stage 1: BoundaryMembrane classification
        if self._boundary_membrane is not None:
            try:
                allowed, reason = self._boundary_membrane.check_input(
                    request.provider, {"type": "api_request", "provider": request.provider},
                )
                if not allowed:
                    self._total_membrane_blocked += 1
                    logger.warning(
                        "ApiGateway: membrane blocked request %s (reason=%s)",
                        request.request_id, reason,
                    )
                    return False
            except Exception:
                logger.debug("ApiGateway: membrane check failed, allowing", exc_info=True)

        # Stage 2: InputValidator on request payload
        if self._input_validator is not None:
            try:
                report = self._input_validator.validate(
                    request.provider, "api_request", request.payload,
                )
                if report.result.value in ("rejected", "quarantine"):
                    self._total_validation_failed += 1
                    logger.warning(
                        "ApiGateway: validation rejected request %s",
                        request.request_id,
                    )
                    return False
            except Exception:
                logger.debug("ApiGateway: validation failed, allowing", exc_info=True)

        # Queue the request
        self._queue.append(request)
        self._total_queued += 1

        if self._bus is not None:
            self._bus.publish(CH_EXTERNAL_QUEUED, {
                "request_id": request.request_id,
                "provider": request.provider,
                "agent_id": request.agent_id,
                "queue_depth": len(self._queue),
            })

        return True

    def step(self, current_step: int = 0) -> None:
        """Process queued requests (called as step hook).

        Non-blocking: submits new requests to a thread pool and
        collects completed results from previous submissions.
        The organism continues stepping while API calls run in background.
        """
        self._step_count = current_step

        # Pick up any new external tasks from the TaskPool
        self._scan_task_pool()

        # Collect completed futures from previous steps
        self._collect_completed()

        # Submit new requests (up to max_per_step minus in-flight)
        available_slots = self._max_per_step - len(self._in_flight)
        submitted = 0
        while self._queue and submitted < available_slots:
            request = self._queue.popleft()
            actual_name, provider = self._resolve_provider(request.provider)
            if provider is None:
                self._total_rejected += 1
                logger.warning(
                    "ApiGateway: no provider '%s' for request %s",
                    request.provider, request.request_id,
                )
                self._publish_failure(request, "no_provider")
                continue

            logger.info(
                "ApiGateway: submitting %s for request %s (agent=%s)",
                actual_name, request.request_id, request.agent_id,
            )
            future = self._executor.submit(self._client.call, provider, request)
            self._in_flight[request.request_id] = (future, request)
            submitted += 1

        # Second collect pass — wait briefly for fast providers (mocks,
        # cached responses) to finish in-thread. Real API calls (1-30s)
        # won't complete in 100ms and will be collected next step.
        if self._in_flight:
            futures_wait(
                [f for f, _ in self._in_flight.values()],
                timeout=0.1,
            )
            self._collect_completed()

    def _scan_task_pool(self) -> None:
        """Scan TaskPool for unclaimed api_call tasks and queue them."""
        if self._task_pool is None:
            return

        pending = get_pending_external_tasks(self._task_pool)
        for task in pending:
            spec = getattr(task, "external_spec", None)
            if spec is None:
                continue

            # Claim the task for the gateway
            task.state = "claimed"
            task.claimed_by = "api_gateway"

            request = ApiRequest(
                request_id=task.task_id,
                provider=spec.provider,
                endpoint="",  # provider-specific
                payload=spec.payload,
                priority=spec.priority,
                agent_id=spec.agent_id,
            )
            self.submit_request(request)

    def _collect_completed(self) -> None:
        """Collect results from completed async API calls."""
        completed_ids = []
        for req_id, (future, request) in self._in_flight.items():
            if future.done():
                completed_ids.append(req_id)
                try:
                    response = future.result()
                    self._handle_response(request, response)
                except Exception as exc:
                    logger.warning(
                        "ApiGateway: async call failed for %s: %s",
                        req_id, exc,
                    )
                    self._publish_failure(request, str(exc))
        for req_id in completed_ids:
            del self._in_flight[req_id]

    def _process_request(self, request: ApiRequest) -> None:
        """Process a single API request synchronously (legacy path)."""
        actual_name, provider = self._resolve_provider(request.provider)
        if provider is None:
            self._total_rejected += 1
            self._publish_failure(request, "no_provider")
            return
        response = self._client.call(provider, request)
        self._handle_response(request, response)

    def _handle_response(self, request: ApiRequest, response: ApiResponse) -> None:
        """Process an API response — validation, task completion, EventBus publish."""
        self._total_processed += 1

        # Mark cooldown on rate limit so rotation skips this provider
        if response.status == ApiResponseStatus.RATE_LIMITED:
            actual_name, _ = self._resolve_provider(request.provider)
            self._mark_cooldown(actual_name, seconds=60.0)

        if response.status == ApiResponseStatus.SUCCESS:
            text = response.payload.get("text", "")[:100] if response.payload else ""
            logger.info(
                "ApiGateway: %s responded — %dms, %d+%d tokens: %s%s",
                request.provider, response.latency_ms,
                response.tokens_in, response.tokens_out,
                text, "..." if len(text) >= 100 else "",
            )
            # Stage 3: Validate response through InputValidator
            if self._input_validator is not None:
                try:
                    report = self._input_validator.validate(
                        request.provider, "api_response", response.payload,
                    )
                    if report.result.value in ("rejected", "quarantine"):
                        self._total_validation_failed += 1
                        logger.warning(
                            "ApiGateway: response validation failed for %s",
                            request.request_id,
                        )
                        self._publish_failure(request, "response_validation_failed")
                        return
                except Exception:
                    logger.debug("ApiGateway: response validation error, passing", exc_info=True)

            # Complete the task in TaskPool
            if self._task_pool is not None:
                complete_external_task(
                    self._task_pool, request.request_id,
                    response={
                        "text": response.payload.get("text", "") if response.payload else "",
                        "tokens_in": response.tokens_in,
                        "tokens_out": response.tokens_out,
                        "latency_ms": response.latency_ms,
                        "provider": response.provider,
                    },
                )

            # Publish response on EventBus
            if self._bus is not None:
                self._bus.publish(CH_EXTERNAL_RESPONSE, {
                    "request_id": request.request_id,
                    "provider": request.provider,
                    "agent_id": request.agent_id,
                    "status": "success",
                    "tokens_in": response.tokens_in,
                    "tokens_out": response.tokens_out,
                    "latency_ms": response.latency_ms,
                })

        else:
            logger.warning(
                "ApiGateway: %s FAILED for request %s — %s: %s",
                request.provider, request.request_id,
                response.status.value, response.error_message,
            )
            self._publish_failure(
                request,
                f"{response.status.value}: {response.error_message}",
            )

    def _publish_failure(self, request: ApiRequest, reason: str) -> None:
        """Publish a failure event."""
        if self._bus is not None:
            self._bus.publish(CH_EXTERNAL_FAILED, {
                "request_id": request.request_id,
                "provider": request.provider,
                "agent_id": request.agent_id,
                "reason": reason,
            })

    def _on_submit_request(self, channel: str, message: Any) -> None:
        """EventBus callback for external.submit_request."""
        try:
            data = json.loads(message) if isinstance(message, str) else message
            if not isinstance(data, dict):
                return

            request = ApiRequest(
                request_id=data.get("request_id", f"evt-{int(time.time())}"),
                provider=data.get("provider", ""),
                endpoint=data.get("endpoint", ""),
                payload=data.get("payload", {}),
                priority=data.get("priority", 5),
                agent_id=data.get("agent_id", ""),
            )
            self.submit_request(request)
        except Exception:
            logger.debug("ApiGateway: failed to parse submit_request", exc_info=True)

    def close(self) -> None:
        """Shut down the thread pool and close provider connections."""
        self._executor.shutdown(wait=False)
        for name, provider in self._providers.items():
            if hasattr(provider, "close"):
                try:
                    provider.close()
                except Exception:
                    pass

    def get_statistics(self) -> dict[str, Any]:
        """Gateway statistics for monitoring."""
        return {
            "providers": list(self._providers.keys()),
            "provider_count": len(self._providers),
            "queue_depth": len(self._queue),
            "in_flight": len(self._in_flight),
            "total_queued": self._total_queued,
            "total_processed": self._total_processed,
            "total_rejected": self._total_rejected,
            "total_membrane_blocked": self._total_membrane_blocked,
            "total_validation_failed": self._total_validation_failed,
            "client": self._client.get_statistics(),
        }
