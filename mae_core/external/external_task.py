"""ExternalTask - Extends TaskPool vocabulary with api_call task type.

Biological analogy: A specialized neurotransmitter. Regular tasks are
glutamate (general excitatory signals). External tasks are acetylcholine —
they cross the blood-brain barrier and trigger motor actions (API calls)
in the outside world.

Agents create external tasks when their internal decision cascade reaches
the prefrontal tier and determines external consultation is needed.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExternalTaskSpec:
    """Specification for an external API task.

    Created by agents via inject_external_task(), consumed by ApiGateway.
    Lives in the TaskPool alongside regular tasks but has extra fields
    for provider routing and payload delivery.
    """

    provider: str  # e.g., "claude", "data_feed", "web_search"
    payload: dict[str, Any]  # provider-specific request data
    priority: int = 5  # 1=highest, 10=lowest
    agent_id: str = ""  # requesting agent
    context: dict[str, Any] = field(default_factory=dict)  # agent state snapshot
    response: Optional[dict[str, Any]] = None  # filled when complete


def inject_external_task(
    task_pool: Any,
    provider: str,
    payload: dict[str, Any],
    agent_id: str = "",
    priority: int = 5,
    deadline: int = 20,
    context: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Inject an api_call task into the TaskPool.

    Creates a task with type "api_call" and attaches an ExternalTaskSpec
    with provider routing info. The ApiGateway picks these up on its
    step() cycle.

    Args:
        task_pool: The TaskPool instance
        provider: Provider name (e.g., "claude")
        payload: Request payload for the provider
        agent_id: Requesting agent ID
        priority: 1=highest, 10=lowest
        deadline: Steps before expiry
        context: Optional agent state snapshot

    Returns:
        task_id if injection succeeded, None otherwise
    """
    if task_pool is None:
        logger.warning("ExternalTask: no TaskPool available")
        return None

    from mae_core.environment.task_pool import Task

    task_id = f"ext-{str(uuid.uuid4())[:8]}"
    difficulty = 0.5  # External tasks have moderate difficulty
    reward_value = 1.5  # Higher reward for external consultation

    task = Task(
        task_id=task_id,
        task_type="api_call",
        difficulty=difficulty,
        reward_value=reward_value,
        deadline=deadline,
        state="available",
        created_step=getattr(task_pool, "_current_step", 0),
    )

    # Attach the external spec
    task.external_spec = ExternalTaskSpec(  # type: ignore[attr-defined]
        provider=provider,
        payload=payload,
        priority=priority,
        agent_id=agent_id,
        context=context or {},
    )

    # Insert directly into pool
    task_pool._tasks[task_id] = task

    logger.debug(
        "ExternalTask injected: %s (provider=%s, agent=%s)",
        task_id, provider, agent_id,
    )
    return task_id


def get_pending_external_tasks(task_pool: Any) -> list[Any]:
    """Get all unclaimed api_call tasks from the pool.

    Used by ApiGateway to find work to process.
    """
    if task_pool is None:
        return []

    return [
        task for task in task_pool._tasks.values()
        if task.task_type == "api_call"
        and task.state == "available"
        and hasattr(task, "external_spec")
    ]


def complete_external_task(
    task_pool: Any,
    task_id: str,
    response: dict[str, Any],
) -> bool:
    """Mark an external task as completed with its response.

    Called by ApiGateway after receiving the API response.
    The agent that claimed the task will see it completed
    in its next _observe() cycle.
    """
    if task_pool is None:
        return False

    task = task_pool._tasks.get(task_id)
    if task is None:
        return False

    task.state = "completed"
    task_pool._total_completed += 1

    spec = getattr(task, "external_spec", None)
    if spec is not None:
        spec.response = response

    logger.debug("ExternalTask completed: %s", task_id)
    return True
