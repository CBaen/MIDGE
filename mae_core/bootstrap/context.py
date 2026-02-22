"""Bootstrap context — shared state between bootstrap phases."""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("mae.bootstrap")

# Minimum agents enforced by Rule of 3
MIN_AGENTS = 3


def create_context(
    num_agents: int = 5,
    cycle_length: int = 100,
    persist_dir: str = "data/mae",
) -> SimpleNamespace:
    """Create the initial bootstrap context with configuration."""
    if num_agents < MIN_AGENTS:
        logger.warning(
            "Rule of 3: minimum %d agents required (requested %d), using %d",
            MIN_AGENTS, num_agents, MIN_AGENTS,
        )
        num_agents = MIN_AGENTS

    return SimpleNamespace(
        num_agents=num_agents,
        cycle_length=cycle_length,
        persist_dir=persist_dir,
    )
