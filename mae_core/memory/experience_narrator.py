"""Experience Narrator - Converts numerical experiences to searchable text.

Template-based narration that turns state vectors and rewards into
structured English text suitable for embedding with mxbai-embed-large.
No LLM required -- uses agent context and state dimension semantics.

Biological analogy: The inner narrator that gives meaning to raw
sensory experience. The prefrontal cortex constructing a story
from hippocampal replay.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default state dimension names (74-dim Mae state vector).
# First 8 dims correspond to the _observe() state vector in MycelialAgent.
# The rest are subsystem-specific and unnamed by default.
DEFAULT_STATE_DIMS: dict[int, str] = {
    0: "energy",
    1: "risk",
    2: "exploration_drive",
    3: "social_connection",
    4: "success_markers",
    5: "danger_markers",
    6: "exploration_markers",
    7: "resource_level",
}

# Action names for discrete action space (4 actions default).
DEFAULT_ACTION_NAMES: dict[int, str] = {
    0: "explore",
    1: "exploit",
    2: "communicate",
    3: "rest",
}

# Reward characterization thresholds.
_REWARD_THRESHOLDS = [
    (0.8, "excellent"),
    (0.5, "good"),
    (0.1, "mild positive"),
    (0.0, "neutral"),
    (-0.3, "mild negative"),
    (-0.6, "poor"),
]


def _characterize_reward(reward: float) -> str:
    """Map reward float to descriptive word."""
    for threshold, label in _REWARD_THRESHOLDS:
        if reward >= threshold:
            return label
    return "very poor"


def _sign_label(reward: float) -> str:
    """Reward sign for payload filtering."""
    if reward > 0.01:
        return "positive"
    if reward < -0.01:
        return "negative"
    return "neutral"


def _level_word(value: float) -> str:
    """Map [0,1] value to descriptive word."""
    if value >= 0.8:
        return "very high"
    if value >= 0.6:
        return "high"
    if value >= 0.4:
        return "moderate"
    if value >= 0.2:
        return "low"
    return "very low"


def _delta_word(delta: float) -> str:
    """Map state delta to direction word."""
    if delta > 0.15:
        return "rising"
    if delta < -0.15:
        return "falling"
    return "steady"


class ExperienceNarrator:
    """Converts Experience tuples to text narratives for Qdrant storage.

    Template-based: no LLM calls, deterministic output.
    Each narration is unique due to timestamp, step, and context.
    """

    def __init__(
        self,
        state_dim_names: dict[int, str] | None = None,
        action_names: dict[int, str] | None = None,
    ) -> None:
        self._state_dims = state_dim_names if state_dim_names is not None else DEFAULT_STATE_DIMS
        self._action_names = action_names if action_names is not None else DEFAULT_ACTION_NAMES

    def characterize_state(self, state: np.ndarray) -> str:
        """Convert state vector to readable description of key dimensions."""
        parts = []
        for idx, name in sorted(self._state_dims.items()):
            if idx < len(state):
                val = float(state[idx])
                parts.append(f"{name}={_level_word(val)}")
        if not parts:
            parts.append(f"{len(state)}-dim state")
        return ", ".join(parts)

    def characterize_transition(
        self, state: np.ndarray, next_state: np.ndarray
    ) -> str:
        """Describe what changed between states."""
        changes = []
        for idx, name in sorted(self._state_dims.items()):
            if idx < len(state) and idx < len(next_state):
                delta = float(next_state[idx] - state[idx])
                direction = _delta_word(delta)
                if direction != "steady":
                    changes.append(f"{name} {direction}")
        if not changes:
            return "state stable"
        return ", ".join(changes)

    def narrate(
        self,
        experience: Any,
        agent_context: dict[str, Any],
    ) -> str:
        """Convert a single Experience to a text narrative.

        Args:
            experience: Experience dataclass with state, action, reward, etc.
            agent_context: Dict with keys like agent_id, agent_role,
                parent_id, peer_count, circadian_phase, step.
        """
        # Agent identity
        agent_id = agent_context.get("agent_id", "unknown")
        role = agent_context.get("agent_role", "GENERALIST")
        parent = agent_context.get("parent_id", "colony")
        peer_count = agent_context.get("peer_count", 0)
        phase = agent_context.get("circadian_phase", "ACTIVE")
        step = agent_context.get("step", 0)

        # Experience fields
        state = np.asarray(experience.state)
        next_state = np.asarray(experience.next_state)
        action = experience.action
        reward = float(experience.reward)
        done = experience.done

        # Action name
        action_name = self._action_names.get(
            int(action) if isinstance(action, (int, float, np.integer)) else -1,
            f"action-{action}",
        )

        # Build narrative
        state_desc = self.characterize_state(state)
        transition_desc = self.characterize_transition(state, next_state)
        reward_desc = _characterize_reward(reward)

        parts = [
            f"Agent {agent_id} ({role} role, child of {parent})",
            f"chose to {action_name}",
            f"from state [{state_desc}]",
            f"receiving {reward_desc} reward ({reward:.3f}).",
            f"Transition: {transition_desc}.",
        ]

        if done:
            parts.append("Episode complete.")

        parts.append(
            f"Context: step {step}, {phase} phase, {peer_count} peers active."
        )

        return " ".join(parts)

    def narrate_batch(
        self,
        experiences: list[Any],
        agent_context: dict[str, Any],
    ) -> list[str]:
        """Batch narration for consolidation phases."""
        return [self.narrate(exp, agent_context) for exp in experiences]

    def narrate_consolidation_summary(
        self,
        consolidation_id: int,
        agent_context: dict[str, Any],
        experiences: list[Any],
    ) -> str:
        """Narrate a summary of a consolidation batch."""
        if not experiences:
            return f"Agent {agent_context.get('agent_id', 'unknown')} consolidated 0 experiences."

        rewards = [float(e.reward) for e in experiences]
        actions = [
            self._action_names.get(
                int(e.action) if isinstance(e.action, (int, float, np.integer)) else -1,
                f"action-{e.action}",
            )
            for e in experiences
        ]

        from collections import Counter

        action_counts = Counter(actions)
        top_action = action_counts.most_common(1)[0][0]

        agent_id = agent_context.get("agent_id", "unknown")
        role = agent_context.get("agent_role", "GENERALIST")

        return (
            f"Consolidation #{consolidation_id} for agent {agent_id} ({role}): "
            f"{len(experiences)} experiences, "
            f"reward range [{min(rewards):.3f}, {max(rewards):.3f}], "
            f"mean reward {np.mean(rewards):.3f}, "
            f"most common action: {top_action} ({action_counts[top_action]} times). "
            f"Actions: {dict(action_counts)}."
        )

    def narrate_pattern(
        self,
        pattern: dict[str, Any],
        contributing_agents: list[str],
    ) -> str:
        """Narrate a distilled pattern for ancestral storage."""
        pattern_type = pattern.get("pattern_type", "behavioral")
        domain = pattern.get("domain", "general")
        occurrences = pattern.get("occurrence_count", 0)
        confidence = pattern.get("confidence", 0.0)
        description = pattern.get("description", "")

        agents_str = ", ".join(contributing_agents[:5])
        if len(contributing_agents) > 5:
            agents_str += f" and {len(contributing_agents) - 5} others"

        return (
            f"Pattern ({pattern_type}, {domain}): {description} "
            f"Observed {occurrences} times across agents [{agents_str}] "
            f"with confidence {confidence:.2f}."
        )

    def build_payload(
        self,
        experience: Any,
        agent_context: dict[str, Any],
        witness_hash: str = "",
        consolidation_id: int = 0,
    ) -> dict[str, Any]:
        """Build Qdrant payload dict for a narrated experience."""
        reward = float(experience.reward)
        action = experience.action

        return {
            "type": "episode",
            "agent_id": agent_context.get("agent_id", "unknown"),
            "agent_role": agent_context.get("agent_role", "GENERALIST"),
            "timestamp": getattr(experience, "timestamp", time.time()),
            "step": agent_context.get("step", 0),
            "circadian_phase": agent_context.get("circadian_phase", "ACTIVE"),
            "mean_reward": reward,
            "reward_sign": _sign_label(reward),
            "action_taken": int(action) if isinstance(action, (int, float, np.integer)) else str(action),
            "episode_complete": bool(experience.done),
            "state_dim": len(experience.state),
            "parent_id": agent_context.get("parent_id", "colony"),
            "organ_id": agent_context.get("organ_id", ""),
            "peer_count": agent_context.get("peer_count", 0),
            "witness_hash": witness_hash,
            "consolidation_id": consolidation_id,
            "primary_store": "midge_narrative",
            "verification_store": "pickle",
            "balance_store": "midge_ancestral",
        }
