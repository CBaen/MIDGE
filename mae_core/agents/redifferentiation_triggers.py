"""Automatic Stem Cell Redifferentiation Triggers.

Two advisory triggers that make stem cell redifferentiation operational
instead of manual-only:

1. Performance trigger — sustained low health causes role switch
2. Role imbalance trigger — empty roles filled from oversaturated ones

Cadenced every 21 steps (Fibonacci). Advisory only — logs, never blocks.
Law 5 (Stem Cell) + Law 6 (Autopoietic Closure).
"""

from __future__ import annotations

import logging
from typing import Any

from mae_core.agents.stem_cell import ROLE_PROFILES

logger = logging.getLogger(__name__)

# EventBus channel for automatic redifferentiation events
CH_AUTO_REDIFFERENTIATED = "stem_cell.auto_redifferentiated"

# Market roles that need live system refs re-attached after rediff
_MARKET_ROLES = frozenset({
    "SEC_WATCHER", "CONTRACT_TRACKER", "MARKET_ANALYST",
    "HYPOTHESIS_EXPLORER", "HYPOTHESIS_VALIDATOR",
})
_HYPOTHESIS_ROLES = frozenset({"HYPOTHESIS_EXPLORER", "HYPOTHESIS_VALIDATOR"})

# Thresholds
HEALTH_THRESHOLD = 0.3       # Below this = critically underperforming
CONSECUTIVE_REQUIRED = 21    # Checks before trigger (21 * 21-step cadence = 441 steps)
MIN_HISTORY = 21             # Minimum reward_history entries before evaluation


class RedifferentiationMonitor:
    """Monitors agent health and colony role balance for automatic redifferentiation.

    Like the bone marrow niche sensing damage signals and initiating
    stem cell mobilization — the colony self-repairs by re-specializing
    its worst-performing members into needed roles.
    """

    def __init__(self, stem_cell_registry: Any, event_bus: Any = None) -> None:
        self._registry = stem_cell_registry
        self._event_bus = event_bus
        self._low_health_counts: dict[str, int] = {}
        self._market_refs: dict[str, Any] | None = None

    def set_market_refs(self, refs: dict[str, Any]) -> None:
        """Attach market system refs for re-attachment after redifferentiation.

        Called from Layer 33 (market bootstrap) after market systems exist.
        Keys: advisory, alerter, regime, engine, registry, ctx.
        """
        self._market_refs = refs

    def step(self, current_step: int = 0) -> None:
        """Run both triggers. Called every 21 steps from model step hook."""
        if current_step % 21 != 0:
            return
        self._check_performance(current_step)
        self._check_role_imbalance(current_step)

    def _agent_health(self, agent: Any) -> float | None:
        """Compute health from reward_history. Returns None if insufficient data."""
        history = getattr(agent, "reward_history", None)
        if history is None or len(history) < MIN_HISTORY:
            return None
        recent = list(history)[-MIN_HISTORY:]
        total = sum(recent)
        return max(0.0, min(1.0, (total / len(recent) + 1.0) / 2.0))

    def _reattach_market_refs(self, agent: Any, new_role: str) -> None:
        """Re-attach market system refs if agent redifferentiated into a market role."""
        if self._market_refs is None or new_role not in _MARKET_ROLES:
            return
        refs = self._market_refs
        agent._market_advisory_ref = refs.get("advisory")
        agent._convergence_alerter_ref = refs.get("alerter")
        agent._regime_classifier_ref = refs.get("regime")
        agent._model_ctx_ref = refs.get("ctx")
        if new_role in _HYPOTHESIS_ROLES:
            agent._hypothesis_engine_ref = refs.get("engine")
            agent._hypothesis_registry_ref = refs.get("registry")

    def _least_populated_role(self, exclude_role: str) -> str:
        """Return the role with fewest agents, excluding current role."""
        dist = self._registry.get_role_distribution()
        candidates = {r: dist.get(r, 0) for r in ROLE_PROFILES if r != exclude_role}
        return min(candidates, key=candidates.get) if candidates else "STEM"

    def _check_performance(self, current_step: int) -> None:
        """Trigger A: Redifferentiate agents with sustained low health."""
        for agent_id, agent in list(self._registry._agents.items()):
            epi = self._registry.get_epigenome(agent_id)
            if epi is None or epi.locked:
                continue
            health = self._agent_health(agent)
            if health is None:
                continue
            if health < HEALTH_THRESHOLD:
                self._low_health_counts[agent_id] = self._low_health_counts.get(agent_id, 0) + 1
            else:
                self._low_health_counts[agent_id] = 0
            if self._low_health_counts.get(agent_id, 0) >= CONSECUTIVE_REQUIRED:
                old_role = epi.role
                new_role = self._least_populated_role(old_role)
                self._registry.redifferentiate(agent_id, new_role, step=current_step)
                self._reattach_market_refs(agent, new_role)
                self._low_health_counts[agent_id] = 0
                logger.info(
                    "Auto-redifferentiation (performance): agent %s %s->%s health=%.2f",
                    agent_id, old_role, new_role, health,
                )
                self._emit(agent_id, old_role, new_role, "performance", health, current_step)

    def _check_role_imbalance(self, current_step: int) -> None:
        """Trigger B: Fill empty roles from oversaturated ones."""
        dist = self._registry.get_role_distribution()
        total = sum(dist.values())
        if total < len(ROLE_PROFILES):
            return
        empty_roles = [r for r in ROLE_PROFILES if dist.get(r, 0) == 0]
        if not empty_roles:
            return
        oversaturated = max(dist, key=dist.get)
        if dist[oversaturated] <= 1:
            return
        agents_in_role = self._registry.get_agents_by_role(oversaturated)
        worst_id, worst_health = None, float("inf")
        for aid in agents_in_role:
            agent = self._registry._agents.get(aid)
            epi = self._registry.get_epigenome(aid)
            if agent is None or (epi is not None and epi.locked):
                continue
            h = self._agent_health(agent) or 1.0
            if h < worst_health:
                worst_id, worst_health = aid, h
        if worst_id is None:
            return
        new_role = empty_roles[0]
        agent = self._registry._agents.get(worst_id)
        self._registry.redifferentiate(worst_id, new_role, step=current_step)
        if agent is not None:
            self._reattach_market_refs(agent, new_role)
        self._low_health_counts[worst_id] = 0
        logger.info(
            "Auto-redifferentiation (imbalance): agent %s %s->%s",
            worst_id, oversaturated, new_role,
        )
        self._emit(worst_id, oversaturated, new_role, "role_imbalance", worst_health, current_step)

    def _emit(self, agent_id: str, old_role: str, new_role: str,
              trigger_type: str, health: float, step: int) -> None:
        """Publish auto-redifferentiation event on EventBus."""
        if self._event_bus is None:
            return
        self._event_bus.publish(CH_AUTO_REDIFFERENTIATED, {
            "agent_id": agent_id,
            "old_role": old_role,
            "new_role": new_role,
            "trigger_type": trigger_type,
            "health_at_trigger": round(health, 3),
            "step": step,
        })

    def get_statistics(self) -> dict[str, Any]:
        """Monitoring statistics."""
        return {
            "tracked_agents": len(self._low_health_counts),
            "agents_near_trigger": sum(
                1 for c in self._low_health_counts.values() if c >= CONSECUTIVE_REQUIRED // 2
            ),
        }
