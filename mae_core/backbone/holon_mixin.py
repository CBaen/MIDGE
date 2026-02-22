"""Holon Mixin - the 10-capability interface for agents.

Biological analogy: The genetic code that ensures every cell, no matter
how specialized, retains the full blueprint of the organism. HolonMixin
is the "stem cell protocol" — the universal interface that every agent
inherits, ensuring fractal self-similarity at the agent level.

10 capabilities: sense, remember, decide, act, learn, heal,
know_self, know_up, know_down, know_peers.

Follows the existing mixin pattern:
  _init_holon()          -- explicit initialization
  _serialize_holon()     -- persistence
  _restore_holon()       -- restore from persistence
  get_holon_statistics() -- introspection
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


ALL_HOLON_CAPABILITIES = frozenset([
    "sense", "remember", "decide", "act", "learn",
    "heal", "know_self", "know_up", "know_down", "know_peers",
])


class HolonMixin:
    """Universal holon protocol -- 10 capabilities at any scale.

    Default implementations introspect available capabilities from
    existing mixins/systems. Override individual holon_* methods to
    specialize behavior.

    Follows the existing mixin pattern:
      _init_holon()          -- explicit initialization
      _serialize_holon()     -- persistence
      _restore_holon()       -- restore from persistence
      get_holon_statistics() -- introspection
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_holon(
        self,
        holon_registry: Any = None,
        somatic_map: Any = None,
        holon_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        agent_config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize holon protocol state.

        Called LAST in MycelialAgent.__init__ so all other mixin state exists.
        """
        self._holon_registry: Any = holon_registry
        self._holon_somatic_map: Any = somatic_map
        self._holon_parent_id: Optional[str] = parent_id

        # Auto-detect holon ID from agent unique_id if not provided
        uid = getattr(self, "unique_id", None)
        self._holon_id: str = holon_id or (str(uid) if uid is not None else "unknown")

        # Simple fallback memory for holons without EpisodicMemoryMixin
        self._holon_memory: dict[str, Any] = {}

        # Detect which capabilities this holon actually has
        self._holon_capabilities: set[str] = self._detect_capabilities()

    def _detect_capabilities(self) -> set[str]:
        """Discover which of the 10 capabilities this holon supports."""
        caps: set[str] = set()

        # sense: always available (at minimum returns empty dict)
        caps.add("sense")

        # remember: EpisodicMemoryMixin or fallback dict
        caps.add("remember")

        # decide: BaseAgent._decide or DecisionRouter
        if callable(getattr(self, "_decide", None)):
            caps.add("decide")

        # act: BaseAgent._act
        if callable(getattr(self, "_act", None)):
            caps.add("act")

        # learn: BaseAgent._learn
        if callable(getattr(self, "_learn", None)):
            caps.add("learn")

        # heal: always available (basic self-assessment)
        caps.add("heal")

        # know_self/up/down/peers: always available
        caps.update({"know_self", "know_up", "know_down", "know_peers"})

        return caps

    # ------------------------------------------------------------------
    # The 10 Holon Capabilities
    # ------------------------------------------------------------------

    def holon_sense(self) -> dict[str, Any]:
        """Perceive local state and neighbors."""
        state: dict[str, Any] = {}

        # Own state from BaseAgent
        get_state = getattr(self, "get_state", None)
        if callable(get_state):
            state["self"] = get_state()

        # Step count
        state["step_count"] = getattr(self, "step_count", 0)

        # Recent reward
        state["last_reward"] = getattr(self, "last_reward", 0.0)

        return state

    def holon_remember(self, key: str, value: Any = None) -> Any:
        """Store or retrieve from memory.

        If value is provided, stores it. If value is None, retrieves by key.
        Delegates to EpisodicMemoryMixin if available, else uses simple dict.
        """
        if value is not None:
            # Store
            store_fn = getattr(self, "store_experience", None)
            if callable(store_fn):
                try:
                    store_fn(state=value, action=0, reward=0.0, next_state=value)
                except Exception:
                    pass
            self._holon_memory[key] = value
            return value
        else:
            # Retrieve
            return self._holon_memory.get(key)

    def holon_decide(self, stimulus: Any = None) -> Any:
        """Choose action via existing decision infrastructure."""
        decide_fn = getattr(self, "_decide", None)
        if callable(decide_fn):
            return decide_fn()
        return None

    def holon_act(self, action: Any = None) -> float:
        """Execute action via existing act infrastructure."""
        act_fn = getattr(self, "_act", None)
        if callable(act_fn) and action is not None:
            return act_fn(action)
        return 0.0

    def holon_learn(self, action: Any = None, reward: float = 0.0) -> None:
        """Learn from outcome via existing learn infrastructure."""
        learn_fn = getattr(self, "_learn", None)
        if callable(learn_fn):
            learn_fn(action, reward)

    def holon_heal(self) -> dict[str, Any]:
        """Self-assessment health check."""
        report: dict[str, Any] = {"holon_id": self._holon_id, "healthy": True, "issues": []}

        # Check reward trend (declining = possible issue)
        reward_history = getattr(self, "reward_history", None)
        if reward_history and len(reward_history) >= 10:
            recent = list(reward_history)[-10:]
            first_half = sum(recent[:5]) / 5
            second_half = sum(recent[5:]) / 5
            if second_half < first_half * 0.5:
                report["issues"].append("reward_declining")
                report["healthy"] = False

        # Check step progress
        step_count = getattr(self, "step_count", 0)
        if step_count == 0:
            report["issues"].append("never_stepped")

        # Report health to SomaticMap if available
        if self._holon_somatic_map and hasattr(self._holon_somatic_map, "heartbeat"):
            health = 1.0 if report["healthy"] else 0.5
            try:
                self._holon_somatic_map.heartbeat(self._holon_id, health=health)
            except Exception:
                pass

        return report

    def _effective_parent_id(self) -> Optional[str]:
        """Get parent ID from registry (source of truth) or local cache."""
        if self._holon_registry:
            registry_parent = self._holon_registry.get_parent(self._holon_id)
            if registry_parent is not None:
                return registry_parent
        return self._holon_parent_id

    def holon_know_self(self) -> dict[str, Any]:
        """Self-model: who am I, what can I do, how am I doing."""
        parent_id = self._effective_parent_id()
        result: dict[str, Any] = {
            "holon_id": self._holon_id,
            "holon_type": "agent",
            "capabilities": sorted(self._holon_capabilities),
            "parent_id": parent_id,
        }

        # Counts from registry
        if self._holon_registry:
            result["children_count"] = len(self._holon_registry.get_children(self._holon_id))
            result["peers_count"] = len(self._holon_registry.get_peers(self._holon_id))
        else:
            result["children_count"] = 0
            result["peers_count"] = 0

        # Health from SomaticMap
        if self._holon_somatic_map:
            get_info = getattr(self._holon_somatic_map, "get_system_info", None)
            if callable(get_info):
                try:
                    info = get_info(self._holon_id)
                    if info and hasattr(info, "health"):
                        result["health"] = info.health
                except Exception:
                    pass

        # Performance summary
        perf = getattr(self, "get_performance_summary", None)
        if callable(perf):
            result["performance"] = perf()

        return result

    def holon_know_up(self) -> Optional[dict[str, Any]]:
        """Aware of parent context. Returns None if root holon."""
        parent_id = self._effective_parent_id()
        if parent_id is None:
            return None

        result: dict[str, Any] = {"parent_id": parent_id}

        if self._holon_registry:
            entry = self._holon_registry.get_entry(parent_id)
            if entry:
                result["parent_type"] = entry.holon_type
                result["parent_children_count"] = len(
                    self._holon_registry.get_children(parent_id)
                )

        return result

    def holon_know_down(self) -> list[dict[str, Any]]:
        """Aware of child components."""
        if not self._holon_registry:
            return []

        children: list[dict[str, Any]] = []
        for child_id in self._holon_registry.get_children(self._holon_id):
            entry = self._holon_registry.get_entry(child_id)
            if entry:
                children.append({
                    "holon_id": child_id,
                    "holon_type": entry.holon_type,
                })
        return children

    def holon_know_peers(self) -> list[dict[str, Any]]:
        """Aware of siblings (same parent)."""
        if not self._holon_registry:
            return []

        peers: list[dict[str, Any]] = []
        for peer_id in self._holon_registry.get_peers(self._holon_id):
            entry = self._holon_registry.get_entry(peer_id)
            if entry:
                peers.append({
                    "holon_id": peer_id,
                    "holon_type": entry.holon_type,
                })
        return peers

    # ------------------------------------------------------------------
    # Persistence (standard mixin pattern)
    # ------------------------------------------------------------------

    def _serialize_holon(self) -> dict[str, Any]:
        """Serialize holon state for persistence."""
        return {
            "holon_id": self._holon_id,
            "parent_id": self._holon_parent_id,
            "holon_memory": dict(self._holon_memory),
        }

    def _restore_holon(self, data: dict[str, Any]) -> None:
        """Restore holon state from persistence."""
        self._holon_id = data.get("holon_id", self._holon_id)
        self._holon_parent_id = data.get("parent_id", self._holon_parent_id)
        memory = data.get("holon_memory")
        if isinstance(memory, dict):
            self._holon_memory.update(memory)

    # ------------------------------------------------------------------
    # Statistics (standard mixin pattern)
    # ------------------------------------------------------------------

    def get_holon_statistics(self) -> dict[str, Any]:
        """Holon protocol statistics."""
        stats: dict[str, Any] = {
            "holon_id": self._holon_id,
            "parent_id": self._holon_parent_id,
            "capabilities": sorted(self._holon_capabilities),
            "memory_keys": len(self._holon_memory),
        }
        if self._holon_registry:
            stats["children_count"] = len(self._holon_registry.get_children(self._holon_id))
            stats["peers_count"] = len(self._holon_registry.get_peers(self._holon_id))
        return stats
