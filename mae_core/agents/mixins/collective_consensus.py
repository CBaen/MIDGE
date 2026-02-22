"""Collective Consensus Mixin - Quorum sensing and swarm wisdom.

Enables population-level coordination through quorum sensing,
collective insights, and consensus-driven behavior modification.

Ported from v5-pivot base_agent.py Phase 3 collective methods.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CollectiveConsensusMixin:
    """Adds quorum sensing and collective decision-making to agents."""

    def _init_collective_consensus(
        self,
        quorum_sensor: Any = None,
        agent_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize collective consensus attributes."""
        config = agent_config or {}
        self.quorum_sensor = quorum_sensor
        self.quorum_sensing_enabled: bool = config.get("quorum_sensing_enabled", False)
        self.last_quorum_check: float = 0.0
        self.quorum_check_interval: float = config.get("quorum_check_interval", 1.0)

    def sense_quorum(self) -> Optional[dict[str, Any]]:
        """Check quorum signals and respond to population-level coordination.

        Reads quorum signals from the shared QuorumSpace and responds
        to threshold-triggered behaviors (synchronization, migration,
        collective decision-making).
        """
        if not self.quorum_sensor or not self.quorum_sensing_enabled:
            return None

        try:
            concentrations = self.quorum_sensor.sense_all()

            thresholds_met: list[str] = []
            config = getattr(self, "agent_config", {})

            for signal_type, concentration in concentrations.items():
                threshold = config.get(f"quorum_threshold_{signal_type}", 0.5)
                if concentration >= threshold:
                    thresholds_met.append(signal_type)

            if thresholds_met:
                self._respond_to_quorum(thresholds_met, concentrations)

            return {
                "concentrations": concentrations,
                "thresholds_met": thresholds_met,
                "timestamp": time.time(),
            }

        except Exception:
            logger.exception("Error in quorum sensing")
            return None

    def _respond_to_quorum(self, signal_types: list[str], concentrations: dict[str, float]) -> None:
        """Respond to quorum threshold being met. Subclasses can override."""
        emit_signal = getattr(self, "emit_signal", None)
        if emit_signal is None:
            return

        for signal_type in signal_types:
            emit_signal(
                "COLLABORATION",
                {
                    "quorum_signal": signal_type,
                    "concentration": concentrations.get(signal_type, 0.0),
                    "responding_agent": getattr(self, "unique_id", "unknown"),
                },
                priority=0.8,
            )

    def get_collective_insights(
        self, count: int = 5, min_priority: Any = None
    ) -> list[Any]:
        """Get top consensus messages from the swarm.

        Returns messages with highest vote counts - what the collective
        wisdom says is important right now.
        """
        if not self.quorum_sensor:
            return []

        return self.quorum_sensor.get_collective_insights(
            count=count, min_priority=min_priority,
        )

    def act_on_consensus(self) -> None:
        """Modify behavior based on collective wisdom.

        Queries top consensus messages and adjusts agent parameters:
        - DANGER consensus -> reduce risk_tolerance, increase caution
        - OPPORTUNITY consensus -> increase exploration_rate
        - RESOURCE consensus -> increase search intensity
        """
        if not self.quorum_sensor:
            return

        insights = self.get_collective_insights(count=3)

        for msg in insights:
            msg_priority = getattr(msg, "priority", None)
            if msg_priority is not None and hasattr(msg_priority, "value"):
                if msg_priority.value < 3:  # Below URGENT
                    continue

            msg_type = getattr(msg, "message_type", "")

            if msg_type == "DANGER":
                risk_tolerance = getattr(self, "risk_tolerance", None)
                if risk_tolerance is not None:
                    self.risk_tolerance = risk_tolerance * 0.8  # type: ignore[attr-defined]

            elif msg_type == "OPPORTUNITY":
                exploration_rate = getattr(self, "exploration_rate", None)
                if exploration_rate is not None:
                    self.exploration_rate = min(1.0, exploration_rate * 1.2)  # type: ignore[attr-defined]

            elif msg_type == "RESOURCE":
                search_intensity = getattr(self, "search_intensity", None)
                if search_intensity is not None:
                    self.search_intensity = min(1.0, search_intensity * 1.3)  # type: ignore[attr-defined]

    def get_consensus_strength(
        self, signal_type: str, min_votes: int = 3
    ) -> float:
        """Calculate consensus strength (0.0-1.0) for a signal type."""
        if not self.quorum_sensor:
            return 0.0

        return self.quorum_sensor.get_consensus_strength(
            signal_type=signal_type, min_votes=min_votes,
        )

    def get_collective_statistics(self) -> Optional[dict[str, Any]]:
        """Get quorum sensing and collective consensus statistics."""
        if not self.quorum_sensor:
            return None

        stats: dict[str, Any] = {
            "quorum_sensing_enabled": self.quorum_sensing_enabled,
            "quorum_check_interval": self.quorum_check_interval,
        }

        get_qstats = getattr(self.quorum_sensor, "get_statistics", None)
        if get_qstats:
            stats["quorum_sensor"] = get_qstats()

        return stats

    def _serialize_collective_consensus(self) -> dict:
        return {
            "quorum_sensing_enabled": getattr(self, "quorum_sensing_enabled", False),
        }

    def _restore_collective_consensus(self, data: dict) -> None:
        if "quorum_sensing_enabled" in data:
            self.quorum_sensing_enabled = data["quorum_sensing_enabled"]
