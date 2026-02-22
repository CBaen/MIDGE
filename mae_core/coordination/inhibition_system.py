"""Inhibition System — Go/No-Go gate (basal ganglia model).

Biological analogy: The basal ganglia implement a Go/No-Go decision
via competing direct (Go) and indirect (No-Go) pathways. The subthalamic
nucleus provides a global brake signal. This system determines whether
an agent should act or suppress action at each step.

References:
- Frank (2006): Computational models of basal ganglia
- Mink (1996): The basal ganglia and involuntary movements
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InhibitionResult:
    """Result of the Go/No-Go decision."""
    inhibited: bool                   # True = suppress action
    go_strength: float               # Direct pathway activation (0-1)
    nogo_strength: float             # Indirect pathway activation (0-1)
    reason: str                      # Human-readable reason
    veto_sources: list[str] = field(default_factory=list)  # What triggered inhibition


class InhibitionSystem:
    """Basal ganglia Go/No-Go gate.

    Computes competing Go and No-Go signals from multiple sources.
    If No-Go > Go + threshold, action is suppressed.

    Sources of inhibition (No-Go pathway):
    - High prediction error (> 0.6): surprise -> pause
    - High risk score (> 0.7): danger -> freeze
    - Low energy (< 0.2): exhaustion -> rest
    - Negative somatic markers: emotional warning
    - External inhibit signals from other agents

    Sources of disinhibition (Go pathway):
    - High goal priority: urgent goal overrides caution
    - Positive reward trend: keep doing what works
    - Social pressure (quorum): group expects action
    - Low prediction error: confident about outcomes

    The balance between Go and No-Go is the core computation.
    """

    def __init__(self, event_bus=None, threshold: float = 0.15):
        """
        Args:
            event_bus: EventBus for publishing inhibition signals
            threshold: Go-NoGo margin needed for inhibition (default 0.15)
        """
        self._bus = event_bus
        self._threshold = threshold

        # Statistics
        self._total_decisions = 0
        self._total_inhibitions = 0
        self._consecutive_inhibitions = 0
        self._max_consecutive = 5  # Safety: don't inhibit forever

    def evaluate(
        self,
        prediction_error: float = 0.0,
        risk_score: float = 0.0,
        energy_level: float = 1.0,
        emotional_valence: float = 0.0,
        emotional_arousal: float = 0.5,
        goal_priority: float = 0.5,
        reward_trend: float = 0.0,
        quorum_pressure: float = 0.0,
        agent_id: str = "",
        step: int = 0,
    ) -> InhibitionResult:
        """Evaluate Go/No-Go balance.

        Returns InhibitionResult indicating whether action should be suppressed.
        """
        veto_sources: list[str] = []

        # --- No-Go pathway (indirect) ---
        nogo = 0.0

        # Surprise brake: high prediction error
        if prediction_error > 0.6:
            nogo += 0.3 * prediction_error
            veto_sources.append(f"surprise:{prediction_error:.2f}")

        # Danger freeze: high risk
        if risk_score > 0.7:
            nogo += 0.4 * risk_score
            veto_sources.append(f"risk:{risk_score:.2f}")

        # Exhaustion rest: low energy
        if energy_level < 0.2:
            nogo += 0.3 * (1.0 - energy_level)
            veto_sources.append(f"energy:{energy_level:.2f}")

        # Somatic warning: negative emotion + high arousal
        if emotional_valence < -0.3 and emotional_arousal > 0.6:
            somatic_nogo = abs(emotional_valence) * emotional_arousal * 0.25
            nogo += somatic_nogo
            veto_sources.append(f"somatic:{somatic_nogo:.2f}")

        # --- Go pathway (direct) ---
        go = 0.0

        # Goal urgency: high priority goal demands action
        go += 0.3 * goal_priority

        # Positive momentum: reward trend is up
        if reward_trend > 0:
            go += 0.2 * min(1.0, reward_trend)

        # Social pressure: group expects action
        if quorum_pressure > 0.5:
            go += 0.15 * quorum_pressure

        # Confidence: low prediction error means we know what we're doing
        if prediction_error < 0.2:
            go += 0.15 * (1.0 - prediction_error)

        # Baseline Go (organisms tend toward action)
        go += 0.2

        # --- Safety: don't inhibit forever ---
        if self._consecutive_inhibitions >= self._max_consecutive:
            go += 0.5  # Force disinhibition after max consecutive inhibitions
            veto_sources.append(f"safety_override:consecutive={self._consecutive_inhibitions}")

        # --- Decision ---
        inhibited = (nogo - go) > self._threshold

        # Update statistics
        self._total_decisions += 1
        if inhibited:
            self._total_inhibitions += 1
            self._consecutive_inhibitions += 1
            reason = f"INHIBIT: NoGo({nogo:.2f}) > Go({go:.2f}) + threshold({self._threshold})"
        else:
            self._consecutive_inhibitions = 0
            reason = f"GO: Go({go:.2f}) >= NoGo({nogo:.2f}) + threshold({self._threshold})"

        result = InhibitionResult(
            inhibited=inhibited,
            go_strength=min(1.0, go),
            nogo_strength=min(1.0, nogo),
            reason=reason,
            veto_sources=veto_sources if inhibited else [],
        )

        # Publish to EventBus
        if self._bus is not None and inhibited:
            try:
                self._bus.publish("coordination.inhibit_signal", {
                    "agent_id": agent_id,
                    "inhibited": True,
                    "go_strength": result.go_strength,
                    "nogo_strength": result.nogo_strength,
                    "veto_sources": result.veto_sources,
                    "step": step,
                })
            except Exception:
                logger.debug("InhibitionSystem: EventBus publish failed", exc_info=True)

        return result

    def get_statistics(self) -> dict[str, Any]:
        """Return inhibition statistics."""
        return {
            "total_decisions": self._total_decisions,
            "total_inhibitions": self._total_inhibitions,
            "inhibition_rate": (
                self._total_inhibitions / self._total_decisions
                if self._total_decisions > 0 else 0.0
            ),
            "consecutive_inhibitions": self._consecutive_inhibitions,
        }

    def reset_statistics(self) -> None:
        """Reset counters."""
        self._total_decisions = 0
        self._total_inhibitions = 0
        self._consecutive_inhibitions = 0
