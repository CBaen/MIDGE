"""Arousal Regulator — Autonomic nervous system balance.

Biological analogy: The autonomic nervous system balances sympathetic
(fight/flight, increases arousal) and parasympathetic (rest/digest,
decreases arousal) activation. The Yerkes-Dodson law describes the
inverted-U relationship between arousal and performance — too little
arousal leads to lethargy, too much leads to panic, optimal is in
the middle and depends on task complexity.

References:
- Yerkes & Dodson (1908): Relation of strength of stimulus to rapidity of habit formation
- Aston-Jones & Cohen (2005): Adaptive gain theory (locus coeruleus-norepinephrine system)

Law compliance:
- Law 6: autopoietic closure - regulation maintains the conditions for regulation
- Law 8 property (8): prediction/error-correction - PID-style arousal targeting

EventBus channel: coordination.regulation_signal
"""

import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

# EventBus channel
CH_REGULATION_SIGNAL = "coordination.regulation_signal"


class ArousalRegulator:
    """Maintains optimal arousal via sympathetic/parasympathetic balance.

    Cadenced at every 21 steps (Fibonacci: autonomic regulation is slow).

    Computes target arousal from recent performance using Yerkes-Dodson:
    - Poor performance + low arousal -> increase (sympathetic activation)
    - Poor performance + high arousal -> decrease (parasympathetic activation)
    - Good performance -> maintain current level

    Adjusts hormones in the EndocrineSystem to move actual arousal toward target.
    """

    def __init__(
        self,
        event_bus=None,
        target_arousal: float = 0.5,
        adjustment_rate: float = 0.15,
        dead_zone: float = 0.1,
        performance_window: int = 20,
    ):
        """Initialize the arousal regulator.

        Args:
            event_bus: EventBus for publishing regulation signals.
            target_arousal: Initial target arousal level (0-1).
            adjustment_rate: How much to adjust per regulation cycle (0-1).
            dead_zone: Minimum |error| before adjusting (prevents noise).
            performance_window: Number of recent rewards to track.
        """
        self._bus = event_bus
        self._target_arousal = max(0.0, min(1.0, target_arousal))
        self._adjustment_rate = max(0.0, min(1.0, adjustment_rate))
        self._dead_zone = max(0.0, dead_zone)

        # Performance tracking
        self._recent_rewards: deque[float] = deque(maxlen=performance_window)

        # Statistics
        self._total_regulations = 0
        self._sympathetic_activations = 0
        self._parasympathetic_activations = 0
        self._no_adjustments = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_reward(self, reward: float) -> None:
        """Record a reward observation for performance tracking.

        Args:
            reward: A scalar reward value. Will be clamped to [0, 1].
        """
        clamped = max(0.0, min(1.0, float(reward)))
        self._recent_rewards.append(clamped)

    def compute_target_arousal(self) -> float:
        """Compute optimal arousal via Yerkes-Dodson.

        Formula: target = 0.5 + 0.3 * mean_recent_reward
        Where mean_recent_reward is the average of recent rewards clamped to [0, 1].

        Good performance -> target ~0.8 (stay energized)
        Poor performance -> target ~0.5 (middle ground)
        No data -> target stays at current value

        Returns:
            The updated target arousal level.
        """
        if len(self._recent_rewards) == 0:
            return self._target_arousal

        mean_reward = sum(self._recent_rewards) / len(self._recent_rewards)
        # Clamp to [0, 1] range
        mean_reward = max(0.0, min(1.0, mean_reward))

        # Yerkes-Dodson: optimal arousal for good performance is moderate-high.
        # For poor performance, we want moderate arousal (not too high, not too low).
        target = 0.5 + 0.3 * mean_reward

        self._target_arousal = max(0.2, min(0.9, target))
        return self._target_arousal

    def regulate(
        self,
        current_arousal: float,
        agent_id: str = "",
        step: int = 0,
    ) -> dict[str, Any]:
        """Compute regulation adjustment for the given arousal level.

        This is the main entry point called every 21 steps during the
        REGULATE lifecycle phase.

        Args:
            current_arousal: The agent's current arousal level (0-1).
            agent_id: Identifier for the agent being regulated.
            step: Current simulation step (for logging/events).

        Returns:
            Dict with:
            - adjustment: float (positive = increase arousal, negative = decrease)
            - direction: str ("sympathetic", "parasympathetic", or "none")
            - target_arousal: float
            - current_arousal: float
            - hormone_commands: list of (hormone_name, delta) tuples
        """
        self._total_regulations += 1
        target = self.compute_target_arousal()
        error = target - current_arousal

        result: dict[str, Any] = {
            "adjustment": 0.0,
            "direction": "none",
            "target_arousal": target,
            "current_arousal": current_arousal,
            "hormone_commands": [],
        }

        if abs(error) <= self._dead_zone:
            # Within dead zone — no adjustment needed
            self._no_adjustments += 1
            result["direction"] = "none"
            return result

        adjustment = error * self._adjustment_rate
        result["adjustment"] = adjustment

        if adjustment > 0:
            # Need more arousal -> sympathetic activation
            self._sympathetic_activations += 1
            result["direction"] = "sympathetic"
            result["hormone_commands"] = [
                ("ADRENALINE", adjustment * 0.6),
                ("CORTISOL", adjustment * 0.3),
                ("DOPAMINE", adjustment * 0.1),
            ]
        else:
            # Need less arousal -> parasympathetic activation
            self._parasympathetic_activations += 1
            result["direction"] = "parasympathetic"
            result["hormone_commands"] = [
                ("SEROTONIN", abs(adjustment) * 0.5),
                ("OXYTOCIN", abs(adjustment) * 0.3),
                ("MELATONIN", abs(adjustment) * 0.2),
            ]

        # Publish regulation signal
        if self._bus is not None:
            try:
                self._bus.publish(CH_REGULATION_SIGNAL, {
                    "agent_id": agent_id,
                    "current_arousal": current_arousal,
                    "target_arousal": target,
                    "adjustment": adjustment,
                    "direction": result["direction"],
                    "step": step,
                })
            except Exception:
                logger.debug(
                    "ArousalRegulator: EventBus publish failed",
                    exc_info=True,
                )

        return result

    # ------------------------------------------------------------------
    # Phi-driven modulation
    # ------------------------------------------------------------------

    def _on_phi_measurement(self, channel: str, message: Any) -> None:
        """Nudge arousal target based on organism integration (phi).

        Low phi (< 0.3) means agents are disconnected — raise target
        arousal slightly to drive more inter-agent communication.
        High phi (> 0.7) means system is well-integrated — lower target
        slightly to conserve energy (no need to push harder).

        Advisory: +-0.05 max, clamped to [0.2, 0.9]. Never blocks.
        """
        import json as _json
        try:
            msg = _json.loads(message) if isinstance(message, str) else message
        except (TypeError, ValueError):
            return
        if not isinstance(msg, dict):
            return

        phi = msg.get("organism_mean_phi")
        if phi is None or not isinstance(phi, (int, float)):
            return

        if phi < 0.3:
            self._target_arousal = min(0.9, self._target_arousal + 0.05)
        elif phi > 0.7:
            self._target_arousal = max(0.2, self._target_arousal - 0.05)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Return regulation statistics for monitoring and diagnostics."""
        return {
            "total_regulations": self._total_regulations,
            "sympathetic_activations": self._sympathetic_activations,
            "parasympathetic_activations": self._parasympathetic_activations,
            "no_adjustments": self._no_adjustments,
            "current_target": self._target_arousal,
            "recent_rewards_count": len(self._recent_rewards),
            "mean_recent_reward": (
                sum(self._recent_rewards) / len(self._recent_rewards)
                if self._recent_rewards
                else 0.0
            ),
        }

    def __repr__(self) -> str:
        return (
            f"ArousalRegulator(target={self._target_arousal:.2f}, "
            f"regulations={self._total_regulations})"
        )
