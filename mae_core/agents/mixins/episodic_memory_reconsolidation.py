"""Episodic Memory Reconsolidation Mixin - Nader et al. 2000 implementation.

Handles memory reconsolidation: when a recalled memory is contradicted by
current experience (prediction error > threshold), it enters a labile state
and can be blended with new evidence before re-stabilizing.

Extracted from episodic_memory.py to stay under 500-line limit.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EpisodicMemoryReconsolidationMixin:
    """Memory reconsolidation: labile windows, blending, stabilization."""

    def _check_reconsolidation(
        self,
        memory: Any,
        current_reward: Optional[float] = None,
    ) -> None:
        """Check if a recalled memory should enter the reconsolidation window.

        Biological basis (Nader, Schafe & Le Doux 2000): When a consolidated
        memory is retrieved and the current context produces a prediction
        error (mismatch between expected and actual outcome), the memory
        trace is destabilized. During this labile period, the memory can
        be updated with new information before being re-stabilized through
        protein synthesis. The reconsolidation window in biology is ~6 hours;
        here it is scaled to 5 simulation steps.

        A prediction error > reconsolidation_threshold (default 0.3) triggers
        the window. During the window, the memory's reward is blended:
            updated = alpha * old + (1-alpha) * new
        where alpha = 0.6 gives the original memory inertia against change.

        Args:
            memory: The recalled experience object.
            current_reward: The reward observed in the current state. If None,
                falls back to the most recent reward in history.
        """
        try:
            mem_reward = getattr(memory, "reward", None)
            if mem_reward is None:
                return

            # Determine current reward context
            if current_reward is None:
                reward_history = getattr(self, "_reward_history", None)
                if reward_history and len(reward_history) > 0:
                    current_reward = reward_history[-1]
                else:
                    return

            # Compute prediction error: |expected - actual|
            prediction_error = abs(float(mem_reward) - float(current_reward))

            threshold = getattr(self, "_reconsolidation_threshold", 0.3)
            if prediction_error <= threshold:
                return

            # Memory enters labile state — reconsolidation window opens
            mem_id = id(memory)
            labile = getattr(self, "_labile_memories", {})
            if mem_id in labile:
                # Already labile — update with new evidence (within window)
                self._reconsolidate_memory(memory, current_reward)
                return

            step_count = getattr(self, "step_count", 0)
            window = getattr(self, "_reconsolidation_window", 5)
            labile[mem_id] = {
                "memory": memory,
                "original_reward": float(mem_reward),
                "labile_step": step_count,
                "window_end": step_count + window,
                "updated": False,
                "prediction_error": prediction_error,
            }
            self._labile_memories = labile

            recon_events = getattr(self, "_reconsolidation_events", 0)
            self._reconsolidation_events = recon_events + 1

            logger.debug(
                "Memory %s entered labile state (prediction_error=%.3f, "
                "window=%d steps)",
                mem_id, prediction_error, window,
            )

            # Immediately apply first update
            self._reconsolidate_memory(memory, current_reward)
        except Exception:
            logger.debug(
                "Reconsolidation check failed gracefully", exc_info=True,
            )

    def _reconsolidate_memory(
        self, memory: Any, current_reward: float,
    ) -> None:
        """Blend old memory with new evidence during the reconsolidation window.

        Updates the memory's reward using exponential blending:
            updated_reward = alpha * old_reward + (1-alpha) * current_reward

        alpha = 0.6 gives the original trace inertia — older memories resist
        change more than recent ones, matching the biological finding that
        strongly consolidated memories are harder to destabilize (Suzuki et al.
        2004).

        Args:
            memory: The experience being reconsolidated.
            current_reward: New evidence (current observed reward).
        """
        try:
            mem_reward = getattr(memory, "reward", None)
            if mem_reward is None:
                return

            alpha = getattr(self, "_reconsolidation_alpha", 0.6)
            updated_reward = alpha * float(mem_reward) + (1.0 - alpha) * float(
                current_reward
            )

            # Apply update — use setattr for flexibility across Experience types
            if hasattr(memory, "reward"):
                try:
                    memory.reward = updated_reward
                except (AttributeError, TypeError):
                    # Immutable Experience — cannot update in-place, log and skip
                    logger.debug(
                        "Cannot update immutable memory reward in-place"
                    )
                    return

            # Track the update
            mem_id = id(memory)
            labile = getattr(self, "_labile_memories", {})
            if mem_id in labile:
                labile[mem_id]["updated"] = True

            updates = getattr(self, "_reconsolidation_updates", 0)
            self._reconsolidation_updates = updates + 1

            logger.debug(
                "Reconsolidated memory %s: reward %.3f -> %.3f",
                mem_id, mem_reward, updated_reward,
            )
        except Exception:
            logger.debug(
                "Reconsolidation update failed gracefully", exc_info=True,
            )

    def _tick_reconsolidation(self) -> None:
        """Advance reconsolidation windows by one step.

        Call this once per agent step. Memories whose reconsolidation
        window has expired are stabilized (removed from the labile set).
        This mirrors the biological process where protein synthesis
        completes and the memory trace is re-consolidated into a stable
        form — potentially with modifications made during the labile period.
        """
        try:
            labile = getattr(self, "_labile_memories", None)
            if not labile:
                return

            step_count = getattr(self, "step_count", 0)
            to_stabilize: list[int] = []

            for mem_id, entry in labile.items():
                if step_count >= entry.get("window_end", 0):
                    to_stabilize.append(mem_id)

            for mem_id in to_stabilize:
                del labile[mem_id]
                stabilizations = getattr(
                    self, "_reconsolidation_stabilizations", 0
                )
                self._reconsolidation_stabilizations = stabilizations + 1
                logger.debug(
                    "Memory %s reconsolidated (window closed)", mem_id
                )
        except Exception:
            logger.debug(
                "Reconsolidation tick failed gracefully", exc_info=True,
            )
