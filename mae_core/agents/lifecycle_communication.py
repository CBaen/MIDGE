"""Communication lifecycle mixin for MycelialAgent.

Extracted from mycelial_agent.py to prevent monolith growth.
These methods are called by step() in mycelial_agent.py.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CommunicationLifecycleMixin:
    """Communication lifecycle methods for MycelialAgent.

    All subsystem access is via getattr(self, ..., None) for graceful
    degradation when subsystems are not injected.
    """

    def _communicate(self) -> None:
        """Process GNN messages, deposit exploration trail, share patterns."""
        self.process_gnn_messages()
        self.deposit_exploration_marker()

        # Triadic pattern sharing (tissue-level communication)
        _sharer = getattr(self, "_pattern_sharer", None)
        _sense_result = getattr(self, "_last_sense_result", None)
        if _sharer is not None and _sense_result is not None:
            _sharer.share(_sense_result.signals)
            _sharer.receive_and_correlate(_sense_result.signals)

        # --- Update predictive field with intention ---
        pred_field = getattr(self, "_predictive_field", None)
        if pred_field is not None:
            try:
                pos = getattr(self, "pos", None)
                if pos is not None and hasattr(pred_field, "broadcast_intention"):
                    pred_field.broadcast_intention(
                        agent_id=self.unique_id,
                        intention=str(getattr(self, "last_action", "idle")),
                        target_position=pos,
                    )
            except Exception:
                logger.debug("predictive field update failed", exc_info=True)

    def _broadcast(self) -> None:
        """BROADCAST: GWT competitive ignition (cadenced every 3 steps).

        Biological analogy: Global Workspace Theory posits that conscious
        access occurs when a coalition of neural processes wins a competition
        and gets "broadcast" to the entire cortex. This allows agents to
        share their strongest signals with the organism-wide workspace.

        Submits the agent's most salient signal to PatternBus for inclusion
        in the next competitive ignition cycle.
        """
        pattern_bus = getattr(self, "_pattern_bus", None)
        if pattern_bus is None:
            return

        try:
            # Build broadcast payload from agent's strongest current signal
            pe = getattr(self, "_prediction_error", 0.0)
            reward = getattr(self, "last_reward", 0.0)
            risk = getattr(self, "risk_score", 0.0)

            # Only broadcast if we have something noteworthy
            salience = max(abs(pe), abs(reward), risk)
            if salience < 0.3:
                return  # Nothing worth broadcasting

            # Submit to EventBus (PatternBus listens on these channels)
            bus = getattr(pattern_bus, "_event_bus", None)
            if bus is None:
                bus = getattr(self, "_event_bus", None)
            if bus is not None:
                bus.publish("agent.broadcast", {
                    "agent_id": str(self.unique_id),
                    "prediction_error": pe,
                    "reward": reward,
                    "risk": risk,
                    "salience": salience,
                    "step": self.step_count,
                })

        except Exception:
            logger.debug(
                "Agent %s: _broadcast failed", self.unique_id, exc_info=True
            )

    def _regulate(self) -> None:
        """REGULATE: Arousal homeostasis (cadenced every 21 steps).

        Biological analogy: The autonomic nervous system maintains arousal
        within the optimal range via the Yerkes-Dodson law. The locus
        coeruleus adjusts norepinephrine based on recent performance:
        poor performance + low arousal -> sympathetic boost,
        poor performance + high arousal -> parasympathetic calming.

        Adjusts the EndocrineSystem to move arousal toward target.
        """
        reg = getattr(self, "_arousal_regulator", None)
        if reg is None:
            return

        try:
            # Get current arousal from organism body state
            body = getattr(self, "_body_state", None)
            current_arousal = 0.5  # default moderate
            if isinstance(body, dict):
                current_arousal = body.get("emotional_arousal", 0.5)

            result = reg.regulate(
                current_arousal=current_arousal,
                agent_id=str(self.unique_id),
                step=self.step_count,
            )

            # Apply hormone commands to EndocrineSystem if available
            endocrine = getattr(self, "_endocrine", None)
            if endocrine is not None and result.get("hormone_commands"):
                for hormone_name, delta in result["hormone_commands"]:
                    try:
                        if hasattr(endocrine, "adjust_hormone"):
                            endocrine.adjust_hormone(hormone_name, delta)
                        elif hasattr(endocrine, "release"):
                            endocrine.release(hormone_name, delta)
                    except Exception:
                        pass  # Graceful: not all hormones may exist

        except Exception:
            logger.debug(
                "Agent %s: _regulate failed", self.unique_id, exc_info=True
            )
