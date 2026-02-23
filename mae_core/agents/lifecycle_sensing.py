"""Sensing lifecycle mixin for MycelialAgent.

Extracted from mycelial_agent.py to prevent monolith growth.
These methods are called by step() in mycelial_agent.py.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SensingLifecycleMixin:
    """Sensing lifecycle methods for MycelialAgent.

    All subsystem access is via getattr(self, ..., None) for graceful
    degradation when subsystems are not injected.
    """

    def _predict(self) -> None:
        """Generate prediction of next observation (FEP: prior expectation).

        Biological analogy: Before opening your eyes, the visual cortex
        already has a prediction of what you will see, based on your
        current internal state and last motor command. This prediction
        is compared against actual observation to compute surprise.

        Graceful when no world_model: sets prediction to None.
        """
        wm = getattr(self, "world_model", None)
        if wm is None:
            self._last_prediction = None
            return

        try:
            # Build state from current internal signals (pre-observation)
            state = self._build_state_vector()
            # Use last action as the motor command that generated the
            # expected transition (biological: efference copy)
            last_action = getattr(self, "last_action", None)
            if last_action is None:
                last_action = 0
            self._last_prediction = wm.predict(state, last_action)

            # Record imagination for validation (biological: predictive tagging)
            vi = getattr(self, "_validated_imagination", None)
            if vi is not None and self._last_prediction is not None:
                try:
                    self._last_prediction_id = vi.record_imagination(
                        agent_id=str(self.unique_id),
                        domain="world_model",
                        state=state,
                        action=last_action,
                        predicted_next_state=self._last_prediction,
                        predicted_reward=0.0,
                        confidence=1.0 - getattr(self, "_prediction_error", 0.0),
                    )
                except Exception:
                    logger.debug("Agent %s: record_imagination failed", self.unique_id, exc_info=True)

        except Exception:
            logger.debug(
                "Agent %s: FEP prediction failed, continuing without",
                self.unique_id, exc_info=True,
            )
            self._last_prediction = None

    def _attend(self) -> None:
        """ATTEND: Precision-weighted attentional gating.

        Biological analogy: The thalamic reticular nucleus (TRN) modulates
        sensory gain based on prediction error and current goals. High
        prediction error increases precision (attend more to surprises).
        Active goals bias attention toward goal-relevant signals.

        Uses AttentionalGate (created in Layer 23, injected in Layer 30).
        """
        gate = getattr(self, "_attentional_gate", None)
        if gate is None:
            return

        try:
            # Feed current prediction error to the gate
            pe = getattr(self, "_prediction_error", 0.0)
            if pe > 0:
                gate.set_prediction_error(pe)

            # Goal-directed attention bias: if we have an active goal,
            # boost attention for goal-relevant domains
            gm = getattr(self, "_goal_manager", None)
            if gm is not None:
                goal_ctx = gm.get_context_for_decision()
                if goal_ctx.get("has_goal", False):
                    # Store goal context for _decide() to use later
                    self._goal_context = goal_ctx
                else:
                    self._goal_context = None
            else:
                self._goal_context = None

        except Exception:
            logger.debug(
                "Agent %s: _attend failed", self.unique_id, exc_info=True
            )

    def _observe(self) -> None:
        """Sense environment: stigmergy markers, working memory decay."""
        # Decay working memory (biological: attention fades without rehearsal)
        coordinator = getattr(self, "memory_coordinator", None)
        if coordinator:
            decay = getattr(coordinator, "decay_working_memory", None)
            if decay:
                decay()

        # Sense stigmergy markers (biological: antennae reading pheromones)
        if self.stigmergy_env is not None:
            self._sensed_markers = self.sense_environment()
        else:
            self._sensed_markers: dict[str, list[Any]] = {}

        # FIX-2: Quorum sensing (biological: social insect density sensing)
        if getattr(self, "quorum_sensing_enabled", False):
            try:
                self._quorum_state = self.sense_quorum()
            except Exception:
                logger.debug("Agent %s: quorum sensing failed", self.unique_id, exc_info=True)
                self._quorum_state = None
        else:
            self._quorum_state = None

        # FIX-3: Follow stigmergy trails (biological: chemotaxis gradient following)
        # Agents deposit markers but never read gradients — fix that
        if self.stigmergy_env is not None and hasattr(self, "follow_trail"):
            try:
                self._success_gradient = self.follow_trail("SUCCESS", attractive=True)
            except Exception:
                self._success_gradient = None
            try:
                self._danger_gradient = self.follow_trail("DANGER", attractive=False)
            except Exception:
                self._danger_gradient = None
        else:
            self._success_gradient = None
            self._danger_gradient = None

        # Build state vector for memory and decision-making
        self._prev_state_vector = getattr(self, "_curr_state_vector", None)
        self._curr_state_vector = self._build_state_vector()
        self.current_state = {"state_vector": self._curr_state_vector}

        # Strange Loop: self-awareness feeds back into perception
        # (biological: insular cortex proprioception — the body sensing itself)
        _know_self = getattr(self, "holon_know_self", None)
        if callable(_know_self):
            try:
                self._self_awareness = _know_self()
            except Exception:
                self._self_awareness = None
        else:
            self._self_awareness = None

        # Read latest pattern advisory (organism-level intelligence → agent awareness)
        _advisory_ref = getattr(self, "_pattern_advisory_ref", None)
        if _advisory_ref is not None:
            self._current_advisory = _advisory_ref.get("advisory")

        # --- Organism body state (interoception) ---
        organism = getattr(self, "_organism_state", None)
        if organism is not None:
            try:
                self._body_state = organism.get_body_state()
            except Exception:
                logger.debug("organism state read failed", exc_info=True)
                self._body_state = None
        else:
            self._body_state = None

        # --- Theory of Mind: feed peer observations (biological: mirror neurons) ---
        tom = getattr(self, "_theory_of_mind", None)
        if tom is not None:
            try:
                qs = getattr(self, "quorum_sensor", None)
                if qs is not None and hasattr(qs, "get_recent_signals"):
                    for sig in qs.get_recent_signals():
                        sender = getattr(sig, "sender_id", None)
                        if sender is not None and sender != self.unique_id:
                            tom.update_model(
                                agent_id=sender,
                                action=getattr(sig, "action", None),
                                signal_type=getattr(sig, "type", None),
                            )
            except Exception:
                logger.debug("Agent %s: theory_of_mind update failed", self.unique_id, exc_info=True)

        # --- Causal Reasoning: feed state observations ---
        ce = getattr(self, "causal_engine", None)
        if ce is not None and hasattr(ce, "observe_correlation"):
            try:
                # Record reward ↔ risk correlation for causal inference
                ce.observe_correlation(
                    variable_a="reward",
                    variable_b="risk",
                    correlation_strength=float(self.last_reward) - float(self.risk_score),
                    context={"step": self.step_count, "agent_id": str(self.unique_id)},
                )
            except Exception:
                logger.debug("Agent %s: causal observe failed", self.unique_id, exc_info=True)

        # --- Predictive field spatial awareness ---
        pred_field = getattr(self, "_predictive_field", None)
        if pred_field is not None:
            try:
                pos = getattr(self, "pos", None)
                if pos is not None:
                    self._collision_risks = pred_field.detect_collision_risk(
                        self.unique_id, threshold=2.0
                    )
                    self._coordination_opps = pred_field.find_coordination_opportunities(
                        self.unique_id, max_distance=5.0
                    )
            except Exception:
                logger.debug("predictive field read failed", exc_info=True)

    def _compare(self) -> None:
        """Compare prediction to observation (FEP: prediction error / surprise).

        The prediction error is the core learning signal in predictive
        processing. High error means the world surprised us — amplifying
        learning. Low error means our model is accurate — damping updates.

        Biological analogy: Mismatch negativity in auditory cortex;
        reward prediction error in dopaminergic neurons.

        Graceful when no prediction: sets error to 0.0.
        """
        if self._last_prediction is None:
            self._prediction_error = 0.0
            return

        try:
            # Get the actual observation produced by _observe()
            observation = getattr(self, "_curr_state_vector", None)
            if observation is None:
                self._prediction_error = 0.0
                return

            # Align dimensions: prediction may differ from observation length
            pred = np.asarray(self._last_prediction, dtype=np.float32).flatten()
            obs = np.asarray(observation, dtype=np.float32).flatten()
            min_len = min(len(pred), len(obs))
            if min_len == 0:
                self._prediction_error = 0.0
                return

            # Mean squared error clamped to [0,1] (biological: population-level surprise)
            # Raw MSE is unbounded when state vector has large-magnitude dimensions
            # (cumulative_reward, marker counts). Clamping prevents runaway inhibition.
            error_vec = pred[:min_len] - obs[:min_len]
            self._prediction_error = float(np.clip(np.mean(error_vec ** 2), 0.0, 1.0))

            # Validate imagination against reality (biological: prediction error feedback)
            vi = getattr(self, "_validated_imagination", None)
            pred_id = getattr(self, "_last_prediction_id", None)
            if vi is not None and pred_id is not None:
                try:
                    vi.validate_with_consensus(
                        prediction_id=pred_id,
                        actual_state=obs,
                        actual_reward=getattr(self, "last_reward", 0.0),
                        consensus_confidence=0.5,
                    )
                except Exception:
                    logger.debug("Agent %s: validate_imagination failed", self.unique_id, exc_info=True)

            # Publish prediction error to event bus (neuromodulatory broadcast)
            emit_fn = getattr(self, "emit_signal", None)
            if emit_fn is not None:
                emit_fn(
                    "PREDICTION_ERROR",
                    {
                        "agent_id": str(self.unique_id),
                        "prediction_error": self._prediction_error,
                        "step": self.step_count,
                    },
                )
        except Exception:
            logger.debug(
                "Agent %s: FEP comparison failed",
                self.unique_id, exc_info=True,
            )
            self._prediction_error = 0.0

    def _build_state_vector(self) -> np.ndarray:
        """Build observation vector from internal state (12 dimensions).

        Dims 0-7: core agent state (generic, all agents).
        Dims 8-11: market perception (via market_awareness module).
        Non-market agents get neutral defaults [0.5, 0.0, 0.0, 0.0].
        """
        from mae_core.market.market_awareness import get_market_state_dims

        sensed = getattr(self, "_sensed_markers", {})
        base = [
            self.step_count / 1000.0,
            self.cumulative_reward,
            self.last_reward,
            self.risk_score,
            float(self.has_reached_convergence),
            self.satisfaction_score,
            float(len(sensed.get("SUCCESS", []))),
            float(len(sensed.get("DANGER", []))),
        ]
        base.extend(get_market_state_dims(self))
        return np.array(base, dtype=np.float32)
