"""Learning lifecycle mixin for MycelialAgent.

Extracted from mycelial_agent.py to prevent monolith growth.
These methods are called by step() in mycelial_agent.py.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class LearningLifecycleMixin:
    """Learning lifecycle methods for MycelialAgent.

    All subsystem access is via getattr(self, ..., None) for graceful
    degradation when subsystems are not injected.
    """

    def _learn(self, action: Any, reward: float) -> None:
        """Learn from experience: store memory, deposit stigmergy, replay."""
        super()._learn(action, reward)

        # --- Report outcome to organism state ---
        organism = getattr(self, "_organism_state", None)
        if organism is not None:
            try:
                organism.report_action_outcome(
                    action=str(action),
                    reward=float(reward),
                    step=self.step_count,
                )
            except Exception:
                logger.debug("organism outcome report failed", exc_info=True)

        # --- Feed prediction vs reality to metacognition (biological: RPE signal) ---
        metacog = getattr(self, "_metacognition", None)
        if metacog is not None:
            try:
                predicted = float(getattr(self, "_predicted_reward", 0.0))
                metacog.record_decision(
                    step=self.step_count,
                    predicted=predicted,
                    actual=float(reward),
                )
            except Exception:
                logger.debug("metacognition record_decision failed", exc_info=True)

        # Intrinsic curiosity reward (biological: dopaminergic novelty signal)
        # CuriosityDrive adds exploration bonus to sparse extrinsic rewards
        prev = getattr(self, "_prev_state_vector", None)
        curr = getattr(self, "_curr_state_vector", None)
        curiosity_drive = getattr(self, "curiosity_drive", None) or getattr(self, "_curiosity_drive", None)
        if curiosity_drive is not None and prev is not None and curr is not None:
            try:
                curiosity_signal = curiosity_drive.compute_curiosity_reward(
                    state=prev, action=action, next_state=curr,
                )
                reward = curiosity_drive.combine_rewards(
                    extrinsic=reward, intrinsic=curiosity_signal.total_reward,
                )
            except Exception:
                logger.debug(
                    "Agent %s: curiosity reward computation failed",
                    self.unique_id, exc_info=True,
                )

        # Store experience in episodic memory (biological: hippocampal encoding)
        # Build signal_context from current advisory/pattern state for consensus priority
        signal_context = self._build_signal_context()
        if prev is not None and curr is not None and self.episodic_memory is not None:
            self.store_experience(prev, action, reward, curr, done=False, signal_context=signal_context)

        # Deposit stigmergy markers based on outcome
        if reward > 0:
            self.deposit_success_marker(reward)
        if self.risk_score > 0.5:
            self.deposit_danger_marker(self.risk_score)

        # Periodic memory replay — every 13 steps (Fibonacci: hippocampal replay)
        if self.step_count % 13 == 0 and self.episodic_memory is not None:
            self.learn_from_memory(num_batches=1)

        # Periodic consolidation check — every 89 steps (Fibonacci: sleep cycle)
        if self.step_count % 89 == 0 and self.should_consolidate():
            self.consolidate_memory()

        # Generative Replay: sample synthetic experiences alongside real replay
        # Circuit B: Dream more when struggling (metacognition-driven amplification)
        gen = getattr(self, "generative_memory", None)
        dream_cadence = 13
        dream_batch = 4
        metacog_dream = getattr(self, "_metacognition", None)
        if metacog_dream is not None and not metacog_dream.is_performing_well():
            dream_cadence = 5
            dream_batch = 8
        if gen is not None and hasattr(gen, "sample") and self.step_count % dream_cadence == 0:
            try:
                synthetic_batch = gen.sample(batch_size=dream_batch, synthetic_ratio=0.5)
                if synthetic_batch:
                    synthetic_weights = np.ones(len(synthetic_batch), dtype=np.float32)
                    self._learn_from_batch(synthetic_batch, synthetic_weights)
            except Exception:
                logger.debug("Agent %s: generative replay failed", self.unique_id, exc_info=True)

        # Memory Bridge: store high-reward patterns as ancestral knowledge
        mb = getattr(self, "_memory_bridge", None)
        if mb is not None and reward > 0.7 and self.step_count % 50 == 0:
            try:
                if hasattr(mb, "store_ancestral_pattern"):
                    mb.store_ancestral_pattern(
                        pattern={"action": str(action), "reward": reward,
                                 "step": self.step_count, "context": signal_context},
                        contributing_agents=[str(self.unique_id)],
                    )
            except Exception:
                logger.debug("Agent %s: ancestral pattern store failed", self.unique_id, exc_info=True)

        # Memory Bridge: periodic meta-memory update
        if mb is not None and self.step_count % 100 == 0:
            try:
                if hasattr(mb, "update_meta_memory"):
                    stats = self.episodic_memory.get_statistics() if self.episodic_memory else {}
                    mb.update_meta_memory(memory_stats=stats)
            except Exception:
                logger.debug("Agent %s: meta_memory update failed", self.unique_id, exc_info=True)

        # FIX-4: Activate 5 passive learning subsystems
        # (engines were created but never invoked from agent lifecycle)

        # -- Federated Reinforcement Learning (dynamic cadence) --
        # Circuit A: Share more often when struggling (metacognition-driven)
        frl = getattr(self, "_frl_engine", None)
        frl_freq = getattr(frl, "_share_frequency", 10) if frl is not None else 10
        metacog_frl = getattr(self, "_metacognition", None)
        if metacog_frl is not None and not metacog_frl.is_performing_well():
            frl_freq = max(3, frl_freq // 2)
        if frl is not None and self.step_count % frl_freq == 0:
            try:
                vdn_snap = getattr(self, "_vdn_engine", None)
                if vdn_snap is not None:
                    policy_state = {
                        "q_table_size": len(getattr(vdn_snap, "_q_table", {})),
                        "lr": float(getattr(vdn_snap, "_lr", 0.01)),
                    }
                else:
                    policy_state = {
                        "action": str(action),
                        "reward": float(reward),
                        "step": self.step_count,
                    }
                frl.share_policy_update(
                    policy_state=policy_state,
                    performance=float(reward),
                    metadata={"agent_id": str(self.unique_id), "step": self.step_count},
                )
                peer_updates = frl.receive_policy_updates(max_updates=5)
                if peer_updates:
                    frl.aggregate_policy_updates(
                        local_policy=policy_state,
                        peer_updates=peer_updates,
                    )
            except Exception:
                logger.debug("Agent %s: FRL share/aggregate failed", self.unique_id, exc_info=True)

        # -- Value Decomposition Networks (every 10 steps) --
        vdn = getattr(self, "_vdn_engine", None)
        if vdn is not None and self.step_count % 10 == 0:
            try:
                if hasattr(vdn, "update_value_function") and prev is not None and curr is not None:
                    # Convert action to int index for tabular Q-learning
                    action_dim = getattr(vdn, "_action_dim", 5)
                    if isinstance(action, int):
                        action_int = action % action_dim
                    else:
                        action_int = hash(str(action)) % action_dim
                    vdn.update_value_function(
                        state=prev,
                        action=action_int,
                        reward=reward,
                        next_state=curr,
                        done=False,
                    )
            except Exception:
                logger.debug("Agent %s: VDN update failed", self.unique_id, exc_info=True)

        # -- MAML Meta-Learning (every 50 steps) --
        maml = getattr(self, "_maml_learner", None)
        if maml is not None and self.step_count % 50 == 0:
            try:
                if hasattr(maml, "meta_train"):
                    maml.meta_train(
                        task_family_id=f"agent-{self.unique_id}",
                        num_iterations=1,
                    )
            except Exception:
                logger.debug("Agent %s: MAML meta_train failed", self.unique_id, exc_info=True)

        # -- Transfer Learning (when reward is high, record + attempt transfer) --
        transfer = getattr(self, "_transfer_engine", None)
        if transfer is not None and reward > 0.5 and self.step_count % 20 == 0:
            try:
                if hasattr(transfer, "record_performance"):
                    transfer.record_performance(
                        agent_id=str(self.unique_id),
                        task_id=f"step-{self.step_count}",
                        performance=reward,
                    )
            except Exception:
                logger.debug("Agent %s: Transfer learning failed", self.unique_id, exc_info=True)

        # -- Imitation Learning (observe own + peer behavior) --
        il = getattr(self, "_imitation_learner", None)
        if il is not None and action is not None and self.step_count % 5 == 0:
            try:
                if hasattr(il, "observe_behavior"):
                    # Observe self
                    il.observe_behavior(
                        actor_id=str(self.unique_id),
                        action=action,
                        context={"reward": reward, "step": self.step_count},
                        outcome=reward,
                    )
                    # Observe successful peers via quorum signals
                    qs = getattr(self, "quorum_sensor", None)
                    if qs is not None and hasattr(qs, "get_recent_signals"):
                        for sig in qs.get_recent_signals():
                            peer_reward = getattr(sig, "reward", 0.0)
                            if peer_reward > 0.5:
                                il.observe_behavior(
                                    actor_id=str(getattr(sig, "sender_id", "unknown")),
                                    action=getattr(sig, "action", None),
                                    context={"source": "quorum", "step": self.step_count},
                                    outcome=peer_reward,
                                )
            except Exception:
                logger.debug("Agent %s: Imitation observe failed", self.unique_id, exc_info=True)

        # Per-agent pattern sense (cell membrane)
        _ps = getattr(self, "_pattern_sense", None)
        if _ps is not None:
            self._last_sense_result = _ps.sense(reward, action, self.step_count)

        # Close the prediction-training loop (Law 6: Autopoietic Closure)
        # WorldModel predicts → prediction error drives learning →
        # learning improves WorldModel → better predictions.
        # (biological: cerebellar forward-model adaptation — the cerebellum
        # continuously refines its internal model from sensorimotor experience)
        wm = getattr(self, "world_model", None)
        if wm is not None and prev is not None and curr is not None:
            try:
                # Encode action for WorldModel: integer actions pass through,
                # dict/string actions map to a default index (WorldModel
                # handles encoding internally via _encode_action)
                wm_action = action if isinstance(action, (int, np.integer)) else 0

                # Train on single experience: wrap scalars into batch-of-one
                wm.train_step(
                    states=prev.reshape(1, -1),
                    actions=np.array([wm_action]),
                    next_states=curr.reshape(1, -1),
                    rewards=np.array([reward], dtype=np.float32),
                )
                self._wm_train_steps += 1
            except Exception:
                logger.debug(
                    "Agent %s: WorldModel train_step failed in _learn",
                    self.unique_id, exc_info=True,
                )

        # Tick reconsolidation windows and spreading activation decay
        tick = getattr(self, "tick_episodic_memory", None)
        if tick:
            try:
                tick()
            except Exception:
                pass

    def _learn_from_batch(
        self, batch: list[Any], weights: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Learn from a batch of experiences with TD-error computation.

        When a world_model is available, computes proper temporal-difference
        errors: td = reward + gamma * V(next_state) - V(state). The
        prediction error from FEP comparison modulates learning rate —
        higher surprise amplifies parameter updates (biological: dopaminergic
        gating of synaptic plasticity).

        Without world_model, falls back to reward-as-TD-error (the
        immediate reward IS the learning signal, equivalent to V=0).

        Returns (td_errors, loss) for priority-buffer updates.
        """
        wm = getattr(self, "world_model", None)
        if wm is None:
            # No value function available — reward is the signal
            return super()._learn_from_batch(batch, weights)

        gamma = 0.99  # Standard temporal discount factor

        td_errors = np.zeros(len(batch), dtype=np.float32)
        for i, exp in enumerate(batch):
            reward = getattr(exp, "reward", 0.0)
            state = getattr(exp, "state", None)
            next_state = getattr(exp, "next_state", None)

            if state is None or next_state is None:
                # Incomplete experience — fall back to reward signal
                td_errors[i] = reward
                continue

            try:
                # V(state) approximated via world_model predicted reward
                # (biological: expected value from hippocampal-prefrontal loop)
                v_state = wm.predict_reward(
                    np.asarray(state, dtype=np.float32),
                    getattr(exp, "action", 0),
                )
                # V(next_state) via world_model one-step lookahead
                v_next = wm.predict_reward(
                    np.asarray(next_state, dtype=np.float32),
                    0,  # Default action for value estimate
                )
                td_errors[i] = reward + gamma * v_next - v_state
            except Exception:
                # World model prediction failed — use raw reward
                td_errors[i] = reward

        # FEP modulation: prediction error amplifies learning
        # (biological: high surprise = enhanced LTP at active synapses)
        pred_error = getattr(self, "_prediction_error", 0.0)
        modulation = 1.0 + min(pred_error, 1.0)  # Clamp [1.0, 2.0]

        # Weighted loss with surprise modulation
        loss = float(np.mean(np.abs(td_errors) * weights) * modulation)

        # Train WorldModel on replayed batch (Law 6: Autopoietic Closure)
        # Memory replay gives the model many more training samples than
        # online-only learning. (biological: hippocampal replay during
        # sleep also updates the cerebellar forward model)
        try:
            # Collect valid experiences with complete (state, next_state) pairs
            valid = [
                exp for exp in batch
                if getattr(exp, "state", None) is not None
                and getattr(exp, "next_state", None) is not None
            ]
            if valid:
                b_states = np.array(
                    [np.asarray(e.state, dtype=np.float32).flatten() for e in valid]
                )
                b_actions = np.array([
                    e.action if isinstance(getattr(e, "action", 0), (int, np.integer)) else 0
                    for e in valid
                ])
                b_next = np.array(
                    [np.asarray(e.next_state, dtype=np.float32).flatten() for e in valid]
                )
                b_rewards = np.array(
                    [getattr(e, "reward", 0.0) for e in valid], dtype=np.float32
                )
                # Use importance-sampling weights for the valid subset
                valid_indices = [
                    i for i, exp in enumerate(batch)
                    if getattr(exp, "state", None) is not None
                    and getattr(exp, "next_state", None) is not None
                ]
                b_weights = weights[valid_indices] if len(valid_indices) == len(valid) else None
                wm.train_step(b_states, b_actions, b_next, b_rewards, b_weights)
                self._wm_train_steps += 1
        except Exception:
            logger.debug(
                "Agent %s: WorldModel batch training failed in _learn_from_batch",
                getattr(self, "unique_id", "?"), exc_info=True,
            )

        return td_errors, loss

    def _manage_goals(self, reward: float) -> None:
        """GOAL: Track progress, detect impasses, manage goal stack.

        Biological analogy: The prefrontal cortex maintains persistent goals
        in working memory. The anterior cingulate cortex (ACC) monitors
        progress and detects when the current strategy isn't working (impasse).
        Impasse triggers strategy shift toward exploration.

        Called after _learn() so reward information is available.
        """
        gm = getattr(self, "_goal_manager", None)
        if gm is None:
            return

        try:
            result = gm.update_progress(
                reward=reward,
                step=self.step_count,
                agent_id=str(self.unique_id),
            )

            if result.get("impasse_detected", False):
                logger.debug(
                    "Agent %s: impasse detected on goal '%s' (%d steps stuck)",
                    self.unique_id,
                    result.get("current_goal", "?"),
                    result.get("steps_since_progress", 0),
                )

            # Feed goal priority to arousal regulator (goals affect arousal target)
            reg = getattr(self, "_arousal_regulator", None)
            if reg is not None:
                reg.record_reward(max(0.0, min(1.0, reward)))

        except Exception:
            logger.debug(
                "Agent %s: _manage_goals failed", self.unique_id, exc_info=True
            )

    def _build_signal_context(self) -> dict[str, Any] | None:
        """Build signal_context for consensus-weighted memory storage.

        Packages current advisory and pattern state so that
        store_experience can compute consensus-based priority
        via the quorum sensor (biological: neuromodulatory tagging
        of salient experiences for preferential consolidation).
        """
        advisory = getattr(self, "_current_advisory", None)
        sense_result = getattr(self, "_last_sense_result", None)
        if advisory is None and sense_result is None:
            return None

        ctx: dict[str, Any] = {}
        if advisory is not None:
            dominant = getattr(advisory, "dominant_pattern", None)
            ctx["signal_type"] = (
                f"{dominant.domain.value}:{dominant.description}"
                if dominant is not None
                else "ambient"
            )
            ctx["metadata"] = {
                "threat_level": getattr(advisory, "threat_level", 0.0),
                "opportunity_level": getattr(advisory, "opportunity_level", 0.0),
                "confidence": getattr(advisory, "confidence", 0.0),
            }
        if sense_result is not None:
            ctx.setdefault("metadata", {})["pattern_signals"] = getattr(
                sense_result, "signals", []
            )
        return ctx
