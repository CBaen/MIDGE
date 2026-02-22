"""Episodic Memory Mixin - Experience replay and consolidation.

Provides prioritized experience replay, memory consolidation ("sleep"),
semantic search over past experiences, generative replay via VAE,
triadic recall verification (Law 1: No Bare Dyads), memory reconsolidation,
and spreading activation for associative recall.

Triadic Recall Verification:
  Every memory recall is verified by three independent witnesses before
  being trusted. Without verification, recall is a bare dyad (agent <-> memory)
  with no witness. Biological analogy: memory reconsolidation — the brain
  cross-checks recalled memories against semantic context, temporal neighbors,
  and statistical baselines before acting on them.

  Witness 1 (Semantic): Does the memory's context match via similarity?
  Witness 2 (Temporal): Are neighboring memories consistent?
  Witness 3 (Statistical): Is the memory's reward within normal bounds?

Memory Reconsolidation (Nader et al. 2000):
  When a consolidated memory is recalled and the current experience
  contradicts it (prediction error > threshold), the memory enters a
  labile state — a reconsolidation window. During this window (5 steps,
  scaled from the biological ~6-hour window), the memory can be updated
  with new evidence. The update blends old and new: alpha * old + (1-alpha)
  * new, where alpha=0.6 gives the original memory inertia. After the
  window closes, the memory re-stabilizes (reconsolidates) with its
  modifications. This mirrors the biological process where retrieval
  destabilizes synaptic traces, requiring de novo protein synthesis to
  restabilize them — during which the memory is susceptible to modification.

  References:
    - Nader, Schafe & Le Doux (2000) "Fear memories require protein
      synthesis in the amygdala for reconsolidation after retrieval" Nature
    - Reconsolidation window ~6h in biology (anisomycin-sensitive period)
    - Prediction error as reconsolidation trigger (Pedreira et al. 2004)

Spreading Activation (Collins & Loftus 1975):
  When a memory is recalled, activation spreads to semantically related
  memories. Activation decays with distance: each hop multiplies by a
  decay factor (0.7). Maximum spread depth is 2 hops. Activated memories
  receive a temporary salience boost that makes them more likely to surface
  on subsequent recalls. Activation decays each step at 0.85 (matching
  the habituation rate), creating a priming window where recent recalls
  prime related memories — just as recalling "doctor" primes "nurse".

  References:
    - Collins & Loftus (1975) "A spreading-activation theory of semantic
      processing" Psychological Review 82:407-428
    - Activation formula: A[j] += A[i] * similarity * decay_factor

Ported from v5-pivot base_agent.py Big Rock 9 memory methods.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EpisodicMemoryMixin:
    """Adds episodic memory, replay, consolidation, and semantic search to agents."""

    def _init_episodic_memory(
        self,
        episodic_memory: Any = None,
        memory_consolidator: Any = None,
        semantic_retriever: Any = None,
        generative_memory: Any = None,
        agent_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize episodic memory attributes."""
        config = agent_config or {}
        self.episodic_memory = episodic_memory
        self.memory_consolidator = memory_consolidator
        self.semantic_retriever = semantic_retriever
        self.generative_memory = generative_memory

        self.replay_enabled: bool = config.get("replay_enabled", False)
        self.consolidation_enabled: bool = config.get("consolidation_enabled", False)
        self.semantic_search_enabled: bool = config.get("semantic_search_enabled", False)
        self.generative_memory_enabled: bool = config.get("generative_memory_enabled", False)
        self.replay_frequency: int = config.get("replay_frequency", 4)
        self.replay_batch_size: int = config.get("replay_batch_size", 32)
        self.steps_since_replay: int = 0
        self.total_replays: int = 0
        self.total_consolidations: int = 0

        # Triadic recall verification: running reward statistics for Witness 3
        self._reward_history: deque[float] = deque(maxlen=1000)
        self._reward_sum: float = 0.0
        self._reward_sq_sum: float = 0.0
        self._recall_verifications: int = 0
        self._recall_trusted: int = 0
        self._recall_partial: int = 0
        self._recall_untrusted: int = 0

        # --- Reconsolidation state (Nader et al. 2000) ---
        # Maps memory id -> {memory, original_reward, labile_step, updated}
        # Labile memories are those currently in the reconsolidation window.
        self._labile_memories: dict[int, dict[str, Any]] = {}
        self._reconsolidation_threshold: float = config.get(
            "reconsolidation_threshold", 0.3,
        )
        self._reconsolidation_window: int = config.get(
            "reconsolidation_window", 5,
        )  # steps (scaled from biological ~6 hours)
        self._reconsolidation_alpha: float = config.get(
            "reconsolidation_alpha", 0.6,
        )  # old memory inertia
        self._reconsolidation_events: int = 0
        self._reconsolidation_updates: int = 0
        self._reconsolidation_stabilizations: int = 0

        # --- Spreading activation state (Collins & Loftus 1975) ---
        # Maps memory id -> current activation level (0.0 to 1.0)
        self._spreading_activation: dict[int, float] = {}
        self._spreading_decay_factor: float = config.get(
            "spreading_decay_factor", 0.7,
        )  # activation weakens per hop
        self._spreading_max_depth: int = config.get(
            "spreading_max_depth", 2,
        )  # primary -> secondary -> tertiary
        self._spreading_step_decay: float = config.get(
            "spreading_step_decay", 0.85,
        )  # per-step decay (matches habituation rate)
        self._spreading_activation_events: int = 0

    def store_experience(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[dict[str, Any]] = None,
        signal_context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Store experience in episodic memory for replay learning.

        Phase 3 integration: If signal_context is provided and a quorum_sensor
        exists, consensus-based priority is used. High-consensus experiences
        are learned from more frequently.

        Phase 4 integration: Also stores in generative memory if enabled,
        providing 100x memory compression via VAE-based generative replay.
        """
        if self.episodic_memory is None or not self.replay_enabled:
            return

        priority = None
        quorum_sensor = getattr(self, "quorum_sensor", None)
        if signal_context and quorum_sensor:
            try:
                priority = quorum_sensor.get_consensus_priority(
                    signal_type=signal_context.get("signal_type"),
                    metadata=signal_context.get("metadata"),
                )
            except Exception:
                logger.exception("Error computing consensus priority")

        from mae_core.memory.experience import Experience

        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {},
        )
        self.episodic_memory.add(experience, priority=priority)

        # Track reward statistics for triadic recall verification (Witness 3)
        self._reward_history.append(reward)
        self._reward_sum += reward
        self._reward_sq_sum += reward * reward

        if self.generative_memory_enabled and self.generative_memory is not None:
            try:
                from mae_core.memory.experience import Experience
                self.generative_memory.store(Experience(
                    state=state, action=action, reward=reward,
                    next_state=next_state, done=done, info=info or {},
                ))
            except Exception:
                logger.exception("Error storing in generative memory")

    def learn_from_memory(
        self,
        num_batches: int = 1,
        batch_size: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """Learn from experiences stored in episodic memory.

        Phase 4: Can use generative memory for hybrid replay
        (mixing real and synthetic experiences).
        """
        if self.generative_memory_enabled and self.generative_memory is not None:
            return self._use_generative_memory(num_batches, batch_size)

        if self.episodic_memory is None or not self.replay_enabled:
            return None

        bs = batch_size or self.replay_batch_size
        if len(self.episodic_memory) < bs:
            return None

        total_loss = 0.0
        total_td_error = 0.0
        experiences_replayed = 0

        for _ in range(num_batches):
            batch, indices, weights = self.episodic_memory.sample(batch_size=bs)
            td_errors, loss = self._learn_from_batch(batch, weights)
            self.episodic_memory.update_priorities(indices, td_errors)
            total_loss += loss
            total_td_error += np.mean(np.abs(td_errors))
            experiences_replayed += len(batch)

        self.total_replays += num_batches

        return {
            "num_batches": num_batches,
            "batch_size": bs,
            "experiences_replayed": experiences_replayed,
            "mean_loss": total_loss / num_batches,
            "mean_td_error": total_td_error / num_batches,
            "total_replays": self.total_replays,
            "memory_size": len(self.episodic_memory),
        }

    def _learn_from_batch(
        self, batch: list[Any], weights: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Learn from a batch of experiences. Subclasses should override.

        Default: use reward as TD error signal. Without a value network,
        the immediate reward IS the learning signal.
        """
        td_errors = np.array(
            [getattr(exp, "reward", 0.0) for exp in batch], dtype=np.float32
        )
        loss = float(np.mean(np.abs(td_errors) * weights))
        return td_errors, loss

    def _use_generative_memory(
        self,
        num_batches: int = 1,
        batch_size: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """Learn from VAE-generated synthetic + real experiences."""
        if not self.generative_memory or not self.generative_memory_enabled:
            return None

        bs = batch_size or self.replay_batch_size
        total_loss = 0.0
        total_td_error = 0.0
        experiences_replayed = 0
        synthetic_count = 0

        for _ in range(num_batches):
            batch = self.generative_memory.sample(batch_size=bs)
            synthetic_count += sum(
                1 for exp in batch if getattr(exp, "info", {}).get("is_synthetic", False)
            )
            weights = np.ones(len(batch)) / len(batch)
            td_errors, loss = self._learn_from_batch(batch, weights)
            total_loss += loss
            total_td_error += np.mean(np.abs(td_errors))
            experiences_replayed += len(batch)

        self.total_replays += num_batches

        return {
            "num_batches": num_batches,
            "batch_size": bs,
            "experiences_replayed": experiences_replayed,
            "synthetic_count": synthetic_count,
            "synthetic_ratio": synthetic_count / experiences_replayed if experiences_replayed > 0 else 0.0,
            "mean_loss": total_loss / num_batches,
            "mean_td_error": total_td_error / num_batches,
            "total_replays": self.total_replays,
            "memory_type": "generative",
        }

    def consolidate_memory(
        self,
        num_steps: Optional[int] = None,
        strategy: Any = None,
    ) -> Optional[Any]:
        """Perform memory consolidation ('sleep' phase) for offline learning."""
        if not self.memory_consolidator or not self.consolidation_enabled:
            return None

        min_size = getattr(self.memory_consolidator, "min_memory_size", 0)
        if self.episodic_memory is not None and len(self.episodic_memory) < min_size:
            return None

        try:
            result = self.memory_consolidator.consolidate(
                agent=self, num_steps=num_steps, strategy=strategy,
            )
            self.total_consolidations += 1

            emit_signal = getattr(self, "emit_signal", None)
            if emit_signal is not None:
                emit_signal(
                    "LEARNING_MILESTONE",
                    {
                        "event": "memory_consolidation",
                        "loss_reduction": getattr(result, "loss_reduction", 0.0),
                    },
                )

            return result
        except Exception:
            logger.exception("Memory consolidation failed")
            return None

    def should_consolidate(self) -> bool:
        """Check if memory consolidation should be triggered."""
        if not self.memory_consolidator or not self.consolidation_enabled:
            return False
        step_count = getattr(self, "step_count", 0)
        return self.memory_consolidator.should_consolidate(current_step=step_count)

    def verify_recall(
        self, memory: Any, query_state: Optional[np.ndarray] = None,
    ) -> tuple[Any, float]:
        """Verify a recalled memory using three independent witnesses (Law 1).

        Every recall is a connection agent<->memory. Without verification
        this is a bare dyad. Three witnesses provide triadic witnessing:

        Witness 1 (Semantic): Check if the memory's state matches the query
          context via cosine similarity. Similarity > 0.3 = corroboration.
        Witness 2 (Temporal): Check if neighboring memories (within +/-2 steps
          of the recalled memory's timestamp) have consistent state vectors.
          Consistency > 0.5 = corroboration.
        Witness 3 (Statistical): Check if the memory's reward is within 2
          standard deviations of the agent's mean reward. Not an extreme
          outlier = corroboration.

        Verdict:
          2+ witnesses corroborate -> trusted (confidence = 1.0)
          1 witness corroborates   -> partial trust (confidence = 0.6)
          0 witnesses corroborate  -> untrusted (confidence = 0.3)

        Returns:
            (memory, confidence) tuple. Graceful: if verification fails
            entirely, returns (memory, 0.8) — benefit of the doubt.
        """
        try:
            corroborations = 0

            # --- Witness 1: Semantic similarity ---
            try:
                semantic = getattr(self, "semantic_retriever", None)
                mem_state = getattr(memory, "state", None)
                if (
                    semantic is not None
                    and query_state is not None
                    and mem_state is not None
                ):
                    encode_state = getattr(semantic, "encode_state", None)
                    encode_exp = getattr(semantic, "encode_experience", None)
                    if encode_state is not None and encode_exp is not None:
                        q_vec = encode_state(query_state)
                        m_vec = encode_exp(memory)
                        # Cosine similarity (both are L2-normalized by SemanticRetriever)
                        q_flat = np.asarray(q_vec, dtype=np.float32).flatten()
                        m_flat = np.asarray(m_vec, dtype=np.float32).flatten()
                        min_len = min(len(q_flat), len(m_flat))
                        if min_len > 0:
                            dot = float(np.dot(q_flat[:min_len], m_flat[:min_len]))
                            norm_q = float(np.linalg.norm(q_flat[:min_len]))
                            norm_m = float(np.linalg.norm(m_flat[:min_len]))
                            if norm_q > 0 and norm_m > 0:
                                similarity = dot / (norm_q * norm_m)
                                if similarity > 0.3:
                                    corroborations += 1
                    elif query_state is not None and mem_state is not None:
                        # Fallback: direct state vector cosine similarity
                        q_flat = np.asarray(query_state, dtype=np.float32).flatten()
                        m_flat = np.asarray(mem_state, dtype=np.float32).flatten()
                        min_len = min(len(q_flat), len(m_flat))
                        if min_len > 0:
                            dot = float(np.dot(q_flat[:min_len], m_flat[:min_len]))
                            norm_q = float(np.linalg.norm(q_flat[:min_len]))
                            norm_m = float(np.linalg.norm(m_flat[:min_len]))
                            if norm_q > 0 and norm_m > 0:
                                similarity = dot / (norm_q * norm_m)
                                if similarity > 0.3:
                                    corroborations += 1
                elif query_state is not None and mem_state is not None:
                    # No semantic retriever — use raw state similarity
                    q_flat = np.asarray(query_state, dtype=np.float32).flatten()
                    m_flat = np.asarray(mem_state, dtype=np.float32).flatten()
                    min_len = min(len(q_flat), len(m_flat))
                    if min_len > 0:
                        dot = float(np.dot(q_flat[:min_len], m_flat[:min_len]))
                        norm_q = float(np.linalg.norm(q_flat[:min_len]))
                        norm_m = float(np.linalg.norm(m_flat[:min_len]))
                        if norm_q > 0 and norm_m > 0:
                            similarity = dot / (norm_q * norm_m)
                            if similarity > 0.3:
                                corroborations += 1
            except Exception:
                logger.debug("Recall verification: Witness 1 (semantic) failed", exc_info=True)

            # --- Witness 2: Temporal consistency ---
            try:
                mem_timestamp = getattr(memory, "timestamp", None)
                mem_state = getattr(memory, "state", None)
                if mem_timestamp is not None and mem_state is not None:
                    # Search for temporally neighboring memories
                    # Use semantic retriever to find similar states (proxy for temporal neighbors)
                    semantic = getattr(self, "semantic_retriever", None)
                    if semantic is not None and getattr(self, "semantic_search_enabled", False):
                        neighbors = semantic.search_by_state(
                            np.asarray(mem_state, dtype=np.float32), k=5,
                        )
                        neighbor_exps = getattr(neighbors, "experiences", []) if neighbors else []
                        # Filter to temporal neighbors: within +/-2 timestamp units
                        temporal_neighbors = []
                        for n in neighbor_exps:
                            n_ts = getattr(n, "timestamp", None)
                            if n_ts is not None and abs(n_ts - mem_timestamp) <= 2.0:
                                if n is not memory:  # Exclude the memory itself
                                    temporal_neighbors.append(n)

                        if temporal_neighbors:
                            # Check state vector consistency with neighbors
                            m_flat = np.asarray(mem_state, dtype=np.float32).flatten()
                            consistencies = []
                            for neighbor in temporal_neighbors:
                                n_state = getattr(neighbor, "state", None)
                                if n_state is not None:
                                    n_flat = np.asarray(n_state, dtype=np.float32).flatten()
                                    min_len = min(len(m_flat), len(n_flat))
                                    if min_len > 0:
                                        dot = float(np.dot(m_flat[:min_len], n_flat[:min_len]))
                                        norm_m = float(np.linalg.norm(m_flat[:min_len]))
                                        norm_n = float(np.linalg.norm(n_flat[:min_len]))
                                        if norm_m > 0 and norm_n > 0:
                                            consistencies.append(dot / (norm_m * norm_n))
                            if consistencies:
                                avg_consistency = sum(consistencies) / len(consistencies)
                                if avg_consistency > 0.5:
                                    corroborations += 1
            except Exception:
                logger.debug("Recall verification: Witness 2 (temporal) failed", exc_info=True)

            # --- Witness 3: Statistical normality ---
            try:
                mem_reward = getattr(memory, "reward", None)
                if mem_reward is not None and len(self._reward_history) >= 3:
                    n = len(self._reward_history)
                    mean = self._reward_sum / n
                    variance = (self._reward_sq_sum / n) - (mean * mean)
                    std = max(np.sqrt(max(variance, 0.0)), 1e-8)
                    z_score = abs(mem_reward - mean) / std
                    # Within 2 standard deviations = not an outlier
                    if z_score <= 2.0:
                        corroborations += 1
                elif mem_reward is not None:
                    # Not enough data for statistics — give benefit of the doubt
                    corroborations += 1
            except Exception:
                logger.debug("Recall verification: Witness 3 (statistical) failed", exc_info=True)

            # --- Verdict ---
            self._recall_verifications += 1
            if corroborations >= 2:
                confidence = 1.0
                self._recall_trusted += 1
            elif corroborations == 1:
                confidence = 0.6
                self._recall_partial += 1
            else:
                confidence = 0.3
                self._recall_untrusted += 1

            return (memory, confidence)

        except Exception:
            logger.debug("Recall verification failed entirely, returning default confidence", exc_info=True)
            return (memory, 0.8)

    # ------------------------------------------------------------------
    # Reconsolidation (Nader et al. 2000)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Spreading Activation (Collins & Loftus 1975)
    # ------------------------------------------------------------------

    def _spread_activation(
        self,
        primary_memories: list[Any],
        primary_scores: list[float],
        query_state: np.ndarray,
    ) -> None:
        """Spread activation from recalled memories to their semantic neighbors.

        Biological basis (Collins & Loftus 1975): When a concept node in
        semantic memory is activated, activation propagates along associative
        links to related nodes. The strength of propagation depends on the
        similarity (link weight) between nodes and decays with distance.
        This creates priming: recalling "doctor" activates "nurse", "hospital",
        "stethoscope" — making them easier to retrieve on subsequent queries.

        Implementation:
          1. Each primary recalled memory gets activation = its relevance score
          2. For each primary, find k nearest neighbors via semantic retriever
          3. Neighbor activation = primary_activation * similarity * decay_factor
          4. For depth 2: each activated neighbor spreads further (with decay^2)
          5. Activations are stored in _spreading_activation dict (keyed by id)
          6. On next recall, boosted memories get salience multiplier

        Args:
            primary_memories: The memories just recalled (search results).
            primary_scores: Relevance scores for each primary memory.
            query_state: The state vector used for the query.
        """
        try:
            semantic = getattr(self, "semantic_retriever", None)
            if semantic is None or not getattr(
                self, "semantic_search_enabled", False
            ):
                return

            decay_factor = getattr(self, "_spreading_decay_factor", 0.7)
            max_depth = getattr(self, "_spreading_max_depth", 2)
            activation_map = getattr(self, "_spreading_activation", {})

            # Set primary activations
            activated_this_round: dict[int, float] = {}
            for i, mem in enumerate(primary_memories):
                score = primary_scores[i] if i < len(primary_scores) else 0.5
                mem_id = id(mem)
                activated_this_round[mem_id] = max(
                    activated_this_round.get(mem_id, 0.0), float(score)
                )

            # Spread through depth levels
            current_frontier = list(activated_this_round.items())  # (id, activation)
            for depth in range(max_depth):
                next_frontier: list[tuple[int, float]] = []
                for _mem_id, activation in current_frontier:
                    if activation < 0.01:
                        continue  # Below threshold — don't propagate noise

                    # Find the memory object for this id from primary_memories
                    source_mem = None
                    for m in primary_memories:
                        if id(m) == _mem_id:
                            source_mem = m
                            break

                    if source_mem is None:
                        # Check if we have it stored from a previous depth
                        continue

                    source_state = getattr(source_mem, "state", None)
                    if source_state is None:
                        continue

                    # Search for neighbors of this memory
                    try:
                        neighbor_result = semantic.search_by_state(
                            np.asarray(source_state, dtype=np.float32),
                            k=5,
                        )
                        if neighbor_result is None:
                            continue
                        neighbor_exps = getattr(
                            neighbor_result, "experiences", []
                        )
                        neighbor_scores = getattr(
                            neighbor_result, "scores", []
                        )
                    except Exception:
                        continue

                    for j, neighbor in enumerate(neighbor_exps):
                        n_id = id(neighbor)
                        if n_id == _mem_id:
                            continue  # Don't self-activate

                        similarity = (
                            float(neighbor_scores[j])
                            if j < len(neighbor_scores)
                            else 0.3
                        )
                        neighbor_activation = (
                            activation * similarity * decay_factor
                        )

                        if neighbor_activation < 0.01:
                            continue

                        # Accumulate (don't replace — stronger path wins)
                        old = activated_this_round.get(n_id, 0.0)
                        new_val = max(old, neighbor_activation)
                        activated_this_round[n_id] = new_val

                        if depth < max_depth - 1:
                            # Track for next hop — store the memory for lookup
                            next_frontier.append((n_id, neighbor_activation))
                            # Add to primary_memories for next-depth lookup
                            if neighbor not in primary_memories:
                                primary_memories.append(neighbor)

                current_frontier = next_frontier

            # Merge into persistent activation map (max of old and new)
            for mem_id, act in activated_this_round.items():
                old_act = activation_map.get(mem_id, 0.0)
                activation_map[mem_id] = max(old_act, act)

            self._spreading_activation = activation_map

            events = getattr(self, "_spreading_activation_events", 0)
            self._spreading_activation_events = events + 1

            logger.debug(
                "Spreading activation: %d memories activated (from %d primaries)",
                len(activated_this_round),
                len(primary_memories),
            )
        except Exception:
            logger.debug(
                "Spreading activation failed gracefully", exc_info=True,
            )

    def _apply_activation_boost(
        self,
        experiences: list[Any],
        scores: list[float],
    ) -> list[float]:
        """Apply spreading activation salience boost to recall scores.

        Memories that were activated by previous spreading activation get
        their relevance scores multiplied by (1 + activation_level). This
        implements the priming effect: recently-activated memories are
        easier to retrieve, mirroring the behavioral finding that response
        times to primed words are faster (Collins & Loftus 1975).

        Args:
            experiences: List of recalled experience objects.
            scores: Their current relevance scores.

        Returns:
            New list of boosted scores.
        """
        try:
            activation_map = getattr(self, "_spreading_activation", None)
            if not activation_map:
                return scores

            boosted: list[float] = []
            for i, exp in enumerate(experiences):
                base_score = scores[i] if i < len(scores) else 0.5
                mem_id = id(exp)
                activation = activation_map.get(mem_id, 0.0)
                # Boost = 1 + activation (so max 2x for activation=1.0)
                boosted_score = base_score * (1.0 + activation)
                boosted.append(boosted_score)

            return boosted
        except Exception:
            logger.debug(
                "Activation boost failed gracefully", exc_info=True,
            )
            return scores

    def _decay_spreading_activation(self) -> None:
        """Decay all spreading activations by the step decay rate.

        Called once per agent step. Activation decays at 0.85 per step
        (matching the habituation rate), so a primed memory loses ~15%
        of its activation each step. This creates a natural priming window:
        memories primed 1-3 steps ago are strongly activated, while older
        activations fade toward zero.

        Activations below 0.01 are pruned to prevent unbounded growth
        of the activation map.
        """
        try:
            activation_map = getattr(self, "_spreading_activation", None)
            if not activation_map:
                return

            step_decay = getattr(self, "_spreading_step_decay", 0.85)
            to_prune: list[int] = []

            for mem_id in activation_map:
                activation_map[mem_id] *= step_decay
                if activation_map[mem_id] < 0.01:
                    to_prune.append(mem_id)

            for mem_id in to_prune:
                del activation_map[mem_id]
        except Exception:
            logger.debug(
                "Activation decay failed gracefully", exc_info=True,
            )

    # ------------------------------------------------------------------
    # Combined step tick for reconsolidation + spreading activation
    # ------------------------------------------------------------------

    def tick_episodic_memory(self) -> None:
        """Advance episodic memory subsystems by one step.

        Call once per agent step to:
        1. Decay spreading activation (priming fades over time)
        2. Check/close reconsolidation windows (labile memories stabilize)

        Both operations are safe no-ops if the respective features have
        not been triggered (no labile memories, no active spreads).
        """
        self._decay_spreading_activation()
        self._tick_reconsolidation()

    # ------------------------------------------------------------------
    # Main recall interface (updated with reconsolidation + spreading)
    # ------------------------------------------------------------------

    def search_similar_experiences(
        self, state: np.ndarray, k: int = 5,
        current_reward: Optional[float] = None,
    ) -> Optional[Any]:
        """Search for similar past experiences using semantic retrieval.

        After retrieval, each memory undergoes triadic recall verification
        (Law 1: No Bare Dyads). Results are re-sorted by (confidence * relevance)
        so the most relevant AND most verified memory wins.

        Post-verification, two additional neuroscience-grounded processes run:

        1. **Reconsolidation** (Nader et al. 2000): If the top recalled memory's
           reward significantly differs from the current reward context
           (prediction error > 0.3), the memory enters a labile state for 5
           steps. During this window it can be updated with blended evidence.

        2. **Spreading activation** (Collins & Loftus 1975): Activation spreads
           from recalled memories to their semantic neighbors. On subsequent
           recalls, primed memories get a salience boost.

        Args:
            state: Current state vector for similarity search.
            k: Number of nearest neighbors to retrieve.
            current_reward: Optional current reward for reconsolidation check.
                If not provided, falls back to most recent reward in history.

        Returns:
            SemanticQuery object with .experiences and .scores reordered by
            verified relevance (with activation boost applied).
        """
        if not self.semantic_retriever or not self.semantic_search_enabled:
            return None

        result = self.semantic_retriever.search_by_state(state, k=k)
        if result is None:
            return None

        experiences = getattr(result, "experiences", None)
        scores = getattr(result, "scores", None)
        if not experiences:
            return result

        # Verify each recalled memory and compute verified relevance
        verified_entries: list[tuple[Any, float, float]] = []  # (exp, score, confidence)
        for i, exp in enumerate(experiences):
            relevance = scores[i] if scores and i < len(scores) else 0.5
            try:
                _exp, confidence = self.verify_recall(exp, query_state=state)
                verified_entries.append((exp, relevance, confidence))
            except Exception:
                # Graceful: unverified memory gets default confidence 0.8
                verified_entries.append((exp, relevance, 0.8))

        # Sort by (confidence * relevance), descending
        verified_entries.sort(key=lambda e: e[1] * e[2], reverse=True)

        # Extract sorted lists
        sorted_experiences = [e[0] for e in verified_entries]
        sorted_scores = [e[1] * e[2] for e in verified_entries]

        # --- Spreading activation: boost primed memories ---
        sorted_scores = self._apply_activation_boost(
            sorted_experiences, sorted_scores,
        )

        # --- Reconsolidation: check top memories for prediction error ---
        for exp in sorted_experiences[:3]:  # Check top 3 recalled memories
            try:
                self._check_reconsolidation(exp, current_reward)
            except Exception:
                logger.debug(
                    "Reconsolidation check skipped for memory", exc_info=True,
                )

        # --- Spreading activation: spread from recalled memories ---
        try:
            self._spread_activation(
                list(sorted_experiences), list(sorted_scores), state,
            )
        except Exception:
            logger.debug(
                "Spreading activation skipped", exc_info=True,
            )

        # Re-sort after activation boost (boost may change ordering)
        combined = list(zip(sorted_experiences, sorted_scores))
        combined.sort(key=lambda x: x[1], reverse=True)

        # Rebuild the result object with reordered data
        try:
            result.experiences = [c[0] for c in combined]
            result.scores = [c[1] for c in combined]
        except (AttributeError, TypeError):
            # Result object is immutable or doesn't support assignment — return as-is
            pass

        return result

    def get_counterfactual_experiences(
        self, state: np.ndarray, action: int, k: int = 3
    ) -> Optional[Any]:
        """Query: 'What happened when I took this action in similar states?'"""
        if not self.semantic_retriever or not self.semantic_search_enabled:
            return None
        return self.semantic_retriever.get_counterfactual_experiences(
            state=state, action=action, k=k,
        )

    def get_episodic_memory_statistics(self) -> Optional[dict[str, Any]]:
        """Get comprehensive statistics about episodic memory system."""
        if self.episodic_memory is None:
            return None

        stats: dict[str, Any] = {
            "replay_enabled": self.replay_enabled,
            "consolidation_enabled": self.consolidation_enabled,
            "semantic_search_enabled": self.semantic_search_enabled,
            "generative_memory_enabled": self.generative_memory_enabled,
            "memory_size": len(self.episodic_memory),
            "total_replays": self.total_replays,
            "total_consolidations": self.total_consolidations,
            "replay_frequency": self.replay_frequency,
            "replay_batch_size": self.replay_batch_size,
        }

        capacity = getattr(self.episodic_memory, "capacity", None)
        if capacity:
            stats["memory_capacity"] = capacity
            stats["memory_utilization"] = len(self.episodic_memory) / capacity

        if self.memory_consolidator:
            get_stats = getattr(self.memory_consolidator, "get_consolidation_statistics", None)
            if get_stats:
                stats["consolidation"] = get_stats()

        if self.semantic_retriever:
            n_indexed = getattr(self.semantic_retriever, "n_indexed", None)
            if n_indexed is not None:
                stats["semantic_retrieval"] = {"n_indexed": n_indexed}

        # Triadic recall verification statistics
        if self._recall_verifications > 0:
            stats["recall_verification"] = {
                "total_verifications": self._recall_verifications,
                "trusted": self._recall_trusted,
                "partial": self._recall_partial,
                "untrusted": self._recall_untrusted,
                "trust_rate": self._recall_trusted / max(self._recall_verifications, 1),
            }

        # Reconsolidation statistics (Nader et al. 2000)
        recon_events = getattr(self, "_reconsolidation_events", 0)
        recon_updates = getattr(self, "_reconsolidation_updates", 0)
        recon_stab = getattr(self, "_reconsolidation_stabilizations", 0)
        labile_count = len(getattr(self, "_labile_memories", {}))
        if recon_events > 0 or labile_count > 0:
            stats["reconsolidation"] = {
                "total_events": recon_events,
                "total_updates": recon_updates,
                "total_stabilizations": recon_stab,
                "currently_labile": labile_count,
                "threshold": getattr(self, "_reconsolidation_threshold", 0.3),
                "window_steps": getattr(self, "_reconsolidation_window", 5),
                "alpha": getattr(self, "_reconsolidation_alpha", 0.6),
            }

        # Spreading activation statistics (Collins & Loftus 1975)
        spread_events = getattr(self, "_spreading_activation_events", 0)
        active_count = len(getattr(self, "_spreading_activation", {}))
        if spread_events > 0 or active_count > 0:
            activation_map = getattr(self, "_spreading_activation", {})
            max_act = max(activation_map.values()) if activation_map else 0.0
            mean_act = (
                sum(activation_map.values()) / len(activation_map)
                if activation_map
                else 0.0
            )
            stats["spreading_activation"] = {
                "total_spread_events": spread_events,
                "currently_active": active_count,
                "max_activation": max_act,
                "mean_activation": mean_act,
                "decay_factor": getattr(self, "_spreading_decay_factor", 0.7),
                "step_decay": getattr(self, "_spreading_step_decay", 0.85),
                "max_depth": getattr(self, "_spreading_max_depth", 2),
            }

        return stats

    def _serialize_episodic_memory(self) -> dict:
        return {
            "replay_enabled": getattr(self, "replay_enabled", False),
            "consolidation_enabled": getattr(self, "consolidation_enabled", False),
            "total_replays": getattr(self, "total_replays", 0),
            "total_consolidations": getattr(self, "total_consolidations", 0),
            "steps_since_replay": getattr(self, "steps_since_replay", 0),
            "recall_verifications": getattr(self, "_recall_verifications", 0),
            "recall_trusted": getattr(self, "_recall_trusted", 0),
            "recall_partial": getattr(self, "_recall_partial", 0),
            "recall_untrusted": getattr(self, "_recall_untrusted", 0),
            # Reconsolidation state
            "reconsolidation_events": getattr(self, "_reconsolidation_events", 0),
            "reconsolidation_updates": getattr(self, "_reconsolidation_updates", 0),
            "reconsolidation_stabilizations": getattr(
                self, "_reconsolidation_stabilizations", 0,
            ),
            # Spreading activation state (activation map is transient — not serialized)
            "spreading_activation_events": getattr(
                self, "_spreading_activation_events", 0,
            ),
        }

    def _restore_episodic_memory(self, data: dict) -> None:
        if "total_replays" in data:
            self.total_replays = data["total_replays"]
        if "total_consolidations" in data:
            self.total_consolidations = data["total_consolidations"]
        if "steps_since_replay" in data:
            self.steps_since_replay = data["steps_since_replay"]
        if "recall_verifications" in data:
            self._recall_verifications = data["recall_verifications"]
        if "recall_trusted" in data:
            self._recall_trusted = data["recall_trusted"]
        if "recall_partial" in data:
            self._recall_partial = data["recall_partial"]
        if "recall_untrusted" in data:
            self._recall_untrusted = data["recall_untrusted"]
        # Reconsolidation counters
        if "reconsolidation_events" in data:
            self._reconsolidation_events = data["reconsolidation_events"]
        if "reconsolidation_updates" in data:
            self._reconsolidation_updates = data["reconsolidation_updates"]
        if "reconsolidation_stabilizations" in data:
            self._reconsolidation_stabilizations = data[
                "reconsolidation_stabilizations"
            ]
        # Spreading activation counter (map itself is transient)
        if "spreading_activation_events" in data:
            self._spreading_activation_events = data[
                "spreading_activation_events"
            ]
