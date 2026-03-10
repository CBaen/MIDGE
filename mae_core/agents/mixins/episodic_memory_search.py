"""Episodic Memory Search Mixin - tick, search, counterfactual, verify_recall.

Handles the main recall interface: semantic search with triadic verification,
counterfactual queries, and the per-step tick that advances reconsolidation
and spreading activation.

Extracted from episodic_memory.py to stay under 500-line limit.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EpisodicMemorySearchMixin:
    """Memory search: verified recall, counterfactuals, and step tick."""

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
                    semantic = getattr(self, "semantic_retriever", None)
                    if semantic is not None and getattr(self, "semantic_search_enabled", False):
                        neighbors = semantic.search_by_state(
                            np.asarray(mem_state, dtype=np.float32), k=5,
                        )
                        neighbor_exps = getattr(neighbors, "experiences", []) if neighbors else []
                        temporal_neighbors = []
                        for n in neighbor_exps:
                            n_ts = getattr(n, "timestamp", None)
                            if n_ts is not None and abs(n_ts - mem_timestamp) <= 2.0:
                                if n is not memory:
                                    temporal_neighbors.append(n)

                        if temporal_neighbors:
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
                reward_history = getattr(self, "_reward_history", [])
                reward_sum = getattr(self, "_reward_sum", 0.0)
                reward_sq_sum = getattr(self, "_reward_sq_sum", 0.0)
                if mem_reward is not None and len(reward_history) >= 3:
                    import numpy as _np
                    n = len(reward_history)
                    mean = reward_sum / n
                    variance = (reward_sq_sum / n) - (mean * mean)
                    std = max(_np.sqrt(max(variance, 0.0)), 1e-8)
                    z_score = abs(mem_reward - mean) / std
                    if z_score <= 2.0:
                        corroborations += 1
                elif mem_reward is not None:
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
