"""Episodic Memory Spreading Activation Mixin - Collins & Loftus 1975.

Handles spreading activation: when memories are recalled, activation
propagates to semantically related memories, creating a priming effect.

Extracted from episodic_memory.py to stay under 500-line limit.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EpisodicMemoryActivationMixin:
    """Spreading activation: priming via semantic neighbor activation spread."""

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
