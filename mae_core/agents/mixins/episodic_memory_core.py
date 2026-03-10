"""Episodic Memory Core Mixin - init, store, consolidate, learn, serialize.

Contains the primary experience storage, memory consolidation, batch learning,
generative replay, serialization, and statistics. Extracted from episodic_memory.py
to stay under 500-line limit.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EpisodicMemoryCoreMixin:
    """Core episodic memory: storage, consolidation, replay, serialize/restore."""

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
        self._labile_memories: dict[int, dict[str, Any]] = {}
        self._reconsolidation_threshold: float = config.get(
            "reconsolidation_threshold", 0.3,
        )
        self._reconsolidation_window: int = config.get(
            "reconsolidation_window", 5,
        )
        self._reconsolidation_alpha: float = config.get(
            "reconsolidation_alpha", 0.6,
        )
        self._reconsolidation_events: int = 0
        self._reconsolidation_updates: int = 0
        self._reconsolidation_stabilizations: int = 0

        # --- Spreading activation state (Collins & Loftus 1975) ---
        self._spreading_activation: dict[int, float] = {}
        self._spreading_decay_factor: float = config.get(
            "spreading_decay_factor", 0.7,
        )
        self._spreading_max_depth: int = config.get(
            "spreading_max_depth", 2,
        )
        self._spreading_step_decay: float = config.get(
            "spreading_step_decay", 0.85,
        )
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
        """Store experience in episodic memory for replay learning."""
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
        """Learn from experiences stored in episodic memory."""
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
        """Learn from a batch of experiences. Subclasses should override."""
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
