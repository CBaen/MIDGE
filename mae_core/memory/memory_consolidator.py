"""Memory Consolidator - Offline "sleep" learning via experience replay.

Schedules periodic consolidation phases where the agent replays
important experiences to strengthen learning without environment
interaction. Supports multiple strategies.

Biological analogy: Hippocampal replay during sleep.
Based on: Stickgold (2005), Wilson & McNaughton (1994)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ConsolidationStrategy(Enum):
    """How to sample experiences during consolidation."""

    UNIFORM = "uniform"
    PRIORITIZED = "prioritized"
    RECENT = "recent"
    ADAPTIVE = "adaptive"
    MIXED = "mixed"


@dataclass
class ConsolidationResult:
    """Outcome of a consolidation phase."""

    consolidation_id: int
    strategy: str
    steps: int
    elapsed: float
    mean_loss: float
    initial_loss: float
    final_loss: float
    loss_reduction: float
    timestamp: float = field(default_factory=time.time)


class MemoryConsolidator:
    """Schedules and executes memory consolidation ("sleep") phases.

    Coordinates with EpisodicMemory to replay high-priority experiences
    and update the agent's policy.

    Args:
        episodic_memory: The EpisodicMemory to consolidate from.
        consolidation_steps: Replay steps per consolidation.
        consolidation_frequency: Trigger every N env steps.
        batch_size: Replay batch size.
        strategy: Default sampling strategy.
        min_memory_size: Minimum experiences before consolidating.
        lr_multiplier: Scale agent's LR during consolidation.
    """

    def __init__(
        self,
        episodic_memory: Any,
        consolidation_steps: int = 100,
        consolidation_frequency: int = 1000,
        batch_size: int = 32,
        strategy: ConsolidationStrategy = ConsolidationStrategy.PRIORITIZED,
        min_memory_size: int = 1000,
        lr_multiplier: float = 0.5,
    ) -> None:
        self._memory = episodic_memory
        self._steps = consolidation_steps
        self._frequency = consolidation_frequency
        self._batch_size = batch_size
        self._strategy = strategy
        self._min_size = min_memory_size
        self._lr_mult = lr_multiplier

        self._steps_since = 0
        self._total_consolidations = 0
        self._history: list[ConsolidationResult] = []

    # -- public API --

    def should_consolidate(self, current_step: int | None = None) -> bool:
        """Check whether it's time to consolidate."""
        if len(self._memory) < self._min_size:
            return False
        if current_step is not None:
            return current_step % self._frequency == 0
        return self._steps_since >= self._frequency

    def step(self, n: int = 1) -> None:
        """Advance the internal step counter."""
        self._steps_since += n

    def consolidate(
        self,
        agent: Any,
        num_steps: int | None = None,
        strategy: ConsolidationStrategy | None = None,
    ) -> ConsolidationResult:
        """Run a consolidation phase.

        Args:
            agent: Must have learn_from_batch(batch, weights) -> (td_errors, loss),
                   get_learning_rate() -> float, set_learning_rate(lr).
            num_steps: Override default step count.
            strategy: Override default strategy.

        Returns:
            ConsolidationResult with statistics.
        """
        if len(self._memory) < self._min_size:
            raise ValueError(
                f"Memory too small: {len(self._memory)} < {self._min_size}"
            )

        num_steps = num_steps or self._steps
        strategy = strategy or self._strategy

        # Lower learning rate during consolidation
        orig_lr = agent.get_learning_rate()
        agent.set_learning_rate(orig_lr * self._lr_mult)

        losses: list[float] = []
        t0 = time.time()

        for _ in range(num_steps):
            batch, indices, weights = self._sample(strategy)
            td_errors, loss = agent.learn_from_batch(batch, weights)
            self._memory.update_priorities(indices, td_errors)
            losses.append(float(loss))

        agent.set_learning_rate(orig_lr)
        elapsed = time.time() - t0

        result = ConsolidationResult(
            consolidation_id=self._total_consolidations,
            strategy=strategy.value,
            steps=num_steps,
            elapsed=elapsed,
            mean_loss=float(np.mean(losses)),
            initial_loss=losses[0],
            final_loss=losses[-1],
            loss_reduction=(losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0.0,
        )

        self._history.append(result)
        self._total_consolidations += 1
        self._steps_since = 0
        return result

    @property
    def total_consolidations(self) -> int:
        return self._total_consolidations

    @property
    def history(self) -> list[ConsolidationResult]:
        return list(self._history)

    # -- internals --

    def _sample(
        self, strategy: ConsolidationStrategy
    ) -> tuple[list, list[int], np.ndarray]:
        """Sample a batch according to strategy."""
        bs = min(self._batch_size, len(self._memory))

        if strategy == ConsolidationStrategy.UNIFORM:
            batch, indices, _ = self._memory.sample(bs)
            return batch, indices, np.ones(len(batch))

        if strategy == ConsolidationStrategy.PRIORITIZED:
            return self._memory.sample(bs)

        if strategy == ConsolidationStrategy.MIXED:
            half = bs // 2
            b1, i1, w1 = self._memory.sample(half)
            b2, i2, _ = self._memory.sample(bs - half)
            return b1 + b2, i1 + i2, np.concatenate([w1, np.ones(len(b2))])

        # RECENT, ADAPTIVE - fall back to prioritized
        return self._memory.sample(bs)

    # -- persistence --

    def serialize(self, persist_dir: "Path") -> dict:
        """Return JSON-safe metadata (all state is primitives, no file needed)."""
        return {
            "steps_since": self._steps_since,
            "total_consolidations": self._total_consolidations,
            "history_len": len(self._history),
        }

    def restore(self, persist_dir: "Path", metadata: dict) -> None:
        """Restore consolidator state from metadata."""
        self._steps_since = metadata.get("steps_since", 0)
        self._total_consolidations = metadata.get("total_consolidations", 0)
        logger.info(
            "Restored consolidator (consolidations=%d)",
            self._total_consolidations,
        )

    def __repr__(self) -> str:
        return (
            f"MemoryConsolidator(consolidations={self._total_consolidations}, "
            f"strategy={self._strategy.value}, "
            f"frequency={self._frequency})"
        )
