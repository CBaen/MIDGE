"""Signal Priority Protocol - Thalamus-like signal triage for agents.

Queues signals between steps, sorts by priority, coalesces same-type
duplicates, enforces a per-step budget, and maps signals to decision
tiers (Reflex/Habit/Prefrontal).

Biological analogy: Thalamic relay — prioritizes sensory input before
it reaches the cortex.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..cognition.decision_router import DecisionTier
from .signal_bus import Signal, SignalBus

logger = logging.getLogger(__name__)

# Default urgency scores by signal type (biological priority)
DEFAULT_URGENCY_MAP: dict[str, float] = {
    "DANGER": 1.0,
    "COLLABORATION_REQUEST": 0.7,
    "CONVERGENCE": 0.6,
    "OPPORTUNITY": 0.5,
    "KNOWLEDGE_SHARE": 0.3,
}


@dataclass
class PriorityConfig:
    """Tunable weights for priority computation."""

    priority_weight: float = 0.5
    urgency_weight: float = 0.3
    recency_weight: float = 0.2
    budget_per_step: int = 10
    defer_overflow: bool = True
    defer_max_age_steps: int = 3
    enable_preemption: bool = True
    coalesce_same_type: bool = True
    preemption_threshold: float = 0.9
    reflex_threshold: float = 0.8
    habit_threshold: float = 0.5


@dataclass
class PrioritizedSignal:
    """A signal with computed priority score and tier assignment."""

    signal: Signal
    computed_score: float = 0.0
    decision_tier: DecisionTier = DecisionTier.NONE
    coalesced_count: int = 1
    coalesced_senders: list[str] = field(default_factory=list)
    coalesced_payloads: list[dict[str, Any]] = field(default_factory=list)
    enqueue_step: int = 0


@dataclass
class StepReport:
    """Outcome of one step's signal processing."""

    step_number: int = 0
    processed: list[PrioritizedSignal] = field(default_factory=list)
    deferred: list[PrioritizedSignal] = field(default_factory=list)
    dropped: list[PrioritizedSignal] = field(default_factory=list)
    preempted: list[PrioritizedSignal] = field(default_factory=list)
    coalesced_count: int = 0
    total_received: int = 0


class SignalPriorityResolver:
    """Per-agent signal triage — queues, sorts, coalesces, budgets.

    Sits between SignalBus and agent signal handlers. Instead of
    immediate dispatch, signals accumulate in an inbox and are
    processed in priority order at the start of each step.

    Args:
        agent_id: Owning agent's ID.
        signal_bus: The shared SignalBus to subscribe to.
        config: Priority tuning parameters.
        urgency_map: Signal type → base urgency score (0–1).
    """

    def __init__(
        self,
        agent_id: str,
        signal_bus: SignalBus,
        config: PriorityConfig | None = None,
        urgency_map: dict[str, float] | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._bus = signal_bus
        self._config = config or PriorityConfig()
        self._urgency_map = dict(DEFAULT_URGENCY_MAP)
        if urgency_map:
            self._urgency_map.update(urgency_map)

        # Optional triage classifier — injected post-construction when
        # biological systems (nociception, threat, endocrine) are available.
        self._triage_classifier: Any = None

        # Signal type → handler callback
        self._handlers: dict[str, Callable] = {}

        # Inbox: signals waiting to be processed
        self._inbox: list[Signal] = []

        # Deferred from previous steps (overflow)
        self._deferred: list[PrioritizedSignal] = []

        # Step counter (incremented each process() call)
        self._current_step: int = 0

        # Last report
        self._last_report: StepReport | None = None

        # Cumulative statistics
        self._total_received: int = 0
        self._total_processed: int = 0
        self._total_deferred: int = 0
        self._total_dropped: int = 0
        self._total_preempted: int = 0
        self._total_coalesced: int = 0

    # -- Registration --

    def register_handler(
        self,
        signal_type: str,
        callback: Callable[[Signal], None],
    ) -> None:
        """Register a handler and subscribe to the SignalBus.

        The handler will NOT be called immediately on signal arrival.
        Instead, signals are queued and dispatched during process().
        """
        self._handlers[signal_type] = callback
        self._bus.subscribe(signal_type, self._enqueue)

    # -- Enqueue (called by SignalBus on signal emission) --

    def _enqueue(self, signal: Signal) -> None:
        """Buffer an incoming signal for prioritized processing."""
        # Skip expired signals
        if signal.is_expired:
            return

        self._total_received += 1

        # Preemption: CRITICAL signals dispatch immediately
        if (
            self._config.enable_preemption
            and signal.priority >= self._config.preemption_threshold
        ):
            self._preempt(signal)
            return

        self._inbox.append(signal)

    def _preempt(self, signal: Signal) -> None:
        """Immediately dispatch a CRITICAL signal, bypassing the queue."""
        handler = self._handlers.get(signal.signal_type)
        if handler is not None:
            try:
                handler(signal)
            except Exception:
                logger.exception(
                    "Error in preempted handler for %s", signal.signal_type
                )
        self._total_preempted += 1

    # -- Process (called at start of each agent step) --

    def process(self) -> StepReport:
        """Sort, coalesce, budget, and dispatch queued signals.

        Called at the start of each agent step. Returns a StepReport
        describing what was processed, deferred, and dropped.
        """
        self._current_step += 1
        cfg = self._config
        report = StepReport(
            step_number=self._current_step,
            total_received=len(self._inbox),
        )

        # 1. Merge deferred signals (with age check)
        still_valid: list[PrioritizedSignal] = []
        for ps in self._deferred:
            age = self._current_step - ps.enqueue_step
            if age <= cfg.defer_max_age_steps:
                still_valid.append(ps)
            else:
                report.dropped.append(ps)
                self._total_dropped += 1

        # 2. Convert inbox signals to PrioritizedSignal
        new_signals: list[PrioritizedSignal] = []
        for signal in self._inbox:
            if signal.is_expired:
                continue
            score = self._compute_score(signal)
            tier = self._score_to_tier(score)
            ps = PrioritizedSignal(
                signal=signal,
                computed_score=score,
                decision_tier=tier,
                coalesced_senders=[signal.sender_id] if signal.sender_id else [],
                coalesced_payloads=[signal.payload],
                enqueue_step=self._current_step,
            )
            new_signals.append(ps)
        self._inbox.clear()

        # 3. Coalesce same-type signals
        if cfg.coalesce_same_type:
            new_signals, coalesced = self._coalesce(new_signals)
            report.coalesced_count = coalesced
            self._total_coalesced += coalesced

        # 4. Combine with deferred and sort by score (descending)
        all_signals = still_valid + new_signals
        all_signals.sort(key=lambda ps: ps.computed_score, reverse=True)

        # 5. Budget: take top N
        processed = all_signals[: cfg.budget_per_step]
        overflow = all_signals[cfg.budget_per_step :]

        # 6. Dispatch processed signals to handlers
        for ps in processed:
            handler = self._handlers.get(ps.signal.signal_type)
            if handler is not None:
                try:
                    handler(ps.signal)
                except Exception:
                    logger.exception(
                        "Error in handler for %s", ps.signal.signal_type
                    )

        # 7. Handle overflow
        if cfg.defer_overflow:
            self._deferred = overflow
            report.deferred = list(overflow)
            self._total_deferred += len(overflow)
        else:
            report.dropped.extend(overflow)
            self._total_dropped += len(overflow)
            self._deferred = []

        report.processed = processed
        self._total_processed += len(processed)
        self._last_report = report
        return report

    # -- Scoring --

    def _compute_score(self, signal: Signal) -> float:
        """Compute composite priority score for a signal.

        Base score = priority*0.5 + urgency*0.3 + recency*0.2.
        If a TriageClassifier is available, adds biological urgency boost
        based on current pain/threat/stress state.
        """
        cfg = self._config
        urgency = self._urgency_map.get(signal.signal_type, 0.5)
        recency = self._compute_recency(signal.timestamp)
        base_score = (
            cfg.priority_weight * signal.priority
            + cfg.urgency_weight * urgency
            + cfg.recency_weight * recency
        )

        # Biological triage boost (if classifier available)
        if self._triage_classifier is not None:
            try:
                result = self._triage_classifier.classify(
                    signal.signal_type, signal.priority,
                )
                base_score += result.urgency_boost
            except Exception:
                pass  # Graceful degradation

        return max(0.0, min(1.0, base_score))

    @staticmethod
    def _compute_recency(timestamp: float) -> float:
        """Recency score: 1.0 for now, decaying exponentially."""
        age = max(0.0, time.time() - timestamp)
        return math.exp(-age / 60.0)  # 60s half-life

    def _score_to_tier(self, score: float) -> DecisionTier:
        """Map score to decision tier."""
        if score >= self._config.reflex_threshold:
            return DecisionTier.REFLEX
        if score >= self._config.habit_threshold:
            return DecisionTier.HABIT
        return DecisionTier.PREFRONTAL

    # -- Coalescing --

    @staticmethod
    def _coalesce(
        signals: list[PrioritizedSignal],
    ) -> tuple[list[PrioritizedSignal], int]:
        """Merge same-type signals within a step.

        Returns (coalesced_list, num_raw_signals_merged).
        """
        if not signals:
            return signals, 0

        groups: dict[str, list[PrioritizedSignal]] = {}
        for ps in signals:
            groups.setdefault(ps.signal.signal_type, []).append(ps)

        coalesced: list[PrioritizedSignal] = []
        total_merged = 0

        for signal_type, group in groups.items():
            if len(group) == 1:
                coalesced.append(group[0])
                continue

            # Merge: take highest-priority signal, boost by log(count)
            group.sort(key=lambda ps: ps.computed_score, reverse=True)
            best = group[0]
            count = len(group)
            boost = 0.1 * math.log(count)
            best.computed_score = min(1.0, best.computed_score + boost)
            best.decision_tier = SignalPriorityResolver._score_to_tier_static(
                best.computed_score,
            )
            best.coalesced_count = count
            best.coalesced_senders = [
                s for ps in group for s in ps.coalesced_senders
            ]
            best.coalesced_payloads = [
                p for ps in group for p in ps.coalesced_payloads
            ]
            coalesced.append(best)
            total_merged += count - 1  # N signals → 1 = (N-1) merged

        return coalesced, total_merged

    @staticmethod
    def _score_to_tier_static(score: float) -> DecisionTier:
        if score >= 0.8:
            return DecisionTier.REFLEX
        if score >= 0.5:
            return DecisionTier.HABIT
        return DecisionTier.PREFRONTAL

    # -- Statistics --

    def get_statistics(self) -> dict[str, Any]:
        return {
            "agent_id": self._agent_id,
            "current_step": self._current_step,
            "inbox_size": len(self._inbox),
            "deferred_size": len(self._deferred),
            "total_received": self._total_received,
            "total_processed": self._total_processed,
            "total_deferred": self._total_deferred,
            "total_dropped": self._total_dropped,
            "total_preempted": self._total_preempted,
            "total_coalesced": self._total_coalesced,
            "handler_count": len(self._handlers),
            "budget_per_step": self._config.budget_per_step,
        }

    def get_last_report(self) -> StepReport | None:
        return self._last_report

    # -- Persistence --

    def serialize(self, persist_dir: "Path") -> dict:
        """Return JSON-safe metadata (deferred signals not persisted)."""
        return {
            "current_step": self._current_step,
            "total_received": self._total_received,
            "total_processed": self._total_processed,
            "total_deferred": self._total_deferred,
            "total_dropped": self._total_dropped,
            "total_preempted": self._total_preempted,
            "total_coalesced": self._total_coalesced,
        }

    def restore(self, persist_dir: "Path", metadata: dict) -> None:
        """Restore cumulative statistics from metadata."""
        self._current_step = metadata.get("current_step", 0)
        self._total_received = metadata.get("total_received", 0)
        self._total_processed = metadata.get("total_processed", 0)
        self._total_deferred = metadata.get("total_deferred", 0)
        self._total_dropped = metadata.get("total_dropped", 0)
        self._total_preempted = metadata.get("total_preempted", 0)
        self._total_coalesced = metadata.get("total_coalesced", 0)

    def __repr__(self) -> str:
        return (
            f"SignalPriorityResolver(agent={self._agent_id}, "
            f"inbox={len(self._inbox)}, deferred={len(self._deferred)}, "
            f"step={self._current_step})"
        )
