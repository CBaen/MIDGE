"""Circadian rhythm - Mae's internal clock.

Biological analogy: All organisms have circadian rhythms that regulate
activity, rest, and maintenance cycles. In mammals, the suprachiasmatic
nucleus (SCN) coordinates the ~24-hour cycle. During sleep, the brain
consolidates memories, repairs tissue, and prunes neural connections.

Mae's circadian cycle has 3 phases:

| Phase         | What Happens                    | Systems Active           |
|---------------|--------------------------------|--------------------------|
| ACTIVE        | Normal operation                | All systems              |
| CONSOLIDATION | Offline learning, maintenance  | Memory consolidation,    |
|               |                                | generative replay,       |
|               |                                | connection strengthening |
| REST          | Minimal activity               | Health recovery,         |
|               |                                | weight decay, cleanup    |

Unlike real organisms, Mae's cycle is not tied to wall-clock time.
Instead, it's driven by simulation steps. This makes it deterministic
and testable. The cycle length and phase ratios are configurable.

Uses Mesa 3.4 universal time (model.time) for synchronization.

EventBus channel: circadian.phase_change
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Optional

from mae_core.backbone.event_bus import EventBus

logger = logging.getLogger(__name__)

# EventBus channel
CH_PHASE_CHANGE = "circadian.phase_change"


class CircadianPhase(Enum):
    """The three phases of Mae's circadian cycle."""

    ACTIVE = "ACTIVE"
    CONSOLIDATION = "CONSOLIDATION"
    REST = "REST"


class CircadianRhythm:
    """Mae's internal clock - controls activity cycles.

    The rhythm divides simulation steps into repeating cycles of
    ACTIVE -> CONSOLIDATION -> REST. Each phase has a configurable
    duration (in steps) and triggers different system behaviors.

    Integration:
    - Publishes phase changes on EventBus
    - Endocrine system responds with melatonin/cortisol modulation
    - Memory consolidator activates during CONSOLIDATION
    - Substrate modulates flow rates per phase
    - DecisionRouter adjusts tier thresholds per phase
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        cycle_length: int = 100,  # Steps per full cycle
        active_ratio: float = 0.6,  # Fraction of cycle in ACTIVE
        consolidation_ratio: float = 0.25,  # Fraction in CONSOLIDATION
        # REST gets the remainder (0.15)
    ) -> None:
        self.event_bus = event_bus or EventBus()

        # Phase durations (in steps)
        self._cycle_length = cycle_length
        active_steps = int(cycle_length * active_ratio)
        consolidation_steps = int(cycle_length * consolidation_ratio)
        rest_steps = cycle_length - active_steps - consolidation_steps

        self._phase_schedule: list[tuple[CircadianPhase, int]] = [
            (CircadianPhase.ACTIVE, active_steps),
            (CircadianPhase.CONSOLIDATION, consolidation_steps),
            (CircadianPhase.REST, rest_steps),
        ]

        # Current state
        self._current_phase = CircadianPhase.ACTIVE
        self._step_count = 0
        self._phase_step = 0  # Steps within current phase
        self._cycle_count = 0  # Complete cycles

        # Phase transition callbacks
        self._on_phase_change: list[Callable[[CircadianPhase, CircadianPhase], None]] = []

        # Phase statistics
        self._phase_durations: dict[str, int] = {p.value: 0 for p in CircadianPhase}

        logger.info(
            "CircadianRhythm: cycle=%d steps (ACTIVE=%d, CONSOL=%d, REST=%d)",
            cycle_length,
            active_steps,
            consolidation_steps,
            rest_steps,
        )

    @property
    def current_phase(self) -> CircadianPhase:
        return self._current_phase

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def cycle_progress(self) -> float:
        """Progress through current cycle (0.0 to 1.0)."""
        return (self._step_count % self._cycle_length) / self._cycle_length

    @property
    def phase_progress(self) -> float:
        """Progress through current phase (0.0 to 1.0)."""
        current_duration = self._get_phase_duration(self._current_phase)
        if current_duration <= 0:
            return 0.0
        return min(1.0, self._phase_step / current_duration)

    # =========================================================================
    # Step
    # =========================================================================

    def step(self) -> Optional[CircadianPhase]:
        """Advance the clock by one step.

        Returns the new phase if a transition occurred, else None.
        """
        self._step_count += 1
        self._phase_step += 1
        self._phase_durations[self._current_phase.value] += 1

        # Check for phase transition
        current_duration = self._get_phase_duration(self._current_phase)
        if self._phase_step >= current_duration:
            old_phase = self._current_phase
            new_phase = self._next_phase()
            self._transition_to(new_phase, old_phase)
            return new_phase

        return None

    def _transition_to(
        self, new_phase: CircadianPhase, old_phase: CircadianPhase
    ) -> None:
        """Execute a phase transition."""
        self._current_phase = new_phase
        self._phase_step = 0

        # Track cycle completion
        if new_phase == CircadianPhase.ACTIVE and old_phase == CircadianPhase.REST:
            self._cycle_count += 1

        # Publish phase change
        self.event_bus.publish(
            CH_PHASE_CHANGE,
            {
                "old_phase": old_phase.value,
                "new_phase": new_phase.value,
                "cycle": self._cycle_count,
                "step": self._step_count,
            },
        )

        # Notify callbacks
        for callback in self._on_phase_change:
            try:
                callback(old_phase, new_phase)
            except Exception:
                logger.exception("Error in phase_change callback")

        logger.info(
            "Circadian: %s -> %s (cycle %d, step %d)",
            old_phase.value,
            new_phase.value,
            self._cycle_count,
            self._step_count,
        )

    def _next_phase(self) -> CircadianPhase:
        """Get the next phase in the cycle."""
        phases = [p for p, _ in self._phase_schedule]
        idx = phases.index(self._current_phase)
        return phases[(idx + 1) % len(phases)]

    def _get_phase_duration(self, phase: CircadianPhase) -> int:
        """Get the duration of a phase in steps."""
        for p, duration in self._phase_schedule:
            if p == phase:
                return duration
        return 0

    # =========================================================================
    # Query Methods (for other systems)
    # =========================================================================

    def is_active(self) -> bool:
        """Is Mae in ACTIVE phase?"""
        return self._current_phase == CircadianPhase.ACTIVE

    def is_consolidating(self) -> bool:
        """Is Mae in CONSOLIDATION phase?"""
        return self._current_phase == CircadianPhase.CONSOLIDATION

    def is_resting(self) -> bool:
        """Is Mae in REST phase?"""
        return self._current_phase == CircadianPhase.REST

    def should_consolidate_memory(self) -> bool:
        """Should memory consolidation happen now?"""
        return self._current_phase == CircadianPhase.CONSOLIDATION

    def should_learn(self) -> bool:
        """Should active learning happen now?"""
        return self._current_phase in (CircadianPhase.ACTIVE, CircadianPhase.CONSOLIDATION)

    def get_activity_multiplier(self) -> float:
        """Activity level multiplier for the current phase.

        ACTIVE: 1.0 (full activity)
        CONSOLIDATION: 0.5 (reduced activity, focus on learning)
        REST: 0.1 (minimal activity)
        """
        multipliers = {
            CircadianPhase.ACTIVE: 1.0,
            CircadianPhase.CONSOLIDATION: 0.5,
            CircadianPhase.REST: 0.1,
        }
        return multipliers.get(self._current_phase, 1.0)

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def on_phase_change(
        self, callback: Callable[[CircadianPhase, CircadianPhase], None]
    ) -> None:
        """Register callback for phase transitions.

        Callback receives (old_phase, new_phase).
        """
        self._on_phase_change.append(callback)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        return {
            "current_phase": self._current_phase.value,
            "phase_progress": self.phase_progress,
            "cycle_progress": self.cycle_progress,
            "cycle_count": self._cycle_count,
            "step_count": self._step_count,
            "cycle_length": self._cycle_length,
            "activity_multiplier": self.get_activity_multiplier(),
            "phase_durations": dict(self._phase_durations),
            "schedule": [
                {"phase": p.value, "steps": d} for p, d in self._phase_schedule
            ],
        }
