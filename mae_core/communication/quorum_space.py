"""Quorum Space - Chemical concentration environment for quorum sensing.

Manages signal concentrations that accumulate from agent deposits
and decay over time. When concentration crosses threshold, quorum
is achieved.

Biological analogy: Bacterial autoinducer accumulation.
Based on: Miller & Bassler (2001), Waters & Bassler (2005).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

from .quorum_signal import DecayType, QuorumSignalType, SenseEvent, get_signal_type

logger = logging.getLogger(__name__)


@dataclass
class ConcentrationState:
    """State of a signal concentration in the quorum space."""

    concentration: float = 0.0
    last_update: float = field(default_factory=time.time)
    contributors: set[str] = field(default_factory=set)
    deposit_count: int = 0
    total_strength: float = 0.0


class QuorumSpace:
    """Chemical concentration environment for quorum sensing.

    Thread-safe. Supports background decay if needed.
    """

    def __init__(
        self,
        max_history: int = 1000,
    ) -> None:
        self._signal_types: dict[str, QuorumSignalType] = {}
        self._states: dict[str, ConcentrationState] = defaultdict(ConcentrationState)
        self._history: deque[SenseEvent] = deque(maxlen=max_history)
        self._lock = threading.RLock()

        # Register default signal types
        from .quorum_signal import SIGNAL_TYPE_REGISTRY
        for name, sig_type in SIGNAL_TYPE_REGISTRY.items():
            self._signal_types[name] = sig_type

    def register_signal_type(self, signal_type: QuorumSignalType) -> None:
        with self._lock:
            self._signal_types[signal_type.name] = signal_type

    def deposit_signal(
        self,
        signal_type: str,
        agent_id: str,
        strength: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Deposit a signal, increasing concentration. Returns True on success."""
        sig_def = self._signal_types.get(signal_type)
        if sig_def is None:
            # Auto-register unknown signal types
            sig_def = QuorumSignalType(name=signal_type)
            self._signal_types[signal_type] = sig_def

        actual_strength = strength if strength is not None else sig_def.base_strength

        with self._lock:
            state = self._states[signal_type]

            # Apply pending decay first
            self._apply_decay(signal_type)

            # Add to concentration (capped at max)
            state.concentration = min(
                sig_def.max_concentration,
                state.concentration + actual_strength,
            )
            state.contributors.add(agent_id)
            state.deposit_count += 1
            state.total_strength += actual_strength
            state.last_update = time.time()

            # Record event
            self._history.append(SenseEvent(
                event_type="deposit",
                signal_type=signal_type,
                agent_id=agent_id,
                concentration=state.concentration,
                metadata=metadata or {},
            ))

        return True

    # Alias
    emit_signal = deposit_signal

    def get_concentration(self, signal_type: str) -> float:
        """Get current decayed concentration of a signal type."""
        with self._lock:
            if signal_type not in self._states:
                return 0.0
            self._apply_decay(signal_type)
            return self._states[signal_type].concentration

    def get_contributor_count(self, signal_type: str) -> int:
        with self._lock:
            return len(self._states[signal_type].contributors) if signal_type in self._states else 0

    def get_contributors(self, signal_type: str) -> set[str]:
        with self._lock:
            state = self._states.get(signal_type)
            return set(state.contributors) if state else set()

    def get_state(self, signal_type: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._states.get(signal_type)
            if state is None:
                return None
            self._apply_decay(signal_type)
            return {
                "concentration": state.concentration,
                "contributors": len(state.contributors),
                "deposit_count": state.deposit_count,
                "total_strength": state.total_strength,
            }

    def get_all_concentrations(self) -> dict[str, float]:
        with self._lock:
            result = {}
            for sig_type in list(self._states.keys()):
                self._apply_decay(sig_type)
                result[sig_type] = self._states[sig_type].concentration
            return result

    # Alias
    get_all_signals = get_all_concentrations

    def reset_signal(self, signal_type: str, clear_contributors: bool = True) -> None:
        with self._lock:
            if signal_type in self._states:
                self._states[signal_type].concentration = 0.0
                if clear_contributors:
                    self._states[signal_type].contributors.clear()

    def reset_all(self, clear_contributors: bool = True) -> None:
        with self._lock:
            for sig_type in self._states:
                self._states[sig_type].concentration = 0.0
                if clear_contributors:
                    self._states[sig_type].contributors.clear()

    def get_event_history(
        self,
        signal_type: str | None = None,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> list[SenseEvent]:
        events = list(self._history)
        if signal_type:
            events = [e for e in events if e.signal_type == signal_type]
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        return events[-limit:]

    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "signal_types": len(self._signal_types),
                "active_signals": sum(
                    1 for s in self._states.values() if s.concentration > 0.01
                ),
                "total_deposits": sum(s.deposit_count for s in self._states.values()),
                "history_size": len(self._history),
            }

    def _apply_decay(self, signal_type: str) -> None:
        """Apply time-based decay to a signal's concentration. Must hold lock."""
        state = self._states.get(signal_type)
        if state is None:
            return

        sig_def = self._signal_types.get(signal_type)
        if sig_def is None or sig_def.decay_type == DecayType.NONE:
            return

        elapsed = time.time() - state.last_update
        if elapsed <= 0:
            return

        state.concentration = sig_def.compute_decay(state.concentration, elapsed)
        state.last_update = time.time()

        # Clear contributors when concentration drops to near-zero
        if state.concentration < 0.001:
            state.contributors.clear()

    def __repr__(self) -> str:
        return (
            f"QuorumSpace(signals={len(self._signal_types)}, "
            f"active={sum(1 for s in self._states.values() if s.concentration > 0.01)})"
        )
