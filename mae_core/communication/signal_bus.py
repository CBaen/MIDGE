"""Signal Bus - Electrical signal communication between agents.

Wraps EventBus with signal-specific semantics: typed signals,
priority levels, TTL, and subscription filtering.

Biological analogy: Nervous system electrical impulses.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..backbone.event_bus import EventBus

logger = logging.getLogger(__name__)

SIGNAL_CHANNEL_PREFIX = "signal."


@dataclass
class Signal:
    """An electrical signal transmitted between agents."""

    signal_type: str
    payload: dict[str, Any]
    sender_id: str = ""
    priority: float = 0.5
    ttl: float = 0.0  # seconds, 0 = no expiry
    timestamp: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        if self.ttl <= 0:
            return False
        return (time.time() - self.timestamp) > self.ttl


class SignalBus:
    """Signal-specific communication layer on top of EventBus.

    Provides the interface expected by SignalProcessingMixin:
    emit_signal(), subscribe(), unsubscribe(), get_metrics().
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus
        self._subscriptions: dict[str, list[tuple[Callable, float | None]]] = defaultdict(list)
        self._signal_history: deque[Signal] = deque(maxlen=1000)

        # Metrics
        self._emitted = 0
        self._delivered = 0
        self._filtered = 0
        self._expired = 0

    def emit_signal(
        self,
        signal_type: str,
        payload: dict[str, Any],
        sender_id: str = "",
        priority: float = 0.5,
        ttl: float = 0.0,
    ) -> bool:
        """Emit a signal to all subscribers. Returns True on success."""
        signal = Signal(
            signal_type=signal_type,
            payload=payload,
            sender_id=sender_id,
            priority=priority,
            ttl=ttl,
        )

        self._emitted += 1
        self._signal_history.append(signal)

        # Deliver to direct subscribers (with priority filtering)
        for callback, min_priority in self._subscriptions.get(signal_type, []):
            if min_priority is not None and priority < min_priority:
                self._filtered += 1
                continue
            try:
                callback(signal)
                self._delivered += 1
            except Exception:
                logger.exception("Error in signal handler for %s", signal_type)

        # Also publish on EventBus for cross-system listeners
        channel = f"{SIGNAL_CHANNEL_PREFIX}{signal_type}"
        self._bus.publish(channel, {
            "signal_type": signal_type,
            "payload": payload,
            "sender_id": sender_id,
            "priority": priority,
            "timestamp": signal.timestamp,
        })

        return True

    # Alias for compatibility
    emit = emit_signal

    def subscribe(
        self,
        signal_type: str,
        callback: Callable[[Signal], None],
        min_priority: float | None = None,
    ) -> bool:
        """Subscribe to a signal type with optional priority filtering."""
        self._subscriptions[signal_type].append((callback, min_priority))
        return True

    def unsubscribe(self, signal_type: str, callback: Callable | None = None) -> bool:
        """Unsubscribe from a signal type. If callback is None, remove all."""
        if signal_type not in self._subscriptions:
            return False
        if callback is None:
            del self._subscriptions[signal_type]
        else:
            self._subscriptions[signal_type] = [
                (cb, mp) for cb, mp in self._subscriptions[signal_type] if cb != callback
            ]
        return True

    def get_metrics(self) -> dict[str, Any]:
        """Return signal bus statistics."""
        return {
            "emitted": self._emitted,
            "delivered": self._delivered,
            "filtered": self._filtered,
            "expired": self._expired,
            "subscriptions": {
                st: len(cbs) for st, cbs in self._subscriptions.items()
            },
            "history_size": len(self._signal_history),
        }

    def get_recent_signals(
        self, signal_type: str | None = None, limit: int = 10
    ) -> list[Signal]:
        """Get recent signals, optionally filtered by type."""
        signals = list(self._signal_history)
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        return signals[-limit:]

    def __repr__(self) -> str:
        return (
            f"SignalBus(emitted={self._emitted}, "
            f"subscriptions={sum(len(v) for v in self._subscriptions.values())})"
        )
