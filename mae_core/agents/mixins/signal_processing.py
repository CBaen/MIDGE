"""Signal Processing Mixin - Electrical signaling.

Provides ultra-fast (sub-millisecond) communication via the
ElectricalSignalBus for critical events like dangers, opportunities,
and convergence notifications.

Ported from v5-pivot base_agent.py Big Rock 5 signal methods.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SignalProcessingMixin:
    """Adds electrical signal emission and subscription to agents."""

    def _init_signal_processing(self, signal_bus: Any = None) -> None:
        """Initialize signal processing attributes."""
        self.signal_bus = signal_bus
        self.signal_subscriptions: deque[str] = deque(maxlen=50)
        self._signal_callbacks: dict[str, Callable[..., None]] = {}

    def emit_signal(
        self,
        signal_type: str,
        payload: dict[str, Any],
        priority: Any = None,
        ttl: float = 0.0,
    ) -> bool:
        """Emit an electrical signal to the mycelial network."""
        if self.signal_bus is None:
            return False

        kwargs: dict[str, Any] = {
            "signal_type": signal_type,
            "payload": payload,
            "sender_id": str(getattr(self, "unique_id", "unknown")),
        }
        if priority is not None:
            kwargs["priority"] = priority
        if ttl > 0:
            kwargs["ttl"] = ttl

        # Support both .emit_signal() and .emit() (fixes known bug)
        emit_fn = getattr(self.signal_bus, "emit_signal", None)
        if emit_fn is None:
            emit_fn = getattr(self.signal_bus, "emit", None)

        if emit_fn is None:
            logger.warning("Signal bus has neither emit_signal nor emit method")
            return False

        try:
            return bool(emit_fn(**kwargs))
        except Exception:
            logger.exception("Error emitting signal %s", signal_type)
            return False

    def subscribe_to_signal(
        self,
        signal_type: str,
        callback: Callable[..., None],
        min_priority: Any = None,
    ) -> bool:
        """Subscribe to signals of a specific type."""
        if self.signal_bus is None:
            return False

        subscribe_fn = getattr(self.signal_bus, "subscribe", None)
        if subscribe_fn is None:
            return False

        kwargs: dict[str, Any] = {
            "signal_type": signal_type,
            "callback": callback,
        }
        if min_priority is not None:
            kwargs["min_priority"] = min_priority

        try:
            success = bool(subscribe_fn(**kwargs))
            if success:
                self.signal_subscriptions.append(signal_type)
                self._signal_callbacks[signal_type] = callback
            return success
        except Exception:
            logger.exception("Error subscribing to %s", signal_type)
            return False

    def unsubscribe_from_signal(self, signal_type: str) -> bool:
        """Unsubscribe from signals of a specific type."""
        if self.signal_bus is None:
            return False

        unsubscribe_fn = getattr(self.signal_bus, "unsubscribe", None)
        if unsubscribe_fn is None:
            return False

        try:
            callback = self._signal_callbacks.get(signal_type)
            success = bool(unsubscribe_fn(signal_type, callback))
            if success:
                if signal_type in self.signal_subscriptions:
                    self.signal_subscriptions.remove(signal_type)
                self._signal_callbacks.pop(signal_type, None)
            return success
        except Exception:
            return False

    def setup_standard_signal_handlers(self) -> None:
        """Setup default handlers for common signal types."""
        if self.signal_bus is None:
            return

        handlers = {
            "DANGER": self._handle_danger_signal,
            "OPPORTUNITY": self._handle_opportunity_signal,
            "CONVERGENCE": self._handle_convergence_signal,
            "COLLABORATION_REQUEST": self._handle_collaboration_signal,
            "KNOWLEDGE_SHARE": self._handle_knowledge_share_signal,
        }

        for signal_type, handler in handlers.items():
            self.subscribe_to_signal(signal_type, handler)

    def _handle_danger_signal(self, signal: Any) -> None:
        """Handle DANGER signal - increase caution."""
        payload = getattr(signal, "payload", {}) if not isinstance(signal, dict) else signal
        risk_level = payload.get("risk_level", 0.5)
        if risk_level > 0.7:
            current_risk = getattr(self, "risk_score", 0.0)
            self.risk_score = min(1.0, current_risk + 0.2)  # type: ignore[attr-defined]

    def _handle_opportunity_signal(self, signal: Any) -> None:
        """Handle OPPORTUNITY signal — reduce caution, note potential reward."""
        payload = getattr(signal, "payload", {}) if not isinstance(signal, dict) else signal
        reward_potential = payload.get("reward_potential", 0.5)
        current_risk = getattr(self, "risk_score", 0.0)
        self.risk_score = max(0.0, current_risk - 0.1)  # type: ignore[attr-defined]
        # Store opportunity context for decision-making
        self._last_opportunity = payload  # type: ignore[attr-defined]

    def _handle_convergence_signal(self, signal: Any) -> None:
        """Handle CONVERGENCE signal - update team satisfaction."""
        payload = getattr(signal, "payload", {}) if not isinstance(signal, dict) else signal
        peer_satisfaction = payload.get("satisfaction_score", 0)
        team_sat = getattr(self, "team_satisfaction", None)
        if team_sat is None:
            self.team_satisfaction = peer_satisfaction  # type: ignore[attr-defined]
        else:
            self.team_satisfaction = 0.9 * team_sat + 0.1 * peer_satisfaction  # type: ignore[attr-defined]

    def _handle_collaboration_signal(self, signal: Any) -> None:
        """Handle COLLABORATION_REQUEST — check capabilities and respond."""
        payload = getattr(signal, "payload", {}) if not isinstance(signal, dict) else signal
        required = set(payload.get("required_capabilities", []))
        sender_id = payload.get("sender_id", "unknown")
        my_caps = getattr(self, "capabilities", set())
        if required and my_caps & required:
            # We can help — emit response signal
            emit = getattr(self, "emit_signal", None)
            if emit:
                emit(
                    "COLLABORATION_RESPONSE",
                    {
                        "responder_id": str(getattr(self, "unique_id", "unknown")),
                        "matching_capabilities": list(my_caps & required),
                        "task": payload.get("task", ""),
                    },
                    priority=0.7,
                )

    def _handle_knowledge_share_signal(self, signal: Any) -> None:
        """Handle KNOWLEDGE_SHARE — learn from others' experiences (mirror neurons)."""
        payload = getattr(signal, "payload", {}) if not isinstance(signal, dict) else signal
        knowledge = payload.get("knowledge", {})
        # Learn from shared risk assessments
        if "risk" in knowledge:
            current_risk = getattr(self, "risk_score", 0.0)
            shared_risk = float(knowledge["risk"])
            # Blend: 80% own assessment, 20% peer assessment
            self.risk_score = 0.8 * current_risk + 0.2 * shared_risk  # type: ignore[attr-defined]

    def get_signal_statistics(self) -> Optional[dict[str, Any]]:
        """Get electrical signaling statistics."""
        if self.signal_bus is None:
            return None

        get_metrics = getattr(self.signal_bus, "get_metrics", None)
        bus_metrics = get_metrics() if get_metrics else {}

        return {
            "subscriptions": list(self.signal_subscriptions),
            "subscription_count": len(self.signal_subscriptions),
            "bus_metrics": bus_metrics,
        }

    def _serialize_signal_processing(self) -> dict:
        return {
            "signal_subscriptions": list(self.signal_subscriptions) if hasattr(self, "signal_subscriptions") else [],
        }

    def _restore_signal_processing(self, data: dict) -> None:
        if "signal_subscriptions" in data and hasattr(self, "signal_subscriptions"):
            for sub in data["signal_subscriptions"]:
                if sub not in self.signal_subscriptions:
                    self.signal_subscriptions.append(sub)
