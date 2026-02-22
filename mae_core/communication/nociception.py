"""Nociception System - Pain signaling for damage awareness.

Biological basis: Sherrington (1906) identified nociceptors as specialized
nerve endings that detect tissue damage. Pain signals demand immediate
attention, overriding normal processing (gate control theory, Melzack &
Wall 1965). Habituation reduces chronic pain over time. Endorphins can
suppress pain signals.

Mae uses nociception to:
1. Detect and report system damage (acute pain)
2. Track chronic degradation via somatic map integration
3. Demand immediate healing attention for severe damage
4. Allow pain suppression when healing is in progress (gate control)

Connection points:
- Subscribes to healing.failure_detected for damage events
- Subscribes to defense.threat_detected for threat-induced pain
- Publishes communication.acute_pain when any pain > 0.7
- Publishes communication.pain_overload when total load > 1.5
- SomaticMap integration: scans for unhealthy systems

Law compliance:
- Law 8 (Consciousness): Property 7 — competition/selection (pain
  demands attention over normal signals, forcing resource reallocation)
- Law 8 (Consciousness): Property 4 — recurrence/feedback (damage ->
  pain -> healing -> less pain forms a self-correcting loop)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PainSignal:
    """A pain signal from a specific source system."""

    source: str
    intensity: float  # 0-1
    pain_type: str = "acute"  # "acute", "chronic", "referred"
    step: int = 0
    habituation: float = 1.0  # Multiplier that decays over time


# Configuration constants
_DEFAULT_PAIN_THRESHOLD = 0.3
_DEFAULT_HABITUATION_RATE = 0.9
_ACUTE_PAIN_ALERT_THRESHOLD = 0.7
_PAIN_OVERLOAD_THRESHOLD = 1.5
_SOMATIC_CHRONIC_HEALTH_THRESHOLD = 0.5
_SOMATIC_CHRONIC_BASE_INTENSITY = 0.4


class NociceptionSystem:
    """Pain signaling system for detecting and communicating damage.

    Maintains a registry of active pain signals keyed by source system.
    Applies habituation (repeated pain at the same source diminishes
    over time). Supports gate control: pain can be suppressed by
    endorphin-like signals.

    Args:
        event_bus: Optional EventBus for pub/sub integration.
        somatic_map: Optional SomaticMap for scanning system health.
    """

    CH_ACUTE_PAIN = "communication.acute_pain"
    CH_PAIN_OVERLOAD = "communication.pain_overload"
    CH_PAIN_UPDATE = "communication.pain_update"

    def __init__(
        self,
        event_bus: Any | None = None,
        somatic_map: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._bus = event_bus
        self._somatic_map = somatic_map

        self._active_pains: dict[str, PainSignal] = {}
        self._pain_threshold: float = _DEFAULT_PAIN_THRESHOLD
        self._habituation_rate: float = _DEFAULT_HABITUATION_RATE
        self._total_pain_load: float = 0.0
        self._current_step: int = 0

        # Statistics
        self._total_damage_reports: int = 0
        self._total_acute_alerts: int = 0
        self._total_overload_alerts: int = 0
        self._total_suppressions: int = 0

        # Subscribe to relevant channels
        if self._bus is not None:
            self._bus.register_callback(
                "healing.failure_detected", self._on_failure_detected
            )
            self._bus.register_callback(
                "defense.threat_detected", self._on_threat_detected
            )

    # ------------------------------------------------------------------
    # Damage Reporting
    # ------------------------------------------------------------------

    def report_damage(
        self,
        source: str,
        intensity: float,
        pain_type: str = "acute",
    ) -> PainSignal:
        """Report damage from a source system.

        If an existing pain signal from the same source exists, the
        intensity is set to the maximum of current and new values.
        Creates a new PainSignal otherwise.

        Args:
            source: Name of the damaged system.
            intensity: Severity of the damage (0-1).
            pain_type: Category: "acute", "chronic", or "referred".

        Returns:
            The active PainSignal for this source.
        """
        intensity = max(0.0, min(1.0, intensity))
        self._total_damage_reports += 1

        existing = self._active_pains.get(source)
        if existing is not None:
            existing.intensity = max(existing.intensity, intensity)
            existing.pain_type = pain_type
            existing.step = self._current_step
            return existing

        signal = PainSignal(
            source=source,
            intensity=intensity,
            pain_type=pain_type,
            step=self._current_step,
            habituation=1.0,
        )
        self._active_pains[source] = signal
        return signal

    # ------------------------------------------------------------------
    # Gate Control (Pain Suppression)
    # ------------------------------------------------------------------

    def suppress_pain(self, source: str, factor: float = 0.5) -> bool:
        """Suppress pain from a specific source (like endorphins).

        Multiplies the pain intensity by `factor`. Used when healing
        is actively addressing the damaged system.

        Args:
            source: The source system whose pain to suppress.
            factor: Multiplication factor (0-1, lower = more suppression).

        Returns:
            True if pain was found and suppressed, False otherwise.
        """
        signal = self._active_pains.get(source)
        if signal is None:
            return False

        factor = max(0.0, min(1.0, factor))
        signal.intensity *= factor
        self._total_suppressions += 1

        # Remove if suppressed below threshold
        if signal.intensity < self._pain_threshold:
            del self._active_pains[source]

        return True

    # ------------------------------------------------------------------
    # Step (Autopoietic Maintenance)
    # ------------------------------------------------------------------

    def step(self, current_step: int = 0) -> None:
        """Advance one time step. Apply habituation, check alerts, scan somatic map.

        Law 6 (Autopoietic Closure): The step method self-maintains by
        habituating old pains, removing subthreshold signals, and
        generating new chronic pain signals from somatic map scanning.
        """
        self._current_step = current_step if current_step > 0 else self._current_step + 1

        # Apply habituation to all active pains
        to_remove: list[str] = []
        for source, signal in self._active_pains.items():
            signal.habituation *= self._habituation_rate
            signal.intensity *= self._habituation_rate

            if signal.intensity < self._pain_threshold:
                to_remove.append(source)

        for source in to_remove:
            del self._active_pains[source]

        # Compute total pain load
        self._total_pain_load = sum(
            s.intensity for s in self._active_pains.values()
        )

        # Check for acute pain alerts (any single pain > 0.7)
        for source, signal in self._active_pains.items():
            if signal.intensity > _ACUTE_PAIN_ALERT_THRESHOLD:
                self._total_acute_alerts += 1
                if self._bus is not None:
                    try:
                        self._bus.publish(self.CH_ACUTE_PAIN, {
                            "source": source,
                            "intensity": round(signal.intensity, 4),
                            "pain_type": signal.pain_type,
                            "urgency": "high",
                            "step": self._current_step,
                        })
                    except Exception:
                        logger.debug("EventBus publish failed for acute_pain")

        # Check for pain overload (total > 1.5)
        if self._total_pain_load > _PAIN_OVERLOAD_THRESHOLD:
            self._total_overload_alerts += 1
            if self._bus is not None:
                try:
                    self._bus.publish(self.CH_PAIN_OVERLOAD, {
                        "total_pain_load": round(self._total_pain_load, 4),
                        "active_pain_count": len(self._active_pains),
                        "step": self._current_step,
                    })
                except Exception:
                    logger.debug("EventBus publish failed for pain_overload")

        # Somatic map integration: scan for unhealthy systems
        if self._somatic_map is not None:
            self._scan_somatic_map()

        # Publish general pain update
        if self._bus is not None:
            try:
                self._bus.publish(self.CH_PAIN_UPDATE, {
                    "total_pain_load": round(self._total_pain_load, 4),
                    "active_pains": len(self._active_pains),
                    "step": self._current_step,
                })
            except Exception:
                logger.debug("EventBus publish failed for pain_update")

    def _scan_somatic_map(self) -> None:
        """Scan the somatic map for systems with low health.

        Systems with health < 0.5 generate chronic pain signals,
        reflecting ongoing degradation that hasn't been healed.
        """
        try:
            # SomaticMap has get_unhealthy_systems(threshold) and get_system_info(id)
            unhealthy = self._somatic_map.get_unhealthy_systems(
                threshold=_SOMATIC_CHRONIC_HEALTH_THRESHOLD
            )
            for system_id in unhealthy:
                info = self._somatic_map.get_system_info(system_id)
                if info is not None:
                    # Generate chronic pain proportional to damage
                    chronic_intensity = _SOMATIC_CHRONIC_BASE_INTENSITY * (
                        1.0 - info.health
                    )
                    self.report_damage(
                        source=system_id,
                        intensity=chronic_intensity,
                        pain_type="chronic",
                    )
        except Exception:
            logger.debug("Somatic map scan failed in nociception")

    # ------------------------------------------------------------------
    # EventBus Handlers
    # ------------------------------------------------------------------

    def _on_failure_detected(self, channel: str, message: Any) -> None:
        """Handle healing.failure_detected events as damage reports."""
        data = self._parse_message(message)
        if data is None:
            return

        source = data.get("system_id", data.get("failure_type", "unknown"))
        severity = data.get("severity", 0.5)
        self.report_damage(source, intensity=severity, pain_type="acute")

    def _on_threat_detected(self, channel: str, message: Any) -> None:
        """Handle defense.threat_detected events as threat-induced pain."""
        data = self._parse_message(message)
        if data is None:
            return

        source = data.get("source", data.get("threat_id", "threat"))
        score = data.get("score", 0.5)
        self.report_damage(source, intensity=score, pain_type="referred")

    @staticmethod
    def _parse_message(message: Any) -> Optional[dict[str, Any]]:
        """Parse an EventBus message to dict."""
        if isinstance(message, dict):
            return message
        if isinstance(message, str):
            try:
                return json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_pain_load(self) -> float:
        """Return the total pain intensity across all active pains."""
        return self._total_pain_load

    def get_worst_pain(self) -> Optional[PainSignal]:
        """Return the most intense active pain signal, or None."""
        if not self._active_pains:
            return None
        return max(self._active_pains.values(), key=lambda s: s.intensity)

    def is_in_pain(self) -> bool:
        """Return True if any active pains are above the threshold."""
        return any(
            s.intensity >= self._pain_threshold
            for s in self._active_pains.values()
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Return current system statistics."""
        return {
            "active_pains": len(self._active_pains),
            "total_pain_load": round(self._total_pain_load, 4),
            "pain_threshold": self._pain_threshold,
            "habituation_rate": self._habituation_rate,
            "total_damage_reports": self._total_damage_reports,
            "total_acute_alerts": self._total_acute_alerts,
            "total_overload_alerts": self._total_overload_alerts,
            "total_suppressions": self._total_suppressions,
            "current_step": self._current_step,
        }

    # ------------------------------------------------------------------
    # Tier 2 Persistence
    # ------------------------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        """Serialize state for Tier 2 persistence."""
        pains_data = {}
        for source, signal in self._active_pains.items():
            pains_data[source] = {
                "source": signal.source,
                "intensity": signal.intensity,
                "pain_type": signal.pain_type,
                "step": signal.step,
                "habituation": signal.habituation,
            }

        return {
            "active_pains": pains_data,
            "pain_threshold": self._pain_threshold,
            "habituation_rate": self._habituation_rate,
            "total_pain_load": self._total_pain_load,
            "total_damage_reports": self._total_damage_reports,
            "total_acute_alerts": self._total_acute_alerts,
            "total_overload_alerts": self._total_overload_alerts,
            "total_suppressions": self._total_suppressions,
            "current_step": self._current_step,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self._pain_threshold = data.get("pain_threshold", _DEFAULT_PAIN_THRESHOLD)
        self._habituation_rate = data.get("habituation_rate", _DEFAULT_HABITUATION_RATE)
        self._total_pain_load = data.get("total_pain_load", 0.0)
        self._total_damage_reports = data.get("total_damage_reports", 0)
        self._total_acute_alerts = data.get("total_acute_alerts", 0)
        self._total_overload_alerts = data.get("total_overload_alerts", 0)
        self._total_suppressions = data.get("total_suppressions", 0)
        self._current_step = data.get("current_step", 0)

        self._active_pains.clear()
        for source, pain_data in data.get("active_pains", {}).items():
            signal = PainSignal(
                source=pain_data["source"],
                intensity=pain_data["intensity"],
                pain_type=pain_data.get("pain_type", "acute"),
                step=pain_data.get("step", 0),
                habituation=pain_data.get("habituation", 1.0),
            )
            self._active_pains[source] = signal

    def __repr__(self) -> str:
        return (
            f"NociceptionSystem(pains={len(self._active_pains)}, "
            f"load={self._total_pain_load:.2f})"
        )
