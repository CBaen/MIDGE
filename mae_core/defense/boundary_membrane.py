"""BoundaryMembrane - Selective permeability and self/non-self recognition.

Biological analogy: Cell membrane with selective permeability. MHC (Major
Histocompatibility Complex) for self/non-self recognition. Skin as barrier organ.

Every living cell defines itself by its membrane. The membrane doesn't just
block threats -- it defines what is "self" versus "non-self" by recognizing
molecular signatures. Known internal molecules pass freely (MHC Class I),
trusted external molecules pass after inspection (antigen presentation),
unknown molecules are quarantined for immune review, and known pathogens
are rejected outright.

Mae's BoundaryMembrane does the same:
- Internal system signals pass on a fast path (self-recognition)
- Trusted external sources pass with minimal inspection
- Unknown sources are quarantined and must be approved
- Known-bad sources are rejected immediately

Connection points:
- InputValidator: downstream consumer of membrane decisions
- ThreatDetector: threat events lower permeability
- PearlDefense: quarantined items may enter pearl processing
- EventBus: publishes membrane status and quarantine events

Law compliance:
- Law 6 (Autopoietic Closure): self-produced boundary defines its own
  self/non-self distinction. The membrane IS the boundary.
- Law 8 (Consciousness): contributes to differentiation (property 2) --
  distinguishes self from non-self at the system boundary.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# EventBus channels
CH_MEMBRANE_STATUS = "defense.membrane_status"
CH_QUARANTINE = "defense.quarantine_event"


@dataclass
class MembranePassport:
    """Identity passport for a source interacting with the membrane."""

    source: str
    signature: str
    trust_level: float = 0.0
    first_seen: int = 0
    access_count: int = 0
    is_self: bool = False
    trust_floor: float = 0.0  # Pre-registered sources never decay below this


class BoundaryMembrane:
    """Selective permeability membrane with self/non-self recognition.

    Maintains a registry of known sources (passports) and classifies
    incoming signals as self, trusted, provisional, quarantined, or
    rejected based on accumulated trust and self-recognition.

    Permeability adapts to threat conditions: more threats tighten
    the membrane, calm periods relax it.
    """

    CH_MEMBRANE_STATUS = CH_MEMBRANE_STATUS
    CH_QUARANTINE = CH_QUARANTINE

    def __init__(self, event_bus: Any = None, **kwargs: Any) -> None:
        self._bus = event_bus
        self._known_sources: dict[str, MembranePassport] = {}
        self._self_signature: str = ""
        self._permeability: float = 0.5  # 0=locked down, 1=fully open
        self._saved_permeability: float = 0.5  # saved before lockdown
        self._quarantine_queue: list[dict[str, Any]] = []
        self._step_count: int = 0

        # Threat tracking for adaptive permeability
        self._recent_threats: int = 0
        self._calm_steps: int = 0

        # Statistics
        self._total_checks: int = 0
        self._self_passes: int = 0
        self._trusted_passes: int = 0
        self._provisional_passes: int = 0
        self._quarantined_count: int = 0
        self._rejected_count: int = 0

        # Subscribe to threat events to adapt permeability
        if self._bus is not None:
            self._bus.register_callback(
                "defense.threat_detected", self._on_threat_detected
            )

        logger.info(
            "BoundaryMembrane initialized (permeability=%.2f)",
            self._permeability,
        )

    # =========================================================================
    # Self-Recognition
    # =========================================================================

    def register_source(self, source: str, trust: float = 0.8) -> None:
        """Pre-register an external source with a given initial trust level.

        Used during bootstrap to give LLM providers trusted status so
        their requests bypass the quarantine path on first contact.
        Trust >= 0.7 means the source passes as "trusted" on check_input().
        """
        passport = MembranePassport(
            source=source,
            signature="external_trusted",
            trust_level=trust,
            first_seen=self._step_count,
            access_count=0,
            is_self=False,
            trust_floor=max(0.35, trust * 0.5),  # Never decay below half initial trust (min 0.35 > rejection threshold 0.3)
        )
        self._known_sources[source] = passport
        logger.info("External source pre-trusted: %s (trust=%.2f)", source, trust)

    def register_self(self, system_names: list[str]) -> None:
        """Compute self-signature from internal system names.

        All internal sources get is_self=True passports. The self-signature
        is a hash of the sorted system names, defining what "self" means
        for this organism instance.
        """
        sorted_names = sorted(system_names)
        signature_input = "|".join(sorted_names)
        self._self_signature = hashlib.sha256(
            signature_input.encode("utf-8")
        ).hexdigest()[:16]

        for name in system_names:
            passport = MembranePassport(
                source=name,
                signature=self._self_signature,
                trust_level=1.0,
                first_seen=self._step_count,
                access_count=0,
                is_self=True,
            )
            self._known_sources[name] = passport

        logger.info(
            "Self registered: %d systems, signature=%s",
            len(system_names),
            self._self_signature,
        )

    # =========================================================================
    # Input Checking
    # =========================================================================

    def check_input(
        self, source: str, data: dict[str, Any]
    ) -> tuple[bool, str]:
        """Check whether an input from a source should be allowed.

        Returns:
            (allowed: bool, reason: str) where reason is one of:
            "self", "trusted", "provisional", "quarantine", "rejected"
        """
        self._total_checks += 1
        passport = self._known_sources.get(source)

        # Fast path: self-recognition
        if passport is not None and passport.is_self:
            passport.access_count += 1
            self._self_passes += 1
            return (True, "self")

        # Known source with high trust
        if passport is not None and passport.trust_level >= 0.7:
            passport.access_count += 1
            self._trusted_passes += 1
            return (True, "trusted")

        # Unknown source: create passport and quarantine
        if passport is None:
            passport = MembranePassport(
                source=source,
                signature="",
                trust_level=0.5,
                first_seen=self._step_count,
                access_count=1,
            )
            self._known_sources[source] = passport
            self._quarantine_queue.append({
                "source": source,
                "data": data,
                "step_added": self._step_count,
            })
            self._quarantined_count += 1

            if self._bus is not None:
                self._bus.publish(CH_QUARANTINE, {
                    "action": "quarantined",
                    "source": source,
                    "step": self._step_count,
                })

            return (False, "quarantine")

        # Known source with very low trust
        if passport.trust_level < 0.3:
            passport.access_count += 1
            self._rejected_count += 1
            return (False, "rejected")

        # Known source building trust (0.3 <= trust < 0.7)
        passport.access_count += 1
        self._provisional_passes += 1
        return (True, "provisional")

    # =========================================================================
    # Quarantine Management
    # =========================================================================

    def approve_quarantined(self, source: str) -> None:
        """Approve a quarantined source. Boosts trust by 0.2."""
        self._quarantine_queue = [
            item for item in self._quarantine_queue
            if item["source"] != source
        ]
        passport = self._known_sources.get(source)
        if passport is not None:
            passport.trust_level = min(1.0, passport.trust_level + 0.2)

        if self._bus is not None:
            self._bus.publish(CH_QUARANTINE, {
                "action": "approved",
                "source": source,
                "step": self._step_count,
            })

    def refresh_trust(self, source: str, boost: float = 0.1) -> None:
        """Refresh trust for a source after a successful interaction.

        Biological analogy: Immune memory. Successfully processed molecules
        get recognized faster next time. Without this, trust decays to zero
        even for sources that are actively and successfully communicating.
        """
        passport = self._known_sources.get(source)
        if passport is not None and not passport.is_self:
            passport.trust_level = min(1.0, passport.trust_level + boost)

    def reject_quarantined(self, source: str) -> None:
        """Reject a quarantined source. Reduces trust by 0.3."""
        self._quarantine_queue = [
            item for item in self._quarantine_queue
            if item["source"] != source
        ]
        passport = self._known_sources.get(source)
        if passport is not None:
            passport.trust_level = max(0.0, passport.trust_level - 0.3)

        if self._bus is not None:
            self._bus.publish(CH_QUARANTINE, {
                "action": "rejected",
                "source": source,
                "step": self._step_count,
            })

    # =========================================================================
    # Lockdown
    # =========================================================================

    def set_lockdown(self, locked: bool) -> None:
        """Lock down or restore membrane permeability.

        When locked: permeability = 0.0 (nothing passes except self).
        When unlocked: restores previous permeability.
        """
        if locked:
            self._saved_permeability = self._permeability
            self._permeability = 0.0
            logger.info("Membrane LOCKDOWN activated")
        else:
            self._permeability = self._saved_permeability
            logger.info(
                "Membrane lockdown released (permeability=%.2f)",
                self._permeability,
            )

    # =========================================================================
    # Step (Periodic Maintenance)
    # =========================================================================

    def step(self, current_step: int = 0) -> None:
        """Per-step maintenance: trust decay, quarantine aging, permeability adaptation."""
        self._step_count = current_step if current_step > 0 else self._step_count + 1

        # Decay trust for unseen sources (non-self only), respecting floor
        for passport in self._known_sources.values():
            if not passport.is_self:
                passport.trust_level = max(
                    passport.trust_floor,
                    passport.trust_level * 0.99,
                )

        # Age quarantine items: auto-reject after 50 steps
        expired: list[str] = []
        for item in self._quarantine_queue:
            age = self._step_count - item["step_added"]
            if age > 50:
                expired.append(item["source"])

        for source in expired:
            self.reject_quarantined(source)

        # Adapt permeability based on threat conditions
        if self._recent_threats > 0:
            # More threats -> lower permeability
            self._permeability = max(0.1, self._permeability - 0.05)
            self._recent_threats = 0
            self._calm_steps = 0
        else:
            self._calm_steps += 1
            if self._calm_steps > 10:
                # Calm period -> slowly raise permeability
                self._permeability = min(0.9, self._permeability + 0.02)

        # Publish status
        if self._bus is not None:
            self_sources = sum(
                1 for p in self._known_sources.values() if p.is_self
            )
            self._bus.publish(CH_MEMBRANE_STATUS, {
                "permeability": self._permeability,
                "known_sources": len(self._known_sources),
                "quarantine_size": len(self._quarantine_queue),
                "self_sources": self_sources,
                "step": self._step_count,
            })

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_threat_detected(self, channel: str, message: Any) -> None:
        """React to threat events by tracking threat count for permeability."""
        self._recent_threats += 1

    def _on_blanket_report(self, channel: str, message: Any) -> None:
        """Adapt permeability based on Markov blanket effectiveness from IntegrationMeter.

        Low blanket effectiveness means boundaries are porous at the fractal level,
        so the membrane tightens slightly. High effectiveness means boundaries are
        well-defined, allowing the membrane to relax slightly.

        Advisory-style: adjustments are small (+-0.05), clamped to [0.1, 1.0].
        """
        import json as _json
        msg = _json.loads(message) if isinstance(message, str) else message
        if not isinstance(msg, dict):
            return

        effectiveness_map = msg.get("blanket_effectiveness")
        if not effectiveness_map or not isinstance(effectiveness_map, dict):
            return

        values = [v for v in effectiveness_map.values() if isinstance(v, (int, float))]
        if not values:
            return

        mean_eff = sum(values) / len(values)
        if mean_eff < 0.5:
            # Weak blankets -> tighten membrane
            self._permeability = max(0.1, self._permeability - 0.05)
        elif mean_eff > 0.8:
            # Strong blankets -> allow slightly more permeable
            self._permeability = min(1.0, self._permeability + 0.05)
        # Between 0.5 and 0.8: no adjustment (stable range)

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Return membrane statistics for observability."""
        self_sources = sum(
            1 for p in self._known_sources.values() if p.is_self
        )
        return {
            "permeability": self._permeability,
            "known_sources": len(self._known_sources),
            "self_sources": self_sources,
            "quarantine_size": len(self._quarantine_queue),
            "total_checks": self._total_checks,
            "self_passes": self._self_passes,
            "trusted_passes": self._trusted_passes,
            "provisional_passes": self._provisional_passes,
            "quarantined_count": self._quarantined_count,
            "rejected_count": self._rejected_count,
            "self_signature": self._self_signature,
            "step": self._step_count,
        }

    # =========================================================================
    # Persistence (Tier 2)
    # =========================================================================

    def serialize(self) -> dict[str, Any]:
        """Serialize membrane state for persistence."""
        passports = {}
        for source, passport in self._known_sources.items():
            passports[source] = {
                "source": passport.source,
                "signature": passport.signature,
                "trust_level": passport.trust_level,
                "first_seen": passport.first_seen,
                "access_count": passport.access_count,
                "is_self": passport.is_self,
                "trust_floor": passport.trust_floor,
            }
        return {
            "known_sources": passports,
            "self_signature": self._self_signature,
            "permeability": self._permeability,
            "saved_permeability": self._saved_permeability,
            "quarantine_queue": self._quarantine_queue,
            "step_count": self._step_count,
            "recent_threats": self._recent_threats,
            "calm_steps": self._calm_steps,
            "total_checks": self._total_checks,
            "self_passes": self._self_passes,
            "trusted_passes": self._trusted_passes,
            "provisional_passes": self._provisional_passes,
            "quarantined_count": self._quarantined_count,
            "rejected_count": self._rejected_count,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore membrane state from serialized data."""
        self._self_signature = data.get("self_signature", "")
        self._permeability = data.get("permeability", 0.5)
        self._saved_permeability = data.get("saved_permeability", 0.5)
        self._quarantine_queue = data.get("quarantine_queue", [])
        self._step_count = data.get("step_count", 0)
        self._recent_threats = data.get("recent_threats", 0)
        self._calm_steps = data.get("calm_steps", 0)
        self._total_checks = data.get("total_checks", 0)
        self._self_passes = data.get("self_passes", 0)
        self._trusted_passes = data.get("trusted_passes", 0)
        self._provisional_passes = data.get("provisional_passes", 0)
        self._quarantined_count = data.get("quarantined_count", 0)
        self._rejected_count = data.get("rejected_count", 0)

        for source, pdata in data.get("known_sources", {}).items():
            self._known_sources[source] = MembranePassport(
                source=pdata["source"],
                signature=pdata["signature"],
                trust_level=pdata["trust_level"],
                first_seen=pdata["first_seen"],
                access_count=pdata["access_count"],
                is_self=pdata["is_self"],
                trust_floor=pdata.get("trust_floor", 0.0),
            )

        logger.info(
            "BoundaryMembrane restored (%d sources, permeability=%.2f)",
            len(self._known_sources),
            self._permeability,
        )
