"""Threat Detection - Multi-strategy biological defense.

Biological analogy: Four defense strategies from nature:
- PORCUPINE: Proactive detection with "quills" (tripwire sensors)
- TURTLE: Passive resilience, shell up under pressure
- LIZARD: Adaptive sacrifice (tail autotomy) to survive
- KANGAROO: Aggressive counterattack when cornered

These strategies are NOT mutually exclusive. Like a real immune system
that layers innate immunity (fast, broad) with adaptive immunity
(slow, precise), Mae layers multiple defense strategies.

Connection points:
- HAVEN provides risk assessments (input)
- EventBus receives threat alerts (output)
- AutoHealer receives failure notifications (cascade)
- Substrate can isolate regions (containment)
- Endocrine triggers adrenaline/cortisol (stress response)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# EventBus channels
CH_THREAT_DETECTED = "defense.threat_detected"
CH_DEFENSE_ACTIVATED = "defense.activated"
CH_THREAT_NEUTRALIZED = "defense.threat_neutralized"


class ThreatLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> ThreatLevel:
        if score < 0.1:
            return cls.NONE
        if score < 0.3:
            return cls.LOW
        if score < 0.6:
            return cls.MEDIUM
        if score < 0.8:
            return cls.HIGH
        return cls.CRITICAL


class DefenseStrategy(Enum):
    PORCUPINE = "porcupine"  # Proactive detection
    TURTLE = "turtle"  # Passive resilience
    LIZARD = "lizard"  # Adaptive sacrifice
    KANGAROO = "kangaroo"  # Aggressive counter


@dataclass
class Threat:
    """A detected threat."""

    threat_id: str
    source: str  # What/who is threatening
    target: str  # What's being threatened
    level: ThreatLevel
    score: float  # [0, 1]
    description: str = ""
    detected_at: float = field(default_factory=time.time)
    neutralized: bool = False
    strategy_used: Optional[DefenseStrategy] = None


@dataclass
class DefenseResponse:
    """Response to a threat."""

    threat: Threat
    strategy: DefenseStrategy
    success: bool
    actions: list[str] = field(default_factory=list)
    cost: float = 0.0  # Resource cost of defense


class ThreatDetector:
    """Multi-strategy biological defense system.

    Layers multiple detection and response strategies:

    Porcupine sensors: Custom detection functions ("quills") that
    scan for specific threat patterns. Fast, proactive, but narrow.

    Turtle shell: Tracks system integrity and hardens when under
    pressure. Slow to activate, broad protection.

    Lizard autotomy: When a subsystem is compromised, sacrifice it
    cleanly (like a lizard dropping its tail) to save the whole.

    Kangaroo counter: When threat source is identified, actively
    neutralize it (quarantine, rollback, terminate).
    """

    def __init__(
        self,
        event_bus: Any = None,
        haven: Any = None,
        energy_budget: float = 100.0,
        cooldown_steps: int = 5,
        somatic_map: Any = None,
    ) -> None:
        self._bus = event_bus
        self._haven = haven
        self._energy = energy_budget
        self._max_energy = energy_budget
        self._cooldown = cooldown_steps
        self._somatic_map = somatic_map
        self._last_activation: dict[DefenseStrategy, int] = {}

        # Porcupine: registered detection functions
        self._quills: list[Callable[..., Optional[Threat]]] = []

        # Turtle: integrity tracking
        self._integrity: float = 1.0  # [0, 1]
        self._shell_active: bool = False
        self._shell_threshold: float = 0.5  # Activate shell below this

        # Lizard: sacrificeable components
        self._sacrificeable: dict[str, float] = {}  # component -> priority (lower = more sacrificeable)

        # Kangaroo: counter-action callbacks
        self._counter_actions: dict[str, Callable] = {}

        # Threat tracking
        self._active_threats: dict[str, Threat] = {}
        self._threat_history: deque[Threat] = deque(maxlen=200)
        self._step_count = 0

        # Detection threshold (lowered when cortisol is high)
        self._detection_threshold: float = 0.5

        # Statistics
        self._stats: dict[str, int] = defaultdict(int)

        self._lock = threading.RLock()

        # Subscribe to endocrine cortisol events
        if self._bus is not None:
            self._bus.register_callback(
                "endocrine.hormone_release", self._on_hormone_release
            )
            if self._somatic_map is not None:
                self._bus.register_callback(
                    CH_DEFENSE_ACTIVATED, self._on_defense_activated
                )

        logger.info("ThreatDetector initialized (energy=%.0f)", energy_budget)

    # =========================================================================
    # Porcupine: Proactive Detection
    # =========================================================================

    def register_quill(self, detector: Callable[..., Optional[Threat]]) -> None:
        """Register a threat detection function (quill sensor)."""
        self._quills.append(detector)

    def scan_threats(self) -> list[Threat]:
        """Run all quill sensors and return detected threats."""
        threats: list[Threat] = []
        for quill in self._quills:
            try:
                threat = quill()
                if threat is not None:
                    threats.append(threat)
                    self._register_threat(threat)
            except Exception as e:
                logger.warning("Quill sensor failed: %s", e)
        return threats

    def report_threat(self, threat: Threat) -> None:
        """Manually report a threat (from external detection)."""
        self._register_threat(threat)

    def _register_threat(self, threat: Threat) -> None:
        """Register a detected threat and publish alert."""
        with self._lock:
            self._active_threats[threat.threat_id] = threat
            self._stats["threats_detected"] += 1

            if self._bus:
                self._bus.publish(CH_THREAT_DETECTED, {
                    "threat_id": threat.threat_id,
                    "source": threat.source,
                    "target": threat.target,
                    "level": threat.level.value,
                    "score": threat.score,
                })

            # Update somatic map health when threat detected
            if self._somatic_map is not None:
                health = max(0.1, 1.0 - threat.score)
                self._somatic_map.heartbeat("defense", health=health)

    # =========================================================================
    # Turtle: Passive Resilience
    # =========================================================================

    def update_integrity(self, delta: float) -> None:
        """Update system integrity score (negative = damage)."""
        with self._lock:
            self._integrity = max(0.0, min(1.0, self._integrity + delta))

            if self._integrity < self._shell_threshold and not self._shell_active:
                self._activate_shell()
            elif self._integrity >= self._shell_threshold and self._shell_active:
                self._deactivate_shell()

    def _activate_shell(self) -> None:
        """Activate turtle shell (defensive mode)."""
        self._shell_active = True
        self._stats["shell_activations"] += 1

        if self._bus:
            self._bus.publish(CH_DEFENSE_ACTIVATED, {
                "strategy": "turtle",
                "integrity": self._integrity,
                "action": "shell_up",
            })
        logger.info("Turtle shell ACTIVATED (integrity=%.2f)", self._integrity)

    def _deactivate_shell(self) -> None:
        self._shell_active = False
        logger.info("Turtle shell deactivated (integrity=%.2f)", self._integrity)

    @property
    def is_shelled(self) -> bool:
        return self._shell_active

    @property
    def integrity(self) -> float:
        return self._integrity

    # =========================================================================
    # Lizard: Adaptive Sacrifice
    # =========================================================================

    def register_sacrificeable(self, component: str, priority: float) -> None:
        """Register a component that can be sacrificed. Lower priority = sacrificed first."""
        self._sacrificeable[component] = priority

    def sacrifice(self, reason: str = "") -> Optional[str]:
        """Sacrifice the lowest-priority component to survive.

        Returns the sacrificed component name, or None if nothing to sacrifice.
        """
        with self._lock:
            if not self._sacrificeable:
                return None

            # Sacrifice lowest priority
            victim = min(self._sacrificeable, key=self._sacrificeable.get)
            del self._sacrificeable[victim]
            self._stats["sacrifices"] += 1

            if self._bus:
                self._bus.publish(CH_DEFENSE_ACTIVATED, {
                    "strategy": "lizard",
                    "component": victim,
                    "reason": reason,
                    "action": "autotomy",
                })

            logger.info("Lizard autotomy: sacrificed '%s' (%s)", victim, reason)
            return victim

    # =========================================================================
    # Kangaroo: Aggressive Counter
    # =========================================================================

    def register_counter_action(
        self, threat_source: str, action: Callable
    ) -> None:
        """Register a counter-action for a specific threat source."""
        self._counter_actions[threat_source] = action

    def counterattack(self, threat: Threat) -> DefenseResponse:
        """Actively neutralize a threat source."""
        with self._lock:
            actions: list[str] = []
            success = False

            # Check cooldown
            if not self._check_cooldown(DefenseStrategy.KANGAROO):
                return DefenseResponse(
                    threat=threat,
                    strategy=DefenseStrategy.KANGAROO,
                    success=False,
                    actions=["cooldown_active"],
                )

            # Check energy
            cost = 10.0 * (1.0 + threat.score)
            if self._energy < cost:
                return DefenseResponse(
                    threat=threat,
                    strategy=DefenseStrategy.KANGAROO,
                    success=False,
                    actions=["insufficient_energy"],
                    cost=0,
                )

            self._energy -= cost

            # Execute counter-action if registered
            counter = self._counter_actions.get(threat.source)
            if counter:
                try:
                    counter(threat)
                    actions.append(f"counter_{threat.source}")
                    success = True
                except Exception as e:
                    actions.append(f"counter_failed: {e}")

            # Default: isolate via HAVEN
            if self._haven and not success:
                self._haven.isolate_agent(threat.source, reason=f"counterattack: {threat.description}")
                actions.append(f"isolated_{threat.source}")
                success = True

            if success:
                threat.neutralized = True
                threat.strategy_used = DefenseStrategy.KANGAROO
                self._stats["counterattacks"] += 1

            self._last_activation[DefenseStrategy.KANGAROO] = self._step_count

            if self._bus:
                self._bus.publish(CH_THREAT_NEUTRALIZED if success else CH_DEFENSE_ACTIVATED, {
                    "strategy": "kangaroo",
                    "threat_id": threat.threat_id,
                    "success": success,
                    "actions": actions,
                })

            return DefenseResponse(
                threat=threat,
                strategy=DefenseStrategy.KANGAROO,
                success=success,
                actions=actions,
                cost=cost,
            )

    # =========================================================================
    # Periodic Scanning
    # =========================================================================

    def periodic_scan(self, agents: list[Any]) -> list[Threat]:
        """Scan a list of agents for anomalies.

        Runs all registered quill sensors against each agent and returns
        any detected threats.
        """
        threats: list[Threat] = []
        for agent in agents:
            for quill in self._quills:
                try:
                    threat = quill(agent)
                    if threat is not None:
                        threats.append(threat)
                        self._register_threat(threat)
                except Exception as e:
                    logger.warning("Quill sensor failed on agent scan: %s", e)
        return threats

    # =========================================================================
    # Endocrine Integration
    # =========================================================================

    def _on_hormone_release(self, channel: str, message: Any) -> None:
        """Handle endocrine.hormone_release events.

        When cortisol is high, lower detection thresholds (more sensitive).
        """
        import json

        data = message
        if isinstance(message, str):
            try:
                data = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return

        if not isinstance(data, dict):
            return

        hormone = data.get("hormone", "")
        level = data.get("level", 0.0)

        if hormone == "cortisol" and level > 0.5:
            # High cortisol = lower threshold = more sensitive detection
            self._detection_threshold = max(0.1, 0.5 - (level - 0.5) * 0.5)
            logger.debug(
                "Cortisol high (%.2f), detection threshold lowered to %.2f",
                level, self._detection_threshold,
            )

    # =========================================================================
    # Somatic Map Integration
    # =========================================================================

    def _on_defense_activated(self, channel: str, message: Any) -> None:
        """Handle defense.activated events to update somatic_map body awareness."""
        if self._somatic_map is None:
            return

        import json

        data = message
        if isinstance(message, str):
            try:
                data = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return

        if not isinstance(data, dict):
            return

        strategy = data.get("strategy", "")
        health = 0.5 if strategy in ("turtle", "lizard") else 0.7
        self._somatic_map.heartbeat("defense", health=health)

    # =========================================================================
    # Automated Response
    # =========================================================================

    def respond_to_threat(self, threat: Threat) -> DefenseResponse:
        """Choose and execute the best defense strategy for a threat."""
        # Strategy selection based on threat level and system state
        if threat.level == ThreatLevel.CRITICAL:
            # Critical: counterattack immediately
            return self.counterattack(threat)
        elif threat.level == ThreatLevel.HIGH and self._shell_active:
            # High + shelled: sacrifice to reduce pressure
            sacrificed = self.sacrifice(reason=threat.description)
            return DefenseResponse(
                threat=threat,
                strategy=DefenseStrategy.LIZARD,
                success=sacrificed is not None,
                actions=[f"sacrificed_{sacrificed}" if sacrificed else "nothing_to_sacrifice"],
            )
        elif threat.level in (ThreatLevel.MEDIUM, ThreatLevel.HIGH):
            # Medium/High: shell up
            self.update_integrity(-0.2)
            return DefenseResponse(
                threat=threat,
                strategy=DefenseStrategy.TURTLE,
                success=True,
                actions=["integrity_reduced", "monitoring"],
            )
        else:
            # Low: just monitor
            return DefenseResponse(
                threat=threat,
                strategy=DefenseStrategy.PORCUPINE,
                success=True,
                actions=["monitoring"],
            )

    # =========================================================================
    # Step and Energy
    # =========================================================================

    def step(self) -> None:
        """Advance one step. Recharge energy, clean old threats."""
        self._step_count += 1

        # Energy recharge
        self._energy = min(self._max_energy, self._energy + 2.0)

        # Integrity natural recovery (slow)
        if not self._shell_active:
            self._integrity = min(1.0, self._integrity + 0.01)

        # Clean neutralized threats
        with self._lock:
            for tid, threat in list(self._active_threats.items()):
                if threat.neutralized:
                    self._threat_history.append(threat)
                    del self._active_threats[tid]

    def _check_cooldown(self, strategy: DefenseStrategy) -> bool:
        last = self._last_activation.get(strategy, -999)
        return (self._step_count - last) >= self._cooldown

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "energy": self._energy,
                "max_energy": self._max_energy,
                "integrity": self._integrity,
                "shell_active": self._shell_active,
                "active_threats": len(self._active_threats),
                "threat_history": len(self._threat_history),
                "quills_registered": len(self._quills),
                "sacrificeable_components": len(self._sacrificeable),
                **dict(self._stats),
            }

    def get_active_threats(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "threat_id": t.threat_id,
                    "source": t.source,
                    "level": t.level.value,
                    "score": t.score,
                }
                for t in self._active_threats.values()
            ]
