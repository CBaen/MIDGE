"""PearlDefense - Nacre-inspired threat encapsulation.

Biological analogy: When an oyster encounters a foreign body it cannot
expel, it coats it in alternating layers of aragonite (hard crystalline
CaCO3) and conchiolin (soft organic protein). Layer by layer, the
irritant becomes a pearl -- neutralized, contained, and sometimes
valuable. The alternating hard/soft structure gives nacre its
extraordinary toughness (3000x stronger than pure aragonite).

Mae's PearlDefense does the same with suspicious inputs:
1. Input fails initial validation but isn't a clear threat -> QUARANTINED
2. Pearl process begins: alternating strict/lenient validation layers
3. Each layer that passes adds a "nacre coating" (trust increment)
4. If all layers pass, the input is released as PEARLED (safe, with metadata)
5. If any layer fails hard, the pearl is dissolved and input REJECTED

This creates graduated response: not everything is binary pass/fail.
Some inputs need time and multiple perspectives to evaluate.

EventBus channels:
- defense.pearl_started: New pearl process began
- defense.pearl_completed: Pearl matured and input released
- defense.pearl_dissolved: Pearl failed and input rejected
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from mae_core.backbone.event_bus import EventBus
from mae_core.defense.input_validator import InputValidator, ValidationReport, ValidationResult

logger = logging.getLogger(__name__)

CH_PEARL_STARTED = "defense.pearl_started"
CH_PEARL_COMPLETED = "defense.pearl_completed"
CH_PEARL_DISSOLVED = "defense.pearl_dissolved"


class PearlStatus(Enum):
    """Status of a pearl in the chamber."""
    FORMING = "forming"       # Still being coated
    COMPLETE = "complete"     # Matured -- safe to release
    DISSOLVED = "dissolved"   # Failed -- rejected


@dataclass
class Pearl:
    """A single pearl wrapping a suspicious input.

    Each pearl tracks its layers (alternating hard/soft validation),
    the original input, and cumulative trust built through layers.
    """
    pearl_id: str
    source: str
    input_type: str
    data: Any
    numeric_value: Optional[float] = None
    status: PearlStatus = PearlStatus.FORMING
    layers: list[dict[str, Any]] = field(default_factory=list)
    trust_accumulated: float = 0.0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def age(self) -> float:
        return time.time() - self.created_at


@dataclass
class PearlConfig:
    """Configuration for pearl defense dynamics."""
    required_layers: int = 5        # Layers needed to complete a pearl
    trust_per_layer: float = 0.08   # Trust gained per passing layer
    trust_threshold: float = 0.35   # Cumulative trust needed for release
    max_pearl_age: float = 300.0    # Seconds before pearl expires (dissolves)
    max_chamber_size: int = 50      # Maximum concurrent pearls
    strict_validators: int = 3      # Extra validators on "hard" layers
    lenient_validators: int = 1     # Extra validators on "soft" layers


class PearlDefense:
    """Nacre-inspired graduated threat encapsulation.

    Wraps InputValidator with a pearl chamber. Quarantined inputs
    enter the chamber and undergo alternating strict/lenient
    validation layers until they either mature (PEARLED) or
    dissolve (REJECTED).

    Integration points:
    - InputValidator: delegates initial validation, runs layer checks
    - ThreatDetector: dissolved pearls increase threat awareness
    - HAVEN: pearl completion/dissolution affects risk calculations
    - SomaticMap: registered for body awareness
    """

    def __init__(
        self,
        input_validator: InputValidator,
        event_bus: Optional[EventBus] = None,
        config: Optional[PearlConfig] = None,
    ) -> None:
        self._validator = input_validator
        self._bus = event_bus
        self._config = config or PearlConfig()

        # Pearl chamber: active pearls being processed
        self._chamber: dict[str, Pearl] = {}
        self._pearl_counter: int = 0

        # Statistics
        self._total_started: int = 0
        self._total_completed: int = 0
        self._total_dissolved: int = 0
        self._history: deque[dict[str, Any]] = deque(maxlen=200)

        logger.info(
            "PearlDefense initialized (layers=%d, trust_threshold=%.2f, max_age=%.0fs)",
            self._config.required_layers,
            self._config.trust_threshold,
            self._config.max_pearl_age,
        )

    def validate(
        self,
        source: str,
        input_type: str,
        data: Any,
        numeric_value: Optional[float] = None,
    ) -> ValidationReport:
        """Validate input with pearl defense.

        1. Run through InputValidator first
        2. If PASSED -> return immediately
        3. If FAILED -> return immediately (clear threat)
        4. If QUARANTINED -> start pearl process
        """
        report = self._validator.validate(source, input_type, data, numeric_value)

        if report.result == ValidationResult.PASSED:
            return report

        if report.result == ValidationResult.FAILED:
            return report

        # QUARANTINED: start or continue pearl process
        return self._process_pearl(source, input_type, data, numeric_value, report)

    def _process_pearl(
        self,
        source: str,
        input_type: str,
        data: Any,
        numeric_value: Optional[float],
        initial_report: ValidationReport,
    ) -> ValidationReport:
        """Start or continue pearl encapsulation for quarantined input."""
        # Check if we already have a pearl for this source+type
        pearl_key = f"{source}:{input_type}"
        pearl = self._chamber.get(pearl_key)

        if pearl is None:
            # Start new pearl
            if len(self._chamber) >= self._config.max_chamber_size:
                # Chamber full -- reject
                return initial_report

            pearl = self._start_pearl(source, input_type, data, numeric_value)
            self._chamber[pearl_key] = pearl

        # Apply next layer
        self._apply_layer(pearl)

        # Check completion
        if pearl.status == PearlStatus.COMPLETE:
            del self._chamber[pearl_key]
            return ValidationReport(
                source=source,
                input_type=input_type,
                result=ValidationResult.PASSED,
                checks_passed=pearl.layer_count,
                checks_failed=0,
                details=[f"pearl_complete (layers={pearl.layer_count}, trust={pearl.trust_accumulated:.2f})"],
            )

        if pearl.status == PearlStatus.DISSOLVED:
            del self._chamber[pearl_key]
            return ValidationReport(
                source=source,
                input_type=input_type,
                result=ValidationResult.FAILED,
                checks_passed=0,
                checks_failed=pearl.layer_count,
                details=[f"pearl_dissolved (layers={pearl.layer_count})"],
            )

        # Still forming -- return quarantined with progress
        return ValidationReport(
            source=source,
            input_type=input_type,
            result=ValidationResult.QUARANTINED,
            checks_passed=sum(1 for l in pearl.layers if l["passed"]),
            checks_failed=sum(1 for l in pearl.layers if not l["passed"]),
            details=[f"pearl_forming (layer={pearl.layer_count}/{self._config.required_layers})"],
        )

    def _start_pearl(
        self,
        source: str,
        input_type: str,
        data: Any,
        numeric_value: Optional[float],
    ) -> Pearl:
        """Begin a new pearl encapsulation."""
        self._pearl_counter += 1
        pearl_id = f"pearl-{self._pearl_counter}"

        pearl = Pearl(
            pearl_id=pearl_id,
            source=source,
            input_type=input_type,
            data=data,
            numeric_value=numeric_value,
        )

        self._total_started += 1

        if self._bus is not None:
            self._bus.publish(CH_PEARL_STARTED, {
                "pearl_id": pearl_id,
                "source": source,
                "input_type": input_type,
            })

        logger.debug("Pearl started: %s from %s", pearl_id, source)
        return pearl

    def _apply_layer(self, pearl: Pearl) -> None:
        """Apply one nacre layer to a pearl.

        Odd layers = hard (aragonite) = strict validation
        Even layers = soft (conchiolin) = lenient validation
        """
        layer_num = pearl.layer_count
        is_hard = (layer_num % 2 == 0)  # 0, 2, 4 = hard; 1, 3 = soft

        # Run validation with appropriate strictness
        report = self._validator.validate(
            pearl.source,
            pearl.input_type,
            pearl.data,
            pearl.numeric_value,
        )

        # Hard layers require zero failures; soft layers tolerate one
        if is_hard:
            passed = report.checks_failed == 0
        else:
            passed = report.checks_failed <= 1

        # Record layer
        pearl.layers.append({
            "layer_num": layer_num,
            "type": "hard" if is_hard else "soft",
            "passed": passed,
            "checks_passed": report.checks_passed,
            "checks_failed": report.checks_failed,
        })

        # Update trust
        if passed:
            pearl.trust_accumulated += self._config.trust_per_layer
        else:
            # Hard layer failure on a hard layer = dissolve immediately
            if is_hard:
                self._dissolve_pearl(pearl)
                return

        # Check completion conditions
        if (
            pearl.layer_count >= self._config.required_layers
            and pearl.trust_accumulated >= self._config.trust_threshold
        ):
            self._complete_pearl(pearl)

    def _complete_pearl(self, pearl: Pearl) -> None:
        """Pearl has matured -- release the input as safe."""
        pearl.status = PearlStatus.COMPLETE
        pearl.completed_at = time.time()
        self._total_completed += 1

        # Boost source trust in the validator
        self._validator.update_trust(pearl.source, passed=True)

        self._history.append({
            "pearl_id": pearl.pearl_id,
            "source": pearl.source,
            "outcome": "completed",
            "layers": pearl.layer_count,
            "trust": pearl.trust_accumulated,
        })

        if self._bus is not None:
            self._bus.publish(CH_PEARL_COMPLETED, {
                "pearl_id": pearl.pearl_id,
                "source": pearl.source,
                "layers": pearl.layer_count,
                "trust": pearl.trust_accumulated,
            })

        logger.debug("Pearl completed: %s (layers=%d)", pearl.pearl_id, pearl.layer_count)

    def _dissolve_pearl(self, pearl: Pearl) -> None:
        """Pearl failed -- reject the input."""
        pearl.status = PearlStatus.DISSOLVED
        self._total_dissolved += 1

        # Decrease source trust
        self._validator.update_trust(pearl.source, passed=False)

        self._history.append({
            "pearl_id": pearl.pearl_id,
            "source": pearl.source,
            "outcome": "dissolved",
            "layers": pearl.layer_count,
        })

        if self._bus is not None:
            self._bus.publish(CH_PEARL_DISSOLVED, {
                "pearl_id": pearl.pearl_id,
                "source": pearl.source,
                "layers": pearl.layer_count,
            })

        logger.debug("Pearl dissolved: %s (layers=%d)", pearl.pearl_id, pearl.layer_count)

    # =========================================================================
    # Periodic maintenance
    # =========================================================================

    def step(self) -> None:
        """Age pearls and dissolve expired ones.

        Called each model step. Pearls that exceed max_age are
        dissolved -- if the organism can't decide in time, reject.
        """
        expired_keys: list[str] = []

        for key, pearl in self._chamber.items():
            if pearl.age > self._config.max_pearl_age:
                self._dissolve_pearl(pearl)
                expired_keys.append(key)

        for key in expired_keys:
            del self._chamber[key]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        return {
            "total_started": self._total_started,
            "total_completed": self._total_completed,
            "total_dissolved": self._total_dissolved,
            "active_pearls": len(self._chamber),
            "completion_rate": (
                self._total_completed / max(self._total_started, 1)
            ),
            "config": {
                "required_layers": self._config.required_layers,
                "trust_threshold": self._config.trust_threshold,
                "max_pearl_age": self._config.max_pearl_age,
            },
        }
