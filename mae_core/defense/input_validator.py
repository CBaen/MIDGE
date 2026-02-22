"""Input Validation - Zero-trust verification at system boundaries.

Biological analogy: Mucosal immune system. The gut, lungs, and skin
don't trust anything that arrives. Every molecule is inspected at the
epithelial barrier before being allowed into the body. Toll-like
receptors pattern-match against known threats. Tight junctions between
cells prevent unauthorized passage.

Mae's input validator is the epithelial barrier. Every signal, message,
policy update, and state change from external sources gets validated
before entering the system. Trust is earned through verification,
never assumed.

Connection points:
- All external-facing systems route through validation
- HAVEN receives validation failures as risk signals
- EventBus publishes validation events
- FRL policy updates are validated before acceptance
- GNN messages are validated before routing
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# EventBus channels
CH_VALIDATION_FAILED = "defense.validation_failed"
CH_TRUST_UPDATED = "defense.trust_updated"


class ValidationResult(Enum):
    PASSED = "passed"
    FAILED = "failed"
    QUARANTINED = "quarantined"  # Suspicious but not blocked


@dataclass
class ValidationReport:
    """Result of input validation."""

    source: str
    input_type: str
    result: ValidationResult
    checks_passed: int
    checks_failed: int
    details: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class InputValidator:
    """Zero-trust input validation at system boundaries.

    Every input is verified against:
    1. Type and range checks (is it well-formed?)
    2. Source trust score (has this source been reliable?)
    3. Anomaly detection (is this input unusual?)
    4. Custom validators (domain-specific checks)

    Trust is dynamic: starts at default, increases with good behavior,
    decreases with violations. Like adaptive immunity building tolerance.
    """

    def __init__(
        self,
        event_bus: Any = None,
        default_trust: float = 0.5,
        trust_increment: float = 0.02,
        trust_decrement: float = 0.1,
        min_trust_for_accept: float = 0.3,
        quarantine_trust: float = 0.2,
        frl_engine: Any = None,
    ) -> None:
        self._bus = event_bus
        self._default_trust = default_trust
        self._trust_inc = trust_increment
        self._trust_dec = trust_decrement
        self._min_trust = min_trust_for_accept
        self._quarantine_trust = quarantine_trust
        self._frl_engine = frl_engine

        # Per-source trust scores
        self._trust_scores: dict[str, float] = {}

        # Custom validators: input_type -> list of (name, checker_fn)
        self._validators: dict[str, list[tuple[str, Callable]]] = defaultdict(list)

        # History for anomaly detection
        self._input_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Statistics
        self._total_validations = 0
        self._passed = 0
        self._failed = 0
        self._quarantined = 0
        self._violations: deque[ValidationReport] = deque(maxlen=200)

        self._lock = threading.RLock()

        # Subscribe to trust update events to sync with FRL
        if self._bus is not None:
            self._bus.register_callback(
                "defense.trust_updated", self._on_trust_updated
            )

        logger.info(
            "InputValidator initialized (default_trust=%.2f, min=%.2f)",
            default_trust, min_trust_for_accept,
        )

    # =========================================================================
    # Trust Management
    # =========================================================================

    def get_trust(self, source: str) -> float:
        """Get trust score for a source. Creates default if new."""
        with self._lock:
            if source not in self._trust_scores:
                self._trust_scores[source] = self._default_trust
            return self._trust_scores[source]

    def update_trust(self, source: str, passed: bool) -> float:
        """Update trust based on validation outcome."""
        with self._lock:
            trust = self.get_trust(source)
            if passed:
                trust = min(1.0, trust + self._trust_inc)
            else:
                trust = max(0.0, trust - self._trust_dec)
            self._trust_scores[source] = trust

            if self._bus:
                self._bus.publish(CH_TRUST_UPDATED, {
                    "source": source,
                    "trust": trust,
                    "passed": passed,
                })
            return trust

    def set_trust(self, source: str, trust: float) -> None:
        """Directly set trust for a source (e.g., initial calibration)."""
        with self._lock:
            self._trust_scores[source] = max(0.0, min(1.0, trust))

    def is_trusted(self, source: str) -> bool:
        """Check if a source meets minimum trust threshold."""
        return self.get_trust(source) >= self._min_trust

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(
        self,
        source: str,
        input_type: str,
        data: Any,
        numeric_value: Optional[float] = None,
    ) -> ValidationReport:
        """Validate an input from a source.

        Runs trust check, registered validators, and anomaly detection.
        """
        with self._lock:
            self._total_validations += 1
            checks_passed = 0
            checks_failed = 0
            details: list[str] = []

            # Check 1: Source trust
            trust = self.get_trust(source)
            if trust < self._quarantine_trust:
                details.append(f"trust_too_low ({trust:.2f})")
                checks_failed += 1
            elif trust < self._min_trust:
                details.append(f"trust_marginal ({trust:.2f})")
                checks_failed += 1
            else:
                details.append(f"trust_ok ({trust:.2f})")
                checks_passed += 1

            # Check 2: Custom validators
            for name, checker in self._validators.get(input_type, []):
                try:
                    if checker(data, source):
                        checks_passed += 1
                        details.append(f"{name}_passed")
                    else:
                        checks_failed += 1
                        details.append(f"{name}_failed")
                except Exception as e:
                    checks_failed += 1
                    details.append(f"{name}_error: {e}")

            # Check 3: Anomaly detection (if numeric value provided)
            if numeric_value is not None:
                history = list(self._input_history[f"{source}:{input_type}"])
                self._input_history[f"{source}:{input_type}"].append(numeric_value)

                if len(history) >= 10:
                    mean = sum(history) / len(history)
                    std = max(0.01, float(np.std(history)))
                    z_score = abs(numeric_value - mean) / std

                    if z_score > 3.0:
                        checks_failed += 1
                        details.append(f"anomaly_detected (z={z_score:.1f})")
                    else:
                        checks_passed += 1
                        details.append(f"anomaly_clear (z={z_score:.1f})")

            # Determine result
            if checks_failed == 0:
                result = ValidationResult.PASSED
                self._passed += 1
            elif trust >= self._min_trust and checks_failed <= 1:
                result = ValidationResult.QUARANTINED
                self._quarantined += 1
            else:
                result = ValidationResult.FAILED
                self._failed += 1

            report = ValidationReport(
                source=source,
                input_type=input_type,
                result=result,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                details=details,
            )

            # Update trust based on result
            self.update_trust(source, result == ValidationResult.PASSED)

            # Publish failure events
            if result == ValidationResult.FAILED:
                self._violations.append(report)
                if self._bus:
                    self._bus.publish(CH_VALIDATION_FAILED, {
                        "source": source,
                        "input_type": input_type,
                        "checks_failed": checks_failed,
                        "trust": trust,
                    })

            return report

    # =========================================================================
    # Validator Registration
    # =========================================================================

    def register_validator(
        self,
        input_type: str,
        name: str,
        checker: Callable[[Any, str], bool],
    ) -> None:
        """Register a custom validation function for an input type.

        The checker receives (data, source_id) and returns True if valid.
        """
        self._validators[input_type].append((name, checker))

    def register_range_validator(
        self,
        input_type: str,
        field_name: str,
        min_val: float,
        max_val: float,
    ) -> None:
        """Register a numeric range check for an input type."""

        def _check(data: Any, source: str) -> bool:
            if isinstance(data, dict):
                val = data.get(field_name)
            elif hasattr(data, field_name):
                val = getattr(data, field_name)
            else:
                return False
            if val is None:
                return False
            return min_val <= float(val) <= max_val

        self.register_validator(input_type, f"range_{field_name}", _check)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def validate_policy_update(
        self, source: str, policy_weights: Any
    ) -> ValidationReport:
        """Validate a policy update from FRL peer sharing."""
        numeric = None
        if hasattr(policy_weights, '__len__'):
            try:
                arr = np.asarray(policy_weights)
                numeric = float(np.mean(np.abs(arr)))
            except (ValueError, TypeError):
                pass
        return self.validate(source, "policy_update", policy_weights, numeric)

    def validate_message(
        self, source: str, message: Any
    ) -> ValidationReport:
        """Validate a GNN/signal message."""
        return self.validate(source, "message", message)

    def validate_state_update(
        self, source: str, state: Any
    ) -> ValidationReport:
        """Validate an agent state update."""
        return self.validate(source, "state_update", state)

    def validate_incoming_message(
        self, source: str, message: Any
    ) -> ValidationReport:
        """Convenience wrapper: validate an incoming message from a source.

        Equivalent to validate_message but with a clearer name for
        boundary-guard usage at system entry points.
        """
        return self.validate(source, "message", message)

    # =========================================================================
    # FRL Integration
    # =========================================================================

    def _on_trust_updated(self, channel: str, message: Any) -> None:
        """Handle defense.trust_updated events to sync peer trust with FRL.

        When trust changes for a source, update the FRL engine's peer
        trust weights if available.
        """
        if self._frl_engine is None:
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

        source = data.get("source")
        trust = data.get("trust")
        if source is not None and trust is not None:
            try:
                self._frl_engine.update_peer_trust(source, trust)
            except Exception as e:
                logger.warning("Failed to update FRL peer trust for %s: %s", source, e)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_validations": self._total_validations,
                "passed": self._passed,
                "failed": self._failed,
                "quarantined": self._quarantined,
                "pass_rate": self._passed / max(self._total_validations, 1),
                "tracked_sources": len(self._trust_scores),
                "registered_validators": sum(
                    len(v) for v in self._validators.values()
                ),
                "recent_violations": len(self._violations),
            }

    def get_trust_report(self) -> dict[str, float]:
        """Get trust scores for all known sources."""
        with self._lock:
            return dict(self._trust_scores)
