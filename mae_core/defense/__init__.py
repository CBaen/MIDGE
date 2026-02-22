"""Defense systems - threat detection, zero-trust validation, immune coordination.

ThreatDetector: Four biological defense strategies (porcupine, turtle, lizard, kangaroo).
InputValidator: Zero-trust input validation at system boundaries.
HAVEN (in learning/): Immune system for policy contagion detection.

These systems protect Mae from internal degradation, policy contagion,
and malformed inputs - like a biological immune system with multiple layers.
"""

from .input_validator import (
    CH_TRUST_UPDATED,
    CH_VALIDATION_FAILED,
    InputValidator,
    ValidationReport,
    ValidationResult,
)
from .boundary_membrane import BoundaryMembrane
from .renal_filter import RenalFilter
from .threat_detector import (
    CH_DEFENSE_ACTIVATED,
    CH_THREAT_DETECTED,
    CH_THREAT_NEUTRALIZED,
    DefenseResponse,
    DefenseStrategy,
    Threat,
    ThreatDetector,
    ThreatLevel,
)

__all__ = [
    # Threat Detection
    "ThreatDetector",
    "Threat",
    "ThreatLevel",
    "DefenseStrategy",
    "DefenseResponse",
    "CH_THREAT_DETECTED",
    "CH_DEFENSE_ACTIVATED",
    "CH_THREAT_NEUTRALIZED",
    # Input Validation
    "InputValidator",
    "ValidationReport",
    "ValidationResult",
    "CH_VALIDATION_FAILED",
    "CH_TRUST_UPDATED",
    # Boundary Membrane
    "BoundaryMembrane",
    # Renal Filter
    "RenalFilter",
]
