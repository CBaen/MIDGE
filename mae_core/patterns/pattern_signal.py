"""Pattern Signal - The universal currency of Mae's pattern recognition ecosystem.

Every sensory neuron in the brain uses the same electrical signal format
regardless of whether the stimulus was light, sound, or pressure. PatternSignal
is Mae's equivalent -- the common action potential that all pattern detectors emit.

Biological analogy: Action potentials are the universal neural code. PatternSignal
is Mae's action potential. Different detectors fire for different reasons, but the
signal format is always the same.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PatternForm(Enum):
    """How a pattern was recognized -- its epistemic status.

    Three forms following the triadic principle:
    - Reactive: fast, single-source (like a pain reflex)
    - Correlated: multi-source agreement (like seeing fire AND feeling heat)
    - Ancestral: survived Rule of Three, permanent wisdom (like genetic memory)
    """

    REACTIVE = "reactive"
    CORRELATED = "correlated"
    ANCESTRAL = "ancestral"


class PatternDomain(Enum):
    """What kind of pattern was detected -- the sensory modality.

    Each domain corresponds to a class of patterns, like sight vs sound vs touch.
    """

    NOVELTY = "novelty"          # Something new detected (CuriosityDrive)
    PREDICTION = "prediction"    # Prediction error or confirmation (WorldModel)
    CAUSATION = "causation"      # Causal link discovered (CausalEngine)
    THREAT = "threat"            # Danger / risk detected (HAVEN, ThreatDetector)
    OPPORTUNITY = "opportunity"  # Beneficial pattern found
    CAPABILITY = "capability"    # New ability emerging (CapabilityDiscovery)
    FAILURE = "failure"          # System failure detected (AutoHealer)
    BEHAVIORAL = "behavioral"    # Action-reward pattern (PatternDistiller, DecisionRouter)
    STATE = "state"              # State configuration pattern (PatternDistiller)
    META = "meta"                # Pattern about patterns (strange loop)


@dataclass
class PatternSignal:
    """Universal pattern format -- Mae's neural action potential.

    Every pattern detector in Mae produces PatternSignals. The PatternBus
    collects them, the PatternCortex integrates them, and the decision
    system acts on them.

    Fields follow the biological model:
    - signal_id: unique identifier (like a spike timestamp)
    - source_system: which detector fired (like which nerve)
    - domain: what kind of pattern (like which sense)
    - form: epistemic status (reactive/correlated/ancestral)
    - confidence: how sure the source is [0, 1]
    - salience: how important/urgent [0, 1]
    - description: human-readable explanation
    - evidence: source-specific supporting data
    """

    source_system: str
    domain: PatternDomain
    form: PatternForm
    confidence: float
    salience: float
    description: str
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    ttl_steps: int = 5
    occurrence_count: int = 1
    related_signals: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.salience = max(0.0, min(1.0, self.salience))

    @property
    def is_expired(self) -> bool:
        return self.ttl_steps <= 0

    def tick(self) -> None:
        """Decrement TTL by one step."""
        self.ttl_steps -= 1

    def __repr__(self) -> str:
        return (
            f"PatternSignal({self.domain.value}/{self.form.value} "
            f"conf={self.confidence:.2f} sal={self.salience:.2f} "
            f"from={self.source_system})"
        )
