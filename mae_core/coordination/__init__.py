"""Coordination systems - endocrine, circadian, resource regulation.

EndocrineSystem: 6 hormones modulating Mae's behavior globally.
CircadianRhythm: ACTIVE/CONSOLIDATION/REST cycle using Mesa time.

These systems provide the global "mood" and "clock" that all other
systems read to modulate their behavior. Hormones create persistent
mood states. Circadian rhythm creates predictable activity cycles.
"""

from .circadian_rhythm import (
    CH_PHASE_CHANGE,
    CircadianPhase,
    CircadianRhythm,
)
from .endocrine_system import (
    CH_HORMONE_RELEASE,
    CH_HORMONE_STATE,
    DEFAULT_HORMONE_CONFIGS,
    HORMONE_CASCADES,
    EndocrineSystem,
    HormoneConfig,
    HormoneType,
)
from .emotional_system import EmotionalSystem
from .homeostasis import HomeostasisRegulator
from .thermoregulation import ThermoregulationSystem
from .digestive_system import DigestiveSystem
from .respiratory_system import RespiratorySystem
from .vestibular_system import VestibularSystem
from .organism_state import CH_ORGANISM_ACTION_OUTCOME, OrganismState

__all__ = [
    # Endocrine
    "EndocrineSystem",
    "HormoneType",
    "HormoneConfig",
    "DEFAULT_HORMONE_CONFIGS",
    "HORMONE_CASCADES",
    "CH_HORMONE_RELEASE",
    "CH_HORMONE_STATE",
    # Circadian
    "CircadianRhythm",
    "CircadianPhase",
    "CH_PHASE_CHANGE",
    # Emotional
    "EmotionalSystem",
    # Homeostasis
    "HomeostasisRegulator",
    # Thermoregulation
    "ThermoregulationSystem",
    # Digestive
    "DigestiveSystem",
    # Respiratory
    "RespiratorySystem",
    # Vestibular
    "VestibularSystem",
    # Organism State
    "OrganismState",
    "CH_ORGANISM_ACTION_OUTCOME",
]
