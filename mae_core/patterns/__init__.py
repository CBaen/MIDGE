"""Mae's Pattern Recognition Ecosystem -- the sensory cortex.

Translates signals from 13 independent pattern detectors into a common
language (PatternSignal), collects and correlates them (PatternBus),
integrates over time (PatternCortex), and feeds into decision-making.

Biological analogy: The thalamus translates raw sensory signals into a
format the cortex can process. The association cortex integrates multiple
senses into unified perception. This package is Mae's thalamus + cortex.
"""

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)
from mae_core.patterns.pattern_bus import (
    PatternBus,
    PatternDigest,
)
from mae_core.patterns.pattern_cortex import (
    PatternAdvisory,
    PatternCortex,
)
from mae_core.patterns.pattern_consolidator import PatternConsolidator
from mae_core.patterns.attentional_gate import AttentionalGate
from mae_core.patterns.global_workspace import (
    GlobalWorkspace,
    IgnitionResult,
    WorkspaceCandidate,
)

__all__ = [
    "AttentionalGate",
    "GlobalWorkspace",
    "IgnitionResult",
    "PatternAdvisory",
    "PatternBus",
    "PatternConsolidator",
    "PatternCortex",
    "PatternDigest",
    "PatternDomain",
    "PatternForm",
    "PatternSignal",
    "WorkspaceCandidate",
]
