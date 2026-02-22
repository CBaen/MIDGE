"""Pattern translators -- sensory receptors for the pattern ecosystem.

Each translator listens to existing EventBus channels from a specific
subsystem and converts raw events into PatternSignals. No existing
systems are modified -- translators are pure listeners.

Biological analogy: Rod cells translate photons into neural signals.
Hair cells translate sound waves. Each translator converts one type
of raw stimulus into the universal PatternSignal format.
"""

from mae_core.patterns.translators.base import PatternTranslator

__all__ = [
    "PatternTranslator",
]
