"""Episodic Memory Mixin - Experience replay and consolidation.

Provides prioritized experience replay, memory consolidation ("sleep"),
semantic search over past experiences, generative replay via VAE,
triadic recall verification (Law 1: No Bare Dyads), memory reconsolidation,
and spreading activation for associative recall.

Triadic Recall Verification:
  Every memory recall is verified by three independent witnesses before
  being trusted. Without verification, recall is a bare dyad (agent <-> memory)
  with no witness. Biological analogy: memory reconsolidation — the brain
  cross-checks recalled memories against semantic context, temporal neighbors,
  and statistical baselines before acting on them.

  Witness 1 (Semantic): Does the memory's context match via similarity?
  Witness 2 (Temporal): Are neighboring memories consistent?
  Witness 3 (Statistical): Is the memory's reward within normal bounds?

Memory Reconsolidation (Nader et al. 2000):
  When a consolidated memory is recalled and the current experience
  contradicts it (prediction error > threshold), the memory enters a
  labile state — a reconsolidation window. During this window (5 steps,
  scaled from the biological ~6-hour window), the memory can be updated
  with new evidence. The update blends old and new: alpha * old + (1-alpha)
  * new, where alpha=0.6 gives the original memory inertia. After the
  window closes, the memory re-stabilizes (reconsolidates) with its
  modifications.

Spreading Activation (Collins & Loftus 1975):
  When a memory is recalled, activation spreads to semantically related
  memories. Activation decays with distance: each hop multiplies by a
  decay factor (0.7). Maximum spread depth is 2 hops.

Ported from v5-pivot base_agent.py Big Rock 9 memory methods.

This file is a thin re-export hub. Implementation lives in:
  - episodic_memory_core.py          (EpisodicMemoryCoreMixin)
  - episodic_memory_reconsolidation.py (EpisodicMemoryReconsolidationMixin)
  - episodic_memory_activation.py    (EpisodicMemoryActivationMixin)
  - episodic_memory_search.py        (EpisodicMemorySearchMixin)
"""

from __future__ import annotations

from mae_core.agents.mixins.episodic_memory_activation import EpisodicMemoryActivationMixin
from mae_core.agents.mixins.episodic_memory_core import EpisodicMemoryCoreMixin
from mae_core.agents.mixins.episodic_memory_reconsolidation import EpisodicMemoryReconsolidationMixin
from mae_core.agents.mixins.episodic_memory_search import EpisodicMemorySearchMixin


class EpisodicMemoryMixin(
    EpisodicMemoryCoreMixin,
    EpisodicMemoryReconsolidationMixin,
    EpisodicMemoryActivationMixin,
    EpisodicMemorySearchMixin,
):
    """Adds episodic memory, replay, consolidation, and semantic search to agents.

    Combines four focused mixins:
    - EpisodicMemoryCoreMixin: init, store, learn, consolidate, statistics, serialize
    - EpisodicMemoryReconsolidationMixin: labile windows, blending, stabilization
    - EpisodicMemoryActivationMixin: spreading activation, priming, decay
    - EpisodicMemorySearchMixin: verified recall, counterfactuals, step tick
    """


__all__ = [
    "EpisodicMemoryMixin",
    "EpisodicMemoryCoreMixin",
    "EpisodicMemoryReconsolidationMixin",
    "EpisodicMemoryActivationMixin",
    "EpisodicMemorySearchMixin",
]
