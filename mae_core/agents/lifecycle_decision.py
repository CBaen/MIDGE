"""Decision and action lifecycle mixin for MycelialAgent.

Extracted from mycelial_agent.py to prevent monolith growth.
These methods are called by step() in mycelial_agent.py.

This file is a thin re-export hub. Implementation lives in:
  - lifecycle_inhibit_decide.py  (InhibitDecideMixin)
  - lifecycle_act.py             (ActMixin)
"""

from __future__ import annotations

from mae_core.agents.lifecycle_act import ActMixin
from mae_core.agents.lifecycle_inhibit_decide import InhibitDecideMixin


class DecisionActionLifecycleMixin(InhibitDecideMixin, ActMixin):
    """Decision and action lifecycle methods for MycelialAgent.

    All subsystem access is via getattr(self, ..., None) for graceful
    degradation when subsystems are not injected.

    Combines two focused mixins:
    - InhibitDecideMixin: _inhibit, _decide, _route_with_advisory
    - ActMixin: _act, _act_explore, _act_exploit, _act_communicate,
                _act_rest, _act_api_call
    """


__all__ = [
    "DecisionActionLifecycleMixin",
    "InhibitDecideMixin",
    "ActMixin",
]
