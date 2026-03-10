"""Holon Protocol - Universal self-awareness interface for Mae's fractal architecture.

Re-export hub. Implementation split into:
  holon_registry.py  — HolonEntry, HolonRegistry
  holon_proxy.py     — HolonProxy
  awareness_pulse.py — AwarenessPulse, CH_AWARENESS_PULSE, CH_AWARENESS_ANOMALY

Every system at every scale implements: sense, remember, decide, act, learn,
heal, know_self, know_up, know_down, know_peers.

HolonMixin (the 10-capability agent interface) lives in holon_mixin.py.

HolonRegistry complements SomaticMap:
  SomaticMap tracks dependencies (what breaks if X fails).
  HolonRegistry tracks containment (what lives inside X, what contains X).
"""

from mae_core.backbone.awareness_pulse import (  # noqa: F401
    AwarenessPulse,
    CH_AWARENESS_ANOMALY,
    CH_AWARENESS_PULSE,
)
from mae_core.backbone.holon_proxy import HolonProxy  # noqa: F401
from mae_core.backbone.holon_registry import HolonEntry, HolonRegistry  # noqa: F401
