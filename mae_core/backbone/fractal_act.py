"""Fractal ACT — action delegation at every scale (Law 3: Holon Protocol).

Re-export hub. Implementation split into:
  fractal_act_subsystem.py  — SubsystemAction, OrganClusterAction
  fractal_act_organ.py      — OrganAction
  fractal_act_organism.py   — OrganismAction, build_fractal_action, CH_FRACTAL_ACT
"""

from mae_core.backbone.fractal_act_organ import OrganAction  # noqa: F401
from mae_core.backbone.fractal_act_organism import (  # noqa: F401
    CH_FRACTAL_ACT,
    OrganismAction,
    build_fractal_action,
)
from mae_core.backbone.fractal_act_subsystem import (  # noqa: F401
    OrganClusterAction,
    SubsystemAction,
)
