"""Bootstrap Layer 33f-h: Market EventBus callbacks, step hooks, and sensing hook.

One job: wire all runtime signal pathways.
Three sub-tasks:
  - EventBus callbacks: endocrine coupling for convergence + hypothesis lifecycle
  - Step hooks: cadenced operations (convergence/1, stats/10, velocity/50, etc.)
  - Sensing hook: MarketSensingHook + advisory bridge

Critical constraint: ctx._cached_alerts is the shared handshake between
_register_market_step_hooks() (writer) and _wire_sensing_hook() (reader).
_register_market_step_hooks() MUST be called before _wire_sensing_hook()
to ensure ctx._cached_alerts exists when the sensing hook wraps it.
Hook registration order also matters: _market_sense_hook runs before
_sensing_step_with_advisory because step hooks run in registration order.

Implementation is split across sub-modules:
  - market_hooks_trades.py       — trade processing functions
  - market_hooks_eventbus.py     — EventBus subscriber registration
  - market_hooks_steps.py        — step hook helpers
  - market_hooks_steps_core.py   — main step hook registration
  - market_hooks_sensing.py      — sensing pipeline wiring
"""

from __future__ import annotations

# Re-export public entry points so all callers using:
#   from mae_core.bootstrap.market_hooks import X
# continue to work without modification.

from mae_core.bootstrap.market_hooks_eventbus import _register_market_eventbus
from mae_core.bootstrap.market_hooks_steps_core import _register_market_step_hooks
from mae_core.bootstrap.market_hooks_sensing import _wire_sensing_hook

# Also re-export trade helpers and step helpers for any direct callers.
from mae_core.bootstrap.market_hooks_trades import (
    BYPASS_ELIGIBLE_SOURCES,
    _check_sweep_bypass,
    _write_paper_trade,
    _translate_and_log_executable_signal,
    _submit_to_alpaca,
)
from mae_core.bootstrap.market_hooks_steps import (
    _write_convergence_heartbeat,
    _run_drift_detector,
)

__all__ = [
    # Primary entry points (called by market_bootstrap.py / main.py)
    "_register_market_eventbus",
    "_register_market_step_hooks",
    "_wire_sensing_hook",
    # Trade helpers
    "BYPASS_ELIGIBLE_SOURCES",
    "_check_sweep_bypass",
    "_write_paper_trade",
    "_translate_and_log_executable_signal",
    "_submit_to_alpaca",
    # Step helpers
    "_write_convergence_heartbeat",
    "_run_drift_detector",
]
