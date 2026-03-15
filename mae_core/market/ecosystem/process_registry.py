"""process_registry.py — MIDGE ecosystem process definitions.

Single source of truth for all long-running MIDGE processes.
The supervisor reads this list to know what to start, watch, and restart.

Tiers:
  core     — The organism itself. If it dies, everything is flying blind.
  analysis — Background miners that improve learning over time. Restartable.
  mining   — Data extraction processes. Slow to restart; state is in files.
"""

from __future__ import annotations

from typing import List, TypedDict


class ProcessSpec(TypedDict):
    name: str          # Human-readable identifier, also used in status output
    module: str        # Python module path (run as: python -m {module} {args})
    args: List[str]    # Additional CLI arguments passed to the module
    tier: str          # "core" | "analysis" | "mining"
    restart_delay: int # Seconds to wait before restarting after a crash
    critical: bool     # True = alert prominently when this process dies


ECOSYSTEM_PROCESSES: List[ProcessSpec] = [
    {
        "name": "midge-daemon",
        "module": "main",
        "args": ["--daemon", "--agents", "5", "--steps", "500", "--pace", "1.5"],
        "tier": "core",
        "restart_delay": 5,
        "critical": True,
    },
    {
        "name": "midge-replay",
        "module": "mae_core.market.parallel.continuous_replay",
        "args": [],
        "tier": "analysis",
        "restart_delay": 10,
        "critical": False,
    },
    {
        "name": "midge-granger",
        "module": "mae_core.market.parallel.continuous_granger",
        "args": [],
        "tier": "analysis",
        "restart_delay": 10,
        "critical": False,
    },
    {
        "name": "midge-postmortem",
        "module": "mae_core.market.parallel.continuous_postmortem",
        "args": [],
        "tier": "analysis",
        "restart_delay": 30,
        "critical": False,
    },
    {
        "name": "midge-raw-miner",
        "module": "mae_core.market.parallel.raw_data_miner",
        "args": [],
        "tier": "mining",
        "restart_delay": 30,
        "critical": False,
    },
    {
        "name": "midge-cross-market",
        "module": "mae_core.market.parallel.cross_market_hunter",
        "args": [],
        "tier": "analysis",
        "restart_delay": 30,
        "critical": False,
    },
]

# All valid tier names — used for --only filtering
ALL_TIERS = {"core", "analysis", "mining"}
